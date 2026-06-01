from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from .miqp_planner import MIQPPlanner
from .model_data import PlannerResult, PlanningParameters, ScenarioState, SolverOptions, TargetLane


class HMAPlanner:
    """Heuristic multi-stage algorithm for the lane-sorting MIQP model."""

    def __init__(self, state: ScenarioState, params: PlanningParameters, options: SolverOptions | None = None):
        self.state = state
        self.params = params
        self.options = options or SolverOptions(verbose=1, time_limit=60.0, mip_gap=0.005)

    def solve(self) -> PlannerResult:
        start_time = time.time()
        active_agents = list(range(self.state.vehicle_num))
        positions = {agent: float(self.state.initial_position[agent]) for agent in active_agents}
        lanes = {agent: int(self.state.initial_lane[agent]) for agent in active_agents}
        speeds = {agent: float(self.state.initial_speed[agent]) for agent in active_agents}
        accels = {agent: 0.0 for agent in active_agents}
        target_lanes = {agent: self.state.target_lane[agent] for agent in active_agents}
        vehicle_types = {agent: int(self.state.vehicle_type[agent]) for agent in active_agents}
        target_positions = {agent: float(self.state.target_position[agent]) for agent in active_agents}

        trajectories: dict[int, dict[str, np.ndarray]] = {
            agent: {
                "position": np.asarray([positions[agent]], dtype=float),
                "lane": np.asarray([lanes[agent]], dtype=float),
                "speed": np.asarray([speeds[agent]], dtype=float),
                "accel": np.asarray([accels[agent]], dtype=float),
            }
            for agent in active_agents
        }

        stage = 0
        status = "SOLVED"
        while active_agents:
            actions = {
                agent: self._select_substage_target_lane(
                    positions[agent],
                    lanes[agent],
                    target_lanes[agent],
                    target_positions[agent],
                )
                for agent in active_agents
            }
            if self.options.verbose >= 1:
                print(f"HMA stage {stage}: target-lane actions = {actions}")

            sub_state, agent_order = self._build_subproblem_state(
                active_agents, positions, lanes, speeds, target_lanes, vehicle_types, target_positions, actions
            )
            sub_params = replace(self.params, T_hat=self.params.T_tilde)
            sub_result = MIQPPlanner(
                state=sub_state,
                params=sub_params,
                options=self.options,
                hma_mode=True,
            ).solve()

            if not sub_result.feasible:
                return PlannerResult(
                    trajectories=trajectories,
                    feasible=False,
                    cpu_time=time.time() - start_time,
                    status=f"HMA_SUBPROBLEM_{sub_result.status}",
                    mip_gap=sub_result.mip_gap,
                )

            delta = self.params.delta_T_tilde
            for local_index, agent in enumerate(agent_order):
                sub_traj = sub_result.trajectories[local_index]
                trajectories[agent]["position"] = np.append(
                    trajectories[agent]["position"], sub_traj["position"][1:delta + 1]
                )
                trajectories[agent]["lane"] = np.append(trajectories[agent]["lane"], sub_traj["lane"][1:delta + 1])
                trajectories[agent]["speed"] = np.append(trajectories[agent]["speed"], sub_traj["speed"][1:delta + 1])
                trajectories[agent]["accel"] = np.append(trajectories[agent]["accel"], sub_traj["accel"][1:delta + 1])
                positions[agent] = float(sub_traj["position"][delta])
                lanes[agent] = int(round(float(sub_traj["lane"][delta])))
                speeds[agent] = float(sub_traj["speed"][delta])
                accels[agent] = float(sub_traj["accel"][delta])

            active_agents = [agent for agent in active_agents if positions[agent] <= self.params.planning_area_length]
            stage += 1
            if stage > 20:
                status = "HMA_STAGE_LIMIT"
                break

        feasible = not active_agents and status == "SOLVED"
        return PlannerResult(trajectories=trajectories, feasible=feasible, cpu_time=time.time() - start_time,
                             status=status, mip_gap=None)

    def _build_subproblem_state(
        self,
        active_agents: list[int],
        positions: dict[int, float],
        lanes: dict[int, int],
        speeds: dict[int, float],
        target_lanes: dict[int, TargetLane],
        vehicle_types: dict[int, int],
        target_positions: dict[int, float],
        actions: dict[int, TargetLane],
    ) -> tuple[ScenarioState, list[int]]:
        initial_position = []
        initial_lane = []
        initial_speed = []
        vehicle_type = []
        target_position = []
        sub_target_lane: list[TargetLane] = []
        hdv_delta_a = []

        for agent in active_agents:
            initial_position.append(positions[agent])
            initial_lane.append(lanes[agent])
            initial_speed.append(speeds[agent])
            vehicle_type.append(vehicle_types[agent])
            if vehicle_types[agent] == 0:
                new_target_position = target_positions[agent]
                sub_lane = target_lanes[agent]
            else:
                if positions[agent] > target_positions[agent]:
                    new_target_position = positions[agent] + self._hma_target_position_buffer()
                else:
                    new_target_position = min(positions[agent] + self._hma_target_position_buffer(), target_positions[agent])
                sub_lane = actions[agent] if new_target_position < target_positions[agent] else target_lanes[agent]
            target_position.append(new_target_position)
            sub_target_lane.append(sub_lane)
            hdv_delta_a.append(self._delta_a_at_position(positions[agent]))

        return (
            ScenarioState(
                initial_position=np.asarray(initial_position, dtype=float),
                initial_lane=np.asarray(initial_lane, dtype=int),
                initial_speed=np.asarray(initial_speed, dtype=float),
                target_position=np.asarray(target_position, dtype=float),
                vehicle_type=np.asarray(vehicle_type, dtype=int),
                target_lane=sub_target_lane,
                hdv_delta_a=np.asarray(hdv_delta_a, dtype=float),
            ),
            active_agents.copy(),
        )

    def _hma_target_position_buffer(self) -> float:
        return float(self.params.v_max * (self.params.delta_T_tilde + self.params.epsilon))

    def _select_substage_target_lane(
        self,
        position: float,
        lane: int,
        target_lane: TargetLane,
        target_position: float,
    ) -> TargetLane:
        lane = int(round(lane))
        target_lane = self._normalize_target_lane(target_lane)
        gamma_1 = 1.0 / 2.0
        gamma_2 = 2.0 / 3.0

        if isinstance(target_lane, int):
            if target_lane <= 0:
                return 0
            if position > gamma_2 * target_position:
                return int(target_lane)
            if position >= gamma_1 * target_position:
                return int(lane + np.sign(target_lane - lane))
            return lane if lane == target_lane else [lane, int(lane + np.sign(target_lane - lane))]

        if lane in target_lane:
            return target_lane

        if lane < min(target_lane):
            nearest_lane = min(target_lane)
        elif lane > max(target_lane):
            nearest_lane = max(target_lane)
        else:
            raise ValueError("Target-lane set is not compatible with current lane.")

        if position > gamma_2 * target_position:
            return target_lane
        if position >= gamma_1 * target_position:
            return target_lane if abs(lane - nearest_lane) == 1 else int(nearest_lane)
        return [lane, int(lane + np.sign(nearest_lane - lane))]

    def _normalize_target_lane(self, target_lane: TargetLane) -> TargetLane:
        if isinstance(target_lane, np.integer):
            return int(target_lane)
        if isinstance(target_lane, list):
            return [int(round(value)) for value in target_lane]
        return int(target_lane)

    def _delta_a_at_position(self, position: float) -> float:
        values = self.params.delta_a_hdv
        if isinstance(values, (float, int)):
            return float(values)
        values_array = np.asarray(values, dtype=float)
        bounds = self._delta_a_bounds(len(values_array))
        for index, value in enumerate(values_array):
            if bounds[index] <= position <= bounds[index + 1]:
                return float(value)
        return float(values_array[-1])

    def _delta_a_bounds(self, segment_num: int) -> np.ndarray:
        if self.params.delta_a_bounds is not None:
            bounds = np.asarray(self.params.delta_a_bounds, dtype=float)
            if len(bounds) != segment_num + 1:
                raise ValueError("delta_a_bounds length must equal len(delta_a_hdv) + 1.")
            return bounds
        return np.linspace(0.0, self.params.planning_area_length, segment_num + 1)
