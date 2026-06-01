from __future__ import annotations

import numpy as np

from .model_data import PlanningParameters, ScenarioState


def compute_metrics(trajectories: dict[int, dict[str, np.ndarray]], params: PlanningParameters) -> dict[str, list[float] | float]:
    """Compute the public case-study metrics.

    Fuel is reported in two forms: ``total_fuel`` is total consumption, while
    ``fuel`` is distance-normalized consumption.
    """

    return {
        **_calculate_speed(trajectories, params),
        **_calculate_fuel(trajectories, params),
    }


def _calculate_speed(
    trajectories: dict[int, dict[str, np.ndarray]],
    params: PlanningParameters,
) -> dict[str, list[float] | float]:
    """Calculate the average speed of each vehicle before it leaves the planning area."""

    vehicle_num = len(trajectories)
    speed = np.full(vehicle_num, np.nan)
    for agent in sorted(trajectories):
        _, _, veh_speed, _, index = _trajectory_until_endpoint(trajectories[agent], params)
        if len(index) > 0:
            speed[agent] = float(np.mean(veh_speed[index]))
    return {"speed": speed.tolist(), "mean_speed": float(np.nanmean(speed))}


def _calculate_fuel(
    trajectories: dict[int, dict[str, np.ndarray]],
    params: PlanningParameters,
) -> dict[str, list[float] | float]:
    """Calculate total and distance-normalized fuel consumption."""

    vehicle_num = len(trajectories)
    fuel = np.full(vehicle_num, np.nan)
    total_fuel = np.full(vehicle_num, np.nan)

    alpha = 0.444
    beta1 = 0.09
    beta2 = 0.04
    b1 = 0.333
    b2 = 0.00108
    mass_factor = 1.2

    for agent in sorted(trajectories):
        pos, _, veh_speed, accel, index = _trajectory_until_endpoint(trajectories[agent], params)
        fuel_total = 0.0
        travel_distance = 0.0

        for local_t in range(len(index) - 1):
            k0 = index[local_t]
            k1 = index[local_t + 1]
            if pos[k0] < params.planning_area_length:
                segment_distance = max(pos[k1] - pos[k0], 0.0)
                rho1 = 1 if -(b1 + b2 * veh_speed[k0] ** 2) / mass_factor <= accel[k0] < 0 else 0
                rho2 = 1 if accel[k0] >= 0 else 0
                fuel_total += (
                    alpha
                    + (rho1 + rho2) * beta1 * veh_speed[k0]
                    * (b1 + b2 * veh_speed[k0] ** 2 + mass_factor * accel[k0])
                    + rho2 * beta2 * mass_factor * veh_speed[k0] * accel[k0] ** 2
                )
                travel_distance += segment_distance
        fuel_total *= params.delta_t
        total_fuel[agent] = fuel_total
        if travel_distance > 0:
            fuel[agent] = fuel_total / travel_distance
    return {
        "fuel": fuel.tolist(),
        "total_fuel": total_fuel.tolist(),
        "mean_fuel": float(np.nanmean(fuel)),
        "mean_total_fuel": float(np.nanmean(total_fuel)),
    }


def _trajectory_until_endpoint(
    trajectory: dict[str, np.ndarray],
    params: PlanningParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(trajectory["position"], dtype=float)
    lane = np.asarray(trajectory["lane"], dtype=float)
    accel = np.asarray(trajectory["accel"], dtype=float)
    speed = np.asarray(trajectory["speed"], dtype=float)
    if len(pos) == 0:
        return pos, lane, speed, accel, np.array([], dtype=int)

    endpoint = params.planning_area_length - params.v_max
    index_before_endpoint = np.where(pos <= endpoint)[0]
    if len(index_before_endpoint) == 0:
        return pos, lane, speed, accel, np.asarray([0])

    end_index = index_before_endpoint[-1] + 1
    missing_steps = end_index - (len(pos) - 1)
    if missing_steps > 0 and speed[-1] > 0:
        extra_pos = pos[-1] + speed[-1] * params.delta_t * np.arange(1, missing_steps + 1)
        pos = np.append(pos, extra_pos)
        lane = np.append(lane, np.repeat(lane[-1], missing_steps))
        speed = np.append(speed, np.repeat(speed[-1], missing_steps))
        accel = np.append(accel, np.zeros(missing_steps))
    end_index = min(end_index, len(pos) - 1)
    index = np.arange(0, end_index + 1)
    return pos, lane, speed, accel, index


def target_lane_success(
    trajectories: dict[int, dict[str, np.ndarray]],
    state: ScenarioState,
    params: PlanningParameters,
) -> dict[str, bool | dict[str, bool]]:
    """Check whether each vehicle reaches an acceptable target lane."""

    success_by_vehicle: dict[str, bool] = {}
    for agent in range(state.vehicle_num):
        trajectory = trajectories.get(agent)
        if not trajectory:
            success_by_vehicle[str(agent)] = False
            continue
        pos = np.asarray(trajectory["position"], dtype=float)
        lane = np.asarray(trajectory["lane"], dtype=float)
        endpoint_index = np.where(pos >= state.target_position[agent])[0]
        index = int(endpoint_index[0]) if len(endpoint_index) else len(lane) - 1
        final_lane = int(round(float(lane[index])))
        target = state.target_lane[agent]
        if isinstance(target, list):
            success_by_vehicle[str(agent)] = final_lane in target
        elif int(target) == 0:
            success_by_vehicle[str(agent)] = 1 <= final_lane <= params.lane_num
        else:
            success_by_vehicle[str(agent)] = final_lane == int(target)
    return {
        "all_target_lanes_satisfied": all(success_by_vehicle.values()),
        "target_lane_success_by_vehicle": success_by_vehicle,
    }
