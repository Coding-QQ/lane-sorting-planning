from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .model_data import PlannerResult, PlanningParameters, ScenarioState, SolverOptions

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:  # pragma: no cover - exercised only without Gurobi installed.
    gp = None
    GRB = None


class MIQPPlanner:
    """MIQP planner for centralized mixed-autonomy lane sorting."""

    def __init__(
        self,
        state: ScenarioState,
        params: PlanningParameters,
        options: SolverOptions | None = None,
        *,
        hma_mode: bool = False,
        iis_path: str | Path | None = None,
    ):
        if gp is None or GRB is None:
            raise ImportError("gurobipy is required to run MIQPPlanner. Install gurobipy and configure a Gurobi license.")
        self.state = state
        self.params = params
        self.options = options or SolverOptions()
        self.hma_mode = hma_mode
        self.iis_path = Path(iis_path) if iis_path is not None else None

        self.hdv_index = state.vehicle_type == 0
        self.cav_index = state.vehicle_type == 1
        self.vehicle_num = state.vehicle_num
        self.model = gp.Model("lane_sorting_miqp")
        self.variables: dict[str, np.ndarray] = {}

    def solve(self) -> PlannerResult:
        start_time = time.time()
        p = self.params
        n = self.vehicle_num
        T = p.T_hat
        M = 4.0 * p.planning_area_length
        M_small = 20.0
        M_mid = 100.0

        x = self.model.addVars(n, T, vtype=GRB.CONTINUOUS, name="x")
        v = self.model.addVars(n, T, lb=0, ub=p.v_max, vtype=GRB.CONTINUOUS, name="v")
        a = self.model.addVars(n, T, lb=p.a_min, ub=p.a_max, vtype=GRB.CONTINUOUS, name="a")
        a_tilde = self.model.addVars(n, T, lb=p.a_min, ub=p.a_max, vtype=GRB.CONTINUOUS, name="a_tilde")
        zeta = self.model.addVars(n, T, vtype=GRB.BINARY, name="zeta")
        l = self.model.addVars(n, T, lb=1, ub=p.lane_num, vtype=GRB.INTEGER, name="l")
        l_tilde = self.model.addVars(n, T, lb=1, ub=p.lane_num, vtype=GRB.INTEGER, name="l_tilde")
        alpha = self.model.addVars(n, T, vtype=GRB.BINARY, name="alpha")
        beta = self.model.addVars(n, T, vtype=GRB.BINARY, name="beta")
        B = self.model.addVars(n, n, T, vtype=GRB.BINARY, name="B")

        self._add_kinematic_constraints(x, v, a, l, l_tilde, alpha, beta, zeta, p, M)
        self._add_leader_order_constraints(x, B, n, T, M)
        self._add_target_lane_constraints(l, zeta, p, M)

        abs_terms = self._add_lane_relation_constraints(l, l_tilde, n, T, M)
        self._add_hdv_behavior_constraints(x, v, a, a_tilde, alpha, beta, zeta, B, abs_terms, p, M, M_mid)
        self._add_safety_constraints(x, v, B, abs_terms, p, M)

        is_in_lane = None
        if self.hma_mode:
            is_in_lane = self._add_target_lane_guidance_constraints(l, n, T, p, M_small)

        self._set_objective(x, a, alpha, beta, is_in_lane, p, n, T)
        self.model.setParam("MipGap", self.options.mip_gap)
        self.model.setParam("TimeLimit", self.options.time_limit)
        self.model.setParam("OutputFlag", 1 if self.options.verbose >= 2 else 0)

        self.model.optimize()
        feasible = self._has_solution()
        status = self._status_name()
        mip_gap = self._mip_gap_or_none()
        if feasible:
            self._extract_variables(x, v, a, a_tilde, alpha, beta, zeta, l, l_tilde, B, n, T)
            trajectories = self._build_trajectories()
            cpu_time = float(self.model.Runtime)
        else:
            if status in {"INFEASIBLE", "INF_OR_UNBD"} and self.iis_path is not None:
                self.iis_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.computeIIS()
                self.model.write(str(self.iis_path))
            trajectories = {}
            cpu_time = float(time.time() - start_time)
        return PlannerResult(trajectories=trajectories, feasible=feasible, cpu_time=cpu_time, status=status,
                             mip_gap=mip_gap)

    def _add_kinematic_constraints(self, x, v, a, l, l_tilde, alpha, beta, zeta, p, M) -> None:
        n = self.vehicle_num
        T = p.T_hat
        for t in range(T - 1):
            for veh in range(n):
                self.model.addConstr(x[veh, t + 1] - x[veh, t] == p.delta_t * v[veh, t + 1], name="euler_x")
                self.model.addConstr(v[veh, t + 1] - v[veh, t] == p.delta_t * a[veh, t + 1], name="euler_v")

        for t in range(1, T):
            for veh in range(n):
                if self.cav_index[veh]:
                    self.model.addConstr(a[veh, t] - a[veh, t - 1] <= p.jerk_max * p.delta_t, name="jerk_upper")
                    self.model.addConstr(a[veh, t] - a[veh, t - 1] >= -p.jerk_max * p.delta_t, name="jerk_lower")

        for veh in range(n):
            for t in range(T - p.lane_change_blocking_steps):
                self.model.addConstr(
                    gp.quicksum(alpha[veh, t + k] for k in range(p.lane_change_blocking_steps))
                    + gp.quicksum(beta[veh, t + k] for k in range(p.lane_change_blocking_steps))
                    <= 1,
                    name="consecutive_lane_change",
                )

        for t in range(T - 1):
            for veh in range(n):
                self.model.addConstr(
                    l_tilde[veh, t + 1] == l_tilde[veh, t] + alpha[veh, t + 1] - beta[veh, t + 1],
                    name="target_lane_update",
                )
        for t in range(T - p.mu - 1):
            for veh in range(n):
                self.model.addConstr(
                    l[veh, t + p.mu + 1] == l[veh, t + p.mu] + alpha[veh, t + 1] - beta[veh, t + 1],
                    name="current_lane_update",
                )

        for t in range(T):
            for veh in range(n):
                self.model.addConstr(-zeta[veh, t] * M + x[veh, t] - self.state.target_position[veh] <= 0)
                self.model.addConstr((1 - zeta[veh, t]) * M + x[veh, t] - self.state.target_position[veh] >= 0)
        for t in range(T - 1):
            for veh in range(n):
                self.model.addConstr(zeta[veh, t + 1] - zeta[veh, t] >= 0, name="zeta_monotonic")

        for veh in range(n):
            self.model.addConstr(x[veh, 0] == self.state.initial_position[veh], name="initial_position")
            self.model.addConstr(v[veh, 0] == self.state.initial_speed[veh], name="initial_speed")
            for t in range(p.mu + 1):
                self.model.addConstr(l[veh, t] == self.state.initial_lane[veh], name="initial_lane")
            self.model.addConstr(l_tilde[veh, 0] == self.state.initial_lane[veh], name="initial_l_tilde")

    def _add_leader_order_constraints(self, x, B, n, T, M) -> None:
        for veh in range(n):
            for other in range(n):
                if veh == other:
                    continue
                for t in range(T):
                    self.model.addConstr(B[veh, other, t] == 1 - B[other, veh, t], name="symmetric_B")
                    self.model.addConstr((B[veh, other, t] - 1) * M - x[veh, t] + x[other, t] <= 0)
                    self.model.addConstr(x[veh, t] - x[other, t] - B[veh, other, t] * M <= 0)

    def _add_target_lane_constraints(self, l, zeta, p, M) -> None:
        aux = self.model.addVars(self.vehicle_num, p.lane_num, vtype=GRB.BINARY, name="target_lane_aux")
        for veh in range(self.vehicle_num):
            target = self.state.target_lane[veh]
            for t in range(p.T_hat):
                if isinstance(target, list):
                    self.model.addConstr(gp.quicksum(aux[veh, lane - 1] for lane in target) == 1)
                    for lane in target:
                        self.model.addConstr((1 - zeta[veh, t]) * M + l[veh, t] - lane >= M * (aux[veh, lane - 1] - 1))
                        self.model.addConstr((1 - zeta[veh, t]) * M - l[veh, t] + lane >= M * (aux[veh, lane - 1] - 1))
                elif target != 0:
                    self.model.addConstr((1 - zeta[veh, t]) * M + l[veh, t] - target >= 0)
                    self.model.addConstr((1 - zeta[veh, t]) * M - l[veh, t] + target >= 0)

    def _add_lane_relation_constraints(self, l, l_tilde, n, T, M) -> dict[str, object]:
        abs_i_j = self.model.addVars(n, n, T, vtype=GRB.INTEGER, name="abs_i_j")
        abs_ti_j = self.model.addVars(n, n, T, vtype=GRB.INTEGER, name="abs_ti_j")
        abs_i_tj = self.model.addVars(n, n, T, vtype=GRB.INTEGER, name="abs_i_tj")
        abs_ti_tj = self.model.addVars(n, n, T, vtype=GRB.INTEGER, name="abs_ti_tj")
        u_i_j_1 = self.model.addVars(n, n, T, vtype=GRB.BINARY, name="u_i_j_1")
        u_i_j_2 = self.model.addVars(n, n, T, vtype=GRB.BINARY, name="u_i_j_2")
        u_ti_j_1 = self.model.addVars(n, n, T, vtype=GRB.BINARY, name="u_ti_j_1")
        u_ti_j_2 = self.model.addVars(n, n, T, vtype=GRB.BINARY, name="u_ti_j_2")
        u_i_tj_1 = self.model.addVars(n, n, T, vtype=GRB.BINARY, name="u_i_tj_1")
        u_i_tj_2 = self.model.addVars(n, n, T, vtype=GRB.BINARY, name="u_i_tj_2")
        u_ti_tj_1 = self.model.addVars(n, n, T, vtype=GRB.BINARY, name="u_ti_tj_1")
        u_ti_tj_2 = self.model.addVars(n, n, T, vtype=GRB.BINARY, name="u_ti_tj_2")

        specs = [
            (abs_i_j, u_i_j_1, u_i_j_2, l, l),
            (abs_ti_j, u_ti_j_1, u_ti_j_2, l_tilde, l),
            (abs_i_tj, u_i_tj_1, u_i_tj_2, l, l_tilde),
            (abs_ti_tj, u_ti_tj_1, u_ti_tj_2, l_tilde, l_tilde),
        ]
        for veh in range(n):
            for other in range(n):
                if veh == other:
                    continue
                for t in range(T):
                    for abs_var, u_1, u_2, lhs_lane, rhs_lane in specs:
                        diff = lhs_lane[veh, t] - rhs_lane[other, t]
                        self.model.addConstr(abs_var[veh, other, t] >= diff)
                        self.model.addConstr(abs_var[veh, other, t] >= -diff)
                        self.model.addConstr(diff - abs_var[veh, other, t] + M * (1 - u_1[veh, other, t]) >= 0)
                        self.model.addConstr(-diff - abs_var[veh, other, t] + M * (1 - u_2[veh, other, t]) >= 0)
                        self.model.addConstr(u_1[veh, other, t] + u_2[veh, other, t] >= 1)
        return {"i_j": abs_i_j, "ti_j": abs_ti_j, "i_tj": abs_i_tj, "ti_tj": abs_ti_tj}

    def _add_hdv_behavior_constraints(self, x, v, a, a_tilde, alpha, beta, zeta, B, abs_terms, p, M, M_mid) -> None:
        n = self.vehicle_num
        T = p.T_hat
        helly_cf = self.model.addVars(n, n, T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="helly_cf")
        a_bar = self.model.addVars(n, T, lb=p.a_min, ub=p.a_max, vtype=GRB.CONTINUOUS, name="a_bar")

        for veh in range(n):
            if not self.hdv_index[veh]:
                continue
            for other in range(n):
                if veh == other:
                    continue
                for t in range(T):
                    self.model.addConstr(
                        helly_cf[veh, other, t]
                        == p.helly_theta_1 * (v[other, t] - v[veh, t])
                        + p.helly_theta_2
                        * (x[other, t] - x[veh, t] - p.vehicle_length - p.d_safe_hdv - p.tau_hdv * v[veh, t]),
                        name="helly_cf",
                    )
                for t in range(T - 1):
                    self.model.addConstr(
                        a_bar[veh, t + 1] <= helly_cf[veh, other, t] + (B[veh, other, t] + abs_terms["i_j"][veh, other, t]) * M
                    )
                    self.model.addConstr(
                        a_bar[veh, t + 1] <= helly_cf[veh, other, t] + (B[veh, other, t] + abs_terms["i_tj"][veh, other, t]) * M
                    )
                    self.model.addConstr(
                        a_tilde[veh, t + 1] <= helly_cf[veh, other, t] + (B[veh, other, t] + abs_terms["ti_j"][veh, other, t]) * M
                    )
                    self.model.addConstr(
                        a_tilde[veh, t + 1] <= helly_cf[veh, other, t] + (B[veh, other, t] + abs_terms["ti_tj"][veh, other, t]) * M
                    )

        for t in range(T - 1):
            for veh in range(n):
                if self.hdv_index[veh]:
                    self.model.addConstr(v[veh, t] + p.delta_t * a_tilde[veh, t + 1] <= p.v_max)
                    self.model.addConstr(v[veh, t] + p.delta_t * a_tilde[veh, t + 1] >= 0)
                    self.model.addConstr(v[veh, t] + p.delta_t * a_bar[veh, t + 1] <= p.v_max)
                    self.model.addConstr(v[veh, t] + p.delta_t * a_bar[veh, t + 1] >= 0)

        for veh in range(n):
            if self.hdv_index[veh]:
                for t in range(T):
                    self.model.addConstr(a[veh, t] <= a_bar[veh, t], name="hdv_accel_following")
                    self.model.addConstr(a[veh, t] <= a_tilde[veh, t], name="hdv_accel_lc")

        if self.hma_mode:
            if self.state.hdv_delta_a is None:
                raise ValueError("HMA-mode MIQP requires ScenarioState.hdv_delta_a.")
            for veh in range(n):
                if self.hdv_index[veh]:
                    delta_a = float(self.state.hdv_delta_a[veh])
                    for t in range(1, T - 1):
                        self.model.addConstr(
                            (1 - alpha[veh, t] - beta[veh, t]) * M
                            + a_tilde[veh, t + 1] - a_bar[veh, t + 1]
                            >= delta_a,
                            name="hdv_lc_incentive_hma",
                        )
            return

        delta_a_values = p.delta_a_hdv
        if isinstance(delta_a_values, (float, int)):
            delta_a = float(delta_a_values)
            for veh in range(n):
                if self.hdv_index[veh]:
                    for t in range(1, T - 1):
                        self.model.addConstr(
                            (1 - alpha[veh, t] - beta[veh, t]) * M
                            + a_tilde[veh, t + 1] - a_bar[veh, t + 1]
                            >= delta_a,
                            name="hdv_lc_incentive",
                        )
            return

        delta_a_array = np.asarray(delta_a_values, dtype=float)
        segment_num = len(delta_a_array)
        bounds = self._get_delta_a_bounds(segment_num)
        b_i_s_t = self.model.addVars(n, segment_num, T, vtype=GRB.BINARY, name="delta_a_segment")
        delta_a_i_t = self.model.addVars(n, T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="delta_a_i_t")
        for veh in range(n):
            if not self.hdv_index[veh]:
                continue
            for t in range(1, T - 1):
                self.model.addConstr(gp.quicksum(b_i_s_t[veh, s, t] for s in range(segment_num)) == 1 - zeta[veh, t])
                for s in range(segment_num):
                    delta_a = float(delta_a_array[s])
                    self.model.addConstr(delta_a - (1 - b_i_s_t[veh, s, t]) * M_mid <= delta_a_i_t[veh, t])
                    self.model.addConstr(delta_a + (1 - b_i_s_t[veh, s, t]) * M_mid >= delta_a_i_t[veh, t])
                    self.model.addConstr(bounds[s] - (1 - b_i_s_t[veh, s, t]) * M <= x[veh, t])
                    self.model.addConstr(bounds[s + 1] + (1 - b_i_s_t[veh, s, t]) * M >= x[veh, t])
                self.model.addConstr(
                    (1 - alpha[veh, t] - beta[veh, t]) * M
                    + a_tilde[veh, t + 1] - a_bar[veh, t + 1]
                    >= delta_a_i_t[veh, t],
                    name="hdv_lc_incentive_segmented",
                )

    def _add_safety_constraints(self, x, v, B, abs_terms, p, M) -> None:
        n = self.vehicle_num
        for veh in range(n):
            for other in range(n):
                if veh == other:
                    continue
                if self.cav_index[veh]:
                    for t in range(p.T_hat):
                        gap = p.d_safe_cav - 0.2 if t == 0 else p.d_safe_cav
                        safe_gap = gap + p.vehicle_length + p.tau_cav * v[veh, t]
                        self.model.addConstr((B[veh, other, t] + abs_terms["i_j"][veh, other, t]) * M
                                             + x[other, t] - x[veh, t] - safe_gap >= 0)
                        self.model.addConstr((B[veh, other, t] + abs_terms["i_tj"][veh, other, t]) * M
                                             + x[other, t] - x[veh, t] - safe_gap >= 0)
                        self.model.addConstr((B[veh, other, t] + abs_terms["ti_j"][veh, other, t]) * M
                                             + x[other, t] - x[veh, t] - safe_gap >= 0)
                        self.model.addConstr((B[veh, other, t] + abs_terms["ti_tj"][veh, other, t]) * M
                                             + x[other, t] - x[veh, t] - safe_gap >= 0)
                else:
                    for t in range(p.T_hat):
                        safe_gap = p.d_safe_hdv + p.vehicle_length
                        self.model.addConstr((B[veh, other, t] + abs_terms["i_j"][veh, other, t]) * M
                                             + x[other, t] - x[veh, t] - safe_gap >= 0)
                        self.model.addConstr((B[veh, other, t] + abs_terms["i_tj"][veh, other, t]) * M
                                             + x[other, t] - x[veh, t] - safe_gap >= 0)

    def _get_delta_a_bounds(self, segment_num: int) -> np.ndarray:
        if self.params.delta_a_bounds is not None:
            bounds = np.asarray(self.params.delta_a_bounds, dtype=float)
            if len(bounds) != segment_num + 1:
                raise ValueError("delta_a_bounds length must equal len(delta_a_hdv) + 1.")
            return bounds
        return np.linspace(0.0, self.params.planning_area_length, segment_num + 1)

    def _add_target_lane_guidance_constraints(self, l, n, T, p, M_small):
        vehicle_indices = range(n)
        is_in_lane = self.model.addVars(vehicle_indices, p.lane_num, T, vtype=GRB.BINARY, name="target_lane_match")
        for veh in vehicle_indices:
            for t in range(T):
                for lane in range(p.lane_num):
                    lane_id = lane + 1
                    self.model.addConstr(l[veh, t] - lane_id >= M_small * (is_in_lane[veh, lane, t] - 1))
                    self.model.addConstr(lane_id - l[veh, t] >= M_small * (is_in_lane[veh, lane, t] - 1))
        return is_in_lane

    def _set_objective(self, x, a, alpha, beta, is_in_lane, p, n, T) -> None:
        obj_travel_dist_all = gp.quicksum(x[veh, t] for veh in range(n) for t in range(T))
        obj_accel_cav = gp.quicksum(a[veh, t] * a[veh, t] for veh in range(n) if self.cav_index[veh] for t in range(T))
        obj_comfort_cav = gp.quicksum(
            (1.0 / (p.delta_t ** 2)) * (a[veh, t] - a[veh, t - 1]) * (a[veh, t] - a[veh, t - 1])
            for veh in range(n) if self.cav_index[veh] for t in range(1, T)
        )
        obj_lc_cav = gp.quicksum(alpha[veh, t] + beta[veh, t] for veh in range(n) if self.cav_index[veh] for t in range(T))
        obj_lc_hdv = gp.quicksum(alpha[veh, t] + beta[veh, t] for veh in range(n) if self.hdv_index[veh] for t in range(T))

        obj_target_lane = 0
        if is_in_lane is not None:
            for veh in range(n):
                target = self.state.target_lane[veh]
                if isinstance(target, list):
                    obj_target_lane += gp.quicksum(is_in_lane[veh, lane - 1, t] for lane in target for t in range(T))
                elif target != 0:
                    obj_target_lane += gp.quicksum(is_in_lane[veh, target - 1, t] for t in range(T))

        self.model.setObjective(
            -0.1 * obj_travel_dist_all
            + obj_accel_cav
            + p.rho_c * obj_comfort_cav
            + 4.0 * obj_lc_cav
            + 2.0 * obj_lc_hdv
            - 1.5 * obj_target_lane,
            GRB.MINIMIZE,
        )

    def _has_solution(self) -> bool:
        return self.model.SolCount > 0 and self._mip_gap_or_none() != float("inf")

    def _status_name(self) -> str:
        if self.model.status == GRB.INFEASIBLE:
            return "INFEASIBLE"
        if self.model.status == GRB.INF_OR_UNBD:
            return "INF_OR_UNBD"
        if self.model.status == GRB.OPTIMAL:
            return "OPTIMAL"
        if self.model.status == GRB.TIME_LIMIT:
            return "TIME_LIMIT"
        return f"STATUS_{self.model.status}"

    def _mip_gap_or_none(self) -> float | None:
        try:
            return float(self.model.MIPGap)
        except gp.GurobiError:
            return None

    def _extract_variables(self, x, v, a, a_tilde, alpha, beta, zeta, l, l_tilde, B, n, T) -> None:
        self.variables = {
            "x": np.zeros((n, T)),
            "v": np.zeros((n, T)),
            "a": np.zeros((n, T)),
            "a_tilde": np.zeros((n, T)),
            "alpha": np.zeros((n, T)),
            "beta": np.zeros((n, T)),
            "zeta": np.zeros((n, T)),
            "l": np.zeros((n, T)),
            "l_tilde": np.zeros((n, T)),
            "B": np.zeros((n, n, T)),
        }
        for veh in range(n):
            for t in range(T):
                self.variables["x"][veh, t] = x[veh, t].X
                self.variables["v"][veh, t] = v[veh, t].X
                self.variables["a"][veh, t] = a[veh, t].X
                self.variables["a_tilde"][veh, t] = a_tilde[veh, t].X
                self.variables["alpha"][veh, t] = alpha[veh, t].X
                self.variables["beta"][veh, t] = beta[veh, t].X
                self.variables["zeta"][veh, t] = zeta[veh, t].X
                self.variables["l"][veh, t] = l[veh, t].X
                self.variables["l_tilde"][veh, t] = l_tilde[veh, t].X
        for veh in range(n):
            for other in range(n):
                for t in range(T):
                    self.variables["B"][veh, other, t] = B[veh, other, t].X

    def _build_trajectories(self) -> dict[int, dict[str, np.ndarray]]:
        trajectories: dict[int, dict[str, np.ndarray]] = {}
        for agent in range(self.vehicle_num):
            trajectories[agent] = {
                "position": self.variables["x"][agent, :].copy(),
                "lane": self.variables["l"][agent, :].copy(),
                "speed": self.variables["v"][agent, :].copy(),
                "accel": self.variables["a"][agent, :].copy(),
            }
        return trajectories
