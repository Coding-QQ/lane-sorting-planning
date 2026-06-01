from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np


TargetLane = int | list[int]


@dataclass(frozen=True)
class PlanningParameters:
    """Parameters used by the lane-sorting planning model."""

    delta_t: float
    a_min: float
    a_max: float
    v_max: float
    lane_num: int
    helly_theta_1: float
    helly_theta_2: float
    vehicle_length: float
    d_safe_hdv: float
    tau_hdv: float
    d_safe_cav: float
    tau_cav: float
    lc_time: float
    planning_area_length: float
    delta_a_hdv: float | list[float]
    T_hat: int
    T_tilde: int
    delta_T_tilde: int
    delta_a_bounds: list[float] | None = None
    epsilon: float = 2.0
    rho_c: float = 1.0
    jerk_max: float = 3.0

    def __post_init__(self) -> None:
        if self.delta_t <= 0:
            raise ValueError("delta_t must be positive.")
        if self.lane_num < 1:
            raise ValueError("lane_num must be positive.")
        if self.lc_time <= 0:
            raise ValueError("lc_time must be positive.")
        lc_steps = self.lc_time / self.delta_t
        if not np.isclose(lc_steps, round(lc_steps)):
            raise ValueError("lc_time must be an integer multiple of delta_t.")
        if int(round(lc_steps)) < 1:
            raise ValueError("lc_time must be at least one discrete time step.")
        if self.T_hat < 2 or self.T_tilde < 2:
            raise ValueError("T_hat and T_tilde must be at least 2.")
        if self.delta_T_tilde < 1:
            raise ValueError("delta_T_tilde must be positive.")

    @property
    def mu(self) -> int:
        return int(round(self.lc_time / self.delta_t)) - 1

    @property
    def lane_change_blocking_steps(self) -> int:
        return self.mu + 1

    def with_horizon(self, horizon: int) -> "PlanningParameters":
        return replace(self, T_hat=int(horizon))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanningParameters":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioState:
    """Initial vehicle states for a lane-sorting case."""

    initial_position: np.ndarray
    initial_lane: np.ndarray
    initial_speed: np.ndarray
    target_position: np.ndarray
    vehicle_type: np.ndarray
    target_lane: list[TargetLane]
    hdv_delta_a: np.ndarray | None = None

    @property
    def vehicle_num(self) -> int:
        return int(len(self.initial_position))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioState":
        vehicle_type = data["vehicle_type"]
        if vehicle_type and isinstance(vehicle_type[0], str):
            vehicle_type = [1 if str(name).upper() == "CAV" else 0 for name in vehicle_type]

        target_lane: list[TargetLane] = []
        for item in data["target_lane"]:
            if isinstance(item, list):
                target_lane.append([int(lane) for lane in item])
            else:
                target_lane.append(int(item))

        hdv_delta_a = data.get("hdv_delta_a")
        return cls(
            initial_position=np.asarray(data["initial_position"], dtype=float),
            initial_lane=np.asarray(data["initial_lane"], dtype=int),
            initial_speed=np.asarray(data["initial_speed"], dtype=float),
            target_position=np.asarray(data["target_position"], dtype=float),
            vehicle_type=np.asarray(vehicle_type, dtype=int),
            target_lane=target_lane,
            hdv_delta_a=None if hdv_delta_a is None else np.asarray(hdv_delta_a, dtype=float),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_num": self.vehicle_num,
            "initial_position": self.initial_position.tolist(),
            "initial_lane": self.initial_lane.astype(int).tolist(),
            "initial_speed": self.initial_speed.tolist(),
            "target_position": self.target_position.tolist(),
            "vehicle_type": ["CAV" if int(value) == 1 else "HDV" for value in self.vehicle_type],
            "target_lane": self.target_lane,
            "hdv_delta_a": None if self.hdv_delta_a is None else self.hdv_delta_a.tolist(),
        }


@dataclass(frozen=True)
class SolverOptions:
    verbose: int = 1
    time_limit: float = 120.0
    mip_gap: float = 0.005


@dataclass
class PlannerResult:
    trajectories: dict[int, dict[str, np.ndarray]]
    feasible: bool
    cpu_time: float
    status: str
    mip_gap: float | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "feasible": bool(self.feasible),
            "cpu_time": float(self.cpu_time),
            "mip_gap": None if self.mip_gap is None else float(self.mip_gap),
        }


def load_case_config(path: str | Path) -> tuple[PlanningParameters, ScenarioState]:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return PlanningParameters.from_dict(payload["parameters"]), ScenarioState.from_dict(payload["state"])


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
