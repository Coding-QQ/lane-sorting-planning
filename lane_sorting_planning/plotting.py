from __future__ import annotations

from pathlib import Path

import numpy as np

from .model_data import PlanningParameters
from .reference_trajectory_generation import generate_quintic_reference_trajectory


LEGEND_FONT = {"family": "Times New Roman", "size": 12}
AXIS_LABEL_FONT = {"family": "Times New Roman", "size": 16}


def plot_case_study(
    trajectories_by_solver: dict[str, dict[int, dict[str, np.ndarray]]],
    params: PlanningParameters,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for solver, trajectories in trajectories_by_solver.items():
        _plot_speed(trajectories, params, output_path / f"{solver}_speed.png")
        _plot_accel(trajectories, params, output_path / f"{solver}_accel.png")
        _plot_trajectory_2d(trajectories, params, output_path / f"{solver}_trajectory_2d.png")
        _plot_trajectory_3d(trajectories, params, output_path / f"{solver}_trajectory_3d.png")


def _visible_indices(position: np.ndarray, params: PlanningParameters) -> np.ndarray:
    return np.where((position > 0) & (position < params.planning_area_length * 1.04))[0]


def _plot_speed(trajectories: dict[int, dict[str, np.ndarray]], params: PlanningParameters, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    markers = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P"]

    for idx, agent in enumerate(trajectories.keys()):
        trajectory = trajectories[agent]
        position = np.asarray(trajectory["position"], dtype=float)
        speed = np.asarray(trajectory["speed"], dtype=float)
        index = _visible_indices(position, params)
        if len(index) == 0:
            continue

        time = index * params.delta_t
        marker = markers[idx % len(markers)]
        ax.plot(time, speed[index], linewidth=1.5, marker=marker, markersize=0, label=f"Vehicle: {agent}")

    ax.set_xlabel("Time (s)", fontdict=AXIS_LABEL_FONT)
    ax.set_ylabel("Speed (m/s)", fontdict=AXIS_LABEL_FONT)
    ax.set_ylim([-0.5, params.v_max + 1])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.axhline(params.v_max, color="r", linestyle="--", linewidth=1.25)
    ax.legend(prop=LEGEND_FONT)
    ax.grid(True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_accel(trajectories: dict[int, dict[str, np.ndarray]], params: PlanningParameters, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    markers = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P"]

    for idx, agent in enumerate(trajectories.keys()):
        trajectory = trajectories[agent]
        position = np.asarray(trajectory["position"], dtype=float)
        accel = np.asarray(trajectory["accel"], dtype=float)
        accel = np.clip(accel, params.a_min, params.a_max)
        index = _visible_indices(position, params)
        if len(index) == 0:
            continue

        time = index * params.delta_t
        accel = accel[index]
        marker = markers[idx % len(markers)]
        ax.plot(time[1:-1], accel[1:-1], linewidth=1.5, marker=marker, markersize=0, label=f"Vehicle: {agent}")

    ax.set_xlabel("Time (s)", fontdict=AXIS_LABEL_FONT)
    ax.set_ylabel("Acceleration (m/s$^2$)", fontdict=AXIS_LABEL_FONT)
    ax.set_ylim([params.a_min - 1, params.a_max + 1])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.axhline(params.a_max, color="r", linestyle="--", linewidth=1.25)
    ax.axhline(params.a_min, color="r", linestyle="--", linewidth=1.25)
    ax.legend(prop=LEGEND_FONT, loc="lower right")
    ax.grid(True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory_3d(
    trajectories: dict[int, dict[str, np.ndarray]],
    params: PlanningParameters,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    vehicle_num = len(trajectories)
    cmap = plt.colormaps.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, vehicle_num))

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.98)

    for agent in trajectories.keys():
        pos_smooth, lane_smooth, index_smooth = generate_quintic_reference_trajectory(trajectories[agent], params)
        ax.plot3D(
            lane_smooth,
            index_smooth * params.delta_t,
            pos_smooth,
            color=colors[int(agent)],
            linewidth=1.5,
            label=f"Vehicle: {agent}",
        )

    ax.view_init(elev=30, azim=-30)
    ax.set_ylabel("Time (s)", fontdict=AXIS_LABEL_FONT)
    ax.set_xlabel("Lane", fontdict=AXIS_LABEL_FONT)
    ax.set_zlabel("")
    fig.text(0.83, 0.52, "Distance (m)", rotation=90, va="center", ha="center", fontdict=AXIS_LABEL_FONT)
    ax.set_zlim([0, params.planning_area_length * 1.1])
    ax.set_xlim([1, params.lane_num])
    ax.grid(True)
    ax.legend(loc="upper left", prop=LEGEND_FONT)
    plt.xticks(np.arange(1, params.lane_num + 1, step=1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory_2d(
    trajectories: dict[int, dict[str, np.ndarray]],
    params: PlanningParameters,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    xlim = [-2, int(params.planning_area_length / params.v_max * 1.5)]
    ylim = [0, params.planning_area_length + 10]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()

    cmap = plt.colormaps.get_cmap("tab10")
    vehicle_num = len(trajectories)
    colors = cmap(np.linspace(0, 1, vehicle_num))
    lane_threshold = np.linspace(1, params.lane_num, params.lane_num + 1)

    for agent in trajectories.keys():
        pos_smooth, lane_smooth, index_smooth = generate_quintic_reference_trajectory(trajectories[agent], params)

        mask_1 = (lane_smooth >= lane_threshold[0]) & (lane_smooth < lane_threshold[1])
        mask_2 = (lane_smooth >= lane_threshold[1]) & (lane_smooth < lane_threshold[2])
        mask_3 = (lane_smooth > lane_threshold[2]) & (lane_smooth <= lane_threshold[3])

        solid_segments = np.split(np.where(mask_1)[0], np.where(np.diff(np.where(mask_1)[0]) != 1)[0] + 1)
        for segment in solid_segments:
            if len(segment) > 0:
                ax.plot(index_smooth[segment], pos_smooth[segment], linestyle="solid", linewidth=1.5, color=colors[int(agent)])
                if index_smooth[segment][0] > 0:
                    ax.plot(
                        index_smooth[segment][0],
                        pos_smooth[segment][0],
                        marker="o",
                        color=colors[int(agent)],
                        markersize=6,
                        markerfacecolor="none",
                    )

        dashed_segments = np.split(np.where(mask_2)[0], np.where(np.diff(np.where(mask_2)[0]) != 1)[0] + 1)
        for segment in dashed_segments:
            if len(segment) > 0:
                ax.plot(index_smooth[segment], pos_smooth[segment], linestyle="dashed", linewidth=1.5, color=colors[int(agent)])
                if index_smooth[segment][0] > 0:
                    ax.plot(
                        index_smooth[segment][0],
                        pos_smooth[segment][0],
                        marker="o",
                        color=colors[int(agent)],
                        markersize=6,
                        markerfacecolor="none",
                    )

        dotted_segments = np.split(np.where(mask_3)[0], np.where(np.diff(np.where(mask_3)[0]) != 1)[0] + 1)
        for segment in dotted_segments:
            if len(segment) > 0:
                ax.plot(index_smooth[segment], pos_smooth[segment], linestyle="dotted", linewidth=1.5, color=colors[int(agent)])
                if index_smooth[segment][0] > 0:
                    ax.plot(
                        index_smooth[segment][0],
                        pos_smooth[segment][0],
                        marker="o",
                        color=colors[int(agent)],
                        markersize=6,
                        markerfacecolor="none",
                    )

    ax.set_ylabel("Position (m)", fontdict=AXIS_LABEL_FONT)
    ax.set_xlabel("Time (s)", fontdict=AXIS_LABEL_FONT)
    ax.grid(True)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.axhline(y=params.planning_area_length - params.v_max, linestyle="--", color="red", linewidth=1.25)

    for agent in trajectories.keys():
        ax.plot([], [], linestyle="solid", linewidth=1.5, color=colors[int(agent)], label=f"Vehicle {agent}")
    agent_legend = ax.legend(loc="upper left", bbox_to_anchor=(0, 1), frameon=True, prop=LEGEND_FONT)
    ax.add_artist(agent_legend)

    lane_legend = [
        plt.Line2D([0], [0], linestyle="solid", linewidth=1.5, color="black", label="Lane 1"),
        plt.Line2D([0], [0], linestyle="dashed", linewidth=1.5, color="black", label="Lane 2"),
        plt.Line2D([0], [0], linestyle="dotted", linewidth=1.5, color="black", label="Lane 3"),
    ]
    ax.legend(handles=lane_legend, loc="lower right", frameon=True, prop=LEGEND_FONT)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
