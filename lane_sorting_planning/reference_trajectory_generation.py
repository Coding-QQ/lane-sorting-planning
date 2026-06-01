from __future__ import annotations

import numpy as np

from .model_data import PlanningParameters


def generate_quintic_reference_trajectory(
    trajectory: dict[str, np.ndarray],
    params: PlanningParameters,
    plot_step: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a reference trajectory with quintic lane-change curves.

    Llane-changing intervals use quintic polynomials with position, speed, acceleration, lane, 
    lateral speed, and lateral acceleration boundary conditions.
    """

    position = np.asarray(trajectory["position"], dtype=float)
    index = np.where((position > 0) & (position < params.planning_area_length * 1.04))[0]
    if len(index) == 0:
        return np.array([]), np.array([]), np.array([])

    pos = position[index]
    lane = np.asarray(trajectory["lane"], dtype=float)[index]
    speed = np.asarray(trajectory["speed"], dtype=float)[index]
    accel = np.asarray(trajectory["accel"], dtype=float)[index]

    lc_points_start = np.where(lane[1:] != lane[:-1])[0] - params.mu
    lc_points_end = lc_points_start + params.mu + 1

    x_start = pos[lc_points_start]
    v_start = speed[lc_points_start]
    a_start = accel[lc_points_start]
    x_end = pos[lc_points_end]
    v_end = speed[lc_points_end]
    a_end = accel[lc_points_end]
    y_start = lane[lc_points_start]
    y_end = lane[lc_points_end]

    interval_num = int(params.delta_t / plot_step)
    pos_smooth = np.array([])
    lane_smooth = np.array([])
    index_smooth = np.array([])

    step = 0
    while step <= len(pos) - 2:
        if not np.isin(step, lc_points_start):
            x_pos = np.linspace(pos[step], pos[step + 1], interval_num + 1)
            y_pos = np.linspace(lane[step], lane[step + 1], interval_num + 1)
            index_append = np.linspace(index[step], index[step + 1], interval_num + 1)
            step += 1
        else:
            ind = np.where(lc_points_start == step)[0][0]
            assert len(np.where(lc_points_start == step)[0]) == 1, "duplicated lane-changing time"

            x_parameter = np.array([x_start[ind], v_start[ind], a_start[ind], x_end[ind], v_end[ind], a_end[ind]])
            y_parameter = np.array([y_start[ind], 0, 0, y_end[ind], 0, 0])
            t0 = lc_points_start[ind]
            t1 = lc_points_end[ind]
            matrix = np.array(
                [
                    [t0**5, t0**4, t0**3, t0**2, t0**1, 1],
                    [5 * t0**4, 4 * t0**3, 3 * t0**2, 2 * t0**1, 1, 0],
                    [20 * t0**3, 12 * t0**2, 6 * t0, 2, 0, 0],
                    [t1**5, t1**4, t1**3, t1**2, t1**1, 1],
                    [5 * t1**4, 4 * t1**3, 3 * t1**2, 2 * t1**1, 1, 0],
                    [20 * t1**3, 12 * t1**2, 6 * t1, 2, 0, 0],
                ]
            )

            x_coeff = np.matmul(x_parameter, np.transpose(np.linalg.inv(matrix)))
            y_coeff = np.matmul(y_parameter, np.transpose(np.linalg.inv(matrix)))

            t = np.linspace(lc_points_start[ind], lc_points_end[ind], interval_num * (lc_points_end[ind] - lc_points_start[ind]) + 1)
            t_matrix = np.array([t**5, t**4, t**3, t**2, t**1, np.ones(len(t))])
            x_pos = np.matmul(x_coeff, t_matrix)
            y_pos = np.clip(np.matmul(y_coeff, t_matrix), min(y_start[ind], y_end[ind]), max(y_start[ind], y_end[ind]))
            index_append = np.linspace(
                index[lc_points_start[ind]],
                index[lc_points_end[ind]],
                interval_num * (lc_points_end[ind] - lc_points_start[ind]) + 1,
            )
            step = lc_points_end[ind]

        pos_smooth = np.append(pos_smooth, x_pos[:-1])
        lane_smooth = np.append(lane_smooth, y_pos[:-1])
        index_smooth = np.append(index_smooth, index_append[:-1])

    pos_smooth = np.append(pos_smooth, pos[-1])
    lane_smooth = np.append(lane_smooth, lane[-1])
    index_smooth = np.append(index_smooth, index[-1])

    return pos_smooth, lane_smooth, index_smooth
