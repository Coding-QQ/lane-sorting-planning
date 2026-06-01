# Infrastructure-Enabled Centralized Trajectory Planning for Lane Sorting

This repository contains the core code for the MIQP and heuristic multi-stage
algorithm (HMA) used for centralized lane-sorting trajectory planning in
mixed-autonomy traffic.

The code requires Gurobi. Install `gurobipy` and make sure a valid Gurobi
license is available before running the planners.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Run The Case Study

Run both MIQP and HMA for the provided case study:

```bash
python -m lane_sorting_planning.case_study --solver all
```

Optional trajectory, speed, and acceleration figures can be generated with:

```bash
python -m lane_sorting_planning.case_study --solver all --plot
```

Outputs are written to `outputs/case_study/` by default:

- `metrics.json`, containing MIQP/HMA case-study metrics
- optional speed, acceleration, 2D trajectory, and 3D trajectory PNG figures when
  `--plot` is used

Note: Numerical values may vary slightly with the Gurobi and runtime environment.
This repository provides the core MIQP/HMA algorithms and one case-study example.

## Project Structure

```text
lane_sorting_planning/
  model_data.py       # dataclasses for parameters, scenarios, and solver results
  miqp_planner.py     # MIQP trajectory planner
  hma_planner.py      # heuristic multi-stage algorithm
  metrics.py          # speed, fuel, and total-fuel metrics
  reference_trajectory_generation.py  # reference trajectory generation using quintic polynomial curves
  plotting.py         # optional case-study plots
  case_study.py       # command-line entry point
configs/
  case_study.json
```
