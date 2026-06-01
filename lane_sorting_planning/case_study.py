from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .hma_planner import HMAPlanner
from .metrics import compute_metrics, target_lane_success
from .miqp_planner import MIQPPlanner
from .model_data import PlannerResult, SolverOptions, load_case_config, save_json


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = DEFAULT_ROOT / "configs" / "case_study.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "outputs" / "case_study"


def run_case_study(args: argparse.Namespace) -> dict[str, Any]:
    params, state = load_case_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    solvers = ["miqp", "hma"] if args.solver == "all" else [args.solver]
    result_payload: dict[str, Any] = {}
    trajectories_by_solver = {}

    for solver in solvers:
        try:
            if solver == "miqp":
                options = SolverOptions(verbose=args.verbose, time_limit=args.miqp_time_limit, mip_gap=args.mip_gap)
                result = MIQPPlanner(state=state, params=params, options=options).solve()
            else:
                options = SolverOptions(verbose=args.verbose, time_limit=args.hma_time_limit, mip_gap=args.mip_gap)
                result = HMAPlanner(state=state, params=params, options=options).solve()
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        trajectories_by_solver[solver] = result.trajectories
        metrics = _summarize_result(result, state, params)
        result_payload[solver] = metrics

    if args.plot:
        from .plotting import plot_case_study

        plot_case_study(trajectories_by_solver, params, output_dir)

    save_json(output_dir / "metrics.json", _build_metrics_output(result_payload, state.vehicle_num))
    _print_summary(result_payload)
    return result_payload


def _summarize_result(result: PlannerResult, state, params) -> dict[str, Any]:
    if result.feasible:
        metrics = compute_metrics(result.trajectories, params)
        target_summary = target_lane_success(result.trajectories, state, params)
    else:
        metrics = {}
        target_summary = {"all_target_lanes_satisfied": False, "target_lane_success_by_vehicle": {}}

    return {
        **result.to_summary_dict(),
        **target_summary,
        **metrics,
    }


def _build_metrics_output(payload: dict[str, Any], vehicle_num: int) -> dict[str, Any]:
    return {"table": _build_metrics_table(payload, vehicle_num)}


def _build_metrics_table(payload: dict[str, Any], vehicle_num: int) -> list[dict[str, Any]]:
    table_specs = [
        ("Avg. Speed (m/s)", "speed", "mean_speed"),
        ("Fuel Consumption (mL)", "total_fuel", "mean_total_fuel"),
    ]
    rows: list[dict[str, Any]] = []
    for metric_name, vehicle_key, mean_key in table_specs:
        for vehicle in range(vehicle_num):
            row: dict[str, Any] = {"metric": metric_name, "vehicle": f"Vehicle {vehicle}"}
            for solver in payload:
                values = payload[solver].get(vehicle_key, [])
                row[solver] = values[vehicle] if vehicle < len(values) else None
            rows.append(row)
        mean_row: dict[str, Any] = {"metric": metric_name, "vehicle": "Mean"}
        for solver in payload:
            mean_row[solver] = payload[solver].get(mean_key)
        rows.append(mean_row)
    return rows

def _print_summary(payload: dict[str, Any]) -> None:
    print("\nFeasibility")
    for solver in ["miqp", "hma"]:
        if solver in payload:
            print(f"{solver.upper()}: feasible={payload[solver].get('feasible')}")

    print("\nCase-study metrics")
    solvers = [solver for solver in ["miqp", "hma"] if solver in payload]
    if not solvers:
        return
    vehicle_num = max((len(payload[solver].get("speed", [])) for solver in solvers), default=0)
    header = f"{'Metric':<24} {'Vehicle':<10} " + " ".join(f"{solver.upper():>10}" for solver in solvers)
    print(header)
    print("-" * len(header))
    for row in _build_metrics_table(payload, vehicle_num):
        values = []
        for solver in solvers:
            value = row.get(solver)
            values.append(f"{float(value):>10.2f}" if value is not None else f"{'NA':>10}")
        print(f"{row['metric']:<24} {row['vehicle']:<10} " + " ".join(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lane-sorting MIQP/HMA case study.")
    parser.add_argument("--solver", choices=["miqp", "hma", "all"], default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--mip-gap", type=float, default=0.005)
    parser.add_argument("--miqp-time-limit", type=float, default=240.0)
    parser.add_argument("--hma-time-limit", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_case_study(args)


if __name__ == "__main__":
    main()
