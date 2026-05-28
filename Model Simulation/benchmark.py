"""
Multi-seed benchmark runner for AERIS simulation policies.

Examples:
    python benchmark.py --scenario baseline --branching on --seeds 20
    python benchmark.py --scenario dual_objective --branching off --seeds 20
    python benchmark.py --scenario dual_objective --branching on --seeds 20 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import optimizer
from config import (
    DEFAULT_SCENARIO, N_STEPS, OBJECTIVE_HEALTH,
    SCENARIO_BASELINE, SCENARIO_DUAL_OBJECTIVE,
    N_UAVS, N_ENEMIES, N_ENEMIES_FLANK,
)
from sim import Simulation, POLICY_GREEDY, POLICY_HORIZON, VALID_POLICIES


OUTPUT_ROOT = Path("outputs")
BENCHMARK_PLOT_DIR = OUTPUT_ROOT / "benchmark_plots"
CSV_OUTPUT_DIR = OUTPUT_ROOT / "csv"
JSON_OUTPUT_DIR = OUTPUT_ROOT / "json"

SERIES_KEYS = [
    "objective",
    "isr_coverage",
    "north_coverage",
    "center_coverage",
    "south_coverage",
    "time_weighted_coverage",
    "area_coverage",
    "conn_fraction",
    "avg_detection_latency",
    "n_alive",
    "n_enemies",
    "north_site_health",
    "south_site_health",
    "objectives_alive",
    "cumulative_strikes",
    "cumulative_uav_kills",
]


PLOT_KEYS = [
    ("objective", "Objective Score", "Score"),
    ("isr_coverage", "ISR Coverage", "Fraction"),
    ("conn_fraction", "Connected UAV Fraction", "Fraction"),
    ("area_coverage", "Area Coverage (fresh cells)", "Fraction"),
    ("center_coverage", "Center Coverage", "Fraction"),
    ("south_coverage", "South Coverage", "Fraction"),
    ("time_weighted_coverage", "Time-Weighted Coverage", "Observed / Alive"),
    ("avg_detection_latency", "Avg Detection Latency", "Timesteps"),
    ("north_site_health", "North Site Health", "Percent"),
]


def _as_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _set_branching(branching: str) -> None:
    if branching == "config":
        return
    optimizer.ENABLE_BRANCHING_CHAIN = branching == "on"


def _set_persistent_watch(persistent_watch: str) -> None:
    if persistent_watch == "config":
        return
    optimizer.ENABLE_PERSISTENT_ISR_WATCH = persistent_watch == "on"


def _set_emergency_intercept(emergency_intercept: str) -> None:
    if emergency_intercept == "config":
        return
    optimizer.ENABLE_EMERGENCY_INTERCEPT = emergency_intercept == "on"


def _set_objective_defense(objective_defense: str) -> None:
    if objective_defense == "config":
        return
    optimizer.ENABLE_OBJECTIVE_DEFENSE = objective_defense == "on"


def _extract_metrics(history) -> dict[str, list[float]]:
    metrics = {key: [] for key in SERIES_KEYS}
    strikes = 0
    uav_kills = 0

    for m in history:
        strikes += m.strikes
        uav_kills += m.kills
        objective_health = m.objective_health or {}

        metrics["objective"].append(_as_float(m.objective))
        metrics["isr_coverage"].append(_as_float(m.isr_coverage))
        metrics["north_coverage"].append(_as_float(m.north_coverage))
        metrics["center_coverage"].append(_as_float(m.center_coverage))
        metrics["south_coverage"].append(_as_float(m.south_coverage))
        metrics["time_weighted_coverage"].append(_as_float(m.time_weighted_coverage))
        metrics["area_coverage"].append(_as_float(m.area_coverage))
        metrics["conn_fraction"].append(_as_float(m.conn_fraction))
        metrics["avg_detection_latency"].append(_as_float(m.avg_detection_latency))
        metrics["n_alive"].append(_as_float(m.n_alive))
        metrics["n_enemies"].append(_as_float(m.n_enemies))
        metrics["north_site_health"].append(_as_float(objective_health.get("North Site")))
        metrics["south_site_health"].append(_as_float(objective_health.get("South Site")))
        metrics["objectives_alive"].append(_as_float(m.objectives_alive if objective_health else None))
        metrics["cumulative_strikes"].append(_as_float(strikes))
        metrics["cumulative_uav_kills"].append(_as_float(uav_kills))

    return metrics


def _run_one(args: tuple) -> dict:
    (seed, scenario, branching, persistent_watch, emergency_intercept,
     objective_defense, policy, steps, n_uavs, n_enemies, n_enemies_flank) = args
    _set_branching(branching)
    _set_persistent_watch(persistent_watch)
    _set_emergency_intercept(emergency_intercept)
    _set_objective_defense(objective_defense)
    np.random.seed(seed)

    sim = Simulation(
        seed=seed, scenario=scenario, policy=policy,
        n_uavs=n_uavs, n_enemies=n_enemies, n_enemies_flank=n_enemies_flank,
    )
    sim.run(n_steps=steps, verbose=False)
    metrics = _extract_metrics(sim.history)
    final = sim.history[-1]
    objective_health = final.objective_health or {}

    return {
        "seed": seed,
        "metrics": metrics,
        "summary": {
            "final_objective": _as_float(final.objective),
            "final_isr_coverage": _as_float(final.isr_coverage),
            "final_conn_fraction": _as_float(final.conn_fraction),
            "final_time_weighted_coverage": _as_float(final.time_weighted_coverage),
            "final_area_coverage": _as_float(final.area_coverage),
            "final_avg_detection_latency": _as_float(final.avg_detection_latency),
            "final_uavs_alive": _as_float(final.n_alive),
            "final_enemies_remaining": _as_float(final.n_enemies),
            "total_strikes": _as_float(sum(m.strikes for m in sim.history)),
            "total_uav_kills": _as_float(sum(m.kills for m in sim.history)),
            "north_site_health": _as_float(objective_health.get("North Site")),
            "south_site_health": _as_float(objective_health.get("South Site")),
            "mean_objective_health": _as_float(
                np.mean(list(objective_health.values())) if objective_health else None
            ),
        },
    }


def run_benchmark(
    scenario: str = DEFAULT_SCENARIO,
    branching: str = "config",
    persistent_watch: str = "config",
    emergency_intercept: str = "config",
    objective_defense: str = "config",
    policy: str = POLICY_GREEDY,
    seeds: int = 20,
    steps: int = N_STEPS,
    workers: int = 1,
    n_uavs: int = N_UAVS,
    n_enemies: int = N_ENEMIES,
    n_enemies_flank: int = N_ENEMIES_FLANK,
) -> list[dict]:
    jobs = [
        (seed, scenario, branching, persistent_watch, emergency_intercept,
         objective_defense, policy, steps, n_uavs, n_enemies, n_enemies_flank)
        for seed in range(seeds)
    ]
    if workers > 1:
        with Pool(processes=workers) as pool:
            return pool.map(_run_one, jobs)
    return [_run_one(job) for job in jobs]


def aggregate_results(results: list[dict]) -> dict:
    aggregated = {}
    for key in SERIES_KEYS:
        stacked = np.array([r["metrics"][key] for r in results], dtype=float)
        if np.all(np.isnan(stacked)):
            continue
        mean = []
        std = []
        p25 = []
        p75 = []
        for col in stacked.T:
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                mean.append(float("nan"))
                std.append(float("nan"))
                p25.append(float("nan"))
                p75.append(float("nan"))
                continue
            mean.append(float(np.mean(valid)))
            std.append(float(np.std(valid)))
            p25.append(float(np.percentile(valid, 25)))
            p75.append(float(np.percentile(valid, 75)))
        aggregated[key] = {
            "mean": np.array(mean),
            "std": np.array(std),
            "p25": np.array(p25),
            "p75": np.array(p75),
        }
    return aggregated


def summarize_results(results: list[dict]) -> dict:
    keys = results[0]["summary"].keys()
    summary = {}
    for key in keys:
        values = np.array([r["summary"][key] for r in results], dtype=float)
        if np.all(np.isnan(values)):
            continue
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "p25": float(np.nanpercentile(values, 25)),
            "p75": float(np.nanpercentile(values, 75)),
        }
    return summary


def plot_benchmark(aggregated: dict, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for ax, (key, label, ylabel) in zip(axes.flat, PLOT_KEYS):
        data = aggregated.get(key)
        if data is None:
            ax.text(0.5, 0.5, "Not applicable", ha="center", va="center",
                    transform=ax.transAxes, color="#64748B")
            ax.set_title(label)
            continue

        steps = np.arange(1, len(data["mean"]) + 1)
        ax.plot(steps, data["mean"], color="#2563EB", linewidth=2, label="Mean")
        ax.fill_between(steps, data["p25"], data["p75"], color="#93C5FD",
                        alpha=0.35, label="p25-p75")
        ax.set_title(label)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(summary: dict, output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "p25", "p75"])
        for key, stats in summary.items():
            writer.writerow([
                key,
                f"{stats['mean']:.6g}",
                f"{stats['std']:.6g}",
                f"{stats['p25']:.6g}",
                f"{stats['p75']:.6g}",
            ])


def print_summary(summary: dict) -> None:
    print("\nSummary")
    print("-" * 76)
    print(f"{'Metric':34} {'Mean':>10} {'Std':>10} {'p25':>10} {'p75':>10}")
    print("-" * 76)
    for key, stats in summary.items():
        print(
            f"{key:34} "
            f"{stats['mean']:10.3f} "
            f"{stats['std']:10.3f} "
            f"{stats['p25']:10.3f} "
            f"{stats['p75']:10.3f}"
        )


def _default_prefix(
    scenario: str, branching: str, persistent_watch: str,
    emergency_intercept: str, objective_defense: str, policy: str,
) -> str:
    base = f"benchmark_{scenario}_{branching}_watch_{persistent_watch}"
    if emergency_intercept != "config":
        base += f"_intercept_{emergency_intercept}"
    if objective_defense != "config":
        base += f"_defense_{objective_defense}"
    if policy != POLICY_GREEDY:
        base += f"_policy_{policy}"
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed AERIS benchmarks.")
    parser.add_argument(
        "--scenario",
        choices=[SCENARIO_BASELINE, SCENARIO_DUAL_OBJECTIVE],
        default=DEFAULT_SCENARIO,
    )
    parser.add_argument(
        "--branching",
        choices=["config", "on", "off"],
        default="config",
        help="Override adaptive branching for this benchmark.",
    )
    parser.add_argument(
        "--persistent-watch",
        choices=["config", "on", "off"],
        default="config",
        help="Override persistent objective-lane ISR watch behavior.",
    )
    parser.add_argument(
        "--emergency-intercept",
        choices=["config", "on", "off"],
        default="config",
        help="Override emergency objective-bound enemy intercept behavior.",
    )
    parser.add_argument(
        "--objective-defense",
        choices=["config", "on", "off"],
        default="config",
        help="Reserve a relay chain + forward ISR per objective.",
    )
    parser.add_argument(
        "--policy",
        choices=VALID_POLICIES,
        default=POLICY_GREEDY,
        help="Control policy: greedy heuristic or finite-horizon optimizer.",
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-uavs", type=int, default=N_UAVS)
    parser.add_argument("--n-enemies", type=int, default=N_ENEMIES,
                        help="Center-band enemies.")
    parser.add_argument("--n-flank", type=int, default=N_ENEMIES_FLANK,
                        help="Enemies per flank band (top and bottom).")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=BENCHMARK_PLOT_DIR,
        help="Directory for benchmark PNG plots.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=CSV_OUTPUT_DIR,
        help="Directory for benchmark CSV summaries.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=JSON_OUTPUT_DIR,
        help="Directory for benchmark JSON summaries.",
    )
    args = parser.parse_args()

    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    prefix = args.output_prefix or _default_prefix(
        args.scenario,
        args.branching,
        args.persistent_watch,
        args.emergency_intercept,
        args.objective_defense,
        args.policy,
    )
    print(
        f"Running benchmark: scenario={args.scenario}, branching={args.branching}, "
        f"persistent_watch={args.persistent_watch}, "
        f"emergency_intercept={args.emergency_intercept}, "
        f"objective_defense={args.objective_defense}, policy={args.policy}, "
        f"seeds={args.seeds}, steps={args.steps}, workers={args.workers}"
    )

    results = run_benchmark(
        scenario=args.scenario,
        branching=args.branching,
        persistent_watch=args.persistent_watch,
        emergency_intercept=args.emergency_intercept,
        objective_defense=args.objective_defense,
        policy=args.policy,
        seeds=args.seeds,
        steps=args.steps,
        workers=args.workers,
        n_uavs=args.n_uavs,
        n_enemies=args.n_enemies,
        n_enemies_flank=args.n_flank,
    )
    aggregated = aggregate_results(results)
    summary = summarize_results(results)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    args.json_dir.mkdir(parents=True, exist_ok=True)

    png_path = args.plot_dir / f"{prefix}.png"
    csv_path = args.csv_dir / f"{prefix}_summary.csv"
    json_path = args.json_dir / f"{prefix}_summary.json"

    title = (
        f"AERIS Benchmark - {args.scenario} - branching={args.branching} "
        f"- watch={args.persistent_watch} ({args.seeds} seeds)"
    )
    plot_benchmark(aggregated, png_path, title)
    write_summary_csv(summary, csv_path)
    json_path.write_text(json.dumps({
        "scenario": args.scenario,
        "branching": args.branching,
        "persistent_watch": args.persistent_watch,
        "emergency_intercept": args.emergency_intercept,
        "objective_defense": args.objective_defense,
        "policy": args.policy,
        "seeds": args.seeds,
        "steps": args.steps,
        "n_uavs": args.n_uavs,
        "n_enemies": args.n_enemies,
        "n_flank": args.n_flank,
        "summary": summary,
    }, indent=2))

    print_summary(summary)
    print(f"\nSaved {png_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
