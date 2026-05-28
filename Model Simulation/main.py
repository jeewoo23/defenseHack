"""
AERIS Simulator entry point.

Run:
    python main.py
    python main.py --scenario dual_objective --prefix aeris_dual_objective

Outputs:
    <prefix>_snapshots.png
    <prefix>_metrics.png
    <prefix>_animation.gif
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import optimizer
from sim import Simulation, POLICY_GREEDY, POLICY_HORIZON, VALID_POLICIES
from plotting import plot_snapshots_grid, plot_metrics, create_animation
from config import (
    DEFAULT_SCENARIO, N_STEPS, N_UAVS, N_ENEMIES, N_ENEMIES_FLANK,
    SCENARIO_BASELINE, SCENARIO_DUAL_OBJECTIVE,
)

SNAPSHOT_STEPS = [1, 100, 200, 300, 400, 500]
IMAGE_OUTPUT_DIR = Path("outputs") / "images"


def _default_prefix(scenario: str) -> str:
    return "aeris_dual_objective" if scenario == SCENARIO_DUAL_OBJECTIVE else "aeris"


def run_scenario(
    scenario: str,
    prefix: str,
    animate_every: int,
    output_dir: Path = IMAGE_OUTPUT_DIR,
    policy: str = POLICY_GREEDY,
) -> Simulation:
    print("=" * 60)
    print("  AERIS: Autonomous Relay-Enabled ISR System")
    print(f"  Scenario: {scenario}")
    print(f"  Policy:   {policy}")
    print("  Simulation starting...")
    print("=" * 60)

    sim = Simulation(scenario=scenario, policy=policy)

    print(f"\nRunning {N_STEPS} steps  (snapshot at steps: {SNAPSHOT_STEPS})\n")
    sim.run(
        n_steps=N_STEPS,
        snapshot_at_steps=SNAPSHOT_STEPS,
        animate_every=animate_every,
        verbose=True,
    )

    print("\nGenerating output plots...")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_snapshots_grid(
        sim,
        save_path=output_dir / f"{prefix}_snapshots.png",
        nrows=2,
        ncols=3,
    )
    plot_metrics(sim.history, save_path=output_dir / f"{prefix}_metrics.png")
    create_animation(sim, save_path=output_dir / f"{prefix}_animation.gif", fps=8)

    final = sim.history[-1]
    best_cov = max(m.isr_coverage for m in sim.history)
    best_conn = max(m.conn_fraction for m in sim.history)
    total_uav_kills = sum(m.kills for m in sim.history)
    total_strikes = sum(m.strikes for m in sim.history)
    total_enemies = N_ENEMIES + 2 * N_ENEMIES_FLANK

    print("\n" + "=" * 60)
    print("  Simulation Summary")
    print("=" * 60)
    print(f"  Final alive UAVs    : {final.n_alive} / {N_UAVS}")
    print(f"  Enemies remaining   : {final.n_enemies} / {total_enemies}")
    print(f"  Enemy strikes       : {total_strikes}")
    print(f"  UAV kills by enemy  : {total_uav_kills}")
    if final.objective_health:
        for name, health in final.objective_health.items():
            print(f"  {name:18}: {health:5.1f}%")
    print(f"  Final ISR coverage  : {final.isr_coverage:.2f}")
    print(f"  Final connectivity  : {final.conn_fraction:.2f}")
    print(f"  Peak ISR coverage   : {best_cov:.2f}")
    print(f"  Peak connectivity   : {best_conn:.2f}")
    print(f"  Final objective     : {final.objective:.3f}")
    print("=" * 60)

    return sim


def main():
    parser = argparse.ArgumentParser(description="Run an AERIS simulation scenario.")
    parser.add_argument(
        "--scenario",
        choices=[SCENARIO_BASELINE, SCENARIO_DUAL_OBJECTIVE],
        default=DEFAULT_SCENARIO,
        help="Scenario layer to run.",
    )
    parser.add_argument("--prefix", default=None, help="Output filename prefix.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IMAGE_OUTPUT_DIR,
        help="Directory for generated simulation plots and animations.",
    )
    parser.add_argument(
        "--animate-every",
        type=int,
        default=2,
        help="Record every Nth step as an animation frame.",
    )
    parser.add_argument(
        "--policy",
        choices=VALID_POLICIES,
        default=POLICY_GREEDY,
        help="Control policy: greedy heuristic or finite-horizon optimizer.",
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
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show().")
    args = parser.parse_args()

    if args.emergency_intercept != "config":
        optimizer.ENABLE_EMERGENCY_INTERCEPT = args.emergency_intercept == "on"
    if args.objective_defense != "config":
        optimizer.ENABLE_OBJECTIVE_DEFENSE = args.objective_defense == "on"

    prefix = args.prefix or _default_prefix(args.scenario)
    if args.prefix is None:
        suffix = []
        if args.policy == POLICY_HORIZON:
            suffix.append("horizon")
        if args.emergency_intercept == "on":
            suffix.append("intercept")
        if args.objective_defense == "on":
            suffix.append("defense")
        if suffix:
            prefix = f"{prefix}_{'_'.join(suffix)}"
    run_scenario(
        args.scenario, prefix, max(1, args.animate_every),
        args.output_dir, policy=args.policy,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
