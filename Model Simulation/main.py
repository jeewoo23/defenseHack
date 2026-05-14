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

import matplotlib.pyplot as plt

from sim import Simulation
from plotting import plot_snapshots_grid, plot_metrics, create_animation
from config import (
    DEFAULT_SCENARIO, N_STEPS, N_UAVS, N_ENEMIES, N_ENEMIES_FLANK,
    SCENARIO_BASELINE, SCENARIO_DUAL_OBJECTIVE,
)

SNAPSHOT_STEPS = [1, 100, 200, 300, 400, 500]


def _default_prefix(scenario: str) -> str:
    return "aeris_dual_objective" if scenario == SCENARIO_DUAL_OBJECTIVE else "aeris"


def run_scenario(scenario: str, prefix: str, animate_every: int) -> Simulation:
    print("=" * 60)
    print("  AERIS: Autonomous Relay-Enabled ISR System")
    print(f"  Scenario: {scenario}")
    print("  Simulation starting...")
    print("=" * 60)

    sim = Simulation(scenario=scenario)

    print(f"\nRunning {N_STEPS} steps  (snapshot at steps: {SNAPSHOT_STEPS})\n")
    sim.run(
        n_steps=N_STEPS,
        snapshot_at_steps=SNAPSHOT_STEPS,
        animate_every=animate_every,
        verbose=True,
    )

    print("\nGenerating output plots...")

    plot_snapshots_grid(sim, save_path=f"{prefix}_snapshots.png")
    plot_metrics(sim.history, save_path=f"{prefix}_metrics.png")
    create_animation(sim, save_path=f"{prefix}_animation.gif", fps=8)

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
        "--animate-every",
        type=int,
        default=2,
        help="Record every Nth step as an animation frame.",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show().")
    args = parser.parse_args()

    prefix = args.prefix or _default_prefix(args.scenario)
    run_scenario(args.scenario, prefix, max(1, args.animate_every))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
