"""
AERIS Simulator — entry point.

Run:
    python main.py

Outputs:
    aeris_snapshots.png  — tactical map at key timesteps
    aeris_metrics.png    — 6-panel time-series dashboard
"""
import matplotlib.pyplot as plt
from sim import Simulation
from plotting import plot_snapshots_grid, plot_metrics, create_animation
from config import N_STEPS, N_UAVS, N_ENEMIES, N_ENEMIES_FLANK

SNAPSHOT_STEPS = [1, 100, 200, 300, 400, 500]


def main():
    print("=" * 60)
    print("  AERIS: Autonomous Relay-Enabled ISR System")
    print("  Simulation starting...")
    print("=" * 60)

    sim = Simulation()

    print(f"\nRunning {N_STEPS} steps  (snapshot at steps: {SNAPSHOT_STEPS})\n")
    sim.run(n_steps=N_STEPS, snapshot_at_steps=SNAPSHOT_STEPS, verbose=True)

    print("\nGenerating output plots...")

    fig_snaps = plot_snapshots_grid(sim, save_path="aeris_snapshots.png")
    fig_metrics = plot_metrics(sim.history, save_path="aeris_metrics.png")
    create_animation(sim, save_path="aeris_animation.gif", fps=8)

    # Summary statistics
    final = sim.history[-1]
    best_cov = max(m.isr_coverage  for m in sim.history)
    best_conn = max(m.conn_fraction for m in sim.history)
    total_uav_kills   = sum(m.kills   for m in sim.history)
    total_strikes     = sum(m.strikes for m in sim.history)

    print("\n" + "=" * 60)
    print("  Simulation Summary")
    print("=" * 60)
    total_enemies = N_ENEMIES + 2 * N_ENEMIES_FLANK
    print(f"  Final alive UAVs    : {final.n_alive} / {N_UAVS}")
    print(f"  Enemies remaining   : {final.n_enemies} / {total_enemies}")
    print(f"  Enemy strikes       : {total_strikes}")
    print(f"  UAV kills by enemy  : {total_uav_kills}")
    print(f"  Final ISR coverage  : {final.isr_coverage:.2f}")
    print(f"  Final connectivity  : {final.conn_fraction:.2f}")
    print(f"  Peak ISR coverage   : {best_cov:.2f}")
    print(f"  Peak connectivity   : {best_conn:.2f}")
    print(f"  Final objective     : {final.objective:.3f}")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
