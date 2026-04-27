"""
Visualization: map snapshots and time-series metrics.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import networkx as nx

from uav import UAVMode
from config import ENEMY_KILL_RANGE, ENEMY_THREAT_RANGE, ISR_SENSOR_RANGE

# Visual style per mode
_MODE_COLOR  = {UAVMode.ISR: "#2196F3", UAVMode.MOBILE_RELAY: "#4CAF50", UAVMode.STATIC_RELAY: "#FF9800"}
_MODE_MARKER = {UAVMode.ISR: "o",       UAVMode.MOBILE_RELAY: "^",       UAVMode.STATIC_RELAY: "D"}


# ---------------------------------------------------------------------------
# Single-step map snapshot
# ---------------------------------------------------------------------------
def plot_snapshot(sim, G: nx.Graph, connected_ids: set, step: int, ax=None):
    """
    Draw the tactical map for one timestep.
    - Terrain shown as background contour
    - Comm links: green if both endpoints connected, red otherwise
    - UAVs: colour/shape by mode, white edge = connected, red edge = disconnected
    - Dead UAVs shown as grey X
    - Enemies shown as red stars with kill-range circle
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    size = sim.terrain.size

    # --- Terrain background ---
    _, _, grid = sim.terrain.build_grid(resolution=60)
    ax.imshow(
        grid,
        extent=[0, size, 0, size],
        origin="lower",
        cmap="terrain",
        alpha=0.35,
        vmin=0.55, vmax=1.35,
        aspect="auto",
    )

    # --- Comm links ---
    node_pos = {"base": sim.base.pos}
    for u in sim.uavs:
        node_pos[u.id] = u.pos

    for u_node, v_node in G.edges():
        p1 = node_pos.get(u_node)
        p2 = node_pos.get(v_node)
        if p1 is None or p2 is None:
            continue
        u_connected = (u_node == "base") or (u_node in connected_ids)
        v_connected = (v_node == "base") or (v_node in connected_ids)
        color  = "#00C853" if (u_connected and v_connected) else "#D50000"
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, alpha=0.45, linewidth=1.2, zorder=2)

    # --- Base station ---
    ax.scatter(*sim.base.pos, s=350, c="black", marker="s", zorder=6)
    ax.annotate("Base", sim.base.pos + np.array([120, 120]),
                fontsize=8, fontweight="bold", color="black")

    # --- UAVs ---
    for uav in sim.uavs:
        if not uav.alive:
            ax.scatter(*uav.pos, s=80, c="gray", marker="x", alpha=0.4, zorder=4)
            continue
        color      = _MODE_COLOR[uav.mode]
        marker     = _MODE_MARKER[uav.mode]
        edge_color = "white" if uav.id in connected_ids else "#FF1744"
        ax.scatter(*uav.pos, s=160, c=color, marker=marker,
                   edgecolors=edge_color, linewidths=2, zorder=5)
        bat_str = f"U{uav.id}\n{uav.battery:.0f}%"
        ax.annotate(bat_str, uav.pos + np.array([80, 80]), fontsize=6.5, zorder=7)

    # --- Enemies ---
    for enemy in sim.enemies:
        ax.scatter(*enemy.pos, s=220, c="#B71C1C", marker="*", zorder=6)
        ax.annotate(f"E{enemy.id}", enemy.pos + np.array([80, 80]),
                    fontsize=8, color="#B71C1C", fontweight="bold")
        # Kill range circle
        ax.add_patch(plt.Circle(enemy.pos, ENEMY_KILL_RANGE,
                                color="#B71C1C", fill=False,
                                alpha=0.5, linewidth=1.2, linestyle="--"))
        # Threat range circle (lighter)
        ax.add_patch(plt.Circle(enemy.pos, ENEMY_THREAT_RANGE,
                                color="#FF6F00", fill=False,
                                alpha=0.25, linewidth=0.8, linestyle=":"))

    # --- Legend ---
    legend_items = [
        mpatches.Patch(color=_MODE_COLOR[UAVMode.ISR],          label="ISR UAV"),
        mpatches.Patch(color=_MODE_COLOR[UAVMode.MOBILE_RELAY], label="Mobile Relay"),
        mpatches.Patch(color=_MODE_COLOR[UAVMode.STATIC_RELAY], label="Static Relay"),
        mpatches.Patch(color="#B71C1C", label="Enemy"),
        mpatches.Patch(color="black",   label="Base"),
        mpatches.Patch(color="#00C853", label="Active link"),
        mpatches.Patch(color="#D50000", label="Broken link"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7.5,
              framealpha=0.85)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_title(f"Step {step}", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    return ax


# ---------------------------------------------------------------------------
# Time-series metrics dashboard
# ---------------------------------------------------------------------------
def plot_metrics(history, save_path: str | None = None):
    steps       = [m.step          for m in history]
    isr_cov     = [m.isr_coverage  for m in history]
    conn_frac   = [m.conn_fraction for m in history]
    n_alive     = [m.n_alive       for m in history]
    avg_bat     = [m.avg_battery   for m in history]
    objective   = [m.objective     for m in history]
    total_kills = np.cumsum([m.kills for m in history])

    isr_c  = [m.role_counts.get("ISR",          0) for m in history]
    mob_c  = [m.role_counts.get("Mobile Relay", 0) for m in history]
    sta_c  = [m.role_counts.get("Static Relay", 0) for m in history]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("AERIS Simulation — Performance Metrics", fontsize=14, fontweight="bold")

    # ISR Coverage
    ax = axes[0, 0]
    ax.plot(steps, isr_cov, color="#2196F3", linewidth=2)
    ax.fill_between(steps, isr_cov, alpha=0.15, color="#2196F3")
    ax.set_title("Valid ISR Coverage")
    ax.set_ylabel("Fraction of Enemies Observed")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # Connected fraction
    ax = axes[0, 1]
    ax.plot(steps, conn_frac, color="#4CAF50", linewidth=2)
    ax.fill_between(steps, conn_frac, alpha=0.15, color="#4CAF50")
    ax.set_title("Connected UAV Fraction")
    ax.set_ylabel("Fraction")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # Alive UAVs + cumulative kills
    ax = axes[0, 2]
    ax.plot(steps, n_alive, color="#9C27B0", linewidth=2, label="Alive")
    ax2 = ax.twinx()
    ax2.plot(steps, total_kills, color="#F44336", linewidth=1.5,
             linestyle="--", label="Cumulative kills")
    ax2.set_ylabel("Cumulative Enemy Kills", color="#F44336")
    ax.set_title("Alive UAVs & Enemy Kills")
    ax.set_ylabel("Alive UAVs")
    ax.set_ylim(-0.5, 11)
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")

    # Role distribution (stacked area)
    ax = axes[1, 0]
    ax.stackplot(steps, isr_c, mob_c, sta_c,
                 labels=["ISR", "Mobile Relay", "Static Relay"],
                 colors=["#2196F3", "#4CAF50", "#FF9800"], alpha=0.82)
    ax.set_title("Role Distribution")
    ax.set_ylabel("UAV Count")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Battery
    ax = axes[1, 1]
    ax.plot(steps, avg_bat, color="#FF9800", linewidth=2)
    ax.fill_between(steps, avg_bat, alpha=0.15, color="#FF9800")
    ax.axhline(20, color="red", linewidth=1, linestyle="--", alpha=0.6, label="Critical (20%)")
    ax.set_title("Average Battery Level")
    ax.set_ylabel("Battery (%)")
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Objective score
    ax = axes[1, 2]
    ax.plot(steps, objective, color="#607D8B", linewidth=2)
    ax.fill_between(steps, objective,
                    [min(objective)] * len(objective),
                    alpha=0.12, color="#607D8B")
    ax.set_title("Objective Score")
    ax.set_ylabel("Score (higher = better)")
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Timestep")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"Saved metrics plot -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Animated GIF
# ---------------------------------------------------------------------------
def create_animation(sim, save_path: str = "aeris_animation.gif",
                     fps: int = 8) -> animation.FuncAnimation:
    """
    Render all recorded frames into an animated GIF.
    Frames are captured by sim.run() every `animate_every` steps.
    """
    frames = sim.frames
    if not frames:
        print("No animation frames recorded.")
        return None

    size = sim.terrain.size
    _, _, terrain_grid = sim.terrain.build_grid(resolution=60)

    fig, ax = plt.subplots(figsize=(8, 8))

    def _draw_frame(frame_data):
        ax.clear()

        # Terrain
        ax.imshow(
            terrain_grid,
            extent=[0, size, 0, size],
            origin="lower",
            cmap="terrain",
            alpha=0.35,
            vmin=0.55, vmax=1.35,
            aspect="auto",
        )

        # Build position lookup for edge drawing
        node_pos = {"base": sim.base.pos}
        for uid, pos in zip(frame_data["uav_id"], frame_data["uav_pos"]):
            node_pos[uid] = pos

        connected_ids = frame_data["connected_ids"]

        # Comm links
        for u_node, v_node in frame_data["edges"]:
            p1 = node_pos.get(u_node)
            p2 = node_pos.get(v_node)
            if p1 is None or p2 is None:
                continue
            u_conn = (u_node == "base") or (u_node in connected_ids)
            v_conn = (v_node == "base") or (v_node in connected_ids)
            color = "#00C853" if (u_conn and v_conn) else "#D50000"
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, alpha=0.45, linewidth=1.2, zorder=2)

        # Base station
        ax.scatter(*sim.base.pos, s=350, c="black", marker="s", zorder=6)
        ax.annotate("Base", sim.base.pos + np.array([120, 120]),
                    fontsize=8, fontweight="bold", color="black")

        # UAVs
        for uid, pos, mode, alive, bat in zip(
            frame_data["uav_id"], frame_data["uav_pos"],
            frame_data["uav_mode"], frame_data["uav_alive"],
            frame_data["uav_battery"],
        ):
            from uav import UAVMode
            if not alive:
                ax.scatter(*pos, s=80, c="gray", marker="x", alpha=0.4, zorder=4)
                continue
            color      = _MODE_COLOR[mode]
            marker     = _MODE_MARKER[mode]
            edge_color = "white" if uid in connected_ids else "#FF1744"
            ax.scatter(*pos, s=160, c=color, marker=marker,
                       edgecolors=edge_color, linewidths=2, zorder=5)
            ax.annotate(f"U{uid}\n{bat:.0f}%", pos + np.array([80, 80]),
                        fontsize=6.5, zorder=7)

        # Enemies
        for eid, epos in zip(frame_data["enemy_id"], frame_data["enemy_pos"]):
            ax.scatter(*epos, s=220, c="#B71C1C", marker="*", zorder=6)
            ax.annotate(f"E{eid}", epos + np.array([80, 80]),
                        fontsize=8, color="#B71C1C", fontweight="bold")
            ax.add_patch(plt.Circle(epos, ENEMY_KILL_RANGE,
                                    color="#B71C1C", fill=False,
                                    alpha=0.5, linewidth=1.2, linestyle="--"))
            ax.add_patch(plt.Circle(epos, ENEMY_THREAT_RANGE,
                                    color="#FF6F00", fill=False,
                                    alpha=0.25, linewidth=0.8, linestyle=":"))

        # Legend
        legend_items = [
            mpatches.Patch(color=_MODE_COLOR[UAVMode.ISR],          label="ISR UAV"),
            mpatches.Patch(color=_MODE_COLOR[UAVMode.MOBILE_RELAY], label="Mobile Relay"),
            mpatches.Patch(color=_MODE_COLOR[UAVMode.STATIC_RELAY], label="Static Relay"),
            mpatches.Patch(color="#B71C1C", label="Enemy"),
            mpatches.Patch(color="black",   label="Base"),
            mpatches.Patch(color="#00C853", label="Active link"),
            mpatches.Patch(color="#D50000", label="Broken link"),
        ]
        ax.legend(handles=legend_items, loc="upper right", fontsize=7.5,
                  framealpha=0.85)

        m = frame_data["metrics"]
        ax.set_title(
            f"Step {frame_data['step']}  |  "
            f"Alive: {m.n_alive}  |  "
            f"ISR cov: {m.isr_coverage:.2f}  |  "
            f"Conn: {m.conn_fraction:.2f}  |  "
            f"Score: {m.objective:.2f}",
            fontsize=9,
        )
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")

    def _update(i):
        _draw_frame(frames[i])
        return []

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=1000 // fps, blit=False
    )

    print(f"Saving animation ({len(frames)} frames) -> {save_path}  (this may take a moment...)")
    writer = animation.PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    print(f"Saved animation -> {save_path}")
    plt.close(fig)
    return anim


# ---------------------------------------------------------------------------
# Multi-snapshot grid
# ---------------------------------------------------------------------------
def plot_snapshots_grid(sim, save_path: str | None = None):
    """Render all stored snapshots in a grid layout."""
    snaps = sim.snapshots
    if not snaps:
        print("No snapshots recorded.")
        return None

    ncols = min(len(snaps), 5)
    nrows = (len(snaps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 6 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    snap_iter = iter(snaps)
    for row in axes:
        for ax in row:
            item = next(snap_iter, None)
            if item is None:
                ax.axis("off")
            else:
                step, G, connected_ids = item
                plot_snapshot(sim, G, connected_ids, step, ax=ax)

    fig.suptitle("AERIS Simulation — Snapshots", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved snapshot grid -> {save_path}")
    return fig
