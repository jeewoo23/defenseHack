"""
Simulation loop.

Timestep ordering:
  1.  Enemies move (hunt static relays)
  2.  Enemies attempt kills on static relays in range
  3.  Build communication graph
  4.  Identify connected UAVs
  5.  Apply greedy policy (role changes + movement decisions)
  6.  Drain batteries (may kill UAVs with empty batteries)
  7.  Rebuild communication graph after movement
  8.  Recompute connectivity + ISR coverage
  9.  Log metrics
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List

import networkx as nx

from config import (
    WORLD_SIZE, BASE_POS, N_UAVS, N_ENEMIES, SIM_SEED,
    ENEMY_SPAWN_X_MIN, ENEMY_SPAWN_X_MAX,
    ENEMY_SPAWN_Y_MIN, ENEMY_SPAWN_Y_MAX,
)

from uav import UAV, UAVMode
from world import BaseStation, Enemy, Terrain
from graph import build_comm_graph, get_connected_uav_ids, compute_isr_coverage
from optimizer import compute_objective, greedy_policy


# ---------------------------------------------------------------------------
# Metric snapshot
# ---------------------------------------------------------------------------
@dataclass
class StepMetrics:
    step:         int
    n_alive:      int
    n_connected:  int
    isr_coverage: float
    conn_fraction: float
    avg_battery:  float
    objective:    float
    kills:        int          # UAVs killed by enemies this step
    role_counts:  dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class Simulation:
    def __init__(
        self,
        n_uavs:    int   = N_UAVS,
        n_enemies: int   = N_ENEMIES,
        seed:      int   = SIM_SEED,
    ):
        rng = np.random.RandomState(seed)

        self.terrain  = Terrain(size=WORLD_SIZE, seed=seed)
        self.base     = BaseStation(pos=BASE_POS)

        # UAVs start clustered near the base station
        self.uavs: list[UAV] = []
        for i in range(n_uavs):
            offset = rng.uniform(-400, 400, 2)
            pos    = np.clip(np.array(BASE_POS) + offset, 0, WORLD_SIZE)
            self.uavs.append(UAV(uav_id=i, pos=pos, mode=UAVMode.ISR))

        # Enemies start in the operational zone (within relay chain corridor)
        self.enemies: list[Enemy] = []
        for i in range(n_enemies):
            pos = [rng.uniform(ENEMY_SPAWN_X_MIN, ENEMY_SPAWN_X_MAX),
                   rng.uniform(ENEMY_SPAWN_Y_MIN, ENEMY_SPAWN_Y_MAX)]
            self.enemies.append(Enemy(enemy_id=i, pos=pos))

        self.step_num:         int               = 0
        self.history:          list[StepMetrics] = []
        self._prev_switch_total: int             = 0

        # Snapshot storage: list of (step, G, connected_ids) for selected steps
        self.snapshots: list[tuple] = []

        # Animation frames: lightweight per-step records
        self.frames: list[dict] = []

    # ------------------------------------------------------------------
    def step(self, record_snapshot: bool = False,
             record_frame: bool = False) -> StepMetrics:
        self.step_num += 1

        # 1. Enemies move
        for enemy in self.enemies:
            enemy.move(self.uavs)

        # 2. Enemy kill attempts
        kills_this_step = 0
        for enemy in self.enemies:
            killed = enemy.attempt_kill(self.uavs)
            kills_this_step += len(killed)

        # 3. Build initial graph
        G            = build_comm_graph(self.uavs, self.base, self.terrain)
        connected_ids = get_connected_uav_ids(G, self.uavs)

        # 4. Greedy policy (modifies modes and positions)
        greedy_policy(self.uavs, connected_ids,
                      self.enemies, self.base, self.terrain, G)

        # 5. Drain batteries
        for uav in self.uavs:
            uav.drain_battery()

        # 6. Rebuild graph after movement & battery deaths
        G             = build_comm_graph(self.uavs, self.base, self.terrain)
        connected_ids = get_connected_uav_ids(G, self.uavs)

        # 7. Metrics
        alive = [u for u in self.uavs if u.alive]
        n_alive = len(alive)

        isr_cov    = compute_isr_coverage(self.uavs, self.enemies,
                                          connected_ids, self.terrain)
        conn_frac  = len(connected_ids) / n_alive if n_alive else 0.0
        avg_bat    = np.mean([u.battery for u in alive]) if alive else 0.0

        total_switches  = sum(u.switch_count for u in self.uavs)
        switch_delta    = total_switches - self._prev_switch_total
        self._prev_switch_total = total_switches

        obj = compute_objective(self.uavs, self.enemies, connected_ids,
                                self.terrain, isr_cov, switch_delta)

        role_counts = {
            "ISR":          sum(1 for u in alive if u.mode == UAVMode.ISR),
            "Mobile Relay": sum(1 for u in alive if u.mode == UAVMode.MOBILE_RELAY),
            "Static Relay": sum(1 for u in alive if u.mode == UAVMode.STATIC_RELAY),
        }

        metrics = StepMetrics(
            step=self.step_num,
            n_alive=n_alive,
            n_connected=len(connected_ids),
            isr_coverage=isr_cov,
            conn_fraction=conn_frac,
            avg_battery=avg_bat,
            objective=obj,
            kills=kills_this_step,
            role_counts=role_counts,
        )
        self.history.append(metrics)

        if record_snapshot:
            self.snapshots.append((self.step_num, G, set(connected_ids)))

        if record_frame:
            self.frames.append({
                "step":          self.step_num,
                "uav_pos":       [u.pos.copy()   for u in self.uavs],
                "uav_mode":      [u.mode         for u in self.uavs],
                "uav_alive":     [u.alive        for u in self.uavs],
                "uav_battery":   [u.battery      for u in self.uavs],
                "uav_id":        [u.id           for u in self.uavs],
                "enemy_pos":     [e.pos.copy()   for e in self.enemies],
                "enemy_id":      [e.id           for e in self.enemies],
                "edges":         list(G.edges()),
                "connected_ids": set(connected_ids),
                "metrics":       metrics,
            })

        return metrics

    # ------------------------------------------------------------------
    def run(
        self,
        n_steps:           int       = 200,
        snapshot_at_steps: list[int] | None = None,
        animate_every:     int       = 2,
        verbose:           bool      = True,
    ) -> list[StepMetrics]:
        if snapshot_at_steps is None:
            snapshot_at_steps = set()
        else:
            snapshot_at_steps = set(snapshot_at_steps)

        for _ in range(n_steps):
            next_step  = self.step_num + 1
            record_snap  = next_step in snapshot_at_steps
            record_frame = (next_step % animate_every == 0) or next_step == 1
            m = self.step(record_snapshot=record_snap, record_frame=record_frame)

            if verbose and self.step_num % 25 == 0:
                print(
                    f"Step {self.step_num:3d} | "
                    f"Alive: {m.n_alive:2d} | "
                    f"Connected: {m.n_connected:2d} | "
                    f"ISR cov: {m.isr_coverage:.2f} | "
                    f"Battery: {m.avg_battery:.1f}% | "
                    f"Score: {m.objective:.3f}"
                    + (f" | Kills: {m.kills}" if m.kills else "")
                )

        return self.history
