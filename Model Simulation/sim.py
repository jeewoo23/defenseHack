"""
Simulation loop.

Timestep ordering:
  1.  Enemies move — using detection state from previous step
         (retreat if detected, else hunt static relays, else patrol)
  2.  Enemies attempt kills on static relays in range
  3.  Build communication graph
  4.  Identify connected UAVs
  5.  Apply greedy policy (role changes + movement decisions)
  6.  Drain batteries (may kill UAVs with empty batteries)
  7.  Rebuild communication graph after movement
  8.  Recompute connectivity + ISR coverage
  9.  Update enemy detection counters; apply FOB strikes (consecutive obs >= threshold)
 10.  Log metrics
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List

import networkx as nx

from config import (
    WORLD_SIZE, BASE_POS, N_UAVS, N_ENEMIES, N_ENEMIES_FLANK, SIM_SEED,
    ENEMY_SPAWN_X_MIN, ENEMY_SPAWN_X_MAX,
    ENEMY_SPAWN_Y_MIN, ENEMY_SPAWN_Y_MAX,
    ENEMY_SPAWN_Y_TOP_MIN, ENEMY_SPAWN_Y_TOP_MAX,
    ENEMY_SPAWN_Y_BOT_MIN, ENEMY_SPAWN_Y_BOT_MAX,
    STRIKE_OBSERVATION_STEPS, SCORE_WEIGHTS, N_STEPS,
    DEFAULT_SCENARIO, SCENARIO_DUAL_OBJECTIVE, SECONDARY_OBJECTIVES,
    OBJECTIVE_ATTACK_RANGE, OBJECTIVE_DAMAGE_PER_STEP, OBJECTIVE_HEALTH,
    RTB_ARRIVAL_DIST, RTB_RECHARGE_STEPS, UAV_MAX_SPEED,
)

from uav import UAV, UAVMode
from world import BaseStation, CriticalObjective, Enemy, Terrain
from graph import (build_comm_graph, get_connected_uav_ids,
                   compute_coverage_from_observed, compute_flank_coverage,
                   get_observed_enemy_ids)
from optimizer import greedy_policy


# ---------------------------------------------------------------------------
# Metric snapshot
# ---------------------------------------------------------------------------
@dataclass
class StepMetrics:
    step:          int
    n_alive:       int
    n_connected:   int
    isr_coverage:  float
    conn_fraction: float
    avg_battery:   float
    objective:     float
    kills:         int    # UAVs killed by enemies this step
    strikes:       int    # enemies eliminated by FOB strike this step
    n_enemies:     int    # alive enemies at end of step
    north_coverage: float | None = None
    center_coverage: float | None = None
    south_coverage: float | None = None
    time_weighted_coverage: float = 0.0
    avg_detection_latency: float | None = None
    detection_latencies: list = field(default_factory=list)
    relays_killed_total: int = 0
    objective_health: dict = field(default_factory=dict)
    objectives_alive: int = 0
    role_counts:   dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Honest score helpers
# ---------------------------------------------------------------------------
def compute_time_weighted_coverage(enemies) -> float:
    """Average fraction of alive-time each enemy has spent under observation."""
    terms = [
        e.observed_steps / e.alive_steps
        for e in enemies
        if e.alive_steps > 0
    ]
    return float(np.mean(terms)) if terms else 0.0


def compute_avg_detection_latency(enemies) -> float | None:
    """Average first-detection latency for enemies that have been detected."""
    latencies = [
        e.first_detection_step - e.spawn_step
        for e in enemies
        if e.spawn_step is not None and e.first_detection_step is not None
    ]
    return float(np.mean(latencies)) if latencies else None


def compute_objective_score(
    enemies, uavs, history, conn_fraction, relays_killed_total, objectives=None
) -> float:
    """Run-to-date score that is less sensitive to enemy attrition timing."""
    w = SCORE_WEIGHTS
    twc = compute_time_weighted_coverage(enemies)

    conn_terms = [m.conn_fraction for m in history] + [conn_fraction]
    avg_conn = sum(conn_terms) / max(1, len(conn_terms))

    avg_latency = compute_avg_detection_latency(enemies)
    latency_for_score = avg_latency if avg_latency is not None else N_STEPS
    latency_term = 1.0 - (latency_for_score / max(1, N_STEPS))
    latency_term = max(0.0, min(1.0, latency_term))

    uavs_lost = sum(1 for u in uavs if not u.alive)
    objectives = objectives or []
    objective_health = (
        np.mean([o.health / OBJECTIVE_HEALTH for o in objectives])
        if objectives else 0.0
    )
    objectives_lost = sum(1 for o in objectives if not o.alive)

    return (
        w["time_weighted_coverage"] * twc
        + w["connectivity"] * avg_conn
        + w["detection_latency"] * latency_term
        + w["objective_health"] * objective_health
        - w["uav_loss_penalty"] * uavs_lost
        - w["relay_loss_penalty"] * relays_killed_total
        - w["objective_loss_penalty"] * objectives_lost
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class Simulation:
    def __init__(
        self,
        n_uavs:         int = N_UAVS,
        n_enemies:      int = N_ENEMIES,
        n_enemies_flank: int = N_ENEMIES_FLANK,
        seed:           int = SIM_SEED,
        scenario:       str = DEFAULT_SCENARIO,
    ):
        rng = np.random.RandomState(seed)

        self.terrain  = Terrain(size=WORLD_SIZE, seed=seed)
        self.base     = BaseStation(pos=BASE_POS)
        self.scenario = scenario
        self.objectives: list[CriticalObjective] = []
        if self.scenario == SCENARIO_DUAL_OBJECTIVE:
            self.objectives = [
                CriticalObjective(name, pos)
                for name, pos in SECONDARY_OBJECTIVES.items()
            ]

        # UAVs start clustered near the base station
        self.uavs: list[UAV] = []
        for i in range(n_uavs):
            offset = rng.uniform(-400, 400, 2)
            pos    = np.clip(np.array(BASE_POS) + offset, 0, WORLD_SIZE)
            self.uavs.append(UAV(uav_id=i, pos=pos, mode=UAVMode.ISR))

        # Enemies spawn far from base — relay chain must extend to reach them
        self.enemies: list[Enemy] = []
        objective_by_name = {o.name: o.pos for o in self.objectives}
        eid = 0
        for _ in range(n_enemies):
            pos = [rng.uniform(ENEMY_SPAWN_X_MIN, ENEMY_SPAWN_X_MAX),
                   rng.uniform(ENEMY_SPAWN_Y_MIN, ENEMY_SPAWN_Y_MAX)]
            self.enemies.append(Enemy(enemy_id=eid, pos=pos))
            eid += 1
        for _ in range(n_enemies_flank):
            pos = [rng.uniform(ENEMY_SPAWN_X_MIN, ENEMY_SPAWN_X_MAX),
                   rng.uniform(ENEMY_SPAWN_Y_TOP_MIN, ENEMY_SPAWN_Y_TOP_MAX)]
            self.enemies.append(Enemy(
                enemy_id=eid,
                pos=pos,
                target_name="North Site" if self.scenario == SCENARIO_DUAL_OBJECTIVE else None,
                target_pos=objective_by_name.get("North Site"),
            ))
            eid += 1
        for _ in range(n_enemies_flank):
            pos = [rng.uniform(ENEMY_SPAWN_X_MIN, ENEMY_SPAWN_X_MAX),
                   rng.uniform(ENEMY_SPAWN_Y_BOT_MIN, ENEMY_SPAWN_Y_BOT_MAX)]
            self.enemies.append(Enemy(
                enemy_id=eid,
                pos=pos,
                target_name="South Site" if self.scenario == SCENARIO_DUAL_OBJECTIVE else None,
                target_pos=objective_by_name.get("South Site"),
            ))
            eid += 1

        self.step_num:           int               = 0
        for enemy in self.enemies:
            enemy.spawn_step = self.step_num

        self.history:            list[StepMetrics] = []
        self.relays_killed_total: int              = 0

        # Snapshot storage: list of (step, G, connected_ids) for selected steps
        self.snapshots: list[tuple] = []

        # Animation frames: lightweight per-step records
        self.frames: list[dict] = []

    # ------------------------------------------------------------------
    def step(self, record_snapshot: bool = False,
             record_frame: bool = False) -> StepMetrics:
        self.step_num += 1

        # 1. Enemies move (detection centroid was set at end of previous step)
        for enemy in self.enemies:
            enemy.move(self.uavs)

        # 1b. Scenario objectives take damage from enemies that reach them.
        if self.objectives:
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                for objective in self.objectives:
                    if objective.alive and np.linalg.norm(enemy.pos - objective.pos) <= OBJECTIVE_ATTACK_RANGE:
                        objective.take_damage(OBJECTIVE_DAMAGE_PER_STEP)

        # 2. Enemy kill attempts (only alive enemies)
        kills_this_step = 0
        for enemy in self.enemies:
            killed = enemy.attempt_kill(self.uavs)
            kills_this_step += len(killed)
            self.relays_killed_total += sum(
                1 for u in self.uavs
                if u.id in killed and u.mode in (UAVMode.MOBILE_RELAY, UAVMode.STATIC_RELAY)
            )

        # 3. Build initial graph
        G             = build_comm_graph(self.uavs, self.base, self.terrain)
        connected_ids = get_connected_uav_ids(G, self.uavs)

        # 4. Greedy policy (modifies modes and positions); pass only alive enemies
        alive_enemies = [e for e in self.enemies if e.alive]
        greedy_policy(self.uavs, connected_ids,
                      alive_enemies, self.base, self.terrain, G)

        # 5. RTB movement and recharge
        _base_arr = np.array(BASE_POS, dtype=float)
        for uav in self.uavs:
            if not uav.alive:
                continue
            if uav.recharge_steps_remaining > 0:
                uav.recharge_steps_remaining -= 1
                if uav.recharge_steps_remaining == 0:
                    uav.battery = 100.0
                    uav.rtb = False
                    uav.set_mode(UAVMode.ISR)
                continue
            if uav.rtb:
                dist = np.linalg.norm(uav.pos - _base_arr)
                if dist <= RTB_ARRIVAL_DIST:
                    uav.recharge_steps_remaining = RTB_RECHARGE_STEPS
                else:
                    delta = _base_arr - uav.pos
                    uav.pos += (delta / np.linalg.norm(delta)) * min(np.linalg.norm(delta), UAV_MAX_SPEED)
                    uav.pos = np.clip(uav.pos, 0.0, WORLD_SIZE)

        # 6. Drain batteries
        for uav in self.uavs:
            uav.drain_battery()

        # 7. Rebuild graph after movement & battery deaths
        G             = build_comm_graph(self.uavs, self.base, self.terrain)
        connected_ids = get_connected_uav_ids(G, self.uavs)

        # 8. Update enemy detection state
        #    get_observed_enemy_ids returns {enemy.id: centroid_of_observers}
        observed = get_observed_enemy_ids(
            self.uavs, self.enemies, connected_ids, self.terrain
        )
        observed_ids = set(observed)
        isr_cov = compute_coverage_from_observed(self.enemies, observed_ids)
        flank_coverage = compute_flank_coverage(self.enemies, observed_ids)

        for enemy in self.enemies:
            if not enemy.alive:
                continue
            enemy.alive_steps += 1
            if enemy.id in observed_ids:
                enemy.observed_steps += 1
                if enemy.first_detection_step is None:
                    enemy.first_detection_step = self.step_num

        strikes_this_step = 0
        for enemy in self.enemies:
            if not enemy.alive:
                enemy._detecting_centroid = None
                continue
            if enemy.id in observed:
                enemy.consecutive_obs += 1
                enemy._detecting_centroid = observed[enemy.id]
                if enemy.consecutive_obs >= STRIKE_OBSERVATION_STEPS:
                    enemy.alive = False
                    enemy._detecting_centroid = None
                    strikes_this_step += 1
            else:
                enemy.consecutive_obs     = 0
                enemy._detecting_centroid = None

        # 9. Metrics
        alive_uavs   = [u for u in self.uavs if u.alive]
        alive_enemies = [e for e in self.enemies if e.alive]
        n_alive       = len(alive_uavs)

        conn_frac  = len(connected_ids) / n_alive if n_alive else 0.0
        avg_bat    = np.mean([u.battery for u in alive_uavs]) if alive_uavs else 0.0

        detection_latencies = [
            e.first_detection_step - e.spawn_step
            for e in self.enemies
            if e.spawn_step is not None and e.first_detection_step is not None
        ]
        time_weighted_coverage = compute_time_weighted_coverage(self.enemies)
        avg_detection_latency = float(np.mean(detection_latencies)) if detection_latencies else None
        obj = compute_objective_score(
            self.enemies, self.uavs, self.history, conn_frac,
            self.relays_killed_total, self.objectives
        )

        role_counts = {
            "ISR":          sum(1 for u in alive_uavs if u.mode == UAVMode.ISR),
            "Mobile Relay": sum(1 for u in alive_uavs if u.mode == UAVMode.MOBILE_RELAY),
            "Static Relay": sum(1 for u in alive_uavs if u.mode == UAVMode.STATIC_RELAY),
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
            strikes=strikes_this_step,
            n_enemies=len(alive_enemies),
            north_coverage=flank_coverage["north"],
            center_coverage=flank_coverage["center"],
            south_coverage=flank_coverage["south"],
            time_weighted_coverage=time_weighted_coverage,
            avg_detection_latency=avg_detection_latency,
            detection_latencies=detection_latencies,
            relays_killed_total=self.relays_killed_total,
            objective_health={o.name: o.health for o in self.objectives},
            objectives_alive=sum(1 for o in self.objectives if o.alive),
            role_counts=role_counts,
        )
        self.history.append(metrics)

        if record_snapshot:
            self.snapshots.append((self.step_num, G, set(connected_ids)))

        if record_frame:
            self.frames.append({
                "step":          self.step_num,
                "uav_pos":       [u.pos.copy()  for u in self.uavs],
                "uav_mode":      [u.mode        for u in self.uavs],
                "uav_alive":     [u.alive       for u in self.uavs],
                "uav_battery":   [u.battery     for u in self.uavs],
                "uav_rtb":       [u.rtb or u.recharge_steps_remaining > 0
                                  for u in self.uavs],
                "uav_id":        [u.id          for u in self.uavs],
                "enemy_pos":     [e.pos.copy()  for e in self.enemies],
                "enemy_alive":   [e.alive       for e in self.enemies],
                "enemy_id":      [e.id          for e in self.enemies],
                "enemy_obs":     [e.consecutive_obs for e in self.enemies],
                "edges":         list(G.edges()),
                "connected_ids": set(connected_ids),
                "objective_name": [o.name for o in self.objectives],
                "objective_pos":  [o.pos.copy() for o in self.objectives],
                "objective_health": [o.health for o in self.objectives],
                "metrics":       metrics,
            })

        return metrics

    # ------------------------------------------------------------------
    def run(
        self,
        n_steps:           int             = 500,
        snapshot_at_steps: list[int] | None = None,
        animate_every:     int             = 2,
        verbose:           bool            = True,
    ) -> list[StepMetrics]:
        if snapshot_at_steps is None:
            snapshot_at_steps = set()
        else:
            snapshot_at_steps = set(snapshot_at_steps)

        for _ in range(n_steps):
            next_step    = self.step_num + 1
            record_snap  = next_step in snapshot_at_steps
            record_frame = (next_step % animate_every == 0) or next_step == 1
            m = self.step(record_snapshot=record_snap, record_frame=record_frame)

            if verbose and self.step_num % 25 == 0:
                extras = []
                if m.kills:
                    extras.append(f"UAV kills: {m.kills}")
                if m.strikes:
                    extras.append(f"Strikes: {m.strikes}")
                print(
                    f"Step {self.step_num:3d} | "
                    f"Alive: {m.n_alive:2d} UAVs | "
                    f"Enemies: {m.n_enemies} | "
                    f"Connected: {m.n_connected:2d} | "
                    f"ISR cov: {m.isr_coverage:.2f} | "
                    f"Battery: {m.avg_battery:.1f}% | "
                    f"Score: {m.objective:.3f}"
                    + (f" | " + ", ".join(extras) if extras else "")
                )

        return self.history
