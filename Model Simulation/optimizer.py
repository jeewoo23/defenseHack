"""
Greedy FDIR policy and objective function.

Relay chain management (proactive):
  - Compute how many hops are needed to reach the enemy centroid.
  - Immediately promote ISR UAVs to fill any missing relay slots.
  - Move each mobile relay toward its assigned chain waypoint.
  - Once stable (steps_in_mode >= threshold) and safe (far from enemies),
    upgrade mobile relays to static relays for extended range.

ISR assignment (round-robin):
  - Each connected ISR UAV is assigned to a DIFFERENT enemy so UAVs spread
    across the battlefield instead of clustering on the same 2-3 targets.
  - Disconnected ISR UAVs retreat toward the nearest relay.
"""
from __future__ import annotations
import numpy as np
from config import (
    ISR_SENSOR_RANGE, ENEMY_THREAT_RANGE,
    MIN_ENEMY_DIST_FOR_STATIC, STEPS_MOBILE_BEFORE_STATIC,
    STATIC_BATTERY_THRESHOLD,
    MAX_NEW_RELAYS_PER_STEP, N_ISR_RESERVE,
    W_ISR, W_CONN, W_ENERGY, W_SWITCH,
)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def compute_objective(
    uavs, enemies, connected_ids: set, terrain,
    isr_coverage: float, switch_delta: int,
) -> float:
    alive = [u for u in uavs if u.alive]
    if not alive:
        return 0.0
    conn_frac = len(connected_ids) / len(alive)
    avg_bat   = np.mean([u.battery for u in alive]) / 100.0
    return (W_ISR    * isr_coverage
            + W_CONN   * conn_frac
            - W_ENERGY * (1.0 - avg_bat)
            - W_SWITCH * switch_delta)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _relay_chain_targets(base_pos, target_pos, n_relays: int) -> list:
    """Evenly-spaced relay waypoints between base and enemy centroid."""
    return [
        base_pos + (target_pos - base_pos) * (k + 1) / (n_relays + 1)
        for k in range(n_relays)
    ]


def _assign_relays_to_targets(relay_uavs, targets: list) -> dict:
    """Match relay UAVs to chain targets in distance-sorted order."""
    if not relay_uavs or not targets:
        return {}
    origin   = np.zeros(2)
    r_sorted = sorted(relay_uavs, key=lambda u: np.linalg.norm(u.pos - origin))
    t_sorted = sorted(targets,    key=lambda p: np.linalg.norm(p  - origin))
    return {r.id: t for r, t in zip(r_sorted, t_sorted)}


# ---------------------------------------------------------------------------
# Greedy policy — called once per timestep
# ---------------------------------------------------------------------------
def greedy_policy(uavs, connected_ids: set, enemies, base, terrain, G) -> None:
    from uav import UAVMode
    alive = [u for u in uavs if u.alive]
    if not alive:
        return

    base_pos = np.array(base.pos, dtype=float)

    # Static relays do NOT flee — they hold position and risk being killed.
    # When an enemy eliminates a static relay the policy detects the loss
    # next step (fewer relays than n_relays_need) and assigns a replacement.

    # ------------------------------------------------------------------
    # 2. Compute relay chain — target = centroid of ALL enemies
    # ------------------------------------------------------------------
    if enemies:
        target_pos = np.mean([e.pos for e in enemies], axis=0)
    else:
        target_pos = base_pos + np.array([4000.0, 0.0])

    chain_dist = max(0.0, np.linalg.norm(target_pos - base_pos) - base.comm_range)

    # Plan using static relay hop range (1300 m) so chain reaches far enough.
    # As mobile relays upgrade to static the plan becomes self-fulfilling.
    HOP_RANGE     = 1_300.0
    n_relays_need = max(2, min(4, int(chain_dist / HOP_RANGE) + 1))
    n_relays_need = min(n_relays_need, max(1, len(alive) // 2))

    # ------------------------------------------------------------------
    # 3. Promote ISR UAVs to Mobile Relay to fill empty chain slots
    # ------------------------------------------------------------------
    current_relays = [u for u in alive
                      if u.mode in (UAVMode.MOBILE_RELAY, UAVMode.STATIC_RELAY)]
    n_short = n_relays_need - len(current_relays)

    if n_short > 0:
        relay_targets  = _relay_chain_targets(base_pos, target_pos, n_relays_need)
        covered_slots: set[int] = set()
        for rel in current_relays:
            if len(covered_slots) >= n_relays_need:
                break
            best_i = min(range(n_relays_need),
                         key=lambda i: np.linalg.norm(rel.pos - relay_targets[i]))
            covered_slots.add(best_i)

        open_targets = [relay_targets[i]
                        for i in range(n_relays_need) if i not in covered_slots]
        # Sort ISR pool by battery ascending; reserve the top N_ISR_RESERVE
        # highest-battery UAVs so they stay available as long-endurance ISR.
        all_isr = sorted(
            [u for u in alive if u.mode == UAVMode.ISR],
            key=lambda u: u.battery,
        )
        isr_pool = all_isr[: max(0, len(all_isr) - N_ISR_RESERVE)]

        promoted = 0
        for tgt in open_targets:
            if not isr_pool or promoted >= MAX_NEW_RELAYS_PER_STEP:
                break
            best = min(isr_pool, key=lambda u: np.linalg.norm(u.pos - tgt))
            best.set_mode(UAVMode.MOBILE_RELAY)
            isr_pool.remove(best)
            promoted += 1

    # ------------------------------------------------------------------
    # 4. Move Mobile Relays toward their chain waypoints
    # ------------------------------------------------------------------
    relay_targets = _relay_chain_targets(base_pos, target_pos, n_relays_need)
    mobile_relays = [u for u in alive if u.mode == UAVMode.MOBILE_RELAY]
    assignment    = _assign_relays_to_targets(mobile_relays, relay_targets)
    for uav in mobile_relays:
        if uav.id in assignment:
            uav.move_toward(assignment[uav.id])

    # ------------------------------------------------------------------
    # 5. Upgrade stable, safe Mobile Relays → Static Relay
    # ------------------------------------------------------------------
    for uav in alive:
        if uav.mode != UAVMode.MOBILE_RELAY:
            continue
        if uav.steps_in_mode < STEPS_MOBILE_BEFORE_STATIC:
            continue
        if uav.battery < STATIC_BATTERY_THRESHOLD:
            continue
        if uav.id not in connected_ids:
            continue
        min_e = min((np.linalg.norm(uav.pos - e.pos) for e in enemies),
                    default=float("inf"))
        if min_e > MIN_ENEMY_DIST_FOR_STATIC:
            uav.set_mode(UAVMode.STATIC_RELAY)

    # ------------------------------------------------------------------
    # 6. ISR round-robin enemy assignment
    #    Each connected ISR UAV is matched to a DIFFERENT enemy so UAVs
    #    spread across the battlefield rather than clustering.
    # ------------------------------------------------------------------
    isr_connected = [u for u in alive
                     if u.mode == UAVMode.ISR and u.id in connected_ids]
    isr_disconn   = [u for u in alive
                     if u.mode == UAVMode.ISR and u.id not in connected_ids]
    relay_set     = {u.id for u in alive
                     if u.mode in (UAVMode.MOBILE_RELAY, UAVMode.STATIC_RELAY)}

    # Greedy one-to-one matching: nearest ISR to each enemy
    remaining_enemy_idx = list(range(len(enemies)))
    isr_target: dict[int, int] = {}   # uav.id -> enemy index

    for uav in sorted(isr_connected,
                      key=lambda u: min(
                          (np.linalg.norm(u.pos - enemies[i].pos)
                           for i in remaining_enemy_idx),
                          default=0.0)):
        if not remaining_enemy_idx:
            break
        best_idx = min(remaining_enemy_idx,
                       key=lambda i: np.linalg.norm(uav.pos - enemies[i].pos))
        isr_target[uav.id] = best_idx
        remaining_enemy_idx.remove(best_idx)

    # Any leftover ISR UAVs target whichever enemy still needs more coverage
    for uav in isr_connected:
        if uav.id in isr_target:
            continue
        if enemies:
            best_idx = min(range(len(enemies)),
                           key=lambda i: np.linalg.norm(uav.pos - enemies[i].pos))
            isr_target[uav.id] = best_idx

    # Move connected ISR UAVs toward their assigned enemy
    for uav in isr_connected:
        if uav.id not in isr_target:
            continue
        enemy      = enemies[isr_target[uav.id]]
        mod        = terrain.get_modifier(uav.pos)
        sens_range = ISR_SENSOR_RANGE * mod
        if np.linalg.norm(uav.pos - enemy.pos) > sens_range * 0.85:
            uav.move_toward(enemy.pos)
        # else: within observation range — hold position

    # Disconnected ISR UAVs retreat toward nearest relay
    for uav in isr_disconn:
        anchors = [u for u in alive if u.id in relay_set]
        if anchors:
            nearest = min(anchors, key=lambda u: np.linalg.norm(u.pos - uav.pos))
            uav.move_toward(nearest.pos)
        else:
            uav.move_toward(base.pos)
