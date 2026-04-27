"""
Greedy FDIR policy and objective function.

Relay chain management (proactive):
  - Chain target = far edge of enemy spawn zone (fixed, never shrinks).
  - Only assign mobile relays to UNCOVERED waypoints (no static relay there).
  - Relays go static only once they've arrived at their assigned waypoint.
  - ISR advance is gated by a link-safety margin to prevent oscillation.

ISR assignment (round-robin):
  - Each connected ISR UAV matched to a different enemy.
  - Disconnected ISR UAVs retreat toward nearest relay.
"""
from __future__ import annotations
import numpy as np
from config import (
    ISR_SENSOR_RANGE, ISR_COMM_RANGE, UAV_MAX_SPEED,
    MIN_ENEMY_DIST_FOR_STATIC, STEPS_MOBILE_BEFORE_STATIC,
    STATIC_BATTERY_THRESHOLD,
    MAX_NEW_RELAYS_PER_STEP, N_ISR_RESERVE,
    ENEMY_SPAWN_X_MAX, BASE_POS,
    W_ISR, W_CONN, W_ENERGY, W_SWITCH,
)

# Relay must be within this distance of its assigned waypoint before going static.
_WAYPOINT_ARRIVAL_DIST = 160.0      # ~1 step of travel
# ISR won't advance if it's within this margin of the link-range limit.
_LINK_SAFETY_MARGIN    = UAV_MAX_SPEED * 1.5   # 225 m


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
    """Evenly-spaced relay waypoints between base and target."""
    return [
        base_pos + (target_pos - base_pos) * (k + 1) / (n_relays + 1)
        for k in range(n_relays)
    ]


def _assign_relays_to_targets(relay_uavs, targets: list) -> dict:
    """
    Match relay UAVs to chain targets in distance-sorted order.
    Closest relay (by distance from map origin) maps to closest target.
    """
    if not relay_uavs or not targets:
        return {}
    origin   = np.zeros(2)
    r_sorted = sorted(relay_uavs, key=lambda u: np.linalg.norm(u.pos - origin))
    t_sorted = sorted(targets,    key=lambda p: np.linalg.norm(p  - origin))
    return {r.id: t for r, t in zip(r_sorted, t_sorted)}


def _wp_covered(wp, static_relays) -> bool:
    """True if any static relay is sitting at this waypoint."""
    return any(
        np.linalg.norm(s.pos - wp) <= _WAYPOINT_ARRIVAL_DIST * 2
        for s in static_relays
    )


# ---------------------------------------------------------------------------
# Greedy policy — called once per timestep
# ---------------------------------------------------------------------------
def greedy_policy(uavs, connected_ids: set, enemies, base, terrain, G) -> None:
    from uav import UAVMode
    alive = [u for u in uavs if u.alive]
    if not alive:
        return

    base_pos = np.array(base.pos, dtype=float)

    # ------------------------------------------------------------------
    # 2. Fixed strategic chain target = far edge of enemy spawn zone.
    #    A fixed target keeps waypoint positions stable for the full sim.
    # ------------------------------------------------------------------
    target_pos = np.array([ENEMY_SPAWN_X_MAX, BASE_POS[1]], dtype=float)

    chain_dist    = max(0.0, np.linalg.norm(target_pos - base_pos) - base.comm_range)
    HOP_RANGE     = 1_300.0
    n_relays_need = max(2, min(5, int(chain_dist / HOP_RANGE) + 1))
    n_relays_need = min(n_relays_need, max(1, len(alive) // 2))

    relay_targets  = _relay_chain_targets(base_pos, target_pos, n_relays_need)
    static_relays  = [u for u in alive if u.mode == UAVMode.STATIC_RELAY]
    mobile_relays  = [u for u in alive if u.mode == UAVMode.MOBILE_RELAY]

    # Waypoints not yet locked in by a static relay
    open_waypoints = [wp for wp in relay_targets if not _wp_covered(wp, static_relays)]

    # ------------------------------------------------------------------
    # 3. Promote ISR UAVs to fill open slots that have no mobile relay
    # ------------------------------------------------------------------
    n_short = max(0, len(open_waypoints) - len(mobile_relays))
    if n_short > 0:
        all_isr  = sorted([u for u in alive if u.mode == UAVMode.ISR],
                          key=lambda u: u.battery)
        isr_pool = all_isr[: max(0, len(all_isr) - N_ISR_RESERVE)]

        # Waypoints that have neither a static nor a mobile relay en route
        unserved = open_waypoints[len(mobile_relays):]

        promoted = 0
        for tgt in unserved:
            if not isr_pool or promoted >= MAX_NEW_RELAYS_PER_STEP:
                break
            best = min(isr_pool, key=lambda u: np.linalg.norm(u.pos - tgt))
            best.set_mode(UAVMode.MOBILE_RELAY)
            isr_pool.remove(best)
            promoted += 1

    # ------------------------------------------------------------------
    # 4. Assign mobile relays to OPEN (uncovered) waypoints only
    # ------------------------------------------------------------------
    mobile_relays = [u for u in alive if u.mode == UAVMode.MOBILE_RELAY]
    assignment    = _assign_relays_to_targets(mobile_relays, open_waypoints)
    for uav in mobile_relays:
        if uav.id in assignment:
            uav.move_toward(assignment[uav.id])

    # ------------------------------------------------------------------
    # 5. Upgrade Mobile Relays → Static once at assigned waypoint
    # ------------------------------------------------------------------
    for uav in mobile_relays:
        if uav.steps_in_mode < STEPS_MOBILE_BEFORE_STATIC:
            continue
        if uav.battery < STATIC_BATTERY_THRESHOLD:
            continue
        if uav.id not in connected_ids:
            continue
        if uav.id in assignment:
            if np.linalg.norm(uav.pos - assignment[uav.id]) > _WAYPOINT_ARRIVAL_DIST:
                continue   # still en route — don't lock in yet
        min_e = min((np.linalg.norm(uav.pos - e.pos) for e in enemies),
                    default=float("inf"))
        if min_e > MIN_ENEMY_DIST_FOR_STATIC:
            uav.set_mode(UAVMode.STATIC_RELAY)

    # ------------------------------------------------------------------
    # 6. ISR round-robin enemy assignment
    # ------------------------------------------------------------------
    isr_connected = [u for u in alive
                     if u.mode == UAVMode.ISR and u.id in connected_ids]
    isr_disconn   = [u for u in alive
                     if u.mode == UAVMode.ISR and u.id not in connected_ids]
    relay_set     = {u.id for u in alive
                     if u.mode in (UAVMode.MOBILE_RELAY, UAVMode.STATIC_RELAY)}
    relay_uavs    = [u for u in alive if u.id in relay_set]

    # Greedy one-to-one matching
    remaining_enemy_idx = list(range(len(enemies)))
    isr_target: dict[int, int] = {}

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

    for uav in isr_connected:
        if uav.id in isr_target:
            continue
        if enemies:
            best_idx = min(range(len(enemies)),
                           key=lambda i: np.linalg.norm(uav.pos - enemies[i].pos))
            isr_target[uav.id] = best_idx

    # Move ISR toward assigned enemy, with link-safety guard to stop oscillation
    for uav in isr_connected:
        if uav.id not in isr_target:
            continue
        enemy      = enemies[isr_target[uav.id]]
        mod        = terrain.get_modifier(uav.pos)
        sens_range = ISR_SENSOR_RANGE * mod
        if np.linalg.norm(uav.pos - enemy.pos) <= sens_range * 0.85:
            continue   # within observation range — hold

        if relay_uavs:
            nearest_relay = min(relay_uavs,
                                key=lambda r: np.linalg.norm(r.pos - uav.pos))
            relay_range   = nearest_relay.comm_range(
                terrain.get_modifier(nearest_relay.pos))
            isr_range     = ISR_COMM_RANGE * terrain.get_modifier(uav.pos)
            max_link      = max(relay_range, isr_range)
            dist_to_relay = np.linalg.norm(uav.pos - nearest_relay.pos)
            if dist_to_relay > max_link - _LINK_SAFETY_MARGIN:
                continue   # advancing would risk disconnecting next step

        uav.move_toward(enemy.pos)

    # Disconnected ISR UAVs retreat toward nearest relay
    for uav in isr_disconn:
        anchors = [u for u in alive if u.id in relay_set]
        if anchors:
            nearest = min(anchors, key=lambda u: np.linalg.norm(u.pos - uav.pos))
            uav.move_toward(nearest.pos)
        else:
            uav.move_toward(base.pos)
