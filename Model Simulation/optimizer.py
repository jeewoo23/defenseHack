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
    ENABLE_BRANCHING_CHAIN, MAX_BRANCHES, BRANCH_TRIGGER_MIN_ENEMIES,
    NORTH_FLANK_Y_THRESHOLD, SOUTH_FLANK_Y_THRESHOLD, RELAY_BUDGET_FRACTION,
    CHAIN_HOP_RANGE, MIN_WAYPOINTS_PER_BRANCH, MAX_WAYPOINTS_PER_BRANCH,
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


def _enemy_band(enemy) -> str:
    if enemy.pos[1] > NORTH_FLANK_Y_THRESHOLD:
        return "north"
    if enemy.pos[1] < SOUTH_FLANK_Y_THRESHOLD:
        return "south"
    return "center"


def _target_for_cluster(enemies) -> np.ndarray:
    enemy_centroid = np.mean([e.pos for e in enemies], axis=0)
    target_positions = [
        np.asarray(e.target_pos, dtype=float)
        for e in enemies
        if getattr(e, "target_pos", None) is not None
    ]
    if target_positions:
        objective_centroid = np.mean(target_positions, axis=0)
        # Defend the approach corridor, not the objective's doorstep.
        return 0.65 * enemy_centroid + 0.35 * objective_centroid
    return enemy_centroid


def _cluster_enemies_by_band(enemies) -> dict:
    clusters = {"center": [], "north": [], "south": []}
    for enemy in enemies:
        clusters[_enemy_band(enemy)].append(enemy)
    return {
        band: members
        for band, members in clusters.items()
        if len(members) >= BRANCH_TRIGGER_MIN_ENEMIES
    }


def _branch_priority(name: str) -> int:
    return {"center": 0, "north": 1, "south": 2}.get(name, 3)


def _has_objective_target(enemies) -> bool:
    return any(getattr(e, "target_pos", None) is not None for e in enemies)


def _branch_weight(enemies) -> float:
    if not enemies:
        return 0.0
    objective_members = [
        e for e in enemies
        if getattr(e, "target_pos", None) is not None
    ]
    if not objective_members:
        return float(len(enemies))
    urgency = sum(
        8_000.0 / max(1_000.0, np.linalg.norm(e.pos - e.target_pos))
        for e in objective_members
    )
    return float(len(enemies)) + urgency


def _branch_waypoints(base_pos, enemies, relay_budget: int) -> list:
    """
    Build adaptive waypoint branches toward active enemy bands/objective sites.
    Falls back to the original centerline chain when threats are not split.
    """
    if relay_budget <= 0:
        return []

    fallback_target = np.array([ENEMY_SPAWN_X_MAX, BASE_POS[1]], dtype=float)
    if not ENABLE_BRANCHING_CHAIN or not enemies:
        n_relays = max(2, min(5, int(
            max(0.0, np.linalg.norm(fallback_target - base_pos) - 2_000.0) / CHAIN_HOP_RANGE
        ) + 1))
        return _relay_chain_targets(base_pos, fallback_target, min(n_relays, relay_budget))

    clusters = _cluster_enemies_by_band(enemies)
    if not clusters:
        return _relay_chain_targets(base_pos, fallback_target, min(2, relay_budget))

    active = sorted(
        clusters.items(),
        key=lambda item: (
            0 if _has_objective_target(item[1]) else 1,
            _branch_priority(item[0]),
            -len(item[1]),
        ),
    )[:MAX_BRANCHES]

    branches = []
    for name, members in active:
        target = _target_for_cluster(members)
        dist = np.linalg.norm(target - base_pos)
        n_waypoints = int(max(MIN_WAYPOINTS_PER_BRANCH, dist / CHAIN_HOP_RANGE))
        n_waypoints = min(MAX_WAYPOINTS_PER_BRANCH, n_waypoints)
        branches.append({
            "name": name,
            "target": target,
            "n_waypoints": n_waypoints,
            "weight": max(1.0, _branch_weight(members)),
        })

    total_requested = sum(b["n_waypoints"] for b in branches)
    if total_requested <= relay_budget:
        allocations = {b["name"]: b["n_waypoints"] for b in branches}
    else:
        allocations = {}
        remaining = relay_budget
        total_weight = sum(b["weight"] for b in branches)
        for b in branches:
            share = max(1, round(relay_budget * b["weight"] / total_weight))
            share = min(share, b["n_waypoints"], remaining)
            allocations[b["name"]] = share
            remaining -= share
        while remaining > 0:
            candidates = [b for b in branches if allocations[b["name"]] < b["n_waypoints"]]
            if not candidates:
                break
            best = max(candidates, key=lambda b: b["weight"])
            allocations[best["name"]] += 1
            remaining -= 1

    waypoints = []
    for branch in sorted(branches, key=lambda b: _branch_priority(b["name"])):
        count = allocations.get(branch["name"], 0)
        waypoints.extend(_relay_chain_targets(base_pos, branch["target"], count))
    return waypoints


def _assign_relays_to_targets(relay_uavs, targets: list) -> dict:
    """
    Greedily match each target to the nearest unassigned relay.
    """
    if not relay_uavs or not targets:
        return {}
    remaining = list(relay_uavs)
    assignment = {}
    for target in targets:
        if not remaining:
            break
        best = min(remaining, key=lambda u: np.linalg.norm(u.pos - target))
        assignment[best.id] = target
        remaining.remove(best)
    return assignment


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
    alive = [u for u in uavs if u.alive and not u.rtb
             and u.recharge_steps_remaining == 0]
    if not alive:
        return

    base_pos = np.array(base.pos, dtype=float)

    # ------------------------------------------------------------------
    # 2. Adaptive branch targets. Split relay waypoints across active
    #    north/center/south threat bands, falling back to the old centerline
    #    chain when the situation is not multi-axis.
    # ------------------------------------------------------------------
    relay_budget = int(max(1, len(alive) * RELAY_BUDGET_FRACTION))
    relay_budget = min(relay_budget, max(1, len(alive) - N_ISR_RESERVE))
    relay_targets = _branch_waypoints(base_pos, enemies, relay_budget)
    static_relays  = [u for u in alive if u.mode == UAVMode.STATIC_RELAY]
    mobile_relays  = [u for u in alive if u.mode == UAVMode.MOBILE_RELAY]

    # Waypoints not yet locked in by a static relay
    open_waypoints = [wp for wp in relay_targets if not _wp_covered(wp, static_relays)]

    # ------------------------------------------------------------------
    # 3. Promote ISR UAVs to fill open slots that have no mobile relay
    # ------------------------------------------------------------------
    relay_slots = max(0, relay_budget - len(static_relays) - len(mobile_relays))
    n_short = min(max(0, len(open_waypoints) - len(mobile_relays)), relay_slots)
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

    # Band-aware matching: seed one ISR per active threat band, then fill the
    # rest greedily. This keeps flank pressure from being starved by centerline
    # nearest-neighbor assignments.
    remaining_enemy_idx = list(range(len(enemies)))
    remaining_isr = list(isr_connected)
    isr_target: dict[int, int] = {}

    active_bands = sorted(
        {_enemy_band(enemy) for enemy in enemies},
        key=lambda band: (
            0 if any(
                _enemy_band(e) == band and getattr(e, "target_pos", None) is not None
                for e in enemies
            ) else 1,
            min(
                (
                    np.linalg.norm(e.pos - e.target_pos)
                    for e in enemies
                    if _enemy_band(e) == band and getattr(e, "target_pos", None) is not None
                ),
                default=float("inf"),
            ),
            _branch_priority(band),
        ),
    )
    band_order = active_bands or ["center", "north", "south"]
    for band in band_order:
        band_indices = [i for i in remaining_enemy_idx if _enemy_band(enemies[i]) == band]
        if not band_indices or not remaining_isr:
            continue
        band_center = np.mean([enemies[i].pos for i in band_indices], axis=0)
        uav = min(remaining_isr, key=lambda u: np.linalg.norm(u.pos - band_center))
        best_idx = min(band_indices, key=lambda i: np.linalg.norm(uav.pos - enemies[i].pos))
        isr_target[uav.id] = best_idx
        remaining_isr.remove(uav)
        remaining_enemy_idx.remove(best_idx)

    for uav in sorted(remaining_isr,
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
