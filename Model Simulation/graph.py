"""
Communication graph construction and analysis.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from config import ISR_SENSOR_RANGE, ENABLE_SITE_AS_RELAY, SITE_COMM_RANGE


def build_comm_graph(uavs, base, terrain, objectives=None) -> nx.Graph:
    """
    Nodes: 'base', alive UAV ids, and (when ENABLE_SITE_AS_RELAY) each alive
    critical objective. Edge exists when distance <= max(range_A, range_B).
    Sites carry a zero-distance hardened backhaul edge to base (fiber/satellite
    uplink), so any UAV within SITE_COMM_RANGE of an alive site is connected.
    """
    G = nx.Graph()
    G.add_node("base", pos=base.pos.copy())

    # Sites as forward relay nodes with hardened backhaul to base.
    alive_sites: list[tuple[str, np.ndarray]] = []
    if ENABLE_SITE_AS_RELAY and objectives:
        for obj in objectives:
            if obj.alive:
                node_id = f"site:{obj.name}"
                G.add_node(node_id, pos=obj.pos.copy())
                G.add_edge("base", node_id, dist=0.0)
                alive_sites.append((node_id, obj.pos))

    alive = [u for u in uavs if u.alive]
    for u in alive:
        G.add_node(u.id, pos=u.pos.copy(), mode=u.mode)

    # Base ↔ UAV links
    for u in alive:
        dist = np.linalg.norm(u.pos - base.pos)
        mod  = terrain.get_modifier(u.pos)
        uav_range  = u.comm_range(mod)
        link_range = max(uav_range, base.comm_range)
        if dist <= link_range:
            G.add_edge("base", u.id, dist=dist)

    # Site ↔ UAV links (forward relay coverage)
    for node_id, site_pos in alive_sites:
        for u in alive:
            dist = np.linalg.norm(u.pos - site_pos)
            mod  = terrain.get_modifier(u.pos)
            uav_range = u.comm_range(mod)
            link_range = max(uav_range, SITE_COMM_RANGE)
            if dist <= link_range:
                G.add_edge(node_id, u.id, dist=dist)

    # UAV ↔ UAV links
    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            u1, u2 = alive[i], alive[j]
            dist   = np.linalg.norm(u1.pos - u2.pos)
            mod1   = terrain.get_modifier(u1.pos)
            mod2   = terrain.get_modifier(u2.pos)
            r1     = u1.comm_range(mod1)
            r2     = u2.comm_range(mod2)
            if dist <= max(r1, r2):
                G.add_edge(u1.id, u2.id, dist=dist)

    return G


def get_connected_uav_ids(G: nx.Graph, uavs) -> set:
    """Set of UAV ids that have a path to 'base' in G."""
    connected = set()
    for u in uavs:
        if u.alive and u.id in G:
            try:
                if nx.has_path(G, "base", u.id):
                    connected.add(u.id)
            except nx.NetworkXError:
                pass
    return connected


def compute_isr_coverage(uavs, enemies, connected_ids: set, terrain) -> float:
    """
    Fraction of *alive* enemies observed by at least one *connected* ISR UAV.
    Returns 1.0 if no alive enemies exist.
    """
    alive_enemies = [e for e in enemies if e.alive]
    if not alive_enemies:
        return 1.0

    from uav import UAVMode
    isr_uavs = [u for u in uavs
                if u.alive and u.mode == UAVMode.ISR and u.id in connected_ids]
    covered = set()
    for uav in isr_uavs:
        mod          = terrain.get_modifier(uav.pos)
        sensor_range = ISR_SENSOR_RANGE * mod
        for idx, enemy in enumerate(alive_enemies):
            if np.linalg.norm(uav.pos - enemy.pos) <= sensor_range:
                covered.add(idx)

    return len(covered) / len(alive_enemies)


def compute_coverage_from_observed(enemies, observed_enemy_ids) -> float:
    """Fraction of alive enemies in observed_enemy_ids."""
    alive_enemies = [e for e in enemies if e.alive]
    if not alive_enemies:
        return 1.0
    observed_ids = set(observed_enemy_ids)
    covered = sum(1 for e in alive_enemies if e.id in observed_ids)
    return covered / len(alive_enemies)


def compute_flank_coverage(enemies, observed_enemy_ids) -> dict:
    """Return north/center/south coverage fractions for alive enemies."""
    observed_ids = set(observed_enemy_ids)

    north = [e for e in enemies if e.alive and e.pos[1] > 6500]
    south = [e for e in enemies if e.alive and e.pos[1] < 3500]
    center = [e for e in enemies if e.alive and 3500 <= e.pos[1] <= 6500]

    def frac(group):
        if not group:
            return None
        observed = sum(1 for e in group if e.id in observed_ids)
        return observed / len(group)

    return {"north": frac(north), "center": frac(center), "south": frac(south)}


def get_observed_enemy_ids(uavs, enemies, connected_ids: set, terrain) -> dict:
    """
    For each alive enemy observed by a connected ISR UAV, return a mapping
    enemy.id -> (centroid of observing ISR positions, number of observers).
    Used to drive enemy retreat and update detection counters; the observer
    count enables cooperative-observation strike acceleration.
    """
    from uav import UAVMode
    isr_uavs = [u for u in uavs
                if u.alive and u.mode == UAVMode.ISR and u.id in connected_ids]

    # enemy_id -> list of observing ISR positions
    observers: dict[int, list] = {}
    for uav in isr_uavs:
        mod          = terrain.get_modifier(uav.pos)
        sensor_range = ISR_SENSOR_RANGE * mod
        for enemy in enemies:
            if not enemy.alive:
                continue
            if np.linalg.norm(uav.pos - enemy.pos) <= sensor_range:
                observers.setdefault(enemy.id, []).append(uav.pos.copy())

    # Collapse lists to (centroid, count).
    return {
        eid: (np.mean(positions, axis=0), len(positions))
        for eid, positions in observers.items()
    }
