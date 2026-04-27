"""
Communication graph construction and analysis.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from config import ISR_SENSOR_RANGE


def build_comm_graph(uavs, base, terrain) -> nx.Graph:
    """
    Nodes: 'base' + alive UAV ids.
    Edge exists when distance <= max(range_A, range_B).
    Using max lets high-power nodes (Static Relay, Base) reach lower-power ones.
    Terrain modifier is evaluated at each node's position.
    """
    G = nx.Graph()
    G.add_node("base", pos=base.pos.copy())

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
    Fraction of enemies observed by at least one *connected* ISR UAV.
    Returns 1.0 if no enemies exist.
    """
    if not enemies:
        return 1.0

    from uav import UAVMode
    isr_uavs = [u for u in uavs
                if u.alive and u.mode == UAVMode.ISR and u.id in connected_ids]
    covered = set()
    for uav in isr_uavs:
        mod          = terrain.get_modifier(uav.pos)
        sensor_range = ISR_SENSOR_RANGE * mod
        for idx, enemy in enumerate(enemies):
            if np.linalg.norm(uav.pos - enemy.pos) <= sensor_range:
                covered.add(idx)

    return len(covered) / len(enemies)
