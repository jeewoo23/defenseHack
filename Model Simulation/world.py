"""
World objects: BaseStation, Enemy, Terrain.
"""
from __future__ import annotations
import numpy as np
from config import (
    WORLD_SIZE, BASE_POS, BASE_COMM_RANGE,
    ENEMY_SPEED, ENEMY_KILL_RANGE,
    TERRAIN_MOD_MIN, TERRAIN_MOD_MAX,
    ISR_SENSOR_RANGE,
)


# ---------------------------------------------------------------------------
# Base station
# ---------------------------------------------------------------------------
class BaseStation:
    def __init__(self, pos=None):
        self.pos        = np.array(pos if pos is not None else BASE_POS, dtype=float)
        self.comm_range = BASE_COMM_RANGE


# ---------------------------------------------------------------------------
# Enemy
# ---------------------------------------------------------------------------
class Enemy:
    """
    Behaviour priority (highest to lowest):
      1. Retreat  — if detected by a connected ISR last step, flee away from
                    the detecting UAV centroid.
      2. Hunt     — move toward nearest Static Relay UAV.
      3. Patrol   — random drift anchored to spawn point.

    Eliminated by a fire-mission strike after STRIKE_OBSERVATION_STEPS
    consecutive steps of ISR detection (handled in sim.py).
    """

    PATROL_RADIUS = 700.0    # max wander distance from spawn when not hunting

    def __init__(self, enemy_id: int, pos):
        self.id            = enemy_id
        self.pos           = np.array(pos, dtype=float)
        self.patrol_center = self.pos.copy()   # home point for idle patrol

        self.alive           = True
        self.consecutive_obs = 0    # consecutive steps observed by connected ISR
        # Centroid of detecting ISR UAVs set by sim at end of each step;
        # used by move() at the start of the next step so enemies react 1 step late.
        self._detecting_centroid: np.ndarray | None = None

    # ------------------------------------------------------------------
    def move(self, uavs) -> None:
        if not self.alive:
            return

        # 1. Retreat from detecting ISR
        if self._detecting_centroid is not None:
            direction = self.pos - self._detecting_centroid
            dist      = np.linalg.norm(direction)
            if dist < 1.0:
                direction = np.array([1.0, 0.0])   # generic eastward retreat
            else:
                direction = direction / dist
            self.pos += direction * ENEMY_SPEED
            self.pos  = np.clip(self.pos, 0.0, WORLD_SIZE)
            return

        from uav import UAVMode
        static_targets = [u for u in uavs
                          if u.alive and u.mode == UAVMode.STATIC_RELAY]
        if static_targets:
            # 2. Hunt nearest Static Relay
            nearest   = min(static_targets,
                            key=lambda u: np.linalg.norm(u.pos - self.pos))
            direction = nearest.pos - self.pos
            dist      = np.linalg.norm(direction)
            if dist > 1.0:
                step = min(ENEMY_SPEED, dist)
                self.pos += (direction / dist) * step
        else:
            # 3. Patrol: random drift pulled back toward home
            drift = np.random.uniform(-ENEMY_SPEED * 0.4, ENEMY_SPEED * 0.4, 2)
            dist_from_home = np.linalg.norm(self.pos - self.patrol_center)
            if dist_from_home > self.PATROL_RADIUS:
                toward_home = self.patrol_center - self.pos
                drift += 0.5 * (toward_home / dist_from_home) * ENEMY_SPEED
            self.pos += drift

        self.pos = np.clip(self.pos, 0.0, WORLD_SIZE)

    # ------------------------------------------------------------------
    def attempt_kill(self, uavs) -> list[int]:
        """Return list of UAV ids eliminated this step."""
        if not self.alive:
            return []
        from uav import UAVMode
        killed = []
        for uav in uavs:
            if not uav.alive or uav.mode != UAVMode.STATIC_RELAY:
                continue
            if np.linalg.norm(uav.pos - self.pos) <= ENEMY_KILL_RANGE:
                uav.alive   = False
                uav.battery = 0.0
                killed.append(uav.id)
        return killed

    def __repr__(self) -> str:
        status = "alive" if self.alive else "struck"
        return f"Enemy(id={self.id}, pos=({self.pos[0]:.0f},{self.pos[1]:.0f}), {status})"


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------
class Terrain:
    """ 
    Smooth terrain using a superposition of Gaussian hills and valleys.
    Returns a modifier in [TERRAIN_MOD_MIN, TERRAIN_MOD_MAX] at any position.
    Higher modifier = better comm/sensor range (favorable high ground).
    No scipy required.
    """

    def __init__(self, size: float = WORLD_SIZE, n_features: int = 25, seed: int = 42):
        self.size = size
        rng       = np.random.RandomState(seed)

        self.features: list[tuple] = []
        for _ in range(n_features):
            center    = rng.uniform(0, size, 2)
            radius    = rng.uniform(800, 3_500)
            amplitude = rng.uniform(-0.25, 0.35)   # negative = valley
            self.features.append((center, radius, amplitude))

        self._base = 0.9

    # ------------------------------------------------------------------
    def get_modifier(self, pos) -> float:
        pos = np.asarray(pos, dtype=float)
        val = self._base
        for center, radius, amp in self.features:
            d   = np.linalg.norm(pos - center)
            val += amp * np.exp(-0.5 * (d / radius) ** 2)
        return float(np.clip(val, TERRAIN_MOD_MIN, TERRAIN_MOD_MAX))

    # ------------------------------------------------------------------
    def build_grid(self, resolution: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, Y, Z) arrays suitable for imshow / contourf."""
        xs   = np.linspace(0, self.size, resolution)
        ys   = np.linspace(0, self.size, resolution)
        grid = np.zeros((resolution, resolution))
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                grid[i, j] = self.get_modifier([x, y])
        X, Y = np.meshgrid(xs, ys)
        return X, Y, grid
