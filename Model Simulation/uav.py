"""
UAV model: state, mode transitions, battery drain, movement.
"""
from __future__ import annotations
import numpy as np
from enum import Enum
from config import (
    DRAIN_RATES, DRAIN_PER_STEP_BASE,
    ISR_COMM_RANGE, MOBILE_RELAY_RANGE, STATIC_RELAY_RANGE,
    UAV_MAX_SPEED, WORLD_SIZE,
    RTB_BATTERY_THRESHOLD,
)


class UAVMode(Enum):
    ISR          = "ISR"
    MOBILE_RELAY = "Mobile Relay"
    STATIC_RELAY = "Static Relay"


_BASE_RANGES = {
    UAVMode.ISR:          ISR_COMM_RANGE,
    UAVMode.MOBILE_RELAY: MOBILE_RELAY_RANGE,
    UAVMode.STATIC_RELAY: STATIC_RELAY_RANGE,
}


class UAV:
    def __init__(self, uav_id: int, pos, mode: UAVMode = UAVMode.ISR):
        self.id   = uav_id
        self.pos  = np.array(pos, dtype=float)
        self.mode = mode

        self.battery      = 100.0
        self.alive        = True
        self.switch_count = 0
        self.steps_in_mode = 0

        self.rtb                    = False
        self.recharge_steps_remaining = 0

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------
    def set_mode(self, new_mode: UAVMode) -> None:
        if new_mode != self.mode:
            self.mode          = new_mode
            self.switch_count += 1
            self.steps_in_mode = 0

    # ------------------------------------------------------------------
    # Battery
    # ------------------------------------------------------------------
    def drain_battery(self) -> None:
        """Called once per timestep."""
        if not self.alive:
            return
        if self.recharge_steps_remaining > 0:
            return  # docked at base — no drain
        rate = DRAIN_RATES[self.mode.value]
        self.battery       -= DRAIN_PER_STEP_BASE * rate
        self.steps_in_mode += 1
        if self.battery <= 0:
            self.battery = 0.0
            self.alive   = False
            return
        if not self.rtb and self.battery <= RTB_BATTERY_THRESHOLD:
            self.rtb = True
            if self.mode != UAVMode.ISR:
                self.set_mode(UAVMode.ISR)  # must be able to move home

    # ------------------------------------------------------------------
    # Communication range (terrain-modified)
    # ------------------------------------------------------------------
    def comm_range(self, terrain_modifier: float = 1.0) -> float:
        return _BASE_RANGES[self.mode] * terrain_modifier

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def move_toward(self, target_pos, max_step: float = UAV_MAX_SPEED) -> None:
        """Move at most max_step meters toward target_pos.  Static relays don't move."""
        if not self.alive or self.mode == UAVMode.STATIC_RELAY:
            return
        delta = np.asarray(target_pos, dtype=float) - self.pos
        dist  = np.linalg.norm(delta)
        if dist < 1.0:
            return
        self.pos += (delta / dist) * min(dist, max_step)
        self._clamp()

    def _clamp(self) -> None:
        self.pos = np.clip(self.pos, 0.0, WORLD_SIZE)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        return (f"UAV(id={self.id}, mode={self.mode.value}, "
                f"bat={self.battery:.1f}%, {status})")
