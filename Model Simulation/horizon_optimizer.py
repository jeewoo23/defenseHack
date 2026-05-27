"""
Finite-horizon (receding-horizon) optimizer for AERIS.

At each replan point, generates a small discrete set of candidate policies,
rolls each one forward H steps in a cloned simulation, scores the terminal
state with the J function from the project writeup, and selects the maximizer.
The selected candidate's overrides are then applied to greedy_policy on the
real sim for K steps before the next replan (model-predictive control).

Score (from aeris_project_update_optimization_focus_2page.tex):

    J = alpha * C_ISR + beta * C_conn + gamma * H_obj + eta * S_strike
        - lambda * D_obj - mu * L_detect - rho * L_UAV - psi * E_battery
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

import optimizer
from config import OBJECTIVE_HEALTH


# --- Receding-horizon parameters --------------------------------------------
HORIZON_STEPS = 20          # H: rollout depth (>= strike-completion window)
REPLAN_EVERY = 5            # K: real-sim steps between replans

# Score weights. Tuned so losing both objectives dominates any other term:
# objective damage cap is 2 * 100 = 200 health, so lambda=1.0 makes total
# objective loss worth -200, larger than the maximum positive contribution.
W_ALPHA  = 3.0      # connected ISR coverage (0..1)
W_BETA   = 1.0      # connectivity fraction  (0..1)
W_GAMMA  = 5.0      # surviving objective health, mean fraction (0..1)
W_ETA    = 4.0      # strikes completed during rollout
W_LAMBDA = 1.0      # objective damage suffered during rollout (health pts)
W_MU     = 0.05     # avg detection latency (steps)
W_RHO    = 3.0      # UAVs lost during rollout
W_PSI    = 1.0      # battery drain (0..1)


@dataclass
class Candidate:
    name: str
    overrides: dict = field(default_factory=dict)


# Small discrete candidate set. Each candidate is just a parameter override on
# the existing greedy policy -- not a separate controller -- per the plan.
CANDIDATES: list[Candidate] = [
    Candidate("keep_branching", {
        "ENABLE_BRANCHING_CHAIN": True,
        "ENABLE_EMERGENCY_INTERCEPT": False,
        "ENABLE_PERSISTENT_ISR_WATCH": False,
        "ENABLE_OBJECTIVE_DEFENSE": False,
    }),
    Candidate("branching_with_intercept", {
        "ENABLE_BRANCHING_CHAIN": True,
        "ENABLE_EMERGENCY_INTERCEPT": True,
        "ENABLE_PERSISTENT_ISR_WATCH": False,
        "ENABLE_OBJECTIVE_DEFENSE": False,
    }),
    Candidate("intercept_no_branch", {
        "ENABLE_BRANCHING_CHAIN": False,
        "ENABLE_EMERGENCY_INTERCEPT": True,
        "ENABLE_PERSISTENT_ISR_WATCH": False,
        "ENABLE_OBJECTIVE_DEFENSE": False,
    }),
    Candidate("single_chain", {
        "ENABLE_BRANCHING_CHAIN": False,
        "ENABLE_EMERGENCY_INTERCEPT": False,
        "ENABLE_PERSISTENT_ISR_WATCH": False,
        "ENABLE_OBJECTIVE_DEFENSE": False,
    }),
    Candidate("watch", {
        "ENABLE_BRANCHING_CHAIN": True,
        "ENABLE_EMERGENCY_INTERCEPT": False,
        "ENABLE_PERSISTENT_ISR_WATCH": True,
        "ENABLE_OBJECTIVE_DEFENSE": False,
    }),
    Candidate("defense_with_branching", {
        "ENABLE_BRANCHING_CHAIN": True,
        "ENABLE_EMERGENCY_INTERCEPT": False,
        "ENABLE_PERSISTENT_ISR_WATCH": False,
        "ENABLE_OBJECTIVE_DEFENSE": True,
    }),
    Candidate("defense_with_intercept", {
        "ENABLE_BRANCHING_CHAIN": True,
        "ENABLE_EMERGENCY_INTERCEPT": True,
        "ENABLE_PERSISTENT_ISR_WATCH": False,
        "ENABLE_OBJECTIVE_DEFENSE": True,
    }),
]


# --- Override plumbing ------------------------------------------------------
class _OptimizerOverrides:
    """Context manager that temporarily mutates optimizer module globals."""

    def __init__(self, overrides: dict):
        self._overrides = overrides
        self._previous: dict = {}

    def __enter__(self):
        for key, value in self._overrides.items():
            self._previous[key] = getattr(optimizer, key)
            setattr(optimizer, key, value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self._previous.items():
            setattr(optimizer, key, value)
        return False


# --- Rollout ----------------------------------------------------------------
def _clone_for_rollout(sim):
    """
    Deepcopy the simulation, but share the immutable terrain (heaviest object)
    to keep rollouts fast.
    """
    saved_terrain = sim.terrain
    saved_history = sim.history
    saved_frames = sim.frames
    saved_snapshots = sim.snapshots

    # Temporarily replace heavy/throwaway state with stubs so deepcopy is cheap.
    sim.terrain = None
    sim.history = []
    sim.frames = []
    sim.snapshots = []

    try:
        clone = copy.deepcopy(sim)
    finally:
        sim.terrain = saved_terrain
        sim.history = saved_history
        sim.frames = saved_frames
        sim.snapshots = saved_snapshots

    clone.terrain = saved_terrain  # shared by reference; Terrain is read-only
    # Force the clone to use greedy policy during rollout. Otherwise its step()
    # would re-enter the horizon planner and recurse infinitely.
    clone.policy = "greedy"
    clone._horizon_selected = None
    clone._horizon_steps_since_replan = 0
    clone._horizon_history = []
    return clone


def _score_state(clone, baseline) -> float:
    """
    Evaluate J on the post-rollout clone state, using `baseline` (a snapshot
    dict captured before the rollout) for delta-style terms.
    """
    history = clone.history
    if not history:
        return 0.0
    terminal = history[-1]

    objectives = getattr(clone, "objectives", []) or []
    if objectives:
        h_obj = float(np.mean([o.health / OBJECTIVE_HEALTH for o in objectives]))
        obj_health_now = sum(o.health for o in objectives)
        d_obj = max(0.0, baseline["obj_health"] - obj_health_now)
    else:
        h_obj = 0.0
        d_obj = 0.0

    s_strike = sum(m.strikes for m in history)
    l_uav = max(0, baseline["alive_uavs"] - terminal.n_alive)

    if terminal.avg_detection_latency is None:
        l_detect = float(len(history))   # nothing detected yet -> worst case
    else:
        l_detect = float(terminal.avg_detection_latency)

    alive_uavs = [u for u in clone.uavs if u.alive]
    if alive_uavs:
        avg_bat = float(np.mean([u.battery for u in alive_uavs])) / 100.0
    else:
        avg_bat = 0.0
    e_battery = max(0.0, 1.0 - avg_bat)

    return (
        W_ALPHA  * float(terminal.isr_coverage)
        + W_BETA   * float(terminal.conn_fraction)
        + W_GAMMA  * h_obj
        + W_ETA    * float(s_strike)
        - W_LAMBDA * float(d_obj)
        - W_MU     * float(l_detect)
        - W_RHO    * float(l_uav)
        - W_PSI    * float(e_battery)
    )


def _baseline_snapshot(sim) -> dict:
    objectives = getattr(sim, "objectives", []) or []
    return {
        "obj_health": sum(o.health for o in objectives),
        "alive_uavs": sum(1 for u in sim.uavs if u.alive),
    }


def _rollout(sim, candidate: Candidate, horizon: int) -> float:
    """Clone, simulate `horizon` steps under candidate, return J."""
    baseline = _baseline_snapshot(sim)
    clone = _clone_for_rollout(sim)

    # Make rollout deterministic without disturbing the real sim's RNG draws.
    rng_state = np.random.get_state()
    np.random.seed(int(sim.step_num) + hash(candidate.name) % 1_000_000)

    try:
        with _OptimizerOverrides(candidate.overrides):
            for _ in range(horizon):
                if not any(u.alive for u in clone.uavs):
                    break
                clone.step(record_snapshot=False, record_frame=False)
    finally:
        np.random.set_state(rng_state)

    return _score_state(clone, baseline)


# --- Public API used by sim.step() ------------------------------------------
def select_candidate(sim,
                     horizon: int = HORIZON_STEPS,
                     candidates: Iterable[Candidate] = CANDIDATES) -> Candidate:
    """Argmax over candidates by H-step rollout score."""
    best_candidate = None
    best_score = -float("inf")
    for candidate in candidates:
        score = _rollout(sim, candidate, horizon)
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate or CANDIDATES[0]


def apply_with_overrides(candidate: Candidate, uavs, connected_ids,
                         enemies, base, terrain, G) -> None:
    """Run the greedy policy on the real sim with the candidate's overrides."""
    with _OptimizerOverrides(candidate.overrides):
        optimizer.greedy_policy(uavs, connected_ids, enemies, base, terrain, G)
