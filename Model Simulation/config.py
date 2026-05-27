"""
AERIS Simulation Configuration
All tunable constants live here.
"""

# --- World ---
WORLD_SIZE = 10_000.0       # meters (square map)
BASE_POS   = [500.0, 5000.0]

# --- Communication ranges (base, before terrain modifier) ---
BASE_COMM_RANGE       = 2_000.0   # Base station reach
ISR_COMM_RANGE        = 500.0
MOBILE_RELAY_RANGE    = 1_500.0
STATIC_RELAY_RANGE    = 1_500.0

# ISR sensor range (camera/sensor footprint — wider than comm range)
ISR_SENSOR_RANGE = 900.0

# --- Terrain modifier bounds ---
TERRAIN_MOD_MIN = 0.6
TERRAIN_MOD_MAX = 1.3

# --- Battery ---
# ISR mode drains 1.0x.  Lifetime = ISR_LIFETIME_STEPS steps.
ISR_LIFETIME_STEPS   = 180          # ~30 min at DT=10s per step
DRAIN_PER_STEP_BASE  = 100.0 / ISR_LIFETIME_STEPS

DRAIN_RATES = {
    "ISR":          1.0,
    "Mobile Relay": 1.4,
    "Static Relay": 0.6,
}

# --- UAV movement ---
UAV_MAX_SPEED = 150.0   # m per timestep (≈15 m/s at DT=10s)

# --- Enemy ---
ENEMY_SPEED      = 40.0    # m per timestep
ENEMY_KILL_RANGE = 600.0   # kills Static Relay UAVs within this range
ENEMY_THREAT_RANGE = 900.0  # Static Relay flees if enemy this close

# --- Scenario layers ---
SCENARIO_BASELINE = "baseline"
SCENARIO_DUAL_OBJECTIVE = "dual_objective"
DEFAULT_SCENARIO = SCENARIO_BASELINE

# Undefended flank sites used by the dual-objective scenario. Enemies in the
# corresponding y-band will try to reach and damage these sites unless detected.
SECONDARY_OBJECTIVES = {
    "North Site": [50.0, 9000.0],
    "South Site": [50.0, 1000.0],
}
OBJECTIVE_HEALTH = 100.0
OBJECTIVE_ATTACK_RANGE = 450.0
OBJECTIVE_DAMAGE_PER_STEP = 1.5

# --- RTB (Return to Base) ---
RTB_BATTERY_THRESHOLD = 50.0   # UAV begins RTB when battery falls to this %
RTB_ARRIVAL_DIST      = 300.0  # metres from base = "arrived"
RTB_RECHARGE_STEPS    = 10     # steps spent at base before redeploying at 100%

# --- Greedy policy thresholds ---
MAX_NEW_RELAYS_PER_STEP = 1   # cap simultaneous ISR->relay promotions
N_ISR_RESERVE           = 3   # highest-battery ISR UAVs exempt from promotion
# Relays go static if enemy is farther than this.  Set above ENEMY_KILL_RANGE
# but do NOT set equal to ENEMY_THREAT_RANGE or the relay oscillates.
# With enemy spawn at x>=3000 and relay 1 at ~x=1700, separation is ~1300m -> safe.
MIN_ENEMY_DIST_FOR_STATIC  = 1_100.0  # relay 1 (1300m from closest enemy) can go static
STEPS_MOBILE_BEFORE_STATIC = 8
STATIC_BATTERY_THRESHOLD   = 50.0

# --- Adaptive relay-chain planning ---
ENABLE_BRANCHING_CHAIN = True
MAX_BRANCHES = 3
BRANCH_TRIGGER_MIN_ENEMIES = 2
NORTH_FLANK_Y_THRESHOLD = 6500.0
SOUTH_FLANK_Y_THRESHOLD = 3500.0
RELAY_BUDGET_FRACTION = 0.7
CHAIN_HOP_RANGE = 1_300.0
MIN_WAYPOINTS_PER_BRANCH = 1
MAX_WAYPOINTS_PER_BRANCH = 3
OBJECTIVE_INTERCEPT_DISTANCE = 2_400.0
OBJECTIVE_MIN_INTERCEPT_DISTANCE = 1_300.0
OBJECTIVE_URGENCY_BUFFER_STEPS = 15
OBJECTIVE_BRANCH_URGENCY_WEIGHT = 6.0
ENABLE_PERSISTENT_ISR_WATCH = False
OBJECTIVE_WATCH_FRACTION = 0.75
OBJECTIVE_WATCH_SENSOR_MARGIN = 0.9
OBJECTIVE_WATCH_HOLD_SENSOR_FRACTION = 0.85

# --- Emergency intercept ---
# When an objective-bound enemy is within this many steps of reaching its
# objective, claim the nearest connected ISR to chase it directly (bypasses the
# normal band-aware round-robin so urgent threats are not deferred).
ENABLE_EMERGENCY_INTERCEPT = False
EMERGENCY_INTERCEPT_TTI_THRESHOLD = 40.0

# --- Proactive objective defense ---
# Reserve a relay chain + forward-station ISR per objective, positioned along
# the enemy's approach corridor so the ISR's sensor footprint catches
# objective-bound enemies before they arrive. The chain extends along
# (base -> station) so the station ISR stays connected. Costs (relays+ISR)
# UAVs per objective.
ENABLE_OBJECTIVE_DEFENSE = False
# Station sits this many metres back along the enemy approach axis from the
# objective (i.e., between objective and enemy spawn zone).
OBJECTIVE_DEFENSE_STATION_OFFSET = 2000.0
# Number of relays in each defensive chain (base -> station). Two keeps the
# UAV budget tight enough that branching can still cover the center band.
OBJECTIVE_DEFENSE_RELAY_COUNT = 2

# --- Enemy spawn zone ---
# Enemies spawn far from base (x=6000-9000) to give allied forces time to set up
# the relay chain before enemies close in.  Tight y-band keeps all enemies inside
# the ISR sensor corridor so coverage metrics are meaningful.
ENEMY_SPAWN_X_MIN = 6_000.0
ENEMY_SPAWN_X_MAX = 9_000.0
ENEMY_SPAWN_Y_MIN = 4_000.0
ENEMY_SPAWN_Y_MAX = 6_000.0

# --- Strike mechanic ---
# Enemy is eliminated after this many *consecutive* steps of being observed by
# a connected ISR UAV (simulates FOB calling a fire mission).
STRIKE_OBSERVATION_STEPS = 20

# --- Objective weights ---
W_ISR    = 3.0
W_CONN   = 2.0
W_ENERGY = 0.5
W_SWITCH = 0.1

# Honest scoring weights. These score policy quality over the run instead of
# rewarding abrupt coverage jumps caused by enemy attrition.
SCORE_WEIGHTS = {
    "time_weighted_coverage": 3.0,
    "connectivity": 1.0,
    "detection_latency": 2.0,
    "uav_loss_penalty": 1.5,
    "relay_loss_penalty": 2.0,
    "objective_health": 3.0,
    "objective_loss_penalty": 5.0,
}

# --- Simulation ---
# Rebalanced so forward objective defense is structurally viable:
# 14 UAVs leaves enough budget to staff two defensive chains (6 UAVs total)
# AND keep center/branching coverage. 3 + 3 + 3 = 9 enemies keeps the strike
# load below the per-ISR strike-window throughput.
N_UAVS         = 14
N_ENEMIES      = 3    # center band
N_ENEMIES_FLANK = 3   # per flank (top + bottom)
SIM_SEED  = 42
N_STEPS   = 500

# --- Flank enemy spawn zones (top and bottom of map) ---
ENEMY_SPAWN_Y_TOP_MIN = 8_000.0
ENEMY_SPAWN_Y_TOP_MAX = 10_000.0
ENEMY_SPAWN_Y_BOT_MIN = 0.0
ENEMY_SPAWN_Y_BOT_MAX = 2_000.0
