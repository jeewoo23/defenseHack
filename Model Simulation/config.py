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
MOBILE_RELAY_RANGE    = 750.0
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

# --- Greedy policy thresholds ---
MAX_NEW_RELAYS_PER_STEP = 1   # cap simultaneous ISR->relay promotions
N_ISR_RESERVE           = 2   # highest-battery ISR UAVs exempt from promotion
# Relays go static if enemy is farther than this.  Set above ENEMY_KILL_RANGE
# but do NOT set equal to ENEMY_THREAT_RANGE or the relay oscillates.
# With enemy spawn at x>=3000 and relay 1 at ~x=1700, separation is ~1300m -> safe.
MIN_ENEMY_DIST_FOR_STATIC  = 1_100.0  # relay 1 (1300m from closest enemy) can go static
STEPS_MOBILE_BEFORE_STATIC = 8
STATIC_BATTERY_THRESHOLD   = 50.0

# --- Enemy spawn zone ---
# x: start at 3000 so relay 1 (~x=1700) has >1100m safe margin before enemy zone
# y: within a 3 km band around base (y=5000) so relay chain corridor covers all enemies
ENEMY_SPAWN_X_MIN = 3_000.0
ENEMY_SPAWN_X_MAX = 6_500.0
ENEMY_SPAWN_Y_MIN = 3_500.0
ENEMY_SPAWN_Y_MAX = 6_500.0

# --- Objective weights ---
W_ISR    = 3.0
W_CONN   = 2.0
W_ENERGY = 0.5
W_SWITCH = 0.1

# --- Simulation ---
N_UAVS    = 10
N_ENEMIES = 5
SIM_SEED  = 42
N_STEPS   = 200
