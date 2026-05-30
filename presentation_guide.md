# AERIS Presentation Build Guide

## Context for Claude

Help me build a PowerPoint presentation for my final project video. I will go slide by slide. For each slide, tell me exactly what to type, where to place images, and any formatting tips. When I say "next slide" move to the next one. When I say "done" on a slide, confirm and move on.

The presentation is a **recorded video** (screen capture over slides), submitted as .mp4. It must be **5–10 minutes** for a solo presenter. Target is ~8 minutes.

---

## Project Summary (read this so you understand the content)

**Project name:** AERIS — Autonomous Relay-Enabled ISR System

**One-line summary:** A fleet of UAVs must conduct surveillance while maintaining a live communication relay chain back to the operator, even as enemies destroy relay nodes and attack critical ground sites. I built a receding-horizon MPC controller that outperforms a greedy baseline in hard scenarios.

**Key concepts:**
- UAVs have 3 modes: ISR (blue, wide sensor, short comm), Mobile Relay (green, moves, bridges gaps), Static Relay (orange, stationary, killable by enemies)
- A UAV's observations only count if connected to base through an unbroken relay chain
- Two critical sites (North + South) act as forward relay nodes AND charging stations (FARP). Lose the site = lose the relay node for that flank
- Enemies hunt static relays and attack the critical sites
- Battery lasts ~30 min; UAVs must return to base or FARP to recharge

**Greedy baseline:** Clusters enemies by geographic band (north/center/south), builds relay chains toward each cluster, assigns ISR drones round-robin with emergency intercept near objectives. One-step-ahead only.

**Horizon MPC controller:** Every 5 timesteps, deep-copies the simulation and runs 7 candidate policies forward 20 steps each. Scores terminal state with multi-objective function J (ISR coverage + connectivity + site health + strikes - UAV losses - battery). Picks best candidate, executes 5 steps, repeats.

**7 candidate policies:** branching, branching+intercept, intercept only, single chain, watch, defense+branching, defense+intercept.

**3 key innovations:**
1. Sites-as-relays: alive objectives act as forward relay nodes
2. Cooperative observation: 2 ISRs on same enemy advance strike counter faster (capped at 2)
3. Persistent area coverage: grid cells go stale after 50 steps, penalizes clustering

**Experiments:** 4 force configs (10v15 outnumbered, 12v12 even, 14v9 nominal, 18v6 abundant), 4 controllers, 20 random seeds each = 320 runs, 500 timesteps per run.

**Key results:**
- Horizon wins outnumbered (obj score -6.48 vs greedy -25, site health 60% vs 20%)
- Horizon wins nominal 14v9 (+1.37, first positive score)
- Greedy wins easy scenarios (12v12, 18v6) where lookahead isn't needed
- Sites-as-relays: mean site health 22-74% → 48-97% across scenarios

---

## Image Files (all relative to project root)

All images are in: `Model Simulation/outputs/`

| Variable Name | File Path |
|---|---|
| SNAP_BASELINE | `Model Simulation/outputs/images/aeris_snapshots.png` |
| SNAP_DUAL | `Model Simulation/outputs/images/aeris_dual_objective_snapshots.png` |
| SNAP_HORIZON | `Model Simulation/outputs/images/aeris_dual_objective_horizon_snapshots.png` |
| METRICS_HORIZON | `Model Simulation/outputs/images/aeris_dual_objective_horizon_metrics.png` |
| METRICS_DUAL | `Model Simulation/outputs/images/aeris_dual_objective_metrics.png` |
| BENCH_GREEDY_OUT | `Model Simulation/outputs/benchmark_plots/benchmark_final_outnumbered_greedy_20seed.png` |
| BENCH_HORIZON_OUT | `Model Simulation/outputs/benchmark_plots/benchmark_final_outnumbered_horizon_20seed.png` |
| BENCH_HORIZON_CUR | `Model Simulation/outputs/benchmark_plots/benchmark_final_current_horizon_20seed.png` |
| BENCH_HORIZON_ABU | `Model Simulation/outputs/benchmark_plots/benchmark_final_abundant_horizon_20seed.png` |
| ANIM_HORIZON | `Model Simulation/outputs/images/aeris_dual_objective_horizon_animation.gif` |

---

## Slide-by-Slide Build Instructions

---

### SLIDE 1 — Title
**Layout:** Title Slide

**Text to add:**
- Title (large): `AERIS: Autonomous Relay-Enabled ISR System`
- Subtitle: `Connectivity-Constrained Multi-UAV Coordination via Receding-Horizon Control`
- Bottom line: `Jeewoo Choi | [Course Name] | [Date]`

**Image:** None

**Spoken script:**
"Hello, I'm Jeewoo Choi, and today I'm presenting AERIS — the Autonomous Relay-Enabled ISR System. This project tackles the challenge of coordinating a fleet of UAVs to conduct surveillance missions while maintaining a live communication link back to the operator, even as the adversary actively tries to destroy that link."

**Timing:** ~25 seconds

---

### SLIDE 2 — Problem Statement
**Layout:** Title + Content (bullets left, image right)

**Title text:** `The Problem`

**Bullet points:**
- UAVs must simultaneously sense, relay, recharge, and defend
- ISR data is worthless without an unbroken link back to base
- Enemies hunt static relay nodes and attack critical ground sites
- ~30 min battery forces constant hard trade-offs

**Image:** Use SNAP_DUAL. Crop to show only the Step 1 panel (top-left panel of the 6-panel grid). Place on right side of slide.

**Spoken script:**
"The core problem is this: a UAV can push deep into enemy territory and get a perfect view of the target — but if the relay chain behind it breaks, that data never arrives. In our scenario, enemies don't just wander — they hunt and destroy static relay nodes, and they move to attack two critical ground sites that serve as our forward operating bases. Lose those sites and we lose both our charging infrastructure and our best forward relay nodes. So the UAV team must sense, relay, recharge, and defend simultaneously, all under a roughly 30-minute battery constraint. The question is: how do you make those decisions in real time?"

**Timing:** ~50 seconds

---

### SLIDE 3 — Prior Work
**Layout:** Title + Content (bullets only)

**Title text:** `Related Work`

**Bullet points:**
- UAV coverage planning (Galceran & Carreras 2013) — maximizes sensor footprint, ignores relay chain constraint
- MILP-based multi-UAV coordination — graph reachability doesn't express cleanly as a linear constraint
- Reinforcement learning — poor generalization to novel adversarial scenarios
- **Our approach: Receding-Horizon MPC** — bounded compute, no training, interpretable policy selection

**Image:** None

**Spoken script:**
"Prior work on multi-UAV coverage planning focuses on maximizing how much ground is observed, but doesn't model the relay chain constraint at all. MILP-based coordination can optimize UAV assignments, but graph reachability — whether a UAV is actually connected to base through living relay nodes — doesn't express cleanly as a linear constraint. Reinforcement learning has been applied to similar problems but tends to overfit training environments and fails against novel adversarial behavior. Our approach draws from Model Predictive Control — a classical planning framework — and adapts it to this discrete-mode, graph-constrained problem. The key advantage is bounded computation, no training required, and interpretable policy choices."

**Timing:** ~45 seconds

---

### SLIDE 4 — System Architecture
**Layout:** Title + full-width image below title

**Title text:** `System Architecture`

**Sub-bullets (small text below title, above image):**
- 10km × 10km map | Base station (left) | North Site + South Site (relay nodes + FARP)
- Three UAV modes: ISR (blue) · Mobile Relay (green) · Static Relay (orange, killable)
- Connectivity rule: observations only count if UAV is connected to base through alive relays

**Image:** Use SNAP_HORIZON (full 6-panel grid). Place as large as possible below the bullet text. This is your best visual — the green relay chains branching to north/south sites are clearly visible.

**BONUS:** If using PowerPoint, also insert ANIM_HORIZON as a GIF on this slide (place it small in a corner or on a duplicate of this slide). It will loop automatically and show the relay chain evolving over time.

**Spoken script:**
"The simulation runs on a 10-kilometer-by-10-kilometer battlefield. The base station sits on the left edge. Two critical sites — North and South — each serve as a forward relay node and a refueling point, what military calls a FARP. UAVs operate in one of three modes: blue ISR drones have a wide sensor footprint but only a 500-meter communication range; green mobile relays bridge the communication gap between the base and forward ISRs; and orange static relays hold position as fixed network nodes — they have good range and low battery drain, but enemies can kill them. The key rule is that an ISR's observations only count if there is an unbroken path of relay nodes back to base. You can see that structure in these snapshots — the green lines show active relay chains branching toward the north and south sites."

**Timing:** ~55 seconds

---

### SLIDE 5 — Greedy Baseline Approach
**Layout:** Title + Content (bullets left, image right)

**Title text:** `Approach: Greedy Baseline Controller`

**Bullet points:**
- Cluster enemies into geographic bands (north / center / south)
- Build relay waypoints toward each cluster, weighted by enemy density
- Assign ISR drones round-robin per band
- Emergency intercept override when enemy is <40 steps from a critical site
- **Limitation:** reactive, one-step-ahead — cannot anticipate future relay losses

**Image:** Use SNAP_BASELINE (5-panel baseline run). Place on right side. Shows the relay chain staying along the center corridor as enemies are in the center band.

**Spoken script:**
"My baseline controller is a greedy heuristic that runs every timestep. It clusters enemies by geographic band — north, center, or south — and builds relay waypoints toward each cluster, allocating relay budget proportional to how many enemies are in each band. ISR drones are assigned round-robin across active bands, with an emergency intercept override when an enemy is closing in on a critical site within 40 timesteps. You can see in this baseline run that the relay chain forms a corridor toward the center enemy cluster and stays there for most of the mission. The greedy policy is fast and works well when the threat is simple — but it only looks one step ahead. It can't anticipate that an enemy flanking south will destroy a static relay and collapse the entire northern chain two minutes later."

**Timing:** ~55 seconds

---

### SLIDE 6 — Finite-Horizon MPC Controller
**Layout:** Title + Content (bullets left, image right)

**Title text:** `Approach: Finite-Horizon MPC Controller`

**Bullet points:**
- Every K=5 steps: snapshot state, simulate **7 candidate policies** H=20 steps forward
- Score terminal state with multi-objective function **J**
- J = ISR coverage + connectivity + site health + strikes − UAV losses − battery drain
- Pick best candidate, execute K steps, repeat
- 7 candidates span: aggressive intercept → persistent watch → proactive site defense

**Image:** Use SNAP_HORIZON, but crop to show only the **Step 1** and **Step 100** panels side by side. These two panels best show the branching relay structure the MPC builds.

**Spoken script:**
"My primary contribution is a receding-horizon MPC controller. Every 5 timesteps — 50 simulated seconds — the controller deep-copies the simulation state and runs 7 candidate policies forward for 20 steps each. It scores the terminal state of each rollout using a multi-objective function J that combines ISR coverage, connectivity fraction, site health, strike count, and penalties for UAV losses and battery drain. The candidate with the best J score is executed on the real simulation for 5 steps, and the whole process repeats. The 7 candidates span a range of strategic postures: aggressive interception, persistent watch near objective lanes, proactive site defense, and combinations. This means the optimizer can switch from offense to defense mid-mission as the threat picture changes — something the greedy policy structurally cannot do."

**Timing:** ~1 minute

---

### SLIDE 7 — Key Innovations
**Layout:** Title + Content (bullets left, image right)

**Title text:** `Key Design Innovations`

**Bullet points:**
- **Sites-as-relays:** Alive objectives act as forward relay nodes — lose the site, lose the chain
- **Cooperative observation:** 2 ISRs on one enemy advance the strike counter faster (capped at 2 by artillery doctrine)
- **Persistent area coverage:** Grid cells go stale after 50 steps — penalizes clustering all ISRs near known enemies

**Image:** Use METRICS_HORIZON. Crop to show only the **Battery & Objective Health** panel (bottom-center panel). It shows north/south site health staying at ~90–100% throughout the horizon optimizer run.

**Spoken script:**
"Three design choices are worth highlighting. First, sites-as-relays: when a critical site is alive, it acts as a hardened forward relay with a direct backhaul to base. This directly couples site defense to mission performance — if the optimizer lets a site fall, it loses the relay node for that entire flank. Second, cooperative observation: two ISRs observing the same enemy advance the strike counter by two per step instead of one, modeling the doctrine that a second observer accelerates a fire mission. A third ISR adds nothing — bottlenecked by artillery rate. Third, persistent area coverage: I track a 25-by-25 grid where each cell is fresh if it was observed within the last 50 steps. This penalizes the degenerate behavior of clustering all ISRs near the same known enemy and rewards spreading the network out."

**Timing:** ~55 seconds

---

### SLIDE 8 — Experiments
**Layout:** Title + Content (table + bullets)

**Title text:** `Experimental Setup`

**Table:**

| Configuration | UAVs vs Enemies |
|---|---|
| Outnumbered | 10 vs 15 |
| Even | 12 vs 12 |
| Nominal | 14 vs 9 |
| Abundant | 18 vs 6 |

**Bullet points below table:**
- 4 controllers: Greedy · Greedy+Intercept · Greedy+Defense · Horizon MPC
- 20 random seeds per configuration (terrain + spawn locations randomized)
- 500 timesteps = ~83 simulated minutes per run
- **320 total simulation runs**

**Image:** None (clean data slide)

**Spoken script:**
"I evaluated 4 controllers across 4 force configurations, running 20 random seeds each — that's 320 total simulation runs. The configurations range from heavily outnumbered at 10 UAVs versus 15 enemies, to force-abundant at 18 UAVs versus 6. Each seed randomizes terrain features and enemy spawn positions, so the mean and confidence intervals reflect genuine robustness rather than cherry-picked scenarios. Each run is 500 timesteps, representing about 83 simulated minutes of mission time."

**Timing:** ~35 seconds

---

### SLIDE 9 — Results
**Layout:** Title + two images side by side + summary table below

**Title text:** `Results`

**Images (side by side, large):**
- Left: BENCH_GREEDY_OUT — label it "Greedy — Outnumbered (10v15)"
- Right: BENCH_HORIZON_OUT — label it "Horizon MPC — Outnumbered (10v15)"

**Key callouts to annotate on the images (use text boxes with arrows):**
- Left image → Objective Score panel: "Drops to −25"
- Left image → North Site Health panel: "Site collapses to ~20%"
- Right image → Objective Score panel: "Holds at −12"
- Right image → North Site Health panel: "Site preserved at ~40%"

**Summary table below images:**

| Scenario | Winner | Final Obj. Score | North Site Health |
|---|---|---|---|
| 10v15 Outnumbered | **Horizon** | −6.48 | 60.3% |
| 12v12 Even | Greedy | −3.27 | 65.0% |
| 14v9 Nominal | **Horizon** | +1.37 | 73.7% |
| 18v6 Abundant | Greedy | +4.47 | 90.5% |

**Spoken script:**
"Here are the main results. I'm showing the outnumbered scenario — 10 UAVs versus 15 enemies — with greedy on the left and the horizon optimizer on the right. The greedy policy's objective score collapses to around negative 25 by the end of the simulation. The horizon optimizer holds at negative 12 — roughly half the damage. Site health tells the same story: greedy loses the north site down to about 20 percent health, while the horizon optimizer preserves it at 40 percent. The pattern across all four scenarios is that the horizon optimizer wins decisively when the problem is hard — outnumbered and nominal configurations — but in the easy scenarios where resources are abundant, greedy performs comparably. The most meaningful result is in the nominal 14-versus-9 configuration, where the horizon optimizer is the first policy to achieve a positive mission score. That means it eliminated more threats than it lost relay nodes and site health — net mission success."

**Timing:** ~1 minute 10 seconds

---

### SLIDE 10 — Conclusions & Future Work
**Layout:** Title + Content (bullets left, image right)

**Title text:** `Conclusions & Future Work`

**Bullet points:**
- Horizon MPC enables multi-axis strategic reasoning unavailable to greedy heuristics
- Sites-as-relays is critical: defending sites directly protects the relay network
- Horizon wins when the threat is hard; greedy is competitive when resources are abundant
- **Limitations:** Fixed horizon H=20, hand-tuned J weights, enemies don't adapt
- **Future work:** Adaptive replan cadence · Learned J weights via offline simulation · Adversarial enemy policies

**Image:** Use METRICS_HORIZON (full 9-panel view). This shows objective score climbing, ISR coverage reaching 1.0, connectivity held near 100%, and site health stable — a strong visual conclusion.

**Spoken script:**
"To conclude: the receding-horizon controller shows that lookahead planning significantly outperforms reactive heuristics when the threat is difficult enough that one-step decisions lead to cascade failures. The most important design insight is that sites-as-relays creates a direct coupling between objective defense and communication network integrity — the optimizer naturally discovers that preserving a site also preserves the relay chain for that entire flank. The main limitations are that the horizon and replan cadence are fixed, the objective weights are hand-tuned rather than learned, and the enemy doesn't adapt its strategy in response to the UAV behavior. Future work would make the horizon adaptive, use offline rollout data to fit the J weights, and introduce an adversarial enemy that learns to counter the relay chain structure. Thank you."

**Timing:** ~55 seconds

---

### SLIDE 11 — References
**Layout:** Title + Content (text only)

**Title text:** `References`

**Text:**
- Galceran, E. & Carreras, M. (2013). A survey on coverage path planning for robotics. *Robotics and Autonomous Systems*, 61(12), 1258–1276.
- Bemporad, A. & Morari, M. (1999). Control of systems integrating logic, dynamics, and constraints. *Automatica*, 35(3), 407–427.
- Hagberg, A., Swart, P., & Chult, D. (2008). Exploring network structure, dynamics, and function using NetworkX.
- Harris, C.R. et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362.
- Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.

**Image:** None

**Spoken script:** *(No spoken script — display briefly, then end recording)*

---

## Build Checklist

- [ ] Slide 1: Title text entered
- [ ] Slide 2: Bullets + SNAP_DUAL Step 1 panel cropped and inserted
- [ ] Slide 3: Bullets entered
- [ ] Slide 4: SNAP_HORIZON inserted full-size + optional ANIM_HORIZON GIF
- [ ] Slide 5: Bullets + SNAP_BASELINE inserted
- [ ] Slide 6: Bullets + SNAP_HORIZON Step 1 and Step 100 cropped and inserted
- [ ] Slide 7: Bullets + METRICS_HORIZON Battery panel cropped and inserted
- [ ] Slide 8: Table + bullets entered
- [ ] Slide 9: BENCH_GREEDY_OUT and BENCH_HORIZON_OUT side by side + annotation text boxes + summary table
- [ ] Slide 10: Bullets + METRICS_HORIZON full inserted
- [ ] Slide 11: References text entered
- [ ] Record audio over slides using PowerPoint recorder or screen capture
- [ ] Export as .mp4

## Timing Check

| Slide | Target Time |
|---|---|
| 1 | 25 sec |
| 2 | 50 sec |
| 3 | 45 sec |
| 4 | 55 sec |
| 5 | 55 sec |
| 6 | 60 sec |
| 7 | 55 sec |
| 8 | 35 sec |
| 9 | 70 sec |
| 10 | 55 sec |
| 11 | 10 sec |
| **Total** | **~8 min 15 sec** |
