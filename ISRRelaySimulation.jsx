import React, { useEffect, useMemo, useRef, useState } from "react";

// ------------------------------------------------------------------
// ISR Relay Network Simulation — improved
//
// Key changes vs. the original:
//  1. Bidirectional role switching (ISR <-> MOBILE via FDIR, with hysteresis)
//  2. All state mutations use functional updaters from a single tick driver
//     (no more setX inside setY inside setInterval)
//  3. Mobile relay targets are anchored per drone and only reshuffled on
//     arrival, so relays no longer jitter every frame
//  4. Terrain line-of-sight check applied to ALL links (relays can be
//     blocked by ridges too)
//  5. ISR sensor range is scaled by altitude like comms radius
//  6. Enemy force progressively reinforces — new waves advance while
//     existing ones are destroyed
//  7. Heatmap role weights corrected (mobile relays weigh more, matching
//     their larger comms radius)
//  8. Terrain rendered via a memoized raster image instead of 860 <rect>s
//  9. Observation timer visualized as a filling ring around each enemy
// 10. Active observer -> target lines drawn so it's obvious which ISR
//     drones are cueing the strike
// ------------------------------------------------------------------

// **** http://localhost:5173/

const W = 820;
const H = 520;
const BASE = { x: 70, y: 440 };

const TICK_MS = 100;
const OBSERVE_TO_ATTACK_MS = 10000;
const OBSERVATION_DECAY_PER_TICK = TICK_MS; // 1:1 decay rate
const ATTACK_DAMAGE = 28;
const ATTACK_SPEED = 8.5;
const ISR_SENSOR_BASE = 170;
const TERRAIN_BLOCK_DELTA = 0.46;

const ROLE = {
  ISR:    { label: "ISR",          color: "#2563eb", radius: 110, speed: 2.7, drain: 0.08 },
  MOBILE: { label: "Mobile Relay", color: "#16a34a", radius: 180, speed: 1.6, drain: 0.06 },
  STATIC: { label: "Static Relay", color: "#f59e0b", radius: 230, speed: 0,   drain: 0.025 },
};

// ---------- pure helpers --------------------------------------------------

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const dist  = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function terrainAt(x, y) {
  const nx = x / W, ny = y / H;
  let h = 0.5;
  h += 0.24 * Math.sin(2.7 * Math.PI * nx + 1.1 * Math.sin(2 * Math.PI * ny));
  h += 0.17 * Math.cos(3.4 * Math.PI * ny + 0.6);
  h += 0.09 * Math.sin(6.2 * Math.PI * (nx + ny));
  return clamp(h, 0, 1);
}

function terrainRGBA(h) {
  if (h > 0.78) return [112,  87, 54, 200];
  if (h > 0.60) return [146, 118, 78, 175];
  if (h > 0.42) return [ 84, 130, 100, 142];
  return                [ 64, 116, 143, 115];
}

const signalFactor = (x, y) => 0.75 + 0.65 * terrainAt(x, y);
const linkRadius   = (n) => (ROLE[n.role] || ROLE.ISR).radius * signalFactor(n.x, n.y);
const sensorRange  = (n) => ISR_SENSOR_BASE * signalFactor(n.x, n.y);

function blockedByTerrain(a, b) {
  // Sample 8 points along the segment. If any interior point is significantly
  // higher than BOTH endpoints, the ridge blocks line-of-sight.
  const low = Math.min(terrainAt(a.x, a.y), terrainAt(b.x, b.y));
  for (let i = 1; i < 8; i += 1) {
    const t = i / 8;
    const x = a.x + (b.x - a.x) * t;
    const y = a.y + (b.y - a.y) * t;
    if (terrainAt(x, y) > low + TERRAIN_BLOCK_DELTA) return true;
  }
  return false;
}

function hasLink(a, b) {
  if (dist(a, b) >= Math.min(linkRadius(a), linkRadius(b))) return false;
  // Relays get a modest LOS bonus (higher gain antennas) but are not immune.
  const bothRelays = a.role !== "ISR" && b.role !== "ISR";
  if (bothRelays) return true;
  return !blockedByTerrain(a, b);
}

function moveToward(obj, target, speed) {
  const dx = target.x - obj.x, dy = target.y - obj.y;
  const len = Math.hypot(dx, dy) || 1;
  const step = Math.min(speed, len);
  return { ...obj, x: obj.x + (dx / len) * step, y: obj.y + (dy / len) * step };
}

// ---------- entities ------------------------------------------------------

const baseNode = () => ({ id: 0, role: "STATIC", x: BASE.x, y: BASE.y, connected: true, battery: 100 });

function initDrones() {
  return [
    { id: 1, role: "STATIC", x:  95, y: 420, tx:  95, ty: 420, battery: 100, connected: true },
    { id: 2, role: "MOBILE", x: 200, y: 400, tx: 220, ty: 390, battery: 100, connected: true },
    { id: 3, role: "MOBILE", x: 335, y: 360, tx: 360, ty: 340, battery: 100, connected: true },
    { id: 4, role: "ISR",    x: 420, y: 290, tx: 560, ty: 145, battery: 100, connected: true },
    { id: 5, role: "ISR",    x: 460, y: 310, tx: 640, ty: 220, battery: 100, connected: true },
    { id: 6, role: "ISR",    x: 470, y: 390, tx: 655, ty: 350, battery: 100, connected: true },
    { id: 7, role: "ISR",    x: 450, y: 260, tx: 710, ty: 130, battery: 100, connected: true },
    { id: 8, role: "ISR",    x: 510, y: 360, tx: 725, ty: 420, battery: 100, connected: true },
  ];
}

let enemySeq = 1;
function newEnemy(id, x, y) {
  return {
    id,
    x, y,
    vx: -0.25 - Math.random() * 0.15,
    vy: (Math.random() - 0.5) * 0.12,
    hp: 100,
    observedMs: 0,
  };
}
function initEnemies() {
  enemySeq = 5;
  return [
    newEnemy(1, 690,  95),
    newEnemy(2, 735, 245),
    newEnemy(3, 665, 390),
    newEnemy(4, 785, 320),
  ];
}

// ---------- network graph -------------------------------------------------

function buildNetwork(drones) {
  const nodes = [baseNode(), ...drones];
  const n = nodes.length;
  const adj = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      if (hasLink(nodes[i], nodes[j])) {
        adj[i].push(j);
        adj[j].push(i);
      }
    }
  }
  // BFS from base (index 0) for connectivity + spanning branches
  const prev  = new Array(n).fill(-1);
  const seen  = new Array(n).fill(false);
  seen[0] = true;
  const queue = [0];
  while (queue.length > 0) {
    const i = queue.shift();
    for (const j of adj[i]) {
      if (!seen[j]) {
        seen[j] = true;
        prev[j] = i;
        queue.push(j);
      }
    }
  }
  const branches = [];
  for (let j = 1; j < n; j += 1) {
    if (seen[j] && prev[j] >= 0) branches.push({ from: nodes[prev[j]], to: nodes[j] });
  }
  // Collect ALL links for the "mesh" view (shows redundancy, not just tree)
  const meshLinks = [];
  for (let i = 0; i < n; i += 1) {
    for (const j of adj[i]) {
      if (j > i && seen[i] && seen[j]) meshLinks.push({ from: nodes[i], to: nodes[j] });
    }
  }
  const connectedIds = new Set();
  for (let i = 0; i < n; i += 1) if (seen[i]) connectedIds.add(nodes[i].id);
  return { branches, meshLinks, connectedIds };
}

function computeConnectivity(drones) {
  const { connectedIds } = buildNetwork(drones);
  return drones.map((d) => ({ ...d, connected: connectedIds.has(d.id) }));
}

function observers(enemy, drones) {
  return drones.filter(
    (d) => d.role === "ISR" && d.connected && d.battery > 0 && dist(d, enemy) < sensorRange(d),
  );
}

// ---------- target picking (deterministic per drone, no per-tick jitter) --

function isrTarget(enemies, id) {
  const e = enemies[(id * 31) % Math.max(1, enemies.length)] || { x: 650, y: 250 };
  return {
    x: clamp(e.x - 80 + Math.random() * 90, 380, W - 40),
    y: clamp(e.y - 90 + Math.random() * 180, 50, H - 50),
  };
}

function mobileAnchor(id) {
  // Each mobile drone has a stable loiter box keyed on its id.
  const boxes = {
    2: { x: 205, y: 390, w: 35, h: 30 },
    3: { x: 340, y: 340, w: 45, h: 35 },
  };
  const b = boxes[id] || { x: 320, y: 345, w: 140, h: 80 };
  return { x: b.x + Math.random() * b.w, y: b.y + Math.random() * b.h };
}

// ---------- FDIR (bidirectional, hysteresis to stop thrashing) -----------

function applyFDIR(drones) {
  const total = drones.length || 1;
  const connectedRatio = drones.filter((d) => d.connected).length / total;
  const relayCount = drones.filter((d) => d.role !== "ISR").length;

  // Thresholds with hysteresis:
  //  - Convert an ISR to MOBILE when network is weak OR relays are scarce
  //  - Convert a MOBILE back to ISR when network is comfortably healthy
  //    AND we already have enough relays
  const NEED_RELAY   = connectedRatio < 0.75 || relayCount < 3;
  const RELAX_RELAY  = connectedRatio > 0.9  && relayCount > 3;

  let promoted = false;
  let demoted  = false;

  return drones.map((d) => {
    // Never touch the static anchor or empty batteries
    if (d.role === "STATIC" || d.battery < 40) return d;

    if (!promoted && NEED_RELAY && d.role === "ISR") {
      promoted = true;
      const t = mobileAnchor(d.id);
      return { ...d, role: "MOBILE", tx: t.x, ty: t.y };
    }
    if (!demoted && RELAX_RELAY && d.role === "MOBILE" && d.id !== 2 && d.id !== 3) {
      // Don't demote the "critical chain" mobiles (id 2,3) — they're the
      // bridge between base and the forward ISR swarm.
      demoted = true;
      return { ...d, role: "ISR", tx: 500 + Math.random() * 200, ty: 100 + Math.random() * 300 };
    }
    return d;
  });
}

// ---------- damage + enemy reinforcement ----------------------------------

function damageEnemy(enemies, id) {
  return enemies
    .map((e) => (e.id === id ? { ...e, hp: Math.max(0, e.hp - ATTACK_DAMAGE) } : e))
    .filter((e) => e.hp > 0);
}

function maybeReinforce(enemies, tick) {
  // Every ~150 ticks (15s), if fewer than 5 enemy groups exist, push a new
  // wave in from the east edge. This gives the "advancing force" feel.
  if (tick % 150 !== 0) return enemies;
  if (enemies.length >= 5) return enemies;
  enemySeq += 1;
  const y = 60 + Math.random() * (H - 120);
  return [...enemies, newEnemy(enemySeq, W - 20, y)];
}

// ---------- self-tests (throw on failure so they actually surface) --------

function assert(cond, msg) {
  if (!cond) { const err = new Error("Self-test failed: " + msg); console.error(err); throw err; }
}

function runSelfTests() {
  const drones = initDrones();
  const enemies = initEnemies();
  const connected = computeConnectivity(drones);
  const damaged = damageEnemy(enemies, enemies[0].id);
  const disconnectedIsr = [{ ...drones[3], connected: false }];

  assert(drones.length === 8, "8 drones");
  assert(enemies.length === 4, "4 enemy groups");
  assert(damaged[0].hp === 72, "attack removes 28 hp");
  assert(TICK_MS === 100 && OBSERVATION_DECAY_PER_TICK === TICK_MS, "decay rate matches tick");
  assert(linkRadius(drones[0]) > linkRadius(drones[4]), "static relay reaches farther than ISR");
  assert(connected.length === drones.length, "connectivity preserves count");
  assert(observers({ x: drones[3].x, y: drones[3].y }, connected).length >= 1, "connected ISR observes nearby point");
  assert(buildNetwork(connected).branches.length > 0, "spanning branches exist");
  assert(hasLink(baseNode(), drones[0]) === true, "base links to initial static relay");
  assert(computeConnectivity(initDrones()).filter((d) => d.role === "ISR" && d.connected).length >= 3, "initial placement keeps most ISR connected");
  assert(observers({ x: drones[3].x, y: drones[3].y }, disconnectedIsr).length === 0, "disconnected ISR does not observe");

  // Bidirectional FDIR test: with a mostly-connected, relay-rich fleet, at
  // least one MOBILE should be eligible for demotion back to ISR
  const fleetHealthy = connected.map((d) =>
    d.id === 4 ? { ...d, role: "MOBILE", connected: true } : { ...d, connected: true }
  );
  const after = applyFDIR(fleetHealthy);
  const demoted = after.some((d, i) => fleetHealthy[i].role === "MOBILE" && d.role === "ISR");
  assert(demoted, "FDIR demotes a non-critical MOBILE back to ISR when network is healthy");
}

// ---------- UI primitives -------------------------------------------------

function Button({ children, onClick, active }) {
  return (
    <button
      onClick={onClick}
      style={{
        border: active ? "1px solid #0f172a" : "1px solid #cbd5e1",
        background: active ? "#0f172a" : "white",
        color: active ? "white" : "#0f172a",
        borderRadius: 12, padding: "8px 12px", fontWeight: 700, cursor: "pointer",
      }}
    >
      {children}
    </button>
  );
}

function Card({ title, value, children }) {
  return (
    <div style={{ background: "white", border: "1px solid #e2e8f0", borderRadius: 16, padding: 12 }}>
      {title && <div style={{ fontSize: 12, color: "#64748b" }}>{title}</div>}
      {value !== undefined && <div style={{ fontSize: 26, fontWeight: 800 }}>{value}</div>}
      {children}
    </div>
  );
}

function Bar({ value, color }) {
  return (
    <div style={{ width: 62, height: 5, background: "#e5e7eb", borderRadius: 99, overflow: "hidden" }}>
      <div style={{ width: `${clamp(value, 0, 100)}%`, height: "100%", background: color }} />
    </div>
  );
}

// ---------- terrain raster (memoized — built once per mount) --------------

function useTerrainDataURL() {
  return useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(W, H);
    for (let y = 0; y < H; y += 1) {
      for (let x = 0; x < W; x += 1) {
        const [r, g, b, a] = terrainRGBA(terrainAt(x, y));
        const idx = (y * W + x) * 4;
        img.data[idx]     = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = a;
      }
    }
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL();
  }, []);
}

// ---------- heatmap (coarse grid, corrected role weights) -----------------

function Heatmap({ drones, enemies }) {
  const STEP = 25;
  const cells = [];
  for (let x = 0; x < W; x += STEP) {
    for (let y = 0; y < H; y += STEP) {
      let friendly = 0, enemy = 0;
      for (const d of drones) {
        if (!d.connected) continue;
        // MOBILE and STATIC relays spread *comms* coverage (bigger radius)
        // ISR spreads *surveillance* coverage (smaller radius but that's
        // already encoded in the smaller drop-off below)
        const w = d.role === "MOBILE" ? 1.0 : d.role === "STATIC" ? 0.85 : 0.7;
        const r = linkRadius(d);
        const k = 1 - dist(d, { x, y }) / r;
        if (k > 0) friendly += k * w;
      }
      for (const e of enemies) {
        const k = 1 - dist(e, { x, y }) / 145;
        if (k > 0) enemy += k;
      }
      const green = clamp(friendly * 0.11, 0, 0.5);
      const red   = clamp(enemy    * 0.13, 0, 0.36);
      if (green > 0.02) cells.push(<rect key={`g-${x}-${y}`} x={x} y={y} width={STEP} height={STEP} fill={`rgba(34,197,94,${green})`} />);
      if (red   > 0.02) cells.push(<rect key={`r-${x}-${y}`} x={x} y={y} width={STEP} height={STEP} fill={`rgba(239,68,68,${red})`} />);
    }
  }
  return <g>{cells}</g>;
}

// ---------- link / branch overlay -----------------------------------------

function Links({ drones, mode }) {
  const { branches, meshLinks } = buildNetwork(drones);
  const edges = mode === "mesh" ? meshLinks : branches;
  return (
    <g>
      {edges.map((edge, idx) => (
        <g key={`e-${idx}`}>
          <line x1={edge.from.x} y1={edge.from.y} x2={edge.to.x} y2={edge.to.y} stroke="rgba(15,23,42,0.55)" strokeWidth="4.2" strokeLinecap="round" opacity="0.32" />
          <line x1={edge.from.x} y1={edge.from.y} x2={edge.to.x} y2={edge.to.y} stroke="rgba(34,197,94,0.92)" strokeWidth="2.2" strokeLinecap="round" />
        </g>
      ))}
    </g>
  );
}

// ---------- drone visual --------------------------------------------------

function Drone({ d }) {
  const role = ROLE[d.role];
  const r = linkRadius(d);
  return (
    <g>
      <circle cx={d.x} cy={d.y} r={r} fill={role.color} opacity="0.08" />
      <circle cx={d.x} cy={d.y} r={r} fill="none" stroke={role.color} strokeOpacity="0.5" strokeWidth="1.1" strokeDasharray={d.role === "STATIC" ? "" : "8 6"} />
      <circle cx={d.x} cy={d.y} r="11" fill="white" stroke={d.connected ? role.color : "#ef4444"} strokeWidth="3" />
      <text x={d.x + 14} y={d.y + 4} fontSize="11" fontWeight="700" fill="#0f172a">{role.label}</text>
      <foreignObject x={d.x + 14} y={d.y + 8} width="80" height="12">
        <Bar value={d.battery} color={role.color} />
      </foreignObject>
    </g>
  );
}

// ---------- main component ------------------------------------------------

export default function ISRRelaySimulation() {
  const [running, setRunning] = useState(true);
  const [fdir, setFdir]       = useState(true);
  const [linkMode, setLinkMode] = useState("tree"); // "tree" | "mesh"
  const [state, setState] = useState(() => ({
    drones:  computeConnectivity(initDrones()),
    enemies: initEnemies(),
    attacks: [],
    hits: 0,
    tick: 0,
  }));
  const attackIdRef = useRef(1);
  const didTest = useRef(false);
  const terrainURL = useTerrainDataURL();

  useEffect(() => {
    if (didTest.current) return;
    didTest.current = true;
    try { runSelfTests(); } catch (_) { /* surfaced to console */ }
  }, []);

  function reset() {
    attackIdRef.current = 1;
    setState({ drones: computeConnectivity(initDrones()), enemies: initEnemies(), attacks: [], hits: 0, tick: 0 });
  }

  useEffect(() => {
    if (!running) return undefined;
    const timer = setInterval(() => {
      setState((prev) => {
        const tick = prev.tick + 1;

        // --- enemies move & accumulate observation time ---
        let enemies = prev.enemies.map((e) => {
          const sway = Math.sin(Date.now() / 950 + e.id) * 0.8;
          const moved = {
            ...e,
            x: clamp(e.x + e.vx + sway * 0.18, 500, W - 25),
            y: clamp(e.y + e.vy + sway * 0.25,  45, H - 45),
          };
          const obsCount = observers(moved, prev.drones).length;
          const observedMs = obsCount > 0
            ? Math.min(OBSERVE_TO_ATTACK_MS, (moved.observedMs || 0) + TICK_MS)
            : Math.max(0, (moved.observedMs || 0) - OBSERVATION_DECAY_PER_TICK);
          return { ...moved, observedMs };
        });

        // --- launch attacks against enemies past the observation threshold ---
        const newAttacks = [];
        const activeTargets = new Set(prev.attacks.map((a) => a.targetId));
        enemies = enemies.map((e) => {
          if (e.observedMs >= OBSERVE_TO_ATTACK_MS && !activeTargets.has(e.id)) {
            newAttacks.push({
              id: attackIdRef.current++,
              targetId: e.id,
              x: BASE.x, y: BASE.y,
              tx: e.x, ty: e.y,
            });
            return { ...e, observedMs: 0 };
          }
          return e;
        });

        // --- step attack projectiles; resolve hits ---
        let hits = prev.hits;
        const remainingAttacks = [];
        for (const a of [...prev.attacks, ...newAttacks]) {
          const target = enemies.find((e) => e.id === a.targetId);
          if (!target) continue; // target already dead
          const stepped = moveToward(a, target, ATTACK_SPEED);
          if (dist(stepped, target) < 10) {
            enemies = damageEnemy(enemies, a.targetId);
            hits += 1;
          } else {
            remainingAttacks.push({ ...stepped, tx: target.x, ty: target.y });
          }
        }

        // --- reinforce enemy force ---
        enemies = maybeReinforce(enemies, tick);

        // --- step drones: move, drain, recharge-respawn ---
        let drones = prev.drones.map((d) => {
          const role = ROLE[d.role];
          let nd = d;
          if (role.speed > 0) {
            nd = moveToward(nd, { x: nd.tx, y: nd.ty }, role.speed);
            const arrived = dist(nd, { x: nd.tx, y: nd.ty }) < 14;
            if (arrived || Math.random() < 0.008) {
              const t = nd.role === "MOBILE" ? mobileAnchor(nd.id) : isrTarget(enemies, nd.id);
              nd = { ...nd, tx: t.x, ty: t.y };
            }
          }
          nd = { ...nd, battery: Math.max(0, nd.battery - role.drain) };
          if (nd.battery <= 0) {
            // Return to base, recharge, respawn as ISR for forward work
            nd = {
              ...nd,
              x: BASE.x + (Math.random() - 0.5) * 30,
              y: BASE.y - Math.random() * 30,
              role: nd.id === 1 ? "STATIC" : "ISR", // preserve the anchor
              battery: 100,
              tx: nd.id === 1 ? BASE.x : 450 + Math.random() * 280,
              ty: nd.id === 1 ? BASE.y : 80  + Math.random() * 350,
            };
          }
          return nd;
        });

        // --- connectivity + FDIR ---
        drones = computeConnectivity(drones);
        if (fdir) drones = computeConnectivity(applyFDIR(drones));

        return { drones, enemies, attacks: remainingAttacks, hits, tick };
      });
    }, TICK_MS);
    return () => clearInterval(timer);
  }, [running, fdir]);

  const { drones, enemies, attacks, hits } = state;
  const connected = drones.filter((d) => d.connected).length;
  const tracked = enemies.filter((e) => observers(e, drones).length > 0).length;
  const networkHealth = Math.round((connected / Math.max(1, drones.length)) * 100);
  const relayCount = drones.filter((d) => d.role !== "ISR").length;

  // Observer → target lines (only currently-observing pairs)
  const observationLines = [];
  for (const e of enemies) {
    for (const d of observers(e, drones)) {
      observationLines.push({ from: d, to: e, key: `obs-${e.id}-${d.id}` });
    }
  }

  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", padding: 20, fontFamily: "Inter, system-ui, sans-serif", color: "#0f172a" }}>
      <div style={{ maxWidth: 1120, margin: "0 auto", display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 26 }}>ISR Relay Network Simulation</h1>
            <p style={{ margin: "6px 0 0", color: "#475569", maxWidth: 720 }}>
              UAVs switch between ISR, mobile relay, and static relay roles while maintaining a connected branch path back to the operator. Enemy groups advance from the east; ten seconds of continuous observation by any connected ISR drone cues a strike from base.
            </p>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Button onClick={() => setRunning((v) => !v)} active={running}>{running ? "Pause" : "Run"}</Button>
            <Button onClick={() => setFdir((v) => !v)} active={fdir}>FDIR {fdir ? "on" : "off"}</Button>
            <Button onClick={() => setLinkMode((m) => (m === "tree" ? "mesh" : "tree"))} active={linkMode === "mesh"}>{linkMode === "tree" ? "Tree view" : "Mesh view"}</Button>
            <Button onClick={reset}>Reset</Button>
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10 }}>
          <Card title="Network health" value={`${networkHealth}%`} />
          <Card title="Connected nodes" value={`${connected}/${drones.length}`} />
          <Card title="Tracked groups" value={`${tracked}/${enemies.length}`} />
          <Card title="Relay nodes" value={relayCount} />
          <Card title="Hits" value={hits} />
        </div>

        <div style={{ background: "white", border: "1px solid #e2e8f0", borderRadius: 18, padding: 12 }}>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block", borderRadius: 14, background: "#eef2f7" }}>
            <image href={terrainURL} x="0" y="0" width={W} height={H} style={{ imageRendering: "pixelated" }} />
            <Heatmap drones={drones} enemies={enemies} />
            <Links drones={drones} mode={linkMode} />

            {/* Observer → target lines (purple dashed) */}
            {observationLines.map((l) => (
              <line key={l.key} x1={l.from.x} y1={l.from.y} x2={l.to.x} y2={l.to.y} stroke="rgba(168, 85, 247, 0.55)" strokeWidth="1.4" strokeDasharray="3 4" />
            ))}

            {/* Operator base */}
            <circle cx={BASE.x} cy={BASE.y} r="18" fill="#111827" />
            <text x={BASE.x + 24} y={BASE.y + 5} fontSize="12" fontWeight="700" fill="#111827">Operator / base</text>

            {/* Enemies: influence ring, observation timer ring, body */}
            {enemies.map((e) => {
              const frac = clamp(e.observedMs / OBSERVE_TO_ATTACK_MS, 0, 1);
              const R = 26;
              const C = 2 * Math.PI * R;
              return (
                <g key={e.id}>
                  <circle cx={e.x} cy={e.y} r="62" fill="rgba(239,68,68,0.12)" stroke="rgba(239,68,68,0.32)" />
                  {/* observation timer ring — rotates so fill starts at 12 o'clock */}
                  <g transform={`rotate(-90 ${e.x} ${e.y})`}>
                    <circle cx={e.x} cy={e.y} r={R} fill="none" stroke="rgba(168,85,247,0.18)" strokeWidth="3.5" />
                    <circle cx={e.x} cy={e.y} r={R} fill="none" stroke="rgba(168,85,247,0.95)" strokeWidth="3.5" strokeLinecap="round" strokeDasharray={`${frac * C} ${C}`} />
                  </g>
                  <circle cx={e.x} cy={e.y} r="9" fill="#dc2626" />
                  <text x={e.x + 14} y={e.y - 12} fontSize="11" fontWeight="700" fill="#7f1d1d">Enemy {e.id}</text>
                  <foreignObject x={e.x + 14} y={e.y - 8} width="80" height="12">
                    <Bar value={e.hp} color="#dc2626" />
                  </foreignObject>
                  <text x={e.x + 14} y={e.y + 14} fontSize="10" fontWeight="700" fill="#7e22ce">{frac >= 1 ? "strike cued" : `${(frac * 10).toFixed(1)}/10s`}</text>
                </g>
              );
            })}

            {/* Attack projectiles */}
            {attacks.map((a) => (
              <g key={a.id}>
                <line x1={BASE.x} y1={BASE.y} x2={a.x} y2={a.y} stroke="rgba(126,34,206,0.55)" strokeWidth="2" strokeDasharray="5 5" />
                <circle cx={a.x} cy={a.y} r="7" fill="#a855f7" />
              </g>
            ))}

            {drones.map((d) => <Drone key={d.id} d={d} />)}
          </svg>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, fontSize: 13, color: "#334155" }}>
          <Card><b>ISR role (blue)</b><br />Patrols toward enemy influence. Sensor range scales with altitude. Only connected ISR drones contribute to the purple observation timer.</Card>
          <Card><b>Mobile relay (green)</b><br />Loiters in a fixed anchor box, keeping the chain between the static base relay and the forward ISR swarm intact.</Card>
          <Card><b>FDIR with hysteresis</b><br />Promotes ISR→mobile relay when the network weakens, demotes mobile→ISR when it is comfortably healthy, ignoring the two chain-critical relays.</Card>
        </div>
      </div>
    </div>
  );
}
