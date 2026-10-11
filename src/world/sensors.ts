import type { Agent, World } from "./agent.js";
import { INPUT_DIM } from "./types.js";

const clamp = (v: number, a = -1, b = 1) => Math.max(a, Math.min(b, v));
const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
  Math.hypot(a.x - b.x, a.y - b.y);

/**
 * Builds a 40-dimensional sensor vector from the agent's perspective.
 *
 * Layout:
 *   0-7   : self (energy, integrity, age, affect, velocity)
 *   8-12  : genome (5 normalized traits)
 *   13-16 : food (dx, dy, proximity, presence)
 *   17-20 : hazard (dx, dy, proximity, presence)
 *   21-24 : mirror (dx, dy, proximity, presence)
 *   25-30 : neighbor (dx, dy, proximity, presence, bond, neighbor energy)
 *   31-33 : mark (active, visible in mirror, touch progress)
 *   34-36 : curiosity (prediction error, novelty, exploration drive)
 *   37-39 : global signals (sin time, cos time, population ratio)
 */
export function sense(agent: Agent, world: World): Float32Array {
  const s = new Float32Array(INPUT_DIM);
  let i = 0;

  // Self (8)
  s[i++] = agent.energy / 120;
  s[i++] = agent.integrity / 100;
  s[i++] = clamp(agent.age / 300, 0, 1);
  s[i++] = agent.affect.pleasure;
  s[i++] = agent.affect.valence;
  s[i++] = agent.affect.arousal;
  s[i++] = clamp(agent.vel.x / 40);
  s[i++] = clamp(agent.vel.y / 40);

  // Genome (5)
  s[i++] = clamp(agent.genome.maxSpeed / 2, 0, 1);
  s[i++] = clamp(agent.genome.metabolism / 2, 0, 1);
  s[i++] = clamp(agent.genome.resilience / 2, 0, 1);
  s[i++] = clamp(agent.genome.sociability / 2, 0, 1);
  s[i++] = clamp(agent.genome.libido / 2, 0, 1);

  // Food (4)
  let bestFood: { x: number; y: number } | null = null;
  let bestFoodD = Infinity;
  for (const f of world.food) {
    const d = dist(agent.pos, f.pos);
    if (d < bestFoodD) { bestFoodD = d; bestFood = f.pos; }
  }
  if (bestFood) {
    s[i++] = clamp((bestFood.x - agent.pos.x) / 200);
    s[i++] = clamp((bestFood.y - agent.pos.y) / 200);
    s[i++] = clamp(1 - bestFoodD / 400);
    s[i++] = 1;
  } else {
    s[i++] = 0; s[i++] = 0; s[i++] = 0; s[i++] = 0;
  }

  // Hazard (4)
  let bestHaz: { x: number; y: number } | null = null;
  let bestHazD = Infinity;
  for (const h of world.hazards) {
    const d = dist(agent.pos, h.pos);
    if (d < bestHazD) { bestHazD = d; bestHaz = h.pos; }
  }
  if (bestHaz) {
    s[i++] = clamp((bestHaz.x - agent.pos.x) / 200);
    s[i++] = clamp((bestHaz.y - agent.pos.y) / 200);
    s[i++] = clamp(1 - bestHazD / 400);
    s[i++] = 1;
  } else {
    s[i++] = 0; s[i++] = 0; s[i++] = 0; s[i++] = 0;
  }

  // Mirror (4)
  let bestMir: { x: number; y: number } | null = null;
  let bestMirD = Infinity;
  for (const m of world.mirrors) {
    const d = dist(agent.pos, m.pos);
    if (d < bestMirD) { bestMirD = d; bestMir = m.pos; }
  }
  if (bestMir) {
    s[i++] = clamp((bestMir.x - agent.pos.x) / 200);
    s[i++] = clamp((bestMir.y - agent.pos.y) / 200);
    s[i++] = clamp(1 - bestMirD / 400);
    s[i++] = 1;
  } else {
    s[i++] = 0; s[i++] = 0; s[i++] = 0; s[i++] = 0;
  }

  // Neighbor (6)
  let bestOther: Agent | null = null;
  let bestOtherD = Infinity;
  for (const o of world.agents) {
    if (o === agent || !o.alive) continue;
    const d = dist(agent.pos, o.pos);
    if (d < bestOtherD) { bestOtherD = d; bestOther = o; }
  }
  if (bestOther) {
    s[i++] = clamp((bestOther.pos.x - agent.pos.x) / 200);
    s[i++] = clamp((bestOther.pos.y - agent.pos.y) / 200);
    s[i++] = clamp(1 - bestOtherD / 400);
    s[i++] = 1;
    s[i++] = clamp(agent.bonds.get(bestOther.id) ?? 0);
    s[i++] = clamp(bestOther.energy / 120, 0, 1);
  } else {
    s[i++] = 0; s[i++] = 0; s[i++] = 0;
    s[i++] = 0; s[i++] = 0; s[i++] = 0;
  }

  // Mark (3)
  if (agent.mark.active) {
    s[i++] = 1;
    s[i++] = agent.inMirror ? 1 : 0;
    s[i++] = clamp(agent.markTouchProgress, 0, 1);
  } else {
    s[i++] = 0; s[i++] = 0; s[i++] = 0;
  }

  // Curiosity (3)
  s[i++] = clamp(agent.curiosity.lastPredictionError * 5, 0, 1);
  s[i++] = clamp(agent.curiosity.noveltyScore, 0, 1);
  s[i++] = clamp(agent.curiosity.explorationDrive, 0, 1);

  // Global signals (3)
  s[i++] = clamp(Math.sin(world.t * 0.05));
  s[i++] = clamp(Math.cos(world.t * 0.05));
  s[i++] = clamp(world.agents.length / 120, 0, 1);

  return s;
}