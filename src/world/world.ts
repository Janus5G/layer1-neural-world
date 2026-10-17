import { sense } from "./sensors.js";
import type { CudaTrainer } from "../neural/cudaTrainer.js";
import type { Agent, Event, World } from "./agent.js";
import { crossoverGenome, makeAgent } from "./agent.js";
import { ACTIONS } from "./types.js";
import {
  OllamaCognition,
  DEFAULT_OLLAMA_CONFIG,
  type OllamaConfig,
} from "../cognition/ollama.js";

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const rand = (a = 1) => Math.random() * a;
const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
  Math.hypot(a.x - b.x, a.y - b.y);

export interface Config {
  worldWidth: number;
  worldHeight: number;
  initialPopulation: number;
  maxPopulation: number;
  foodTarget: number;
  hazardCount: number;
  mirrorCount: number;
  mutationRate: number;
  trainingInterval: number;
  learningRate: number;
  gamma: number;
  lambda: number;
  maxRuntimeMinutes: number;
  curiosityCoef: number;
  noveltyDecay: number;
  markChance: number;
  ideaThreshold: number;
  entropyCoef: number;
}

export const DEFAULT_CONFIG: Config = {
  worldWidth: 960,
  worldHeight: 640,
  initialPopulation: 24,
  maxPopulation: 120,
  foodTarget: 140,
  hazardCount: 4,
  mirrorCount: 7,
  mutationRate: 0.15,
  trainingInterval: 128,
  learningRate: 3e-4,
  gamma: 0.99,
  lambda: 0.95,
  maxRuntimeMinutes: 90,
  curiosityCoef: 0.5,
  noveltyDecay: 0.02,
  markChance: 0.30,
  ideaThreshold: 0.85,
  entropyCoef: 0.04,
};

export class Sim {
  cfg: Config;
  world: World;
  private startedAt = Date.now();
  paused = false;
  cognition: OllamaCognition;
  private deliberationCooldowns = new Map<number, number>();
  private traceAgentId: number | null = null;
  private cognitionRunning = false;
  cudaTrainer: CudaTrainer | null = null;
  cudaError: string | null = null;
  cudaBusy = false;

  constructor(cfg: Config = DEFAULT_CONFIG, oc: OllamaConfig = DEFAULT_OLLAMA_CONFIG) {
    this.cfg = cfg;
    this.cognition = new OllamaCognition(oc);
    this.world = this.freshWorld();
  }

  reconfigureCognition(oc: OllamaConfig) {
    this.cognition.reconfigure(oc);
  }

  private idc = 1;
  private nextId = () => this.idc++;

  private freshWorld(): World {
    this.idc = 1;
    this.startedAt = Date.now();
    const w: World = {
      t: 0,
      w: this.cfg.worldWidth,
      h: this.cfg.worldHeight,
      agents: [],
      food: [],
      hazards: [],
      mirrors: [],
      events: [],
      nextId: 0,
      births: 0,
      deaths: 0,
    };
    const helper = { nextId: () => this.nextId() };
    for (let i = 0; i < this.cfg.initialPopulation; i++) {
      w.agents.push(
        makeAgent(
          this.nextId(),
          { x: rand(w.w), y: rand(w.h) },
          helper,
          0,
          this.cfg.markChance,
        ),
      );
    }
    for (let i = 0; i < this.cfg.foodTarget; i++) {
      w.food.push({ id: this.nextId(), pos: { x: rand(w.w), y: rand(w.h) } });
    }
    for (let i = 0; i < this.cfg.hazardCount; i++) {
      w.hazards.push({
        id: this.nextId(),
        pos: { x: 80 + rand(w.w - 160), y: 80 + rand(w.h - 160) },
        radius: 30 + rand(30),
      });
    }
    for (let i = 0; i < this.cfg.mirrorCount; i++) {
      w.mirrors.push({
        id: this.nextId(),
        pos: { x: 60 + rand(w.w - 120), y: 60 + rand(w.h - 120) },
        radius: 55,
      });
    }
    return w;
  }

  reset() {
    this.cudaTrainer?.invalidate();
    this.world = this.freshWorld();
    this.deliberationCooldowns.clear();
  }

  pushEvent(e: Omit<Event, "t">) {
    this.world.events.push({ t: this.world.t, ...e });
    if (this.world.events.length > 500) {
      this.world.events.splice(0, this.world.events.length - 500);
    }
  }

  addIdea(a: Agent, text: string, source: "curiosity" | "llm" | "mirror", strength: number) {
    a.ideas.push({ t: this.world.t, text, source, strength });
    if (a.ideas.length > 10) a.ideas.shift();
    this.pushEvent({ kind: "idea", agentId: a.id, text: `${a.name} thinks: "${text}"` });
  }

  setTraceAgent(id: number | null) {
    this.traceAgentId = id;
  }

  step(dt: number, gpu?: Map<number,{state:Float32Array;probs:Float32Array;value:number}>) {
    if (this.paused) return;
    const cfg = this.cfg;
    const w = this.world;
    w.t += dt;

    if (Date.now() - this.startedAt > cfg.maxRuntimeMinutes * 60_000) {
      this.paused = true;
      this.pushEvent({ kind: "system", text: "Auto-paused: max runtime." });
      return;
    }

    if (w.food.length < cfg.foodTarget && Math.random() < dt * 4) {
      w.food.push({ id: this.nextId(), pos: { x: rand(w.w), y: rand(w.h) } });
    }

    const newborns: Agent[] = [];

    for (const a of w.agents) {
      if (!a.alive) continue;

      // ---- Curiosity update ----
      const cur = a.curiosity;
      const precomputed = gpu?.get(a.id);
      const state = precomputed?.state ?? sense(a, w);

      if (cur.recentStates.length > 0) {
        let minDist = Infinity;
        for (const prev of cur.recentStates) {
          let d = 0;
          for (let i = 0; i < state.length; i++) d += (state[i] - prev[i]) ** 2;
          d = Math.sqrt(d / state.length);
          if (d < minDist) minDist = d;
        }
        cur.lastPredictionError = Math.max(0, 1 - minDist);
        cur.noveltyScore = clamp(
          cur.noveltyScore * (1 - cfg.noveltyDecay) + minDist * 0.3,
          0, 1,
        );
        cur.explorationDrive = clamp(
          cur.noveltyScore * 0.7 + cur.lastPredictionError * 0.3,
          0, 1,
        );
      }
      cur.recentStates.push(state.slice());
      if (cur.recentStates.length > 16) cur.recentStates.shift();

      // ---- Mirror detection ----
      a.inMirror = w.mirrors.some((m) => dist(a.pos, m.pos) < m.radius);

      // ---- Brain forward + action ----
      if (precomputed) { a.brain.probs.set(precomputed.probs); a.brain.value = precomputed.value; }
      else a.brain.forward(state);
      const { action, logProb } = a.brain.sampleAction();
      const estimatedValue = a.brain.value;
      a.lastAction = action;
      a.lastActionName = ACTIONS[action];

      if (this.traceAgentId === a.id) {
        a.lastBrainTrace = a.brain.forwardWithTrace(state);
      }

      let reward = 0.01;
      let done = false;

      const drain = (0.018 + a.genome.metabolism * 0.010) * dt;
      a.energy -= drain;
      a.age += dt;

      // ---- Action ----
      const speed = a.genome.maxSpeed * 42;

      if (action === 0) a.vel.y -= speed * dt * 8;
      else if (action === 1) a.vel.y += speed * dt * 8;
      else if (action === 2) a.vel.x += speed * dt * 8;
      else if (action === 3) a.vel.x -= speed * dt * 8;
      else if (action === 4) {
        a.energy = clamp(a.energy + 9 * dt, 0, 130);
        reward += a.energy < 40 ? 0.3 : -0.05;
      } else if (action === 5) {
        let ate = false;
        for (let i = w.food.length - 1; i >= 0; i--) {
          const f = w.food[i];
          if (dist(a.pos, f.pos) < 12) {
            w.food.splice(i, 1);
            a.energy = clamp(a.energy + 28, 0, 130);
            a.affect.pleasure = clamp(a.affect.pleasure + 0.4, 0, 1);
            reward += 1.5;
            ate = true;
            break;
          }
        }
        if (!ate) reward -= 0.02;
      } else if (action === 6) {
        let socialized = false;
        for (const o of w.agents) {
          if (o === a || !o.alive) continue;
          if (dist(a.pos, o.pos) < 18) {
            const c = a.bonds.get(o.id) ?? 0;
            a.bonds.set(o.id, clamp(c + 0.15 * dt, -1, 1));
            o.bonds.set(a.id, clamp((o.bonds.get(a.id) ?? 0) + 0.10 * dt, -1, 1));
            a.affect.pleasure = clamp(a.affect.pleasure + 0.2 * dt, 0, 1);
            reward += 0.05;
            socialized = true;
            break;
          }
        }
        if (!socialized) reward -= 0.02;
      } else if (action === 7) {
        for (const o of w.agents) {
          if (o === a || !o.alive) continue;
          if (dist(a.pos, o.pos) < 16) {
            const mutual = (o.bonds.get(a.id) ?? 0) > 0.3;
            if (
              mutual &&
              a.reproCooldown <= 0 && o.reproCooldown <= 0 &&
              a.energy > 70 && o.energy > 70 &&
              w.agents.length + newborns.length < cfg.maxPopulation
            ) {
              a.energy -= 34; o.energy -= 34;
              a.reproCooldown = 25; o.reproCooldown = 25;
              a.affect.pleasure = clamp(a.affect.pleasure + 0.8, 0, 1);
              o.affect.pleasure = clamp(o.affect.pleasure + 0.8, 0, 1);
              reward += 3.0;

              const childGenome = crossoverGenome(a.genome, o.genome, cfg.mutationRate);
              const child = makeAgent(
                this.nextId(),
                {
                  x: (a.pos.x + o.pos.x) / 2 + (Math.random() - 0.5) * 10,
                  y: (a.pos.y + o.pos.y) / 2 + (Math.random() - 0.5) * 10,
                },
                { nextId: () => this.nextId() },
                Math.max(a.generation, o.generation) + 1,
                cfg.markChance,
              );

              child.brain.l1.inheritFrom(a.brain.l1, o.brain.l1, cfg.mutationRate);
              child.brain.l2.inheritFrom(a.brain.l2, o.brain.l2, cfg.mutationRate);
              child.brain.policyHead.inheritFrom(a.brain.policyHead, o.brain.policyHead, cfg.mutationRate);
              child.brain.valueHead.inheritFrom(a.brain.valueHead, o.brain.valueHead, cfg.mutationRate);

              child.genome = childGenome;
              child.color = `hsl(${childGenome.hue} 70% 60%)`;
              child.bonds.set(a.id, 0.5);
              child.bonds.set(o.id, 0.5);
              a.bonds.set(child.id, 0.6);
              o.bonds.set(child.id, 0.6);
              newborns.push(child);
              w.births++;
              this.pushEvent({
                kind: "birth",
                agentId: a.id,
                otherId: o.id,
                text: `${a.name} and ${o.name} had ${child.name} (gen ${child.generation})`,
              });
              break;
            } else {
              reward -= 0.05;
            }
          }
        }
      } else if (action === 8) {
        let nearest: { x: number; y: number } | null = null;
        let bd = Infinity;
        for (const h of w.hazards) {
          const d = dist(a.pos, h.pos);
          if (d < bd) { bd = d; nearest = h.pos; }
        }
        if (nearest && bd < 200) {
          const dx = a.pos.x - nearest.x;
          const dy = a.pos.y - nearest.y;
          const m = Math.hypot(dx, dy) || 1;
          a.vel.x += (dx / m) * speed * dt * 10;
          a.vel.y += (dy / m) * speed * dt * 10;
          reward += 0.02;
        }
      } else if (action === 9) {
        // touch_mark
        if (a.mark.active && a.inMirror && !a.markTouched) {
          a.markTouchProgress = clamp(a.markTouchProgress + 0.3 * dt, 0, 1);
          reward += 0.5 * dt;
          a.affect.pleasure = clamp(a.affect.pleasure + 0.2 * dt, 0, 1);
          if (a.markTouchProgress >= 1) {
            a.markTouched = true;
            reward += 5.0;
            a.affect.pleasure = clamp(a.affect.pleasure + 0.6, 0, 1);
            this.pushEvent({
              kind: "mirror_test",
              agentId: a.id,
              text: `${a.name} completed the mirror-mark task — touched their mark!`,
            });
            this.addIdea(a, "I touched the mark. It is mine. I am me.", "mirror", 0.9);
          }
        } else {
          reward -= 0.1;
        }
      } else if (action === 10) {
        // seek_novel
        const targetX = rand(w.w);
        const targetY = rand(w.h);
        const dx = targetX - a.pos.x;
        const dy = targetY - a.pos.y;
        const m = Math.hypot(dx, dy) || 1;
        a.vel.x += (dx / m) * speed * dt * 6;
        a.vel.y += (dy / m) * speed * dt * 6;
        const intrinsicReward = cur.noveltyScore * cfg.curiosityCoef;
        reward += intrinsicReward * dt;
        a.affect.pleasure = clamp(a.affect.pleasure + intrinsicReward * 0.1 * dt, 0, 1);
      }

      // ---- Reward shaping: an agent with a mark is encouraged toward mirrors ----
      if (a.mark.active && !a.markTouched) {
        for (const m of w.mirrors) {
          const d = dist(a.pos, m.pos);
          if (d < m.radius * 2.5) {
            const closeness = 1 - d / (m.radius * 2.5);
            reward += 0.25 * closeness * dt;
            a.affect.pleasure = clamp(a.affect.pleasure + 0.15 * closeness * dt, 0, 1);
          }
        }
      }

      // Hazard exposure
      for (const h of w.hazards) {
        if (dist(a.pos, h.pos) < h.radius) {
          a.integrity -= (6 * dt) / a.genome.resilience;
          reward -= 0.5 * dt;
          a.affect.valence = clamp(a.affect.valence - 0.8 * dt, -1, 1);
          a.affect.arousal = clamp(a.affect.arousal + 0.9 * dt, 0, 1);
        }
      }

      // Affect decay (slower so the signals persist longer)
      a.affect.pleasure = clamp(a.affect.pleasure * (1 - 0.25 * dt), 0, 1);
      a.affect.valence = clamp(a.affect.valence * (1 - 0.20 * dt), -1, 1);
      a.affect.arousal = clamp(a.affect.arousal * (1 - 0.35 * dt), 0, 1);

      // Death
      if (a.energy <= 0 || a.integrity <= 0) {
        a.alive = false;
        reward -= 10;
        done = true;
        w.deaths++;
        this.pushEvent({ kind: "death", agentId: a.id, text: `${a.name} died (gen ${a.generation})` });
      }

      // Physics
      const maxSpeed = a.genome.maxSpeed * 44;
      const sp = Math.hypot(a.vel.x, a.vel.y);
      if (sp > maxSpeed) {
        a.vel.x = (a.vel.x / sp) * maxSpeed;
        a.vel.y = (a.vel.y / sp) * maxSpeed;
      }
      a.pos.x += a.vel.x * dt;
      a.pos.y += a.vel.y * dt;
      a.vel.x *= 0.94;
      a.vel.y *= 0.94;

      if (a.pos.x < 0) a.pos.x += w.w;
      if (a.pos.x > w.w) a.pos.x -= w.w;
      if (a.pos.y < 0) a.pos.y += w.h;
      if (a.pos.y > w.h) a.pos.y -= w.h;

      a.reproCooldown = Math.max(0, a.reproCooldown - dt);
      a.lastReward = reward;
      a.totalReward += reward;

      a.brain.buffer.push({ state, action, logProb, value: estimatedValue, reward, done });
      if (!this.cudaTrainer && a.brain.buffer.length >= cfg.trainingInterval) {
        a.brain.train(cfg.gamma, cfg.lambda, cfg.learningRate, 0.5, cfg.entropyCoef);
      }

      // Idea generation without an LLM (curiosity-driven, now less frequent)
      if (cur.explorationDrive > cfg.ideaThreshold && Math.random() < dt * 0.08) {
        const ideas = [
          "What lies beyond the horizon?",
          "I want to find a place I have never been.",
          "Maybe there is something to learn over there.",
          "I feel drawn toward the unknown.",
          "There must be more than food and danger.",
        ];
        const idea = ideas[Math.floor(Math.random() * ideas.length)];
        this.addIdea(a, idea, "curiosity", cur.explorationDrive);
      }
    }

    for (const n of newborns) w.agents.push(n);
    w.agents = w.agents.filter((a) => a.alive);

    if (w.agents.length < 4) {
      const helper = { nextId: () => this.nextId() };
      for (let i = 0; i < 6; i++) {
        w.agents.push(
          makeAgent(
            this.nextId(),
            { x: rand(w.w), y: rand(w.h) },
            helper,
            0,
            cfg.markChance,
          ),
        );
      }
      this.pushEvent({ kind: "system", text: "Population replenished." });
    }
  }

  /** GPU path uses batched CUDA policy inference and CUDA training.
   * The independent CPU step() remains the reference behavior. In GPU mode,
   * agents observe one pre-action world snapshot per tick rather than
   * seeing earlier agents' intra-tick movement.
   */
  async stepCuda(dt: number): Promise<void> {
    if(!this.cudaTrainer) { this.step(dt); return; }
    if(this.paused) return;
    this.cudaBusy = true;
    try {
      // Construct sensor vectors before mutating the world; the independent
      // CPU path retains its exact original sequential sensing semantics.
      const futureWorld = {...this.world, t:this.world.t+dt};
      const inputs = this.world.agents.filter(a=>a.alive).map(a=>({
        id:a.id,brain:a.brain,state:sense(a,futureWorld)
      }));
      const outputs = inputs.length ? await this.cudaTrainer.inferBatch(inputs) :
        new Map<number,{state:Float32Array;probs:Float32Array;value:number}>();
      if(this.paused)return;
      this.step(dt,outputs);
      if(this.paused)return;
      const ready = this.world.agents.filter(a => a.alive && a.brain.buffer.length >= this.cfg.trainingInterval);
      if(ready.length)await this.cudaTrainer.trainBatch(ready.map(a=>({id:a.id,brain:a.brain})),this.cfg);
    } catch(e) {
      this.cudaError=String(e).slice(0,220);
      this.cudaTrainer?.close();this.cudaTrainer=null;
      this.paused=true;
      this.pushEvent({kind:"system",text:"CUDA error: simulation paused; restart required, no silent CPU fallback."});
    } finally { this.cudaBusy=false; }
  }

  async stepCognition() {
    if (this.cognitionRunning) return;
    if (!this.paused) {
      this.cognitionRunning = true;
      try {
        let budget = 2;
        for (const a of this.world.agents) {
          if (budget <= 0) break;
          if (!a.alive) continue;
          const cd = this.deliberationCooldowns.get(a.id) ?? 0;
          if (this.world.t < cd) continue;

          this.deliberationCooldowns.set(a.id, this.world.t + 20 + Math.random() * 20);
          budget--;

          const payload = {
            self: {
              name: a.name,
              generation: a.generation,
              energy: +a.energy.toFixed(1),
              integrity: +a.integrity.toFixed(1),
              pleasure: +a.affect.pleasure.toFixed(2),
              valence: +a.affect.valence.toFixed(2),
              arousal: +a.affect.arousal.toFixed(2),
              curiosity: +a.curiosity.explorationDrive.toFixed(2),
              novelty: +a.curiosity.noveltyScore.toFixed(2),
              has_mark: a.mark.active,
              in_mirror: a.inMirror,
              mark_touched: a.markTouched,
              current_narrative: a.narrative,
              recent_ideas: a.ideas.slice(-3).map((i) => i.text),
            },
            world: {
              population: this.world.agents.length,
              t: +this.world.t.toFixed(1),
            },
          };

          const result = await this.cognition.deliberate(payload);
          if (result) {
            if (result.self_reflection && typeof result.self_reflection === "string") {
              a.narrative = result.self_reflection.slice(0, 300);
            }
            if (result.idea && typeof result.idea === "string") {
              this.addIdea(a, result.idea.slice(0, 200), "llm", 0.6);
            }
            if (result.speech && typeof result.speech === "string") {
              this.pushEvent({
                kind: "speech",
                agentId: a.id,
                text: `${a.name}: "${result.speech.slice(0, 160)}"`,
              });
            }
          }
        }
      } finally {
        this.cognitionRunning = false;
      }
    }
  }

  snapshot() {
    const w = this.world;
    const avgPleasure = w.agents.length
      ? w.agents.reduce((s, a) => s + a.affect.pleasure, 0) / w.agents.length
      : 0;
    const avgReward = w.agents.length
      ? w.agents.reduce((s, a) => s + a.totalReward / Math.max(1, a.age), 0) / w.agents.length
      : 0;
    const maxGen = w.agents.reduce((m, a) => Math.max(m, a.generation), 0);
    const mirrorsPassed = w.events.filter((e) => e.kind === "mirror_test").length;

    return {
      t: +w.t.toFixed(2),
      paused: this.paused,
      w: w.w,
      h: w.h,
      agents: w.agents.map((a) => {
        const lastIdea = a.ideas.length ? a.ideas[a.ideas.length - 1] : null;
        return {
          id: a.id,
          name: a.name,
          x: +a.pos.x.toFixed(2),
          y: +a.pos.y.toFixed(2),
          energy: +a.energy.toFixed(1),
          integrity: +a.integrity.toFixed(1),
          age: +a.age.toFixed(1),
          generation: a.generation,
          pleasure: +a.affect.pleasure.toFixed(3),
          valence: +a.affect.valence.toFixed(3),
          arousal: +a.affect.arousal.toFixed(3),
          color: a.color,
          action: a.lastActionName,
          reward: +a.lastReward.toFixed(3),
          totalReward: +a.totalReward.toFixed(2),
          hue: a.genome.hue,
          hasMark: a.mark.active,
          markHue: a.mark.hue,
          inMirror: a.inMirror,
          markTouched: a.markTouched,
          markProgress: +a.markTouchProgress.toFixed(2),
          curiosity: +a.curiosity.explorationDrive.toFixed(3),
          novelty: +a.curiosity.noveltyScore.toFixed(3),
          lastIdea: lastIdea ? lastIdea.text : null,
          lastIdeaT: lastIdea ? +lastIdea.t.toFixed(2) : 0,
          lastIdeaSource: lastIdea ? lastIdea.source : null,
          narrative: a.narrative,
        };
      }),
      food: w.food.map((f) => ({ id: f.id, x: f.pos.x, y: f.pos.y })),
      hazards: w.hazards.map((h) => ({ id: h.id, x: h.pos.x, y: h.pos.y, r: h.radius })),
      mirrors: w.mirrors.map((m) => ({ id: m.id, x: m.pos.x, y: m.pos.y, r: m.radius })),
      events: w.events.slice(-60),
      stats: {
        population: w.agents.length,
        births: w.births,
        deaths: w.deaths,
        avgPleasure: +avgPleasure.toFixed(3),
        avgRewardPerSec: +avgReward.toFixed(3),
        maxGeneration: maxGen,
        mirrorTestsPassed: mirrorsPassed,
        ollama: this.cognition.getStats(),
        neuralBackend: this.cudaTrainer ? "CUDA" : "CPU",
        cudaError: this.cudaError,
      },
    };
  }

  getBrainTrace(agentId: number) {
    const a = this.world.agents.find((x) => x.id === agentId);
    if (!a || !a.lastBrainTrace) return null;
    const t = a.lastBrainTrace;
    return {
      input: Array.from(t.input as Float32Array),
      h1: Array.from(t.h1 as Float32Array),
      h2: Array.from(t.h2 as Float32Array),
      logits: Array.from(t.logits as Float32Array),
      probs: Array.from(t.probs as Float32Array),
      value: t.value,
      actions: ACTIONS,
    };
  }
}