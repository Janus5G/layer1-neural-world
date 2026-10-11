import { Brain } from "../neural/network.js";
import { randn } from "../neural/network.js";
import type { Affect, Genome, Idea, Mark, Vec } from "./types.js";
import { HIDDEN_DIM, INPUT_DIM, NUM_ACTIONS } from "./types.js";

export interface Food { id: number; pos: Vec; }
export interface Hazard { id: number; pos: Vec; radius: number; }
export interface Mirror { id: number; pos: Vec; radius: number; }

export interface Event {
  t: number;
  kind: string;
  text: string;
  agentId?: number;
  otherId?: number;
}

export interface World {
  t: number;
  w: number;
  h: number;
  agents: Agent[];
  food: Food[];
  hazards: Hazard[];
  mirrors: Mirror[];
  events: Event[];
  nextId: number;
  births: number;
  deaths: number;
}

export interface CuriosityState {
  lastPredictionError: number;
  noveltyScore: number;
  explorationDrive: number;
  recentStates: Float32Array[];
}

export interface Agent {
  id: number;
  name: string;
  pos: Vec;
  vel: Vec;
  energy: number;
  integrity: number;
  age: number;
  alive: boolean;
  genome: Genome;
  affect: Affect;
  bonds: Map<number, number>;
  brain: Brain;
  lastAction: number;
  lastActionName: string;
  lastReward: number;
  totalReward: number;
  reproCooldown: number;
  color: string;
  generation: number;

  mark: Mark;
  inMirror: boolean;
  markTouchProgress: number;
  markTouched: boolean;

  curiosity: CuriosityState;
  ideas: Idea[];
  narrative: string;

  lastBrainTrace: any;
}

const NAMES = [
  "Ari","Bel","Cyr","Dax","Eli","Fen","Gia","Hux","Ira","Juno",
  "Kai","Lux","Mira","Nox","Ora","Pax","Quin","Rhea","Sol","Tao",
  "Uma","Vex","Wren","Xan","Yara","Zev","Ash","Bri","Cael","Dune",
];

const NARRATIVES = [
  "The world is new. I do not yet know what I am.",
  "I feel hunger and something else — an urge.",
  "There are others like me. They are moving.",
  "Everything feels cold, but I do not want to stay still.",
  "I do not know why, but I want to see what is out there.",
];

export function randomGenome(): Genome {
  return {
    maxSpeed: 0.6 + Math.random() * 1.0,
    metabolism: 0.6 + Math.random() * 0.8,
    resilience: 0.5 + Math.random() * 0.8,
    sociability: Math.random() * 1.5,
    libido: Math.random() * 1.2,
    hue: Math.random() * 360,
  };
}

export function crossoverGenome(a: Genome, b: Genome, rate: number): Genome {
  const m = (x: number, y: number, lo: number, hi: number) => {
    const v = Math.random() < 0.5 ? x : y;
    const noise = Math.random() < rate ? randn() * 0.1 : 0;
    return Math.max(lo, Math.min(hi, v + noise));
  };
  return {
    maxSpeed: m(a.maxSpeed, b.maxSpeed, 0.2, 2.5),
    metabolism: m(a.metabolism, b.metabolism, 0.2, 2),
    resilience: m(a.resilience, b.resilience, 0.1, 2),
    sociability: m(a.sociability, b.sociability, 0, 2),
    libido: m(a.libido, b.libido, 0, 2),
    hue: (Math.random() < 0.5 ? a.hue : b.hue) + (Math.random() < rate ? randn() * 15 : 0),
  };
}

export function makeAgent(
  id: number,
  pos: Vec,
  _world: { nextId: () => number },
  generation = 0,
  markChance = 0.30,
): Agent {
  const genome = randomGenome();
  return {
    id,
    name: NAMES[(id - 1) % NAMES.length] + "-" + id,
    pos: { ...pos },
    vel: { x: 0, y: 0 },
    energy: 70,
    integrity: 100,
    age: 0,
    alive: true,
    genome,
    affect: { pleasure: 0.25, valence: 0.05, arousal: 0.2 },
    bonds: new Map(),
    brain: new Brain(INPUT_DIM, HIDDEN_DIM, NUM_ACTIONS),
    lastAction: 0,
    lastActionName: "north",
    lastReward: 0,
    totalReward: 0,
    reproCooldown: 10 + Math.random() * 10,
    color: `hsl(${genome.hue} 70% 60%)`,
    generation,

    mark: {
      active: Math.random() < markChance,
      x: (Math.random() - 0.5) * 0.6,
      y: (Math.random() - 0.5) * 0.6,
      hue: Math.random() * 360,
    },
    inMirror: false,
    markTouchProgress: 0,
    markTouched: false,

    curiosity: {
      lastPredictionError: 0.5,
      noveltyScore: 0.3,
      explorationDrive: 0.5,
      recentStates: [],
    },

    ideas: [],
    narrative: NARRATIVES[Math.floor(Math.random() * NARRATIVES.length)],
    lastBrainTrace: null,
  };
}