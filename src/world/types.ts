export type Vec = { x: number; y: number };

export interface Genome {
  maxSpeed: number;
  metabolism: number;
  resilience: number;
  sociability: number;
  libido: number;
  hue: number;
}

export interface Affect {
  pleasure: number;
  valence: number;
  arousal: number;
}

export const ACTIONS = [
  "north",
  "south",
  "east",
  "west",
  "rest",
  "eat",
  "social",
  "court",
  "flee",
  "touch_mark",
  "seek_novel",
] as const;

export type ActionName = (typeof ACTIONS)[number];
export const NUM_ACTIONS = ACTIONS.length;

export const INPUT_DIM = 40;
export const HIDDEN_DIM = 48;

export interface Mark {
  active: boolean;
  x: number;
  y: number;
  hue: number;
}

export interface Idea {
  t: number;
  text: string;
  source: "curiosity" | "llm" | "mirror";
  strength: number;
}