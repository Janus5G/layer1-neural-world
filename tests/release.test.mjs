import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { Sim, DEFAULT_CONFIG } from "../dist/world/world.js";
import { sense } from "../dist/world/sensors.js";
import { NUM_ACTIONS, INPUT_DIM, HIDDEN_DIM } from "../dist/world/types.js";
import { DEFAULT_OLLAMA_CONFIG, OllamaCognition } from "../dist/cognition/ollama.js";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const get = (p) => readFileSync(resolve(root, p), "utf8");

test("the English viewer and Ollama instructions are present", () => {
  assert.match(get("public/index.html"), /lang="en"/);
  assert.match(get("public/index.html"), /Brain activity/);
  assert.match(get("public/viewer.js"), /Click an agent to inspect it/);
  assert.match(get("src/cognition/ollama.ts"), /All natural-language output must be in English/);
});

test("neural dimensions and optional local model defaults are intact", () => {
  assert.equal(INPUT_DIM, 40);
  assert.equal(HIDDEN_DIM, 48);
  assert.equal(NUM_ACTIONS, 11);
  assert.equal(DEFAULT_OLLAMA_CONFIG.enabled, false);
  assert.equal(DEFAULT_OLLAMA_CONFIG.host, "http://127.0.0.1:11434");
});

test("simulation initializes and steps without Ollama", () => {
  const sim = new Sim(DEFAULT_CONFIG, DEFAULT_OLLAMA_CONFIG);
  assert.equal(sim.world.agents.length, 24);
  assert.equal(sense(sim.world.agents[0], sim.world).length, 40);
  sim.step(0.05);
  assert.equal(sim.world.t, 0.05);
  assert.ok(sim.world.agents.length > 0);
  const state = sim.snapshot();
  assert.equal(state.stats.population, sim.world.agents.length);
  assert.equal(state.mirrors.length, 7);
  assert.equal(state.hazards.length, 4);
});

test("pause and reset controls preserve their baseline state transitions", () => {
  const sim = new Sim(DEFAULT_CONFIG, DEFAULT_OLLAMA_CONFIG);
  sim.paused = true;
  sim.step(0.05);
  assert.equal(sim.world.t, 0);
  sim.paused = false;
  sim.step(0.05);
  assert.ok(sim.world.t > 0);
  sim.reset();
  assert.equal(sim.world.t, 0);
  assert.equal(sim.world.agents.length, 24);
});


test("Ollama requests use an English system prompt and parse JSON response", async () => {
  let payload;
  const server = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ message: { content: JSON.stringify({
      speech: "I would like to explore this place.",
      self_reflection: "I am curious about my surroundings.",
      idea: "Look beyond the horizon."
    }) } }));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  try {
    const port = server.address().port;
    const cognition = new OllamaCognition({
      ...DEFAULT_OLLAMA_CONFIG, host: `http://127.0.0.1:${port}`, enabled: true
    });
    const result = await cognition.deliberate({ self: { energy: 70 } });
    assert.equal(result?.speech, "I would like to explore this place.");
    assert.equal(cognition.getStats().calls, 1);
    assert.equal(payload.stream, false);
    assert.equal(payload.format, "json");
    assert.match(payload.messages[0].content, /All natural-language output must be in English/);
    assert.doesNotMatch(payload.messages[0].content, /på dansk|Svar KUN/);
  } finally {
    server.close();
    await once(server, "close");
  }
});
