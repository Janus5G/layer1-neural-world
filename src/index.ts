import express from "express";
import http from "node:http";
import path from "node:path";
import { WebSocketServer, WebSocket } from "ws";
import { Sim, DEFAULT_CONFIG } from "./world/world.js";
import { DEFAULT_OLLAMA_CONFIG, type OllamaConfig } from "./cognition/ollama.js";
import { startEngineBridge } from "./engineBridge/server.js";

let ollamaConfig: OllamaConfig = { ...DEFAULT_OLLAMA_CONFIG };
const sim = new Sim(DEFAULT_CONFIG, ollamaConfig);

const app = express();
app.use(express.json({ limit: "128kb" }));
app.use(express.static(path.resolve(process.cwd(), "public")));

// ---- Admin: Ollama config ----
app.get("/api/admin/ollama", (_req, res) => {
  res.json({ ok: true, config: { ...ollamaConfig } });
});

app.post("/api/admin/ollama", (req, res) => {
  const p = req.body ?? {};
  ollamaConfig = {
    host: String(p.host ?? ollamaConfig.host),
    model: String(p.model ?? ollamaConfig.model),
    enabled: Boolean(p.enabled ?? ollamaConfig.enabled),
    temperature: Number(p.temperature ?? ollamaConfig.temperature),
    maxTokens: Number(p.maxTokens ?? ollamaConfig.maxTokens),
    timeoutMs: Number(p.timeoutMs ?? ollamaConfig.timeoutMs),
  };
  sim.reconfigureCognition(ollamaConfig);
  res.json({ ok: true, config: ollamaConfig });
});

// ---- Control ----
app.post("/api/control", (req, res) => {
  const action = req.body?.action;
  if (action === "pause") sim.paused = true;
  else if (action === "resume") sim.paused = false;
  else if (action === "reset") sim.reset();
  else return res.status(400).json({ ok: false });
  res.json({ ok: true });
});

app.get("/api/status", (_req, res) => {
  res.json({ ok: true, paused: sim.paused, t: sim.world.t, pop: sim.world.agents.length });
});

// ---- Brain trace ----
app.get("/api/brain/:id", (req, res) => {
  const id = Number(req.params.id);
  const trace = sim.getBrainTrace(id);
  res.json({ ok: !!trace, trace });
});

app.post("/api/trace", (req, res) => {
  const id = req.body?.agentId;
  sim.setTraceAgent(typeof id === "number" && !Number.isNaN(id) ? id : null);
  res.json({ ok: true });
});

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });

function broadcast(payload: unknown) {
  const data = JSON.stringify(payload);
  for (const c of wss.clients) {
    if (c.readyState === WebSocket.OPEN) c.send(data);
  }
}

wss.on("connection", (ws) => {
  ws.send(JSON.stringify({ type: "hello" }));
});

// Sim-loop: 20 Hz
const TICK_HZ = 20;
const tickMs = 1000 / TICK_HZ;
let last = performance.now();
setInterval(() => {
  const now = performance.now();
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  sim.step(dt);
}, tickMs);

// Cognition loop: runs independently
setInterval(() => {
  void sim.stepCognition();
}, 5000);

// Broadcast: 10 Hz
setInterval(() => {
  broadcast({ type: "state", payload: sim.snapshot() });
}, 100);

const PORT = Number(process.env.PORT ?? 8787);
server.listen(PORT, () => {
  console.log(`Layer 1+ is running at http://localhost:${PORT}`);
  console.log(
    `  Ollama: ${
      ollamaConfig.enabled
        ? ollamaConfig.model + " @ " + ollamaConfig.host
        : "disabled"
    }`,
  );
});
// Opt-in loopback-only engine interface. Not part of the default world simulation.
if (process.env.ENGINE_BRIDGE_ENABLED === "1") {
  const port = Number(process.env.ENGINE_BRIDGE_PORT ?? "9090");
  const bridge = startEngineBridge({ port });
  bridge.on("listening", () => console.log(`Engine Bridge: ws://127.0.0.1:${port}/engine/ws`));
  bridge.on("error", (err) => console.error("Engine Bridge failed:", err));
}
