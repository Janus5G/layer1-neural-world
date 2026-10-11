/**
 * Opt-in local-only neural action service for external simulation engines.
 * Each WebSocket connection owns independent, ephemeral Brain instances.
 * The game engine remains authoritative for the world, sensors, and rewards.
 */
import { WebSocket, WebSocketServer } from "ws";
import { Brain, type Transition } from "../neural/network.js";

export const ENGINE_BRIDGE_VERSION = 1;
export const BRIDGE_LIMITS = Object.freeze({
  maxClients: 4,
  maxAgentsPerClient: 8,
  maxInputDim: 256,
  maxActions: 32,
  maxMessageBytes: 16_384,
  maxMessagesPerSecond: 40,
  trainingInterval: 32,
});

type AgentController = {
  actions: string[];
  brain: Brain;
  learn: boolean;
  pending: Transition | null;
};

export interface BridgeOptions {
  /** Port 0 is supported for ephemeral-port tests. */
  port?: number;
  /** Override only for unit tests; never bind this service to a public interface. */
  trainingInterval?: number;
}

function validId(x: unknown): x is string {
  return typeof x === "string" && /^[A-Za-z0-9_-]{1,48}$/.test(x);
}

function finiteVector(x: unknown, dim: number): x is number[] {
  return Array.isArray(x) && x.length === dim && x.every((n) => typeof n === "number" && Number.isFinite(n) && Math.abs(n) <= 1);
}

function send(ws: WebSocket, data: object): void {
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data));
}

/** Starts a separate loopback listener; does not mutate the built-in Sim. */
export function startEngineBridge(options: BridgeOptions = {}): WebSocketServer {
  const trainingInterval = options.trainingInterval ?? BRIDGE_LIMITS.trainingInterval;
  if (!Number.isInteger(trainingInterval) || trainingInterval < 2 || trainingInterval > 128) {
    throw new RangeError("Invalid training interval");
  }
  if (!Number.isInteger(options.port ?? 9090) || (options.port ?? 9090) < 0 || (options.port ?? 9090) > 65535) {
    throw new RangeError("Invalid bridge port");
  }
  const wss = new WebSocketServer({
    host: "127.0.0.1",
    port: options.port ?? 9090,
    path: "/engine/ws",
    maxPayload: BRIDGE_LIMITS.maxMessageBytes,
    perMessageDeflate: false,
    clientTracking: true,
    // Prevent browser pages on unrelated origins from controlling a local neural service.
    // Dedicated Unity/Godot clients normally do not send an Origin header.
    verifyClient: (info: { origin?: string }) => !info.origin,
  });

  wss.on("connection", (ws) => {
    if (wss.clients.size > BRIDGE_LIMITS.maxClients) {
      ws.close(1008, "Connection limit reached");
      return;
    }
    const agents = new Map<string, AgentController>();
    let windowStart = Date.now();
    let framesThisSecond = 0;
    const fail = (code: string, requestId?: string) => send(ws, { type: "error", code, ...(requestId ? { requestId } : {}) });

    send(ws, { type: "hello", version: ENGINE_BRIDGE_VERSION, protocol: "layer1-engine-bridge", limits: BRIDGE_LIMITS });

    ws.on("message", (raw, isBinary) => {
      const now = Date.now();
      if (now - windowStart >= 1000) { windowStart = now; framesThisSecond = 0; }
      if (++framesThisSecond > BRIDGE_LIMITS.maxMessagesPerSecond) {
        ws.close(1008, "Message rate exceeded");
        return;
      }
      if (isBinary) { fail("text_json_only"); return; }
      let msg: Record<string, unknown>;
      try {
        const parsed: unknown = JSON.parse(raw.toString());
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) { fail("invalid_message"); return; }
        msg = parsed as Record<string, unknown>;
      } catch { fail("invalid_json"); return; }
      const requestId = validId(msg.requestId) ? msg.requestId : undefined;

      if (msg.type === "create") {
        if (!validId(msg.agentId)) { fail("invalid_agent_id", requestId); return; }
        if (agents.has(msg.agentId)) { fail("agent_exists", requestId); return; }
        if (agents.size >= BRIDGE_LIMITS.maxAgentsPerClient) { fail("agent_limit", requestId); return; }
        if (!Number.isInteger(msg.inputDim) || (msg.inputDim as number) < 1 || (msg.inputDim as number) > BRIDGE_LIMITS.maxInputDim) {
          fail("invalid_input_dim", requestId); return;
        }
        if (!Array.isArray(msg.actions) || msg.actions.length < 2 || msg.actions.length > BRIDGE_LIMITS.maxActions ||
            !msg.actions.every((v: unknown) => validId(v)) || new Set(msg.actions).size !== msg.actions.length) {
          fail("invalid_actions", requestId); return;
        }
        if (msg.learn !== undefined && typeof msg.learn !== "boolean") { fail("invalid_learn_flag", requestId); return; }
        const controller: AgentController = {
          actions: [...msg.actions],
          brain: new Brain(msg.inputDim as number, 48, msg.actions.length),
          learn: msg.learn === true,
          pending: null,
        };
        agents.set(msg.agentId, controller);
        send(ws, { type: "created", agentId: msg.agentId, inputDim: msg.inputDim, actions: controller.actions,
          learn: controller.learn, ...(requestId ? { requestId } : {}) });
        return;
      }

      if (msg.type === "step") {
        if (!validId(msg.agentId)) { fail("invalid_agent_id", requestId); return; }
        const a = agents.get(msg.agentId);
        if (!a) { fail("unknown_agent", requestId); return; }
        const done = msg.done === true;
        if (msg.done !== undefined && typeof msg.done !== "boolean") { fail("invalid_done", requestId); return; }
        if (!done && !finiteVector(msg.observation, a.brain.inputDim)) { fail("invalid_observation", requestId); return; }
        if (a.pending && a.learn && (typeof msg.reward !== "number" || !Number.isFinite(msg.reward) || Math.abs(msg.reward) > 10)) {
          fail("invalid_reward", requestId); return;
        }
        if (msg.reward !== undefined && (typeof msg.reward !== "number" || !Number.isFinite(msg.reward) || Math.abs(msg.reward) > 10)) {
          fail("invalid_reward", requestId); return;
        }
        if (a.pending) {
          if (a.learn) {
            a.brain.buffer.push({ ...a.pending, reward: msg.reward as number, done });
            if (a.brain.buffer.length >= trainingInterval || (done && a.brain.buffer.length >= 2)) a.brain.train();
          }
          a.pending = null;
        }
        if (done) {
          // Episode boundary: discard incomplete short batches rather than carrying across episodes.
          a.brain.buffer.length = 0;
          send(ws, { type: "episode_end", agentId: msg.agentId, updates: a.brain.updates,
            ...(requestId ? { requestId } : {}) });
          return;
        }
        const state = new Float32Array(msg.observation as number[]);
        a.brain.forward(state);
        const { action, logProb, value } = a.brain.sampleAction();
        a.pending = { state, action, logProb, value, reward: 0, done: false };
        send(ws, { type: "action", agentId: msg.agentId, index: action, name: a.actions[action],
          value, updates: a.brain.updates, ...(requestId ? { requestId } : {}) });
        return;
      }

      if (msg.type === "remove") {
        if (!validId(msg.agentId)) { fail("invalid_agent_id", requestId); return; }
        if (!agents.delete(msg.agentId)) { fail("unknown_agent", requestId); return; }
        send(ws, { type: "removed", agentId: msg.agentId, ...(requestId ? { requestId } : {}) });
        return;
      }
      fail("unknown_type", requestId);
    });
  });
  return wss;
}
