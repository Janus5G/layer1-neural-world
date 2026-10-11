# Engine Bridge v1 — Unity and Godot development interface

**Status:** Local TypeScript WebSocket bridge with automated contract tests; **not** an experimentally validated AI, gameplay system, physical AI-to-AI network, or proof of learning. The Unity and Godot examples are **unverified in their respective editors**. The original Layer 1+ simulation remains separate and unchanged by this feature unless a developer explicitly integrates the two.

## The division of responsibilities

- **External engine** (Unity, Godot or another client): constructs the entire world, controls objects, rendering, physics, goals, collisions, sensor observations and reward values.
- **Layer 1+ engine bridge:** creates a separate neural policy for each requested agent, receives a bounded observation vector, samples an action from the existing `Brain` class, and optionally learns from rewards returned on the next step.
- **Standard Layer 1+ simulation:** unchanged; its original browser viewer, WebSocket and `Sim` state are not connected to engine-bridge agents. A Unity/Godot world does **not** appear in the original viewer.

The bridge does not provide persistent storage, shared models across client connections, pre-trained policies, external neural-network links, AI-to-AI messaging, gameplay AI, or a security-hardened WAN service. All weights and agent states disappear when connections close.

## Low-load standalone mode (recommended)

```sh
npm ci
npm run bridge
```

This runs **only the bridge** at `ws://127.0.0.1:9090/engine/ws` (no CPU-intensive built-in artificial-life world). Optionally set `ENGINE_BRIDGE_PORT=NNNN` before starting. The service is **OFF by default** for the existing `npm start` workflow. If you explicitly want the bridge beside the original simulation, use `ENGINE_BRIDGE_ENABLED=1` with `npm start` (environment variable syntax varies by OS). The listener is always bound to the local IPv4 loopback address, regardless of environment variables. It is **not** suitable for public or untrusted network exposure; remote/network setups require additional authentication, transport security and permissions implemented independently.

## Message contract (JSON text over WebSocket)

All client frames must be UTF-8 JSON objects. Protocol version is `1`. When a client connects, the server sends:

```json
{"type":"hello","version":1,"protocol":"layer1-engine-bridge","limits":{"maxClients":4,"maxAgentsPerClient":8,"maxInputDim":256,"maxActions":32,"maxMessageBytes":16384,"maxMessagesPerSecond":40,"trainingInterval":32}}
```

Create one *ephemeral, connection-owned* brain:

```json
{"type":"create","agentId":"villager_1","inputDim":6,"actions":["north","south","east","west","idle"],"learn":false}
```

The response is `{"type":"created","agentId":"villager_1","inputDim":6,"actions":["north","south","east","west","idle"],"learn":false}`.

Send observations (all exactly six elements, all finite in `[-1,1]`) to request decisions:

```json
{"type":"step","agentId":"villager_1","observation":[0.1,0.2,-0.3,0.4,0.6,0.9],"reward":0.0}
```

Example response (values vary because policies are initially random):

```json
{"type":"action","agentId":"villager_1","index":2,"name":"east","value":0.01,"updates":0}
```

Your engine performs the returned action in **its own world**, and sends the resulting reward **in the following `step` message**. Reward applies to the *previous* action; the initial step has no prior action. Rewards must be finite and between `-10` and `10`; the bridge rejects invalid values. When `learn:true`, a reward is mandatory after each previous action; with `learn:false`, training is skipped. Send an episode boundary with `{"type":"step","agentId":"villager_1","done":true,"reward":0}`; the bridge replies `episode_end` and does not choose another action. `{"type":"remove","agentId":"villager_1"}` removes the agent. Optional `requestId` (1–48 ASCII letters/digits/underscores/hyphens) is echoed back in responses.

The incoming `create` action labels and observations define the agent's *entire* action and sensor schema; the example's 6 sensor values represent normalized position (x,z), normalized direction to goal (dx,dz), scaled distance, and a constant placeholder for energy. These are a **demo contract**, not part of the base simulation's 40-dimensional sensors or 11 original actions. To change them, adjust both the client's encoder and its `create` message. The bridge cannot translate unknown sensor semantics for you.

## Learning and resource limits

Policies begin with freshly randomized weights. With `learn:true`, the bridge calls the same handwritten A2C-style training implementation used in the project after 32 complete agent transitions or on an episode boundary with at least two buffered transitions. This does not prove a useful or converged policy, and the original learning algorithm is supplied as experimental code. Only one outstanding action per agent is supported. An incomplete short batch at an episode boundary is dropped. Neural states are client-private and volatile (not saved/recovered).

To limit resource use: maximum 4 clients, maximum 8 agents per client, 1–256 input dimensions, 2–32 actions, 16 KiB/frame and 40 messages/second/connection. Server-side observations are checked for finiteness and range; malformed frames receive an `error` response (with a stable `code`) or the connection is closed for excessive traffic. There is **no** network authentication because the listener is loopback-only. Browser connections carrying an `Origin` header are rejected to reduce cross-site WebSocket abuse; dedicated Unity/Godot clients normally omit it. This is not protection against untrusted local processes, so do not override or proxy the listener onto the public network.

The bridge computes neural actions on the server **CPU**, so it is still possible to overload a weak computer when many clients request frequent training. Begin with one agent and `learn:false`, increase load gradually, and monitor CPU temperature. Ollama is **not** required and is not called by the engine bridge.

## Reference clients

- [`examples/unity/Assets/Scripts/Layer1BrainClient.cs`](../examples/unity/Assets/Scripts/Layer1BrainClient.cs) — Unity desktop C# example using built-in `ClientWebSocket`.
- [`examples/godot/Layer1BrainClient.gd`](../examples/godot/Layer1BrainClient.gd) — Godot 4 GDScript example using `WebSocketPeer`.

Both examples intentionally remain small. They drive an agent toward a scene target through a sampled movement action with a simple reward signal. They are demonstrations of message flow, not ready-made game NPC controllers. Unity and Godot runtime/editor verification must be performed independently.

## Validation and next steps for contributors

The repository `tests/engine-bridge.test.mjs` covers the protocol handshake, agent creation, action selection, bounded observation/reward validation, learning call, episode completion, agent removal, client isolation and agent quota. Existing regression tests also run without launching the full server. Reviewers can execute:

```sh
npm run typecheck
npm test
```

**Not verified:** Unity compilation/runtime, Godot compilation/runtime, real game-engine physics, long-term learning and convergence, WAN/LAN transport, hardware connections, externally hosted models or multi-node/AI-to-AI communication. Full local results are documented in [Engine Bridge validation](ENGINE_BRIDGE_VALIDATION.md). Developers should add engine editor smoke tests and real gameplay scenarios before claiming production readiness.
