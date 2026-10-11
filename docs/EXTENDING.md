# Contributor Guide — Extending Layer 1+

The original author considers Layer 1+ a completed **experiment**, not an ongoing development roadmap. This document translates the original contributor-oriented suggestions for people who may wish to build independent extensions.

Examples below are illustrative, not verified drop-in patches. Please examine the current source before editing.

## Before you start

1. Confirm the project runs locally with `npm start`.
2. Study the data flow: `sense()` → `brain.forward()` → action → reward → `brain.buffer` → `brain.train()`.
3. Read `src/world/world.ts`; the original notes identify it as the core simulation implementation.
4. Define an observable hypothesis: which agent behavior should change, and how will you measure it?
5. Work in your own branch or fork. Preserve the experimental baseline unless changes are intentional and tested.

## Add a new action

To extend the simulation, edit `ACTIONS` in `src/world/types.ts`:

```ts
export const ACTIONS = [
  "north", "south", "east", "west", "rest",
  "eat", "social", "court", "flee",
  "touch_mark", "seek_novel",
  "dig", // New illustrative action.
] as const;
```

If `NUM_ACTIONS` is derived from the action array, it should become 12. Add an appropriate action handler in `src/world/world.ts`:

```ts
} else if (action === 11) {
  // Example: dig / hide to reduce arousal.
  a.affect.arousal = clamp(a.affect.arousal - 0.3 * dt, 0, 1);
  reward += 0.05;
}
```

If a new action needs additional observation data, extend the sensor layout and corresponding input dimension. A changed action-count changes the neural output shape, so saved models, if they ever exist, may be incompatible and may need a controlled migration or reset.

## Add a sensor

1. Increase `INPUT_DIM` in `src/world/types.ts` (example: `40` to `44`).
2. Add precisely the new observations to `src/world/sensors.ts` in the correct order.
3. Update every sensor-layout comment and consumer of that dimension.
4. Reset or migrate incompatible agents and checkpoints.
5. Test numerical ranges, absent-target cases, and normalization.

## Add a reward source

Identify the relevant branch in the simulation step and add a reward with a clear measured purpose:

```ts
reward += /* your justified incremental reward */ 0.05;
```

The original notes suggest rewards on the order of `0.01` to `5.0`, cautioning that large rewards (such as above `10`) could destabilize training, and recommending sparing use of negative rewards. These are heuristics, **not universal rules**.

Log reward distributions and compare multiple random seeds. Avoid the risk that reward shaping makes a desired behavior appear “emergent” when it is directly incentivized.

## Try another RL algorithm (PPO example)

The original notes proposed copying `src/neural/network.ts` to a separate PPO implementation, then adjusting the training loop. This is only a sketch:

```ts
train(gamma, lambda, clipRatio = 0.2, lr, epochs = 4) {
  // Preserve the old action log-probabilities from each transition.
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let t = 0; t < buffer.length; t++) {
      const { probs } = this.forward(buffer[t].state);
      const ratio = Math.exp(
        Math.log(probs[buffer[t].action]) - buffer[t].logProb
      );
      const clipped = Math.max(
        Math.min(ratio, 1 + clipRatio), 1 - clipRatio
      );
      const advantage = buffer[t].advantage; // Compute before PPO training.
      // Optimize -min(ratio * advantage, clipped * advantage).
      // Implement correct gradients, normalization, value and entropy losses.
    }
  }
}
```

A complete PPO implementation requires appropriate transition storage, returns and advantage handling, clipping, numerical stability, optimizer updates, test coverage, and a validation strategy. It is not achieved by copying the snippet alone.

## Add persistence

The original notes include a partial `src/persistence.ts` example and recommend serializing the following fields:

```ts
export function serializeAgent(a: Agent): SerializedAgent {
  return {
    id: a.id,
    name: a.name,
    generation: a.generation,
    genome: a.genome,
    brain: {
      l1: {
        W: Array.from(a.brain.l1.W),
        b: Array.from(a.brain.l1.b),
      },
      l2: {
        W: Array.from(a.brain.l2.W),
        b: Array.from(a.brain.l2.b),
      },
      policy: {
        W: Array.from(a.brain.policyHead.W),
        b: Array.from(a.brain.policyHead.b),
      },
      value: {
        W: Array.from(a.brain.valueHead.W),
        b: Array.from(a.brain.valueHead.b),
      },
    },
    bonds: Array.from(a.bonds.entries()),
    ideas: a.ideas,
    narrative: a.narrative,
    mark: a.mark,
  };
}
```

This example is intentionally partial. A working snapshot system also needs position, velocity, timers, RNG state where reproducibility matters, enough optimizer state to resume training, world objects, versioning, and migration/error handling. `Float32Array` and `Map` require deliberate serialization. JSON is simple but can become large; binary formats may be more compact.

## Add an LLM provider

The original `provider.ts` sketch uses:

```ts
export interface CognitionProvider {
  deliberate(payload: AgentPayload): Promise<DeliberationResult | null>;
  getStats(): ProviderStats;
  reconfigure(cfg: unknown): void;
}
```

A contributor might implement different adapters and make the cognition loop talk to a router. Take care with different API request formats, structured JSON support, timeouts, per-provider usage limits, credentials, and costs. External models must be explicitly opt-in. A provider fallback should **not** silently send local information to a cloud service.

## Modify the viewer without changing the simulation

The notes describe the viewer as files under `public/`:

- `style.css`: layout, colors, and typography.
- `index.html`: accessible structure and controls.
- `viewer.js`: visualization and WebSocket handling.

If the UI only reads existing WebSocket state and documented APIs, its behavior can usually be modified separately from the simulation. To add new data, update `Sim.snapshot()` (described as living in `world.ts`) and the corresponding viewer logic. Preserve compatibility for existing consumers.

## Troubleshooting from the original notes

**An agent appears inactive:** Verify that `brain.buffer.length` can reach the configured training interval, that the simulation is not paused, and that actions are being selected. A shorter interval may be useful for diagnosis, but affects learning behavior.

**A policy is stuck on one action:** Inspect probabilities in the brain trace. A value around `0.99` for one action may indicate premature policy concentration. Inspect reward scales and the entropy term before changing coefficients.

**Agents die too often:** Evaluate metabolism, target food count, hazard count, and agent navigation. Change one variable at a time and measure outcomes.

**Learning is not visible:** Inspect reward magnitude, advantage variance, loss/entropy trends, and behavior compared with an untrained or random baseline. Avoid concluding that there is no learning from a single run.

**Ollama does not respond:** Check service status, available models and the configured name. Examples from the original notes:

```bash
ollama ps
ollama list
curl http://127.0.0.1:11434/api/version
```

**Memory consumption grows:** Check event logs, idea histories, viewer trails, live agent buffers, and any new persisted objects. The source notes suggest event retention around 500 in memory, displayed snapshot events around 60, 10 ideas per agent, and 30 trail entries per agent. Verify those limits in the original implementation.

## Tests to add

This release includes a small Node built-in regression suite (`tests/release.test.mjs`), but it is **not** a full behavioral test suite. Possible additional independent contributions:

- Unit tests: `Linear.forward()`, `Linear.backward()`, `Linear.adam()`.
- Integration tests: run controlled simulated ticks, verify finite tensors and a viable agent population under documented conditions.
- Reproducibility: fixed RNG seed, same initial state, same output after N ticks if the simulation is intended to be deterministic (disable or mock timing/LLM calls).
- Task tests: reward-shaping mechanics, mirror-mark completion conditions, and agent state resets.
- UI smoke tests: viewer launch, Pause, Reset, and disabled Ollama behavior.

Suggested structure:

```text
tests/
├── neural.test.ts
├── world.test.ts
└── integration.test.ts
```

The original notes suggested Vitest. This experimental release instead uses Node's built-in test runner without adding an extra test framework. Proposed additional tests above are not yet implemented.

## Pull request guidelines

Describe the change, name the affected behaviors, include reproducible tests, and report any failure modes. When a contribution touches agent “feelings” or simulated suffering, explain which numerical state variables were changed, rather than making experiential claims. Maintain English-language documentation and user-facing text for the public release.
