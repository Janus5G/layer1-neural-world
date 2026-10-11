# Architecture — Layer 1+

This document explains the experimental implementation in `src/`. It is a technical guide, not scientific validation of emergent behavior.

## Runtime overview

Three scheduled activities share the in-memory `Sim` world state:

1. **Simulation loop (described as 20 Hz):** physics, sensor collection, action selection, reward, and neural training.
2. **Cognition loop (every 5 seconds):** optional requests to a local Ollama model for selected agents.
3. **Broadcast loop (described as 10 Hz):** world snapshots sent to browsers over WebSocket.

The viewer uses Canvas 2D to show agents, neural activity, and optional Ollama settings.

## Neural architecture

**Source file:** `src/neural/network.ts`.

```text
input (40)
  └── Linear(40 → 48) → ReLU → Linear(48 → 48) → ReLU
                               ├── policy_head(48 → 11) → softmax
                               └── value_head(48 → 1)
```

- **Input:** 40 sensor values.
- **Hidden layers:** two layers of 48 units with ReLU.
- **Policy output:** 11 action logits converted to probabilities with softmax.
- **Value output:** one estimated future reward scalar.

### Adam optimizer

Parameters: `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`, learning rate `3e-4`. Each `Linear` layer is described as having its own first/second moment arrays (`mW`, `vW`, `mb`, `vb`) and gradient buffers (`gW`, `gb`). Gradients propagate through the policy and value branches and both hidden layers.

## Observation vector (40 dimensions)

**Source file:** `src/world/sensors.ts`.

| Indices | Count | Meaning |
|---|---:|---|
| 0–7 | 8 | Self: energy, integrity, age, pleasure, valence, arousal, velocity x, velocity y |
| 8–12 | 5 | Genome: max speed, metabolism, resilience, sociability, libido |
| 13–16 | 4 | Nearest food: x/y offset, proximity, presence |
| 17–20 | 4 | Nearest hazard: x/y offset, proximity, presence |
| 21–24 | 4 | Nearest mirror: x/y offset, proximity, presence |
| 25–30 | 6 | Nearest neighbor: x/y offset, proximity, presence, bond, neighbor's energy |
| 31–33 | 3 | Mirror mark: active, visible via mirror, touch progress |
| 34–36 | 3 | Curiosity: prediction-error-like variable, novelty, exploration drive |
| 37–39 | 3 | Global: sin(time), cos(time), population ratio |

The original notes indicate normalized values in `[-1, 1]` or `[0, 1]`.

## Action space (11 actions)

**Source file:** `src/world/types.ts`.

| Index | Action | Described effect |
|---:|---|---|
| 0 | `north` | Accelerate upward |
| 1 | `south` | Accelerate downward |
| 2 | `east` | Accelerate right |
| 3 | `west` | Accelerate left |
| 4 | `rest` | Recover energy, more when energy is low |
| 5 | `eat` | Consume food when sufficiently close |
| 6 | `social` | Increase bond strength near a neighbor |
| 7 | `court` | Reproduce if both agents are eligible/willing |
| 8 | `flee` | Move away from the nearest hazard |
| 9 | `touch_mark` | Touch own mark in the mirror-mark task |
| 10 | `seek_novel` | Travel toward an exploration point with novelty-related reward |

## Reward design

The original description assigns a reward for each agent tick and uses these values as the primary learning signal:

| Condition / action | Described reward |
|---|---:|
| Baseline | `+0.01` |
| Eating food | `+1.5` |
| Social interaction | `+0.05` |
| Reproduction | `+3.0` |
| Rest at low energy | `+0.3` |
| Rest at high energy | `-0.05` |
| Near mirror with active mark | `+0.25 × proximity × dt` |
| In hazard zone | `-0.5 × dt` |
| Death | `-10` |
| `seek_novel` | `+novelty × 0.5 × dt` |

Mirror proximity reward shaping makes the target behavior more likely in the specified training scheme. Consequently, completing this task **is not an independent demonstration of spontaneous self-recognition**.

## A2C training with GAE

**Source file:** `src/neural/network.ts`, `Brain.train()`.

### 1. Experience buffer

Each agent is described as accumulating transitions with fields such as `state`, `action`, `logProb`, `value`, `reward`, and `done`. Training is triggered at `trainingInterval` (described default: 128 transitions).

### 2. Generalized Advantage Estimation

```text
delta_t  = reward_t + gamma * value_(t+1) - value_t
A_t      = delta_t + gamma * lambda * A_(t+1)
return_t = value_t + A_t
```

Settings: `gamma = 0.99` and `lambda = 0.95`. Terminal-state handling and bootstrap details must be confirmed in code.

### 3. Advantage normalization

The notes describe normalizing advantages as z-scores over the training buffer for stability.

### 4. Policy gradient

Illustrative derivatives given in the notes:

```ts
// For the selected action:
d_logits[action] = (probs[action] - 1) * advantage;
// For each other action:
d_logits[i] = probs[i] * advantage;
```

The sign must be interpreted in the context of the actual loss and optimizer implementation; this fragment is not a complete derivation.

### 5. Entropy bonus

```text
H = -sum_i(probs[i] * log(probs[i]))
```

The described `entropyCoef` is `0.04`. The original notes say it is used to discourage premature policy collapse.

### 6. Value objective

Illustrative derivative: `d_value = (value - return) * valueCoef` with described `valueCoef = 0.5`.

### 7. Backpropagation / update

Policy and value gradients join at the shared second hidden layer and propagate through both hidden layers before Adam updates.

## Reproduction and inherited weights

**Source file:** `src/world/world.ts`, `court` action.

- Each genome trait is selected from a parent with a mutation probability that adds Gaussian noise.
- Neural layer weights are mixed similarly for `l1`, `l2`, `policyHead`, and `valueHead`.
- The original notes describe new child energy as `55`.
- Parent–child bond strength is initially described around `0.5–0.6`.

Passing along learned weights resembles **Lamarckian inheritance** in a computational analogy. It is not meant to represent realistic biological heredity.

## Curiosity heuristic

**Source file:** `src/world/world.ts`, inside `step()`.

The project notes describe each agent retaining up to 16 recent observation vectors in `curiosity.recentStates` and computing a minimum distance to previously encountered states. The following updates are described:

```text
predictionError = 1 - minDist
novelty = novelty * (1 - decay) + minDist * 0.3
explorationDrive = novelty * 0.7 + predictionError * 0.3
```

When `explorationDrive` exceeds `ideaThreshold` (described default `0.85`), a curiosity-related idea may be emitted with a small probability (described as `dt * 0.08`).

**Interpretation caution:** The variable named `predictionError` is a heuristic expression in the notes, not evidence of a learned predictive world model. The distance metric, normalization, and exact update logic must be verified in source.

## Mirror-mark task

**Source file:** `src/world/world.ts`, `touch_mark` action.

The original task mechanism is:

1. An agent is born with an active mark at a configurable probability (described default: 30%).
2. A mirror detection zone may set `inMirror = true`.
3. Sensor positions 31–33 provide mark-active, mirror-visible, and touch-progress inputs.
4. The agent can select action `9` (`touch_mark`).
5. When in the mirror zone and marked, the notes describe `markTouchProgress += 0.3 * dt`.
6. At progress `>= 1`, the system sets `markTouched = true`, awards `+5`, and logs an event.
7. An additional shaped reward favors approaching a mirror with an untouched mark; the notes describe `0.25 * (1 - distance / (radius * 2.5)) * dt` while within `2.5 × radius`.

This is a deliberately constructed task with explicit sensors, reward shaping, and completion conditions. It should be labeled a **simulated mirror-mark task**, not used as a stand-alone measure of real self-recognition.

## Optional Ollama integration

**Source file:** `src/cognition/ollama.ts`.

The code posts to a local Ollama HTTP endpoint:

```http
POST http://127.0.0.1:11434/api/chat
Content-Type: application/json
```

```json
{
  "model": "qwen3:4b-instruct",
  "messages": [
    { "role": "system", "content": "<SYSTEM_PROMPT>" },
    { "role": "user", "content": "<JSON payload>" }
  ],
  "stream": false,
  "format": "json",
  "options": { "temperature": 0.8, "num_predict": 300 }
}
```

The reported output is parsed as JSON with fields such as `intent`, `mood`, `speech`, `self_reflection`, `valence_shift`, and `idea`. Described limits are 2 concurrent requests and a budget of 2 calls per cognition round. Verify that the installed model and Ollama version support the selected response format.

The released runtime uses English for its user-facing text, initial narratives, curiosity ideas, and Ollama system prompt.

## WebSocket protocol

**Source file:** `src/index.ts`.

Described messages:

```json
{"type":"state","payload":{"t":0,"paused":false,"w":960,"h":640,"agents":[],"food":[],"hazards":[],"mirrors":[],"events":[],"stats":{}}}
```

```json
{"type":"hello"}
```

A state snapshot is described as containing time, pause status, world dimensions; arrays for live agents, food, hazards, and mirrors; up to 60 recent events; and summary statistics (population, births, deaths, average “pleasure”, Ollama counters). These are illustrative shapes, not a validated API contract.

## Performance considerations

Neural updates can dominate CPU usage. The original notes estimated 128 transitions and four layers per update as 512 per-transition layer calculations per training cycle; actual operation count depends on batch loops, layer sizes, agent count, and implementation.

Possible reductions:

- Lower `maxPopulation` or network width.
- Increase `trainingInterval` to perform updates less often; note that batch sizes rise and memory/timing effects change.
- Reduce simulation ticks or broadcasts.
- Evaluate a GPU-enabled implementation only as a separate project; it is **not** part of this baseline.

Ollama CPU/GPU costs vary by hardware, context length, model, inference settings, and load; do not treat GPU inference as computationally free.
