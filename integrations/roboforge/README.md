# Layer 1+ → RoboForge MCP handover (proposal only)

This folder is **not** a RoboForge installation. No tools are automatically
registered or authorized. RoboForge's repository must be updated and tested
independently before it can expose any Layer 1+ tools through its managed MCP
gateway.

## Components and ownership

- Layer 1+ owns the artificial-life world and optional local CUDA worker.
- RoboForge owns MCP tool registration, launch policy, input validation and
  optional invocation through its game-development workspace.
- ChromaNeural owns local MCP connection approval, per-tool fingerprint
  approval, task-level authorization and resource accounting.
- DeepSeek Janus stays a separately installed local model. The currently
  inspected RoboForge Janus module is **status only**, not verified inference.

## Current Layer 1+ transport

`npm run mcp:stdio` runs a local read-only adapter using Node.js 20+,
without remote downloads. It requires an already-running Layer 1+ instance
bound to `127.0.0.1:8787`. Environment variable `LAYER1_PORT` can select
another local port. The two exposed tools are:

| MCP tool | Arguments | Return |
| --- | --- | --- |
| `layer1_status_v1` | `{}` | current status and backend |
| `layer1_brain_trace_v1` | `{"agent_id":42}` | cached trace, if present |

These tools are **read-only**. They do not launch simulations, train agents,
execute arbitrary code, invoke Ollama or perform Janus inference. Outputs are
untrusted data.

## Integration acceptance checklist (RoboForge repository)

1. Back up and inspect RoboForge's exact current branch and MCP gateway.
2. Register a bounded, disabled-by-default local adapter as an **additional**
   module. Do not overwrite the current plugin registry or existing Janus
   module; verify plugin count and IDs against the current repository.
3. For a first increment, expose only status and cached trace tools through
   a fixed allow-list. Re-validate schemas and wire length at the RoboForge
   boundary. Never provide arbitrary executable paths or URL arguments.
4. Keep RoboForge's own gateway loopback-only and preserve the current
   metadata-only vs controlled-tool distinctions. Do not let REST provide a
   bypass around MCP approvals.
5. If later adding start/pause/reset/training tools, treat every one as a
   **new controlled tool**; extend signed grants and OS-enforced CPU/GPU/RAM
   budgets only after an explicit review. Current managed-grant verifier is
   scoped to Refract preview, **not** these tools.
6. Test catalog, fingerprints, denied tool calls, hostile inputs, offline
   Layer 1+, restarts, bounded execution, and actual ChromaNeural end-to-end
   connection. No Internet Identity, WorkQueue, ledger, storage, ChromaSpeechAI,
   or legacy core-module changes are authorized.

## Local game-development workflow (future)

ChromaNeural → RoboForge → Layer 1+ for observed agent behavior; optional
DeepSeek Janus (separate GPU runtime) for art concepts; Refract/other tools
only with separately approved capabilities. Outputs are proposals/data, not
commands to run. No inference images or game assets are claimed as
implemented in this release candidate.
