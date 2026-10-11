# Engine Bridge validation — 2026-10-11

This report describes **local software checks on a Linux test container**; it is not a certification of intelligence, game behavior, reliability under load, or hardware interoperability.

| Check | Result | Evidence |
|---|---|---|
| TypeScript typecheck (`tsc --noEmit`) | PASS | Entire application plus new `src/engineBridge` compiled |
| Compiled Node test suite (`tsc && node --test tests/*.test.mjs`) | PASS | 10/10 tests, including all 5 new bridge tests and 5 previous release regression tests |
| Bridge protocol actions and optional reward training | PASS | Contract test with five decisions and at least one optimizer update |
| Invalid observations, dimensions, actions and reward bounds | PASS | Rejection paths exercised |
| Agent quotas and per-client state isolation | PASS | Quota and isolation tests |
| Browser-origin rejection | PASS | Cross-origin WebSocket handshake rejected |
| Standalone compiled Node process (`node dist/engineBridge/index.js`) | PASS | Started loopback listener and returned a `hello` frame to a test client |
| `npm run bridge` through `tsx` on this Linux container | NOT VERIFIED | The provided `node_modules` originated on Windows; the Linux-specific esbuild executable was unavailable. Run a fresh `npm ci` on your target platform. |
| Unity Editor and real game runtime | NOT VERIFIED | Unity environment unavailable |
| Godot Editor and real game runtime | NOT VERIFIED | Godot environment unavailable |
| Long-running learning, convergence or production-grade performance | NOT VERIFIED | No benchmarks or extended load tests |
| LAN/WAN, physical robots and cross-machine AI-to-AI communication | NOT IMPLEMENTED | Local-only service; not a production transport |

## Reproduce on your own workstation

Install a platform-native Node.js 20+ toolchain and run `npm ci`, `npm run typecheck`, `npm test`, then `npm run bridge`. In Unity or Godot, follow the separate example README. The default `npm start` still starts the original world and is not required for engine integration.

**Operational safety:** The bridge is disabled unless deliberately started, listens exclusively on `127.0.0.1`, rejects browser `Origin` headers, caps clients and training load, and never accesses the original world `Sim`. None of these safeguards should be described as a verified solution for internet-facing operation.
