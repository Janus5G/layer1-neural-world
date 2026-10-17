# Layer 1+ v1.2.0 — Linux CUDA and local MCP release evidence

Release scope: original experimental Layer 1+ artificial-life world, with an
**optional** Linux-only PyTorch CUDA neural backend, and local, read-only MCP
stdio bridge. Original CPU mode remains available.

## Observed evidence (17 October 2026)

- GitHub Actions (head 85256a3): TypeScript/Node/MCP/Engine Bridge regression
  and CUDA/MCP CPU-side workflow both passed. Additional checks required on
  the release commit, as it contains installer/documentation updates.
- Owner's real Ubuntu WSL2 test, NVIDIA GeForce RTX 5050 Laptop GPU (8 GiB):
  12/12 Node.js regression checks, 5/5 offline Python neural contract checks.
- Real CUDA smoke test: CUDA inference, A2C/Adam update, and GPU weight-cache
  update passed.
- 35-second integrated GPU run: 25.59 simulated seconds; peak observed GPU
  temperature 53 °C.
- 20-minute integrated run: 1200 real seconds, 678.1 simulated seconds
  (0.565× real time); 239 GPU observations, maximum 64 °C, peak 138 MiB GPU
  VRAM; CUDA remained selected; launched process group stopped cleanly.
- CPU **temperature** was **not measured** under WSL2. WSL CPU utilization
  figures do not substitute for thermal measurements.
- GPU utilization was low/intermittent. No speedup relative to CPU was established.
  Model convergence, CPU-vs-GPU behavior parity, Unity/Godot native-editor
  integration, RoboForge registration, ChromaNeural integration, and local Janus
  inference have **not** been verified.

Source evidence: owner-provided JSON/text logs dated
`layer1-final-validation-20261017-201231` and
`layer1-thermal-20261017-201946`.
These logs are not embedded because they contain user-local system paths.
This is an experimental source release, **not** hardware certification.

## Installation (Ubuntu/Linux)

Require Node.js >=20, npm, Python with venv/pip support, and an existing
compatible NVIDIA driver for optional CUDA mode. Do not install a separate
NVIDIA Linux driver inside WSL2. Install npm packages using `npm ci`.
For optional GPU mode:

```sh
python3 scripts/setup_gpu.py
python3 scripts/setup_gpu.py --install
.venv/bin/python scripts/verify_cuda_worker.py
npm run start:cuda
```

The installer selects a compatible PyTorch CUDA wheel index for supported
NVIDIA devices. It installs only into the local Python venv. To put the venv in
a separate Linux-native directory, set `LAYER1_VENV_DIR` while installing
and set `LAYER1_PYTHON` to that environment's `bin/python` when running.
Linux-native ext4 performs better than the WSL `/mnt/e` mount for Python
dependency installation. PyTorch wheels download additional large GPU
libraries; ensure available disk space and authorize downloads explicitly.
The scripts do not install GPU drivers or system-wide toolkits.

The default CPU mode `npm start` does not need PyTorch. GPU defaults to
10 Hz and max 32 agents. This is workload limiting, not a guarantee against
CPU/GPU heat. CUDA failures do not silently fall back to CPU.

## MCP and security

`npm run mcp:stdio` launches a **read-only local adapter** which expects
a separate Layer 1+ server to be running on loopback. Its tools are
`layer1_status_v1` and `layer1_brain_trace_v1`; there are **no**
remote execution, write, training, or simulation-control tools.

RoboForge inclusion is a separate project change; see
`integrations/roboforge/README.md`. ChromaNeural is not modified.
Neither saved neural model checkpoints nor private per-studio model hosting
is implemented. There are no unsupported consciousness claims.

## Package policy

Linux source release only: git sources and GitHub-generated tagged source
archives. No preinstalled Python environment, proprietary NVIDIA driver,
model weights, Windows executable, or binary `.deb` is claimed for this
release. A future `.deb` requires a separately tested reproducible Linux
package. The original MIT license and author attribution remain unchanged.
