# Layer 1+ v1.2.0-rc.1 development: CUDA and RoboForge MCP

Status: **experimental; not hardware verified**. Linux-only. No system CUDA driver
or toolkit is installed by this repository. This work is not a certified
performance, thermal, AI, or security release.

## Local GPU setup
Run `python3 scripts/setup_gpu.py` to inspect the available NVIDIA GPU via
`nvidia-smi`, and then `python3 scripts/setup_gpu.py --install` to explicitly
install the CUDA-enabled PyTorch wheel into the repository-local `.venv`.
The selected CUDA wheel channel is cu126 or cu128 based on reported NVIDIA driver capability and GPU generation; the installer validates actual CUDA
execution and stops on failure. It does not upgrade the host GPU driver. You
must install a compatible NVIDIA driver using your OS package management.
PyTorch CUDA wheel support and driver compatibility must be checked on the
target machine.

Run `npm ci`, then `LAYER1_NEURAL_BACKEND=cuda npm start` to enable
PyTorch CUDA **training**. Forward passes, action sampling, sensors, and world
physics remain in TypeScript on CPU. CPU use is therefore not eliminated.
Training work is performed asynchronously on the GPU worker using serialized
snapshots of each agent's weights, Adam moments and transitions. Unlike a
fully GPU-resident neural simulation this may not improve performance for
small populations. It must be profiled on real hardware; 8 GB VRAM alone is
not a guarantee of speed or acceptable temperatures.

CPU baseline: `npm start`. Failure of CUDA startup does **not** silently
activate CPU. Failure during training pauses the world; inspect the status
endpoint before any restart. Do not run long stress tests without temperature
monitoring. Pause, reset and model consistency require regression acceptance
before any production use.

## MCP integration boundary
Layer 1+ remains an independent simulation engine. RoboForge exposes a
curated local-only MCP gateway and should integrate this project through an
explicit, bounded, disabled-by-default adapter, rather than adding generic
remote code execution or port-forwarding. ChromaNeural then connects to
RoboForge as its existing optional MCP module. No direct ChromaNeural
auth/ledger/storage/Internet Identity changes are allowed.

Proposed read-only tools: world status, current agent stats, recent events.
Proposed controlled tools: pause, resume, reset, finite-time run, bounded
simulation test. Never enable a controlled action in ChromaNeural managed
mode until a reviewed grant verifier, CPU/RAM/GPU limits, and end-to-end tests
cover it; current RoboForge managed grants support only Refract compilation.

DeepSeek Janus is a **separate local runtime**. RoboForge currently provides
status only; Janus inference has NOT been integrated. Do not bundle weights,
download models silently or claim existing game asset generation.
Unity/Godot Engine Bridge is independent from GPU training and is unchanged.
