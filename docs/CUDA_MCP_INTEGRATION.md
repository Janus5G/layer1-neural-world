# Layer 1+ v1.2.0: CUDA and RoboForge MCP

Status: **experimental; CUDA verified on one RTX 5050 laptop; performance and CPU thermal safety not established**. Linux-only. No system CUDA driver
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

Run `npm ci`, then `npm run start:cuda` to enable PyTorch CUDA
**batched neural forward inference and training**. The CUDA worker caches
agent networks on the GPU for forward inference and accepts bounded
state/optimizer/transition snapshots for training. Action sampling, sensors,
and world physics remain in TypeScript on CPU. GPU mode uses one pre-action
world snapshot per tick rather than sequential sensing, so behavior may differ
from the unchanged CPU baseline. The transfer overhead for tiny models may
outweigh the benefits of GPU parallelism; on the owner's RTX 5050, a 20-minute run reached 64 °C maximum GPU temperature and 138 MiB peak VRAM; simulation progressed 678.1s in 1200s (0.565× speed). No acceleration against the CPU baseline has been shown. See [release evidence](RELEASE_V1.2.0.md).

In CUDA mode the conservative defaults are 10 simulation ticks/second and
maximum 32 agents. You may configure `LAYER1_GPU_TICK_HZ` from 1 to 20;
increasing it requires separate heat and stability measurements. The limits
reduce work, but they cannot guarantee safe temperatures.

After installation, run the **real GPU smoke test**:
```sh
.venv/bin/python scripts/verify_cuda_worker.py
```
It checks CUDA inference and training on the selected NVIDIA GPU; the owner's separate 20-minute GPU run also passed. Neither proves CPU thermal safety. Record GPU/CPU utilization and
temperatures separately before extending run times.

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
