#!/usr/bin/env python3
"""Linux-only optional CUDA PyTorch bootstrap; never installs or changes system drivers."""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ENV=Path(os.environ.get("LAYER1_VENV_DIR", str(ROOT/".venv"))).expanduser()
def detect():
    executable = shutil.which("nvidia-smi")
    if executable is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        return {"available":False,"reason":"nvidia-smi not found (no driver installed by this script)"}
    try:
        result=subprocess.run([executable,"--query-gpu=name,compute_cap,memory.total","--format=csv,noheader,nounits"],
                              capture_output=True,text=True,timeout=8,check=True)
    except (OSError,subprocess.SubprocessError) as exc:
        return {"available":False,"reason":str(exc)[:200]}
    gpus=[]
    try:
        header=subprocess.run([executable],capture_output=True,text=True,timeout=8,check=True).stdout
        match=re.search(r"CUDA(?:\s+UMD)?\s+Version:\s*(\d+)\.(\d+)",header,flags=re.IGNORECASE)
        driver_cuda=tuple(map(int,match.groups())) if match else None
    except (OSError,subprocess.SubprocessError):driver_cuda=None
    for line in result.stdout.strip().splitlines():
        cols=[part.strip() for part in line.split(",")]
        if len(cols)<3:continue
        try:
            major,minor=(int(x) for x in cols[1].split("."))
            memory=int(cols[2])
            gpus.append({"name":cols[0],"compute":[major,minor],"vramMiB":memory})
        except ValueError:continue
    return {"available":bool(gpus),"gpus":gpus,"driverCuda":list(driver_cuda) if driver_cuda else None}

def main():
    if sys.platform!="linux":
        raise SystemExit("This installer supports Linux only.")
    info=detect()
    print(json.dumps(info,indent=2))
    if not info["available"]:
        raise SystemExit("No detected NVIDIA CUDA GPU. No packages installed; CPU baseline remains available.")
    if not any(tuple(g["compute"]) >= (7,5) for g in info["gpus"]):
        raise SystemExit("Unsupported CUDA compute capability; no changes made.")
    capability=max(tuple(g["compute"]) for g in info["gpus"])
    supported=tuple(info.get("driverCuda") or (0,0))
    if capability >= (12,0) and supported < (12,8):
        raise SystemExit("Blackwell GPU requires driver CUDA capability >= 12.8; update the driver manually.")
    if supported >= (12,8):channel="cu128"
    elif supported >= (12,6) and capability < (12,0):channel="cu126"
    else:raise SystemExit("No approved PyTorch CUDA wheel matches current driver; no changes made.")
    index="https://download.pytorch.org/whl/"+channel
    print("Planned isolated install: PyTorch from",index)
    if "--install" not in sys.argv:
        print("Dry run only. Run python3 scripts/setup_gpu.py --install to consent.")
        return
    if not ENV.is_absolute():
        raise SystemExit("LAYER1_VENV_DIR must be absolute; no changes made.")
    if not ENV.exists():venv.EnvBuilder(with_pip=True).create(ENV)
    python=str(ENV/"bin/python")
    subprocess.run([python,"-m","pip","install","--no-cache-dir","--index-url",index,"torch==2.10.0"],check=True)
    test=subprocess.run([python,"-c","import torch; assert torch.cuda.is_available(); x=torch.ones((8,8),device='cuda'); y=x@x; torch.cuda.synchronize(); assert float(y[0,0])==8; print(torch.cuda.get_device_name(0))"],capture_output=True,text=True)
    if test.returncode:
        raise SystemExit("CUDA verification FAILED. No CPU substitution. "+test.stderr[-300:])
    print("CUDA verification PASS:",test.stdout.strip())

if __name__=="__main__":main()
