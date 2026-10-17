#!/usr/bin/env python3
"""Linux-only optional CUDA PyTorch bootstrap; never installs or changes system drivers."""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import venv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/".venv"
def detect():
    try:
        result=subprocess.run(["nvidia-smi","--query-gpu=name,compute_cap,memory.total","--format=csv,noheader,nounits"],
                              capture_output=True,text=True,timeout=8,check=True)
    except (OSError,subprocess.SubprocessError) as exc:
        return {"available":False,"reason":str(exc)[:200]}
    gpus=[]
    for line in result.stdout.strip().splitlines():
        cols=[part.strip() for part in line.split(",")]
        if len(cols)<3:continue
        try:
            major,minor=(int(x) for x in cols[1].split("."))
            memory=int(cols[2])
            gpus.append({"name":cols[0],"compute":[major,minor],"vramMiB":memory})
        except ValueError:continue
    return {"available":bool(gpus),"gpus":gpus}

def main():
    if sys.platform!="linux":
        raise SystemExit("This installer supports Linux only.")
    info=detect()
    print(json.dumps(info,indent=2))
    if not info["available"]:
        raise SystemExit("No detected NVIDIA CUDA GPU. No packages installed; CPU baseline remains available.")
    if not any(tuple(g["compute"]) >= (7,0) for g in info["gpus"]):
        raise SystemExit("Unsupported CUDA compute capability; no changes made.")
    # Compatible tested wheels must be explicitly chosen and audited; using this
    # index asks PyTorch for the corresponding CUDA-enabled distribution.
    index="https://download.pytorch.org/whl/cu128"
    print("Planned isolated install: PyTorch from",index)
    if "--install" not in sys.argv:
        print("Dry run only. Run python3 scripts/setup_gpu.py --install to consent.")
        return
    if not ENV.exists():venv.EnvBuilder(with_pip=True).create(ENV)
    python=str(ENV/"bin/python")
    subprocess.run([python,"-m","pip","install","--index-url",index,"torch"],check=True)
    test=subprocess.run([python,"-c","import torch; assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))"],capture_output=True,text=True)
    if test.returncode:
        raise SystemExit("CUDA verification FAILED. No CPU substitution. "+test.stderr[-300:])
    print("CUDA verification PASS:",test.stdout.strip())

if __name__=="__main__":main()
