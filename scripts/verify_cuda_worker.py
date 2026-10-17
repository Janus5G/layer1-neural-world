#!/usr/bin/env python3
"""CUDA acceptance smoke test. Requires real NVIDIA GPU and installed torch CUDA.

Run with .venv/bin/python scripts/verify_cuda_worker.py
No network access, driver installation, model downloads or world mutations.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src/neural/torch_worker.py"
LAYERS = ((40, 48), (48, 48), (48, 11), (48, 1))

def layer(i,o):
    return {"W":[0.0]*(i*o),"b":[0.0]*o,"mW":[0.0]*(i*o),"vW":[0.0]*(i*o),"mb":[0.0]*o,"vb":[0.0]*o}

def trans(action,reward,done):
    return {"state":[0.0]*40,"action":action,"value":0.0,"reward":reward,"done":done}

def main():
    layers=[layer(*dims) for dims in LAYERS]
    infer_layers=[{"W":x["W"],"b":x["b"]} for x in layers]
    commands=[
        {"id":1,"method":"ping"},
        {"id":2,"method":"infer_batch","activeIds":[1],"jobs":[{"id":1,"state":[0.0]*40,"layers":infer_layers}]},
        {"id":3,"method":"train_batch","options":{"gamma":0.99,"lambda":0.95,"learningRate":0.0003,"valueCoef":0.5,"entropyCoef":0.04},
         "jobs":[{"id":1,"adamT":0,"layers":layers,"transitions":[trans(0,1.0,False),trans(1,0.0,True)]}]},
        {"id":4,"method":"infer_batch","jobs":[{"id":1,"state":[0.0]*40}]}
    ]
    proc=subprocess.run([sys.executable,"-u",str(WORKER)],cwd=ROOT,
         input="\n".join(json.dumps(c,separators=(",",":")) for c in commands)+"\n",
         text=True,capture_output=True,timeout=180,
         env={**__import__("os").environ,"LAYER1_WORKER_DEVICE":"cuda"})
    if proc.returncode!=0:
        raise SystemExit("FAIL: CUDA worker unavailable: "+proc.stderr[-500:])
    replies=[json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(replies)!=4 or any(not r.get("ok") or r.get("id")!=i+1 for i,r in enumerate(replies)):
        raise SystemExit("FAIL: CUDA worker command sequence")
    if replies[0]["backend"]!="cuda:0":
        raise SystemExit("FAIL: Not a CUDA device")
    before=replies[1]["results"][0]["probs"]
    after=replies[3]["results"][0]["probs"]
    trained=replies[2]["results"][0]
    if len(before)!=11 or len(after)!=11 or trained["adamT"]!=1 or before==after:
        raise SystemExit("FAIL: Training did not change CUDA predictions")
    print("PASS: CUDA worker device:",replies[0]["gpu"])
    print("PASS: batched inference on CUDA")
    print("PASS: A2C/Adam training update on CUDA")
    print("PASS: GPU cached weights updated after training")
    print("NOTE: this is a functional smoke test, not a performance/thermal soak test.")

if __name__=="__main__":
    main()
