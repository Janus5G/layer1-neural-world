#!/usr/bin/env python3
"""Read-only diagnostic helper for local CUDA configuration."""
import json
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"scripts"))
from setup_gpu import detect
result={"hardware":detect(),"python":sys.version.split()[0],"torchInstalled":False,"cudaAvailable":False}
try:
    import torch
    result["torchInstalled"]=True
    result["torchVersion"]=torch.__version__
    result["cudaAvailable"]=torch.cuda.is_available()
    if result["cudaAvailable"]:
        result["device"]=torch.cuda.get_device_name(0)
except ImportError:pass
print(json.dumps(result,indent=2))
