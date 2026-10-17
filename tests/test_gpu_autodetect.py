"""Offline NVIDIA CLI regression tests, including WSL CUDA UMD header."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
source=Path(__file__).resolve().parents[1]/"scripts/setup_gpu.py"
spec=importlib.util.spec_from_file_location("layer1_gpu_setup",source)
setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)

class GpuDetectionTests(unittest.TestCase):
    def test_wsl_cuda_umd(self):
        calls=[]
        def fake(cmd,**kw):
            calls.append(cmd)
            if "--query-gpu=name,compute_cap,memory.total" in cmd:
                return SimpleNamespace(stdout="NVIDIA GeForce RTX 5050 Laptop GPU, 12.0, 8151\n")
            return SimpleNamespace(stdout="NVIDIA-SMI 610.53    KMD Version: 610.74    CUDA UMD Version: 13.3\n")
        with patch.object(setup.shutil,"which",return_value="/usr/lib/wsl/lib/nvidia-smi"),patch.object(setup.subprocess,"run",side_effect=fake):
            info=setup.detect()
        self.assertEqual(info["driverCuda"],[13,3])
        self.assertEqual(info["gpus"][0]["compute"],[12,0])
        self.assertEqual(len(calls),2)
    def test_standard_cuda_header(self):
        def fake(cmd,**kw):
            return SimpleNamespace(stdout="GeForce RTX, 8.9, 8192\n" if "--query-gpu=name,compute_cap,memory.total" in cmd else "CUDA Version: 12.8")
        with patch.object(setup.shutil,"which",return_value="/usr/bin/nvidia-smi"),patch.object(setup.subprocess,"run",side_effect=fake):
            self.assertEqual(setup.detect()["driverCuda"],[12,8])
    def test_absent_gpu(self):
        with patch.object(setup.shutil,"which",return_value=None),patch.object(setup.Path,"is_file",return_value=False):
            self.assertFalse(setup.detect()["available"])

if __name__=="__main__": unittest.main()
