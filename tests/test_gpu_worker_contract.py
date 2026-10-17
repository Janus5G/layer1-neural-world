"""Offline validation of GPU worker wire contract; does not claim GPU execution."""
import importlib.util
import pathlib
import unittest

PATH=pathlib.Path(__file__).resolve().parents[1]/"src/neural/torch_worker.py"
spec=importlib.util.spec_from_file_location("layer1_torch_worker",PATH)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def valid():
    result=[]
    for n_in,n_out in module.LAYERS:
        result.append({
            "W":[0.]* (n_in*n_out),"b":[0.]*n_out,
            "mW":[0.]*(n_in*n_out),"vW":[0.]*(n_in*n_out),
            "mb":[0.]*n_out,"vb":[0.]*n_out
        })
    return {"id":1,"adamT":0,"layers":result,
            "transitions":[{"state":[0.]*40,"action":0,"value":0.,"reward":0.,"done":False},
                           {"state":[0.]*40,"action":10,"value":0.,"reward":1.,"done":True}]}

class ContractTests(unittest.TestCase):
    def test_reference_dimensions(self):
        self.assertEqual(module.LAYERS,((40,48),(48,48),(48,11),(48,1)))
        module.verify_job(valid())

    def test_wrong_input_dimension_rejected(self):
        task=valid()
        task["transitions"][0]["state"]=[0.]*39
        with self.assertRaises(ValueError):module.verify_job(task)

    def test_out_of_range_action_rejected(self):
        task=valid()
        task["transitions"][1]["action"]=11
        with self.assertRaises(ValueError):module.verify_job(task)

    def test_nonfinite_gradients_rejected(self):
        task=valid()
        task["layers"][0]["mW"][1]=float("nan")
        with self.assertRaises(ValueError):module.verify_job(task)

    def test_refuse_unbounded_training_batch(self):
        task=valid()
        task["transitions"]=task["transitions"]*129
        with self.assertRaises(ValueError):module.verify_job(task)

if __name__=="__main__":unittest.main()
