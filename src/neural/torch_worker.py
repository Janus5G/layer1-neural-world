#!/usr/bin/env python3
"""Opt-in local PyTorch training worker. JSON lines on stdin/stdout; no network or shell.

The built-in TypeScript policy remains the reference implementation. This worker
moves its bounded A2C/GAE training updates onto the selected CUDA device.
Each request supplies an immutable snapshot; results are committed by Node only
when the request completes. This process never installs drivers or packages.
"""
from __future__ import annotations

import json
import math
import os
import sys

MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_JOBS = 32
MAX_TRANSITIONS = 256
LAYERS = ((40, 48), (48, 48), (48, 11), (48, 1))
FIELDS = ("W", "b", "mW", "vW", "mb", "vb")
torch = None


def respond(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def finite_number(x: object) -> bool:
    return type(x) in (float, int) and math.isfinite(x)


def verify_vector(x: object, n: int) -> None:
    if not isinstance(x, list) or len(x) != n or not all(finite_number(v) and abs(v) < 1e9 for v in x):
        raise ValueError("Invalid or nonfinite neural vector")


def verify_job(job: dict) -> None:
    if not isinstance(job, dict) or type(job.get("id")) is not int or job["id"] < 1:
        raise ValueError("Invalid agent ID")
    if type(job.get("adamT")) is not int or not 0 <= job["adamT"] < 100000000:
        raise ValueError("Invalid Adam step")
    layers = job.get("layers")
    if not isinstance(layers, list) or len(layers) != 4:
        raise ValueError("Invalid layer list")
    for obj, (input_dim, output_dim) in zip(layers, LAYERS):
        if not isinstance(obj, dict) or set(obj) != set(FIELDS):
            raise ValueError("Invalid layer state")
        for key in ("W", "mW", "vW"):
            verify_vector(obj[key], input_dim * output_dim)
        for key in ("b", "mb", "vb"):
            verify_vector(obj[key], output_dim)
    transitions = job.get("transitions")
    if not isinstance(transitions, list) or not 2 <= len(transitions) <= MAX_TRANSITIONS:
        raise ValueError("Invalid training sequence")
    for tr in transitions:
        if not isinstance(tr, dict) or set(tr) != {"state", "action", "value", "reward", "done"}:
            raise ValueError("Invalid transition fields")
        verify_vector(tr["state"], 40)
        if type(tr["action"]) is not int or not 0 <= tr["action"] < 11:
            raise ValueError("Invalid action")
        if type(tr["done"]) is not bool or not finite_number(tr["reward"]) or not finite_number(tr["value"]):
            raise ValueError("Invalid reward/value")
        if abs(tr["reward"]) > 1e5 or abs(tr["value"]) > 1e5:
            raise ValueError("Unbounded transition")


def train_job(job: dict, device, options: dict) -> dict:
    verify_job(job)
    gamma = options["gamma"]
    lam = options["lambda"]
    lr = options["learningRate"]
    value_coef = options["valueCoef"]
    entropy_coef = options["entropyCoef"]
    tr = job["transitions"]
    n = len(tr)
    advantages = [0.] * n
    returns = [0.] * n
    last_adv = 0.
    next_value = 0.
    for t in range(n - 1, -1, -1):
        x = tr[t]
        mask = 0. if x["done"] else 1.
        delta = x["reward"] + gamma * next_value * mask - x["value"]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
        returns[t] = x["value"] + last_adv
        next_value = x["value"]
    mean = sum(advantages) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in advantages) / n) + 1e-6
    advantages = [(v - mean) / std for v in advantages]

    params = []
    moments = []
    for raw, (d_in, d_out) in zip(job["layers"], LAYERS):
        w = torch.tensor(raw["W"], dtype=torch.float32, device=device).reshape(d_out, d_in).requires_grad_()
        b = torch.tensor(raw["b"], dtype=torch.float32, device=device).requires_grad_()
        params.extend((w, b))
        moments.extend((
            torch.tensor(raw["mW"], dtype=torch.float32, device=device).reshape(d_out, d_in),
            torch.tensor(raw["vW"], dtype=torch.float32, device=device).reshape(d_out, d_in),
            torch.tensor(raw["mb"], dtype=torch.float32, device=device),
            torch.tensor(raw["vb"], dtype=torch.float32, device=device)
        ))
    adam_t = job["adamT"] + 1
    bc1 = 1 - .9 ** adam_t
    bc2 = 1 - .999 ** adam_t
    policy_total = value_total = entropy_total = 0.
    for t, x in enumerate(tr):
        inp = torch.tensor(x["state"], dtype=torch.float32, device=device)
        h1 = torch.relu(torch.nn.functional.linear(inp, params[0], params[1]))
        h2 = torch.relu(torch.nn.functional.linear(h1, params[2], params[3]))
        logits = torch.nn.functional.linear(h2, params[4], params[5])
        value = torch.nn.functional.linear(h2, params[6], params[7]).squeeze(0)
        probs = torch.softmax(logits, dim=0)
        logp = torch.log(probs[x["action"]] + 1e-9)
        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        policy_loss = -logp * advantages[t] - entropy_coef * entropy
        value_loss = .5 * value_coef * (value - returns[t]).square()
        loss = policy_loss + value_loss
        loss.backward()
        with torch.no_grad():
            policy_total += float((-logp * advantages[t]).item())
            value_total += float(((returns[t] - value).square()).item())
            entropy_total += float(entropy.item())
            for j, p in enumerate(params):
                grad = p.grad
                m, v = moments[2*j], moments[2*j+1]
                m.mul_(.9).add_(grad, alpha=.1)
                v.mul_(.999).addcmul_(grad, grad, value=.001)
                p.addcdiv_(m / bc1, (v / bc2).sqrt().add(1e-8), value=-lr)
                p.grad = None

    result_layers = []
    for k in range(4):
        w, b = params[2*k], params[2*k+1]
        result_layers.append({
            "W": w.detach().cpu().flatten().tolist(),
            "b": b.detach().cpu().tolist(),
            "mW": moments[4*k].cpu().flatten().tolist(),
            "vW": moments[4*k+1].cpu().flatten().tolist(),
            "mb": moments[4*k+2].cpu().tolist(),
            "vb": moments[4*k+3].cpu().tolist(),
        })
    return {
        "id": job["id"], "layers": result_layers, "adamT": adam_t,
        "lastLoss": policy_total/n, "lastValueLoss": value_total/n,
        "lastEntropy": entropy_total/n,
    }


def dispatch(req: dict, device) -> dict:
    if not isinstance(req, dict) or type(req.get("id")) is not int or req["id"] < 1:
        raise ValueError("Invalid worker message")
    method = req.get("method")
    if method == "ping":
        return {"id": req["id"], "ok": True, "backend": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "torch": torch.__version__}
    if method != "train_batch":
        raise ValueError("Unknown worker method")
    jobs = req.get("jobs")
    opt = req.get("options")
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= MAX_JOBS or not isinstance(opt, dict):
        raise ValueError("Invalid batch")
    if set(opt) != {"gamma", "lambda", "learningRate", "valueCoef", "entropyCoef"}:
        raise ValueError("Invalid hyperparameters")
    if not all(finite_number(v) for v in opt.values()) or not 0 < opt["learningRate"] <= .01:
        raise ValueError("Invalid training options")
    if not (0 <= opt["gamma"] <= 1 and 0 <= opt["lambda"] <= 1 and
            0 <= opt["valueCoef"] <= 10 and 0 <= opt["entropyCoef"] <= 1):
        raise ValueError("Invalid training range")
    if len({j.get("id") for j in jobs if isinstance(j, dict)}) != len(jobs):
        raise ValueError("Duplicate agent ID")
    results = [train_job(j, device, opt) for j in jobs]
    return {"id": req["id"], "ok": True, "results": results, "backend": str(device)}


def main() -> None:
    global torch
    import torch as pytorch
    torch = pytorch
    # CUDA is an explicit opt-in: never claim GPU while running on CPU.
    if os.environ.get("LAYER1_WORKER_DEVICE") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; worker refuses CPU substitution")
    device = torch.device("cuda:0")
    for raw in sys.stdin.buffer:
        try:
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError("Worker request exceeds 8 MiB")
            req = json.loads(raw)
            respond(dispatch(req, device))
        except Exception as exc:
            request_id = req.get("id", 0) if isinstance(locals().get("req"), dict) else 0
            respond({"id": request_id, "ok": False, "error": str(exc)[:180]})
            req = None


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write("GPU worker failed: " + str(exc)[:240] + "\n")
        sys.exit(2)
