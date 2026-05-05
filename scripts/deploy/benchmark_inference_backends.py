#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from apps.streamlit_ecg_demo.services.classifier_backend import (
    DEFAULT_CKPT,
    DEFAULT_ONNX,
    ONNXRuntimeClassifierBackend,
    PyTorchClassifierBackend,
)


def bench_pytorch(ckpt: str, device: str, batch_size: int, repeats: int) -> dict:
    backend = PyTorchClassifierBackend(ckpt_path=ckpt, device=device)
    x = np.random.default_rng(42).normal(size=(batch_size, 12, 1000)).astype(np.float32)
    x_t = torch.from_numpy(x).to(backend.device)
    with torch.no_grad():
        for _ in range(5):
            _ = backend.model(x_t)
        if backend.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = backend.model(x_t)
        if backend.device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "backend": "pytorch",
        "device": str(backend.device),
        "batch_size": batch_size,
        "repeats": repeats,
        "latency_ms_per_batch": elapsed * 1000.0 / repeats,
        "throughput_ecg_per_s": batch_size * repeats / elapsed,
    }


def bench_onnx(onnx_path: str, batch_size: int, repeats: int) -> dict:
    backend = ONNXRuntimeClassifierBackend(onnx_path=onnx_path)
    x = np.random.default_rng(42).normal(size=(batch_size, 12, 1000)).astype(np.float32)
    for _ in range(5):
        _ = backend.session.run(None, {backend.input_name: x})
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = backend.session.run(None, {backend.input_name: x})
    elapsed = time.perf_counter() - t0
    return {
        "backend": "onnxruntime",
        "device": backend.provider,
        "batch_size": batch_size,
        "repeats": repeats,
        "latency_ms_per_batch": elapsed * 1000.0 / repeats,
        "throughput_ecg_per_s": batch_size * repeats / elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--out", default="/root/autodl-tmp/streamlit_ecg_demo/reports/inference_benchmark.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = [bench_pytorch(args.ckpt, args.device, b, args.repeats) for b in args.batch_sizes]
    if Path(args.onnx).exists():
        results.extend(bench_onnx(args.onnx, b, args.repeats) for b in args.batch_sizes)
    report = {
        "checkpoint": args.ckpt,
        "onnx": args.onnx,
        "results": results,
        "note": "TensorRT engine still requires trtexec; ONNXRuntime is the current deployable fallback.",
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
