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
    DEFAULT_TRT_ENGINE,
    ONNXRuntimeClassifierBackend,
    PyTorchClassifierBackend,
    TensorRTClassifierBackend,
)
from apps.streamlit_ecg_demo.services.paths import APP_DATA_ROOT


def latency_summary(latencies_ms: list[float], batch_size: int) -> dict:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    total_s = float(arr.sum() / 1000.0)
    return {
        "batch_size": batch_size,
        "repeats": int(arr.size),
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "throughput_ecg_per_s": float(batch_size * arr.size / total_s),
    }


def bench_pytorch(ckpt: str, device: str, batch_size: int, repeats: int) -> dict:
    backend = PyTorchClassifierBackend(ckpt_path=ckpt, device=device)
    x = np.random.default_rng(42).normal(size=(batch_size, 12, 1000)).astype(np.float32)
    x_t = torch.from_numpy(x).to(backend.device)
    with torch.no_grad():
        for _ in range(5):
            _ = backend.model(x_t)
        if backend.device.type == "cuda":
            torch.cuda.synchronize()
        latencies = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = backend.model(x_t)
            if backend.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    result = {
        "backend": "pytorch",
        "device": str(backend.device),
    }
    result.update(latency_summary(latencies, batch_size))
    return result


def bench_onnx(onnx_path: str, batch_size: int, repeats: int) -> dict:
    backend = ONNXRuntimeClassifierBackend(onnx_path=onnx_path)
    x = np.random.default_rng(42).normal(size=(batch_size, 12, 1000)).astype(np.float32)
    for _ in range(5):
        _ = backend.session.run(None, {backend.input_name: x})
    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = backend.session.run(None, {backend.input_name: x})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    result = {
        "backend": "onnxruntime",
        "device": backend.provider,
    }
    result.update(latency_summary(latencies, batch_size))
    return result


def bench_tensorrt(engine_path: str, batch_size: int, repeats: int) -> dict:
    backend = TensorRTClassifierBackend(engine_path=engine_path)
    x = np.random.default_rng(42).normal(size=(batch_size, 12, 1000)).astype(np.float32)
    for _ in range(5):
        _ = backend.predict_batch_array(x)
    latencies = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = backend.predict_batch_array(x)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    result = {
        "backend": "tensorrt",
        "device": "cuda",
        "engine": engine_path,
    }
    result.update(latency_summary(latencies, batch_size))
    return result


def validate_tensorrt(ckpt: str, engine_path: str, batch_size: int) -> dict:
    pt = PyTorchClassifierBackend(ckpt_path=ckpt, device="cuda")
    trt_backend = TensorRTClassifierBackend(engine_path=engine_path)
    x = np.random.default_rng(123).normal(size=(batch_size, 12, 1000)).astype(np.float32)
    x_t = torch.from_numpy(x).to(pt.device)
    with torch.no_grad():
        logits_pt = pt.model(x_t).detach().cpu().numpy()
    logits_trt = trt_backend.predict_batch_array(x)
    probs_pt = 1.0 / (1.0 + np.exp(-np.clip(logits_pt, -50, 50)))
    probs_trt = 1.0 / (1.0 + np.exp(-np.clip(logits_trt, -50, 50)))
    return {
        "batch_size": batch_size,
        "max_abs_logit_diff": float(np.max(np.abs(logits_pt - logits_trt))),
        "mean_abs_logit_diff": float(np.mean(np.abs(logits_pt - logits_trt))),
        "max_abs_prob_diff": float(np.max(np.abs(probs_pt - probs_trt))),
        "mean_abs_prob_diff": float(np.mean(np.abs(probs_pt - probs_trt))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--engine", default=DEFAULT_TRT_ENGINE)
    ap.add_argument("--out", default=str(APP_DATA_ROOT / "reports/inference_benchmark.json"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = [bench_pytorch(args.ckpt, args.device, b, args.repeats) for b in args.batch_sizes]
    if Path(args.onnx).exists():
        results.extend(bench_onnx(args.onnx, b, args.repeats) for b in args.batch_sizes)
    validation = None
    if Path(args.engine).exists():
        validation = validate_tensorrt(args.ckpt, args.engine, batch_size=min(max(args.batch_sizes), 8))
        validation_path = out.with_name("tensorrt_validation.json")
        validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        results.extend(bench_tensorrt(args.engine, b, args.repeats) for b in args.batch_sizes)
    report = {
        "checkpoint": args.ckpt,
        "onnx": args.onnx,
        "engine": args.engine,
        "results": results,
        "tensorrt_validation": validation,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
