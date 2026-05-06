#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_TRT_VENDOR = "/root/autodl-tmp/streamlit_ecg_demo/python_pkgs/tensorrt_cu12"


def add_vendor_path(vendor_path: str | None) -> None:
    if not vendor_path:
        return
    path = Path(vendor_path)
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def import_tensorrt():
    try:
        import tensorrt as trt
    except ModuleNotFoundError:
        import tensorrt_bindings as trt
    return trt


def build_with_python_tensorrt(args: argparse.Namespace) -> dict:
    add_vendor_path(args.vendor_path)
    trt = import_tensorrt()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"missing ONNX model: {onnx_path}")
    parsed = parser.parse(onnx_path.read_bytes())
    if not parsed:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(args.workspace_mib) * 1024 * 1024,
        )
    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    profile.set_shape(
        input_name,
        (args.min_batch, 12, 1000),
        (args.opt_batch, 12, 1000),
        (args.max_batch, 12, 1000),
    )
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    out = Path(args.engine)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(serialized))
    return {
        "status": "ok",
        "builder": "python_tensorrt",
        "tensorrt_version": trt.__version__,
        "input_name": input_name,
        "min_shape": [args.min_batch, 12, 1000],
        "opt_shape": [args.opt_batch, 12, 1000],
        "max_shape": [args.max_batch, 12, 1000],
        "fp16_requested": bool(args.fp16),
        "fp16_enabled": bool(args.fp16 and builder.platform_has_fast_fp16),
        "workspace_mib": int(args.workspace_mib),
        "build_seconds": build_s,
    }


def build_with_trtexec(args: argparse.Namespace, trtexec: str) -> dict:
    cmd = [
        trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        f"--minShapes=ecg:{args.min_batch}x12x1000",
        f"--optShapes=ecg:{args.opt_batch}x12x1000",
        f"--maxShapes=ecg:{args.max_batch}x12x1000",
        f"--memPoolSize=workspace:{args.workspace_mib}",
    ]
    if args.fp16:
        cmd.append("--fp16")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = Path(args.engine)
    return {
        "status": "ok" if proc.returncode == 0 and out.exists() else "failed",
        "builder": "trtexec",
        "returncode": proc.returncode,
        "command": cmd,
        "log_tail": proc.stdout[-4000:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/root/autodl-tmp/streamlit_ecg_demo/models/efficientnetv2_super5.onnx")
    ap.add_argument("--engine", default="/root/autodl-tmp/streamlit_ecg_demo/models/efficientnetv2_super5_fp16.engine")
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false")
    ap.add_argument("--backend", choices=["auto", "trtexec", "python"], default="auto")
    ap.add_argument("--vendor_path", default=DEFAULT_TRT_VENDOR)
    ap.add_argument("--workspace_mib", type=int, default=2048)
    ap.add_argument("--min_batch", type=int, default=1)
    ap.add_argument("--opt_batch", type=int, default=8)
    ap.add_argument("--max_batch", type=int, default=32)
    args = ap.parse_args()

    report = {
        "onnx": args.onnx,
        "engine": args.engine,
        "fp16": bool(args.fp16),
        "backend": args.backend,
        "vendor_path": args.vendor_path,
        "status": "not_started",
    }
    out = Path(args.engine)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        trtexec = shutil.which("trtexec")
        if args.backend in {"auto", "trtexec"} and trtexec:
            report.update(build_with_trtexec(args, trtexec))
        elif args.backend == "trtexec":
            report.update({"status": "unavailable", "reason": "trtexec not found"})
        else:
            report.update(build_with_python_tensorrt(args))
    except Exception as exc:
        report.update({"status": "failed", "reason": str(exc)})

    report_path = out.with_suffix(".build.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
