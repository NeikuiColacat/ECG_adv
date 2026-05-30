#!/usr/bin/env python
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def _package_version(distribution: str, module: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"version": None, "importable": False, "error": None}
    try:
        record["version"] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        record["error"] = "distribution not installed"
    if module:
        try:
            imported = importlib.import_module(module)
            record["importable"] = True
            record["module_version"] = getattr(imported, "__version__", None)
        except Exception as exc:  # pragma: no cover - depends on host CUDA/TensorRT libs
            record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _torch_snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    torch_record = _package_version("torch", "torch")
    cuda: dict[str, Any] = {
        "available": False,
        "torch_cuda_version": None,
        "device_count": 0,
        "device_names": [],
    }
    if torch_record.get("importable"):
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        cuda["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        cuda["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        cuda["device_names"] = [
            torch.cuda.get_device_name(i) for i in range(int(cuda["device_count"]))
        ]
    return torch_record, cuda


def collect_environment_snapshot() -> dict[str, Any]:
    torch_record, cuda = _torch_snapshot()
    packages = {
        "torch": torch_record,
        "onnx": _package_version("onnx", "onnx"),
        "onnxruntime-gpu": _package_version("onnxruntime-gpu", "onnxruntime"),
        "tensorrt": _package_version("tensorrt-cu12", "tensorrt"),
        "streamlit": _package_version("streamlit", "streamlit"),
        "numpy": _package_version("numpy", "numpy"),
        "scipy": _package_version("scipy", "scipy"),
    }
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "cuda": cuda,
        "packages": packages,
        "paths": {
            "cwd": str(Path.cwd()),
            "ECG_ADV_DATA_ROOT": os.environ.get("ECG_ADV_DATA_ROOT"),
            "ECG_ADV_GRAD_ROOT": os.environ.get("ECG_ADV_GRAD_ROOT"),
            "ECG_ADV_APP_DATA_ROOT": os.environ.get("ECG_ADV_APP_DATA_ROOT"),
            "ECG_ADV_FINAL_ROUND_ROOT": os.environ.get("ECG_ADV_FINAL_ROUND_ROOT"),
        },
    }


def main() -> None:
    snapshot = collect_environment_snapshot()
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
