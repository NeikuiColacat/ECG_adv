#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run_cmd(cmd: list[str]) -> None:
    print("[cmd] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--synth_npz", required=True)
    ap.add_argument("--init_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--synth_ratio", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_tensorrt", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_out = out_dir / "best_model.pt"
    onnx_out = out_dir / "efficientnetv2_super5.onnx"
    engine_out = out_dir / "efficientnetv2_super5_fp16.engine"

    train_cmd = [
        sys.executable,
        "scripts/streamlit_demo/train_hospital_classifier.py",
        "--dataset_npz",
        args.dataset_npz,
        "--output_dir",
        str(out_dir),
        "--synth_npz",
        args.synth_npz,
        "--init_ckpt",
        args.init_ckpt,
        "--epochs",
        str(int(args.epochs)),
        "--batch_size",
        str(int(args.batch_size)),
        "--synth_ratio",
        str(float(args.synth_ratio)),
        "--lr",
        str(float(args.lr)),
        "--device",
        args.device,
    ]
    run_cmd(train_cmd)

    export_cmd = [
        sys.executable,
        "scripts/deploy/export_efficientnetv2_onnx.py",
        "--ckpt",
        str(pt_out),
        "--out",
        str(onnx_out),
        "--device",
        args.device,
    ]
    run_cmd(export_cmd)

    trt_report_path = engine_out.with_suffix(".build.json")
    trt_report = {
        "status": "skipped",
        "reason": "--skip_tensorrt was set",
        "engine": str(engine_out),
    }
    if not args.skip_tensorrt:
        trt_cmd = [
            sys.executable,
            "scripts/deploy/build_tensorrt_engine.py",
            "--onnx",
            str(onnx_out),
            "--engine",
            str(engine_out),
            "--backend",
            "auto",
            "--fp16",
        ]
        run_cmd(trt_cmd)
        if trt_report_path.exists():
            trt_report = json.loads(trt_report_path.read_text(encoding="utf-8"))
        else:
            trt_report = {
                "status": "missing_report",
                "engine": str(engine_out),
            }

    summary = {
        "dataset_npz": args.dataset_npz,
        "synth_npz": args.synth_npz,
        "init_ckpt": args.init_ckpt,
        "epochs": int(args.epochs),
        "pt": str(pt_out),
        "onnx": str(onnx_out),
        "tensorrt_engine": str(engine_out) if engine_out.exists() else "",
        "tensorrt_build": trt_report,
    }
    (out_dir / "streamlit_train_export_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("[done] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
