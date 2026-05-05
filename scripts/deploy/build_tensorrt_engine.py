#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/root/autodl-tmp/streamlit_ecg_demo/models/efficientnetv2_super5.onnx")
    ap.add_argument("--engine", default="/root/autodl-tmp/streamlit_ecg_demo/models/efficientnetv2_super5_fp16.engine")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--min_batch", type=int, default=1)
    ap.add_argument("--opt_batch", type=int, default=8)
    ap.add_argument("--max_batch", type=int, default=32)
    args = ap.parse_args()

    report = {
        "onnx": args.onnx,
        "engine": args.engine,
        "fp16": bool(args.fp16),
        "status": "not_started",
    }
    out = Path(args.engine)
    out.parent.mkdir(parents=True, exist_ok=True)
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        report.update({"status": "unavailable", "reason": "trtexec not found"})
        print(json.dumps(report, indent=2))
        return

    cmd = [
        trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        "--minShapes=ecg:%d x12x1000" % args.min_batch,
        "--optShapes=ecg:%d x12x1000" % args.opt_batch,
        "--maxShapes=ecg:%d x12x1000" % args.max_batch,
    ]
    cmd = [x.replace(" ", "") for x in cmd]
    if args.fp16:
        cmd.append("--fp16")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report.update({
        "status": "ok" if proc.returncode == 0 and out.exists() else "failed",
        "returncode": proc.returncode,
        "log_tail": proc.stdout[-4000:],
    })
    report_path = out.with_suffix(".build.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

