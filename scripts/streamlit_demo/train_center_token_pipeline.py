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
    ap.add_argument("--center", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--prompt_bank", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--floor_per_class", type=int, default=10)
    ap.add_argument("--total_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    prepare_cmd = [
        sys.executable,
        "scripts/streamlit_demo/prepare_hospital_prompt_cache.py",
        "--dataset_npz",
        args.dataset_npz,
        "--center",
        args.center,
        "--out_root",
        args.cache_root,
        "--K",
        str(int(args.K)),
        "--floor_per_class",
        str(int(args.floor_per_class)),
        "--device",
        args.device,
    ]
    run_cmd(prepare_cmd)

    train_cmd = [
        sys.executable,
        "scripts/ecgtwin_gen/train_center_prompt_tokens.py",
        "--centers",
        args.center,
        "--cache_root",
        args.cache_root,
        "--prompt_bank",
        args.prompt_bank,
        "--save_dir",
        args.save_dir,
        "--K",
        str(int(args.K)),
        "--total_steps",
        str(int(args.total_steps)),
        "--batch_size",
        str(int(args.batch_size)),
        "--num_workers",
        str(int(args.num_workers)),
        "--device",
        args.device,
        "--amp_dtype",
        "bf16",
        "--token_mode",
        "direct",
        "--n_token_vectors",
        "4",
        "--sample_strategy",
        "center_class_balanced",
        "--log_every",
        "20",
    ]
    run_cmd(train_cmd)

    summary = {
        "center": args.center,
        "dataset_npz": args.dataset_npz,
        "cache_root": args.cache_root,
        "save_dir": args.save_dir,
        "K": int(args.K),
        "floor_per_class": int(args.floor_per_class),
        "total_steps": int(args.total_steps),
        "batch_size": int(args.batch_size),
    }
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "streamlit_center_token_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("[done] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
