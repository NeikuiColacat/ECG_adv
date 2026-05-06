#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


def run_cmd(cmd: list[str], *, allow_fail: bool = False) -> int:
    print("[cmd] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO))
    if proc.returncode and not allow_fail:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return int(proc.returncode)


def load_gated(path: Path) -> dict | None:
    npz = path / "gated_samples.npz"
    if not npz.exists():
        return None
    with np.load(npz, allow_pickle=True) as data:
        return {
            "signals": data["signals"].astype(np.float32),
            "raw_signal_ct": data["raw_signal_ct"].astype(np.float32),
            "latents": data["latents"].astype(np.float32),
            "labels": data["labels"].astype(np.float32),
            "class_names": [str(x) for x in data["class_names"]],
            "source_indices": data["source_indices"].astype(np.int64)
            if "source_indices" in data.files
            else np.arange(data["labels"].shape[0], dtype=np.int64),
        }


def class_from_label(row: np.ndarray, class_names: list[str]) -> str:
    return class_names[int(np.argmax(row))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True)
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--prompt_bank", required=True)
    ap.add_argument("--token_bank", required=True)
    ap.add_argument("--token_center", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_per_class", type=int, required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--victim_ckpt", required=True)
    ap.add_argument("--token_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rounds", type=int, default=5)
    ap.add_argument("--oversample_factor", type=float, default=3.0)
    ap.add_argument("--min_candidates_per_class", type=int, default=8)
    ap.add_argument("--min_target_prob", type=float, default=0.30)
    args = ap.parse_args()

    if args.target_per_class < 1:
        raise ValueError("--target_per_class must be >= 1")
    if args.max_rounds < 1:
        raise ValueError("--max_rounds must be >= 1")

    out_root = Path(args.out_dir)
    center_root = out_root / args.center
    final_dir = center_root / "gated"
    rounds_root = out_root / "rounds"
    final_dir.mkdir(parents=True, exist_ok=True)
    rounds_root.mkdir(parents=True, exist_ok=True)

    target = int(args.target_per_class)
    keep: dict[str, list[dict]] = {cls: [] for cls in args.classes}
    round_summaries = []
    t0 = time.time()

    for round_idx in range(1, args.max_rounds + 1):
        remaining = {cls: target - len(items) for cls, items in keep.items()}
        remaining = {cls: n for cls, n in remaining.items() if n > 0}
        if not remaining:
            break

        candidate_n = max(
            int(args.min_candidates_per_class),
            int(math.ceil(max(remaining.values()) * float(args.oversample_factor))),
        )
        round_root = rounds_root / f"round_{round_idx:02d}"
        round_seed = int(args.seed + round_idx * 10000)
        gen_cmd = [
            sys.executable,
            "scripts/ecgtwin_gen/generate_center_prompt_token_synth.py",
            "--center",
            args.center,
            "--classes",
            *remaining.keys(),
            "--cache_root",
            args.cache_root,
            "--prompt_bank",
            args.prompt_bank,
            "--token_bank",
            args.token_bank,
            "--token_center",
            args.token_center,
            "--out_dir",
            str(round_root),
            "--n_per_class",
            str(candidate_n),
            "--steps",
            str(args.steps),
            "--seed",
            str(round_seed),
            "--device",
            args.device,
            "--victim_ckpt",
            args.victim_ckpt,
            "--token_scale",
            str(args.token_scale),
        ]
        run_cmd(gen_cmd)

        raw_dir = round_root / args.center
        gate_dir = raw_dir / "gated"
        gate_cmd = [
            sys.executable,
            "scripts/ecgtwin_gen/gate_prompt_token_synth.py",
            "--input_dir",
            str(raw_dir),
            "--out_dir",
            str(gate_dir),
            "--cache_root",
            args.cache_root,
            "--classes",
            *remaining.keys(),
            "--min_target_prob",
            str(args.min_target_prob),
        ]
        gate_code = run_cmd(gate_cmd, allow_fail=True)

        added = defaultdict(int)
        gated = load_gated(gate_dir)
        if gated is not None:
            class_names = gated["class_names"]
            for i, label in enumerate(gated["labels"]):
                cls = class_from_label(label, class_names)
                if cls not in keep or len(keep[cls]) >= target:
                    continue
                keep[cls].append({
                    "signals": gated["signals"][i],
                    "raw_signal_ct": gated["raw_signal_ct"][i],
                    "latents": gated["latents"][i],
                    "labels": gated["labels"][i],
                    "source_round": round_idx,
                    "source_index": int(gated["source_indices"][i]),
                })
                added[cls] += 1

        round_summary = {
            "round": round_idx,
            "remaining_before": remaining,
            "candidate_n_per_class": candidate_n,
            "seed": round_seed,
            "gate_returncode": gate_code,
            "added": dict(added),
            "kept_counts": {cls: len(items) for cls, items in keep.items()},
        }
        round_summaries.append(round_summary)
        print("[round] " + json.dumps(round_summary, ensure_ascii=False), flush=True)

    counts = {cls: len(items) for cls, items in keep.items()}
    missing = {cls: target - n for cls, n in counts.items() if n < target}
    if missing:
        partial = {
            "status": "failed_insufficient_gated_samples",
            "target_per_class": target,
            "counts": counts,
            "missing": missing,
            "rounds": round_summaries,
        }
        (center_root / "generation_partial_summary.json").write_text(
            json.dumps(partial, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise RuntimeError(f"could not reach requested gated count per class: missing={missing}")

    rows = [item for cls in args.classes for item in keep[cls][:target]]
    signals = np.stack([row["signals"] for row in rows]).astype(np.float32)
    raw_signal_ct = np.stack([row["raw_signal_ct"] for row in rows]).astype(np.float32)
    latents = np.stack([row["latents"] for row in rows]).astype(np.float32)
    labels = np.stack([row["labels"] for row in rows]).astype(np.float32)
    source_round = np.asarray([row["source_round"] for row in rows], dtype=np.int64)
    source_indices = np.asarray([row["source_index"] for row in rows], dtype=np.int64)

    np.savez_compressed(
        final_dir / "gated_samples.npz",
        signals=signals,
        raw_signal_ct=raw_signal_ct,
        latents=latents,
        labels=labels,
        center_name=args.center,
        class_names=np.asarray(CLASS_NAMES_SUPER5),
        source_round=source_round,
        source_indices=source_indices,
    )
    summary = {
        "status": "ok",
        "target_per_class": target,
        "classes": list(args.classes),
        "counts": counts,
        "n_samples": int(signals.shape[0]),
        "npz": str(final_dir / "gated_samples.npz"),
        "elapsed_sec": round(time.time() - t0, 2),
        "rounds": round_summaries,
    }
    (center_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
