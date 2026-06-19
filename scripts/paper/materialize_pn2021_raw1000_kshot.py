#!/usr/bin/env python3
"""Materialize PN2021 K-shot ref-meta selections as raw1000 ECG artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ecg_adv_gen.data.pn2021_raw_kshot import (  # noqa: E402
    materialize_selected_raw1000_from_ref_meta,
    save_raw1000_kshot_npz,
)


DEFAULT_CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")


def build_output_base(subset_root: Path, center: str, k: int, seed: int) -> Path:
    return subset_root / center / f"k{k}_seed{seed}" / f"{center}_real_k{k}_seed{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pn2021_root", required=True)
    parser.add_argument("--subset_root", required=True)
    parser.add_argument("--centers", nargs="+", default=list(DEFAULT_CENTERS))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary_json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pn2021_root = Path(args.pn2021_root)
    subset_root = Path(args.subset_root)
    records: list[dict[str, object]] = []
    for seed in args.seeds:
        for center in args.centers:
            base = build_output_base(subset_root, center, int(args.k), int(seed))
            ref_meta = base.with_suffix(".ref_meta.json")
            out_path = Path(str(base) + ".raw1000.npz")
            if not ref_meta.exists():
                raise FileNotFoundError(ref_meta)
            if out_path.exists() and not args.force:
                records.append(
                    {
                        "center": center,
                        "seed": int(seed),
                        "path": str(out_path),
                        "status": "exists",
                    }
                )
                print(f"[skip] {out_path}", flush=True)
                continue
            materialized = materialize_selected_raw1000_from_ref_meta(
                center=center,
                ref_meta_json=ref_meta,
                pn2021_root=pn2021_root,
            )
            save_raw1000_kshot_npz(out_path, materialized)
            records.append(
                {
                    "center": center,
                    "seed": int(seed),
                    "path": str(out_path),
                    "status": "written",
                    "n": int(materialized.signals.shape[0]),
                    "mean": float(materialized.signals.mean()),
                    "std": float(materialized.signals.std()),
                }
            )
            print(f"[write] {out_path} n={materialized.signals.shape[0]}", flush=True)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
        print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
