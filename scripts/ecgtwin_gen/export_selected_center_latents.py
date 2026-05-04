"""Export selected PN2021 center ECGTwin latents for source-aware AT."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--center", required=True)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--selection_seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--mmap_root", default="",
                    help="Optional PN2021 mmap eval cache root; when set, also export selected real signals.")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    center_pt = cache_root / "center_full_latents" / f"{args.center}.pt"
    selection_json = cache_root / "ref_selection" / f"{args.center}_k{args.K}_seed{args.selection_seed}.json"
    cache = torch.load(center_pt, map_location="cpu", weights_only=False)
    with open(selection_json) as f:
        selection = json.load(f)
    selected = np.asarray(selection["selected_indices_in_full_cache"], dtype=np.int64)
    if selected.size == 0:
        raise ValueError("empty selected index list")

    latents = cache["latents"][selected].detach().cpu().numpy().astype(np.float32)
    labels = cache["super5_multi_hot"][selected].detach().cpu().numpy().astype(np.float32)
    primary = [str(cache["primary_class"][int(i)]) for i in selected]
    record_ids = [str(cache["record_ids"][int(i)]) for i in selected]
    if labels.shape[1] != len(CLASS_NAMES_SUPER5):
        raise ValueError(f"expected {len(CLASS_NAMES_SUPER5)} labels, got {labels.shape}")
    if not np.isfinite(latents).all():
        raise ValueError("non-finite latents")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.center}_real_k{args.K}_seed{args.selection_seed}"
    latent_path = out_dir / f"{tag}.latent.npz"
    trust_path = out_dir / f"{tag}.class_trust.json"
    meta_path = out_dir / f"{tag}.ref_meta.json"
    report_path = out_dir / f"{tag}.report.json"
    signal_path = out_dir / f"{tag}.signals.npz"

    np.savez_compressed(
        latent_path,
        latents=latents,
        labels=labels,
        center_name=np.asarray(args.center),
        class_names=np.asarray(CLASS_NAMES_SUPER5),
        source_indices=selected,
        record_ids=np.asarray(record_ids),
        primary_class=np.asarray(primary),
    )
    class_counts = {
        cls: int((labels[:, j] > 0.5).sum())
        for j, cls in enumerate(CLASS_NAMES_SUPER5)
    }
    class_trust = {cls: 0.0 for cls in CLASS_NAMES_SUPER5}
    for cls in ["NORM", "MI", "STTC"]:
        class_trust[cls] = 1.0 if class_counts.get(cls, 0) > 0 else 0.0
    with open(trust_path, "w") as f:
        json.dump({
            "tag": tag,
            "center": args.center,
            "latent_npz": str(latent_path),
            "class_trust": class_trust,
            "class_counts": class_counts,
            "policy": "Real target-center K selection: trust NORM/MI/STTC when present; HYP/CD gated off.",
        }, f, indent=2)
    with open(meta_path, "w") as f:
        json.dump({
            "center": args.center,
            "K": int(selected.size),
            "selection_seed": int(args.selection_seed),
            "ref_record_ids": record_ids,
            "source_indices": selected.tolist(),
        }, f, indent=2)
    with open(report_path, "w") as f:
        report = {
            "tag": tag,
            "center": args.center,
            "K": int(selected.size),
            "latent_npz": str(latent_path),
            "class_trust": str(trust_path),
            "ref_meta": str(meta_path),
            "primary_counts": dict(Counter(primary)),
            "class_counts": class_counts,
        }
        json.dump(report, f, indent=2)
    print(f"[export] wrote {latent_path}")
    print(f"[export] class counts: {class_counts}")

    if args.mmap_root:
        mmap_dir = (
            Path(args.mmap_root)
            / f"super5_{args.center}_100hz1000_v3_super5_normsuppress"
        )
        signals = np.load(mmap_dir / "signals.npy", mmap_mode="r")
        mmap_labels = np.load(mmap_dir / "labels.npy", mmap_mode="r")
        mmap_record_ids = np.load(mmap_dir / "record_ids.npy", allow_pickle=True).astype(str)
        rid_to_i = {str(rid): i for i, rid in enumerate(mmap_record_ids)}
        signal_indices = [rid_to_i[rid] for rid in record_ids if rid in rid_to_i]
        if len(signal_indices) != len(record_ids):
            missing = sorted(set(record_ids) - set(rid_to_i))
            raise ValueError(f"{len(missing)} selected record ids missing from mmap cache; first={missing[:5]}")
        sig = np.asarray(signals[signal_indices], dtype=np.float32)
        lab = np.asarray(mmap_labels[signal_indices], dtype=np.float32)
        np.savez_compressed(
            signal_path,
            signals=sig,
            labels=lab,
            record_ids=np.asarray(record_ids),
            center_name=np.asarray(args.center),
            class_names=np.asarray(CLASS_NAMES_SUPER5),
        )
        report["signal_npz"] = str(signal_path)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[export] wrote {signal_path}")


if __name__ == "__main__":
    main()
