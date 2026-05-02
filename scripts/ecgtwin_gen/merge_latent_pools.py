"""Merge generic ECGTwin latent pools for online AT.

Inputs are `.latent.npz` files with:

  latents: (N, 4, 128)
  labels:  (N, 5)
  center_name

Optional class-trust/ref-meta JSON files can be passed separately. The output is
compatible with `scripts/pgd_cross_center/synth_online_at_super5.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


def load_json(path: str | None) -> Dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_npzs", nargs="+", required=True)
    ap.add_argument("--source_names", nargs="*", default=[],
                    help="Optional per-input source names, e.g. real_anchor prompt_token")
    ap.add_argument("--class_trust_jsons", nargs="*", default=[])
    ap.add_argument("--ref_meta_jsons", nargs="*", default=[])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.source_names and len(args.source_names) != len(args.latent_npzs):
        raise ValueError("--source_names must be empty or match --latent_npzs length")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latents: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    centers: List[str] = []
    source_ids: List[np.ndarray] = []
    source_local_indices: List[np.ndarray] = []
    source_names = list(args.source_names) if args.source_names else [
        Path(p).with_suffix("").with_suffix("").name for p in args.latent_npzs
    ]
    for p in args.latent_npzs:
        with np.load(p, allow_pickle=True) as data:
            z = data["latents"].astype(np.float32)
            y = data["labels"].astype(np.float32)
            source_idx = len(latents)
            latents.append(z)
            labels.append(y)
            centers.append(str(data["center_name"]) if "center_name" in data.files else "?")
            source_ids.append(np.full((z.shape[0],), source_idx, dtype=np.int16))
            source_local_indices.append(np.arange(z.shape[0], dtype=np.int32))
    if len(set(centers)) != 1:
        raise ValueError(f"input centers differ: {centers}")
    center_name = centers[0]
    merged_latents = np.concatenate(latents, axis=0)
    merged_labels = np.concatenate(labels, axis=0)
    merged_source_ids = np.concatenate(source_ids, axis=0)
    merged_source_local_indices = np.concatenate(source_local_indices, axis=0)
    if merged_latents.ndim != 3 or merged_latents.shape[1:] != (4, 128):
        raise ValueError(f"bad merged latent shape: {merged_latents.shape}")
    if merged_labels.shape[0] != merged_latents.shape[0]:
        raise ValueError("merged latents and labels length mismatch")

    class_ids = merged_labels.argmax(axis=1)
    counts = Counter(CLASS_NAMES_SUPER5[int(i)] for i in class_ids)

    trust_blobs = [load_json(p) for p in args.class_trust_jsons]
    class_trust = {}
    for cls in CLASS_NAMES_SUPER5:
        vals = [float(b.get("class_trust", {}).get(cls, 0.0)) for b in trust_blobs]
        class_trust[cls] = 1.0 if counts.get(cls, 0) > 0 and (not vals or max(vals) > 0.0) else 0.0
    class_trust["HYP"] = 0.0
    class_trust["CD"] = 0.0

    ref_blobs = [load_json(p) for p in args.ref_meta_jsons]
    ref_ids = sorted({
        str(rid)
        for blob in ref_blobs
        for rid in blob.get("ref_record_ids", [])
        if str(rid)
    })

    tag = args.tag or center_name
    latent_path = out_dir / "merged.latent.npz"
    trust_path = out_dir / "merged.class_trust.json"
    ref_meta_path = out_dir / "merged.ref_meta.json"
    report_path = out_dir / "merge_report.json"

    np.savez_compressed(
        latent_path,
        latents=merged_latents,
        labels=merged_labels,
        center_name=center_name,
        class_names=np.asarray(CLASS_NAMES_SUPER5),
        source_ids=merged_source_ids,
        source_names=np.asarray(source_names),
        source_local_indices=merged_source_local_indices,
        source_latent_npzs=np.asarray(args.latent_npzs),
    )
    with open(trust_path, "w") as f:
        json.dump({
            "tag": tag,
            "center": center_name,
            "source_latent_npzs": list(args.latent_npzs),
            "source_names": source_names,
            "class_trust": class_trust,
            "counts": dict(counts),
            "policy": "trust is enabled for present classes unless all supplied trusts reject them; HYP/CD hardcoded 0",
        }, f, indent=2)
    with open(ref_meta_path, "w") as f:
        json.dump({
            "center": center_name,
            "ref_record_ids": ref_ids,
            "policy": "Union of supplied ref ids for downstream quick-eval exclusion.",
        }, f, indent=2)
    with open(report_path, "w") as f:
        json.dump({
            "tag": tag,
            "center": center_name,
            "source_latent_npzs": list(args.latent_npzs),
            "source_names": source_names,
            "n_samples": int(merged_labels.shape[0]),
            "counts": dict(counts),
            "latent_pool": str(latent_path),
            "class_trust": str(trust_path),
            "ref_meta": str(ref_meta_path),
            "source_class_trust_jsons": list(args.class_trust_jsons),
            "source_ref_meta_jsons": list(args.ref_meta_jsons),
        }, f, indent=2)

    print(f"[merge] wrote {latent_path}")
    print(f"[merge] counts: {dict(counts)}")
    print(f"[merge] trust: {class_trust}")
    print(f"[merge] ref ids: {len(ref_ids)}")


if __name__ == "__main__":
    main()
