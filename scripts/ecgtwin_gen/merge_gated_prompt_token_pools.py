"""Merge gated ECGTwin prompt-token pools.

Each input directory should contain outputs from `gate_prompt_token_synth.py`:

  gated_samples.npz
  gated_samples.latent.npz
  gated_samples.class_trust.json
  gated_samples.ref_meta.json

The merged output keeps the same file naming convention so it can be passed
directly to `scripts/pgd_cross_center/synth_online_at_super5.py`.
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


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dirs", nargs="+", required=True,
                    help="Directories containing gated_samples.* outputs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_blobs = [load_npz(d / "gated_samples.npz") for d in input_dirs]
    latent_blobs = [load_npz(d / "gated_samples.latent.npz") for d in input_dirs]
    trust_blobs = [load_json(d / "gated_samples.class_trust.json") for d in input_dirs]
    ref_blobs = [load_json(d / "gated_samples.ref_meta.json") for d in input_dirs]

    center_names = [str(b.get("center_name", "?")) for b in sample_blobs]
    if len(set(center_names)) > 1:
        raise ValueError(f"input centers differ: {center_names}")
    center_name = center_names[0]
    tag = args.tag or center_name

    merged_samples = {
        "signals": np.concatenate([b["signals"].astype(np.float32) for b in sample_blobs], axis=0),
        "raw_signal_ct": np.concatenate([b["raw_signal_ct"].astype(np.float32) for b in sample_blobs], axis=0),
        "latents": np.concatenate([b["latents"].astype(np.float32) for b in sample_blobs], axis=0),
        "labels": np.concatenate([b["labels"].astype(np.float32) for b in sample_blobs], axis=0),
        "center_name": center_name,
        "class_names": np.asarray(CLASS_NAMES_SUPER5),
    }
    merged_latents = {
        "latents": np.concatenate([b["latents"].astype(np.float32) for b in latent_blobs], axis=0),
        "labels": np.concatenate([b["labels"].astype(np.float32) for b in latent_blobs], axis=0),
        "center_name": center_name,
        "class_names": np.asarray(CLASS_NAMES_SUPER5),
    }
    if merged_samples["latents"].shape != merged_latents["latents"].shape:
        raise ValueError("sample and latent merged pools disagree")
    if not np.allclose(merged_samples["latents"], merged_latents["latents"]):
        raise ValueError("sample and latent arrays differ")
    if not np.allclose(merged_samples["labels"], merged_latents["labels"]):
        raise ValueError("sample and latent labels differ")

    class_ids = merged_samples["labels"].argmax(axis=1)
    counts = Counter(CLASS_NAMES_SUPER5[int(i)] for i in class_ids)
    class_trust = {}
    for cls in CLASS_NAMES_SUPER5:
        vals = []
        for blob in trust_blobs:
            vals.append(float(blob.get("class_trust", {}).get(cls, 0.0)))
        class_trust[cls] = 1.0 if counts.get(cls, 0) > 0 and max(vals or [0.0]) > 0.0 else 0.0
    class_trust["HYP"] = 0.0
    class_trust["CD"] = 0.0

    ref_ids = sorted({
        str(rid)
        for blob in ref_blobs
        for rid in blob.get("ref_record_ids", [])
        if str(rid)
    })

    samples_path = out_dir / "gated_samples.npz"
    latent_path = out_dir / "gated_samples.latent.npz"
    trust_path = out_dir / "gated_samples.class_trust.json"
    ref_meta_path = out_dir / "gated_samples.ref_meta.json"
    report_path = out_dir / "merge_report.json"

    np.savez_compressed(samples_path, **merged_samples)
    np.savez_compressed(latent_path, **merged_latents)
    with open(trust_path, "w") as f:
        json.dump({
            "tag": tag,
            "center": center_name,
            "source_dirs": [str(d) for d in input_dirs],
            "gated_latent_pool": str(latent_path),
            "class_trust": class_trust,
            "counts": dict(counts),
            "policy": "trust is enabled for classes with merged samples and any trusted source; HYP/CD hardcoded 0",
        }, f, indent=2)
    with open(ref_meta_path, "w") as f:
        json.dump({
            "center": center_name,
            "ref_record_ids": ref_ids,
            "policy": "Union of source ref ids for downstream quick-eval exclusion.",
        }, f, indent=2)
    with open(report_path, "w") as f:
        json.dump({
            "tag": tag,
            "center": center_name,
            "source_dirs": [str(d) for d in input_dirs],
            "n_samples": int(merged_samples["labels"].shape[0]),
            "counts": dict(counts),
            "samples": str(samples_path),
            "latents": str(latent_path),
            "class_trust": str(trust_path),
            "ref_meta": str(ref_meta_path),
        }, f, indent=2)

    print(f"[merge] wrote {samples_path}")
    print(f"[merge] wrote {latent_path}")
    print(f"[merge] counts: {dict(counts)}")
    print(f"[merge] trust: {class_trust}")
    print(f"[merge] ref ids: {len(ref_ids)}")


if __name__ == "__main__":
    main()
