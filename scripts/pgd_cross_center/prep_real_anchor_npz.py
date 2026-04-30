"""Convert existing K=200 .pt ref pool → .latent.npz for real-anchored AT.

Module Ablation Study #1 (2026-04-27): Strip ECGTwin diffusion entirely.
Use real target-center records (already VAE-encoded in .pt) as PGD anchors
instead of synth latents.

Pipeline change vs Plan Rev 13.2:
  ❌ ECGTwin DDPM 50-step synthesis (Stage 1)
  ✅ K=200 real records → existing .pt ref pool already has VAE latents
  ✅ PGD on real latents (same K_pgd=10, ε=2.0, K_anchor=300)

Input:  /root/autodl-tmp/center_token_super5/{tag}_k200.pt
        - samples: list of {data: tensor(4,128), label: {diagnostic_class, ...}}

Output:
  - {out_npz}: latents (K,4,128) + labels (K,5) one-hot of primary super5 class
  - {out_class_trust}: class_trust.json with NORM/MI/STTC=1.0 (HYP/CD will be
    hardcoded 0.0 by DEFAULT_TRUST_HARDCODE in synth_online_at_super5.py)

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \\
    scripts/pgd_cross_center/prep_real_anchor_npz.py \\
    --ref_pt /root/autodl-tmp/center_token_super5/extra_k200.pt \\
    --center_name cpsc_2018_extra \\
    --out_npz /root/autodl-tmp/real_anchored_super5/synth_latents/extra_real_k200.latent.npz \\
    --out_class_trust /root/autodl-tmp/real_anchored_super5/synth_latents/extra_real_k200.class_trust.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5, NUM_SUPER5, SUPER5_TO_IDX,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_pt", required=True,
                    help="K=200 ref pool .pt (output of prep_center_dataset_super5.py)")
    ap.add_argument("--center_name", required=True,
                    help="PN2021 center, e.g. cpsc_2018_extra")
    ap.add_argument("--out_npz", required=True, help="Output .latent.npz path")
    ap.add_argument("--out_class_trust", required=True,
                    help="Output class_trust.json path")
    args = ap.parse_args()

    print(f"[real-prep] loading {args.ref_pt}")
    samples = torch.load(args.ref_pt, map_location="cpu", weights_only=False)
    print(f"[real-prep] loaded {len(samples)} samples")

    latents = np.stack([s["data"].numpy() for s in samples], axis=0).astype(np.float32)
    labels_oh = np.zeros((len(samples), NUM_SUPER5), dtype=np.float32)
    primary_counts = Counter()
    for i, s in enumerate(samples):
        primary = s["label"]["diagnostic_class"]
        if primary not in SUPER5_TO_IDX:
            raise SystemExit(f"sample {i} primary class {primary!r} not in SUPER5_TO_IDX")
        labels_oh[i, SUPER5_TO_IDX[primary]] = 1.0
        primary_counts[primary] += 1

    assert latents.shape[1:] == (4, 128), f"bad latent shape: {latents.shape}"
    assert np.isfinite(latents).all(), "non-finite latents"

    print(f"[real-prep] latents: {latents.shape}  labels: {labels_oh.shape}")
    print(f"[real-prep] primary class counts: {dict(primary_counts)}")

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        latents=latents,
        labels=labels_oh,
        center_name=args.center_name,
    )
    print(f"[real-prep] wrote latent pool → {out_path}")

    class_trust = {c: 0.0 for c in CLASS_NAMES_SUPER5}
    for c in ["NORM", "MI", "STTC"]:
        class_trust[c] = 1.0

    ct_path = Path(args.out_class_trust)
    ct_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ct_path, "w") as f:
        json.dump({
            "tag": f"{args.center_name}_real_k200",
            "synth_pool": str(out_path),
            "per_class_auroc": None,
            "class_trust": class_trust,
            "policy": "Module Ablation #1: real anchors → trust NORM/MI/STTC=1.0; "
                      "HYP/CD=0 (Plan Rev 11 hardcoded by DEFAULT_TRUST_HARDCODE)",
        }, f, indent=2)
    print(f"[real-prep] wrote class_trust → {ct_path}")


if __name__ == "__main__":
    main()
