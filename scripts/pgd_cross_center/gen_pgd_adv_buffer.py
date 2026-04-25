"""Generate PGD adversarial buffer for a single (center, K) variant.

Pipeline:
  1. Load pre-encoded per-center .pt (center_token_ablation datasets)
  2. Filter fold<=5 + stratified subsample to K anchors (6-class Tier-M)
  3. For each anchor: generate n_per_anchor adv samples via VAE-PGD
  4. Decode anchor latents (for Gate 2 reference)
  5. Run Gate 1 (ASR) + Gate 2 (semantics) validation
  6. Save {out_npz} + {out_json} validation record

Output npz schema (matches SynthCenterDataset):
  signals: (K*n_per_anchor, 12, 1000) float32  PTBXL-order, z-scored
  labels:  (K*n_per_anchor, 6)        float32  one-hot on anchor's diagnostic_class

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/pgd_cross_center/gen_pgd_adv_buffer.py \
    --full_pt /root/autodl-tmp/center_token_ablation/datasets/extra_full.pt \
    --center_tag extra_k100 --K 100 \
    --out /root/autodl-tmp/pgd_adv_ablation/buffers/extra_k100_eps1.0.npz \
    --epsilon 1.0 --K_pgd 10 --n_per_anchor 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from adversarial.efficientnet_victim_tierM import (  # noqa: E402
    EfficientNetVictimTierM, DEFAULT_TIERM_CKPT,
)
from adversarial.pgd_advdiff import PGDAdvDiffGenerator  # noqa: E402
from adversarial.adv_validation import validate_buffer  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402

TIER_M_CLS = ["NSR", "STach", "AF", "IAVB", "LBBB", "RBBB"]
TIER_M_CLS_TO_IDX = {c: i for i, c in enumerate(TIER_M_CLS)}


def stratified_sample_anchors(pool, k, seed):
    """Stratified: ~k/6 per class, backfill with random."""
    rng = np.random.default_rng(seed)
    by_cls_idx = {c: [] for c in TIER_M_CLS}
    for i, s in enumerate(pool):
        cls = s["label"].get("diagnostic_class")
        if cls in by_cls_idx:
            by_cls_idx[cls].append(i)

    per_cls_target = max(1, k // len(TIER_M_CLS))
    picked = set()
    for c, idxs in by_cls_idx.items():
        if not idxs:
            continue
        take = min(per_cls_target, len(idxs))
        chosen = rng.choice(len(idxs), size=take, replace=False)
        for j in chosen:
            picked.add(idxs[int(j)])
    if len(picked) < k:
        remaining = [i for i in range(len(pool)) if i not in picked]
        need = min(k - len(picked), len(remaining))
        if need > 0:
            chosen = rng.choice(len(remaining), size=need, replace=False)
            for j in chosen:
                picked.add(remaining[int(j)])
    return [pool[i] for i in sorted(picked)]


def decode_anchors_to_ptbxl_1000(z0_batch: torch.Tensor, victim, device) -> np.ndarray:
    """Decode anchor latents (N, 4, 128) → (N, 12, 1000) PTBXL-order z-scored.

    Mirrors PGDAdvDiffGenerator._decode_to_ptbxl_1000 for reference/validation.
    """
    import torch.nn.functional as F
    from adversarial.efficientnet_victim_tierM import TIERM_AMP_CLAMP, TIERM_PREPROC_LENGTH
    z0_batch = z0_batch.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, z0_batch.shape[0], 16):
            chunk = z0_batch[i:i + 16]
            ecg_tc = victim._decode_latent_differentiable(chunk)    # (B, 1024, 12)
            ecg_ct = ecg_tc.transpose(-1, -2)
            ecg_ct = ecg_ct[:, ECGTWIN_TO_PTBXL_INDICES, :]
            ecg_ct = torch.clamp(ecg_ct, -TIERM_AMP_CLAMP, TIERM_AMP_CLAMP)
            ecg_ct = F.interpolate(ecg_ct, size=TIERM_PREPROC_LENGTH,
                                   mode="linear", align_corners=True)
            ecg_ct = victim._global_zscore(ecg_ct)
            out.append(ecg_ct.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_pt", required=True,
                    help="Per-center full .pt (from prep_center_dataset.py), list of {data, label}")
    ap.add_argument("--center_tag", required=True, help="e.g. extra_k100")
    ap.add_argument("--K", type=int, required=True, help="Number of anchor samples")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--victim_ckpt", default=DEFAULT_TIERM_CKPT)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--K_pgd", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=None,
                    help="PGD step size; default 2ε/K_pgd")
    ap.add_argument("--delta_init_scale", type=float, default=0.1)
    ap.add_argument("--n_per_anchor", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_validation", action="store_true",
                    help="Skip Gates 1+2 (for debugging only)")
    ap.add_argument("--allow_validation_fail", action="store_true",
                    help="Save buffer even if validation fails (for analysis)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    val_json_path = args.out.replace(".npz", "_validation.json")

    print("=" * 72)
    print(f"[pgd-gen] center_tag={args.center_tag}  K={args.K}  ε={args.epsilon}  "
          f"K_pgd={args.K_pgd}  n_per_anchor={args.n_per_anchor}")
    print("=" * 72)

    # --- 1. Load full .pt and subsample K anchors ---
    print(f"[1/5] load {args.full_pt}")
    pool = torch.load(args.full_pt, map_location="cpu", weights_only=False)
    fold_max_full = max(int(s["label"].get("strat_fold", 1)) for s in pool)
    ref_half = [s for s in pool if int(s["label"].get("strat_fold", 1)) <= 5]
    print(f"    N_total={len(pool)}  N_refhalf(fold≤5)={len(ref_half)}  "
          f"fold_max_full={fold_max_full}")

    if args.K > len(ref_half):
        print(f"[WARN] K={args.K} > ref_half pool {len(ref_half)}; truncating")
        anchors = list(ref_half)
    else:
        anchors = stratified_sample_anchors(ref_half, args.K, seed=args.seed)

    cls_dist = Counter(s["label"]["diagnostic_class"] for s in anchors)
    print(f"    K_actual={len(anchors)}  classes={dict(cls_dist)}")

    # --- 2. Load ECGTwin wrapper + victim ---
    print("[2/5] load ECGTwin (no encoder, no text model — using pre-encoded z0)")
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)
    print("[2/5] load Tier-M victim (frozen baseline)")
    victim = EfficientNetVictimTierM(
        weight_path=args.victim_ckpt, device=args.device, ecgtwin_wrapper=wrapper,
    )
    victim.eval()

    # --- 3. Build PGD generator ---
    gen = PGDAdvDiffGenerator(
        ecgtwin_wrapper=wrapper, victim=victim,
        epsilon=args.epsilon, K_pgd=args.K_pgd, alpha=args.alpha,
        delta_init_scale=args.delta_init_scale, device=args.device,
    )

    # --- 4. Generate adv samples: K anchors × n_per_anchor = K*n ---
    print(f"[3/5] generating {len(anchors)} × {args.n_per_anchor} = "
          f"{len(anchors) * args.n_per_anchor} adv samples")
    t0 = time.time()
    all_signals, all_labels, all_delta_norms = [], [], []
    for i, anc in enumerate(anchors):
        z0 = anc["data"].float()                      # (4, 128)
        cls_name = anc["label"]["diagnostic_class"]
        cls_idx = TIER_M_CLS_TO_IDX[cls_name]
        y_onehot = np.zeros(6, dtype=np.float32); y_onehot[cls_idx] = 1.0

        signals, labels, info = gen.generate_for_anchor(
            z0, y_onehot, n_per_anchor=args.n_per_anchor,
        )
        all_signals.append(signals)
        all_labels.append(labels)
        all_delta_norms.extend(info["final_delta_norms"])

        if (i + 1) % 20 == 0 or (i + 1) == len(anchors):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    [{i+1}/{len(anchors)}] {elapsed:.0f}s  {rate:.2f} anchor/s  "
                  f"ETA={((len(anchors)-i-1)/max(rate,1e-6)):.0f}s", flush=True)

    adv_signals = np.concatenate(all_signals, axis=0).astype(np.float32)    # (N, 12, 1000)
    adv_labels = np.concatenate(all_labels, axis=0).astype(np.float32)      # (N, 6)
    assert adv_signals.shape[0] == adv_labels.shape[0]
    assert np.isfinite(adv_signals).all(), "non-finite adv signals produced!"
    print(f"[3/5] generated {adv_signals.shape} labels={adv_labels.shape} in {time.time()-t0:.0f}s")
    print(f"    delta norms: mean={np.mean(all_delta_norms):.3f}  max={np.max(all_delta_norms):.3f}  "
          f"(target ε={args.epsilon})")

    # --- 5. Run validation gates ---
    if not args.skip_validation:
        print("[4/5] running validation Gate 1 (ASR) + Gate 2 (semantics)")
        # Decode anchor latents to (12,1000) PTBXL for Gate 2 reference
        anc_z0_batch = torch.stack([a["data"].float() for a in anchors], dim=0)
        anchor_signals = decode_anchors_to_ptbxl_1000(anc_z0_batch, victim, args.device)

        val_result = validate_buffer(
            adv_signals=adv_signals,
            adv_labels_multi_hot=adv_labels,
            anchor_signals=anchor_signals,
            victim=victim,
            device=args.device,
        )
        print(f"    Gate 1 (ASR): PASS={val_result['gate1']['PASS']}  "
              f"asr_overall={val_result['gate1']['asr_overall']:.3f}  "
              f"per_class={val_result['gate1']['per_class_asr']}")
        print(f"    Gate 2 (semantics): PASS={val_result['gate2']['PASS']}  "
              f"hr_delta={val_result['gate2']['hr_mean_delta']:.1f}  "
              f"qrs_ratio={val_result['gate2']['qrs_amp_ratio']:.2f}  "
              f"einthoven_p95={val_result['gate2']['einthoven_mean_p95']}")
        print(f"    OVERALL PASS={val_result['PASS']}  reasons={val_result['fail_reasons']}")

        # Save validation JSON (serializable)
        val_serializable = json.loads(json.dumps(val_result, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))
        with open(val_json_path, "w") as f:
            json.dump(val_serializable, f, indent=2)
        print(f"    validation → {val_json_path}")

        if not val_result["PASS"] and not args.allow_validation_fail:
            print(f"[5/5] VALIDATION FAILED — not saving buffer. "
                  f"Use --allow_validation_fail to override.")
            return 1
    else:
        print("[4/5] validation SKIPPED (--skip_validation)")

    # --- 6. Save buffer ---
    print(f"[5/5] saving {args.out}")
    meta = {
        "center_tag": args.center_tag,
        "K_req": int(args.K),
        "K_actual": int(len(anchors)),
        "n_per_anchor": int(args.n_per_anchor),
        "total_samples": int(adv_signals.shape[0]),
        "epsilon": float(args.epsilon),
        "K_pgd": int(args.K_pgd),
        "alpha": float(args.alpha) if args.alpha is not None else None,
        "delta_init_scale": float(args.delta_init_scale),
        "seed": int(args.seed),
        "victim_ckpt": args.victim_ckpt,
        "class_dist_anchors": dict(cls_dist),
        "delta_norms_mean": float(np.mean(all_delta_norms)),
        "delta_norms_max": float(np.max(all_delta_norms)),
        "label_scheme": "one-hot on anchor's diagnostic_class (primary Tier-M class)",
    }
    np.savez_compressed(
        args.out, signals=adv_signals, labels=adv_labels,
        meta=json.dumps(meta, default=str),
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"    → {args.out}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
