"""Pre-generate adv buffer → .npz for AugMix × Adv combo ablation.

Phase 0 of /root/.claude/plans/concurrent-finding-hoare.md.

Produces a standalone .npz containing adversarial ECG samples in the exact
format expected by ``SynthCenterDataset`` at
``scripts/crosscenter_tierM/train_ptbxl_tierM.py:110``:

    signals : (N, 12, 1000) float32   @ 100 Hz, canonical leads, z-scored
    labels  : (N, 6)       float32    Tier-M 6-class soft labels in [0,1]

Label scheme ("ref+soft_target"):
  - Non-target dims = reference sample's true binary Tier-M label (multi-hot)
  - Target dim      = victim's probability p_target on the generated adv sample
                      (typically in accept range [0.5, 0.6])

This preserves the ref's real co-occurring conditions for non-target dims,
while the target dim carries boundary-confidence — avoids the 'hard 1.0 on
weak-confidence adv' problem observed in the online/offline ablation.

Pipeline reuses the offline gen path (zero duplication of the budget/gen
loop), only the labeling + 1000-length retention differs from
``augmix_inject_and_buffer``.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from adversarial.adv_generate import BoundaryAdvDiffGenerator  # noqa: E402
from adversarial.efficientnet_victim_tierM import (  # noqa: E402
    EfficientNetVictimTierM, DEFAULT_TIERM_CKPT,
)
from adversarial.tierM_labels import TIER_M_CLASSES  # noqa: E402
from methods.augmix.latent_viz.latent_augmix import latent_augmix_on_signal  # noqa: E402
from scripts.crosscenter_tierM.refs_per_center import (  # noqa: E402
    sample_refs_per_center, DEFAULT_PN2021_DIR, MAIN_CENTERS_4,
)
from scripts.crosscenter_tierM.online_adv_train_tierM import (  # noqa: E402
    TIERM_GEN_HYPERPARAMS_DEFAULT,
    build_budget,
    encode_refs_per_center,
    generate_adv_for_epoch,
)

DEFAULT_OUT_NPZ = "/root/autodl-tmp/adv_buffer_tierM/adv_buffer_n1800_soft.npz"
BASELINE_EVAL_JSON = "/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json"
ONLINE_REFS_CACHE = "/root/autodl-tmp/crosscenter_tierM_online/refs_cache_k32.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_npz", default=DEFAULT_OUT_NPZ)
    p.add_argument("--victim_ckpt", default=DEFAULT_TIERM_CKPT)
    p.add_argument("--baseline_eval", default=BASELINE_EVAL_JSON)
    p.add_argument("--refs_cache", default=ONLINE_REFS_CACHE)
    p.add_argument("--data_dir", default=DEFAULT_PN2021_DIR)
    p.add_argument("--centers", nargs="+", default=MAIN_CENTERS_4)
    p.add_argument("--n_adv_total", type=int, default=1800)
    p.add_argument("--max_attempts_per_entry", type=int, default=150)
    p.add_argument("--k_per_center", type=int, default=32)
    p.add_argument("--accept_prob_low", type=float, default=0.5)
    p.add_argument("--accept_prob_high", type=float, default=0.6)
    p.add_argument("--augmix_width", type=int, default=3)
    p.add_argument("--augmix_severity", type=int, default=5)
    p.add_argument("--augmix_depth", type=int, default=-1)
    p.add_argument("--best_of_k", type=int, default=5)
    p.add_argument("--gen_batch_size", type=int, default=4)
    p.add_argument("--cell_skip_threshold", type=float, default=1.1)
    p.add_argument("--min_alloc_per_cell", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_soft_label(
    ref_label_6: np.ndarray,
    target_idx: int,
    p_target: float,
) -> np.ndarray:
    """Ref multi-hot binary labels, target dim replaced with boundary victim prob."""
    lbl = np.asarray(ref_label_6, dtype=np.float32).copy()
    lbl[target_idx] = float(np.clip(p_target, 0.0, 1.0))
    return lbl


def augmix_and_collect(
    adv_items: List[Dict[str, Any]],
    ecgtwin: ECGTwinWrapper,
    augmix_width: int,
    augmix_severity: int,
    augmix_depth: int,
    device: str,
):
    """For each adv item: run latent-AugMix (1 adv chain + (width-1) time-domain chains),
    keep full (12, 1000), build ref+soft_target label. Returns two stacked np arrays."""
    signals_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    t0 = time.time()
    n = len(adv_items)
    for i, item in enumerate(adv_items):
        ref_sig_ct = np.asarray(item["ref_signal_ct_1000_canonical"], dtype=np.float32)
        z_adv = item["z_adv"]
        if z_adv.dim() == 2:
            z_adv = z_adv.unsqueeze(0)                    # (1, 4, 128)
        res = latent_augmix_on_signal(
            wrapper=ecgtwin,
            signal_ct_100hz=ref_sig_ct,
            severity=augmix_severity,
            width=augmix_width,
            depth=augmix_depth,
            alpha=1.0,
            device=device,
            inject_latents=[z_adv],
        )
        augmix_ct_1000 = np.asarray(res.augmix_latent_ct, dtype=np.float32)  # (12, 1000)
        lbl = build_soft_label(
            ref_label_6=item["ref_true_label_6"],
            target_idx=item["target_idx"],
            p_target=float(item["probs_6"][item["target_idx"]]),
        )
        signals_list.append(augmix_ct_1000)
        labels_list.append(lbl)
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  [inject] {i+1}/{n}  elapsed={time.time()-t0:.0f}s", flush=True)
    signals = np.stack(signals_list, axis=0)    # (N, 12, 1000)
    labels = np.stack(labels_list, axis=0)      # (N, 6)
    return signals, labels


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 72)
    print("AugMix×Adv combo: Phase 1 — pre-generate adv buffer → npz")
    print("=" * 72)
    print(f"  output_npz:   {args.output_npz}")
    print(f"  victim_ckpt:  {args.victim_ckpt}")
    print(f"  n_adv_total:  {args.n_adv_total}")
    print(f"  accept_prob:  [{args.accept_prob_low}, {args.accept_prob_high}]")
    print(f"  augmix w={args.augmix_width} s={args.augmix_severity}")
    print("-" * 72)

    # ECGTwin + victim (frozen baseline)
    print("[setup] loading ECGTwin...")
    ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=True)
    print("[setup] loading baseline victim (frozen)...")
    victim = EfficientNetVictimTierM(
        weight_path=args.victim_ckpt, device=args.device, ecgtwin_wrapper=ecgtwin,
    )
    victim.model.eval()

    # Refs (reuse online cache)
    refs = sample_refs_per_center(
        k_per_center=args.k_per_center, centers=args.centers,
        data_dir=args.data_dir, cache_path=args.refs_cache, seed=args.seed,
        verbose=True,
    )
    print("[setup] encoding refs → VAE latents...")
    ref_items_by_center = encode_refs_per_center(refs, ecgtwin, args.device)

    # Budget
    with open(args.baseline_eval) as f:
        baseline_eval = json.load(f)
    budget = build_budget(
        eval_json=baseline_eval, n_adv_total=args.n_adv_total,
        centers=args.centers, tier_m_classes=TIER_M_CLASSES,
        cell_skip_threshold=args.cell_skip_threshold,
        min_alloc_per_cell=args.min_alloc_per_cell,
    )
    total_alloc = sum(e["n_alloc"] for e in budget)
    print(f"[budget] total_alloc={total_alloc}")
    for e in budget:
        print(f"    {e['center']:<22} × {e['class']:<6} "
              f"AUROC={e['baseline_auroc']:.3f} n_alloc={e['n_alloc']}")

    # Generator
    gen_hp = dict(TIERM_GEN_HYPERPARAMS_DEFAULT)
    gen_hp["acceptance_range"] = [args.accept_prob_low, args.accept_prob_high]
    gen_hp["batch_size"] = args.gen_batch_size
    generator = BoundaryAdvDiffGenerator(
        ecgtwin_wrapper=ecgtwin, victim=victim, device=args.device, hyperparams=gen_hp,
    )

    print("\n" + "=" * 72)
    print(f"[GEN] target={args.n_adv_total} samples, frozen baseline victim")
    print("=" * 72)
    z_hist: Dict = {}
    t0 = time.time()
    adv_items, gen_stats = generate_adv_for_epoch(
        generator=generator, ecgtwin=ecgtwin, budget=budget,
        ref_items_by_center=ref_items_by_center, z_hist=z_hist,
        device=args.device, best_of_k=args.best_of_k,
        max_attempts_per_entry=args.max_attempts_per_entry,
    )
    gen_elapsed = time.time() - t0
    accept_rate = gen_stats["accepted"] / max(1, gen_stats["attempted"])
    print(f"[GEN] accepted={gen_stats['accepted']} attempted={gen_stats['attempted']} "
          f"rate={accept_rate:.2f} entries_completed={gen_stats['entries_completed']} "
          f"elapsed={gen_elapsed:.0f}s")
    if accept_rate < 0.20:
        print(f"[WARN] accept rate {accept_rate:.2f} < 0.20 — consider relaxing accept_prob "
              f"(currently [{args.accept_prob_low},{args.accept_prob_high}])")

    print("\n" + "=" * 72)
    print(f"[INJECT] running latent-AugMix on {len(adv_items)} adv samples (keep full 12×1000)")
    print("=" * 72)
    signals, labels = augmix_and_collect(
        adv_items=adv_items, ecgtwin=ecgtwin,
        augmix_width=args.augmix_width, augmix_severity=args.augmix_severity,
        augmix_depth=args.augmix_depth, device=args.device,
    )
    print(f"[INJECT] done. signals={signals.shape} labels={labels.shape}")

    # Sanity
    assert signals.shape[1] == 12 and signals.shape[2] == 1000, \
        f"unexpected signal shape {signals.shape}"
    assert labels.shape[1] == 6 and labels.shape[0] == signals.shape[0], \
        f"label/signal mismatch: {labels.shape} vs {signals.shape}"
    print(f"[stats] label target-dim mean={labels[np.arange(len(labels)), [item['target_idx'] for item in adv_items]].mean():.3f} "
          f"(expect ≈accept-range midpoint)")
    print(f"[stats] label full-array mean={labels.mean():.3f}  "
          f"min={labels.min():.3f} max={labels.max():.3f}")

    # Save npz
    meta = {
        "args": vars(args),
        "gen_stats": gen_stats,
        "accept_rate": round(accept_rate, 3),
        "gen_elapsed_s": round(gen_elapsed, 1),
        "budget": [{"center": e["center"], "class": e["class"],
                    "n_alloc": e["n_alloc"], "baseline_auroc": e["baseline_auroc"]}
                   for e in budget],
        "label_scheme": "ref+soft_target (non-target=ref binary, target=victim_prob clipped)",
        "n_samples": int(signals.shape[0]),
    }
    np.savez_compressed(
        args.output_npz,
        signals=signals, labels=labels,
        meta=json.dumps(meta, default=str),
    )
    print(f"\n[DONE] saved {signals.shape[0]} samples → {args.output_npz}")
    print(f"       file size: {os.path.getsize(args.output_npz)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
