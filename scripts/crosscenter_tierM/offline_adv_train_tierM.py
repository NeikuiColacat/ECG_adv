"""Tier-M 6-class 离线对抗训练 (Offline: gen-once + fine-tune)

与 online_adv_train_tierM 的唯一差异：
  Phase 0  用 **冻结 baseline victim** 一次性生成 N_adv_total 个 adv 样本，
           augmix 注入 buffer（固定，无 eviction）
  Phase 1  复用 online 完全相同的 train_one_epoch_tierM + mixed-loader 结构
           训练 20 epochs，每 eval_every 跑 quick_eval 选 best checkpoint

**不做**：per-epoch 重生成、budget 重分配、buffer rescore / eviction。

Outputs: /root/autodl-tmp/crosscenter_tierM_offline/
  best_model.pt, training_log.json, train_result.json, sample_adv_offline.png
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path("/root/autodl-tmp/models/DeepECG/notebooks")))

from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from adversarial.adv_generate import BoundaryAdvDiffGenerator  # noqa: E402
from adversarial.efficientnet_victim_tierM import (  # noqa: E402
    EfficientNetVictimTierM, DEFAULT_TIERM_CKPT, TIERM_INPUT_LENGTH,
)
from adversarial.tierM_labels import TIER_M_CLASSES  # noqa: E402
from scripts.crosscenter_tierM.refs_per_center import (  # noqa: E402
    sample_refs_per_center, DEFAULT_PN2021_DIR, MAIN_CENTERS_4,
)
# Reuse every helper from the online script — zero duplication
from scripts.crosscenter_tierM.online_adv_train_tierM import (  # noqa: E402
    TIERM_GEN_HYPERPARAMS_DEFAULT,
    QualityAwareBufferTierM,
    build_budget,
    encode_refs_per_center,
    generate_adv_for_epoch,
    augmix_inject_and_buffer,
    load_ptbxl_tierM_train_val,
    build_roundtrip_anchor_dataset,
    build_quick_eval_subset,
    quick_eval,
    compute_pos_weight_tierM,
    train_one_epoch_tierM,
    save_adv_viz,
)


DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/crosscenter_tierM_offline"
BASELINE_EVAL_JSON = "/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json"
# Online caches we can reuse to skip ~5 min of preprocessing
ONLINE_REFS_CACHE = "/root/autodl-tmp/crosscenter_tierM_online/refs_cache_k32.npz"
ONLINE_ROUNDTRIP_CACHE = "/root/autodl-tmp/crosscenter_tierM_online/roundtrip_anchor_n1500.npz"
ONLINE_QUICK_EVAL_CACHE = "/root/autodl-tmp/crosscenter_tierM_online/quick_eval_subset_n800.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--init_ckpt", default=DEFAULT_TIERM_CKPT)
    p.add_argument("--baseline_eval_json", default=BASELINE_EVAL_JSON)
    p.add_argument("--data_dir", default=DEFAULT_PN2021_DIR)
    p.add_argument("--centers", nargs='+', default=MAIN_CENTERS_4)
    # --- offline-specific ---
    p.add_argument("--n_adv_total", type=int, default=1000,
                   help="One-shot total adv sample target. Online v5 accumulated ~1900 over 20 epochs. "
                        "We use 1000 as a quick-ablation budget (cuts gen time in half).")
    p.add_argument("--max_attempts_per_entry", type=int, default=150,
                   help="Cap on attempts per (center,class) cell for offline gen.")
    # --- training (mirror online defaults) ---
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--k_per_center", type=int, default=32)
    p.add_argument("--accept_prob_low", type=float, default=0.5)
    p.add_argument("--accept_prob_high", type=float, default=0.6)
    p.add_argument("--augmix_width", type=int, default=3)
    p.add_argument("--augmix_severity", type=int, default=5)
    p.add_argument("--qab_size", type=int, default=3000,
                   help="Large enough so no eviction happens in offline mode.")
    p.add_argument("--anchor_lambda", type=float, default=0.05)
    p.add_argument("--ewa_decay", type=float, default=0.999)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--soft_labels", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--quick_eval_n_per_center", type=int, default=800,
                   help="Default 800 to reuse online's existing quick_eval_subset_n800.npz cache.")
    p.add_argument("--best_of_k", type=int, default=5)
    p.add_argument("--roundtrip_anchor_n", type=int, default=1500)
    p.add_argument("--roundtrip_weight", type=float, default=0.5)
    p.add_argument("--cell_skip_threshold", type=float, default=1.1)
    p.add_argument("--min_alloc_per_cell", type=int, default=1)
    p.add_argument("--adv_label_mode", choices=["ref", "target_only"], default="target_only")
    # --- cache reuse (read-only) ---
    p.add_argument("--refs_cache", default=ONLINE_REFS_CACHE)
    p.add_argument("--roundtrip_cache", default=ONLINE_ROUNDTRIP_CACHE)
    p.add_argument("--quick_eval_cache", default=ONLINE_QUICK_EVAL_CACHE)
    return p.parse_args()


def _build_mixed_loader(train_ds, roundtrip_ds, buffer, args):
    """Replicates Phase C of online's epoch loop. Returns a DataLoader."""
    buf_ds = buffer.to_dataset()
    streams: List[Tuple[Any, float, Optional[List[float]]]] = [(train_ds, 1.0, None)]
    if roundtrip_ds is not None:
        streams.append((roundtrip_ds, args.roundtrip_weight, None))
    if buf_ds is not None and len(buf_ds) > 0:
        streams.append((buf_ds, 2.0, buffer.get_sampling_weights()))

    if len(streams) == 1:
        return DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )

    weights: List[float] = []
    total_n = 0
    for ds_i, base_w, per_sample_w in streams:
        n = len(ds_i)
        if per_sample_w is None:
            weights.extend([base_w] * n)
        else:
            weights.extend([base_w * max(w, 0.05) for w in per_sample_w])
        total_n += n
    sampler = WeightedRandomSampler(weights, num_samples=total_n, replacement=True)
    combined = ConcatDataset([s[0] for s in streams])
    return DataLoader(
        combined, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 72)
    print("Tier-M 离线对抗训练 (offline_adv_train_tierM)")
    print("=" * 72)
    print(f"  init_ckpt:    {args.init_ckpt}")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  n_adv_total:  {args.n_adv_total}   (ONE-SHOT, frozen baseline victim)")
    print(f"  n_epochs:     {args.n_epochs}")
    print(f"  accept_prob:  [{args.accept_prob_low}, {args.accept_prob_high}]")
    print(f"  augmix_width: {args.augmix_width}  severity={args.augmix_severity}")
    print(f"  anchor_lambda:{args.anchor_lambda}  ewa_decay={args.ewa_decay}")
    print(f"  lr:           {args.lr}")
    print("-" * 72)

    # ── ECGTwin ─────────────────────────────────────────────────────────────
    print("[setup] Loading ECGTwin (encoder + text model)...")
    ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=True)

    # ── Victim (baseline, frozen during gen, trained in Phase 1) ────────────
    print("[setup] Loading Tier-M baseline victim...")
    victim = EfficientNetVictimTierM(
        weight_path=args.init_ckpt, device=args.device, ecgtwin_wrapper=ecgtwin,
    )
    # EWA anchor snapshot on TRAINABLE params
    ewa_params = [p.data.clone().detach()
                  for p in victim.model.parameters() if p.requires_grad]

    # ── Refs (reuse online cache; sample_refs_per_center handles cache-hit) ─
    refs = sample_refs_per_center(
        k_per_center=args.k_per_center, centers=args.centers,
        data_dir=args.data_dir, cache_path=args.refs_cache, seed=args.seed,
        verbose=True,
    )
    print("[setup] Encoding refs -> VAE latents...")
    ref_items_by_center = encode_refs_per_center(refs, ecgtwin, args.device)

    # ── Baseline budget (ONE-SHOT, no refresh during training) ──────────────
    with open(args.baseline_eval_json) as f:
        baseline_eval = json.load(f)
    budget = build_budget(
        eval_json=baseline_eval, n_adv_total=args.n_adv_total,
        centers=args.centers, tier_m_classes=TIER_M_CLASSES,
        cell_skip_threshold=args.cell_skip_threshold,
        min_alloc_per_cell=args.min_alloc_per_cell,
    )
    total_alloc = sum(e['n_alloc'] for e in budget)
    print(f"[budget] one-shot allocation (sum={total_alloc}):")
    for e in budget:
        print(f"    {e['center']:<22} × {e['class']:<6} "
              f"AUROC={e['baseline_auroc']:.3f} n_alloc={e['n_alloc']}")

    # ── PTBXL train / val ───────────────────────────────────────────────────
    print("[setup] Loading PTBXL train/val...")
    train_ds, val_ds, train_labels_6 = load_ptbxl_tierM_train_val()
    pos_weight = compute_pos_weight_tierM(train_labels_6).to(args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"  PTBXL train n={len(train_ds)}, val n={len(val_ds)}")
    print(f"  pos_weight={pos_weight.cpu().tolist()}")

    # ── Roundtrip anchor (reuse online cache) ───────────────────────────────
    roundtrip_ds: Optional[TensorDataset] = None
    if args.roundtrip_anchor_n > 0:
        roundtrip_ds = build_roundtrip_anchor_dataset(
            train_ds=train_ds, ecgtwin=ecgtwin, n_samples=args.roundtrip_anchor_n,
            device=args.device, crop_len=TIERM_INPUT_LENGTH,
            seed=args.seed, cache_path=args.roundtrip_cache,
        )
        print(f"[setup] roundtrip-anchor ready: n={len(roundtrip_ds)}, "
              f"weight={args.roundtrip_weight}")

    # ── Quick-eval subset (for best-model selection) ────────────────────────
    quick_subset = build_quick_eval_subset(
        centers=args.centers, data_dir=args.data_dir,
        n_per_center=args.quick_eval_n_per_center,
        cache_path=args.quick_eval_cache, seed=args.seed,
    )
    baseline_quick = quick_eval(victim.model, quick_subset, args.device)
    print(f"[baseline] quick-eval avg macro AUROC={baseline_quick['avg_macro_auroc']}, "
          f"AUPRC={baseline_quick['avg_macro_auprc']}")
    for c, info in baseline_quick["per_center"].items():
        print(f"    {c}: AUROC={info['macro_auroc']}  AUPRC={info['macro_auprc']} (n={info['n']})")

    # ── Optimizer / scheduler ───────────────────────────────────────────────
    trainable_params = [p for p in victim.model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.lr * 0.01)

    # ── Generator + buffer ──────────────────────────────────────────────────
    gen_hp = dict(TIERM_GEN_HYPERPARAMS_DEFAULT)
    gen_hp["acceptance_range"] = [args.accept_prob_low, args.accept_prob_high]
    gen_hp["batch_size"] = args.gen_batch_size
    generator = BoundaryAdvDiffGenerator(
        ecgtwin_wrapper=ecgtwin, victim=victim, device=args.device, hyperparams=gen_hp,
    )
    buffer = QualityAwareBufferTierM(max_size=args.qab_size)
    z_hist: Dict[Tuple[str, str], torch.Tensor] = {}  # unused in offline but API needs it

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 0: One-shot offline generation (frozen baseline victim)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print(f"[PHASE 0] offline gen: target={args.n_adv_total} samples  "
          f"max_attempts/entry={args.max_attempts_per_entry}")
    print("=" * 72)
    victim.model.eval()
    t0 = time.time()
    adv_items, gen_stats = generate_adv_for_epoch(
        generator=generator, ecgtwin=ecgtwin, budget=budget,
        ref_items_by_center=ref_items_by_center, z_hist=z_hist,
        device=args.device, best_of_k=args.best_of_k,
        max_attempts_per_entry=args.max_attempts_per_entry,
    )
    gen_elapsed = time.time() - t0
    accept_rate = gen_stats["accepted"] / max(1, gen_stats["attempted"])
    print(f"[PHASE 0] accepted={gen_stats['accepted']}  attempted={gen_stats['attempted']}  "
          f"rate={accept_rate:.2f}  entries_completed={gen_stats['entries_completed']}  "
          f"elapsed={gen_elapsed:.0f}s")

    if accept_rate < 0.20:
        print(f"[PHASE 0 WARN] accept rate {accept_rate:.2f} < 0.20 — "
              f"acceptance range [{args.accept_prob_low},{args.accept_prob_high}] "
              f"may be too tight for baseline victim. Consider relaxing to [0.45, 0.65].")

    print("[PHASE 0] injecting augmix + filling buffer...")
    augmix_inject_and_buffer(
        adv_items=adv_items, ecgtwin_encoder_wrapper=ecgtwin,
        buffer=buffer, augmix_width=args.augmix_width,
        augmix_severity=args.augmix_severity, soft_labels=args.soft_labels,
        device=args.device, adv_label_mode=args.adv_label_mode,
    )
    print(f"[PHASE 0] buffer filled: {len(buffer)} fixed adv samples (no eviction)")

    # Snapshot visualization before training
    if adv_items and len(buffer) > 0:
        try:
            recent = buffer.ecg_list[-min(4, len(buffer)):]
            save_adv_viz(
                adv_items[:4], recent,
                save_path=os.path.join(args.output_dir, "sample_adv_offline.png"),
            )
        except Exception as e:
            print(f"  [warn] viz failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Training (FIXED buffer, no regen, no budget refresh)
    # ══════════════════════════════════════════════════════════════════════
    best_avg_auroc = baseline_quick["avg_macro_auroc"]
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save(victim.model.state_dict(), best_ckpt_path)   # bootstrap

    log: Dict[str, Any] = {
        "args": vars(args),
        "baseline_quick_eval": baseline_quick,
        "phase0_gen_stats": gen_stats,
        "phase0_accept_rate": round(accept_rate, 3),
        "phase0_buffer_size": len(buffer),
        "phase0_gen_elapsed_s": round(gen_elapsed, 1),
        "budget": [{"center": e["center"], "class": e["class"],
                    "n_alloc": e["n_alloc"], "baseline_auroc": e["baseline_auroc"]}
                   for e in budget],
        "epochs": [],
    }
    log_path = os.path.join(args.output_dir, "training_log.json")

    print("\n" + "=" * 72)
    print(f"[PHASE 1] {args.n_epochs} epochs, buffer frozen at {len(buffer)}")
    print("=" * 72)
    for epoch in range(1, args.n_epochs + 1):
        epoch_t0 = time.time()

        train_loader = _build_mixed_loader(train_ds, roundtrip_ds, buffer, args)

        # Train
        train_loss = train_one_epoch_tierM(
            victim.model, train_loader, optimizer, criterion, args.device,
            grad_clip=1.0, ewa_params=ewa_params,
            anchor_lambda=args.anchor_lambda, ewa_decay=args.ewa_decay,
        )
        scheduler.step()

        # Val loss on PTBXL val
        victim.model.eval()
        val_losses = []
        with torch.no_grad():
            for sigs, labels in val_loader:
                sigs = sigs.to(args.device)
                labels = labels.to(args.device)
                vl = criterion(victim.model(sigs), labels).mean()
                val_losses.append(vl.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

        elapsed = time.time() - epoch_t0
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "buffer_size": len(buffer),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "time_s": round(elapsed, 1),
        }
        print(f"Ep {epoch:2d}/{args.n_epochs} | train={train_loss:.4f} val={val_loss:.4f} | "
              f"{elapsed:.0f}s")

        # Periodic quick-eval for best-model selection
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.n_epochs)
        if do_eval:
            qe = quick_eval(victim.model, quick_subset, args.device)
            entry["quick_eval"] = qe
            print(f"   quick eval: avg AUROC={qe['avg_macro_auroc']}  "
                  f"AUPRC={qe['avg_macro_auprc']}")
            for c, info in qe["per_center"].items():
                print(f"      {c}: AUROC={info['macro_auroc']}  AUPRC={info['macro_auprc']}")
            if qe["avg_macro_auroc"] > best_avg_auroc:
                best_avg_auroc = qe["avg_macro_auroc"]
                torch.save(victim.model.state_dict(), best_ckpt_path)
                print(f"   ** saved best @ ep{epoch}: avg AUROC {best_avg_auroc}")
                entry["best_update"] = True

        log["epochs"].append(entry)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2, default=str)

    # ── Final summary ───────────────────────────────────────────────────────
    final_result = {
        "args": vars(args),
        "baseline_quick_eval": baseline_quick,
        "phase0_gen_stats": gen_stats,
        "phase0_accept_rate": round(accept_rate, 3),
        "phase0_buffer_size": len(buffer),
        "best_avg_macro_auroc": best_avg_auroc,
        "n_epochs_run": len(log["epochs"]),
        "last_epoch_quick_eval": log["epochs"][-1].get("quick_eval") if log["epochs"] else None,
    }
    with open(os.path.join(args.output_dir, "train_result.json"), "w") as f:
        json.dump(final_result, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"OFFLINE training done. best_avg_macro_auroc={best_avg_auroc}")
    print(f"Best ckpt: {best_ckpt_path}")
    print(f"Run full eval:")
    print(f"  /root/miniforge3/envs/ECGTwin/bin/python \\")
    print(f"    scripts/crosscenter_tierM/eval_crosscenter_tierM.py \\")
    print(f"    --model_dir {args.output_dir}")


if __name__ == "__main__":
    main()
