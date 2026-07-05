"""Offline Super5 PGD-AT from an ECGTwin synthetic latent pool.

This script implements the PTB-XL-only experiment described in
docs/pipelines/ptbxl_ecgtwin_offline_pgd_at_pipeline.md:

  1. load a frozen ECGTwin synthetic latent pool (.latent.npz),
  2. generate an offline adversarial buffer in ECGTwin VAE latent space,
  3. fine-tune the Super5 EfficientNet1DV2 baseline on real PTB-XL plus the
     adversarial buffer,
  4. evaluate PTB-XL fold9/fold10 macro AUROC/AUPRC.

The adversarial labels are target-only masked labels: the intended class is 1,
all other classes are -1 and ignored by the BCE loss. This keeps same-label
latent attacks from forcing incorrect negatives in multilabel PTB-XL semantics.
"""

from __future__ import annotations

import argparse
import json
import math
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
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path("/root/autodl-tmp/models/DeepECG/notebooks")))

from adversarial.adv_validation import compute_asr, compute_semantic_gate  # noqa: E402
from adversarial.efficientnet_victim_tierM import (  # noqa: E402
    EfficientNetVictimTierM,
    TIERM_INPUT_LENGTH,
)
from adversarial.latent_hull_pgd import LatentHullPGDGenerator  # noqa: E402
from adversarial.pgd_advdiff import PGDAdvDiffGenerator  # noqa: E402
from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from ecg_adv_gen.training.tierm_online_adv import (  # noqa: E402
    QualityAwareBufferTierM as QualityAwareBuffer,
    train_one_epoch_tierM as train_one_epoch_masked_bce,
)
from ecg_adv_gen.runner.synth_online_at_super5 import (  # noqa: E402
    SameLabelLatentIndex,
    load_synth_pool,
    push_adv_to_buffer,
    run_pgd_on_synth_pool,
    set_all_seeds,
)
from ecg_adv_gen.labels import CLASS_NAMES_SUPER5, NUM_SUPER5, get_super5_scheme  # noqa: E402
from ecg_adv_gen.data.ptbxl import get_ptbxl_labels_for_scheme, preprocess_ptbxl_all  # noqa: E402
from ecg_adv_gen.evaluation import compute_macro_auroc_auprc  # noqa: E402
from ecg_adv_gen.training import PTBXLDatasetScheme, compute_pos_weight, evaluate  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402


DEFAULT_SUPER5_CKPT = "/root/autodl-tmp/triple_labels/super5/best_model.pt"
DEFAULT_PTBXL_RAW = "/root/autodl-tmp/ptbxl/raw100.npy"
DEFAULT_PTBXL_CSV = "/root/autodl-tmp/ptbxl/ptbxl_database.csv"
DEFAULT_PTBXL_PREP = "/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy"
DEFAULT_LABEL_CACHE = "/root/autodl-tmp/triple_labels/super5/ptbxl_labels"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def masked_bce_scalar(pos_weight: torch.Tensor):
    def criterion(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = (labels >= 0).float()
        labels_safe = labels.clamp(min=0.0)
        per_elem = nn.functional.binary_cross_entropy_with_logits(
            logits, labels_safe, pos_weight=pos_weight, reduction="none"
        )
        return (per_elem * mask).sum() / mask.sum().clamp_min(1.0)

    return criterion


@torch.no_grad()
def logits_for_signals_ct(
    victim: EfficientNetVictimTierM,
    signals_ct: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    victim.eval()
    out: List[np.ndarray] = []
    for i in range(0, signals_ct.shape[0], batch_size):
        x = torch.from_numpy(signals_ct[i:i + batch_size]).float().to(device)
        logits = victim.compute_logits_from_ecg(x)
        out.append(logits.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def build_efficientnet_super5(device: str) -> EfficientNet1DV2:
    model = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=NUM_SUPER5,
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    ).to(device)
    return model


def load_super5_model(weight_path: str, device: str) -> EfficientNet1DV2:
    model = build_efficientnet_super5(device)
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    return model


def build_ptbxl_datasets(args: argparse.Namespace) -> Tuple[PTBXLDatasetScheme, PTBXLDatasetScheme, PTBXLDatasetScheme, np.ndarray, np.ndarray, np.ndarray]:
    scheme = get_super5_scheme()
    if args.split_json:
        all_idx, all_labels, _ = get_ptbxl_labels_for_scheme(
            args.ptbxl_csv, scheme, args.label_cache_prefix, folds=None
        )
        with open(args.split_json, "r") as f:
            split = json.load(f)
        train_idx = [int(i) for i in split["train_indices"]]
        val_idx = [int(i) for i in split["val_indices"]]
        test_idx = [int(i) for i in split["test_indices"]]
        max_idx = len(all_idx) - 1
        for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            if any(i < 0 or i > max_idx for i in idxs):
                raise ValueError(f"{name} split has index outside [0,{max_idx}]")
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]
        test_labels = all_labels[test_idx]
        print(f"[split] custom PTB-XL split loaded: {args.split_json}", flush=True)
    else:
        train_idx, train_labels, _ = get_ptbxl_labels_for_scheme(
            args.ptbxl_csv, scheme, args.label_cache_prefix, folds=list(range(1, 9))
        )
        val_idx, val_labels, _ = get_ptbxl_labels_for_scheme(
            args.ptbxl_csv, scheme, args.label_cache_prefix, folds=[9]
        )
        test_idx, test_labels, _ = get_ptbxl_labels_for_scheme(
            args.ptbxl_csv, scheme, args.label_cache_prefix, folds=[10]
        )

    all_sig = preprocess_ptbxl_all(
        args.ptbxl_raw,
        args.ptbxl_prep,
        preprocess_mode=args.preprocess_mode,
        norm_mode=args.norm_mode,
    )
    train_signals = np.asarray(all_sig[train_idx])
    val_signals = np.asarray(all_sig[val_idx])
    test_signals = np.asarray(all_sig[test_idx])

    if args.smoke and args.smoke_ptbxl_limit > 0:
        rng = np.random.default_rng(args.seed)

        def _take(signals: np.ndarray, labels: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
            if signals.shape[0] <= limit:
                return signals, labels
            pick = rng.choice(np.arange(signals.shape[0]), size=limit, replace=False)
            pick.sort()
            return signals[pick], labels[pick]

        train_signals, train_labels = _take(train_signals, train_labels, args.smoke_ptbxl_limit)
        val_signals, val_labels = _take(val_signals, val_labels, max(200, args.smoke_ptbxl_limit // 4))
        test_signals, test_labels = _take(test_signals, test_labels, max(200, args.smoke_ptbxl_limit // 4))

    train_ds = PTBXLDatasetScheme(train_signals, train_labels, crop_len=args.crop_len, mode="train")
    val_ds = PTBXLDatasetScheme(val_signals, val_labels, crop_len=args.crop_len, mode="eval")
    test_ds = PTBXLDatasetScheme(test_signals, test_labels, crop_len=args.crop_len, mode="eval")
    return train_ds, val_ds, test_ds, train_labels, val_labels, test_labels


def balanced_pick_indices(labels: np.ndarray, n_total: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = labels.astype(np.float32, copy=False)
    n_classes = labels.shape[1]
    quotas = np.full(n_classes, n_total // n_classes, dtype=np.int64)
    quotas[: n_total % n_classes] += 1
    picks: List[np.ndarray] = []
    for c in range(n_classes):
        pool = np.where(labels[:, c] > 0.5)[0]
        if pool.size == 0 or quotas[c] <= 0:
            continue
        replace = pool.size < quotas[c]
        cls_pick = rng.choice(pool, size=int(quotas[c]), replace=replace)
        picks.append(cls_pick.astype(np.int64))
    if not picks:
        raise ValueError("No class-positive samples found in synthetic labels")
    out = np.concatenate(picks, axis=0)
    rng.shuffle(out)
    return out


def load_class_trust(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {c: 1.0 for c in CLASS_NAMES_SUPER5}
    with open(path, "r") as f:
        raw = json.load(f)
    if "class_trust" in raw:
        raw = raw["class_trust"]
    return {c: float(raw.get(c, 1.0)) for c in CLASS_NAMES_SUPER5}


def save_buffer_npz(path: str, buffer: QualityAwareBuffer, meta: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    signals = torch.stack(buffer.ecg_list).numpy().astype(np.float32) if buffer.ecg_list else np.empty((0, 12, 250), dtype=np.float32)
    labels = torch.stack(buffer.label_list).numpy().astype(np.float32) if buffer.label_list else np.empty((0, NUM_SUPER5), dtype=np.float32)
    scores = np.asarray(buffer.score_list, dtype=np.float32)
    np.savez_compressed(out, signals_ct_250=signals, labels=labels, scores=scores, meta_json=json.dumps(meta, default=_json_default))


def load_buffer_npz(path: str, max_size: int) -> QualityAwareBuffer:
    data = np.load(path, allow_pickle=True)
    signals = data["signals_ct_250"]
    labels = data["labels"]
    scores = data["scores"] if "scores" in data.files else np.ones((signals.shape[0],), dtype=np.float32)
    buffer = QualityAwareBuffer(max_size=max(max_size, int(signals.shape[0])))
    for sig, lbl, score in zip(signals, labels, scores):
        buffer.add_one(torch.from_numpy(sig).float(), torch.from_numpy(lbl).float(), float(score))
    return buffer


def make_mixed_loader(
    train_ds: PTBXLDatasetScheme,
    adv_ds: TensorDataset,
    args: argparse.Namespace,
) -> DataLoader:
    combo = ConcatDataset([train_ds, adv_ds])
    real_weight = np.ones(len(train_ds), dtype=np.float64) * float(args.ptbxl_weight) / max(len(train_ds), 1)
    adv_weight = np.ones(len(adv_ds), dtype=np.float64) * float(args.adv_weight) / max(len(adv_ds), 1)
    weights = np.concatenate([real_weight, adv_weight], axis=0)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_ds), replacement=True)
    kwargs: Dict[str, Any] = dict(
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.num_workers >= 0,
        drop_last=True,
    )
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    return DataLoader(combo, **kwargs)


def make_eval_loader(ds: PTBXLDatasetScheme, args: argparse.Namespace) -> DataLoader:
    kwargs: Dict[str, Any] = dict(
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.num_workers >= 0,
    )
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    return DataLoader(ds, **kwargs)


def build_pgd_generator(
    args: argparse.Namespace,
    wrapper: ECGTwinWrapper,
    victim: EfficientNetVictimTierM,
) -> PGDAdvDiffGenerator:
    if args.attack_mode == "latent_hull":
        return LatentHullPGDGenerator(
            ecgtwin_wrapper=wrapper,
            victim=victim,
            epsilon=args.pgd_eps,
            hull_lambda=args.hull_lambda,
            hull_steps=args.hull_steps,
            hull_lr=args.hull_lr,
            weight_mode=args.hull_weight_mode,
            dirichlet_alpha=args.hull_dirichlet_alpha,
            device=args.device,
        )
    return PGDAdvDiffGenerator(
        ecgtwin_wrapper=wrapper,
        victim=victim,
        epsilon=args.pgd_eps,
        K_pgd=args.pgd_K,
        alpha=args.pgd_alpha,
        delta_init_scale=args.delta_init_scale,
        device=args.device,
    )


def build_or_load_adv_buffer(args: argparse.Namespace) -> Tuple[QualityAwareBuffer, Dict[str, Any]]:
    if args.offline_adv_npz and os.path.exists(args.offline_adv_npz) and not args.rebuild_adv:
        print(f"[buffer] loading existing offline buffer: {args.offline_adv_npz}", flush=True)
        buffer = load_buffer_npz(args.offline_adv_npz, args.qab_size)
        return buffer, {"loaded_from": args.offline_adv_npz, "n_loaded": len(buffer)}

    print(f"[synth] loading latent pool: {args.synth_npz}", flush=True)
    synth_latents, synth_labels, center_name, source_meta = load_synth_pool(args.synth_npz)
    if synth_labels.shape[1] != NUM_SUPER5:
        raise ValueError(f"Expected Super5 labels with C=5, got {synth_labels.shape}")
    print(f"[synth] latents={synth_latents.shape} labels={synth_labels.shape} center={center_name}", flush=True)
    class_counts = {
        c: int((synth_labels[:, j] > 0.5).sum())
        for j, c in enumerate(CLASS_NAMES_SUPER5)
    }
    print(f"[synth] class counts: {class_counts}", flush=True)

    picked = balanced_pick_indices(
        synth_labels,
        n_total=min(args.n_adv_total, max(args.n_adv_total, 1)),
        seed=args.seed,
    )
    if args.smoke:
        picked = picked[: min(len(picked), args.n_adv_total)]
    print(f"[attack] selected {len(picked)} anchors for offline PGD", flush=True)

    print("[attack] loading ECGTwin decoder/wrapper and Super5 victim", flush=True)
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)
    victim = EfficientNetVictimTierM(
        weight_path=args.init_ckpt,
        device=args.device,
        ecgtwin_wrapper=wrapper,
        num_classes=NUM_SUPER5,
        crop_len=args.crop_len,
    )
    pgd_gen = build_pgd_generator(args, wrapper, victim)
    latent_hull_index = None
    if args.attack_mode == "latent_hull":
        latent_hull_index = SameLabelLatentIndex(
            synth_latents,
            synth_labels,
            label_mode=args.hull_label_mode,
            seed=args.seed,
        )
        print(f"[attack] latent-hull pools: {latent_hull_index.class_sizes()}", flush=True)

    class_trust = load_class_trust(args.class_trust)
    print(f"[buffer] class_trust={class_trust}", flush=True)
    buffer = QualityAwareBuffer(max_size=args.qab_size)
    rng = np.random.default_rng(args.seed)
    attack_logs: List[Dict[str, Any]] = []

    chunk_size = max(args.offline_chunk_size, args.pgd_batch)
    t0 = time.time()
    for chunk_id, start in enumerate(range(0, len(picked), chunk_size), start=1):
        chunk_pick = picked[start:start + chunk_size]
        adv_signals, anchor_signals, target_oh, delta_stats = run_pgd_on_synth_pool(
            pgd_gen=pgd_gen,
            synth_latents=synth_latents,
            synth_labels=synth_labels,
            K_anchor=len(chunk_pick),
            pgd_batch=args.pgd_batch,
            rng=rng,
            device=args.device,
            picked_indices=chunk_pick,
            attack_mode=args.attack_mode,
            latent_hull_index=latent_hull_index,
            hull_M=args.hull_M,
        )
        if adv_signals.shape[0] == 0:
            continue

        asr_info = compute_asr(
            victim=victim,
            signals_ct_1000=adv_signals,
            labels_multi_hot=target_oh,
            device=args.device,
            batch_size=args.eval_batch_size,
        )
        sem_info = {"PASS": True, "fail_reasons": []}
        if not args.skip_semantic_gate:
            sem_info = compute_semantic_gate(
                adv_signals_ct_1000=adv_signals,
                anchor_signals_ct_1000=anchor_signals,
                einthoven_p95_max=args.einthoven_p95_max,
            )

        gate_pass = bool(sem_info.get("PASS", False)) or args.allow_semantic_gate_fail
        pushed = {"n_pushed": 0, "n_dropped_by_trust": 0}
        if gate_pass:
            logits = logits_for_signals_ct(victim, adv_signals, args.device, args.eval_batch_size)
            pushed = push_adv_to_buffer(
                buffer=buffer,
                adv_signals_ct=adv_signals,
                target_one_hot=target_oh,
                victim_logits=logits,
                crop_len=args.crop_len,
                class_trust=class_trust,
                boundary_prob_min=args.boundary_prob_min,
                boundary_prob_max=args.boundary_prob_max,
            )

        entry = {
            "chunk_id": chunk_id,
            "n_attack": int(adv_signals.shape[0]),
            "n_buffer": int(len(buffer)),
            "delta": delta_stats,
            "asr_overall": float(asr_info.get("asr_overall", float("nan"))),
            "asr_PASS": bool(asr_info.get("PASS", False)),
            "semantic_PASS": bool(sem_info.get("PASS", False)),
            "semantic_fail_reasons": sem_info.get("fail_reasons", []),
            "pushed": pushed,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        attack_logs.append(entry)
        print(
            f"[attack] chunk {chunk_id} n={entry['n_attack']} "
            f"asr={entry['asr_overall']:.3f} sem={entry['semantic_PASS']} "
            f"pushed={pushed['n_pushed']} buffer={len(buffer)}",
            flush=True,
        )

        if len(buffer) >= args.n_adv_total:
            break

    if len(buffer) == 0:
        raise RuntimeError("Offline adversarial buffer is empty; lower gates or inspect ECGTwin/victim path.")

    meta = {
        "synth_npz": args.synth_npz,
        "center_name": center_name,
        "class_counts": class_counts,
        "source_meta": {
            "source_names": source_meta.get("source_names", []),
            "has_source_metadata": bool(source_meta.get("has_source_metadata", False)),
        },
        "attack_mode": args.attack_mode,
        "n_requested": args.n_adv_total,
        "n_buffer": len(buffer),
        "args": vars(args),
        "attack_logs": attack_logs,
    }
    if args.offline_adv_npz:
        save_buffer_npz(args.offline_adv_npz, buffer, meta)
        print(f"[buffer] saved {len(buffer)} samples -> {args.offline_adv_npz}", flush=True)
    with open(Path(args.output_dir) / "offline_attack_log.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    return buffer, meta


def train_offline_at(args: argparse.Namespace, buffer: QualityAwareBuffer, buffer_meta: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device(args.device)
    train_ds, val_ds, test_ds, train_labels, _, _ = build_ptbxl_datasets(args)
    adv_ds = buffer.to_dataset()
    if adv_ds is None:
        raise RuntimeError("Cannot train without adv buffer samples")
    print(f"[data] PTBXL train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} adv={len(adv_ds)}", flush=True)

    val_loader = make_eval_loader(val_ds, args)
    test_loader = make_eval_loader(test_ds, args)

    model = load_super5_model(args.init_ckpt, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] loaded Super5 EfficientNet1DV2 from {args.init_ckpt} params={n_params:,}", flush=True)

    pos_weight_np = compute_pos_weight(train_labels, NUM_SUPER5, clip_max=args.pos_weight_clip_max)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=device)
    criterion_none = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    criterion_eval = masked_bce_scalar(pos_weight)
    print("[loss] pos_weight: " + ", ".join(f"{c}={w:.2f}" for c, w in zip(CLASS_NAMES_SUPER5, pos_weight_np)), flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.n_epochs, 1), eta_min=args.lr * 0.05)
    ewa_params = [p.detach().clone() for p in model.parameters()] if args.anchor_lambda > 0 else None

    best_val = -math.inf
    best_epoch = 0
    patience = 0
    log: List[Dict[str, Any]] = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.n_epochs + 1):
        if args.rescore_interval > 0 and epoch > 1 and (epoch - 1) % args.rescore_interval == 0:
            print("[buffer] rescoring quality-aware buffer", flush=True)
            buffer.rescore(model, device=args.device)
            adv_ds = buffer.to_dataset()
        assert adv_ds is not None

        train_loader = make_mixed_loader(train_ds, adv_ds, args)
        t0 = time.time()
        train_loss = train_one_epoch_masked_bce(
            model,
            train_loader,
            optimizer,
            criterion_none,
            args.device,
            grad_clip=args.grad_clip,
            ewa_params=ewa_params,
            anchor_lambda=args.anchor_lambda,
            ewa_decay=args.ewa_decay,
        )
        scheduler.step()
        val_loss, vy_true, vy_score = evaluate(model, val_loader, criterion_eval, args.device)
        val_metrics = compute_macro_auroc_auprc(vy_true, vy_score, CLASS_NAMES_SUPER5)
        elapsed = time.time() - t0
        entry = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_macro_auroc": float(val_metrics["macro_auroc"]),
            "val_macro_auprc": float(val_metrics["macro_auprc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_sec": round(elapsed, 1),
            "adv_buffer_size": int(len(buffer)),
        }
        log.append(entry)

        improved = entry["val_macro_auroc"] > best_val
        if improved:
            best_val = entry["val_macro_auroc"]
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience += 1
        marker = " *" if improved else ""
        print(
            f"[train] ep {epoch:03d}/{args.n_epochs} loss={train_loss:.5f} "
            f"val_auroc={entry['val_macro_auroc']:.4f} "
            f"val_auprc={entry['val_macro_auprc']:.4f} "
            f"lr={entry['lr']:.6g} {elapsed:.0f}s{marker}",
            flush=True,
        )
        with open(out_dir / "training_log.json", "w") as f:
            json.dump(log, f, indent=2, default=_json_default)
        if patience >= args.patience:
            print(f"[train] early stop at epoch {epoch} patience={args.patience}", flush=True)
            break

    best_state = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_state)
    test_loss, ty_true, ty_score = evaluate(model, test_loader, criterion_eval, args.device)
    test_metrics = compute_macro_auroc_auprc(ty_true, ty_score, CLASS_NAMES_SUPER5)
    val_loss, vy_true, vy_score = evaluate(model, val_loader, criterion_eval, args.device)
    final_val_metrics = compute_macro_auroc_auprc(vy_true, vy_score, CLASS_NAMES_SUPER5)

    result = {
        "scheme": "super5",
        "class_names": list(CLASS_NAMES_SUPER5),
        "best_epoch": best_epoch,
        "best_val_macro_auroc": float(best_val),
        "final_val_loss": float(val_loss),
        "final_val_metrics": final_val_metrics,
        "test_loss": float(test_loss),
        "test_macro_auroc": float(test_metrics["macro_auroc"]),
        "test_macro_auprc": float(test_metrics["macro_auprc"]),
        "test_per_class": test_metrics["per_class"],
        "adv_buffer_size": int(len(buffer)),
        "buffer_meta": buffer_meta,
        "pos_weight": pos_weight_np.tolist(),
        "config": vars(args),
    }
    with open(out_dir / "train_result.json", "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(
        f"[done] test AUROC={result['test_macro_auroc']:.4f} "
        f"AUPRC={result['test_macro_auprc']:.4f} saved={out_dir}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--synth_npz", required=True, help="ECGTwin synthetic latent npz, usually *.latent.npz")
    p.add_argument("--init_ckpt", default=DEFAULT_SUPER5_CKPT)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--offline_adv_npz", default="", help="Save/load offline adv buffer here")
    p.add_argument("--rebuild_adv", action="store_true", help="Ignore existing --offline_adv_npz and regenerate")

    p.add_argument("--ptbxl_raw", default=DEFAULT_PTBXL_RAW)
    p.add_argument("--ptbxl_csv", default=DEFAULT_PTBXL_CSV)
    p.add_argument("--ptbxl_prep", default=DEFAULT_PTBXL_PREP)
    p.add_argument("--label_cache_prefix", default=DEFAULT_LABEL_CACHE)
    p.add_argument("--split_json", default="", help="Optional custom PTB-XL split JSON")
    p.add_argument(
        "--preprocess_mode",
        default="legacy_ecgfounder_filter",
        choices=["minimal_resample", "legacy_ecgfounder_filter", "raw_for_generation_or_digital"],
    )
    p.add_argument("--norm_mode", default="per_sample_global", choices=["per_sample_global", "none"])

    p.add_argument("--n_adv_total", type=int, default=10000)
    p.add_argument("--offline_chunk_size", type=int, default=256)
    p.add_argument("--attack_mode", choices=["pgd", "latent_hull"], default="pgd")
    p.add_argument("--pgd_eps", type=float, default=2.0)
    p.add_argument("--pgd_K", type=int, default=10)
    p.add_argument("--pgd_batch", type=int, default=32)
    p.add_argument("--pgd_alpha", type=float, default=None)
    p.add_argument("--delta_init_scale", type=float, default=0.1)
    p.add_argument("--hull_M", type=int, default=10)
    p.add_argument("--hull_lambda", type=float, default=0.25)
    p.add_argument("--hull_steps", type=int, default=5)
    p.add_argument("--hull_lr", type=float, default=0.3)
    p.add_argument("--hull_weight_mode", choices=["optimized", "one_hot", "uniform", "dirichlet"], default="optimized")
    p.add_argument("--hull_dirichlet_alpha", type=float, default=1.0)
    p.add_argument("--hull_label_mode", choices=["primary", "exact"], default="primary")

    p.add_argument("--class_trust", default=None)
    p.add_argument("--boundary_prob_min", type=float, default=0.0)
    p.add_argument("--boundary_prob_max", type=float, default=1.0)
    p.add_argument("--skip_semantic_gate", action="store_true")
    p.add_argument("--allow_semantic_gate_fail", action="store_true")
    p.add_argument("--einthoven_p95_max", type=float, default=0.5)

    p.add_argument("--ptbxl_weight", type=float, default=1.0)
    p.add_argument("--adv_weight", type=float, default=0.5)
    p.add_argument("--qab_size", type=int, default=20000)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--anchor_lambda", type=float, default=0.05)
    p.add_argument("--ewa_decay", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--rescore_interval", type=int, default=3)
    p.add_argument("--pos_weight_clip_max", type=float, default=50.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--crop_len", type=int, default=TIERM_INPUT_LENGTH)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_ptbxl_limit", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("/root/autodl-tmp/tmp").mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available() and "cuda" in args.device:
        torch.backends.cudnn.benchmark = True
    with open(Path(args.output_dir) / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=_json_default)

    buffer, meta = build_or_load_adv_buffer(args)
    train_offline_at(args, buffer, meta)


if __name__ == "__main__":
    main()
