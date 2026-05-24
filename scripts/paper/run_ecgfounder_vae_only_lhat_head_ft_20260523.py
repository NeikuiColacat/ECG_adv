#!/usr/bin/env python3
"""ECGFounder Super5 head fine-tune with VAE-only latent-hull online AT.

This runner keeps ECGFounder's encoder frozen and updates only the Super5
linear head.  It is the ECGFounder counterpart to the EfficientNet1DV2
VAE-only real-anchor latent-hull AT route:

  target-center real ECGTwin VAE latents
  -> same-label latent-hull adversarial search against the current ECGFounder
     Super5 head
  -> decode ECGTwin VAE latent to ECG
  -> frozen ECGFounder encoder feature
  -> train only the 5-class head with PTB-XL source features, target real
     features, and online adversarial features.

No DiT synthetic ECG and no center prompt token samples are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        str(_MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")),
    )
)
ECGFOUNDER_ROOT = Path(os.environ.get("ECGFOUNDER_ROOT", str(DATA_ROOT / "ecgfounder")))
for path in [str(REPO_ROOT), str(ECGFOUNDER_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from net1d import Net1D  # noqa: E402
from physionet2021_dataset import EXPECTED_LEADS, TARGET_POINTS  # noqa: E402

from adversarial.latent_hull_pgd import LatentHullPGDGenerator  # noqa: E402
from scripts.paper.eval_ecgfounder_super5_zero_shot_20260517 import TARGET_CENTERS  # noqa: E402
from scripts.paper.run_ecgfounder_linear_probe_super5_20260517 import (  # noqa: E402
    REF_ROOT,
    compute_metrics,
    evaluate_pn2021_views,
)
from scripts.pgd_cross_center.synth_online_at_super5 import (  # noqa: E402
    SameLabelLatentIndex,
    StratifiedPoolWalker,
    build_anchor_preserving_soft_labels,
)
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, SUPER5_TO_IDX  # noqa: E402
from scripts.triple_labels.train_ptbxl import compute_pos_weight, masked_bce_with_logits  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402


_V5_LINEAR_PROBE_DIR = (
    DATA_ROOT
    / "paper_foundation_baselines_20260522/ecgfounder_linear_probe_v5_seed42_official"
)
_LEGACY_LINEAR_PROBE_DIR = (
    DATA_ROOT / "paper_foundation_baselines_20260517/ecgfounder_linear_probe_super5"
)
DEFAULT_LINEAR_PROBE_DIR = Path(
    os.environ.get(
        "ECGFOUNDER_LINEAR_PROBE_DIR",
        str(_V5_LINEAR_PROBE_DIR if _V5_LINEAR_PROBE_DIR.exists() else _LEGACY_LINEAR_PROBE_DIR),
    )
)
DEFAULT_OUT_DIR = DATA_ROOT / "paper_ecgfounder_vae_only_lhat_headft_20260523"
CHECKPOINT = ECGFOUNDER_ROOT / "checkpoint/12_lead_ECGFounder.pth"
REAL_ROOTS = [
    DATA_ROOT / "ecgtwin_prompt_token_super5/real_anchor_selected_v2",
    DATA_ROOT / "ecgtwin_prompt_token_super5/real_anchor_selected_v1",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_ecgfounder_feature_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150,
        return_features=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


class ECGFounderHeadVictim(nn.Module):
    """Differentiable ECGTwin-latent -> ECGFounder-head logits wrapper."""

    def __init__(
        self,
        feature_model: nn.Module,
        head: nn.Module,
        ecgtwin: ECGTwinWrapper,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.feature_model = feature_model
        self.head = head
        self.ecgtwin = ecgtwin
        self.device = device
        self.num_classes = len(CLASS_NAMES_SUPER5)
        self.model = nn.ModuleDict({"feature_model": self.feature_model, "head": self.head})

    @staticmethod
    def _global_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp(min=eps)
        return (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)

    def _decode_latent_differentiable(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent / 0.18215
        decoder = self.ecgtwin.decoder
        for module in decoder:
            x = module(x)
        return x.transpose(1, 2)

    def _ecgtwin_latent_to_ecg1000(self, latent: torch.Tensor) -> torch.Tensor:
        ecg_tc = self._decode_latent_differentiable(latent)
        ecg_ct = ecg_tc.transpose(-1, -2)
        ecg_ct = ecg_ct[:, ECGTWIN_TO_PTBXL_INDICES, :]
        ecg_ct = torch.clamp(ecg_ct, min=-3.0, max=3.0)
        ecg_ct = F.interpolate(ecg_ct, size=1000, mode="linear", align_corners=True)
        return self._global_zscore(ecg_ct)

    def features_from_ecg1000(self, ecg_ct_1000: torch.Tensor, grad: bool) -> torch.Tensor:
        x = F.interpolate(ecg_ct_1000, size=TARGET_POINTS, mode="linear", align_corners=True)
        x = self._global_zscore(x)
        if grad:
            _, features = self.feature_model(x)
        else:
            with torch.no_grad():
                _, features = self.feature_model(x)
        return features

    def forward_from_latent_to_logits(self, latent: torch.Tensor) -> torch.Tensor:
        ecg_ct = self._ecgtwin_latent_to_ecg1000(latent)
        features = self.features_from_ecg1000(ecg_ct, grad=True)
        return self.head(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class ResidualAdapterHead(nn.Module):
    """Frozen or trainable source linear head plus a zero-init residual adapter."""

    def __init__(
        self,
        base_head: nn.Linear,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        scale: float = 1.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_head = base_head
        self.scale = float(scale)
        in_dim = int(base_head.in_features)
        out_dim = int(base_head.out_features)
        hidden_dim = max(1, int(hidden_dim))
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, out_dim),
        )
        final = self.adapter[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if freeze_base:
            for p in self.base_head.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_head(x) + self.scale * self.adapter(x)


def real_anchor_base(center: str) -> Path:
    for root in REAL_ROOTS:
        base = root / center / f"{center}_real_k500_seed42"
        if base.with_suffix(".latent.npz").exists():
            return base
    searched = ", ".join(str(root / center / f"{center}_real_k500_seed42") for root in REAL_ROOTS)
    raise FileNotFoundError(f"missing real-anchor files for {center}; searched: {searched}")


def load_anchor_pool(center: str, k: int, seed: int, pn_payload: dict[str, np.ndarray]) -> dict[str, Any]:
    base = real_anchor_base(center)
    with np.load(base.with_suffix(".latent.npz"), allow_pickle=True) as d:
        latents_all = d["latents"].astype(np.float32)
        record_ids_all = d["record_ids"].astype(str)

    pn_centers = pn_payload["centers"].astype(str)
    pn_record_ids = pn_payload["record_ids"].astype(str)
    pn_labels = pn_payload["labels"].astype(np.float32)
    key_to_label = {
        (str(c), str(rid)): pn_labels[i]
        for i, (c, rid) in enumerate(zip(pn_centers, pn_record_ids))
    }
    labels_all = []
    keep = []
    for i, rid in enumerate(record_ids_all):
        label = key_to_label.get((center, str(rid)))
        if label is None:
            continue
        labels_all.append(label)
        keep.append(i)
    if not keep:
        raise RuntimeError(f"{center}: no anchor record ids matched PN2021 feature cache")
    latents_all = latents_all[np.asarray(keep, dtype=np.int64)]
    record_ids_all = record_ids_all[np.asarray(keep, dtype=np.int64)]
    labels_all = np.stack(labels_all).astype(np.float32)

    if k > len(labels_all):
        raise ValueError(f"{center}: K={k} exceeds matched anchors {len(labels_all)}")
    if k < len(labels_all):
        rng = np.random.default_rng(seed)
        primary = np.argmax(labels_all, axis=1)
        pieces = []
        counts = np.bincount(primary, minlength=len(CLASS_NAMES_SUPER5))
        raw = counts / max(counts.sum(), 1) * k
        alloc = np.floor(raw).astype(int)
        for cls in range(len(CLASS_NAMES_SUPER5)):
            if counts[cls] > 0 and alloc[cls] == 0:
                alloc[cls] = 1
        while alloc.sum() > k:
            cls = int(np.argmax(alloc))
            alloc[cls] -= 1
        while alloc.sum() < k:
            for cls in np.argsort(-(raw - np.floor(raw))):
                if alloc.sum() >= k:
                    break
                if alloc[cls] < counts[cls]:
                    alloc[cls] += 1
        for cls, n_take in enumerate(alloc):
            idx = np.where(primary == cls)[0]
            if n_take > 0 and len(idx) > 0:
                pieces.append(rng.permutation(idx)[: min(n_take, len(idx))])
        selected = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        if len(selected) < k:
            rest = np.setdiff1d(np.arange(len(labels_all)), selected, assume_unique=False)
            selected = np.concatenate([selected, rng.permutation(rest)[: k - len(selected)]])
        selected = np.sort(selected[:k])
    else:
        selected = np.arange(len(labels_all), dtype=np.int64)

    latents = latents_all[selected]
    labels = labels_all[selected]
    record_ids = record_ids_all[selected]
    present = labels.sum(axis=0) > 0
    classes_in_scope = [c for c, ok in zip(CLASS_NAMES_SUPER5, present) if ok]
    return {
        "latents": latents,
        "labels": labels,
        "record_ids": record_ids,
        "classes_in_scope": classes_in_scope,
        "label_counts": dict(zip(CLASS_NAMES_SUPER5, labels.sum(axis=0).astype(int).tolist())),
        "source_base": str(base),
    }


def ref_ids_for_all_centers(center: str, selected_ids: np.ndarray) -> dict[str, set[str]]:
    ref_ids = {}
    try:
        from scripts.paper.eval_ecgfounder_super5_zero_shot_20260517 import load_ref_ids

        ref_ids = load_ref_ids(REF_ROOT, TARGET_CENTERS)
    except Exception:
        ref_ids = {c: set() for c in TARGET_CENTERS}
    ref_ids[center] = set(str(x) for x in selected_ids)
    return ref_ids


@torch.no_grad()
def predict_head(head: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    head.eval()
    scores = []
    loader = DataLoader(torch.from_numpy(features).float(), batch_size=batch_size, shuffle=False)
    for x in loader:
        logits = head(x.to(device)).cpu().numpy()
        scores.append(1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50))))
    return np.concatenate(scores, axis=0)


def eval_ptbxl_fold10(head: nn.Module, ptbxl: dict[str, np.ndarray], device: torch.device, batch_size: int) -> dict:
    folds = ptbxl["folds"].astype(np.int64)
    mask = folds == 10
    scores = predict_head(head, ptbxl["features"][mask].astype(np.float32), batch_size, device)
    return compute_metrics(ptbxl["labels"][mask].astype(np.float32), scores, min_pos=1)


def eval_pn(
    head: nn.Module,
    pn: dict[str, np.ndarray],
    ref_ids: dict[str, set[str]],
    device: torch.device,
    batch_size: int,
    report_drop_all_zero: bool = False,
) -> dict:
    scores = predict_head(head, pn["features"].astype(np.float32), batch_size, device)
    return evaluate_pn2021_views(
        pn["labels"].astype(np.float32),
        scores,
        pn["centers"].astype(str),
        pn["record_ids"].astype(str),
        ref_ids,
        report_drop_all_zero=report_drop_all_zero,
    )


def make_weighted_loader(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    adv_x: np.ndarray,
    adv_y: np.ndarray,
    args: argparse.Namespace,
) -> DataLoader:
    target_class_weights = parse_class_weight_string(args.target_class_sample_weights)
    adv_class_weights = parse_class_weight_string(args.adv_class_sample_weights)

    def sample_weights(labels: np.ndarray, base_weight: float, class_weights: dict[str, float]) -> list[float]:
        if not class_weights:
            return [base_weight] * len(labels)
        weights_vec = np.ones((len(CLASS_NAMES_SUPER5),), dtype=np.float32)
        for cls, value in class_weights.items():
            weights_vec[SUPER5_TO_IDX[cls]] = float(value)
        pos = labels > 0.5
        out = []
        for row in pos:
            if row.any():
                out.append(base_weight * float(np.max(weights_vec[row])))
            else:
                out.append(base_weight)
        return out

    datasets = []
    weights = []
    if args.source_weight > 0:
        stream = torch.zeros((len(source_x),), dtype=torch.long)
        ds = TensorDataset(torch.from_numpy(source_x).float(), torch.from_numpy(source_y).float(), stream)
        datasets.append(ds)
        weights.extend([args.source_weight] * len(ds))
    if args.target_real_weight > 0:
        stream = torch.ones((len(target_x),), dtype=torch.long)
        ds = TensorDataset(torch.from_numpy(target_x).float(), torch.from_numpy(target_y).float(), stream)
        datasets.append(ds)
        weights.extend(sample_weights(target_y, args.target_real_weight, target_class_weights))
    if len(adv_x) > 0 and args.adv_weight > 0:
        stream = torch.full((len(adv_x),), 2, dtype=torch.long)
        ds = TensorDataset(torch.from_numpy(adv_x).float(), torch.from_numpy(adv_y).float(), stream)
        datasets.append(ds)
        weights.extend(sample_weights(adv_y, args.adv_weight, adv_class_weights))
    combined = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(combined, batch_size=args.batch_size, sampler=sampler, drop_last=False)


def parse_class_weight_string(raw: str | None) -> dict[str, float]:
    """Parse CLASS=weight comma-separated strings."""
    if not raw:
        return {}
    out: dict[str, float] = {}
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"bad class weight item {item!r}; expected CLASS=weight")
        cls, value = item.split("=", 1)
        cls = cls.strip().upper()
        if cls not in SUPER5_TO_IDX:
            raise ValueError(f"unknown class {cls!r}; valid={list(CLASS_NAMES_SUPER5)}")
        out[cls] = float(value)
    return out


def parse_class_list(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        items = str(raw).replace(",", " ").split()
    out = []
    for item in items:
        cls = item.strip().upper()
        if not cls:
            continue
        if cls not in SUPER5_TO_IDX:
            raise ValueError(f"unknown class {cls!r}; valid={list(CLASS_NAMES_SUPER5)}")
        out.append(cls)
    return out


def pairwise_rank_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_indices: list[int],
    *,
    class_weights: dict[int, float] | None = None,
    temperature: float,
) -> torch.Tensor:
    losses = []
    weights = []
    temp = max(float(temperature), 1e-6)
    for cls_idx in class_indices:
        y = labels[:, cls_idx]
        pos = logits[y > 0.5, cls_idx]
        neg = logits[y <= 0.0, cls_idx]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        margins = (pos[:, None] - neg[None, :]) / temp
        losses.append(F.softplus(-margins).mean())
        weights.append(float((class_weights or {}).get(cls_idx, 1.0)))
    if not losses:
        return logits.new_zeros(())
    loss_t = torch.stack(losses)
    weight_t = torch.tensor(weights, dtype=loss_t.dtype, device=loss_t.device)
    return (loss_t * weight_t).sum() / weight_t.sum().clamp_min(1e-6)


def train_one_center(
    center: str,
    args: argparse.Namespace,
    ptbxl: dict[str, np.ndarray],
    pn: dict[str, np.ndarray],
    feature_model: nn.Module,
    ecgtwin: ECGTwinWrapper,
    out_dir: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    run_dir = out_dir / "runs" / (
        f"{center}_K{args.k}_M{args.hull_m}_lam{str(args.hull_lambda).replace('.', 'p')}"
        f"_ep{args.epochs}_seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "eval_result.json"
    if result_path.exists() and not args.force:
        with result_path.open() as f:
            return json.load(f)

    pool = load_anchor_pool(center, args.k, args.seed, pn)
    selected_ids = pool["record_ids"].astype(str)
    ref_ids = ref_ids_for_all_centers(center, selected_ids)

    pn_mask = (pn["centers"].astype(str) == center) & np.asarray(
        [str(rid) in set(selected_ids) for rid in pn["record_ids"].astype(str)],
        dtype=bool,
    )
    target_x = pn["features"][pn_mask].astype(np.float32)
    target_y = pn["labels"][pn_mask].astype(np.float32)
    if len(target_x) != len(selected_ids):
        print(f"[warn] {center}: selected_ids={len(selected_ids)} target_feature_rows={len(target_x)}", flush=True)

    folds = ptbxl["folds"].astype(np.int64)
    source_mask = np.isin(folds, np.arange(1, 9))
    if args.source_train_limit > 0:
        src_idx = np.nonzero(source_mask)[0][: args.source_train_limit]
    else:
        src_idx = np.nonzero(source_mask)[0]
    source_x = ptbxl["features"][src_idx].astype(np.float32)
    source_y = ptbxl["labels"][src_idx].astype(np.float32)

    base_head = nn.Linear(source_x.shape[1], len(CLASS_NAMES_SUPER5)).to(device)
    base_head.load_state_dict(torch.load(Path(args.linear_probe_dir) / "best_head.pt", map_location=device))
    source_teacher_head = nn.Linear(source_x.shape[1], len(CLASS_NAMES_SUPER5)).to(device)
    source_teacher_head.load_state_dict(torch.load(Path(args.linear_probe_dir) / "best_head.pt", map_location=device))
    source_teacher_head.eval()
    for p in source_teacher_head.parameters():
        p.requires_grad_(False)
    if args.head_type == "linear":
        head = base_head
    else:
        head = ResidualAdapterHead(
            base_head=base_head,
            hidden_dim=args.adapter_hidden,
            dropout=args.adapter_dropout,
            scale=args.adapter_scale,
            freeze_base=args.freeze_base_head,
        ).to(device)
    if args.init_head_path:
        init_path = Path(args.init_head_path)
        print(f"[init] loading head state from {init_path}", flush=True)
        head.load_state_dict(torch.load(init_path, map_location=device))
    target_teacher_head = None
    if args.target_logit_anchor_weight > 0:
        if not args.target_logit_anchor_path:
            raise ValueError("--target_logit_anchor_weight requires --target_logit_anchor_path")
        teacher_path = Path(args.target_logit_anchor_path)
        print(f"[teacher] loading target logit anchor from {teacher_path}", flush=True)
        target_teacher_head = nn.Linear(source_x.shape[1], len(CLASS_NAMES_SUPER5)).to(device)
        target_teacher_head.load_state_dict(torch.load(teacher_path, map_location=device))
        target_teacher_head.eval()
        for p in target_teacher_head.parameters():
            p.requires_grad_(False)
    head_anchor = {
        name: param.detach().clone()
        for name, param in head.named_parameters()
    }
    head_anchor_denom = torch.zeros((), device=device)
    for param in head_anchor.values():
        head_anchor_denom = head_anchor_denom + torch.sum(param ** 2)
    head_anchor_denom = head_anchor_denom.clamp(min=1e-12)
    victim = ECGFounderHeadVictim(feature_model=feature_model, head=head, ecgtwin=ecgtwin, device=device).to(device)
    pgd_gen = LatentHullPGDGenerator(
        ecgtwin_wrapper=ecgtwin,
        victim=victim,
        epsilon=args.pgd_eps,
        hull_lambda=args.hull_lambda,
        hull_steps=args.hull_steps,
        hull_lr=args.hull_lr,
        weight_mode=args.hull_weight_mode,
        dirichlet_alpha=args.hull_dirichlet_alpha,
        device=args.device,
    )
    for p in feature_model.parameters():
        p.requires_grad_(False)
    if isinstance(head, ResidualAdapterHead):
        for p in head.adapter.parameters():
            p.requires_grad_(True)
        if args.freeze_base_head:
            for p in head.base_head.parameters():
                p.requires_grad_(False)
    else:
        for p in head.parameters():
            p.requires_grad_(True)

    classes_in_scope = list(args.classes_in_scope or pool["classes_in_scope"])
    classes_in_scope = [c for c in classes_in_scope if pool["label_counts"].get(c, 0) > 0]
    if not classes_in_scope:
        raise RuntimeError(f"{center}: no classes in scope after label-count filtering")
    walker = StratifiedPoolWalker(
        labels_one_hot=pool["labels"],
        classes_in_scope=classes_in_scope,
        class_to_idx=SUPER5_TO_IDX,
        seed=args.seed,
    )
    index = SameLabelLatentIndex(
        pool["latents"],
        pool["labels"],
        label_mode=args.hull_label_mode,
        seed=args.seed,
        include_self=args.hull_include_anchor,
    )

    if args.pos_weight_data == "source":
        pos_weight_labels = source_y
    elif args.pos_weight_data == "target":
        pos_weight_labels = target_y
    elif args.pos_weight_data == "source_target":
        pos_weight_labels = np.concatenate([source_y, target_y], axis=0)
    else:
        raise ValueError(f"unsupported pos_weight_data={args.pos_weight_data!r}")
    pos_weight = torch.tensor(
        compute_pos_weight(pos_weight_labels, len(CLASS_NAMES_SUPER5), clip_max=50.0),
        dtype=torch.float32,
        device=device,
    )
    class_loss_weights_map = parse_class_weight_string(args.class_loss_weights)
    class_loss_weight = torch.ones((len(CLASS_NAMES_SUPER5),), dtype=torch.float32, device=device)
    for cls, value in class_loss_weights_map.items():
        class_loss_weight[SUPER5_TO_IDX[cls]] = float(value)
    rank_loss_class_indices = [
        SUPER5_TO_IDX[cls]
        for cls in parse_class_list(args.target_rank_loss_classes)
    ]
    rank_loss_class_weights = {
        SUPER5_TO_IDX[cls]: float(value)
        for cls, value in parse_class_weight_string(args.target_rank_loss_class_weights).items()
    }

    def criterion(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not class_loss_weights_map:
            return masked_bce_with_logits(logits, y, pos_weight)
        mask = (y >= 0).float()
        labels_safe = torch.where(mask.bool(), y, torch.zeros_like(y))
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels_safe,
            pos_weight=pos_weight,
            reduction="none",
        )
        weighted = bce * mask * class_loss_weight.view(1, -1)
        denom = (mask * class_loss_weight.view(1, -1)).sum().clamp_min(1.0)
        return weighted.sum() / denom

    trainable_head_params = [p for p in head.parameters() if p.requires_grad]
    if not trainable_head_params:
        param_state = ", ".join(f"{name}:{param.requires_grad}" for name, param in head.named_parameters())
        raise RuntimeError(
            "No trainable head parameters. Check --head_type/--freeze_base_head. "
            f"Parameter states: {param_state}"
        )
    opt = torch.optim.AdamW(trainable_head_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    baseline_views = eval_pn(
        head,
        pn,
        ref_ids,
        device,
        args.eval_batch_size,
        report_drop_all_zero=args.report_drop_all_zero_pn2021,
    )
    baseline_ptbxl = eval_ptbxl_fold10(head, ptbxl, device, args.eval_batch_size)
    def selection_score(target_row: dict[str, Any], ptbxl_row: dict[str, Any]) -> float:
        target_auprc = float(target_row["macro_auprc"])
        source_auprc = float(ptbxl_row["macro_auprc"])
        if args.selection_metric == "target_auprc":
            return target_auprc
        if args.selection_metric == "target_plus_source_auprc":
            return target_auprc + args.source_selection_weight * source_auprc
        if args.selection_metric == "target_source_hmean_auprc":
            denom = target_auprc + source_auprc
            return (2.0 * target_auprc * source_auprc / denom) if denom > 0 else -float("inf")
        if args.selection_metric == "target_under_source_floor":
            if source_auprc < args.source_auprc_floor:
                return target_auprc - args.source_floor_penalty * (args.source_auprc_floor - source_auprc)
            return target_auprc
        raise ValueError(f"unsupported selection_metric={args.selection_metric}")

    best_score = (
        -float("inf")
        if args.selection_metric == "last_epoch"
        else selection_score(
            baseline_views[center]["per_center"][center],
            baseline_ptbxl,
        )
    )
    best_epoch = 0
    torch.save(head.state_dict(), run_dir / "best_head.pt")

    logs = []
    for epoch in range(1, args.epochs + 1):
        victim.eval()
        adv_features, adv_labels = [], []
        delta_norms = []
        if not args.disable_adv_stream and args.adv_weight > 0 and args.k_anchor > 0:
            per_cls = max(1, args.k_anchor // len(classes_in_scope))
            k_per_cls = {c: per_cls for c in classes_in_scope}
            for i in range(args.k_anchor - per_cls * len(classes_in_scope)):
                k_per_cls[classes_in_scope[i % len(classes_in_scope)]] += 1
            drawn = walker.sample(k_per_cls)
            picks = np.concatenate([drawn[c] for c in classes_in_scope if drawn[c].size > 0])
            if picks.size > 1:
                np.random.default_rng(args.seed + epoch).shuffle(picks)
            for i in range(0, len(picks), args.pgd_batch):
                batch_idx = picks[i:i + args.pgd_batch]
                z = torch.from_numpy(pool["latents"][batch_idx]).float().to(device)
                y = torch.from_numpy(pool["labels"][batch_idx]).float().to(device)
                cand = torch.from_numpy(index.candidates_for(batch_idx, args.hull_m)).float().to(device)
                x_adv, delta = pgd_gen.attack_from_latent(z, y, candidate_latents=cand)
                with torch.no_grad():
                    feats = victim.features_from_ecg1000(x_adv, grad=False)
                adv_features.append(feats.float().cpu().numpy())
                if args.hull_mix_label_mode == "anchor":
                    adv_labels.append(pool["labels"][batch_idx].astype(np.float32))
                elif args.hull_mix_label_mode == "anchor_soft":
                    cand_idx = getattr(index, "last_candidate_indices", None)
                    weights_t = getattr(pgd_gen, "last_weights", None)
                    if cand_idx is None or weights_t is None:
                        raise RuntimeError(
                            "latent-hull soft labels require candidate indices and weights"
                        )
                    weights_np = weights_t.numpy().astype(np.float32, copy=False)
                    cand_labels = pool["labels"][cand_idx]
                    adv_labels.append(
                        build_anchor_preserving_soft_labels(
                            pool["labels"][batch_idx].astype(np.float32, copy=False),
                            cand_labels,
                            weights_np,
                            lambda_y=args.hull_label_lambda_y,
                            positive_value=args.hull_label_positive,
                            negative_floor=args.hull_label_negative_floor,
                            new_class_cap=args.hull_label_new_class_cap,
                        )
                    )
                else:
                    raise ValueError(
                        "hull_mix_label_mode must be anchor|anchor_soft, "
                        f"got {args.hull_mix_label_mode!r}"
                    )
                delta_norms.extend(delta.detach().flatten(1).norm(dim=1).cpu().tolist())
        adv_x = (
            np.concatenate(adv_features, axis=0).astype(np.float32)
            if adv_features
            else np.empty((0, source_x.shape[1]), dtype=np.float32)
        )
        adv_y = (
            np.concatenate(adv_labels, axis=0).astype(np.float32)
            if adv_labels
            else np.empty((0, len(CLASS_NAMES_SUPER5)), dtype=np.float32)
        )

        loader = make_weighted_loader(source_x, source_y, target_x, target_y, adv_x, adv_y, args)
        head.train()
        losses = []
        for x, y, stream in loader:
            x = x.to(device)
            y = y.to(device)
            stream = stream.to(device)
            opt.zero_grad(set_to_none=True)
            logits = head(x)
            loss = criterion(logits, y)
            if args.target_rank_loss_weight > 0 and rank_loss_class_indices:
                if args.target_rank_loss_stream == "target_real":
                    target_mask = stream == 1
                elif args.target_rank_loss_stream == "target_adv":
                    target_mask = stream == 2
                else:
                    target_mask = stream > 0
                if bool(target_mask.any()):
                    loss = loss + args.target_rank_loss_weight * pairwise_rank_loss(
                        logits[target_mask],
                        y[target_mask],
                        rank_loss_class_indices,
                        class_weights=rank_loss_class_weights,
                        temperature=args.target_rank_loss_temperature,
                    )
            if args.source_logit_anchor_weight > 0:
                source_mask = stream == 0
                if bool(source_mask.any()):
                    with torch.no_grad():
                        teacher_logits = source_teacher_head(x[source_mask])
                    source_logit_loss = F.mse_loss(logits[source_mask], teacher_logits)
                    loss = loss + args.source_logit_anchor_weight * source_logit_loss
            if args.target_logit_anchor_weight > 0:
                target_mask = stream > 0
                if bool(target_mask.any()):
                    assert target_teacher_head is not None
                    with torch.no_grad():
                        teacher_logits = target_teacher_head(x[target_mask])
                    target_logit_loss = F.mse_loss(logits[target_mask], teacher_logits)
                    loss = loss + args.target_logit_anchor_weight * target_logit_loss
            if args.head_l2_anchor > 0:
                anchor_loss = torch.zeros((), device=device)
                for name, param in head.named_parameters():
                    delta = param - head_anchor[name]
                    if args.head_anchor_mode == "mean":
                        anchor_loss = anchor_loss + torch.mean(delta ** 2)
                    else:
                        anchor_loss = anchor_loss + torch.sum(delta ** 2)
                if args.head_anchor_mode == "relative":
                    anchor_loss = anchor_loss / head_anchor_denom
                loss = loss + args.head_l2_anchor * anchor_loss
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "n_adv": int(len(adv_x)),
            "classes_in_scope": classes_in_scope,
            "delta_mean": float(np.mean(delta_norms)) if delta_norms else None,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            views = eval_pn(
                head,
                pn,
                ref_ids,
                device,
                args.eval_batch_size,
                report_drop_all_zero=args.report_drop_all_zero_pn2021,
            )
            ptbxl_fold10 = eval_ptbxl_fold10(head, ptbxl, device, args.eval_batch_size)
            target = views[center]["per_center"][center]
            entry["target_macro_auroc"] = target["macro_auroc"]
            entry["target_macro_auprc"] = target["macro_auprc"]
            entry["ptbxl_macro_auroc"] = ptbxl_fold10["macro_auroc"]
            entry["ptbxl_macro_auprc"] = ptbxl_fold10["macro_auprc"]
            cur_score = (
                float(epoch)
                if args.selection_metric == "last_epoch"
                else selection_score(target, ptbxl_fold10)
            )
            entry["selection_metric"] = args.selection_metric
            entry["selection_score"] = cur_score
            if cur_score > best_score:
                best_score = float(cur_score)
                best_epoch = epoch
                torch.save(head.state_dict(), run_dir / "best_head.pt")
                entry["best_update"] = True
            else:
                entry["best_update"] = False
            print(
                f"[{center}] ep={epoch:03d} loss={entry['train_loss']:.4f} "
                f"target={target['macro_auroc']:.4f}/{target['macro_auprc']:.4f} "
                f"ptbxl={ptbxl_fold10['macro_auroc']:.4f}/{ptbxl_fold10['macro_auprc']:.4f} "
                f"best_ep={best_epoch}",
                flush=True,
            )
        else:
            print(f"[{center}] ep={epoch:03d} loss={entry['train_loss']:.4f}", flush=True)
        logs.append(entry)
        with (run_dir / "training_log.json").open("w") as f:
            json.dump(logs, f, indent=2)

    head.load_state_dict(torch.load(run_dir / "best_head.pt", map_location=device))
    final_views = eval_pn(
        head,
        pn,
        ref_ids,
        device,
        args.eval_batch_size,
        report_drop_all_zero=args.report_drop_all_zero_pn2021,
    )
    final_ptbxl = eval_ptbxl_fold10(head, ptbxl, device, args.eval_batch_size)
    result = {
        "method": (
            "ECGFounder frozen encoder + real-only target-center head adaptation"
            if args.disable_adv_stream
            else "ECGFounder frozen encoder + VAE-only real-anchor latent-hull online AT head fine-tune"
        ),
        "center": center,
        "class_names": list(CLASS_NAMES_SUPER5),
        "K": int(args.k),
        "selected_ref_record_ids": selected_ids.tolist(),
        "classes_in_scope": classes_in_scope,
        "label_counts": pool["label_counts"],
        "class_loss_weights": class_loss_weights_map,
        "head_type": args.head_type,
        "baseline_pn2021_views": baseline_views,
        "baseline_ptbxl_fold10": baseline_ptbxl,
        "final_pn2021_views": final_views,
        "final_ptbxl_fold10": final_ptbxl,
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "config": vars(args),
        "anchor_source_base": pool["source_base"],
        "preprocess": {
            "ecgfounder_input": f"12 x {TARGET_POINTS}",
            "lead_order": EXPECTED_LEADS,
            "latent_decode_path": "ECGTwin VAE latent -> PTB-XL-order 1000 samples -> linear interpolate to 5000 -> per-sample global z-score",
        },
    }
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    return result


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ecgfounder_vae_only_lhat_headft_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "center",
            "baseline_target_auroc",
            "baseline_target_auprc",
            "lhat_target_auroc",
            "lhat_target_auprc",
            "baseline_drop_all_zero_target_auroc",
            "baseline_drop_all_zero_target_auprc",
            "lhat_drop_all_zero_target_auroc",
            "lhat_drop_all_zero_target_auprc",
            "delta_target_auroc",
            "delta_target_auprc",
            "baseline_ptbxl_auroc",
            "baseline_ptbxl_auprc",
            "lhat_ptbxl_auroc",
            "lhat_ptbxl_auprc",
            "eval_path",
        ])
        for r in rows:
            c = r["center"]
            b = r["baseline_pn2021_views"][c]["per_center"][c]
            m = r["final_pn2021_views"][c]["per_center"][c]
            b_ptb = r["baseline_ptbxl_fold10"]
            m_ptb = r["final_ptbxl_fold10"]
            run_dir = out_dir / "runs" / (
                f"{c}_K{r['K']}_M{r['config']['hull_m']}_lam{str(r['config']['hull_lambda']).replace('.', 'p')}"
                f"_ep{r['config']['epochs']}_seed{r['config']['seed']}"
            )
            w.writerow([
                c,
                b["macro_auroc"],
                b["macro_auprc"],
                m["macro_auroc"],
                m["macro_auprc"],
                b.get("drop_all_zero_macro_auroc"),
                b.get("drop_all_zero_macro_auprc"),
                m.get("drop_all_zero_macro_auroc"),
                m.get("drop_all_zero_macro_auprc"),
                None if b["macro_auroc"] is None or m["macro_auroc"] is None else m["macro_auroc"] - b["macro_auroc"],
                None if b["macro_auprc"] is None or m["macro_auprc"] is None else m["macro_auprc"] - b["macro_auprc"],
                b_ptb["macro_auroc"],
                b_ptb["macro_auprc"],
                m_ptb["macro_auroc"],
                m_ptb["macro_auprc"],
                str(run_dir / "eval_result.json"),
            ])
    print(f"[summary] wrote {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=["cpsc_2018"])
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--linear_probe_dir", default=str(DEFAULT_LINEAR_PROBE_DIR))
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--k_anchor", type=int, default=300)
    p.add_argument("--hull_m", type=int, default=20)
    p.add_argument("--hull_lambda", type=float, default=0.15)
    p.add_argument("--hull_steps", type=int, default=5)
    p.add_argument("--hull_lr", type=float, default=0.25)
    p.add_argument("--hull_weight_mode", choices=["optimized", "one_hot", "uniform", "dirichlet"], default="optimized")
    p.add_argument("--hull_dirichlet_alpha", type=float, default=1.0)
    p.add_argument("--hull_label_mode", choices=["primary", "exact", "compatible"], default="primary")
    p.add_argument("--hull_mix_label_mode", choices=["anchor", "anchor_soft"], default="anchor")
    p.add_argument("--hull_label_lambda_y", type=float, default=0.5)
    p.add_argument("--hull_label_positive", type=float, default=0.95)
    p.add_argument("--hull_label_negative_floor", type=float, default=0.0)
    p.add_argument("--hull_label_new_class_cap", type=float, default=0.5)
    p.add_argument("--hull_include_anchor", action="store_true")
    p.add_argument("--pgd_eps", type=float, default=2.0)
    p.add_argument("--pgd_batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--eval_batch_size", type=int, default=4096)
    p.add_argument(
        "--head_type",
        choices=["linear", "residual_adapter"],
        default="linear",
        help="linear updates the Super5 head directly; residual_adapter keeps a source head plus trainable zero-init adapter.",
    )
    p.add_argument("--adapter_hidden", type=int, default=128)
    p.add_argument("--adapter_dropout", type=float, default=0.0)
    p.add_argument("--adapter_scale", type=float, default=1.0)
    p.add_argument(
        "--freeze_base_head",
        action="store_true",
        help="For residual_adapter, freeze the PTB-XL source linear head and train only adapter parameters.",
    )
    p.add_argument(
        "--init_head_path",
        default="",
        help=(
            "Optional checkpoint for initializing the current head before online AT. "
            "Useful for second-stage experiments, e.g. real-only adapter -> VAE-only adversarial refinement."
        ),
    )
    p.add_argument("--source_weight", type=float, default=1.0)
    p.add_argument("--target_real_weight", type=float, default=20.0)
    p.add_argument("--adv_weight", type=float, default=10.0)
    p.add_argument(
        "--source_logit_anchor_weight",
        type=float,
        default=0.0,
        help="MSE penalty that keeps adapted logits close to the frozen PTB-XL source head on source-stream batches.",
    )
    p.add_argument(
        "--target_logit_anchor_path",
        default="",
        help="Optional linear-head checkpoint used as a target/adv stream logit teacher.",
    )
    p.add_argument(
        "--target_logit_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "MSE penalty that keeps target-real and target-adv logits close to "
            "--target_logit_anchor_path. Useful for testing whether a K-shot "
            "head can stabilize VAE-online AT without score ensembling."
        ),
    )
    p.add_argument(
        "--class_loss_weights",
        default="",
        help="Optional comma-separated class loss weights, e.g. MI=3,HYP=3.",
    )
    p.add_argument(
        "--target_class_sample_weights",
        default="",
        help="Optional positive-class sampling multipliers for target-real stream.",
    )
    p.add_argument(
        "--adv_class_sample_weights",
        default="",
        help="Optional positive-class sampling multipliers for online adversarial stream.",
    )
    p.add_argument(
        "--target_rank_loss_weight",
        type=float,
        default=0.0,
        help="Optional pairwise ranking-loss weight on target-real and target-adv streams.",
    )
    p.add_argument(
        "--target_rank_loss_classes",
        nargs="*",
        default=[],
        help="Classes for target pairwise ranking loss, e.g. HYP STTC.",
    )
    p.add_argument(
        "--target_rank_loss_class_weights",
        default="",
        help="Optional class weights inside target ranking loss, e.g. HYP=1,STTC=0.1.",
    )
    p.add_argument(
        "--target_rank_loss_stream",
        choices=["target_real", "target_adv", "target_adv_real"],
        default="target_adv_real",
        help="Which target stream receives pairwise ranking loss.",
    )
    p.add_argument(
        "--target_rank_loss_temperature",
        type=float,
        default=1.0,
        help="Temperature for target pairwise ranking margins.",
    )
    p.add_argument(
        "--head_l2_anchor",
        type=float,
        default=0.0,
        help="L2 penalty to keep the adapted Super5 head close to the PTB-XL linear-probe head.",
    )
    p.add_argument(
        "--head_anchor_mode",
        choices=["relative", "mean", "sum"],
        default="relative",
        help="Reduction for --head_l2_anchor. relative uses ||w-w0||^2 / ||w0||^2.",
    )
    p.add_argument("--source_train_limit", type=int, default=0)
    p.add_argument(
        "--pos_weight_data",
        choices=["source", "target", "source_target"],
        default="source",
        help=(
            "Label distribution used for BCE pos_weight. The historical default "
            "is source; target can be useful for K-shot center adaptation where "
            "the target class prior differs from PTB-XL."
        ),
    )
    p.add_argument(
        "--disable_adv_stream",
        action="store_true",
        help="Train with source + target real features only; no VAE latent-hull adversarial stream.",
    )
    p.add_argument(
        "--report_drop_all_zero_pn2021",
        action="store_true",
        help="Also report PN2021 metrics after excluding rows with no positive Super5 label.",
    )
    p.add_argument(
        "--selection_metric",
        choices=[
            "last_epoch",
            "target_auprc",
            "target_plus_source_auprc",
            "target_source_hmean_auprc",
            "target_under_source_floor",
        ],
        default="target_auprc",
        help=(
            "Metric used to save best_head.pt during online AT. "
            "Use last_epoch for fixed-horizon runs without target-test checkpoint selection."
        ),
    )
    p.add_argument("--source_selection_weight", type=float, default=0.25)
    p.add_argument("--source_auprc_floor", type=float, default=0.79)
    p.add_argument("--source_floor_penalty", type=float, default=5.0)
    p.add_argument("--classes_in_scope", nargs="*", default=[])
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    linear_dir = Path(args.linear_probe_dir)
    ptbxl_path = linear_dir / f"ptbxl_ecgfounder_features_{args.preprocess_policy}.npz"
    pn_path = linear_dir / f"pn2021_ecgfounder_features_{args.preprocess_policy}.npz"
    if not ptbxl_path.exists():
        ptbxl_path = linear_dir / "ptbxl_ecgfounder_features.npz"
    if not pn_path.exists():
        pn_path = linear_dir / "pn2021_ecgfounder_features.npz"
    if not ptbxl_path.exists() or not pn_path.exists():
        raise FileNotFoundError(f"missing ECGFounder feature cache: {ptbxl_path}, {pn_path}")
    if not (linear_dir / "best_head.pt").exists():
        raise FileNotFoundError(f"missing PTB-XL Super5 head: {linear_dir / 'best_head.pt'}")

    ptbxl_npz = np.load(ptbxl_path, allow_pickle=True)
    pn_npz = np.load(pn_path, allow_pickle=True)
    ptbxl = {k: ptbxl_npz[k] for k in ptbxl_npz.files}
    pn = {k: pn_npz[k] for k in pn_npz.files}

    device = torch.device(args.device)
    print("[setup] loading ECGFounder encoder", flush=True)
    feature_model = build_ecgfounder_feature_model(Path(args.checkpoint), device)
    print("[setup] loading ECGTwin VAE decoder", flush=True)
    ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)

    rows = []
    for center in args.centers:
        result = train_one_center(center, args, ptbxl, pn, feature_model, ecgtwin, out_dir)
        rows.append(result)
        c = result["center"]
        b = result["baseline_pn2021_views"][c]["per_center"][c]
        m = result["final_pn2021_views"][c]["per_center"][c]
        print(
            f"[result] {c}: baseline={b['macro_auroc']:.4f}/{b['macro_auprc']:.4f} "
            f"lhat={m['macro_auroc']:.4f}/{m['macro_auprc']:.4f}",
            flush=True,
        )
        write_summary(rows, out_dir)
    write_summary(rows, out_dir)


if __name__ == "__main__":
    main()
