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
import hashlib
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
from torch.utils.data import DataLoader


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
from ecg_adv_gen.adaptation import (  # noqa: E402
    SameLabelLatentIndex,
    StratifiedPoolWalker,
    build_anchor_preserving_soft_labels,
    build_k500_internal_val_mask,
    build_latent_augmix_branch_signals,
    linear_warmup_value,
    split_anchor_sample_mode,
)
from ecg_adv_gen.adaptation.anchor_sampling import (  # noqa: E402
    feature_anchor_difficulty_weights,
    sample_hard_feature_anchors,
)
from ecg_adv_gen.adaptation.latent_hull_torch import initial_hull_latent  # noqa: E402
from ecg_adv_gen.data import (  # noqa: E402
    find_real_anchor_base,
    load_real_anchor_pool,
)
from ecg_adv_gen.evaluation import (  # noqa: E402
    compute_ecgfounder_lhat_selection_score,
    validate_ecgfounder_lhat_runtime_selection,
)
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, SUPER5_TO_IDX  # noqa: E402
from ecg_adv_gen.labels import pn2021_super5_label_mapping_payload  # noqa: E402
from ecg_adv_gen.models import ecgfounder_k500_head_path, ecgfounder_lhat_run_dir  # noqa: E402
from ecg_adv_gen.models.ecgfounder_heads import (  # noqa: E402
    FeatureAdapterHead,
    ResidualAdapterHead,
    clone_linear_head,
)
from ecg_adv_gen.models.ecgfounder_inference import (  # noqa: E402
    evaluate_feature_head,
    evaluate_pn2021_feature_head,
    evaluate_ptbxl_fold_head,
    predict_feature_head,
)
from ecg_adv_gen.training import (  # noqa: E402
    WeightedFeatureStream,
    attack_success_stats,
    build_weighted_feature_stream_loader,
    compute_pos_weight,
    masked_bce_with_logits,
    merge_attack_success_stats,
    pairwise_rank_loss,
)
from methods.augmix.severity import AVAILABLE_OPS, build_op  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402


_V6_LINEAR_PROBE_DIR = (
    DATA_ROOT
    / "paper_foundation_baselines_20260524/ecgfounder_linear_probe_v6_from_legacy_cache"
)
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
        str(
            _V6_LINEAR_PROBE_DIR
            if _V6_LINEAR_PROBE_DIR.exists()
            else (_V5_LINEAR_PROBE_DIR if _V5_LINEAR_PROBE_DIR.exists() else _LEGACY_LINEAR_PROBE_DIR)
        ),
    )
)
DEFAULT_OUT_DIR = DATA_ROOT / "paper_ecgfounder_vae_only_lhat_headft_v6_20260524"
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


def real_anchor_base(center: str, args: argparse.Namespace | None = None) -> Path:
    return find_real_anchor_base(
        center,
        anchor_base_root=getattr(args, "anchor_base_root", "") if args is not None else None,
        k=int(getattr(args, "k", 500)) if args is not None else 500,
        seed=int(getattr(args, "seed", 42)) if args is not None else 42,
        default_roots=REAL_ROOTS,
    )


def load_anchor_pool(
    center: str,
    k: int,
    seed: int,
    pn_payload: dict[str, np.ndarray],
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    return load_real_anchor_pool(
        center,
        k=k,
        seed=seed,
        pn_payload=pn_payload,
        anchor_base_root=getattr(args, "anchor_base_root", "") if args is not None else None,
        default_roots=REAL_ROOTS,
        class_names=CLASS_NAMES_SUPER5,
    )


def ref_ids_for_all_centers(
    center: str,
    selected_ids: np.ndarray,
    ref_root: str | Path = REF_ROOT,
) -> dict[str, set[str]]:
    ref_ids = {}
    try:
        from scripts.paper.eval_ecgfounder_super5_zero_shot_20260517 import load_ref_ids

        ref_ids = load_ref_ids(ref_root, TARGET_CENTERS)
    except Exception:
        ref_ids = {c: set() for c in TARGET_CENTERS}
    ref_ids[center] = set(str(x) for x in selected_ids)
    return ref_ids


def center_kshot_head_path(run_root: Path, center: str, k: int, seed: int) -> Path:
    path = ecgfounder_k500_head_path(run_root, center=center, k=k, seed=seed, source_k=k)
    if not path.exists():
        raise FileNotFoundError(f"missing direct head for {center} K={k} under {run_root}: {path}")
    return path


def predict_head(head: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    return predict_feature_head(head, features, batch_size=batch_size, device=device)


def eval_ptbxl_fold10(head: nn.Module, ptbxl: dict[str, np.ndarray], device: torch.device, batch_size: int) -> dict:
    return evaluate_ptbxl_fold_head(
        head,
        ptbxl,
        compute_metrics,
        fold=10,
        batch_size=batch_size,
        device=device,
        min_pos=1,
    )


def eval_pn(
    head: nn.Module,
    pn: dict[str, np.ndarray],
    ref_ids: dict[str, set[str]],
    device: torch.device,
    batch_size: int,
    report_drop_all_zero: bool = False,
) -> dict:
    return evaluate_pn2021_feature_head(
        head,
        pn,
        ref_ids,
        evaluate_pn2021_views,
        batch_size=batch_size,
        device=device,
        report_drop_all_zero=report_drop_all_zero,
    )


@torch.no_grad()
def eval_feature_subset(
    head: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    return evaluate_feature_head(
        head,
        features,
        labels,
        compute_metrics,
        batch_size=batch_size,
        device=device,
        min_pos=1,
    )


def anchor_difficulty_weights(
    head: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    mode: str,
    power: float,
    min_weight: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    return feature_anchor_difficulty_weights(
        head,
        features,
        labels,
        mode=mode,
        power=power,
        min_weight=min_weight,
        batch_size=batch_size,
        device=device,
    )


def sample_hard_anchors(
    *,
    head: nn.Module,
    pool_features: np.ndarray,
    pool_labels: np.ndarray,
    k_anchor: int,
    mode: str,
    power: float,
    min_weight: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    return sample_hard_feature_anchors(
        head=head,
        pool_features=pool_features,
        pool_labels=pool_labels,
        k_anchor=k_anchor,
        mode=mode,
        power=power,
        min_weight=min_weight,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )


def apply_augmix_op_np(sig_ct: np.ndarray, op_name: str, op_severity: int) -> np.ndarray:
    """Apply one ECG AugMix op to a channels-first 100 Hz ECG sample."""

    sig_t = torch.from_numpy(sig_ct.copy()).float()
    return build_op(op_name, int(op_severity))(sig_t).cpu().numpy().astype(np.float32, copy=False)


def summarize_latent_augmix_stats(stats_batches: list[dict[str, Any]], *, enabled: bool) -> dict[str, Any]:
    """Merge per-PGD-batch latent AugMix diagnostics for the epoch log."""

    if not enabled:
        return {"enabled": False, "n_generated": 0}
    if not stats_batches:
        return {"enabled": True, "n_generated": 0}
    out: dict[str, Any] = {
        "enabled": True,
        "n_generated": int(sum(int(item.get("n_generated", 0)) for item in stats_batches)),
    }
    for key in ["copies", "severity", "width", "depth", "alpha", "latent_weight_cap", "renorm", "clip_abs", "ops"]:
        if key in stats_batches[-1]:
            out[key] = stats_batches[-1][key]
    for key in ["latent_weight_mean", "latent_weight_max", "beta_m_mean", "chain_depth_mean"]:
        values = [
            float(item[key])
            for item in stats_batches
            if key in item and np.isfinite(float(item[key]))
        ]
        if values:
            reducer = max if key == "latent_weight_max" else np.mean
            out[key] = float(reducer(values))
    return out


def make_weighted_loader(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    adv_x: np.ndarray,
    adv_y: np.ndarray,
    args: argparse.Namespace,
    *,
    adv_weight: float | None = None,
) -> DataLoader:
    target_class_weights = parse_class_weight_string(args.target_class_sample_weights)
    adv_class_weights = parse_class_weight_string(args.adv_class_sample_weights)
    effective_adv_weight = float(args.adv_weight if adv_weight is None else adv_weight)
    streams = [
        WeightedFeatureStream(source_x, source_y, stream_id=0, base_weight=args.source_weight),
        WeightedFeatureStream(
            target_x,
            target_y,
            stream_id=1,
            base_weight=args.target_real_weight,
            class_weights=target_class_weights,
        ),
        WeightedFeatureStream(
            adv_x,
            adv_y,
            stream_id=2,
            base_weight=effective_adv_weight,
            class_weights=adv_class_weights,
        ),
    ]
    return build_weighted_feature_stream_loader(
        streams,
        batch_size=args.batch_size,
        class_to_idx=SUPER5_TO_IDX,
        num_classes=len(CLASS_NAMES_SUPER5),
    )


def scheduled_adv_weight(args: argparse.Namespace, epoch: int) -> float:
    """Linear warmup for the adversarial stream sampler weight."""
    return linear_warmup_value(
        args.adv_weight,
        epoch,
        args.adv_weight_warmup_epochs,
        start=args.adv_weight_start,
    )


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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append_checkpoint_index_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, ensure_ascii=True, default=str) + "\n")


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
    run_dir = ecgfounder_lhat_run_dir(
        out_dir,
        center=center,
        k=args.k,
        hull_m=args.hull_m,
        hull_lambda=args.hull_lambda,
        epochs=args.epochs,
        seed=args.seed,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "eval_result.json"
    if result_path.exists() and not args.force:
        with result_path.open() as f:
            return json.load(f)
    checkpoint_index_path = run_dir / "checkpoint_index.jsonl"
    if checkpoint_index_path.exists():
        checkpoint_index_path.unlink()
    selection_safety = getattr(args, "selection_safety_record", None)
    if selection_safety is None:
        selection_safety = validate_ecgfounder_lhat_runtime_selection(
            selection_source=args.selection_source,
            selection_metric=args.selection_metric,
            target_real_val_fraction=args.target_real_val_fraction,
            target_real_val_seed=args.target_real_val_seed,
            allow_pn2021_heldout_selection=getattr(args, "allow_pn2021_heldout_selection", False),
        )
    with (run_dir / "selection_safety.json").open("w", encoding="utf-8") as f:
        json.dump(selection_safety, f, indent=2, sort_keys=True)

    pool = load_anchor_pool(center, args.k, args.seed, pn, args)
    selected_ids_all = pool["record_ids"].astype(str)
    ref_ids = ref_ids_for_all_centers(center, selected_ids_all, args.ref_root)

    pn_mask = (pn["centers"].astype(str) == center) & np.asarray(
        [str(rid) in set(selected_ids_all) for rid in pn["record_ids"].astype(str)],
        dtype=bool,
    )
    target_x = pn["features"][pn_mask].astype(np.float32)
    target_y = pn["labels"][pn_mask].astype(np.float32)
    target_record_ids = pn["record_ids"][pn_mask].astype(str)
    if len(target_x) != len(selected_ids_all):
        print(f"[warn] {center}: selected_ids={len(selected_ids_all)} target_feature_rows={len(target_x)}", flush=True)
    target_feature_by_id = {
        str(rid): target_x[i].astype(np.float32, copy=False)
        for i, rid in enumerate(target_record_ids.astype(str))
    }

    target_val_x = np.empty((0, target_x.shape[1]), dtype=np.float32)
    target_val_y = np.empty((0, len(CLASS_NAMES_SUPER5)), dtype=np.float32)
    target_val_record_ids: list[str] = []
    target_train_record_ids = target_record_ids.astype(str).tolist()
    original_pool_label_counts = dict(pool["label_counts"])
    if args.selection_source == "target_real_val":
        val_mask_pool = build_k500_internal_val_mask(
            pool["labels"],
            val_fraction=args.target_real_val_fraction,
            seed=args.target_real_val_seed,
        )
        target_val_ids = set(str(x) for x in pool["record_ids"][val_mask_pool].astype(str))
        target_train_ids = set(str(x) for x in pool["record_ids"][~val_mask_pool].astype(str))
        target_val_mask = np.asarray([str(rid) in target_val_ids for rid in target_record_ids], dtype=bool)
        target_train_mask = np.asarray([str(rid) in target_train_ids for rid in target_record_ids], dtype=bool)
        target_val_x = target_x[target_val_mask].astype(np.float32)
        target_val_y = target_y[target_val_mask].astype(np.float32)
        target_val_record_ids = target_record_ids[target_val_mask].astype(str).tolist()
        target_x = target_x[target_train_mask].astype(np.float32)
        target_y = target_y[target_train_mask].astype(np.float32)
        target_train_record_ids = target_record_ids[target_train_mask].astype(str).tolist()
        for key in ("latents", "labels", "record_ids"):
            pool[key] = pool[key][~val_mask_pool]
        pool["label_counts"] = dict(
            zip(CLASS_NAMES_SUPER5, pool["labels"].sum(axis=0).astype(int).tolist())
        )
        pool["classes_in_scope"] = [
            c for c, count in pool["label_counts"].items() if int(count) > 0
        ]
        print(
            f"[setup] target-real internal selection split: "
            f"train={len(target_x)} val={len(target_val_x)} "
            f"fraction={args.target_real_val_fraction} seed={args.target_real_val_seed}",
            flush=True,
        )

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
    elif args.head_type == "residual_adapter":
        head = ResidualAdapterHead(
            base_head=base_head,
            hidden_dim=args.adapter_hidden,
            dropout=args.adapter_dropout,
            scale=args.adapter_scale,
            freeze_base=args.freeze_base_head,
        ).to(device)
    elif args.head_type == "feature_adapter":
        head = FeatureAdapterHead(
            base_head=base_head,
            hidden_dim=args.adapter_hidden,
            dropout=args.adapter_dropout,
            scale=args.adapter_scale,
            freeze_base=args.freeze_base_head,
        ).to(device)
    else:
        raise ValueError(f"unsupported head_type={args.head_type!r}")
    if args.init_base_head_from_k500_root:
        k500_head_path = center_kshot_head_path(
            Path(args.init_base_head_from_k500_root),
            center,
            k=int(args.k),
            seed=int(args.seed),
        )
        print(
            f"[init] loading center K500 direct head into base head from {k500_head_path}",
            flush=True,
        )
        k500_state = torch.load(k500_head_path, map_location=device)
        if isinstance(head, (ResidualAdapterHead, FeatureAdapterHead)):
            head.base_head.load_state_dict(k500_state)
        else:
            head.load_state_dict(k500_state)
    if args.init_head_path:
        init_path = Path(args.init_head_path)
        print(f"[init] loading head state from {init_path}", flush=True)
        head.load_state_dict(torch.load(init_path, map_location=device))
    anchor_base_head = clone_linear_head(head, source_x.shape[1], device)
    target_teacher_head = None
    if args.target_logit_anchor_weight > 0:
        if args.target_logit_anchor_source == "initial_head":
            target_teacher_head = anchor_base_head
            print("[teacher] using frozen initial/direct head as target logit anchor", flush=True)
        else:
            if not args.target_logit_anchor_path:
                raise ValueError(
                    "--target_logit_anchor_weight with --target_logit_anchor_source=path "
                    "requires --target_logit_anchor_path"
                )
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
    if isinstance(head, (ResidualAdapterHead, FeatureAdapterHead)):
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
    missing_pool_features = [
        str(rid) for rid in pool["record_ids"].astype(str)
        if str(rid) not in target_feature_by_id
    ]
    if missing_pool_features:
        raise RuntimeError(
            f"{center}: {len(missing_pool_features)} pool anchors missing cached ECGFounder features"
        )
    pool_features = np.stack(
        [target_feature_by_id[str(rid)] for rid in pool["record_ids"].astype(str)]
    ).astype(np.float32)
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
        distance_space=args.hull_neighbor_distance_space,
        neighbor_mode=args.hull_neighbor_mode,
        neighbor_pool_size=args.hull_neighbor_pool_size,
        neighbor_pool_multiplier=args.hull_neighbor_pool_multiplier,
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

    def criterion(logits: torch.Tensor, y: torch.Tensor, stream: torch.Tensor | None = None) -> torch.Tensor:
        mask = (y >= 0).float()
        labels_safe = torch.where(mask.bool(), y, torch.zeros_like(y))
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels_safe,
            pos_weight=pos_weight,
            reduction="none",
        )
        if class_loss_weights_map:
            weighted = bce * mask * class_loss_weight.view(1, -1)
            denom = (mask * class_loss_weight.view(1, -1)).sum(dim=1).clamp_min(1.0)
        else:
            weighted = bce * mask
            denom = mask.sum(dim=1).clamp_min(1.0)
        per_sample = weighted.sum(dim=1) / denom
        if stream is None:
            return per_sample.mean()
        stream_weights = torch.ones_like(per_sample)
        stream_weights = torch.where(
            stream == 0,
            torch.full_like(stream_weights, float(args.source_bce_loss_weight)),
            stream_weights,
        )
        stream_weights = torch.where(
            stream == 1,
            torch.full_like(stream_weights, float(args.target_real_bce_loss_weight)),
            stream_weights,
        )
        stream_weights = torch.where(
            stream == 2,
            torch.full_like(stream_weights, float(args.adv_bce_loss_weight)),
            stream_weights,
        )
        return (per_sample * stream_weights).sum() / stream_weights.sum().clamp_min(1e-6)

    trainable_head_params = [p for p in head.parameters() if p.requires_grad]
    if not trainable_head_params:
        param_state = ", ".join(f"{name}:{param.requires_grad}" for name, param in head.named_parameters())
        raise RuntimeError(
            "No trainable head parameters. Check --head_type/--freeze_base_head. "
            f"Parameter states: {param_state}"
        )
    opt = torch.optim.AdamW(trainable_head_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    baseline_ptbxl = eval_ptbxl_fold10(head, ptbxl, device, args.eval_batch_size)
    if args.selection_source == "target_real_val":
        if len(target_val_x) == 0:
            raise RuntimeError("target_real_val selection requested but validation split is empty")
        baseline_views = None
        baseline_selection_target = eval_feature_subset(
            head,
            target_val_x,
            target_val_y,
            device,
            args.eval_batch_size,
        )
        print(
            f"[baseline] K500-val={baseline_selection_target['macro_auroc']:.4f}/"
            f"{baseline_selection_target['macro_auprc']:.4f} "
            f"PTBXL={baseline_ptbxl['macro_auroc']:.4f}/{baseline_ptbxl['macro_auprc']:.4f}",
            flush=True,
        )
    else:
        baseline_views = eval_pn(
            head,
            pn,
            ref_ids,
            device,
            args.eval_batch_size,
            report_drop_all_zero=args.report_drop_all_zero_pn2021,
        )
        baseline_selection_target = baseline_views[center]["per_center"][center]
    def selection_score(target_row: dict[str, Any], ptbxl_row: dict[str, Any]) -> float:
        return compute_ecgfounder_lhat_selection_score(
            selection_metric=args.selection_metric,
            target_metrics=target_row,
            source_metrics=ptbxl_row,
            source_selection_weight=args.source_selection_weight,
            source_auprc_floor=args.source_auprc_floor,
            source_floor_penalty=args.source_floor_penalty,
        )

    best_score = (
        -float("inf")
        if args.selection_metric == "last_epoch"
        else selection_score(
            baseline_selection_target,
            baseline_ptbxl,
        )
    )
    best_epoch = 0
    torch.save(head.state_dict(), run_dir / "best_head.pt")
    torch.save(head.state_dict(), run_dir / "initial_head.pt")
    append_checkpoint_index_record(
        checkpoint_index_path,
        {
            "event": "initial_checkpoint",
            "epoch": 0,
            "center": center,
            "selected_as_best": True,
            "selection_source": args.selection_source,
            "selection_metric": args.selection_metric,
            "selection_score": None if not np.isfinite(best_score) else float(best_score),
            "target_metric_view": (
                "target_k500_internal_val"
                if args.selection_source == "target_real_val"
                else "pn2021_heldout"
            ),
            "target_metrics": baseline_selection_target,
            "source_metrics": baseline_ptbxl,
            "source_auprc_floor": float(args.source_auprc_floor),
            "checkpoint_path": str(run_dir / "best_head.pt"),
            "checkpoint_sha256": file_sha256(run_dir / "best_head.pt"),
            "selection_safety": selection_safety["selection_safety"],
        },
    )

    logs = []
    for epoch in range(1, args.epochs + 1):
        epoch_adv_weight = scheduled_adv_weight(args, epoch)
        victim.eval()
        adv_features, adv_labels = [], []
        delta_norms = []
        n_adv_generated = 0
        n_adv_filtered_boundary = 0
        attack_init_stats_batches: list[dict[str, float]] = []
        attack_anchor_stats_batches: list[dict[str, float]] = []
        latent_augmix_rng = np.random.default_rng(args.seed + 170000 + epoch)
        latent_augmix_stats_batches: list[dict[str, Any]] = []
        anchor_head_source, anchor_difficulty_mode = split_anchor_sample_mode(args.anchor_sample_mode)
        anchor_sample_stats: dict[str, Any] = {
            "mode": args.anchor_sample_mode,
            "head_source": anchor_head_source,
            "difficulty_mode": anchor_difficulty_mode,
            "adv_weight_effective": float(epoch_adv_weight),
        }
        if not args.disable_adv_stream and epoch_adv_weight > 0 and args.k_anchor > 0:
            if args.anchor_sample_mode == "stratified":
                per_cls = max(1, args.k_anchor // len(classes_in_scope))
                k_per_cls = {c: per_cls for c in classes_in_scope}
                for i in range(args.k_anchor - per_cls * len(classes_in_scope)):
                    k_per_cls[classes_in_scope[i % len(classes_in_scope)]] += 1
                drawn = walker.sample(k_per_cls)
                picks = np.concatenate([drawn[c] for c in classes_in_scope if drawn[c].size > 0])
                if picks.size > 1:
                    np.random.default_rng(args.seed + epoch).shuffle(picks)
                anchor_sample_stats["k_per_cls"] = k_per_cls
            else:
                if anchor_head_source == "current":
                    sample_head = head
                elif anchor_head_source == "base":
                    sample_head = anchor_base_head
                elif anchor_head_source == "target_teacher":
                    if target_teacher_head is None:
                        raise ValueError(
                            f"--anchor_sample_mode {args.anchor_sample_mode!r} requires "
                            "--target_logit_anchor_path"
                        )
                    sample_head = target_teacher_head
                else:
                    raise ValueError(f"unsupported anchor head source {anchor_head_source!r}")
                picks, hard_stats = sample_hard_anchors(
                    head=sample_head,
                    pool_features=pool_features,
                    pool_labels=pool["labels"],
                    k_anchor=args.k_anchor,
                    mode=anchor_difficulty_mode,
                    power=args.anchor_sample_power,
                    min_weight=args.anchor_sample_min_weight,
                    batch_size=args.eval_batch_size,
                    device=device,
                    seed=args.seed + epoch,
                )
                anchor_sample_stats.update(hard_stats)
            for i in range(0, len(picks), args.pgd_batch):
                batch_idx = picks[i:i + args.pgd_batch]
                z = torch.from_numpy(pool["latents"][batch_idx]).float().to(device)
                y = torch.from_numpy(pool["labels"][batch_idx]).float().to(device)
                cand = torch.from_numpy(index.candidates_for(batch_idx, args.hull_m)).float().to(device)
                with torch.no_grad():
                    anchor_x = victim._ecgtwin_latent_to_ecg1000(z)
                    anchor_logits = victim.forward_from_latent_to_logits(z)
                    z_init = initial_hull_latent(
                        z,
                        cand,
                        weight_mode=args.hull_weight_mode,
                        hull_lambda=args.hull_lambda,
                        init_logit_gap=pgd_gen.init_logit_gap,
                    )
                    init_logits = victim.forward_from_latent_to_logits(z_init)
                x_adv, delta = pgd_gen.attack_from_latent(z, y, candidate_latents=cand)
                with torch.no_grad():
                    feats = victim.features_from_ecg1000(x_adv, grad=False)
                    adv_logits = head(feats)
                    attack_init_stats_batches.append(
                        attack_success_stats(
                            init_logits,
                            adv_logits,
                            y,
                            margin=args.attack_success_margin,
                        )
                    )
                    attack_anchor_stats_batches.append(
                        attack_success_stats(
                            anchor_logits,
                            adv_logits,
                            y,
                            margin=args.attack_success_margin,
                        )
                    )
                probs = torch.sigmoid(adv_logits)
                pos_mask = y > 0.5
                pos_counts = pos_mask.sum(dim=1)
                pos_prob_mean = (probs * pos_mask.float()).sum(dim=1) / pos_counts.clamp(min=1).float()
                keep_mask = (pos_counts == 0) | (
                    (pos_prob_mean >= float(args.adv_boundary_prob_min))
                    & (pos_prob_mean <= float(args.adv_boundary_prob_max))
                )
                n_adv_generated += int(len(batch_idx))
                n_adv_filtered_boundary += int((~keep_mask).sum().item())
                keep_np = keep_mask.cpu().numpy().astype(bool)
                if args.hull_mix_label_mode == "anchor":
                    batch_labels = pool["labels"][batch_idx].astype(np.float32)
                elif args.hull_mix_label_mode == "anchor_soft":
                    cand_idx = getattr(index, "last_candidate_indices", None)
                    weights_t = getattr(pgd_gen, "last_weights", None)
                    if cand_idx is None or weights_t is None:
                        raise RuntimeError(
                            "latent-hull soft labels require candidate indices and weights"
                        )
                    weights_np = weights_t.numpy().astype(np.float32, copy=False)
                    cand_labels = pool["labels"][cand_idx]
                    batch_labels = build_anchor_preserving_soft_labels(
                        pool["labels"][batch_idx].astype(np.float32, copy=False),
                        cand_labels,
                        weights_np,
                        lambda_y=args.hull_label_lambda_y,
                        positive_value=args.hull_label_positive,
                        negative_floor=args.hull_label_negative_floor,
                        new_class_cap=args.hull_label_new_class_cap,
                    )
                else:
                    raise ValueError(
                        "hull_mix_label_mode must be anchor|anchor_soft, "
                        f"got {args.hull_mix_label_mode!r}"
                    )
                if bool(keep_mask.any()):
                    kept_labels = batch_labels[keep_np].astype(np.float32, copy=False)
                    adv_features.append(feats[keep_mask].float().cpu().numpy())
                    adv_labels.append(kept_labels)
                    if args.enable_latent_augmix_branch:
                        anchor_np = anchor_x[keep_mask].detach().cpu().numpy().astype(np.float32, copy=False)
                        adv_np = x_adv[keep_mask].detach().cpu().numpy().astype(np.float32, copy=False)
                        mixed_np, mixed_stats = build_latent_augmix_branch_signals(
                            anchor_np,
                            adv_np,
                            copies=args.latent_augmix_copies,
                            severity=args.latent_augmix_severity,
                            width=args.latent_augmix_width,
                            depth=args.latent_augmix_depth,
                            alpha=args.latent_augmix_alpha,
                            latent_weight_cap=args.latent_augmix_latent_weight_cap,
                            ops=list(args.latent_augmix_ops),
                            rng=latent_augmix_rng,
                            op_apply_fn=apply_augmix_op_np,
                            available_ops=AVAILABLE_OPS,
                            renorm=not args.no_latent_augmix_renorm,
                            clip_abs=args.latent_augmix_clip_abs,
                        )
                        latent_augmix_stats_batches.append(mixed_stats)
                        if mixed_np.shape[0] > 0:
                            labels_rep = np.tile(
                                kept_labels,
                                (max(1, int(args.latent_augmix_copies)), 1),
                            )[: mixed_np.shape[0]].astype(np.float32, copy=False)
                            mixed_feature_chunks = []
                            with torch.no_grad():
                                for j in range(0, mixed_np.shape[0], 128):
                                    mixed_t = torch.from_numpy(mixed_np[j:j + 128]).float().to(device)
                                    mixed_feature_chunks.append(
                                        victim.features_from_ecg1000(mixed_t, grad=False).float().cpu().numpy()
                                    )
                            if mixed_feature_chunks:
                                adv_features.append(np.concatenate(mixed_feature_chunks, axis=0).astype(np.float32))
                                adv_labels.append(labels_rep)
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

        loader = make_weighted_loader(
            source_x,
            source_y,
            target_x,
            target_y,
            adv_x,
            adv_y,
            args,
            adv_weight=epoch_adv_weight,
        )
        head.train()
        losses = []
        for x, y, stream in loader:
            x = x.to(device)
            y = y.to(device)
            stream = stream.to(device)
            opt.zero_grad(set_to_none=True)
            logits = head(x)
            loss = criterion(logits, y, stream)
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
                if args.target_logit_anchor_stream == "target_real":
                    target_mask = stream == 1
                elif args.target_logit_anchor_stream == "target_adv":
                    target_mask = stream == 2
                else:
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
            "n_adv_generated": int(n_adv_generated),
            "n_adv_filtered_boundary": int(n_adv_filtered_boundary),
            "adv_boundary_prob_min": float(args.adv_boundary_prob_min),
            "adv_boundary_prob_max": float(args.adv_boundary_prob_max),
            "adv_weight_effective": float(epoch_adv_weight),
            "classes_in_scope": classes_in_scope,
            "anchor_sample_stats": anchor_sample_stats,
            "attack_success": merge_attack_success_stats(attack_init_stats_batches),
            "attack_vs_anchor": merge_attack_success_stats(attack_anchor_stats_batches),
            "latent_augmix_stats": summarize_latent_augmix_stats(
                latent_augmix_stats_batches,
                enabled=bool(args.enable_latent_augmix_branch),
            ),
            "delta_mean": float(np.mean(delta_norms)) if delta_norms else None,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ptbxl_fold10 = eval_ptbxl_fold10(head, ptbxl, device, args.eval_batch_size)
            if args.selection_source == "target_real_val":
                target = eval_feature_subset(
                    head,
                    target_val_x,
                    target_val_y,
                    device,
                    args.eval_batch_size,
                )
                entry["target_val_macro_auroc"] = target["macro_auroc"]
                entry["target_val_macro_auprc"] = target["macro_auprc"]
            else:
                views = eval_pn(
                    head,
                    pn,
                    ref_ids,
                    device,
                    args.eval_batch_size,
                    report_drop_all_zero=args.report_drop_all_zero_pn2021,
                )
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
            append_checkpoint_index_record(
                checkpoint_index_path,
                {
                    "event": "eval_checkpoint_decision",
                    "epoch": int(epoch),
                    "center": center,
                    "selected_as_best": bool(entry["best_update"]),
                    "best_epoch": int(best_epoch),
                    "selection_source": args.selection_source,
                    "selection_metric": args.selection_metric,
                    "selection_score": float(cur_score),
                    "target_metric_view": (
                        "target_k500_internal_val"
                        if args.selection_source == "target_real_val"
                        else "pn2021_heldout"
                    ),
                    "target_metrics": target,
                    "source_metrics": ptbxl_fold10,
                    "source_auprc_floor": float(args.source_auprc_floor),
                    "checkpoint_path": str(run_dir / "best_head.pt"),
                    "checkpoint_sha256": (
                        file_sha256(run_dir / "best_head.pt")
                        if entry["best_update"]
                        else None
                    ),
                    "selection_safety": selection_safety["selection_safety"],
                },
            )
            atk = entry["attack_success"]
            atk_anchor = entry["attack_vs_anchor"]
            atk_msg = ""
            if isinstance(atk, dict) and atk.get("success_rate") is not None:
                atk_msg = (
                    f" atk_init={float(atk['success_rate']):.2f}"
                    f"/{float(atk['loss_gain_mean']):+.4f}"
                )
                if isinstance(atk_anchor, dict) and atk_anchor.get("success_rate") is not None:
                    atk_msg += (
                        f" atk_anchor={float(atk_anchor['success_rate']):.2f}"
                        f"/{float(atk_anchor['loss_gain_mean']):+.4f}"
                    )
            print(
                f"[{center}] ep={epoch:03d} loss={entry['train_loss']:.4f} "
                f"adv_w={epoch_adv_weight:.3g} "
                f"{args.selection_source}={target['macro_auroc']:.4f}/{target['macro_auprc']:.4f} "
                f"ptbxl={ptbxl_fold10['macro_auroc']:.4f}/{ptbxl_fold10['macro_auprc']:.4f} "
                f"best_ep={best_epoch}{atk_msg}",
                flush=True,
            )
        else:
            print(
                f"[{center}] ep={epoch:03d} loss={entry['train_loss']:.4f} "
                f"adv_w={epoch_adv_weight:.3g}",
                flush=True,
            )
        logs.append(entry)
        with (run_dir / "training_log.json").open("w") as f:
            json.dump(logs, f, indent=2)

    if baseline_views is None:
        head.load_state_dict(torch.load(run_dir / "initial_head.pt", map_location=device))
        baseline_views = eval_pn(
            head,
            pn,
            ref_ids,
            device,
            args.eval_batch_size,
            report_drop_all_zero=args.report_drop_all_zero_pn2021,
        )
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
        "label_mapping": pn2021_super5_label_mapping_payload(),
        "class_names": list(CLASS_NAMES_SUPER5),
        "K": int(args.k),
        "selected_ref_record_ids": selected_ids_all.tolist(),
        "target_train_record_ids": target_train_record_ids,
        "target_val_record_ids": target_val_record_ids,
        "classes_in_scope": classes_in_scope,
        "label_counts": pool["label_counts"],
        "original_k500_label_counts": original_pool_label_counts,
        "class_loss_weights": class_loss_weights_map,
        "head_type": args.head_type,
        "baseline_pn2021_views": baseline_views,
        "baseline_ptbxl_fold10": baseline_ptbxl,
        "final_pn2021_views": final_views,
        "final_ptbxl_fold10": final_ptbxl,
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "selection_record": selection_safety,
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
            run_dir = ecgfounder_lhat_run_dir(
                out_dir,
                center=c,
                k=int(r["K"]),
                hull_m=int(r["config"]["hull_m"]),
                hull_lambda=r["config"]["hull_lambda"],
                epochs=int(r["config"]["epochs"]),
                seed=int(r["config"]["seed"]),
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
    p.add_argument(
        "--ref_root",
        default=str(REF_ROOT),
        help="Root containing per-center K-shot ref_meta files used for non-target ref-excluded eval views.",
    )
    p.add_argument("--k_anchor", type=int, default=300)
    p.add_argument(
        "--anchor_sample_mode",
        choices=[
            "stratified",
            "hard_bce",
            "uncertainty",
            "base_hard_bce",
            "base_uncertainty",
            "target_hard_bce",
            "target_uncertainty",
        ],
        default="stratified",
        help=(
            "How to choose K-shot latent anchors for online VAE adversarial samples. "
            "hard_bce/uncertainty use the current head; base_* freezes the initial "
            "direct/K500 head; target_* uses --target_logit_anchor_path. All modes "
            "score only the known K-shot training anchors, without looking at "
            "held-out center distribution."
        ),
    )
    p.add_argument("--anchor_sample_power", type=float, default=1.0)
    p.add_argument("--anchor_sample_min_weight", type=float, default=1e-4)
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
    p.add_argument(
        "--hull_neighbor_distance_space",
        choices=["raw", "standardized"],
        default="raw",
        help="Distance space for latent-hull same-label neighbor selection.",
    )
    p.add_argument(
        "--hull_neighbor_mode",
        choices=["nearest", "local_random", "random"],
        default="nearest",
        help=(
            "nearest keeps historical deterministic kNN; local_random samples "
            "from top-K same-label neighbors; random samples from the full "
            "same-label pool."
        ),
    )
    p.add_argument("--hull_neighbor_pool_size", type=int, default=0)
    p.add_argument("--hull_neighbor_pool_multiplier", type=int, default=4)
    p.add_argument("--pgd_eps", type=float, default=2.0)
    p.add_argument("--pgd_batch", type=int, default=16)
    p.add_argument(
        "--adv_boundary_prob_min",
        type=float,
        default=0.0,
        help=(
            "Filter online adversarial features by mean post-attack probability "
            "over positive labels. Values below this are considered too strong."
        ),
    )
    p.add_argument(
        "--adv_boundary_prob_max",
        type=float,
        default=1.0,
        help=(
            "Filter online adversarial features by mean post-attack probability "
            "over positive labels. Values above this are considered too weak."
        ),
    )
    p.add_argument(
        "--attack_success_margin",
        type=float,
        default=1e-4,
        help="Per-sample BCE gain threshold for counting latent PGD as successful.",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--eval_batch_size", type=int, default=4096)
    p.add_argument(
        "--head_type",
        choices=["linear", "residual_adapter", "feature_adapter"],
        default="linear",
        help=(
            "linear updates the Super5 head directly; residual_adapter adds a "
            "zero-init logit residual; feature_adapter adds a zero-init feature "
            "residual before the Super5 head."
        ),
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
        "--init_base_head_from_k500_root",
        default="",
        help=(
            "Optional runs/ directory from ECGFounder K500 direct fine-tuning. "
            "Each center loads its direct-ft best_head.pt into the base linear "
            "head before training the residual adapter."
        ),
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
        "--source_bce_loss_weight",
        type=float,
        default=1.0,
        help="Per-sample BCE multiplier for the PTB-XL source stream.",
    )
    p.add_argument(
        "--target_real_bce_loss_weight",
        type=float,
        default=1.0,
        help="Per-sample BCE multiplier for the real K-shot target stream.",
    )
    p.add_argument(
        "--adv_bce_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Per-sample BCE multiplier for online VAE adversarial samples. "
            "Values below 1 turn the adversarial stream into a gentler "
            "consistency/distillation branch instead of hard-label retraining."
        ),
    )
    p.add_argument(
        "--adv_weight_start",
        type=float,
        default=None,
        help=(
            "Optional starting sampler weight for the adversarial stream. "
            "When --adv_weight_warmup_epochs > 0 and this is omitted, warmup starts at 0."
        ),
    )
    p.add_argument(
        "--adv_weight_warmup_epochs",
        type=int,
        default=0,
        help=(
            "Linearly warm up adversarial stream sampler weight from "
            "--adv_weight_start to --adv_weight over this many epochs."
        ),
    )
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
        "--target_logit_anchor_source",
        choices=["path", "initial_head"],
        default="path",
        help=(
            "Source for target/adv logit anchoring. initial_head freezes the "
            "head after any K500-direct initialization, so it is paper-safe and "
            "does not require a center-specific held-out teacher path."
        ),
    )
    p.add_argument(
        "--target_logit_anchor_stream",
        choices=["target_real", "target_adv", "target_adv_real"],
        default="target_adv_real",
        help="Which non-source stream receives the target logit anchor loss.",
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
        "--enable_latent_augmix_branch",
        action="store_true",
        help=(
            "After each kept latent-hull adversarial decode, mix x_adv as one "
            "AugMix branch with ECG corruption chains from the clean anchor and "
            "push the resulting ECGFounder features into the target_adv stream."
        ),
    )
    p.add_argument("--latent_augmix_copies", type=int, default=1)
    p.add_argument("--latent_augmix_width", type=int, default=3)
    p.add_argument("--latent_augmix_depth", type=int, default=-1)
    p.add_argument("--latent_augmix_alpha", type=float, default=1.0)
    p.add_argument("--latent_augmix_severity", type=int, default=2)
    p.add_argument("--latent_augmix_latent_weight_cap", type=float, default=0.25)
    p.add_argument(
        "--latent_augmix_ops",
        nargs="+",
        default=["powerline_noise", "emg_noise", "baseline_wander", "baseline_shift"],
        choices=AVAILABLE_OPS,
        help="ECG corruption ops for non-latent AugMix branches. Random lead masking is excluded by default.",
    )
    p.add_argument(
        "--no_latent_augmix_renorm",
        action="store_true",
        help="Do not global-zscore the final latent-branch AugMix waveform before feature extraction.",
    )
    p.add_argument(
        "--latent_augmix_clip_abs",
        type=float,
        default=6.0,
        help="Clip final latent-branch AugMix waveform after optional zscore; <=0 disables clipping.",
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
            "target_auroc",
            "target_auprc",
            "target_plus_source_auroc",
            "target_plus_source_auprc",
            "target_source_hmean_auroc",
            "target_source_hmean_auprc",
            "target_under_source_floor",
        ],
        default="target_auprc",
        help=(
            "Metric used to save best_head.pt during online AT. "
            "Use last_epoch for fixed-horizon runs without target-test checkpoint selection."
        ),
    )
    p.add_argument(
        "--selection_source",
        choices=["pn2021_heldout", "target_real_val"],
        default="target_real_val",
        help=(
            "Checkpoint-selection source. target_real_val uses a deterministic "
            "validation split from the known K-shot target records and does not "
            "evaluate PN2021 held-out target data during training."
        ),
    )
    p.add_argument(
        "--allow_pn2021_heldout_selection",
        action="store_true",
        help=(
            "Allow explicit diagnostic checkpoint selection on PN2021 held-out "
            "target labels. Do not use for paper-safe runs."
        ),
    )
    p.add_argument("--target_real_val_fraction", type=float, default=0.2)
    p.add_argument("--target_real_val_seed", type=int, default=20260531)
    p.add_argument("--source_selection_weight", type=float, default=0.25)
    p.add_argument("--source_auprc_floor", type=float, default=0.79)
    p.add_argument("--source_floor_penalty", type=float, default=5.0)
    p.add_argument("--classes_in_scope", nargs="*", default=[])
    p.add_argument(
        "--anchor_base_root",
        default="",
        help=(
            "Optional root for per-center K-shot ECGTwin VAE anchors. Supports "
            "<root>/<center>/k{k}_seed{seed>/<center>_real_k{k}_seed{seed}."
        ),
    )
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.selection_safety_record = validate_ecgfounder_lhat_runtime_selection(
        selection_source=args.selection_source,
        selection_metric=args.selection_metric,
        target_real_val_fraction=args.target_real_val_fraction,
        target_real_val_seed=args.target_real_val_seed,
        allow_pn2021_heldout_selection=args.allow_pn2021_heldout_selection,
    )
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
