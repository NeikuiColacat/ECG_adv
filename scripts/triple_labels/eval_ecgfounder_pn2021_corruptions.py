#!/usr/bin/env python3
"""Evaluate saved ECGFounder heads on PN2021-C corruptions.

ECGFounder uses a frozen 500 Hz encoder plus a feature-space Super5 head, so it
cannot be evaluated through the 100 Hz CNN PN2021-C entrypoint.  This script
keeps the same corruption and aggregation contract while routing corrupted ECGs
through the ECGFounder encoder before applying the saved head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        "/home/linbinhao/ECG_adv_data",
    )
)
ECGFOUNDER_ROOT = Path(os.environ.get("ECGFOUNDER_ROOT", str(DATA_ROOT / "ecgfounder")))
for path in (REPO_ROOT, ECGFOUNDER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from net1d import Net1D  # noqa: E402
from finetune_model import ft_12lead_ECGFounder  # noqa: E402
from physionet2021_dataset import TARGET_POINTS  # noqa: E402

from ecg_adv_gen.data.contracts import PREPROCESS_CONTRACT_ID  # noqa: E402
from ecg_adv_gen.evaluation.pn2021_corruptions import (  # noqa: E402
    aggregate_corruption_summary,
    clean_mmap_cache_path,
    clean_npz_cache_path,
    filter_record_indices,
    load_npz_metadata,
)
from ecg_adv_gen.models.ecgfounder_heads import (  # noqa: E402
    FeatureAdapterHead,
    OperatorConditionedLogitAdapter,
    ResidualAdapterHead,
)
from ecg_adv_gen.models.ecgfounder_inference import sigmoid_clipped  # noqa: E402
from ecg_adv_gen.models.ecgfounder_torch import ecg1000_to_ecgfounder_input  # noqa: E402
from scripts.triple_labels.build_pn2021_corruptions import (  # noqa: E402
    DEFAULT_CENTERS,
    DEFAULT_CORRUPTIONS,
    PN2021_C_CACHE_VERSION,
    PUBLIC_TO_INTERNAL_SEVERITY,
)
from scripts.triple_labels.eval_crosscenter import (  # noqa: E402
    PN2021_EVAL_CACHE_VERSION,
    compute_macro_auroc_auprc,
)
from scripts.triple_labels.eval_pn2021_corruptions import (  # noqa: E402
    STRESS_PROFILE_CHOICES,
    _build_corruption_op,
    _load_raw_first_center,
    _load_native_raw_first_center,
    RawFirstCleanPN2021Dataset,
    RawFirstCorruptedPN2021Dataset,
    NativeRawFirstCleanPN2021Dataset,
    NativeRawFirstCorruptedPN2021Dataset,
)
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, get_scheme  # noqa: E402


CHECKPOINT = ECGFOUNDER_ROOT / "checkpoint/12_lead_ECGFounder.pth"


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(p) for p in (base_seed,) + parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:4], "little")


def _ref_ids_sha256(ids: set[str]) -> str:
    values = sorted(str(item) for item in ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _pn2021_scheme(args: argparse.Namespace) -> dict[str, Any]:
    return get_scheme(args.scheme)


def _clean_mmap_path(cache_dir: str, scheme: str, center: str) -> str:
    return clean_mmap_cache_path(cache_dir, scheme, center, PN2021_EVAL_CACHE_VERSION)


def _clean_npz_path(cache_dir: str, scheme: str, center: str) -> str:
    return clean_npz_cache_path(cache_dir, scheme, center, PN2021_EVAL_CACHE_VERSION)


def _load_clean_center(args: argparse.Namespace, center: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str]:
    mmap_root = _clean_mmap_path(args.clean_mmap_cache_dir, args.scheme, center)
    sig_path = os.path.join(mmap_root, "signals.npy")
    lab_path = os.path.join(mmap_root, "labels.npy")
    rid_path = os.path.join(mmap_root, "record_ids.npy")
    meta_path = os.path.join(mmap_root, "metadata.json")
    if os.path.isdir(mmap_root) and os.path.exists(sig_path) and os.path.exists(lab_path) and os.path.exists(rid_path):
        metadata: dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
        return (
            np.load(sig_path, mmap_mode="r"),
            np.load(lab_path, mmap_mode="r"),
            np.load(rid_path, allow_pickle=True).astype(str),
            metadata,
            "mmap",
        )

    npz_path = _clean_npz_path(args.clean_cache_dir, args.scheme, center)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"clean cache not found for {center}: {mmap_root} or {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["signals"].astype(np.float32, copy=False),
        data["labels"].astype(np.float32, copy=False),
        data["record_ids"].astype(str),
        load_npz_metadata(data),
        "npz",
    )


class ECGFounderStreamingCorruptedDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        corruption: str,
        public_severity: int,
        *,
        seed: int,
        crop_len: int,
        severity_profile: str,
    ) -> None:
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.corruption = str(corruption)
        self.public_severity = int(public_severity)
        self.internal_severity = PUBLIC_TO_INTERNAL_SEVERITY[self.public_severity]
        self.seed = int(seed)
        self.crop_len = int(crop_len)
        self.severity_profile = str(severity_profile)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])
        sample_seed = _stable_seed(self.seed, self.corruption, self.public_severity, real_idx)
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        op = _build_corruption_op(self.corruption, self.public_severity, self.severity_profile)

        sig_tc = self.signals[real_idx]
        if self.crop_len and self.crop_len < sig_tc.shape[0]:
            start = max((sig_tc.shape[0] - self.crop_len) // 2, 0)
            sig_tc = sig_tc[start : start + self.crop_len]
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        corrupt_ct = op(sig_ct)
        label = torch.from_numpy(
            np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        ).float()
        return corrupt_ct.float(), label


class ECGFounderCleanDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        crop_len: int,
    ) -> None:
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop_len = int(crop_len)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])
        sig_tc = self.signals[real_idx]
        if self.crop_len and self.crop_len < sig_tc.shape[0]:
            start = max((sig_tc.shape[0] - self.crop_len) // 2, 0)
            sig_tc = sig_tc[start : start + self.crop_len]
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        label = torch.from_numpy(
            np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        ).float()
        return sig_ct.float(), label


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
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)


def _load_result(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "eval_result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    with result_path.open() as f:
        result = json.load(f)
    if "center" not in result:
        raise ValueError(f"{result_path} does not record center")
    return result


def _detect_eval_mode(run_dir: Path) -> str:
    """Return how an ECGFounder run should be evaluated."""

    if (run_dir / "best_head.pt").exists():
        return "feature_head"
    if (run_dir / "best_model.pt").exists():
        return "fullft_model"
    raise FileNotFoundError(
        f"expected either best_head.pt or best_model.pt under {run_dir}"
    )


def _infer_feature_dim(head_state: dict[str, torch.Tensor]) -> int:
    for key in ("weight", "base_head.weight"):
        if key in head_state:
            return int(head_state[key].shape[1])
    raise RuntimeError("cannot infer ECGFounder feature dim from best_head.pt")


def _build_head(run_dir: Path, result: dict[str, Any], device: torch.device) -> nn.Module:
    head_path = run_dir / "best_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(head_path)
    state = torch.load(head_path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError(f"unsupported head checkpoint payload: {head_path}")
    feature_dim = _infer_feature_dim(state)
    cfg = dict(result.get("config") or {})
    head_type = str(result.get("head_type") or cfg.get("head_type") or "linear")
    base_head = nn.Linear(feature_dim, len(CLASS_NAMES_SUPER5)).to(device)
    if head_type == "linear":
        head: nn.Module = base_head
    elif head_type == "residual_adapter":
        head = ResidualAdapterHead(
            base_head=base_head,
            hidden_dim=int(cfg.get("adapter_hidden", 128)),
            dropout=float(cfg.get("adapter_dropout", 0.0)),
            scale=float(cfg.get("adapter_scale", 1.0)),
            freeze_base=bool(cfg.get("freeze_base_head", False)),
        ).to(device)
    elif head_type == "feature_adapter":
        head = FeatureAdapterHead(
            base_head=base_head,
            hidden_dim=int(cfg.get("adapter_hidden", 128)),
            dropout=float(cfg.get("adapter_dropout", 0.0)),
            scale=float(cfg.get("adapter_scale", 1.0)),
            freeze_base=bool(cfg.get("freeze_base_head", False)),
        ).to(device)
    else:
        raise ValueError(f"unsupported ECGFounder head_type={head_type!r}")
    head.load_state_dict(state)
    head.eval()
    return head


def _build_fullft_model(run_dir: Path, checkpoint: Path, device: torch.device) -> nn.Module:
    model = ft_12lead_ECGFounder(device, str(checkpoint), len(CLASS_NAMES_SUPER5), linear_prob=False)
    model_path = run_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    payload = torch.load(model_path, map_location=device)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    if not isinstance(state, dict):
        raise RuntimeError(f"unsupported fullFT checkpoint payload: {model_path}")
    state = {str(k).removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)
    if isinstance(payload, dict) and "op_adapter_state_dict" in payload:
        cfg = dict(payload.get("op_adapter_config") or {})
        if not cfg:
            raise RuntimeError(f"{model_path} has op_adapter_state_dict but no op_adapter_config")
        op_adapter = OperatorConditionedLogitAdapter(
            feature_dim=int(cfg["feature_dim"]),
            num_classes=int(cfg["num_classes"]),
            op_names=list(cfg["op_names"]),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            dropout=float(cfg.get("dropout", 0.0)),
            scale=float(cfg.get("scale", 1.0)),
        ).to(device)
        op_adapter.load_state_dict(payload["op_adapter_state_dict"])
        op_adapter.eval()
        return OperatorConditionedFullFTModel(model, op_adapter).to(device)
    return model


def _forward_logits_features(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(model, "return_features"):
        raise RuntimeError("op-conditioned ECGFounder eval requires model.return_features support")
    previous = bool(getattr(model, "return_features"))
    try:
        setattr(model, "return_features", True)
        output = model(x)
    finally:
        setattr(model, "return_features", previous)
    if not (isinstance(output, tuple) and len(output) == 2):
        raise RuntimeError("op-conditioned ECGFounder eval expected model(x) to return (logits, features)")
    logits, features = output
    return logits, features


class OperatorConditionedFullFTModel(nn.Module):
    def __init__(self, base_model: nn.Module, op_adapter: OperatorConditionedLogitAdapter) -> None:
        super().__init__()
        self.base_model = base_model
        self.op_adapter = op_adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    def forward_with_operator(self, x: torch.Tensor, operator_name: str) -> torch.Tensor:
        logits, features = _forward_logits_features(self.base_model, x)
        op_ids = self.op_adapter.op_ids_from_names(
            [str(operator_name)] * int(x.shape[0]),
            device=x.device,
        )
        return logits + self.op_adapter(features, op_ids)


def _clean_metric_for_center(result: dict[str, Any], center: str) -> dict[str, Any]:
    views = result.get("final_pn2021_views")
    if isinstance(views, dict) and center in views:
        row = views[center].get("per_center", {}).get(center, {})
        if row:
            return row
    if center == str(result.get("center")):
        row = result.get("target_excluding_ref")
        if isinstance(row, dict) and row:
            return row
    target_view = result.get("target_view")
    if isinstance(target_view, dict):
        row = target_view.get("per_center", {}).get(center, {})
        if row:
            return row
    all_views = result.get("all_views")
    if isinstance(all_views, dict) and center in all_views:
        row = all_views[center].get("per_center", {}).get(center, {})
        if row:
            return row
    return {}


def _input_stabilizer_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.ecgfounder_input_bandpass_low_hz is not None:
        kwargs["bandpass_low_hz"] = float(args.ecgfounder_input_bandpass_low_hz)
    if args.ecgfounder_input_bandpass_high_hz is not None:
        kwargs["bandpass_high_hz"] = float(args.ecgfounder_input_bandpass_high_hz)
    if bool(args.ecgfounder_input_repair_flat_leads):
        kwargs["repair_flat_leads"] = True
    if args.ecgfounder_input_clip_abs is not None:
        kwargs["clip_abs"] = float(args.ecgfounder_input_clip_abs)
    return kwargs


@torch.no_grad()
def infer_ecgfounder(
    feature_model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
    apply_input_zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    feature_model.eval()
    head.eval()
    for ecg_ct, labels in loader:
        ecg_ct = ecg_ct.to(device, non_blocking=True)
        x = ecg1000_to_ecgfounder_input(
            ecg_ct,
            target_points=TARGET_POINTS,
            apply_global_zscore=bool(apply_input_zscore),
            **(input_stabilizer_kwargs or {}),
        )
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            _, features = feature_model(x)
            logits = head(features)
        y_true.append(labels.numpy())
        y_score.append(sigmoid_clipped(logits.detach().float().cpu().numpy()))
    return np.concatenate(y_true, axis=0), np.concatenate(y_score, axis=0)


@torch.no_grad()
def infer_ecgfounder_fullft(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    operator_name: str | None = None,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
    apply_input_zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    model.eval()
    for ecg_ct, labels in loader:
        ecg_ct = ecg_ct.to(device, non_blocking=True)
        x = ecg1000_to_ecgfounder_input(
            ecg_ct,
            target_points=TARGET_POINTS,
            apply_global_zscore=bool(apply_input_zscore),
            **(input_stabilizer_kwargs or {}),
        )
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            if operator_name and hasattr(model, "forward_with_operator"):
                logits = model.forward_with_operator(x, operator_name)
            else:
                logits = model(x)
        y_true.append(labels.numpy())
        y_score.append(sigmoid_clipped(logits.detach().float().cpu().numpy()))
    return np.concatenate(y_true, axis=0), np.concatenate(y_score, axis=0)


def eval_clean_center(
    eval_mode: str,
    model_or_feature_model: nn.Module,
    head: nn.Module | None,
    result: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    center: str,
    *,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
    apply_input_zscore: bool = True,
) -> dict[str, Any]:
    sample_rates = None
    if args.corruption_input == "native_raw_first":
        (
            signals,
            labels,
            record_ids,
            metadata,
            clean_kind,
            sample_rates,
        ) = _load_native_raw_first_center(
            args,
            _pn2021_scheme(args),
            center,
        )
    elif args.corruption_input == "raw_first":
        signals, labels, record_ids, metadata, clean_kind = _load_raw_first_center(
            args,
            _pn2021_scheme(args),
            center,
        )
    else:
        signals, labels, record_ids, metadata, clean_kind = _load_clean_center(args, center)
    selected_ids = set(str(x) for x in result.get("selected_ref_record_ids", []))
    exclude_ids = selected_ids if center == str(result["center"]) else set()
    indices = filter_record_indices(record_ids, exclude_ids, args.limit)
    if args.corruption_input == "native_raw_first":
        ds = NativeRawFirstCleanPN2021Dataset(
            signals,
            labels,
            sample_rates,
            indices,
            crop_len=args.crop_len,
            input_stabilizer_config={},
        )
    elif args.corruption_input == "raw_first":
        ds = RawFirstCleanPN2021Dataset(
            signals,
            labels,
            indices,
            crop_len=args.crop_len,
            input_stabilizer_config={},
        )
    else:
        ds = ECGFounderCleanDataset(
            signals,
            labels,
            indices,
            crop_len=args.crop_len,
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if eval_mode == "feature_head":
        if head is None:
            raise RuntimeError("feature_head evaluation requires a head module")
        y_true, y_score = infer_ecgfounder(
            model_or_feature_model,
            head,
            loader,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
            apply_input_zscore=apply_input_zscore,
        )
    elif eval_mode == "fullft_model":
        y_true, y_score = infer_ecgfounder_fullft(
            model_or_feature_model,
            loader,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
            apply_input_zscore=apply_input_zscore,
        )
    else:
        raise ValueError(f"unknown ECGFounder eval mode: {eval_mode}")
    metrics = compute_macro_auroc_auprc(
        y_true,
        y_score,
        CLASS_NAMES_SUPER5,
        min_pos=args.min_pos,
    )
    print(
        f"  {center:<18} clean{'':<22} n={len(ds):>5} "
        f"AUROC={metrics['macro_auroc']:.4f} AUPRC={metrics['macro_auprc']:.4f}",
        flush=True,
    )
    return {
        "cache_source": f"stream:{clean_kind}",
        "metadata": metadata,
        "n_records": int(len(ds)),
        "n_excluded_ref_ids_for_center": int(len(exclude_ids)),
        "ref_record_ids_sha256": _ref_ids_sha256(exclude_ids),
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "n_classes_used": metrics["n_classes_used"],
        "per_class": metrics["per_class"],
    }


def eval_one(
    eval_mode: str,
    model_or_feature_model: nn.Module,
    head: nn.Module | None,
    result: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    center: str,
    corruption: str,
    severity: int,
    *,
    clean_metric_override: dict[str, Any] | None = None,
    input_stabilizer_kwargs: dict[str, Any] | None = None,
    apply_input_zscore: bool = True,
) -> dict[str, Any]:
    sample_rates = None
    if args.corruption_input == "native_raw_first":
        (
            signals,
            labels,
            record_ids,
            metadata,
            clean_kind,
            sample_rates,
        ) = _load_native_raw_first_center(
            args,
            _pn2021_scheme(args),
            center,
        )
    elif args.corruption_input == "raw_first":
        signals, labels, record_ids, metadata, clean_kind = _load_raw_first_center(
            args,
            _pn2021_scheme(args),
            center,
        )
    else:
        signals, labels, record_ids, metadata, clean_kind = _load_clean_center(args, center)
    selected_ids = set(str(x) for x in result.get("selected_ref_record_ids", []))
    exclude_ids = selected_ids if center == str(result["center"]) else set()
    indices = filter_record_indices(record_ids, exclude_ids, args.limit)
    if args.corruption_input == "native_raw_first":
        ds = NativeRawFirstCorruptedPN2021Dataset(
            signals,
            labels,
            sample_rates,
            corruption,
            severity,
            seed=args.seed,
            crop_len=args.crop_len,
            severity_profile=args.severity_profile,
            indices=indices,
            input_stabilizer_config={},
        )
    elif args.corruption_input == "raw_first":
        ds = RawFirstCorruptedPN2021Dataset(
            signals,
            labels,
            corruption,
            severity,
            seed=args.seed,
            crop_len=args.crop_len,
            severity_profile=args.severity_profile,
            indices=indices,
            input_stabilizer_config={},
        )
    else:
        ds = ECGFounderStreamingCorruptedDataset(
            signals,
            labels,
            indices,
            corruption,
            severity,
            seed=args.seed,
            crop_len=args.crop_len,
            severity_profile=args.severity_profile,
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    t0 = time.time()
    if eval_mode == "feature_head":
        if head is None:
            raise RuntimeError("feature_head evaluation requires a head module")
        y_true, y_score = infer_ecgfounder(
            model_or_feature_model,
            head,
            loader,
            device,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
            apply_input_zscore=apply_input_zscore,
        )
    elif eval_mode == "fullft_model":
        y_true, y_score = infer_ecgfounder_fullft(
            model_or_feature_model,
            loader,
            device,
            operator_name=corruption,
            input_stabilizer_kwargs=input_stabilizer_kwargs,
            apply_input_zscore=apply_input_zscore,
        )
    else:
        raise ValueError(f"unknown ECGFounder eval mode: {eval_mode}")
    metrics = compute_macro_auroc_auprc(
        y_true,
        y_score,
        CLASS_NAMES_SUPER5,
        min_pos=args.min_pos,
    )
    nonzero_mask = np.asarray(y_true).sum(axis=1) > 0
    drop_all_zero_metrics = None
    if np.any(nonzero_mask):
        drop_all_zero_metrics = compute_macro_auroc_auprc(
            y_true[nonzero_mask],
            y_score[nonzero_mask],
            CLASS_NAMES_SUPER5,
            min_pos=args.min_pos,
        )
    clean = clean_metric_override if clean_metric_override is not None else _clean_metric_for_center(result, center)
    if clean_metric_override is not None:
        clean_metric_source = (
            "recomputed_native_raw_first_clean"
            if args.corruption_input == "native_raw_first"
            else "recomputed_raw_first_clean"
            if args.corruption_input == "raw_first"
            else "recomputed_input_stabilizer"
        )
    else:
        clean_metric_source = "saved_eval_result"
    auroc_drop = None
    auprc_drop = None
    if clean.get("macro_auroc") is not None:
        auroc_drop = float(clean["macro_auroc"] - metrics["macro_auroc"])
    if clean.get("macro_auprc") is not None:
        auprc_drop = float(clean["macro_auprc"] - metrics["macro_auprc"])
    print(
        f"  {center:<18} {corruption:<22} s{severity} n={len(ds):>5} "
        f"AUROC={metrics['macro_auroc']:.4f} AUPRC={metrics['macro_auprc']:.4f} "
        f"drop=({auroc_drop if auroc_drop is not None else float('nan'):.4f}, "
        f"{auprc_drop if auprc_drop is not None else float('nan'):.4f}) "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {
        "cache_path": None,
        "cache_source": f"stream:{clean_kind}",
        "metadata": metadata,
        "metadata_compatibility": {
            "compatible": None,
            "reason": "ecgfounder_clean_eval_result",
            "preprocess_contract_id": PREPROCESS_CONTRACT_ID,
        },
        "n_excluded_ref_ids_for_center": int(len(exclude_ids)),
        "ref_record_ids_sha256": _ref_ids_sha256(exclude_ids),
        "corruption": {
            "name": corruption,
            "severity_profile": args.severity_profile,
            "public_severity": int(severity),
            "internal_severity": int(PUBLIC_TO_INTERNAL_SEVERITY[int(severity)]),
            "seed": int(args.seed),
        },
        "n_records": int(len(ds)),
        "n_all_zero_labels": int((~nonzero_mask).sum()),
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "n_classes_used": metrics["n_classes_used"],
        "per_class": metrics["per_class"],
        "drop_all_zero_n_records": int(nonzero_mask.sum()),
        "drop_all_zero_macro_auroc": (
            drop_all_zero_metrics["macro_auroc"]
            if drop_all_zero_metrics is not None
            else None
        ),
        "drop_all_zero_macro_auprc": (
            drop_all_zero_metrics["macro_auprc"]
            if drop_all_zero_metrics is not None
            else None
        ),
        "drop_all_zero_n_classes_used": (
            drop_all_zero_metrics["n_classes_used"]
            if drop_all_zero_metrics is not None
            else 0
        ),
        "drop_all_zero_per_class": (
            drop_all_zero_metrics["per_class"]
            if drop_all_zero_metrics is not None
            else {}
        ),
        "clean_macro_auroc": clean.get("macro_auroc"),
        "clean_macro_auprc": clean.get("macro_auprc"),
        "clean_metric_source": clean_metric_source,
        "auroc_drop_vs_clean": auroc_drop,
        "auprc_drop_vs_clean": auprc_drop,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="super5", choices=["super5"])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--variant", default="")
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument(
        "--clean_mmap_cache_dir",
        default=str(DATA_ROOT / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"),
    )
    p.add_argument(
        "--clean_cache_dir",
        default=str(DATA_ROOT / "triple_labels/pn2021_eval_cache_minresample_perglobal"),
    )
    p.add_argument(
        "--corruption_input",
        default="preprocessed_cache",
        choices=["preprocessed_cache", "raw_first", "native_raw_first"],
        help=(
            "preprocessed_cache keeps historical behavior: corrupt the "
            "100Hz/1000 per-sample-z-scored clean cache and let ECGFounder "
            "input conversion z-score again. raw_first reads WFDB records, "
            "resamples/pads without z-score, applies corruption, z-scores once, "
            "then only resamples to ECGFounder length. native_raw_first applies "
            "corruption on WFDB native-fs/native-length signals after only lead "
            "reorder and NaN repair, then runs model preprocessing."
        ),
    )
    p.add_argument(
        "--pn2021_root",
        default=str(DATA_ROOT / "physionet2021"),
        help="PN2021 root containing training/<center>; used by --corruption_input raw_first.",
    )
    p.add_argument("--required_cache_version", default=PN2021_C_CACHE_VERSION)
    p.add_argument("--centers", nargs="+", default=None)
    p.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--severity_profile", default="standard", choices=STRESS_PROFILE_CHOICES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--min_pos", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ecgfounder_input_bandpass_low_hz", type=float, default=None)
    p.add_argument("--ecgfounder_input_bandpass_high_hz", type=float, default=None)
    p.add_argument("--ecgfounder_input_repair_flat_leads", action="store_true")
    p.add_argument("--ecgfounder_input_clip_abs", type=float, default=None)
    p.add_argument(
        "--recompute_clean_with_input_stabilizer",
        action="store_true",
        help="Recompute clean PN2021 metrics through the same ECGFounder input stabilizer before reporting drops.",
    )
    p.add_argument("--output_path", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    result = _load_result(run_dir)
    centers = list(args.centers or [str(result["center"])])
    unknown = sorted(set(centers).difference(DEFAULT_CENTERS))
    if unknown:
        raise ValueError(f"unknown PN2021-C centers: {unknown}")

    device = torch.device(args.device)
    eval_mode = _detect_eval_mode(run_dir)
    if eval_mode == "feature_head":
        model_or_feature_model = build_ecgfounder_feature_model(Path(args.checkpoint), device)
        head = _build_head(run_dir, result, device)
    else:
        model_or_feature_model = _build_fullft_model(run_dir, Path(args.checkpoint), device)
        head = None
    input_stabilizer_kwargs = _input_stabilizer_kwargs_from_args(args)
    apply_input_zscore = args.corruption_input not in {"raw_first", "native_raw_first"}
    clean_metric_overrides: dict[str, dict[str, Any]] = {}
    if (
        args.recompute_clean_with_input_stabilizer
        or args.corruption_input in {"raw_first", "native_raw_first"}
    ):
        for center in centers:
            clean_metric_overrides[center] = eval_clean_center(
                eval_mode,
                model_or_feature_model,
                head,
                result,
                args,
                device,
                center,
                input_stabilizer_kwargs=input_stabilizer_kwargs,
                apply_input_zscore=apply_input_zscore,
            )

    output: dict[str, Any] = {
        "scheme": args.scheme,
        "model_name": "ECGFounder",
        "eval_mode": eval_mode,
        "variant": args.variant,
        "run_dir": str(run_dir),
        "corruption_input": args.corruption_input,
        "pn2021_root": (
            args.pn2021_root
            if args.corruption_input in {"raw_first", "native_raw_first"}
            else None
        ),
        "clean_eval_json": str(run_dir / "eval_result.json"),
        "head_path": str(run_dir / "best_head.pt") if (run_dir / "best_head.pt").exists() else None,
        "model_path": str(run_dir / "best_model.pt") if (run_dir / "best_model.pt").exists() else None,
        "checkpoint": str(Path(args.checkpoint)),
        "center_from_run": str(result["center"]),
        "centers": centers,
        "corruptions": list(args.corruptions),
        "severities": list(args.severities),
        "severity_profile": args.severity_profile,
        "crop_len": int(args.crop_len),
        "required_cache_version": args.required_cache_version,
        "pn2021_c_cache_version": args.required_cache_version,
        "input_stabilizer": {
            "kwargs": dict(input_stabilizer_kwargs),
            "recompute_clean_with_input_stabilizer": bool(args.recompute_clean_with_input_stabilizer),
            "ecgfounder_apply_input_zscore": bool(apply_input_zscore),
        },
        "corruption_order": (
            [
                "wfdb_read_native_fs_native_length",
                "lead_reorder_nan_guard_no_resample_no_zscore",
                "corruption_with_native_sample_rate",
                "resample_pad_or_truncate_to_100hz_1000",
                "per_sample_global_zscore",
                "center_crop",
                "ecgfounder_resample_1000_to_5000_no_second_zscore",
                "model",
            ]
            if args.corruption_input == "native_raw_first"
            else (
                [
                    "wfdb_read",
                    "lead_reorder_nan_guard_resample_pad_no_zscore",
                    "corruption",
                    "per_sample_global_zscore",
                    "center_crop",
                    "ecgfounder_resample_1000_to_5000_no_second_zscore",
                    "model",
                ]
                if args.corruption_input == "raw_first"
                else [
                    "load_100hz1000_per_sample_global_zscore_cache",
                    "center_crop_or_full_signal",
                    "corruption",
                    "ecgfounder_resample_1000_to_5000_and_global_zscore",
                    "model",
                ]
            )
        ),
        "selected_ref_record_ids_count": int(len(result.get("selected_ref_record_ids", []))),
        "label_mapping": result.get("label_mapping"),
        "class_names": list(CLASS_NAMES_SUPER5),
        "clean_metric_overrides": clean_metric_overrides,
        "per_center": {},
    }
    for center in centers:
        output["per_center"].setdefault(center, {})
        for corruption in args.corruptions:
            output["per_center"][center].setdefault(corruption, {})
            for severity in args.severities:
                output["per_center"][center][corruption][str(severity)] = eval_one(
                    eval_mode,
                    model_or_feature_model,
                    head,
                    result,
                    args,
                    device,
                    center,
                    corruption,
                    int(severity),
                    clean_metric_override=clean_metric_overrides.get(center),
                    input_stabilizer_kwargs=input_stabilizer_kwargs,
                    apply_input_zscore=apply_input_zscore,
                )
    output["aggregate_by_corruption_severity"] = aggregate_corruption_summary(output)

    out_path = Path(args.output_path) if args.output_path else run_dir / "eval_pn2021_c_ecgfounder.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[done] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
