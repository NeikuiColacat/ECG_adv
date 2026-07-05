#!/usr/bin/env python
"""Select an EfficientNet checkpoint with K500 internal corrupted validation.

This is a post-hoc diagnostic/selection helper for VAE-LHAT raw-AugMix runs.
It uses only the target K500 internal validation split, not PN2021 heldout
labels, so it can test a paper-safe robust checkpoint-selection rule.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ecg_adv_gen.adaptation import build_k500_internal_val_mask
from ecg_adv_gen.data.waveform_datasets import apply_effnet_input_stabilizer, has_input_stabilizer
from ecg_adv_gen.evaluation import compute_macro_auroc_auprc
from ecg_adv_gen.evaluation.inference import infer_dataset
from ecg_adv_gen.evaluation.pn2021_corruptions import stable_corruption_seed
from ecg_adv_gen.evaluation.pn2021c import build_corruption_op as _build_corruption_op
from ecg_adv_gen.labels import CLASS_NAMES_SUPER5, NUM_SUPER5
from ecg_adv_gen.models.super5_model_zoo import build_super5_model, normalize_model_name


@dataclass(frozen=True)
class CheckpointCandidate:
    name: str
    path: Path
    kind: str
    epoch: int | None = None


@dataclass(frozen=True)
class CandidateScore:
    name: str
    clean_macro_auprc: float
    corrupted_macro_auprc: float
    clean_macro_auroc: float
    corrupted_macro_auroc: float
    n_val: int
    path: str = ""
    kind: str = ""
    epoch: int | None = None
    per_op: dict[str, dict[str, float]] | None = None
    clean_pass: bool = True


def _safe_float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _load_checkpoint_epoch(path: Path) -> int | None:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if isinstance(payload, dict) and "epoch" in payload:
        return int(payload["epoch"])
    return None


def discover_checkpoint_candidates(run_dir: str | Path) -> list[CheckpointCandidate]:
    run_path = Path(run_dir)
    candidates: list[CheckpointCandidate] = []
    selected = run_path / "best_model.pt"
    if selected.exists():
        candidates.append(
            CheckpointCandidate(name="selected", path=selected, kind="selected", epoch=None)
        )
    latest = run_path / "checkpoints" / "checkpoint_latest.pt"
    if latest.exists():
        candidates.append(
            CheckpointCandidate(
                name="latest",
                path=latest,
                kind="latest",
                epoch=_load_checkpoint_epoch(latest),
            )
        )
    return candidates


def select_best_candidate(
    scores: Sequence[CandidateScore],
    *,
    clean_auprc_floor: float,
    clean_auroc_floor: float = 0.0,
) -> CandidateScore:
    if not scores:
        raise ValueError("scores must not be empty")
    eligible = [
        score
        for score in scores
        if score.clean_macro_auprc >= clean_auprc_floor
        and score.clean_macro_auroc >= clean_auroc_floor
    ]
    if not eligible:
        eligible = list(scores)
    return max(
        eligible,
        key=lambda score: (
            score.corrupted_macro_auprc,
            score.corrupted_macro_auroc,
            score.clean_macro_auprc,
        ),
    )


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint does not contain a state_dict: {path}")
    return {str(k).removeprefix("_orig_mod."): v for k, v in payload.items()}


def _load_model(candidate: CheckpointCandidate, model_name: str, device: str) -> torch.nn.Module:
    model = build_super5_model(
        normalize_model_name(model_name),
        num_classes=NUM_SUPER5,
    ).to(device)
    model.load_state_dict(_load_state_dict(candidate.path))
    model.eval()
    return model


def _load_target_val_subset(
    target_real_npz: str | Path,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(target_real_npz, allow_pickle=True) as data:
        signals = np.asarray(data["signals"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        record_ids = (
            data["record_ids"].astype(str)
            if "record_ids" in data.files
            else np.asarray([str(i) for i in range(labels.shape[0])])
        )
    if signals.ndim != 3:
        raise ValueError(f"signals must be 3D, got {signals.shape}")
    if signals.shape[1:] == (12, 1000):
        signals = signals.transpose(0, 2, 1)
    if signals.shape[1:] != (1000, 12):
        raise ValueError(f"signals must be (N,1000,12), got {signals.shape}")
    if labels.shape[0] != signals.shape[0] or labels.shape[1] != NUM_SUPER5:
        raise ValueError(f"labels mismatch: signals={signals.shape} labels={labels.shape}")
    mask = build_k500_internal_val_mask(labels, val_fraction=val_fraction, seed=seed)
    return signals[mask].astype(np.float32), labels[mask].astype(np.float32), record_ids[mask]


def _center_crop_ct(signals_ct: np.ndarray, crop_len: int) -> np.ndarray:
    start = max((signals_ct.shape[-1] - int(crop_len)) // 2, 0)
    return signals_ct[..., start : start + int(crop_len)].astype(np.float32, copy=False)


def _apply_input_stabilizer_batch(
    signals_ct: np.ndarray,
    input_stabilizer_config: dict[str, Any] | None,
) -> np.ndarray:
    config = dict(input_stabilizer_config or {})
    if not has_input_stabilizer(config):
        return np.asarray(signals_ct, dtype=np.float32)
    return np.stack(
        [
            apply_effnet_input_stabilizer(
                torch.from_numpy(np.ascontiguousarray(sample)).float(),
                config,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
            for sample in np.asarray(signals_ct, dtype=np.float32)
        ],
        axis=0,
    )


def _prepare_eval_signals_ct(
    signals_tc: np.ndarray,
    *,
    crop_len: int,
    input_stabilizer_config: dict[str, Any] | None = None,
) -> np.ndarray:
    signals_ct = np.transpose(signals_tc, (0, 2, 1)).astype(np.float32)
    config = dict(input_stabilizer_config or {})
    stage = str(config.get("stage", "post_crop"))
    if stage == "pre_crop":
        return _center_crop_ct(_apply_input_stabilizer_batch(signals_ct, config), crop_len)
    cropped = _center_crop_ct(signals_ct, crop_len)
    return _apply_input_stabilizer_batch(cropped, config)


def _infer_metrics(
    model: torch.nn.Module,
    signals_ct: np.ndarray,
    labels: np.ndarray,
    *,
    device: str,
    batch_size: int,
    min_pos: int,
) -> dict[str, Any]:
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(signals_ct)).float(),
        torch.from_numpy(np.asarray(labels, dtype=np.float32)).float(),
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0)
    y_true, y_score = infer_dataset(model, loader, device)
    return compute_macro_auroc_auprc(y_true, y_score, CLASS_NAMES_SUPER5, min_pos=min_pos)


def _corrupt_signals(
    signals_ct: np.ndarray,
    *,
    op_name: str,
    severity: int,
    severity_profile: str,
    seed: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    for idx, sample in enumerate(signals_ct):
        sample_seed = stable_corruption_seed(seed, op_name, severity, idx)
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        op = _build_corruption_op(op_name, severity, severity_profile)
        corrupted = op(torch.from_numpy(np.ascontiguousarray(sample)).float())
        out.append(corrupted.numpy().astype(np.float32, copy=False))
    return np.stack(out).astype(np.float32)


def _prepare_corrupted_eval_signals_ct(
    signals_tc: np.ndarray,
    *,
    op_name: str,
    severity: int,
    severity_profile: str,
    seed: int,
    crop_len: int,
    input_stabilizer_config: dict[str, Any] | None = None,
) -> np.ndarray:
    signals_ct = np.transpose(signals_tc, (0, 2, 1)).astype(np.float32)
    config = dict(input_stabilizer_config or {})
    if str(config.get("stage", "post_crop")) == "pre_crop":
        corrupted = _corrupt_signals(
            signals_ct,
            op_name=op_name,
            severity=severity,
            severity_profile=severity_profile,
            seed=seed,
        )
        return _center_crop_ct(_apply_input_stabilizer_batch(corrupted, config), crop_len)
    corrupted = _corrupt_signals(
        _center_crop_ct(signals_ct, crop_len),
        op_name=op_name,
        severity=severity,
        severity_profile=severity_profile,
        seed=seed,
    )
    return _apply_input_stabilizer_batch(corrupted, config)


def evaluate_candidate(
    candidate: CheckpointCandidate,
    *,
    target_real_npz: str | Path,
    model_name: str,
    device: str,
    val_fraction: float,
    val_seed: int,
    corruptions: Sequence[str],
    severity: int,
    severity_profile: str,
    corruption_seed: int,
    crop_len: int,
    batch_size: int,
    min_pos: int,
    input_stabilizer_config: dict[str, Any] | None = None,
) -> CandidateScore:
    signals_tc, labels, _record_ids = _load_target_val_subset(
        target_real_npz,
        val_fraction=val_fraction,
        seed=val_seed,
    )
    signals_ct = _prepare_eval_signals_ct(
        signals_tc,
        crop_len=crop_len,
        input_stabilizer_config=input_stabilizer_config,
    )
    model = _load_model(candidate, model_name, device)
    clean = _infer_metrics(
        model,
        signals_ct,
        labels,
        device=device,
        batch_size=batch_size,
        min_pos=min_pos,
    )

    op_scores: dict[str, dict[str, float]] = {}
    aurocs: list[float] = []
    auprcs: list[float] = []
    for op_name in corruptions:
        corrupted_ct = _prepare_corrupted_eval_signals_ct(
            signals_tc,
            op_name=op_name,
            severity=severity,
            severity_profile=severity_profile,
            seed=corruption_seed,
            crop_len=crop_len,
            input_stabilizer_config=input_stabilizer_config,
        )
        metrics = _infer_metrics(
            model,
            corrupted_ct,
            labels,
            device=device,
            batch_size=batch_size,
            min_pos=min_pos,
        )
        auroc = _safe_float(metrics["macro_auroc"])
        auprc = _safe_float(metrics["macro_auprc"])
        aurocs.append(auroc)
        auprcs.append(auprc)
        op_scores[str(op_name)] = {"macro_auroc": auroc, "macro_auprc": auprc}

    return CandidateScore(
        name=candidate.name,
        path=str(candidate.path),
        kind=candidate.kind,
        epoch=candidate.epoch,
        clean_macro_auroc=_safe_float(clean["macro_auroc"]),
        clean_macro_auprc=_safe_float(clean["macro_auprc"]),
        corrupted_macro_auroc=float(np.nanmean(aurocs)),
        corrupted_macro_auprc=float(np.nanmean(auprcs)),
        n_val=int(labels.shape[0]),
        per_op=op_scores,
    )


def run_selection(args: argparse.Namespace) -> dict[str, Any]:
    candidates = discover_checkpoint_candidates(args.run_dir)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint candidates found under {args.run_dir}")
    input_stabilizer_config = {
        "bandpass_low_hz": args.input_bandpass_low_hz,
        "bandpass_high_hz": args.input_bandpass_high_hz,
        "repair_flat_leads": bool(args.input_repair_flat_leads),
        "clip_abs": args.input_clip_abs,
        "renorm_after_stabilizer": bool(args.input_renorm_after_stabilizer),
        "sample_rate_hz": float(args.input_sample_rate_hz),
        "stage": str(args.input_stabilizer_stage),
    }
    scores = [
        evaluate_candidate(
            candidate,
            target_real_npz=args.target_real_npz,
            model_name=args.model_name,
            device=args.device,
            val_fraction=args.target_real_val_fraction,
            val_seed=args.target_real_val_seed,
            corruptions=args.corruptions,
            severity=args.severity,
            severity_profile=args.severity_profile,
            corruption_seed=args.seed,
            crop_len=args.crop_len,
            batch_size=args.batch_size,
            min_pos=args.min_pos,
            input_stabilizer_config=input_stabilizer_config,
        )
        for candidate in candidates
    ]
    best_clean_auprc = max(score.clean_macro_auprc for score in scores)
    best_clean_auroc = max(score.clean_macro_auroc for score in scores)
    clean_auprc_floor = max(0.0, best_clean_auprc - float(args.clean_auprc_floor_delta))
    clean_auroc_floor = max(0.0, best_clean_auroc - float(args.clean_auroc_floor_delta))
    adjusted_scores = [
        CandidateScore(
            **{
                **asdict(score),
                "clean_pass": (
                    score.clean_macro_auprc >= clean_auprc_floor
                    and score.clean_macro_auroc >= clean_auroc_floor
                ),
            }
        )
        for score in scores
    ]
    selected = select_best_candidate(
        adjusted_scores,
        clean_auprc_floor=clean_auprc_floor,
        clean_auroc_floor=clean_auroc_floor,
    )
    return {
        "schema_version": 1,
        "run_dir": str(args.run_dir),
        "target_real_npz": str(args.target_real_npz),
        "model_name": args.model_name,
        "selection_rule": {
            "metric": "corrupted_macro_auprc",
            "clean_auprc_floor": clean_auprc_floor,
            "clean_auroc_floor": clean_auroc_floor,
            "clean_auprc_floor_delta": float(args.clean_auprc_floor_delta),
            "clean_auroc_floor_delta": float(args.clean_auroc_floor_delta),
            "severity_profile": args.severity_profile,
            "severity": int(args.severity),
            "corruptions": list(args.corruptions),
            "target_real_val_fraction": float(args.target_real_val_fraction),
            "target_real_val_seed": int(args.target_real_val_seed),
            "input_stabilizer": (
                input_stabilizer_config if has_input_stabilizer(input_stabilizer_config) else {}
            ),
            "heldout_target_labels_used": False,
        },
        "selected": asdict(selected),
        "candidates": [asdict(score) for score in adjusted_scores],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--target_real_npz", required=True)
    parser.add_argument("--model_name", default="efficientnet1dv2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target_real_val_fraction", type=float, default=0.2)
    parser.add_argument("--target_real_val_seed", type=int, default=20260601)
    parser.add_argument("--corruptions", nargs="+", default=[
        "powerline_noise",
        "emg_noise",
        "baseline_wander",
        "baseline_shift",
        "random_leads_masking",
    ])
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--severity_profile", default="calibrated_10to20pp")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--crop_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_pos", type=int, default=10)
    parser.add_argument("--clean_auprc_floor_delta", type=float, default=0.02)
    parser.add_argument("--clean_auroc_floor_delta", type=float, default=0.03)
    parser.add_argument("--input_bandpass_low_hz", type=float, default=None)
    parser.add_argument("--input_bandpass_high_hz", type=float, default=None)
    parser.add_argument("--input_repair_flat_leads", action="store_true")
    parser.add_argument("--input_clip_abs", type=float, default=None)
    parser.add_argument("--input_renorm_after_stabilizer", action="store_true")
    parser.add_argument("--input_sample_rate_hz", type=float, default=100.0)
    parser.add_argument("--input_stabilizer_stage", choices=["post_crop", "pre_crop"], default="post_crop")
    parser.add_argument("--output_json", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_selection(args)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
