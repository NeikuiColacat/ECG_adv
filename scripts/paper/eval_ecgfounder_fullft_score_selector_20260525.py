#!/usr/bin/env python3
"""K500-val selector over frozen ECGFounder heads and fullFT models.

The older ECGFounder selector can only blend frozen-feature linear heads. This
script adds saved full-fine-tuned ECGFounder models as candidates while keeping
selection paper-safe: candidate/global/classwise choices are made only on the
random K=500 target subset internal validation split, then reported on
ref-excluded PN2021 target-center records.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper.eval_ecgfounder_k500val_multicandidate_selector_20260525 import (  # noqa: E402
    DEFAULT_CANDIDATE_SPECS,
    apply_classwise_options,
    apply_global_option,
    choose_classwise_options,
    choose_global_option,
    find_candidate_head,
    metric_for_scores,
    parse_candidate_specs,
)
from scripts.paper.eval_ecgfounder_score_blend_20260524 import (  # noqa: E402
    DATA_ROOT,
    DEFAULT_CENTERS,
    DEFAULT_K500_RUN_ROOT,
    DEFAULT_LINEAR_PROBE_DIR,
    center_k500_run,
    head_logits,
    sigmoid,
)
from scripts.paper.run_ecgfounder_fullft_super5_pilot_20260523 import (  # noqa: E402
    CHECKPOINT,
    CachedSignalDataset,
    ft_12lead_ECGFounder,
)
from scripts.pgd_cross_center.synth_online_at_super5 import build_k500_internal_val_mask  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


DEFAULT_FULLFT_SPECS = [
    "fullft_direct="
    + str(DATA_ROOT / "paper_ecgfounder_fullft_direct_lowlr_threads8_seed20260531_v6_20260525/runs"),
    "fullft_vae_aw8="
    + str(DATA_ROOT / "paper_ecgfounder_fullft_vae_lowlr_threads8_seed20260531_v6_20260525/runs"),
    "inithead_direct="
    + str(DATA_ROOT / "paper_ecgfounder_fullft_direct_inithead_lowlr_threads8_seed20260531_v6_20260525/runs"),
    "inithead_vae_aw8="
    + str(DATA_ROOT / "paper_ecgfounder_fullft_vae_inithead_lowlr_threads8_seed20260531_v6_20260525/runs"),
    "inithead_vae_aw20="
    + str(DATA_ROOT / "paper_ecgfounder_fullft_vae_inithead_aw20_lowlr_threads8_seed20260531_v6_20260525/runs"),
]


def single_match(paths: list[Path], *, label: str) -> Path:
    uniq = sorted({p for p in paths if p.exists()})
    if len(uniq) != 1:
        found = "\n".join(str(p) for p in uniq[:20])
        raise FileNotFoundError(f"expected one {label}, found {len(uniq)}\n{found}")
    return uniq[0]


def find_fullft_model(root: Path, center: str) -> Path:
    candidates: list[Path] = []
    candidates.extend(root.glob(f"{center}_K500_*/best_model.pt"))
    candidates.extend((root / "runs").glob(f"{center}_K500_*/best_model.pt"))
    return single_match(candidates, label=f"fullFT model for {center} under {root}")


def load_signal_payload(cache_dir: Path, name: str, preprocess_policy: str) -> dict[str, np.ndarray]:
    signal_path = cache_dir / f"{name}_{preprocess_policy}.signals.npy"
    meta_path = cache_dir / f"{name}_{preprocess_policy}.meta.npz"
    if not signal_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing fullFT signal cache: {signal_path} / {meta_path}")
    meta = np.load(meta_path, allow_pickle=True)
    return {
        "signals": np.load(signal_path, mmap_mode="r"),
        "labels": meta["labels"].astype(np.float32),
        "centers": meta["centers"].astype(str),
        "record_ids": meta["record_ids"].astype(str),
        "folds": meta["folds"].astype(np.int64),
    }


def align_center_logits(
    logits_all: np.ndarray,
    feature_centers: np.ndarray,
    feature_record_ids: np.ndarray,
    center: str,
    target_record_ids: np.ndarray,
) -> np.ndarray:
    mask = feature_centers.astype(str) == center
    center_ids = feature_record_ids[mask].astype(str)
    center_logits = logits_all[mask]
    by_id = {str(rid): center_logits[i] for i, rid in enumerate(center_ids)}
    missing = [str(rid) for rid in target_record_ids.astype(str) if str(rid) not in by_id]
    if missing:
        raise RuntimeError(f"{center}: missing {len(missing)} frozen-feature logits; first={missing[:5]}")
    return np.stack([by_id[str(rid)] for rid in target_record_ids.astype(str)]).astype(np.float32)


@torch.no_grad()
def predict_logits(model: nn.Module, ds: CachedSignalDataset, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    outs: list[np.ndarray] = []
    model.eval()
    for x, _y in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
        outs.append(logits.detach().float().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def fullft_logits_cached(
    *,
    model_path: Path,
    cache_path: Path,
    signals: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as d:
            return d["logits"].astype(np.float32)
    model = ft_12lead_ECGFounder(device, str(CHECKPOINT), 5, linear_prob=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    logits = predict_logits(
        model,
        CachedSignalDataset(signals, labels, np.asarray(indices, dtype=np.int64)),
        batch_size,
        device,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, logits=logits, model_path=np.asarray([str(model_path)], dtype=str))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits


def load_selected_ref_ids(k500_eval_path: Path, center: str) -> set[str]:
    with k500_eval_path.open() as f:
        payload = json.load(f)
    by_center = payload.get("selected_ref_record_ids_by_center")
    if isinstance(by_center, dict) and center in by_center:
        return {str(x) for x in by_center[center]}
    return {str(x) for x in payload["selected_ref_record_ids"]}


def metrics_on_indices(labels: np.ndarray, scores: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    return metric_for_scores(labels[indices].astype(np.float32), scores[indices].astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    ap.add_argument("--frozen_candidate", action="append", dest="frozen_candidates", default=[])
    ap.add_argument("--fullft_candidate", action="append", dest="fullft_candidates", default=[])
    ap.add_argument("--use_default_frozen_candidates", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use_default_fullft_candidates", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--blend_space", choices=["prob", "logit"], default="prob")
    ap.add_argument("--linear_probe_dir", default=str(DEFAULT_LINEAR_PROBE_DIR))
    ap.add_argument("--k500_run_root", default=str(DEFAULT_K500_RUN_ROOT))
    ap.add_argument("--fullft_cache_dir", default="")
    ap.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    ap.add_argument("--target_real_val_fraction", type=float, default=0.2)
    ap.add_argument("--target_real_val_seed", type=int, default=20260531)
    ap.add_argument(
        "--class_selection_metric",
        choices=["val_auprc", "val_plus_source_auprc", "val_source_hmean_auprc"],
        default="val_plus_source_auprc",
    )
    ap.add_argument("--source_selection_weight", type=float, default=0.2)
    ap.add_argument(
        "--auroc_selection_weight",
        type=float,
        default=0.0,
        help="Optional classwise selector weight for K500-val AUROC in the primary score.",
    )
    ap.add_argument(
        "--source_auroc_selection_weight",
        type=float,
        default=0.0,
        help="Optional classwise selector weight for PTB-XL fold10 AUROC in the primary score.",
    )
    ap.add_argument("--min_val_auprc_gain", type=float, default=0.0)
    ap.add_argument("--min_val_auroc_gain", type=float, default=0.0)
    ap.add_argument(
        "--min_val_pos_for_candidate",
        type=int,
        default=1,
        help="Require at least this many K500-val positives before allowing classwise candidate selection.",
    )
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--out_dir",
        default=str(DATA_ROOT / "paper_ecgfounder_fullft_score_selector_v6_20260525"),
    )
    args = ap.parse_args()

    frozen_specs = list(DEFAULT_CANDIDATE_SPECS if args.use_default_frozen_candidates else [])
    frozen_specs.extend(args.frozen_candidates)
    frozen_roots = parse_candidate_specs(frozen_specs)
    fullft_specs = list(DEFAULT_FULLFT_SPECS if args.use_default_fullft_candidates else [])
    fullft_specs.extend(args.fullft_candidates)
    fullft_roots = parse_candidate_specs(fullft_specs)
    candidate_names = list(frozen_roots.keys()) + list(fullft_roots.keys())
    if not candidate_names:
        raise RuntimeError("no candidates configured")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits_dir = out_dir / "logits"
    device = torch.device(args.device)

    linear_dir = Path(args.linear_probe_dir)
    pn_feat = np.load(
        linear_dir / f"pn2021_ecgfounder_features_{args.preprocess_policy}.npz",
        allow_pickle=True,
    )
    ptbxl_feat = np.load(
        linear_dir / f"ptbxl_ecgfounder_features_{args.preprocess_policy}.npz",
        allow_pickle=True,
    )
    pn_features = pn_feat["features"].astype(np.float32)
    pn_centers = pn_feat["centers"].astype(str)
    pn_record_ids = pn_feat["record_ids"].astype(str)
    ptbxl_features = ptbxl_feat["features"].astype(np.float32)
    ptbxl_labels_feat = ptbxl_feat["labels"].astype(np.float32)
    fold10 = ptbxl_feat["folds"].astype(np.int64) == 10

    fullft_cache_dir = (
        Path(args.fullft_cache_dir)
        if args.fullft_cache_dir
        else DATA_ROOT / "paper_ecgfounder_fullft_vae_lowlr_seed20260531_v6_20260525/cache"
    )
    ptbxl_payload = load_signal_payload(fullft_cache_dir, "ptbxl", args.preprocess_policy)
    source_indices = np.nonzero(ptbxl_payload["folds"] == 10)[0]
    source_labels = ptbxl_payload["labels"][source_indices].astype(np.float32)
    if len(source_labels) != int(fold10.sum()):
        raise RuntimeError("PTB-XL fold10 size mismatch between feature cache and fullFT signal cache")

    payload: dict[str, Any] = {
        "method": "ECGFounder frozen/fullFT score selector selected on K500-internal validation",
        "class_names": list(CLASS_NAMES_SUPER5),
        "blend_space": args.blend_space,
        "alphas": args.alphas,
        "frozen_candidate_roots": {k: str(v) for k, v in frozen_roots.items()},
        "fullft_candidate_roots": {k: str(v) for k, v in fullft_roots.items()},
        "target_real_val_fraction": args.target_real_val_fraction,
        "target_real_val_seed": args.target_real_val_seed,
        "class_selection_metric": args.class_selection_metric,
        "source_selection_weight": args.source_selection_weight,
        "auroc_selection_weight": args.auroc_selection_weight,
        "source_auroc_selection_weight": args.source_auroc_selection_weight,
        "centers": {},
    }
    rows: list[dict[str, Any]] = []

    for center in args.centers:
        print(f"[center] {center}", flush=True)
        center_payload = load_signal_payload(fullft_cache_dir, center, args.preprocess_policy)
        labels = center_payload["labels"].astype(np.float32)
        record_ids = center_payload["record_ids"].astype(str)
        all_indices = np.arange(len(record_ids), dtype=np.int64)

        k500_run = center_k500_run(Path(args.k500_run_root), center)
        direct_head = k500_run / "best_head.pt"
        k500_eval_path = k500_run / "eval_result.json"
        ref_ids = load_selected_ref_ids(k500_eval_path, center)
        k500_idx = np.asarray([i for i, rid in enumerate(record_ids) if str(rid) in ref_ids], dtype=np.int64)
        if len(k500_idx) == 0:
            raise RuntimeError(f"{center}: no K500 records found in fullFT signal cache")
        internal_val = build_k500_internal_val_mask(
            labels[k500_idx],
            val_fraction=args.target_real_val_fraction,
            seed=args.target_real_val_seed,
        )
        val_indices = k500_idx[internal_val]
        eval_indices = np.asarray([i for i, rid in enumerate(record_ids) if str(rid) not in ref_ids], dtype=np.int64)

        head_paths = {"direct": direct_head}
        for name, root in frozen_roots.items():
            head_paths[name] = find_candidate_head(root, center)
        fullft_paths = {name: find_fullft_model(root, center) for name, root in fullft_roots.items()}

        logits: dict[str, np.ndarray] = {}
        source_logits: dict[str, np.ndarray] = {}
        for name, path in head_paths.items():
            pn_all_logits = head_logits(path, pn_features)
            logits[name] = align_center_logits(pn_all_logits, pn_centers, pn_record_ids, center, record_ids)
            source_logits[name] = head_logits(path, ptbxl_features)[fold10].astype(np.float32)
        for name, path in fullft_paths.items():
            logits[name] = fullft_logits_cached(
                model_path=path,
                cache_path=logits_dir / f"{center}_{name}_target_logits.npz",
                signals=center_payload["signals"],
                labels=labels,
                indices=all_indices,
                batch_size=args.eval_batch_size,
                device=device,
            )
            source_logits[name] = fullft_logits_cached(
                model_path=path,
                cache_path=logits_dir / f"{center}_{name}_source_fold10_logits.npz",
                signals=ptbxl_payload["signals"],
                labels=ptbxl_payload["labels"],
                indices=source_indices,
                batch_size=args.eval_batch_size,
                device=device,
            )

        val_logits = {name: arr[val_indices] for name, arr in logits.items()}
        y_val = labels[val_indices]
        global_choice = choose_global_option(
            y_val,
            val_logits,
            candidate_names=candidate_names,
            alphas=args.alphas,
            blend_space=args.blend_space,
        )
        class_choices = choose_classwise_options(
            y_val,
            val_logits,
            source_y_true=source_labels,
            source_logits=source_logits,
            candidate_names=candidate_names,
            alphas=args.alphas,
            blend_space=args.blend_space,
            class_selection_metric=args.class_selection_metric,
            source_selection_weight=args.source_selection_weight,
            auroc_selection_weight=args.auroc_selection_weight,
            source_auroc_selection_weight=args.source_auroc_selection_weight,
            min_val_auprc_gain=args.min_val_auprc_gain,
            min_val_auroc_gain=args.min_val_auroc_gain,
        )
        if args.min_val_pos_for_candidate > 1:
            for idx, cls in enumerate(CLASS_NAMES_SUPER5):
                n_pos = int((y_val[:, idx] > 0.5).sum())
                if n_pos < int(args.min_val_pos_for_candidate):
                    old_choice = dict(class_choices[cls])
                    class_choices[cls] = {
                        "candidate": "direct",
                        "alpha": 0.0,
                        "n_pos": n_pos,
                        "n_neg": int(len(y_val) - n_pos),
                        "selection": "direct_fallback_min_val_pos",
                        "rejected_best": old_choice,
                    }

        method_scores = {"direct": sigmoid(logits["direct"])}
        for name in candidate_names:
            method_scores[name] = sigmoid(logits[name])
        method_scores["global_k500val_selector"] = apply_global_option(
            logits, global_choice, blend_space=args.blend_space
        )
        method_scores["classwise_k500val_selector"] = apply_classwise_options(
            logits, class_choices, blend_space=args.blend_space
        )

        source_scores = {"direct": sigmoid(source_logits["direct"])}
        for name in candidate_names:
            source_scores[name] = sigmoid(source_logits[name])
        source_scores["global_k500val_selector"] = apply_global_option(
            source_logits, global_choice, blend_space=args.blend_space
        )
        source_scores["classwise_k500val_selector"] = apply_classwise_options(
            source_logits, class_choices, blend_space=args.blend_space
        )
        source_metrics = {
            name: metric_for_scores(source_labels, scores.astype(np.float32))
            for name, scores in source_scores.items()
        }

        target_metrics: dict[str, Any] = {
            name: metrics_on_indices(labels, scores, eval_indices)
            for name, scores in method_scores.items()
        }
        direct_target = target_metrics["direct"]
        for method_name, target in target_metrics.items():
            source = source_metrics[method_name]
            rows.append(
                {
                    "center": center,
                    "method": method_name,
                    "target_auroc": target["macro_auroc"],
                    "target_auprc": target["macro_auprc"],
                    "delta_vs_direct_auroc": target["macro_auroc"] - direct_target["macro_auroc"],
                    "delta_vs_direct_auprc": target["macro_auprc"] - direct_target["macro_auprc"],
                    "ptbxl_fold10_auroc": source["macro_auroc"],
                    "ptbxl_fold10_auprc": source["macro_auprc"],
                    "global_choice": json.dumps(global_choice["best"], sort_keys=True),
                    "classwise_choices": json.dumps(
                        {
                            k: {"candidate": v["candidate"], "alpha": v["alpha"]}
                            for k, v in class_choices.items()
                        },
                        sort_keys=True,
                    ),
                }
            )

        payload["centers"][center] = {
            "head_paths": {k: str(v) for k, v in head_paths.items()},
            "fullft_paths": {k: str(v) for k, v in fullft_paths.items()},
            "k500_eval_path": str(k500_eval_path),
            "k500_val": {
                "n": int(len(val_indices)),
                "positive_counts": labels[val_indices].sum(axis=0).astype(int).tolist(),
                "record_ids": record_ids[val_indices].tolist(),
            },
            "global_selection": global_choice,
            "classwise_selection": class_choices,
            "target": target_metrics,
            "ptbxl_fold10": source_metrics,
        }
        selected = target_metrics["classwise_k500val_selector"]
        print(
            f"  classwise={selected['macro_auroc']:.6f}/{selected['macro_auprc']:.6f} "
            f"delta={(selected['macro_auroc'] - direct_target['macro_auroc']):+.6f}/"
            f"{(selected['macro_auprc'] - direct_target['macro_auprc']):+.6f}",
            flush=True,
        )

    for method_name in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method_name]
        rows.append(
            {
                "center": "MEAN",
                "method": method_name,
                "target_auroc": float(np.mean([float(r["target_auroc"]) for r in subset])),
                "target_auprc": float(np.mean([float(r["target_auprc"]) for r in subset])),
                "delta_vs_direct_auroc": float(np.mean([float(r["delta_vs_direct_auroc"]) for r in subset])),
                "delta_vs_direct_auprc": float(np.mean([float(r["delta_vs_direct_auprc"]) for r in subset])),
                "ptbxl_fold10_auroc": float(np.mean([float(r["ptbxl_fold10_auroc"]) for r in subset])),
                "ptbxl_fold10_auprc": float(np.mean([float(r["ptbxl_fold10_auprc"]) for r in subset])),
                "global_choice": "",
                "classwise_choices": "",
            }
        )

    csv_path = out_dir / f"ecgfounder_fullft_score_selector_{args.blend_space}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / f"ecgfounder_fullft_score_selector_{args.blend_space}.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[summary] wrote {csv_path}", flush=True)
    print(f"[summary] wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
