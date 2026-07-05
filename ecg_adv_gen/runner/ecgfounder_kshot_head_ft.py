#!/usr/bin/env python3
"""ECGFounder frozen-feature K-shot head fine-tune baseline.

Requires ECGFounder linear-probe feature caches.
For each target center, initialize from the PTB-XL Super5 linear-probe head,
fine-tune only that head on the same K=500 target records, and evaluate PN2021
with those K=500 records excluded from the target-center test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        str(_MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")),
    )
)

from ecg_adv_gen.runner.ecgfounder_linear_probe import (  # noqa: E402
    DEFAULT_OUT_DIR as LINEAR_PROBE_DIR,
    REF_ROOT,
    compute_metrics,
    evaluate_pn2021_views,
)
from ecg_adv_gen.labels import CLASS_NAMES_SUPER5  # noqa: E402
from ecg_adv_gen.data import (  # noqa: E402
    PN2021_TARGET_CENTERS_4,
    load_kshot_ref_record_ids,
    select_kshot_indices_from_ref_root,
)
from ecg_adv_gen.labels import pn2021_super5_label_mapping_payload  # noqa: E402
from ecg_adv_gen.models import ecgfounder_kshot_head_run_dir  # noqa: E402
from ecg_adv_gen.models.ecgfounder_inference import predict_feature_head, sigmoid_clipped  # noqa: E402
from ecg_adv_gen.training import compute_pos_weight, masked_bce_with_logits, random_split_indices  # noqa: E402


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
DEFAULT_OUT_DIR = DATA_ROOT / "paper_foundation_baselines_20260524/ecgfounder_kshot_head_ft_v6_from_legacy_cache"
TARGET_CENTERS = list(PN2021_TARGET_CENTERS_4)


def load_ref_ids(
    center: str,
    k: int,
    seed: int,
    source_k: int | None = None,
    ref_root: str | Path = REF_ROOT,
) -> list[str]:
    return load_kshot_ref_record_ids(
        ref_root,
        center,
        k=k,
        seed=seed,
        source_k=source_k,
    )


def select_kshot_indices(
    labels: np.ndarray,
    centers: np.ndarray,
    record_ids: np.ndarray,
    center: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str]]:
    """Select K records from an existing source-K ref pool.

    If an exact K ref-meta exists we use it. Otherwise, fall back to the
    source_k pool, usually K=500, and draw a deterministic stratified subset.
    """
    return select_kshot_indices_from_ref_root(
        labels,
        centers,
        record_ids,
        center=center,
        ref_root=args.ref_root,
        k=args.k,
        source_k=args.source_k,
        subset_seed=args.subset_seed,
        seed=args.seed,
        n_classes=len(CLASS_NAMES_SUPER5),
    )


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return random_split_indices(n, val_fraction, seed, zero_val_policy="permuted")


def predict_head(head: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    return predict_feature_head(head, features, batch_size=batch_size, device=device)


@torch.no_grad()
def eval_subset(head: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    logits = head(x.to(device)).cpu().numpy()
    scores = sigmoid_clipped(logits)
    metric = compute_metrics(y.numpy(), scores, min_pos=1)
    return float(metric["macro_auprc"]) if metric["macro_auprc"] is not None else -float("inf")


def train_one(center: str, args: argparse.Namespace, pn_payload: dict, base_head_path: Path, out_dir: Path) -> dict:
    device = torch.device(args.device)
    center_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(center))
    run_seed = int(args.seed + center_offset + args.k)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)
    run_dir = ecgfounder_kshot_head_run_dir(
        out_dir,
        center=center,
        k=args.k,
        source_k=args.source_k,
        epochs=args.epochs,
        seed=args.seed,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "eval_result.json"
    if result_path.exists() and not args.force:
        with result_path.open() as f:
            return json.load(f)

    features = pn_payload["features"].astype(np.float32, copy=False)
    labels = pn_payload["labels"].astype(np.float32, copy=False)
    centers = pn_payload["centers"].astype(str)
    record_ids = pn_payload["record_ids"].astype(str)
    idx_all, selected_ref_ids = select_kshot_indices(labels, centers, record_ids, center, args)
    train_rel, val_rel = split_indices(len(idx_all), args.val_fraction, args.seed)
    train_idx = idx_all[train_rel]
    val_idx = idx_all[val_rel]

    head = nn.Linear(features.shape[1], len(CLASS_NAMES_SUPER5)).to(device)
    head.load_state_dict(torch.load(base_head_path, map_location=device))
    if args.reset_head:
        head.reset_parameters()

    x_train = torch.from_numpy(features[train_idx]).float()
    y_train = torch.from_numpy(labels[train_idx]).float()
    x_val = torch.from_numpy(features[val_idx]).float()
    y_val = torch.from_numpy(labels[val_idx]).float()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    pos_weight = torch.tensor(
        compute_pos_weight(labels[train_idx], len(CLASS_NAMES_SUPER5), clip_max=50.0),
        dtype=torch.float32,
        device=device,
    )

    def criterion(logits, y):
        return masked_bce_with_logits(logits, y, pos_weight)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    best = -float("inf")
    logs = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(head(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        val_auprc = eval_subset(head, x_val, y_val, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_auprc": val_auprc,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        logs.append(row)
        if val_auprc > best or not (run_dir / "best_head.pt").exists():
            best = val_auprc
            torch.save(head.state_dict(), run_dir / "best_head.pt")
        with (run_dir / "training_log.json").open("w") as f:
            json.dump(logs, f, indent=2)
    head.load_state_dict(torch.load(run_dir / "best_head.pt", map_location=device))
    scores = predict_head(head, features, args.eval_batch_size, device)
    ref_by_center = {c: set() for c in TARGET_CENTERS}
    selected_by_center = {c: [] for c in TARGET_CENTERS}
    selected_by_center[center] = selected_ref_ids
    ref_by_center[center] = set(selected_ref_ids)
    views = evaluate_pn2021_views(labels, scores, centers, record_ids, ref_by_center)
    result = {
        "method": "ECGFounder frozen feature + K-shot target head fine-tune",
        "center": center,
        "K": int(args.k),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "selected_ref_record_ids": selected_ref_ids,
        "selected_ref_record_ids_by_center": selected_by_center,
        "best_val_macro_auprc": best,
        "target_view": views[center],
        "all_views": views,
        "base_head": str(base_head_path),
        "label_mapping": pn2021_super5_label_mapping_payload(),
        "reset_head": bool(args.reset_head),
        "pos_weight": pos_weight.detach().cpu().tolist(),
        "config": vars(args),
        "run_seed": run_seed,
    }
    with result_path.open("w") as f:
        json.dump(result, f, indent=2)
    return result


def write_summary(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ecgfounder_kshot_head_ft_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["center", "target_auroc", "target_auprc", "pn2021_avg_auroc", "pn2021_avg_auprc", "eval_path"])
        for row in rows:
            c = row["center"]
            view = row["target_view"]
            target = view["per_center"][c]
            writer.writerow([
                c,
                target["macro_auroc"],
                target["macro_auprc"],
                view["avg_macro_auroc"],
                view["avg_macro_auprc"],
                str(
                    ecgfounder_kshot_head_run_dir(
                        out_dir,
                        center=c,
                        k=int(row["K"]),
                        source_k=int(row["config"].get("source_k", 500)),
                        epochs=int(row["config"]["epochs"]),
                        seed=int(row["config"]["seed"]),
                    )
                    / "eval_result.json"
                ),
            ])
    print(f"[summary] wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=TARGET_CENTERS)
    p.add_argument("--linear_probe_dir", default=str(DEFAULT_LINEAR_PROBE_DIR))
    p.add_argument(
        "--preprocess_policy",
        default="official_ptbxl_eval",
        choices=["official_ptbxl_eval", "filtered_dataset"],
        help="Feature-cache suffix from the ECGFounder linear-probe cache.",
    )
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument(
        "--ref_root",
        default=str(REF_ROOT),
        help="Root containing per-center K-shot ref_meta files used for selection and ref-excluded eval.",
    )
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--source_k", type=int, default=500,
                   help="Existing ref pool size to subsample from when exact K metadata is absent.")
    p.add_argument("--subset_seed", type=int, default=20260531)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=4096)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--reset_head", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    linear_dir = Path(args.linear_probe_dir)
    out_dir = Path(args.out_dir)
    pn_feature_candidates = [
        linear_dir / f"pn2021_ecgfounder_features_{args.preprocess_policy}.npz",
        linear_dir / "pn2021_ecgfounder_features.npz",
    ]
    pn_feature_path = next((p for p in pn_feature_candidates if p.exists()), pn_feature_candidates[0])
    base_head_path = linear_dir / "best_head.pt"
    if not pn_feature_path.exists() or not base_head_path.exists():
        raise FileNotFoundError(f"missing feature/head cache: {pn_feature_path}, {base_head_path}")
    pn = np.load(pn_feature_path, allow_pickle=True)
    pn_payload = {k: pn[k] for k in pn.files}
    rows = []
    for center in args.centers:
        result = train_one(center, args, pn_payload, base_head_path, out_dir)
        rows.append(result)
        c = result["center"]
        target = result["target_view"]["per_center"][c]
        print(
            f"[result] {c} target={target['macro_auroc']:.4f}/{target['macro_auprc']:.4f} "
            f"avg={result['target_view']['avg_macro_auroc']:.4f}/{result['target_view']['avg_macro_auprc']:.4f}",
            flush=True,
        )
        write_summary(rows, out_dir)
    write_summary(rows, out_dir)


if __name__ == "__main__":
    main()
