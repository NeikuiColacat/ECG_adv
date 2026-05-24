#!/usr/bin/env python3
"""Re-evaluate saved ECGFounder Super5 head-adaptation runs.

This is intentionally evaluation-only.  It reloads each run's best_head.pt,
reconstructs the head from eval_result.json/config, and reports PN2021 metrics
with and without all-zero Super5 rows.  It avoids rerunning latent-hull training
just to audit whether a saved result was inflated by all-zero labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


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

from scripts.paper.run_ecgfounder_vae_only_lhat_head_ft_20260523 import (  # noqa: E402
    DEFAULT_LINEAR_PROBE_DIR,
    ResidualAdapterHead,
    eval_pn,
    eval_ptbxl_fold10,
    ref_ids_for_all_centers,
)
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_config(result: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(result.get("config") or {})
    if "head_type" not in cfg and result.get("head_type"):
        cfg["head_type"] = result["head_type"]
    return cfg


def build_head(result: dict[str, Any], feature_dim: int, device: torch.device) -> tuple[nn.Module, nn.Module]:
    cfg = load_config(result)
    linear_probe_dir = Path(cfg.get("linear_probe_dir") or DEFAULT_LINEAR_PROBE_DIR)
    source_head = nn.Linear(feature_dim, len(CLASS_NAMES_SUPER5)).to(device)
    source_head.load_state_dict(torch.load(linear_probe_dir / "best_head.pt", map_location=device))
    source_head.eval()

    base_head = nn.Linear(feature_dim, len(CLASS_NAMES_SUPER5)).to(device)
    base_head.load_state_dict(torch.load(linear_probe_dir / "best_head.pt", map_location=device))
    if cfg.get("head_type", "linear") == "residual_adapter":
        head = ResidualAdapterHead(
            base_head=base_head,
            hidden_dim=int(cfg.get("adapter_hidden", 128)),
            dropout=float(cfg.get("adapter_dropout", 0.0)),
            scale=float(cfg.get("adapter_scale", 1.0)),
            freeze_base=bool(cfg.get("freeze_base_head", False)),
        ).to(device)
    else:
        head = base_head
    return source_head, head


def reevaluate_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    result_path = run_dir / "eval_result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    result = json.load(result_path.open())
    cfg = load_config(result)
    linear_probe_dir = Path(cfg.get("linear_probe_dir") or DEFAULT_LINEAR_PROBE_DIR)
    preprocess_policy = cfg.get("preprocess_policy", args.preprocess_policy)
    ptbxl_path = linear_probe_dir / f"ptbxl_ecgfounder_features_{preprocess_policy}.npz"
    pn_path = linear_probe_dir / f"pn2021_ecgfounder_features_{preprocess_policy}.npz"
    if not ptbxl_path.exists():
        ptbxl_path = linear_probe_dir / "ptbxl_ecgfounder_features.npz"
    if not pn_path.exists():
        pn_path = linear_probe_dir / "pn2021_ecgfounder_features.npz"
    ptbxl = load_npz(ptbxl_path)
    pn = load_npz(pn_path)

    device = torch.device(args.device)
    source_head, head = build_head(result, int(ptbxl["features"].shape[1]), device)
    head.load_state_dict(torch.load(run_dir / "best_head.pt", map_location=device))
    head.eval()

    center = result["center"]
    selected_ids = np.asarray(result["selected_ref_record_ids"], dtype=str)
    ref_ids = ref_ids_for_all_centers(center, selected_ids)
    eval_batch_size = int(cfg.get("eval_batch_size", args.eval_batch_size))

    baseline_pn = eval_pn(
        source_head,
        pn,
        ref_ids,
        device,
        eval_batch_size,
        report_drop_all_zero=True,
    )
    final_pn = eval_pn(
        head,
        pn,
        ref_ids,
        device,
        eval_batch_size,
        report_drop_all_zero=True,
    )
    baseline_ptbxl = eval_ptbxl_fold10(source_head, ptbxl, device, eval_batch_size)
    final_ptbxl = eval_ptbxl_fold10(head, ptbxl, device, eval_batch_size)
    out = {
        "run_dir": str(run_dir),
        "center": center,
        "method": result.get("method"),
        "class_names": list(CLASS_NAMES_SUPER5),
        "config_subset": {
            "head_type": cfg.get("head_type", "linear"),
            "k": result.get("K", cfg.get("k")),
            "hull_m": cfg.get("hull_m"),
            "hull_lambda": cfg.get("hull_lambda"),
            "epochs": cfg.get("epochs"),
            "source_weight": cfg.get("source_weight"),
            "target_real_weight": cfg.get("target_real_weight"),
            "adv_weight": cfg.get("adv_weight"),
            "source_logit_anchor_weight": cfg.get("source_logit_anchor_weight"),
        },
        "baseline_pn2021_views": baseline_pn,
        "final_pn2021_views": final_pn,
        "baseline_ptbxl_fold10": baseline_ptbxl,
        "final_ptbxl_fold10": final_ptbxl,
    }
    out_path = run_dir / "drop_all_zero_reeval.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dirs", nargs="+", required=True)
    p.add_argument("--out_csv", default="")
    p.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    p.add_argument("--eval_batch_size", type=int, default=4096)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    rows = [reevaluate_run(Path(x), args) for x in args.run_dirs]
    if args.out_csv:
        csv_path = Path(args.out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "center",
                "run_dir",
                "baseline_target_auroc",
                "baseline_target_auprc",
                "final_target_auroc",
                "final_target_auprc",
                "baseline_drop_all_zero_auroc",
                "baseline_drop_all_zero_auprc",
                "final_drop_all_zero_auroc",
                "final_drop_all_zero_auprc",
                "final_ptbxl_auroc",
                "final_ptbxl_auprc",
            ])
            for r in rows:
                c = r["center"]
                b = r["baseline_pn2021_views"][c]["per_center"][c]
                m = r["final_pn2021_views"][c]["per_center"][c]
                p10 = r["final_ptbxl_fold10"]
                w.writerow([
                    c,
                    r["run_dir"],
                    b["macro_auroc"],
                    b["macro_auprc"],
                    m["macro_auroc"],
                    m["macro_auprc"],
                    b.get("drop_all_zero_macro_auroc"),
                    b.get("drop_all_zero_macro_auprc"),
                    m.get("drop_all_zero_macro_auroc"),
                    m.get("drop_all_zero_macro_auprc"),
                    p10["macro_auroc"],
                    p10["macro_auprc"],
                ])
        print(f"[done] wrote {csv_path}")
    for r in rows:
        c = r["center"]
        m = r["final_pn2021_views"][c]["per_center"][c]
        print(
            f"{c}: final={m['macro_auroc']:.4f}/{m['macro_auprc']:.4f} "
            f"drop={m.get('drop_all_zero_macro_auroc'):.4f}/"
            f"{m.get('drop_all_zero_macro_auprc'):.4f}"
        )


if __name__ == "__main__":
    main()
