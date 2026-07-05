"""Shared helpers for EfficientNet direct K-shot fine-tune wrappers."""

from __future__ import annotations

import json
import os
from pathlib import Path


_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
_DEFAULT_DATA_ROOT = _MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")

PYTHON = os.environ.get(
    "ECGTWIN_PYTHON",
    "/home/linbinhao/micromamba/envs/ECGTwin/bin/python"
    if Path("/home/linbinhao/micromamba/envs/ECGTwin/bin/python").exists()
    else "/root/miniforge3/envs/ECGTwin/bin/python",
)
PROJECT_ROOT = Path(os.environ.get("ECG_ADV_GEN_PROJECT_ROOT", str(_DEFAULT_PROJECT_ROOT)))
DATA_ROOT = Path(os.environ.get("ECG_ADV_GEN_DATA_ROOT", str(_DEFAULT_DATA_ROOT)))
BASELINE_CKPT = str(
    DATA_ROOT
    / "triple_labels"
    / "super5_minresample_full10_perglobal_20260503"
    / "best_model.pt"
)
PN2021_CACHE_DIR = str(DATA_ROOT / "triple_labels" / "pn2021_eval_cache_minresample_perglobal")
PN2021_MMAP_CACHE_DIR = str(
    DATA_ROOT / "triple_labels" / "pn2021_eval_cache_mmap_minresample_perglobal"
)


def eval_metrics(eval_path: Path) -> dict[str, float]:
    data = json.load(eval_path.open())
    ptbxl = data["ptbxl_test"]
    pn2021 = data["pn2021"]
    out = {
        "ptbxl_auroc": float(ptbxl["macro_auroc"]),
        "ptbxl_auprc": float(ptbxl["macro_auprc"]),
        "pn2021_avg_auroc": float(pn2021["avg_macro_auroc"]),
        "pn2021_avg_auprc": float(pn2021["avg_macro_auprc"]),
    }
    for center, metrics in pn2021["per_center"].items():
        out[f"{center}_auroc"] = float(metrics["macro_auroc"])
        out[f"{center}_auprc"] = float(metrics["macro_auprc"])
        out[f"{center}_n"] = int(metrics.get("effective_n", metrics.get("n_records", 0)))
    return out
