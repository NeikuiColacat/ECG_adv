"""Shared ECGFounder linear-probe helpers for managed paper runners."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


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
for _path in (str(REPO_ROOT), str(ECGFOUNDER_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from physionet2021_dataset import (  # noqa: E402
    EXPECTED_LEADS,
    TARGET_POINTS,
    filter_bandpass,
    resample_to_target,
    scan_records,
    z_score_normalize,
)

from ecg_adv_gen.data import (  # noqa: E402
    PN2021_EVAL_CENTERS_7,
    PN2021_LEAK_EXCLUDED_CENTERS,
    PN2021_TARGET_CENTERS_4,
    center_from_record_path,
    record_id_from_path,
)
from ecg_adv_gen.evaluation import assemble_target_refexcluded_views, compute_macro_metric_dict  # noqa: E402
from ecg_adv_gen.labels import CLASS_NAMES_SUPER5, get_super5_scheme  # noqa: E402
from ecg_adv_gen.labels.super5_mapping import snomed_list_to_super5  # noqa: E402


DEFAULT_OUT_DIR = DATA_ROOT / "paper_foundation_baselines_20260517/ecgfounder_linear_probe_super5"
PTBXL_ROOT = DATA_ROOT / "ptbxl"
PTBXL_CSV = PTBXL_ROOT / "ptbxl_database.csv"
PN2021_ROOT = DATA_ROOT / "physionet2021/training"
CHECKPOINT = ECGFOUNDER_ROOT / "checkpoint/12_lead_ECGFounder.pth"
REF_ROOT = DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516/subsets"
PN2021_CENTERS = list(PN2021_EVAL_CENTERS_7)
PN2021_FORBIDDEN = set(PN2021_LEAK_EXCLUDED_CENTERS)
TARGET_CENTERS = list(PN2021_TARGET_CENTERS_4)


def normalize_lead_name(name: str) -> str:
    mapping = {
        "AVR": "aVR",
        "AVL": "aVL",
        "AVF": "aVF",
        "avr": "aVR",
        "avl": "aVL",
        "avf": "aVF",
    }
    return mapping.get(str(name), str(name))


def preprocess_record(
    record_path: str,
    max_duration_sec: int = 10,
    preprocess_policy: str = "official_ptbxl_eval",
) -> np.ndarray:
    data, fields = wfdb.rdsamp(record_path)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).T
    fs = int(fields.get("fs", 500))
    sig_names = [normalize_lead_name(x) for x in fields.get("sig_name", [])]
    if sig_names and sig_names != EXPECTED_LEADS:
        try:
            data = data[[sig_names.index(lead) for lead in EXPECTED_LEADS]]
        except ValueError:
            pass
    max_samples = int(fs * max_duration_sec)
    if data.shape[1] > max_samples:
        data = data[:, :max_samples]
    if preprocess_policy == "filtered_dataset" and data.shape[1] > 10 and fs > 0:
        try:
            data = filter_bandpass(data, fs)
        except Exception:
            pass
    if data.shape[1] < max_samples:
        data = np.pad(data, ((0, 0), (0, max_samples - data.shape[1])), mode="constant")
    data = resample_to_target(data, data.shape[1], TARGET_POINTS)
    data = z_score_normalize(data)
    return data.astype(np.float32, copy=False)


def build_ptbxl_items(limit: int = 0) -> tuple[list[dict], np.ndarray]:
    df = pd.read_csv(PTBXL_CSV)
    scheme = get_super5_scheme()
    labels = np.stack([scheme["ptbxl_fn"](x) for x in df.scp_codes]).astype(np.float32)
    items = [
        {
            "path": str(PTBXL_ROOT / row.filename_hr),
            "label": labels[idx],
            "center": "ptbxl",
            "record_id": str(row.ecg_id),
            "strat_fold": int(row.strat_fold),
        }
        for idx, row in df.iterrows()
    ]
    if limit > 0:
        items = items[:limit]
        labels = labels[:limit]
    folds = np.asarray([item["strat_fold"] for item in items], dtype=np.int64)
    return items, folds


def build_pn2021_items(manifest_cache: Path, limit_per_center: int = 0) -> list[dict]:
    if manifest_cache.exists():
        with manifest_cache.open() as f:
            records = json.load(f)
    else:
        records = scan_records(str(PN2021_ROOT))
        manifest_cache.parent.mkdir(parents=True, exist_ok=True)
        with manifest_cache.open("w") as f:
            json.dump(records, f)
    counts = {center: 0 for center in PN2021_CENTERS}
    items = []
    for rec in records:
        center = center_from_record_path(rec["path"])
        if center.lower() in PN2021_FORBIDDEN or center not in counts:
            continue
        if limit_per_center and counts[center] >= limit_per_center:
            continue
        counts[center] += 1
        items.append(
            {
                "path": rec["path"],
                "label": snomed_list_to_super5(rec.get("snomed", [])),
                "center": center,
                "record_id": record_id_from_path(rec["path"]),
            }
        )
    return items


def compute_metrics(labels: np.ndarray, scores: np.ndarray, min_pos: int = 10) -> dict:
    return compute_macro_metric_dict(
        labels,
        scores,
        class_names=CLASS_NAMES_SUPER5,
        min_pos=min_pos,
    )


def evaluate_pn2021_views(
    labels: np.ndarray,
    scores: np.ndarray,
    centers: np.ndarray,
    record_ids: np.ndarray,
    ref_ids_by_center: dict[str, set[str]],
    report_drop_all_zero: bool = False,
) -> dict:
    eval_centers = [c for c in PN2021_CENTERS if c.lower() not in PN2021_FORBIDDEN]
    return assemble_target_refexcluded_views(
        labels,
        scores,
        centers,
        record_ids,
        metric_fn=lambda y_true, y_score: compute_metrics(y_true, y_score),
        ref_ids_by_center=ref_ids_by_center,
        target_centers=TARGET_CENTERS,
        eval_centers=eval_centers,
        report_drop_all_zero=report_drop_all_zero,
        drop_none_if_empty=True,
    )


__all__ = [
    "CHECKPOINT",
    "DEFAULT_OUT_DIR",
    "PTBXL_CSV",
    "REF_ROOT",
    "build_pn2021_items",
    "build_ptbxl_items",
    "compute_metrics",
    "evaluate_pn2021_views",
    "preprocess_record",
]
