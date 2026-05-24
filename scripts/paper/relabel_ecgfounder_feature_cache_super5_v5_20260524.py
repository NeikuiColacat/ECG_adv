#!/usr/bin/env python3
"""Create a current Super5-label ECGFounder feature cache without WFDB reread.

The legacy ECGFounder feature cache stores features, centers, record_ids, and
the PN2021 Super5 labels that were current when the cache was generated.  When
the PN2021 mapping changes, the ECGFounder features remain valid, but the label
matrix must be rebuilt from the original PhysioNet manifest.

This utility writes a new cache directory and leaves the legacy cache untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


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
ECGFOUNDER_ROOT = Path(os.environ.get("ECGFOUNDER_ROOT", str(DATA_ROOT / "ecgfounder")))
if str(ECGFOUNDER_ROOT) not in sys.path:
    sys.path.insert(0, str(ECGFOUNDER_ROOT))

from scripts.paper.eval_ecgfounder_super5_zero_shot_20260517 import (  # noqa: E402
    PN2021_FORBIDDEN,
    center_from_record_path,
    record_id_from_record_path,
)
from scripts.triple_labels.label_schemes import (  # noqa: E402
    get_super5_pn2021_mapping_metadata,
    snomed_list_to_super5,
)


DEFAULT_SOURCE_DIR = DATA_ROOT / "paper_foundation_baselines_20260517/ecgfounder_linear_probe_super5"
DEFAULT_OUT_DIR = (
    DATA_ROOT / "paper_foundation_baselines_20260524/ecgfounder_linear_probe_v5_from_legacy_cache"
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def feature_path(base: Path, stem: str, preprocess_policy: str) -> Path:
    candidates = [
        base / f"{stem}_ecgfounder_features_{preprocess_policy}.npz",
        base / f"{stem}_ecgfounder_features.npz",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def current_labels_from_manifest(
    manifest_path: Path,
    centers: np.ndarray,
    record_ids: np.ndarray,
) -> tuple[np.ndarray, int]:
    with manifest_path.open() as f:
        records = json.load(f)
    label_by_key: dict[tuple[str, str], np.ndarray] = {}
    for rec in records:
        center = center_from_record_path(rec["path"])
        if center.lower() in PN2021_FORBIDDEN:
            continue
        record_id = record_id_from_record_path(rec["path"])
        label_by_key[(center, record_id)] = snomed_list_to_super5(rec.get("snomed", []))

    labels = []
    missing = 0
    for center, record_id in zip(centers.astype(str), record_ids.astype(str)):
        label = label_by_key.get((str(center), str(record_id)))
        if label is None:
            missing += 1
            label = np.zeros((5,), dtype=np.float32)
        labels.append(label.astype(np.float32, copy=False))
    return np.stack(labels).astype(np.float32), missing


def copy_if_present(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", default=str(DEFAULT_SOURCE_DIR))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--preprocess_policy", default="official_ptbxl_eval")
    p.add_argument("--manifest", default=str(ECGFOUNDER_ROOT / "physionet2021_manifest.json"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ptbxl = out_dir / f"ptbxl_ecgfounder_features_{args.preprocess_policy}.npz"
    out_pn = out_dir / f"pn2021_ecgfounder_features_{args.preprocess_policy}.npz"
    if (out_ptbxl.exists() or out_pn.exists()) and not args.force:
        raise FileExistsError(f"output cache exists in {out_dir}; pass --force to replace")

    src_ptbxl = feature_path(source_dir, "ptbxl", args.preprocess_policy)
    src_pn = feature_path(source_dir, "pn2021", args.preprocess_policy)
    if not src_ptbxl.exists() or not src_pn.exists():
        raise FileNotFoundError(f"missing source cache: {src_ptbxl}, {src_pn}")
    if not Path(args.manifest).exists():
        raise FileNotFoundError(args.manifest)

    pn = load_npz(src_pn)
    current_labels, missing = current_labels_from_manifest(
        Path(args.manifest),
        pn["centers"],
        pn["record_ids"],
    )
    old_labels = pn["labels"].astype(np.float32, copy=False)
    changed = np.any(np.abs(old_labels - current_labels) > 0, axis=1)

    shutil.copy2(src_ptbxl, out_ptbxl)
    pn["labels"] = current_labels
    np.savez_compressed(out_pn, **pn)
    for name in ["best_head.pt", "train_result.json", "head_training_log.json"]:
        copy_if_present(source_dir / name, out_dir / name)

    metadata = {
        "source_dir": str(source_dir),
        "source_ptbxl_cache": str(src_ptbxl),
        "source_pn2021_cache": str(src_pn),
        "output_ptbxl_cache": str(out_ptbxl),
        "output_pn2021_cache": str(out_pn),
        "manifest": str(Path(args.manifest)),
        "pn2021_mapping": get_super5_pn2021_mapping_metadata(),
        "n_records": int(len(current_labels)),
        "n_missing_manifest_keys": int(missing),
        "n_rows_changed": int(changed.sum()),
        "old_positive_counts": old_labels.sum(axis=0).astype(int).tolist(),
        "new_positive_counts": current_labels.sum(axis=0).astype(int).tolist(),
    }
    with (out_dir / "cache_relabel_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
