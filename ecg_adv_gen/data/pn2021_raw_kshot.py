"""Raw-first PN2021 K-shot waveform materialization helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ecg_adv_gen.labels.super5_mapping import CLASS_NAMES_SUPER5, snomed_list_to_super5
from ecg_adv_gen.preprocessing import unified_preprocess_to_1000

from .kshot_artifacts import read_ref_meta_record_ids
from .pn2021_index import scan_pn2021_center_records


@dataclass(frozen=True)
class Raw1000KShot:
    """Selected PN2021 K-shot records before model-input z-scoring."""

    signals: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    metadata: dict[str, Any]


def _default_read_record(record_path: str | Path) -> Any:
    import wfdb

    return wfdb.rdrecord(str(record_path))


def _default_label_fn(snomeds: Sequence[int]) -> np.ndarray:
    return snomed_list_to_super5(tuple(int(code) for code in snomeds)).astype(np.float32, copy=False)


def _center_dir(pn2021_root: str | Path, center: str) -> Path:
    root = Path(pn2021_root)
    training_center = root / "training" / center
    if training_center.exists():
        return training_center
    return root / center


def materialize_selected_raw1000_from_ref_meta(
    *,
    center: str,
    ref_meta_json: str | Path,
    pn2021_root: str | Path,
    scan_center_records: Callable[[str | Path], Sequence[Any]] = scan_pn2021_center_records,
    read_record: Callable[[str | Path], Any] = _default_read_record,
    preprocess_signal: Callable[..., np.ndarray | None] = unified_preprocess_to_1000,
    label_fn: Callable[[Sequence[int]], np.ndarray] = _default_label_fn,
    class_names: Sequence[str] = CLASS_NAMES_SUPER5,
) -> Raw1000KShot:
    """Materialize K-shot PN2021 records as raw1000 waveforms in ref-meta order.

    The returned ``signals`` are lead-reordered, 100 Hz, length-1000 ECGs with
    no z-score. Model-input normalization is intentionally deferred until the
    training dataset consumes this artifact.
    """

    ref_ids = read_ref_meta_record_ids(ref_meta_json, expected_center=center).record_ids
    record_by_id = {str(record.record_id): record for record in scan_center_records(_center_dir(pn2021_root, center))}
    missing = [rid for rid in ref_ids if rid not in record_by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{ref_meta_json} references {len(missing)} records absent from PN2021 center {center}: {preview}")

    signals: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    failed: list[str] = []
    t0 = time.time()
    for rid in ref_ids:
        record = record_by_id[str(rid)]
        try:
            wfdb_record = read_record(record.record_path)
            raw = getattr(wfdb_record, "p_signal", None)
            if raw is None:
                failed.append(str(rid))
                continue
            source_leads = [lead.strip() for lead in getattr(wfdb_record, "sig_name", [])] or None
            processed = preprocess_signal(
                np.asarray(raw, dtype=np.float32),
                fs=getattr(wfdb_record, "fs"),
                source_leads=source_leads,
                target_fs=100,
                target_len=1000,
                preprocess_mode="minimal_resample",
                norm_mode="none",
            )
        except Exception:
            failed.append(str(rid))
            continue
        if processed is None:
            failed.append(str(rid))
            continue
        processed = np.asarray(processed, dtype=np.float32)
        if processed.shape != (1000, 12):
            raise ValueError(f"{rid} preprocessed to {processed.shape}, expected (1000, 12)")
        signals.append(processed)
        labels.append(np.asarray(label_fn(record.snomeds), dtype=np.float32))

    if failed:
        preview = ", ".join(failed[:5])
        raise RuntimeError(f"failed to materialize {len(failed)} selected records for {center}: {preview}")
    if not signals:
        raise RuntimeError(f"no selected records materialized for {center}")

    metadata = {
        "center": str(center),
        "source_ref_meta_json": str(ref_meta_json),
        "pn2021_root": str(pn2021_root),
        "n_records": int(len(signals)),
        "class_names": list(class_names),
        "record_ids_source": "ref_meta_order",
        "signals_are_pre_zscore": True,
        "load_time_s": float(time.time() - t0),
        "preprocess": {
            "target_fs": 100,
            "target_len": 1000,
            "preprocess_mode": "minimal_resample",
            "norm_mode": "none",
        },
        "model_input_preprocess": {
            "norm_mode": "per_sample_global",
            "crop_len": 1000,
        },
        "order": [
            "wfdb_read",
            "lead_reorder_nan_guard_resample_pad_no_zscore",
            "optional_corruption_or_augmix",
            "per_sample_global_zscore",
            "model",
        ],
    }
    return Raw1000KShot(
        signals=np.stack(signals, axis=0).astype(np.float32, copy=False),
        labels=np.stack(labels, axis=0).astype(np.float32, copy=False),
        record_ids=np.asarray(ref_ids, dtype=str),
        metadata=metadata,
    )


def save_raw1000_kshot_npz(path: str | Path, materialized: Raw1000KShot) -> Path:
    """Write a raw1000 K-shot artifact with lightweight metadata."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        signals=materialized.signals.astype(np.float32, copy=False),
        labels=materialized.labels.astype(np.float32, copy=False),
        record_ids=materialized.record_ids.astype(str),
        metadata=np.asarray([materialized.metadata], dtype=object),
    )
    return out
