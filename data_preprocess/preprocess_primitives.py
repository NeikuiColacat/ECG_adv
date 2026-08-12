"""Shared waveform primitives for the retained ECG cache builders."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DATA_ROOT_TOKEN = "${data_root}"


class WaveformQualityError(ValueError):
    """A record-level waveform rejection with machine-readable QC details."""

    def __init__(self, reason: str, details: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


def _load_yaml_mapping(
    dataset_name: str,
    config_path: str | Path,
) -> dict[str, Any]:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{dataset_name} config must be a YAML mapping: {path}")
    return payload


def _repair_nonfinite_per_lead(
    signal_tc: np.ndarray,
    *,
    max_record_fraction: float,
    max_lead_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair sparse NaN/Inf values using within-lead linear interpolation.

    ``numpy.interp`` uses the nearest valid value outside the valid index range,
    which gives the intended edge-fill behavior without inventing cross-lead
    information. Records with an all-invalid lead or excessive corruption are
    rejected before interpolation.
    """

    repaired = np.asarray(signal_tc, dtype=np.float64).copy()
    if repaired.ndim != 2 or repaired.shape[0] == 0 or repaired.shape[1] == 0:
        raise WaveformQualityError(
            "invalid_waveform_shape",
            {"shape": list(repaired.shape)},
        )

    invalid = ~np.isfinite(repaired)
    invalid_count = int(invalid.sum())
    total_values = int(repaired.size)
    record_fraction = invalid_count / total_values
    per_lead_counts = invalid.sum(axis=0).astype(int)
    per_lead_fractions = per_lead_counts / repaired.shape[0]
    details: dict[str, Any] = {
        "nonfinite_count": invalid_count,
        "nonfinite_fraction": record_fraction,
        "nonfinite_count_by_lead": per_lead_counts.tolist(),
        "nonfinite_fraction_by_lead": per_lead_fractions.tolist(),
        "repaired_nonfinite_count": 0,
        "repair_method": "none",
    }
    if invalid_count == 0:
        return repaired, details
    if record_fraction > max_record_fraction:
        raise WaveformQualityError("record_nonfinite_fraction_exceeded", details)

    sample_indices = np.arange(repaired.shape[0])
    for lead_index in range(repaired.shape[1]):
        invalid_lead = invalid[:, lead_index]
        if not invalid_lead.any():
            continue
        valid_lead = ~invalid_lead
        if not valid_lead.any():
            raise WaveformQualityError("all_nonfinite_lead", details)
        if per_lead_fractions[lead_index] > max_lead_fraction:
            raise WaveformQualityError("lead_nonfinite_fraction_exceeded", details)
        repaired[invalid_lead, lead_index] = np.interp(
            sample_indices[invalid_lead],
            sample_indices[valid_lead],
            repaired[valid_lead, lead_index],
        )

    if not np.isfinite(repaired).all():
        raise WaveformQualityError("nonfinite_remains_after_repair", details)
    details["repaired_nonfinite_count"] = invalid_count
    details["repair_method"] = "linear_per_lead_nearest_edge"
    return repaired, details


def _validate_nonfinite_policy(
    dataset_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    policy = config.get("nonfinite_policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{dataset_name} config must define nonfinite_policy")
    if str(policy.get("repair_method")) != "linear_per_lead_nearest_edge":
        raise ValueError(
            f"{dataset_name} nonfinite repair_method must be "
            "'linear_per_lead_nearest_edge'"
        )
    max_record_fraction = float(policy["max_record_fraction"])
    max_lead_fraction = float(policy["max_lead_fraction"])
    for name, value in (
        ("max_record_fraction", max_record_fraction),
        ("max_lead_fraction", max_lead_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{dataset_name} {name} must be in [0, 1], got {value}"
            )
    return {
        "repair_method": "linear_per_lead_nearest_edge",
        "max_record_fraction": max_record_fraction,
        "max_lead_fraction": max_lead_fraction,
        "reject_all_nonfinite_lead": True,
    }


def _resolve_path(value: str, data_root: str | Path | None) -> Path:
    text = str(value)
    if DATA_ROOT_TOKEN in text:
        if data_root is None:
            raise ValueError(
                f"Path {text!r} requires data_root in the YAML config"
            )
        text = text.replace(DATA_ROOT_TOKEN, str(Path(data_root).expanduser()))
    return Path(text).expanduser().resolve()


def _normalise_lead_name(name: str) -> str:
    compact = str(name).strip().replace(" ", "").upper()
    aliases = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
    return aliases.get(compact, compact)


def _reorder_leads(
    signal_tc: np.ndarray,
    source_leads: list[str],
    target_leads: list[str],
) -> np.ndarray:
    normalised_source = [_normalise_lead_name(name) for name in source_leads]
    normalised_target = [_normalise_lead_name(name) for name in target_leads]
    if len(set(normalised_source)) != len(normalised_source):
        raise ValueError(f"Duplicate source leads: {source_leads}")
    try:
        indices = [normalised_source.index(name) for name in normalised_target]
    except ValueError as exc:
        raise ValueError(
            f"Cannot map source leads {source_leads} to target leads {target_leads}"
        ) from exc
    return np.asarray(signal_tc)[:, indices]


def _read_wfdb(record_path: Path) -> tuple[np.ndarray, float, list[str]]:
    import wfdb

    signal_tc, fields = wfdb.rdsamp(str(record_path))
    return (
        np.asarray(signal_tc),
        float(fields["fs"]),
        [str(name) for name in fields["sig_name"]],
    )


def _record_hash(dataset: str, dataset_version: str, record_key: str) -> str:
    identity = f"{dataset}\0{dataset_version}\0{record_key}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _fixed_unicode_array(values: list[str]) -> np.ndarray:
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _linear_interpolate_time_batch(
    signals_ntc: np.ndarray,
    target_num_samples: int,
) -> np.ndarray:
    """Resize a batch with the current ECGFounder aligned-corner interpolation."""

    signals_ntc = np.asarray(signals_ntc)
    if signals_ntc.ndim != 3 or signals_ntc.shape[1] < 2:
        raise ValueError(
            "Expected batched time-channel signals with at least two time points, "
            f"got {signals_ntc.shape}"
        )
    if target_num_samples < 2:
        raise ValueError(
            f"target_num_samples must be at least 2, got {target_num_samples}"
        )
    import torch
    import torch.nn.functional as F

    source_array = np.array(signals_ntc, copy=True, order="C")
    source = torch.from_numpy(source_array).permute(0, 2, 1)
    with torch.no_grad():
        derived = F.interpolate(
            source,
            size=int(target_num_samples),
            mode="linear",
            align_corners=True,
        )
    return derived.permute(0, 2, 1).contiguous().numpy()
