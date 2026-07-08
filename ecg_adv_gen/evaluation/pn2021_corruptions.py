"""CPU-safe helpers for PN2021-C corruption evaluation outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def corruption_cache_path(
    cache_dir: os.PathLike[str] | str,
    scheme: str,
    center: str,
    corruption: str,
    severity: int | str,
    cache_version: str,
) -> str:
    name = (
        f"{scheme}_{center}_{corruption}_s{int(severity)}_100hz1000_"
        f"{cache_version}.npz"
    )
    return os.path.join(os.fspath(cache_dir), name)


def clean_mmap_cache_path(
    cache_dir: os.PathLike[str] | str,
    scheme: str,
    center: str,
    eval_cache_version: str,
) -> str:
    name = f"{scheme}_{center}_100hz1000_{eval_cache_version}"
    return os.path.join(os.fspath(cache_dir), name)


def clean_npz_cache_path(
    cache_dir: os.PathLike[str] | str,
    scheme: str,
    center: str,
    eval_cache_version: str,
) -> str:
    name = f"{scheme}_{center}_100hz1000_{eval_cache_version}.npz"
    return os.path.join(os.fspath(cache_dir), name)


def load_npz_metadata(data: Any) -> dict[str, Any]:
    if "metadata_json" not in getattr(data, "files", []):
        return {}
    raw = data["metadata_json"]
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        decoded = json.loads(str(raw))
    except Exception:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def stable_corruption_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(p) for p in (base_seed,) + parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:4], "little")


def filter_record_indices(
    record_ids: Sequence[object] | np.ndarray,
    exclude_ids: set[str] | frozenset[str] | None,
    limit: int | None = None,
) -> np.ndarray:
    ids = np.asarray(record_ids).astype(str)
    if exclude_ids:
        keep = np.asarray([rid not in exclude_ids for rid in ids], dtype=bool)
        indices = np.nonzero(keep)[0].astype(np.int64)
    else:
        indices = np.arange(len(ids), dtype=np.int64)
    if limit and int(limit) < len(indices):
        indices = indices[: int(limit)]
    return indices


def ref_ids_sha256(ids: set[str] | frozenset[str] | Sequence[object]) -> str:
    values = sorted(str(item) for item in ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def load_clean_metric_lookup(clean_eval_json: os.PathLike[str] | str | None) -> dict[str, dict[str, float | None]]:
    if not clean_eval_json:
        return {}
    with Path(clean_eval_json).open() as f:
        data = json.load(f)
    per_center = data.get("pn2021", {}).get("per_center", {})
    return {
        center: {
            "macro_auroc": vals.get("macro_auroc"),
            "macro_auprc": vals.get("macro_auprc"),
        }
        for center, vals in per_center.items()
    }


def require_clean_eval_json(
    clean_eval_json: os.PathLike[str] | str | None,
    *,
    diagnostic_without_clean: bool = False,
) -> None:
    """Require a clean PN2021 eval JSON unless the caller opts into diagnostic mode."""

    if clean_eval_json or diagnostic_without_clean:
        return
    raise ValueError("--clean_eval_json is required for paper PN2021-C evaluation")


def load_json_payload(path: os.PathLike[str] | str | None) -> dict[str, Any]:
    """Load a JSON object from path, returning an empty object for absent paths."""

    if not path:
        return {}
    with Path(path).open() as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(row[key])
    ]


def aggregate_corruption_summary(results: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for center_results in results.get("per_center", {}).values():
        for corruption, severity_results in center_results.items():
            for severity, vals in severity_results.items():
                by_key.setdefault((str(corruption), str(severity)), []).append(vals)

    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for (corruption, severity), rows in by_key.items():
        auroc = _finite_values(rows, "macro_auroc")
        auprc = _finite_values(rows, "macro_auprc")
        drop_all_zero_auroc = _finite_values(rows, "drop_all_zero_macro_auroc")
        drop_all_zero_auprc = _finite_values(rows, "drop_all_zero_macro_auprc")
        auroc_drop = _finite_values(rows, "auroc_drop_vs_clean")
        auprc_drop = _finite_values(rows, "auprc_drop_vs_clean")
        out.setdefault(corruption, {})[severity] = {
            "n_centers": len(rows),
            "mean_macro_auroc": float(np.mean(auroc)) if auroc else float("nan"),
            "mean_macro_auprc": float(np.mean(auprc)) if auprc else float("nan"),
            "mean_drop_all_zero_macro_auroc": (
                float(np.mean(drop_all_zero_auroc)) if drop_all_zero_auroc else float("nan")
            ),
            "mean_drop_all_zero_macro_auprc": (
                float(np.mean(drop_all_zero_auprc)) if drop_all_zero_auprc else float("nan")
            ),
            "mean_auroc_drop_vs_clean": float(np.mean(auroc_drop)) if auroc_drop else None,
            "mean_auprc_drop_vs_clean": float(np.mean(auprc_drop)) if auprc_drop else None,
        }
    return out


__all__ = [
    "aggregate_corruption_summary",
    "clean_mmap_cache_path",
    "clean_npz_cache_path",
    "corruption_cache_path",
    "filter_record_indices",
    "load_json_payload",
    "load_clean_metric_lookup",
    "load_npz_metadata",
    "ref_ids_sha256",
    "require_clean_eval_json",
    "stable_corruption_seed",
]
