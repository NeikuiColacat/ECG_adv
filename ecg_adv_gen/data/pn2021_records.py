"""PN2021 Super5 record-level metadata helpers.

This module owns pure header metadata and K-shot selection policy shared by
prompt-token and center-dataset scripts. It intentionally does not read WFDB
signals, build caches, or touch torch/GPU state.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ecg_adv_gen.labels.super5_mapping import (
    CLASS_NAMES_SUPER5,
    SNOMED_TO_SUPER5,
    SNOMED_TO_SUPER5_POSITIVE,
    SUPER5_TO_IDX,
    snomed_list_to_super5,
)

from .pn2021_index import parse_header_snomeds, record_id_from_path

SUPER5_PRIMARY_PRIORITY = ("MI", "HYP", "CD", "STTC", "NORM")


def parse_pn2021_header_metadata(header_path: str | Path) -> dict[str, Any]:
    """Parse demographic metadata from PN2021 ``.hea`` comments."""
    meta: dict[str, Any] = {"age": None, "sex": None, "hr": None}
    try:
        with Path(header_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("#"):
                    continue
                body = line[1:].strip()
                if body.startswith("Age:"):
                    value = body.split(":", 1)[1].strip()
                    try:
                        age = float(value)
                    except ValueError:
                        continue
                    if math.isfinite(age):
                        meta["age"] = age
                elif body.startswith("Sex:"):
                    value = body.split(":", 1)[1].strip().upper()
                    if value.startswith("M"):
                        meta["sex"] = "M"
                    elif value.startswith("F"):
                        meta["sex"] = "F"
                    else:
                        meta["sex"] = "U"
    except OSError:
        pass
    return meta


def pn2021_hash_fold(record_id: str) -> int:
    """Return the stable legacy 1..10 hash fold for a PN2021 record id."""
    hashed = int(hashlib.sha1(record_id.encode()).hexdigest()[:8], 16)
    return (hashed % 10) + 1


def pn2021_primary_class(
    label: Sequence[float] | np.ndarray,
    *,
    priority: Sequence[str] = SUPER5_PRIMARY_PRIORITY,
) -> str | None:
    """Pick the primary Super5 class with the legacy severity priority."""
    arr = np.asarray(label)
    for cls in priority:
        if float(arr[SUPER5_TO_IDX[cls]]) == 1.0:
            return cls
    return None


def pn2021_primary_snomed(
    snomed_codes: Iterable[int],
    primary: str,
    *,
    include_norm_candidate: bool = True,
) -> int | None:
    """Pick the first SNOMED in a record that maps to ``primary``."""
    mapping = SNOMED_TO_SUPER5 if include_norm_candidate else SNOMED_TO_SUPER5_POSITIVE
    for code in snomed_codes:
        int_code = int(code)
        if mapping.get(int_code) == primary:
            return int_code
    return None


def record_to_prompt_token_cache_item(header_path: str | Path) -> dict[str, Any] | None:
    """Convert one PN2021 header into the shared legacy record-dict shape."""
    header = Path(header_path)
    codes = parse_header_snomeds(header)
    if not codes:
        return None
    label = snomed_list_to_super5(tuple(codes)).astype(np.float32, copy=False)
    primary = pn2021_primary_class(label)
    if primary is None:
        return None

    record_path = header.with_suffix("")
    record_id = record_id_from_path(record_path)
    meta = parse_pn2021_header_metadata(header)
    primary_snomed = pn2021_primary_snomed(codes, primary, include_norm_candidate=False)
    primary_code = pn2021_primary_snomed(codes, primary, include_norm_candidate=True)
    return {
        "hea_path": str(header),
        "record_path": str(record_path),
        "record_id": record_id,
        "path": str(record_path),
        "snomed_codes": codes,
        "label": label,
        "multi_hot": label,
        "primary": primary,
        "primary_class": primary,
        "primary_class_idx": SUPER5_TO_IDX[primary],
        "primary_snomed": primary_snomed,
        "primary_code": primary_code,
        "age": meta["age"],
        "sex": meta["sex"] or "U",
        "hr": meta["hr"],
        "strat_fold": pn2021_hash_fold(record_id),
    }


def _record_primary(record: Mapping[str, Any], class_key: str | None) -> str:
    if class_key is not None:
        return str(record[class_key])
    if "primary_class" in record:
        return str(record["primary_class"])
    return str(record["primary"])


def hybrid_select_pn2021_records(
    records: Sequence[Mapping[str, Any]],
    *,
    k: int,
    floor_per_class: int,
    seed: int,
    class_key: str | None = None,
) -> list[Mapping[str, Any]]:
    """Select records with per-class floor plus natural-distribution fill."""
    rng = random.Random(seed)
    by_class: dict[str, list[Mapping[str, Any]]] = {cls: [] for cls in CLASS_NAMES_SUPER5}
    for record in records:
        by_class[_record_primary(record, class_key)].append(record)

    selected: list[Mapping[str, Any]] = []
    for cls in CLASS_NAMES_SUPER5:
        pool = by_class[cls]
        n_take = min(floor_per_class, len(pool))
        if n_take:
            selected.extend(rng.sample(pool, n_take))

    selected_ids = {str(record["record_id"]) for record in selected}
    remaining = [record for record in records if str(record["record_id"]) not in selected_ids]
    n_more = k - len(selected)
    if n_more > 0 and remaining:
        rng.shuffle(remaining)
        selected.extend(remaining[:n_more])
    return selected[:k]
