"""CPU-safe PN2021 waveform materialization helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PN2021MaterializedCenter:
    signals: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    fail: int
    load_time_s: float


ReadRecordFn = Callable[[str], Any]
PreprocessSignalFn = Callable[..., np.ndarray | None]
LabelFn = Callable[[Sequence[int]], np.ndarray]


def _record_source_leads(record: Any) -> list[str] | None:
    sig_name = getattr(record, "sig_name", None)
    if sig_name is None:
        return None
    return [str(name).strip() for name in sig_name]


def materialize_pn2021_center_records(
    record_paths: Sequence[str],
    snomed_lists: Sequence[Sequence[int]],
    *,
    read_record: ReadRecordFn,
    preprocess_signal: PreprocessSignalFn,
    label_fn: LabelFn,
    num_classes: int,
    preprocess_kwargs: dict[str, Any] | None = None,
    signal_shape: tuple[int, int] = (1000, 12),
) -> PN2021MaterializedCenter:
    """Read and preprocess PN2021 records into cached eval arrays."""

    signals: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    record_ids: list[str] = []
    fail = 0
    t0 = time.time()
    kwargs = dict(preprocess_kwargs or {})

    for path, codes in zip(record_paths, snomed_lists):
        try:
            record = read_record(str(path))
        except Exception:
            fail += 1
            continue
        raw_signal = getattr(record, "p_signal", None)
        if raw_signal is None or raw_signal.shape[1] < 12:
            fail += 1
            continue
        processed = preprocess_signal(
            np.asarray(raw_signal).astype(np.float32),
            fs=getattr(record, "fs", None),
            source_leads=_record_source_leads(record),
            **kwargs,
        )
        if processed is None:
            fail += 1
            continue
        signals.append(np.asarray(processed, dtype=np.float32))
        labels.append(np.asarray(label_fn(codes), dtype=np.float32))
        record_ids.append(os.path.basename(str(path)))

    if signals:
        signal_arr = np.stack(signals).astype(np.float32, copy=False)
        label_arr = np.stack(labels).astype(np.float32, copy=False)
        id_arr = np.asarray(record_ids, dtype=str)
    else:
        signal_arr = np.zeros((0, *signal_shape), dtype=np.float32)
        label_arr = np.zeros((0, int(num_classes)), dtype=np.float32)
        id_arr = np.asarray([], dtype=str)

    return PN2021MaterializedCenter(
        signals=signal_arr,
        labels=label_arr,
        record_ids=id_arr,
        fail=fail,
        load_time_s=time.time() - t0,
    )
