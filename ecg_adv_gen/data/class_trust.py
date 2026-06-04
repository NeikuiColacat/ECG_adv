"""Class-trust metadata helpers for real-anchor K-shot artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ecg_adv_gen.labels.super5 import CLASS_NAMES_SUPER5

DEFAULT_REAL_ALL_PRESENT_POLICY = (
    "Trust every Super5 class present in the real target-center K subset."
)


def write_real_all_present_trust(
    signal_npz: Path | str,
    out_dir: Path | str,
    center: str,
    *,
    classes_in_scope: Iterable[str] | None = None,
    policy: str = DEFAULT_REAL_ALL_PRESENT_POLICY,
) -> Path:
    """Write class-trust JSON by trusting classes present in a signal NPZ."""
    signal_npz = Path(signal_npz)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = signal_npz.name.removesuffix(".signals.npz")
    out_path = out_dir / f"{tag}.real_all_present.class_trust.json"

    with np.load(signal_npz) as data:
        labels = data["labels"].astype(np.float32)
    if labels.ndim != 2 or labels.shape[1] != len(CLASS_NAMES_SUPER5):
        raise ValueError(
            f"labels must have shape (N, {len(CLASS_NAMES_SUPER5)}), got {labels.shape}"
        )

    counts = labels.sum(axis=0).astype(int).tolist()
    allowed = set(classes_in_scope or CLASS_NAMES_SUPER5)
    unknown = allowed.difference(CLASS_NAMES_SUPER5)
    if unknown:
        raise ValueError(f"Unknown Super5 classes in classes_in_scope: {sorted(unknown)}")

    blob = {
        "tag": f"{tag}_real_all_present",
        "center": center,
        "signal_npz": str(signal_npz),
        "class_trust": {
            cls: (1.0 if counts[i] > 0 and cls in allowed else 0.0)
            for i, cls in enumerate(CLASS_NAMES_SUPER5)
        },
        "class_counts": dict(zip(CLASS_NAMES_SUPER5, counts)),
        "policy": policy,
    }
    out_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return out_path
