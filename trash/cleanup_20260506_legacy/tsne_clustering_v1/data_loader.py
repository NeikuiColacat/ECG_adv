"""Load PhysioNet 2021 multi-center real samples with per-record metadata.

Each center yields up to N records preprocessed via unified_preprocess_to_1000
(bandpass + resample to 100Hz + z-score) so the embedder sees identically
conditioned signals across centers. Per-record age/sex/HR/Dx are extracted
from the .hea header and kept alongside the signal tensor — they are needed
later as the ECGTwin ref_label.
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import wfdb

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000


@dataclass
class RealSample:
    signal: np.ndarray          # (12, 1000) float32, 100Hz, canonical lead order
    center: str
    record_id: str
    age: int                    # default 50 if missing
    sex: str                    # "Male" / "Female" / "Unknown"
    snomed: list[int]
    hr: Optional[float]         # estimated from Lead II; None if not estimable


def _parse_header_meta(header_path: str) -> dict:
    """Extract Age, Sex, Dx (SNOMED list) from a PhysioNet 2021 .hea header."""
    meta = {"age": None, "sex": None, "snomed": []}
    with open(header_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                continue
            low = line.lower()
            if "age" in low and ":" in line:
                try:
                    meta["age"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif "sex" in low and ":" in line:
                meta["sex"] = line.split(":", 1)[1].strip()
            elif "dx" in low and ":" in line:
                codes = line.split(":", 1)[1].strip()
                try:
                    meta["snomed"] = [int(c.strip()) for c in codes.split(",") if c.strip()]
                except ValueError:
                    pass
    return meta


def _scan_records(center_dir: str) -> list[str]:
    out = []
    for root, _, files in os.walk(center_dir):
        for f in files:
            if f.endswith(".hea"):
                out.append(os.path.join(root, f[:-4]))
    return sorted(out)


def _estimate_hr_bpm(signal_ct_100hz: np.ndarray) -> Optional[float]:
    """find_peaks on Lead II z-scored at 100 Hz."""
    from scipy.signal import find_peaks

    lead_ii = signal_ct_100hz[1]
    std = float(lead_ii.std())
    if std < 1e-6:
        return None
    z = (lead_ii - lead_ii.mean()) / std
    peaks, _ = find_peaks(z, height=0.5, distance=max(1, int(0.25 * 100)))
    if len(peaks) < 3:
        return None
    duration_sec = len(lead_ii) / 100.0
    return 60.0 * (len(peaks) - 1) / duration_sec


def load_center(
    center: str,
    pn2021_root: str,
    n_per_center: int,
    seed: int = 0,
) -> list[RealSample]:
    """Return up to n_per_center RealSample from a single center.

    Records that fail preprocessing are silently skipped; we keep sampling
    until n_per_center valid records are collected or the pool is exhausted.
    """
    center_dir = os.path.join(pn2021_root, center)
    candidates = _scan_records(center_dir)
    if not candidates:
        raise FileNotFoundError(f"No .hea records under {center_dir}")
    rng = random.Random(seed)
    rng.shuffle(candidates)

    collected: list[RealSample] = []
    for rec_path in candidates:
        if len(collected) >= n_per_center:
            break
        try:
            rec = wfdb.rdrecord(rec_path)
        except Exception:
            continue
        sig = rec.p_signal
        if sig is None or sig.shape[1] < 12:
            continue
        fs = rec.fs
        sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, "sig_name", None) else None

        proc_tc = unified_preprocess_to_1000(
            sig.astype(np.float32),
            fs=fs,
            source_leads=sig_names,
            target_fs=100,
            target_len=1000,
            apply_filter=True,
            apply_zscore=True,
        )
        if proc_tc is None:
            continue

        signal_ct = proc_tc.T.astype(np.float32)  # (12, 1000) canonical
        meta = _parse_header_meta(rec_path + ".hea")
        age = meta["age"] if isinstance(meta["age"], int) and 0 < meta["age"] < 120 else 50
        sex_raw = meta["sex"] or "Unknown"
        sex = (
            "Male" if sex_raw.lower().startswith("m")
            else ("Female" if sex_raw.lower().startswith("f") else "Unknown")
        )
        hr = _estimate_hr_bpm(signal_ct)

        collected.append(
            RealSample(
                signal=signal_ct,
                center=center,
                record_id=os.path.basename(rec_path),
                age=age,
                sex=sex,
                snomed=meta["snomed"],
                hr=hr,
            )
        )
    return collected


def load_all_centers(
    centers: list[str],
    pn2021_root: str,
    n_per_center: int,
    seed: int = 0,
) -> list[RealSample]:
    """Concatenate samples from all requested centers in order."""
    out: list[RealSample] = []
    for i, c in enumerate(centers):
        out.extend(load_center(c, pn2021_root, n_per_center, seed=seed + i))
    return out


def samples_to_arrays(samples: list[RealSample], centers: list[str]) -> dict:
    """Pack a list of samples into stacked arrays for downstream use.

    Returns:
        {
          "signals":    (N, 12, 1000) float32,
          "center_ids": (N,) int64  — indices into `centers`,
          "center_names": centers (list[str]),
          "ages":  (N,) int32,
          "sexes": (N,) list[str],
          "hrs":   (N,) float32  (NaN if unknown),
          "snomed": list[list[int]] — variable length,
          "record_ids": list[str],
        }
    """
    center_to_id = {c: i for i, c in enumerate(centers)}
    signals = np.stack([s.signal for s in samples], axis=0)
    center_ids = np.array([center_to_id[s.center] for s in samples], dtype=np.int64)
    ages = np.array([s.age for s in samples], dtype=np.int32)
    sexes = [s.sex for s in samples]
    hrs = np.array([s.hr if s.hr is not None else np.nan for s in samples], dtype=np.float32)
    snomed = [s.snomed for s in samples]
    record_ids = [s.record_id for s in samples]
    return {
        "signals": signals,
        "center_ids": center_ids,
        "center_names": list(centers),
        "ages": ages,
        "sexes": sexes,
        "hrs": hrs,
        "snomed": snomed,
        "record_ids": record_ids,
    }
