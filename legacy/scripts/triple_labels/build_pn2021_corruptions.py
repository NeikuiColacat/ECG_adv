"""Build PN2021-C corruption caches from clean PN2021 v3 eval caches.

Inputs are the clean caches written by eval_crosscenter.py:
  /root/autodl-tmp/triple_labels/pn2021_eval_cache/
    super5_<center>_100hz1000_v3_super5_normsuppress.npz

Outputs:
  /root/autodl-tmp/triple_labels/pn2021_c_cache/
    super5_<center>_<corruption>_s<severity>_100hz1000_v1.npz

The public severity is 1..5, mapped to the existing ECG op severity
[2, 4, 6, 8, 10]. Labels and record_ids are copied unchanged.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.augmix.severity import AVAILABLE_OPS, build_op  # noqa: E402
from scripts.triple_labels.eval_crosscenter import PN2021_EVAL_CACHE_VERSION  # noqa: E402


DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DEFAULT_CORRUPTIONS = [
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
]
PUBLIC_TO_INTERNAL_SEVERITY = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
PN2021_C_CACHE_VERSION = "v1"


def _stable_seed(base_seed, *parts):
    payload = "|".join(str(p) for p in (base_seed,) + parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:4], "little")


def _clean_cache_path(clean_cache_dir, scheme, center):
    name = f"{scheme}_{center}_100hz1000_{PN2021_EVAL_CACHE_VERSION}.npz"
    return os.path.join(clean_cache_dir, name)


def _out_cache_path(output_dir, scheme, center, corruption, severity):
    name = f"{scheme}_{center}_{corruption}_s{severity}_100hz1000_{PN2021_C_CACHE_VERSION}.npz"
    return os.path.join(output_dir, name)


def _load_metadata(data):
    if "metadata_json" not in data.files:
        return {}
    raw = data["metadata_json"]
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _apply_corruption(signals_tc, corruption, public_severity, seed):
    internal_severity = PUBLIC_TO_INTERNAL_SEVERITY[int(public_severity)]
    op = build_op(corruption, internal_severity)
    out = np.empty_like(signals_tc, dtype=np.float32)
    for i, sig_tc in enumerate(signals_tc):
        sample_seed = _stable_seed(seed, corruption, public_severity, i)
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
        corrupt_ct = op(sig_ct).detach().cpu().numpy().astype(np.float32, copy=False)
        out[i] = corrupt_ct.T
    return out


def build_one(args, center, corruption, public_severity):
    clean_path = _clean_cache_path(args.clean_cache_dir, args.scheme, center)
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"clean cache not found: {clean_path}")

    out_path = _out_cache_path(args.output_dir, args.scheme, center, corruption, public_severity)
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[skip] exists: {out_path}")
        return out_path

    data = np.load(clean_path, allow_pickle=True)
    signals = data["signals"].astype(np.float32, copy=False)
    labels = data["labels"].astype(np.float32, copy=False)
    record_ids = data["record_ids"].astype(str)
    if args.limit and args.limit < len(signals):
        signals = signals[:args.limit]
        labels = labels[:args.limit]
        record_ids = record_ids[:args.limit]

    if signals.ndim != 3 or signals.shape[1:] != (1000, 12):
        raise ValueError(f"{clean_path}: expected signals (N,1000,12), got {signals.shape}")
    if len(signals) != len(labels) or len(signals) != len(record_ids):
        raise ValueError(f"{clean_path}: signals/labels/record_ids length mismatch")

    print(
        f"[build] {center:<18} {corruption:<22} s{public_severity} "
        f"n={len(signals)} -> {out_path}"
    )
    corrupt = _apply_corruption(signals, corruption, public_severity, args.seed)
    if not np.isfinite(corrupt).all():
        raise ValueError(f"{out_path}: corruption produced NaN/Inf")

    metadata = _load_metadata(data)
    metadata["pn2021_c"] = {
        "cache_version": PN2021_C_CACHE_VERSION,
        "source_clean_cache": clean_path,
        "corruption": corruption,
        "public_severity": int(public_severity),
        "internal_severity": int(PUBLIC_TO_INTERNAL_SEVERITY[int(public_severity)]),
        "seed": int(args.seed),
        "limit": int(args.limit) if args.limit else None,
        "ops_source": "methods/augmix/ecg_ops.py",
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        signals=corrupt.astype(np.float32, copy=False),
        labels=labels.astype(np.float32, copy=False),
        record_ids=record_ids,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="super5")
    p.add_argument("--clean_cache_dir", default="/root/autodl-tmp/triple_labels/pn2021_eval_cache")
    p.add_argument("--output_dir", default="/root/autodl-tmp/triple_labels/pn2021_c_cache")
    p.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    p.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS,
                   choices=AVAILABLE_OPS)
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--limit", type=int, default=None,
                   help="Build only the first N records per center for smoke tests.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    for severity in args.severities:
        if severity not in PUBLIC_TO_INTERNAL_SEVERITY:
            raise ValueError(f"public severity must be in 1..5, got {severity}")

    built = []
    for center in args.centers:
        for corruption in args.corruptions:
            for severity in args.severities:
                built.append(build_one(args, center, corruption, severity))

    print(f"[done] built/verified {len(built)} PN2021-C caches in {args.output_dir}")


if __name__ == "__main__":
    main()
