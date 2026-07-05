"""Build PN2021-C corruption caches from clean PN2021 v3 eval caches.

Inputs are the clean caches written by pn2021_clean_eval.py:
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
from ecg_adv_gen.evaluation.pn2021_eval_cache import PN2021_EVAL_CACHE_VERSION  # noqa: E402
from ecg_adv_gen.data.contracts import PREPROCESS_CONTRACT_ID  # noqa: E402
from ecg_adv_gen.evaluation.pn2021_corruptions import (  # noqa: E402
    load_json_payload,
    require_clean_eval_json,
)
from ecg_adv_gen.evaluation.pn2021c_metadata import (  # noqa: E402
    build_pn2021c_metadata_payload,
)
from ecg_adv_gen.evaluation.pn2021c import (  # noqa: E402
    PN2021C_DEFAULT_CENTERS,
    PN2021C_DEFAULT_CORRUPTIONS,
    PN2021_C_CACHE_VERSION,
    PUBLIC_TO_INTERNAL_SEVERITY,
)


DEFAULT_CENTERS = PN2021C_DEFAULT_CENTERS
DEFAULT_CORRUPTIONS = PN2021C_DEFAULT_CORRUPTIONS


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


def _center_clean_ref_count(clean_eval_payload, center):
    per_center = clean_eval_payload.get("pn2021", {}).get("per_center", {})
    row = per_center.get(center, {})
    return int(row.get("n_excluded_ref", 0) or 0)


def _center_clean_ref_hash(clean_eval_payload, center):
    hashes = clean_eval_payload.get("pn2021", {}).get("eval_protocol", {}).get("target_ref_id_hashes", {})
    return hashes.get(center)


def _merge_clean_metadata(cache_metadata, clean_eval_payload):
    metadata = dict(cache_metadata)
    if clean_eval_payload:
        if "label_mapping" in clean_eval_payload:
            metadata["label_mapping"] = clean_eval_payload["label_mapping"]
        if "preprocess" in clean_eval_payload:
            metadata["preprocess"] = clean_eval_payload["preprocess"]
    return metadata


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

    clean_eval_payload = getattr(args, "_clean_eval_payload", {})
    metadata = build_pn2021c_metadata_payload(
        clean_metadata=_merge_clean_metadata(_load_metadata(data), clean_eval_payload),
        center=center,
        corruption=corruption,
        public_severity=int(public_severity),
        internal_severity=int(PUBLIC_TO_INTERNAL_SEVERITY[int(public_severity)]),
        cache_version=PN2021_C_CACHE_VERSION,
        n_excluded_ref=_center_clean_ref_count(clean_eval_payload, center),
        preprocess_contract_id=PREPROCESS_CONTRACT_ID,
        ref_record_ids_sha256=_center_clean_ref_hash(clean_eval_payload, center),
        source_clean_cache=clean_path,
        seed=int(args.seed),
        limit=args.limit,
    )

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
    p.add_argument("--clean_eval_json", default=None,
                   help="Clean ref-excluded PN2021 eval JSON used to stamp "
                        "label mapping, preprocess contract, and ref-exclusion counts.")
    p.add_argument("--diagnostic_without_clean", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Build only the first N records per center for smoke tests.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    try:
        require_clean_eval_json(
            args.clean_eval_json,
            diagnostic_without_clean=args.diagnostic_without_clean,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args._clean_eval_payload = load_json_payload(args.clean_eval_json)

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
