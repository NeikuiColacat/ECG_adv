"""Evaluate EfficientNet1DV2 on PN2021-C corruption caches.

This script intentionally stays separate from eval_crosscenter.py so the clean
PN2021 benchmark remains unchanged.
"""

import argparse
import json
import os
import sys
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "model", "DeepECG", "notebooks"))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from scripts.triple_labels.build_pn2021_corruptions import (  # noqa: E402
    DEFAULT_CENTERS,
    DEFAULT_CORRUPTIONS,
    PUBLIC_TO_INTERNAL_SEVERITY,
    PN2021_C_CACHE_VERSION,
)
from scripts.triple_labels.eval_crosscenter import (  # noqa: E402
    PN2021_EVAL_CACHE_VERSION,
    PN2021CachedCenterDataset,
    compute_macro_auroc_auprc,
    infer_dataset,
)
from scripts.triple_labels.label_schemes import get_scheme  # noqa: E402
from methods.augmix.ecg_ops import (  # noqa: E402
    BaselineShift,
    BaselineWander,
    EMGNoise,
    PowerlineNoise,
    RandomLeadsMask,
)
from methods.augmix.severity import build_op  # noqa: E402
from ecg_adv_gen.data.kshot import load_ref_record_ids_by_center  # noqa: E402
from ecg_adv_gen.evaluation.pn2021_corruptions import (  # noqa: E402
    aggregate_corruption_summary,
    clean_mmap_cache_path,
    clean_npz_cache_path,
    corruption_cache_path,
    filter_record_indices,
    load_clean_metric_lookup,
    load_npz_metadata,
    stable_corruption_seed,
)


STRESS_PROFILE_CHOICES = ("standard", "stress_v2")


_STRESS_V2_PARAMS = {
    # Stress profile calibrated for z-scored 100Hz/10s ECG. It is deliberately
    # stronger than the training-time AugMix severity table and should only be
    # used as a corruption benchmark.
    "powerline_noise": {
        1: {"max_amplitude": 0.15},
        2: {"max_amplitude": 0.30},
        3: {"max_amplitude": 0.60},
        4: {"max_amplitude": 0.90},
        5: {"max_amplitude": 1.20},
    },
    "emg_noise": {
        1: {"max_amplitude": 0.10},
        2: {"max_amplitude": 0.25},
        3: {"max_amplitude": 0.50},
        4: {"max_amplitude": 0.80},
        5: {"max_amplitude": 1.20},
    },
    "baseline_wander": {
        1: {"max_amplitude": 0.20, "k": 3},
        2: {"max_amplitude": 0.40, "k": 3},
        3: {"max_amplitude": 0.80, "k": 4},
        4: {"max_amplitude": 1.20, "k": 4},
        5: {"max_amplitude": 1.60, "k": 5},
    },
    "baseline_shift": {
        1: {"max_amplitude": 0.20, "shift_ratio": 0.10, "num_segment": 1},
        2: {"max_amplitude": 0.50, "shift_ratio": 0.20, "num_segment": 1},
        3: {"max_amplitude": 0.80, "shift_ratio": 0.35, "num_segment": 2},
        4: {"max_amplitude": 1.10, "shift_ratio": 0.55, "num_segment": 2},
        5: {"max_amplitude": 1.40, "shift_ratio": 0.75, "num_segment": 3},
    },
    "random_leads_masking": {
        1: {"mask_leads_prob": 0.15},
        2: {"mask_leads_prob": 0.30},
        3: {"mask_leads_prob": 0.50},
        4: {"mask_leads_prob": 0.70},
        5: {"mask_leads_prob": 0.85},
    },
}


def _build_stress_v2_op(corruption, public_severity):
    if corruption not in _STRESS_V2_PARAMS:
        raise ValueError(f"stress_v2 does not define corruption: {corruption}")
    params = dict(_STRESS_V2_PARAMS[corruption][int(public_severity)])
    params.setdefault("p", 1.0)
    if corruption == "powerline_noise":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("freq", 100)
        params.setdefault("dependency", False)
        return PowerlineNoise(**params)
    if corruption == "emg_noise":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("dependency", False)
        return EMGNoise(**params)
    if corruption == "baseline_wander":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("min_freq", 0.03)
        params.setdefault("max_freq", 0.50)
        params.setdefault("freq", 100)
        params.setdefault("dependency", False)
        return BaselineWander(**params)
    if corruption == "baseline_shift":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("freq", 100)
        params.setdefault("dependency", False)
        return BaselineShift(**params)
    if corruption == "random_leads_masking":
        params.setdefault("mask_leads_selection", "random")
        return RandomLeadsMask(**params)
    raise ValueError(f"unknown corruption: {corruption}")


def _build_corruption_op(corruption, public_severity, severity_profile):
    if severity_profile == "standard":
        return build_op(corruption, PUBLIC_TO_INTERNAL_SEVERITY[int(public_severity)])
    if severity_profile == "stress_v2":
        return _build_stress_v2_op(corruption, public_severity)
    raise ValueError(f"unknown severity_profile: {severity_profile}")


def _cache_path(cache_dir, scheme, center, corruption, severity):
    return corruption_cache_path(
        cache_dir, scheme, center, corruption, severity, PN2021_C_CACHE_VERSION
    )


def _clean_mmap_path(cache_dir, scheme, center):
    return clean_mmap_cache_path(cache_dir, scheme, center, PN2021_EVAL_CACHE_VERSION)


def _clean_npz_path(cache_dir, scheme, center):
    return clean_npz_cache_path(cache_dir, scheme, center, PN2021_EVAL_CACHE_VERSION)


def _load_metadata(data):
    return load_npz_metadata(data)


def _stable_seed(base_seed, *parts):
    return stable_corruption_seed(base_seed, *parts)


class StreamingCorruptedPN2021Dataset(Dataset):
    def __init__(self, signals, labels, corruption, public_severity,
                 seed=20260501, crop_len=250, severity_profile="standard",
                 indices=None):
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = (
            np.arange(len(signals), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.corruption = corruption
        self.public_severity = int(public_severity)
        self.internal_severity = PUBLIC_TO_INTERNAL_SEVERITY[self.public_severity]
        self.severity_profile = severity_profile
        self.seed = int(seed)
        self.crop_len = crop_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        sample_seed = _stable_seed(self.seed, self.corruption, self.public_severity, real_idx)
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        op = _build_corruption_op(
            self.corruption, self.public_severity, self.severity_profile
        )

        sig_tc = self.signals[real_idx]
        start = max((sig_tc.shape[0] - self.crop_len) // 2, 0)
        crop = sig_tc[start:start + self.crop_len]
        sig_ct = torch.from_numpy(np.ascontiguousarray(crop.T)).float()
        corrupt_ct = op(sig_ct)
        label = np.array(self.labels[real_idx], dtype=np.float32, copy=True)
        return corrupt_ct.float(), torch.from_numpy(label).float()


def _filter_indices(record_ids, exclude_ids, limit=None):
    return filter_record_indices(record_ids, exclude_ids, limit)


def _load_clean_center(args, center):
    mmap_root = _clean_mmap_path(args.clean_mmap_cache_dir, args.scheme, center)
    if os.path.isdir(mmap_root):
        sig_path = os.path.join(mmap_root, "signals.npy")
        lab_path = os.path.join(mmap_root, "labels.npy")
        rid_path = os.path.join(mmap_root, "record_ids.npy")
        meta_path = os.path.join(mmap_root, "metadata.json")
        if os.path.exists(sig_path) and os.path.exists(lab_path) and os.path.exists(rid_path):
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    metadata = json.load(f)
            return (
                np.load(sig_path, mmap_mode="r"),
                np.load(lab_path, mmap_mode="r"),
                np.load(rid_path, allow_pickle=True).astype(str),
                metadata,
                "mmap",
            )
    npz_path = _clean_npz_path(args.clean_cache_dir, args.scheme, center)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"clean cache not found for {center}: {mmap_root} or {npz_path}"
        )
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["signals"].astype(np.float32, copy=False),
        data["labels"].astype(np.float32, copy=False),
        data["record_ids"].astype(str),
        _load_metadata(data),
        "npz",
    )


def _load_model(args, scheme, device):
    model = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=scheme["num_classes"],
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    ).to(device)
    ckpt = os.path.join(args.model_dir, "best_model.pt")
    sd = torch.load(ckpt, map_location=device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"[model] loaded {ckpt}")
    return model


def _clean_lookup(clean_eval_json):
    return load_clean_metric_lookup(clean_eval_json)


def eval_one(model, scheme, args, device, center, corruption, severity, clean_by_center):
    path = None
    metadata = {}
    cache_source = args.mode
    exclude_ids = getattr(args, "exclude_ref_ids_by_center", {}).get(center, set())
    if args.mode == "stream":
        signals, labels, record_ids, metadata, clean_kind = _load_clean_center(args, center)
        cache_source = f"stream:{clean_kind}"
        indices = _filter_indices(record_ids, exclude_ids, args.limit)
        ds = StreamingCorruptedPN2021Dataset(
            signals, labels, corruption, severity,
            seed=args.seed, crop_len=args.crop_len,
            severity_profile=args.severity_profile,
            indices=indices,
        )
    else:
        if args.severity_profile != "standard":
            raise ValueError(
                "cache mode only supports --severity_profile standard because "
                "prebuilt PN2021-C caches encode the standard profile"
            )
        path = _cache_path(args.cache_dir, args.scheme, center, corruption, severity)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing PN2021-C cache: {path}")
        data = np.load(path, allow_pickle=True)
        signals = data["signals"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.float32, copy=False)
        record_ids = data["record_ids"].astype(str)
        indices = _filter_indices(record_ids, exclude_ids, args.limit)
        signals = signals[indices]
        labels = labels[indices]
        ds = PN2021CachedCenterDataset(signals, labels, crop_len=args.crop_len)
        metadata = _load_metadata(data)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    t0 = time.time()
    y_true, y_score = infer_dataset(model, loader, device)
    metrics = compute_macro_auroc_auprc(
        y_true, y_score, scheme["class_names"], min_pos=args.min_pos
    )
    clean = clean_by_center.get(center, {})
    auroc_drop = None
    auprc_drop = None
    if clean.get("macro_auroc") is not None:
        auroc_drop = float(clean["macro_auroc"] - metrics["macro_auroc"])
    if clean.get("macro_auprc") is not None:
        auprc_drop = float(clean["macro_auprc"] - metrics["macro_auprc"])

    print(
        f"  {center:<18} {corruption:<22} s{severity} n={len(ds):>5} "
        f"AUROC={metrics['macro_auroc']:.4f} AUPRC={metrics['macro_auprc']:.4f} "
        f"drop=({auroc_drop if auroc_drop is not None else float('nan'):.4f}, "
        f"{auprc_drop if auprc_drop is not None else float('nan'):.4f}) "
        f"({time.time() - t0:.0f}s)"
    )
    return {
        "cache_path": path,
        "cache_source": cache_source,
        "metadata": metadata,
        "n_excluded_ref_ids_for_center": int(len(exclude_ids)),
        "corruption": {
            "name": corruption,
            "severity_profile": args.severity_profile,
            "public_severity": int(severity),
            "internal_severity": int(PUBLIC_TO_INTERNAL_SEVERITY[int(severity)]),
            "seed": int(args.seed),
        },
        "n_records": int(len(ds)),
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"],
        "n_classes_used": metrics["n_classes_used"],
        "per_class": metrics["per_class"],
        "clean_macro_auroc": clean.get("macro_auroc"),
        "clean_macro_auprc": clean.get("macro_auprc"),
        "auroc_drop_vs_clean": auroc_drop,
        "auprc_drop_vs_clean": auprc_drop,
    }


def _aggregate(results):
    return aggregate_corruption_summary(results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", default="super5", choices=["super5", "sub23", "pn26"])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--mode", default="stream", choices=["stream", "cache"],
                   help="stream: corrupt clean cache on the fly; cache: read prebuilt PN2021-C npz files")
    p.add_argument("--cache_dir", default="/root/autodl-tmp/triple_labels/pn2021_c_cache")
    p.add_argument("--clean_mmap_cache_dir",
                   default="/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap")
    p.add_argument("--clean_cache_dir",
                   default="/root/autodl-tmp/triple_labels/pn2021_eval_cache")
    p.add_argument("--clean_eval_json",
                   default="/root/autodl-tmp/triple_labels/super5/eval_result_v3_super5_normsuppress.json")
    p.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    p.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--severity_profile", default="standard",
                   choices=STRESS_PROFILE_CHOICES,
                   help="standard uses methods/augmix/severity.py via public "
                        "severity 1..5 -> internal 2/4/6/8/10; stress_v2 is "
                        "a stronger streaming-only robustness sweep and does "
                        "not affect training-time AugMix defaults.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--crop_len", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--min_pos", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument(
        "--exclude_ref_ids",
        nargs="*",
        default=[],
        help=(
            "Optional K-shot ref-meta JSON file(s). Records listed under "
            "ref_record_ids are excluded from the matching center before "
            "corruption evaluation."
        ),
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N records per cache for smoke tests.")
    p.add_argument("--output_path", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    scheme = get_scheme(args.scheme)
    model = _load_model(args, scheme, device)
    clean_by_center = _clean_lookup(args.clean_eval_json)
    args.exclude_ref_ids_by_center = load_ref_record_ids_by_center(args.exclude_ref_ids)
    if args.exclude_ref_ids_by_center:
        print(
            "[exclude] "
            + ", ".join(
                f"{center}:{len(ids)}"
                for center, ids in sorted(args.exclude_ref_ids_by_center.items())
            )
        )

    output = {
        "scheme": args.scheme,
        "model_dir": args.model_dir,
        "mode": args.mode,
        "cache_dir": args.cache_dir,
        "clean_mmap_cache_dir": args.clean_mmap_cache_dir,
        "clean_cache_dir": args.clean_cache_dir,
        "clean_eval_json": args.clean_eval_json,
        "centers": list(args.centers),
        "corruptions": list(args.corruptions),
        "severities": list(args.severities),
        "severity_profile": args.severity_profile,
        "exclude_ref_ids": list(args.exclude_ref_ids),
        "exclude_ref_ids_by_center_counts": {
            center: len(ids)
            for center, ids in sorted(args.exclude_ref_ids_by_center.items())
        },
        "pn2021_c_cache_version": PN2021_C_CACHE_VERSION,
        "per_center": {},
    }

    for center in args.centers:
        output["per_center"].setdefault(center, {})
        for corruption in args.corruptions:
            output["per_center"][center].setdefault(corruption, {})
            for severity in args.severities:
                output["per_center"][center][corruption][str(severity)] = eval_one(
                    model, scheme, args, device, center, corruption, severity, clean_by_center
                )

    output["aggregate_by_corruption_severity"] = _aggregate(output)

    out_path = args.output_path or os.path.join(args.model_dir, "eval_pn2021_c.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[done] saved {out_path}")


if __name__ == "__main__":
    main()
