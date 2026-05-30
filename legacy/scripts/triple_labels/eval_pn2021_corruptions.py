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
from methods.augmix.severity import build_op  # noqa: E402


def _cache_path(cache_dir, scheme, center, corruption, severity):
    name = f"{scheme}_{center}_{corruption}_s{severity}_100hz1000_{PN2021_C_CACHE_VERSION}.npz"
    return os.path.join(cache_dir, name)


def _clean_mmap_path(cache_dir, scheme, center):
    return os.path.join(
        cache_dir,
        f"{scheme}_{center}_100hz1000_{PN2021_EVAL_CACHE_VERSION}",
    )


def _clean_npz_path(cache_dir, scheme, center):
    return os.path.join(
        cache_dir,
        f"{scheme}_{center}_100hz1000_{PN2021_EVAL_CACHE_VERSION}.npz",
    )


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


def _stable_seed(base_seed, *parts):
    import hashlib
    payload = "|".join(str(p) for p in (base_seed,) + parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:4], "little")


class StreamingCorruptedPN2021Dataset(Dataset):
    def __init__(self, signals, labels, corruption, public_severity,
                 seed=20260501, crop_len=250):
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.corruption = corruption
        self.public_severity = int(public_severity)
        self.internal_severity = PUBLIC_TO_INTERNAL_SEVERITY[self.public_severity]
        self.seed = int(seed)
        self.crop_len = crop_len

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sample_seed = _stable_seed(self.seed, self.corruption, self.public_severity, idx)
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        op = build_op(self.corruption, self.internal_severity)

        sig_tc = self.signals[idx]
        start = max((sig_tc.shape[0] - self.crop_len) // 2, 0)
        crop = sig_tc[start:start + self.crop_len]
        sig_ct = torch.from_numpy(np.ascontiguousarray(crop.T)).float()
        corrupt_ct = op(sig_ct)
        label = np.array(self.labels[idx], dtype=np.float32, copy=True)
        return corrupt_ct.float(), torch.from_numpy(label).float()


def _load_clean_center(args, center):
    mmap_root = _clean_mmap_path(args.clean_mmap_cache_dir, args.scheme, center)
    if os.path.isdir(mmap_root):
        sig_path = os.path.join(mmap_root, "signals.npy")
        lab_path = os.path.join(mmap_root, "labels.npy")
        meta_path = os.path.join(mmap_root, "metadata.json")
        if os.path.exists(sig_path) and os.path.exists(lab_path):
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    metadata = json.load(f)
            return (
                np.load(sig_path, mmap_mode="r"),
                np.load(lab_path, mmap_mode="r"),
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
    if not clean_eval_json:
        return {}
    with open(clean_eval_json) as f:
        data = json.load(f)
    per_center = data.get("pn2021", {}).get("per_center", {})
    return {
        center: {
            "macro_auroc": vals.get("macro_auroc"),
            "macro_auprc": vals.get("macro_auprc"),
        }
        for center, vals in per_center.items()
    }


def eval_one(model, scheme, args, device, center, corruption, severity, clean_by_center):
    path = None
    metadata = {}
    cache_source = args.mode
    if args.mode == "stream":
        signals, labels, metadata, clean_kind = _load_clean_center(args, center)
        cache_source = f"stream:{clean_kind}"
        if args.limit and args.limit < len(signals):
            signals = signals[:args.limit]
            labels = labels[:args.limit]
        ds = StreamingCorruptedPN2021Dataset(
            signals, labels, corruption, severity,
            seed=args.seed, crop_len=args.crop_len,
        )
    else:
        path = _cache_path(args.cache_dir, args.scheme, center, corruption, severity)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing PN2021-C cache: {path}")
        data = np.load(path, allow_pickle=True)
        signals = data["signals"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.float32, copy=False)
        if args.limit and args.limit < len(signals):
            signals = signals[:args.limit]
            labels = labels[:args.limit]
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
        "corruption": {
            "name": corruption,
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
    by_key = {}
    for center, center_results in results["per_center"].items():
        for corruption, severity_results in center_results.items():
            for severity, vals in severity_results.items():
                key = (corruption, severity)
                by_key.setdefault(key, []).append(vals)

    out = {}
    for (corruption, severity), rows in by_key.items():
        auroc = [r["macro_auroc"] for r in rows if np.isfinite(r["macro_auroc"])]
        auprc = [r["macro_auprc"] for r in rows if np.isfinite(r["macro_auprc"])]
        auroc_drop = [
            r["auroc_drop_vs_clean"] for r in rows
            if r["auroc_drop_vs_clean"] is not None and np.isfinite(r["auroc_drop_vs_clean"])
        ]
        auprc_drop = [
            r["auprc_drop_vs_clean"] for r in rows
            if r["auprc_drop_vs_clean"] is not None and np.isfinite(r["auprc_drop_vs_clean"])
        ]
        out.setdefault(corruption, {})[str(severity)] = {
            "n_centers": len(rows),
            "mean_macro_auroc": float(np.mean(auroc)) if auroc else float("nan"),
            "mean_macro_auprc": float(np.mean(auprc)) if auprc else float("nan"),
            "mean_auroc_drop_vs_clean": float(np.mean(auroc_drop)) if auroc_drop else None,
            "mean_auprc_drop_vs_clean": float(np.mean(auprc_drop)) if auprc_drop else None,
        }
    return out


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
    p.add_argument("--device", default="cuda")
    p.add_argument("--crop_len", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--min_pos", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N records per cache for smoke tests.")
    p.add_argument("--output_path", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    scheme = get_scheme(args.scheme)
    model = _load_model(args, scheme, device)
    clean_by_center = _clean_lookup(args.clean_eval_json)

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
