"""No-leak prompt-token style validation on PN2021 mmap caches.

This script runs two fast, auditable probes:

1. Per-target binary center-style probe.
   Train only on real PN2021 ECG, exclude the K500 prompt-token anchors by
   record_id, and match positives/negatives by primary super5 class.

2. Same-label C2ST.
   Compare held-out real target-center ECG against each synthetic arm with the
   same primary class mix. Lower C2ST balanced accuracy is better.

The probe uses downsampled full-10s waveform logistic regression. It is not a
replacement for an end-to-end EfficientNet probe, but it closes the no-leak and
same-label audit gap quickly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
DEFAULT_TARGET_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DEFAULT_REAL_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
REPO = Path(__file__).resolve().parent.parent.parent
DEEPECG_NB = REPO / "model" / "DeepECG" / "notebooks"
if str(DEEPECG_NB) not in sys.path:
    sys.path.insert(0, str(DEEPECG_NB))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402


def primary_class(label: np.ndarray, allowed: set[str]) -> str | None:
    vals = {CLASS_NAMES[i]: float(label[i]) for i in range(len(CLASS_NAMES))}
    # Treat NORM as pure NORM only; abnormal positives dominate.
    for cls in ["MI", "STTC", "CD", "HYP"]:
        if cls in allowed and vals.get(cls, 0.0) > 0.5:
            return cls
    if "NORM" in allowed and vals.get("NORM", 0.0) > 0.5:
        return "NORM"
    return None


def load_anchor_ids(cache_root: Path, center: str, k: int, seed: int) -> set[str]:
    path = cache_root / "ref_selection" / f"{center}_k{k}_seed{seed}.json"
    if not path.exists():
        return set()
    with path.open() as f:
        data = json.load(f)
    ids = data.get("ref_record_ids") or data.get("selected_record_ids") or []
    return {str(x) for x in ids}


def mmap_center_dir(root: Path, center: str) -> Path:
    return root / f"super5_{center}_100hz1000_v3_super5_normsuppress"


def load_center_index(
    root: Path,
    center: str,
    anchor_ids: set[str],
    allowed: set[str],
    seed: int,
) -> Dict:
    d = mmap_center_dir(root, center)
    signals = np.load(d / "signals.npy", mmap_mode="r")
    labels = np.load(d / "labels.npy", mmap_mode="r")
    record_ids = np.load(d / "record_ids.npy", allow_pickle=True, mmap_mode="r")
    by_class: Dict[str, List[int]] = defaultdict(list)
    excluded = 0
    for i, rid in enumerate(record_ids):
        if str(rid) in anchor_ids:
            excluded += 1
            continue
        cls = primary_class(np.asarray(labels[i]), allowed)
        if cls is not None:
            by_class[cls].append(i)

    rng = np.random.default_rng(seed)
    splits: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "val": {}, "test": {}}
    for cls, idx_list in by_class.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * 0.60)
        n_val = int(n * 0.20)
        splits["train"][cls] = idx[:n_train]
        splits["val"][cls] = idx[n_train:n_train + n_val]
        splits["test"][cls] = idx[n_train + n_val:]

    return {
        "center": center,
        "signals": signals,
        "labels": labels,
        "record_ids": record_ids,
        "excluded_anchor_count": int(excluded),
        "by_class_counts": {k: len(v) for k, v in by_class.items()},
        "splits": splits,
    }


def features_from_real(center_data: Dict, indices: np.ndarray, stride: int) -> np.ndarray:
    # mmap layout is (N,1000,12); convert to downsampled channel-first flatten.
    sig = np.asarray(center_data["signals"][indices], dtype=np.float32)
    sig = sig.transpose(0, 2, 1)
    if FEATURE_EXTRACTOR is not None:
        return FEATURE_EXTRACTOR.extract(sig)
    sig = sig[:, :, ::stride]
    return sig.reshape(sig.shape[0], -1)


def features_from_synth(signals_ct: np.ndarray, stride: int) -> np.ndarray:
    if FEATURE_EXTRACTOR is not None:
        return FEATURE_EXTRACTOR.extract(signals_ct.astype(np.float32, copy=False))
    sig = signals_ct[:, :, ::stride].astype(np.float32, copy=False)
    return sig.reshape(sig.shape[0], -1)


class EfficientNetFeatureExtractor:
    def __init__(
        self,
        ckpt: str,
        device: str = "cuda:0",
        crop_len: int = 250,
        batch_size: int = 256,
    ) -> None:
        self.device = torch.device(device)
        self.crop_len = int(crop_len)
        self.batch_size = int(batch_size)
        self.model = EfficientNet1DV2(
            variant="s_v2",
            input_channels=12,
            num_classes=len(CLASS_NAMES),
            activation="leaky_relu",
            stochastic_depth_prob=0.304,
            dropout_rate=0.0,
            use_se=True,
            norm_type="batch",
        )
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.to(self.device).eval()

    def _center_crop(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if length == self.crop_len:
            return x
        if length < self.crop_len:
            pad = self.crop_len - length
            return F.pad(x, (pad // 2, pad - pad // 2))
        start = (length - self.crop_len) // 2
        return x[..., start:start + self.crop_len]

    @torch.no_grad()
    def extract(self, signals_ct: np.ndarray) -> np.ndarray:
        outputs = []
        for i in range(0, len(signals_ct), self.batch_size):
            x = torch.from_numpy(np.ascontiguousarray(signals_ct[i:i + self.batch_size])).float()
            x = self._center_crop(x).to(self.device, non_blocking=True)
            m = self.model
            feat = m.initial_conv(x)
            feat = m.features(feat)
            feat = m.final_conv(feat)
            feat = m.final_norm(feat)
            feat = F.adaptive_avg_pool1d(feat, 1).flatten(1)
            outputs.append(feat.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)


FEATURE_EXTRACTOR: EfficientNetFeatureExtractor | None = None


def sample_indices(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or len(pool) == 0:
        return np.asarray([], dtype=np.int64)
    replace = len(pool) < n
    return rng.choice(pool, size=n, replace=replace).astype(np.int64)


def build_binary_split(
    centers: Dict[str, Dict],
    target: str,
    split: str,
    classes: List[str],
    max_per_class: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    X_parts, y_parts = [], []
    target_data = centers[target]
    for cls in classes:
        pos_pool = target_data["splits"][split].get(cls, np.asarray([], dtype=np.int64))
        if len(pos_pool) == 0:
            continue
        n_pos = min(len(pos_pool), max_per_class)
        pos_idx = sample_indices(pos_pool, n_pos, rng)
        neg_pools = [
            data["splits"][split].get(cls, np.asarray([], dtype=np.int64))
            for c, data in centers.items()
            if c != target
        ]
        neg_pool = np.concatenate([p for p in neg_pools if len(p) > 0]) if neg_pools else np.asarray([], dtype=np.int64)
        if len(neg_pool) == 0:
            continue
        neg_idx = sample_indices(neg_pool, n_pos, rng)
        # Need center-specific feature extraction for negatives.
        X_parts.append(features_from_real(target_data, pos_idx, build_binary_split.stride))
        y_parts.append(np.ones(n_pos, dtype=np.int64))

        neg_x = []
        remaining = n_pos
        for c, data in centers.items():
            if c == target:
                continue
            pool = data["splits"][split].get(cls, np.asarray([], dtype=np.int64))
            if len(pool) == 0:
                continue
            take = min(remaining, max(1, int(np.ceil(n_pos / max(1, len(centers) - 1)))))
            chosen = sample_indices(pool, take, rng)
            neg_x.append(features_from_real(data, chosen, build_binary_split.stride))
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            for c, data in centers.items():
                if c == target:
                    continue
                pool = data["splits"][split].get(cls, np.asarray([], dtype=np.int64))
                if len(pool) == 0:
                    continue
                chosen = sample_indices(pool, remaining, rng)
                neg_x.append(features_from_real(data, chosen, build_binary_split.stride))
                break
        neg = np.concatenate(neg_x, axis=0)[:n_pos]
        X_parts.append(neg)
        y_parts.append(np.zeros(len(neg), dtype=np.int64))

    if not X_parts:
        raise RuntimeError(f"No binary data for target={target} split={split}")
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


def fit_probe(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return scaler, clf


def eval_probe(scaler: StandardScaler, clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    Xs = scaler.transform(X)
    prob = clf.predict_proba(Xs)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "auroc": float(roc_auc_score(y, prob)),
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
    }


def synth_records(input_dir: Path, npz_name: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    with np.load(input_dir / npz_name, allow_pickle=True) as z:
        signals = z["signals"].astype(np.float32)
        labels = z["labels"].astype(np.float32)
        center = str(z["center_name"]) if "center_name" in z.files else input_dir.name
    summary_path = input_dir / "summary.json"
    if input_dir.name == "gated":
        summary_path = input_dir.parent / "summary.json"
    arm = input_dir.parent.name if input_dir.name == "gated" else input_dir.parent.name
    if summary_path.exists():
        rows = json.loads(summary_path.read_text()).get("records", [])
        if rows and rows[0].get("arm"):
            arm = str(rows[0]["arm"])
    return signals, labels, center, arm


def score_synth(
    probes: Dict[str, Tuple[StandardScaler, LogisticRegression]],
    input_dirs: Iterable[str],
    npz_name: str,
    classes: List[str],
    stride: int,
) -> List[Dict]:
    out = []
    allowed = set(classes)
    for raw_dir in input_dirs:
        input_dir = Path(raw_dir)
        if not (input_dir / npz_name).exists():
            continue
        signals, labels, center, arm = synth_records(input_dir, npz_name)
        if center not in probes:
            continue
        cls_rows = [primary_class(labels[i], allowed) for i in range(len(labels))]
        X = features_from_synth(signals, stride)
        scaler, clf = probes[center]
        probs = clf.predict_proba(scaler.transform(X))[:, 1]
        for cls in classes:
            idx = np.asarray([i for i, c in enumerate(cls_rows) if c == cls], dtype=np.int64)
            if len(idx) == 0:
                continue
            vals = probs[idx]
            out.append({
                "center": center,
                "arm": arm,
                "class": cls,
                "npz_name": npz_name,
                "n": int(len(idx)),
                "mean_p_target": float(vals.mean()),
                "median_p_target": float(np.median(vals)),
                "top_half_rate": float(np.mean(vals >= 0.5)),
                "input_dir": input_dir.as_posix(),
            })
    return out


def run_c2st(
    centers: Dict[str, Dict],
    input_dirs: Iterable[str],
    npz_name: str,
    classes: List[str],
    stride: int,
    max_real_per_class: int,
    repeats: int,
    seed: int,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    allowed = set(classes)
    rows = []
    for raw_dir in input_dirs:
        input_dir = Path(raw_dir)
        if not (input_dir / npz_name).exists():
            continue
        signals, labels, center, arm = synth_records(input_dir, npz_name)
        if center not in centers:
            continue
        synth_cls = [primary_class(labels[i], allowed) for i in range(len(labels))]
        real_x_parts, synth_x_parts = [], []
        for cls in classes:
            synth_idx = np.asarray([i for i, c in enumerate(synth_cls) if c == cls], dtype=np.int64)
            if len(synth_idx) == 0:
                continue
            real_pool = centers[center]["splits"]["test"].get(cls, np.asarray([], dtype=np.int64))
            if len(real_pool) < 4:
                real_pool = np.concatenate([
                    centers[center]["splits"][sp].get(cls, np.asarray([], dtype=np.int64))
                    for sp in ["train", "val", "test"]
                ])
            if len(real_pool) < 4:
                continue
            n = min(len(synth_idx), len(real_pool), max_real_per_class)
            if n < 4:
                continue
            real_idx = sample_indices(real_pool, n, rng)
            synth_take = sample_indices(synth_idx, n, rng)
            real_x_parts.append(features_from_real(centers[center], real_idx, stride))
            synth_x_parts.append(features_from_synth(signals[synth_take], stride))
        if not real_x_parts:
            continue
        real_X = np.concatenate(real_x_parts, axis=0)
        synth_X = np.concatenate(synth_x_parts, axis=0)
        X = np.concatenate([real_X, synth_X], axis=0)
        y = np.concatenate([np.zeros(len(real_X), dtype=np.int64), np.ones(len(synth_X), dtype=np.int64)])
        bacc, auroc = [], []
        for rep in range(repeats):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.35, stratify=y, random_state=seed + rep
            )
            scaler, clf = fit_probe(X_tr, y_tr)
            m = eval_probe(scaler, clf, X_te, y_te)
            bacc.append(m["balanced_accuracy"])
            auroc.append(m["auroc"])
        rows.append({
            "center": center,
            "arm": arm,
            "npz_name": npz_name,
            "n_real": int(len(real_X)),
            "n_synth": int(len(synth_X)),
            "balanced_accuracy_mean": float(np.mean(bacc)),
            "balanced_accuracy_std": float(np.std(bacc)),
            "auroc_mean": float(np.mean(auroc)),
            "auroc_std": float(np.std(auroc)),
            "input_dir": input_dir.as_posix(),
        })
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global FEATURE_EXTRACTOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmap_root", default="/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap")
    ap.add_argument("--anchor_cache_root", default="/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1")
    ap.add_argument("--target_centers", nargs="+", default=DEFAULT_TARGET_CENTERS)
    ap.add_argument("--real_centers", nargs="+", default=DEFAULT_REAL_CENTERS)
    ap.add_argument("--classes", nargs="+", default=["NORM", "MI", "STTC"])
    ap.add_argument("--input_dirs", nargs="+", required=True)
    ap.add_argument("--npz_name", default="samples.npz")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--selection_seed", type=int, default=42)
    ap.add_argument("--feature_stride", type=int, default=10)
    ap.add_argument("--feature_backend", choices=["waveform", "efficientnet"], default="waveform")
    ap.add_argument("--feature_ckpt", default="/root/autodl-tmp/triple_labels/super5/best_model.pt")
    ap.add_argument("--feature_device", default="cuda:0")
    ap.add_argument("--feature_batch_size", type=int, default=256)
    ap.add_argument("--feature_crop_len", type=int, default=250)
    ap.add_argument("--max_train_per_class", type=int, default=600)
    ap.add_argument("--max_eval_per_class", type=int, default=300)
    ap.add_argument("--max_c2st_real_per_class", type=int, default=80)
    ap.add_argument("--c2st_repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.feature_backend == "efficientnet":
        FEATURE_EXTRACTOR = EfficientNetFeatureExtractor(
            ckpt=args.feature_ckpt,
            device=args.feature_device,
            crop_len=args.feature_crop_len,
            batch_size=args.feature_batch_size,
        )
        print(
            f"[feature] efficientnet ckpt={args.feature_ckpt} "
            f"crop_len={args.feature_crop_len} batch={args.feature_batch_size}"
        )
    else:
        FEATURE_EXTRACTOR = None
        print(f"[feature] waveform stride={args.feature_stride}")

    build_binary_split.stride = int(args.feature_stride)
    allowed = set(args.classes)
    mmap_root = Path(args.mmap_root)
    anchor_root = Path(args.anchor_cache_root)
    centers: Dict[str, Dict] = {}
    for center in args.real_centers:
        anchors = load_anchor_ids(anchor_root, center, args.K, args.selection_seed)
        centers[center] = load_center_index(mmap_root, center, anchors, allowed, args.seed)
        print(
            f"[real] {center}: excluded_anchors={centers[center]['excluded_anchor_count']} "
            f"counts={centers[center]['by_class_counts']}"
        )

    rng = np.random.default_rng(args.seed)
    probes: Dict[str, Tuple[StandardScaler, LogisticRegression]] = {}
    real_rows = []
    for target in args.target_centers:
        if target not in centers:
            continue
        X_tr, y_tr = build_binary_split(centers, target, "train", args.classes, args.max_train_per_class, rng)
        X_va, y_va = build_binary_split(centers, target, "val", args.classes, args.max_eval_per_class, rng)
        X_te, y_te = build_binary_split(centers, target, "test", args.classes, args.max_eval_per_class, rng)
        scaler, clf = fit_probe(X_tr, y_tr)
        probes[target] = (scaler, clf)
        for split, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
            m = eval_probe(scaler, clf, X, y)
            m.update({"center": target, "split": split})
            real_rows.append(m)
        print(
            f"[probe] {target}: test_bacc={real_rows[-1]['balanced_accuracy']:.3f} "
            f"test_auroc={real_rows[-1]['auroc']:.3f}"
        )

    out_dir = Path(args.output_dir)
    write_csv(out_dir / f"real_binary_probe_{args.npz_name.replace('.', '_')}.csv", real_rows)
    synth_rows = score_synth(probes, args.input_dirs, args.npz_name, args.classes, args.feature_stride)
    write_csv(out_dir / f"synth_binary_scores_{args.npz_name.replace('.', '_')}.csv", synth_rows)
    c2st_rows = run_c2st(
        centers,
        args.input_dirs,
        args.npz_name,
        args.classes,
        args.feature_stride,
        args.max_c2st_real_per_class,
        args.c2st_repeats,
        args.seed,
    )
    write_csv(out_dir / f"same_label_c2st_{args.npz_name.replace('.', '_')}.csv", c2st_rows)

    summary = {
        "config": vars(args),
        "real_centers": {
            c: {
                "excluded_anchor_count": d["excluded_anchor_count"],
                "by_class_counts": d["by_class_counts"],
            }
            for c, d in centers.items()
        },
        "n_real_probe_rows": len(real_rows),
        "n_synth_score_rows": len(synth_rows),
        "n_c2st_rows": len(c2st_rows),
    }
    (out_dir / f"summary_{args.npz_name.replace('.', '_')}.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
