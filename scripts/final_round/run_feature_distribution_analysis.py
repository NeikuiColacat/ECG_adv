#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model" / "DeepECG" / "notebooks"))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402


CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]

DEFAULT_MODEL = "/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt"
DEFAULT_REAL_CACHE = "/root/autodl-tmp/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"
DEFAULT_LABELS = "/root/autodl-tmp/graduate_project/method_a_real2000_seed42/ptbxl_labels.C5.all.npy"
DEFAULT_SPLIT = "/root/autodl-tmp/graduate_project/splits/ptbxl_super5_seed42_train2000_val2000.json"
DEFAULT_SYNTH = (
    "/root/autodl-tmp/graduate_project/"
    "self_distill_v2_filtered_v46_ptbxl_contrast_seed42/"
    "synth_v2_filtered_top4000_gamma03.npz"
)
DEFAULT_OUT = "/root/autodl-tmp/final_round_ablation_20260504/feature_distribution"
DEFAULT_EVIDENCE = str(REPO / "final" / "artifacts" / "evidence_pack" / "feature_distribution")


@dataclass(frozen=True)
class FeatureSet:
    features: np.ndarray
    labels: np.ndarray
    class_index: np.ndarray
    source: str


def jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def build_model(checkpoint: Path, device: torch.device) -> EfficientNet1DV2:
    model = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=5,
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    ).to(device)
    sd = torch.load(checkpoint, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {str(k).removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def efficientnet_penultimate(model: EfficientNet1DV2, x: torch.Tensor) -> torch.Tensor:
    x = model.initial_conv(x)
    x = model.features(x)
    x = model.final_conv(x)
    x = model.final_norm(x)
    x = F.adaptive_avg_pool1d(x, 1).flatten(1)
    return x


def to_channels_first_batch(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D ECG batch, got {arr.shape}")
    if arr.shape[1:] == (1000, 12):
        arr = arr.transpose(0, 2, 1)
    elif arr.shape[1:] == (12, 1000):
        pass
    else:
        raise ValueError(f"expected (N,1000,12) or (N,12,1000), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def extract_features(
    model: EfficientNet1DV2,
    signals: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    feats: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = to_channels_first_batch(signals[batch_idx])
        x = torch.from_numpy(batch).to(device=device, dtype=torch.float32, non_blocking=True)
        feat = efficientnet_penultimate(model, x).float().cpu().numpy()
        feats.append(feat)
    return np.concatenate(feats, axis=0) if feats else np.empty((0, 640), dtype=np.float32)


def select_by_class(
    labels: np.ndarray,
    source_indices: np.ndarray,
    class_idx: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = source_indices[np.asarray(labels[source_indices, class_idx] > 0.5)]
    if len(candidates) < n:
        raise ValueError(f"class {CLASS_NAMES[class_idx]} only has {len(candidates)} candidates, requested {n}")
    return np.asarray(rng.choice(candidates, size=n, replace=False), dtype=np.int64)


def select_synth_by_class(
    synth_labels: np.ndarray,
    target_class: np.ndarray | None,
    class_idx: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_class is not None:
        candidates = np.where(target_class == class_idx)[0]
    else:
        candidates = np.where(synth_labels[:, class_idx] > 0.5)[0]
    if len(candidates) < n:
        raise ValueError(f"synth class {CLASS_NAMES[class_idx]} only has {len(candidates)} candidates, requested {n}")
    return np.asarray(rng.choice(candidates, size=n, replace=False), dtype=np.int64)


def covariance(x: np.ndarray) -> np.ndarray:
    return np.cov(x, rowvar=False).astype(np.float64)


def fid_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    mu_x = x64.mean(axis=0)
    mu_y = y64.mean(axis=0)
    cov_x = covariance(x64) + np.eye(x64.shape[1]) * eps
    cov_y = covariance(y64) + np.eye(y64.shape[1]) * eps
    covmean = sqrtm(cov_x @ cov_y)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = float(np.sum((mu_x - mu_y) ** 2) + np.trace(cov_x + cov_y - 2.0 * covmean))
    return max(value, 0.0)


def mean_l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(np.mean(x, axis=0) - np.mean(y, axis=0)))


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def rbf_mmd2(x: np.ndarray, y: np.ndarray, max_kernel_n: int = 512) -> float:
    scaler = StandardScaler()
    z = scaler.fit_transform(np.concatenate([x, y], axis=0))
    xz = z[: len(x)]
    yz = z[len(x):]
    rng = np.random.default_rng(123)
    zx = xz[rng.choice(len(xz), size=min(len(xz), max_kernel_n), replace=False)]
    zy = yz[rng.choice(len(yz), size=min(len(yz), max_kernel_n), replace=False)]
    zz = np.concatenate([zx, zy], axis=0)
    d = pairwise_sq_dists(zz, zz)
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    gamma = 1.0 / max(2.0 * med, 1e-6)
    kxx = np.exp(-gamma * pairwise_sq_dists(zx, zx))
    kyy = np.exp(-gamma * pairwise_sq_dists(zy, zy))
    kxy = np.exp(-gamma * pairwise_sq_dists(zx, zy))
    if len(zx) > 1:
        np.fill_diagonal(kxx, 0.0)
        xx = kxx.sum() / (len(zx) * (len(zx) - 1))
    else:
        xx = 0.0
    if len(zy) > 1:
        np.fill_diagonal(kyy, 0.0)
        yy = kyy.sum() / (len(zy) * (len(zy) - 1))
    else:
        yy = 0.0
    xy = kxy.mean()
    return float(max(xx + yy - 2.0 * xy, 0.0))


def c2st_balanced_accuracy(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    features = np.concatenate([x, y], axis=0)
    labels = np.concatenate([np.zeros(len(x), dtype=np.int64), np.ones(len(y), dtype=np.int64)])
    if min(len(x), len(y)) < 20:
        return float("nan"), float("nan")
    splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
    scores = []
    for train_idx, test_idx in splitter.split(features, labels):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(features[train_idx], labels[train_idx])
        pred = clf.predict(features[test_idx])
        scores.append(balanced_accuracy_score(labels[test_idx], pred))
    return float(np.mean(scores)), float(np.std(scores))


def compare_sets(name: str, class_name: str, a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    c2st_mean, c2st_std = c2st_balanced_accuracy(a, b, seed)
    return {
        "comparison": name,
        "class": class_name,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "fid": fid_distance(a, b),
        "mean_l2": mean_l2_distance(a, b),
        "mmd_rbf": rbf_mmd2(a, b),
        "c2st_bal_acc_mean": c2st_mean,
        "c2st_bal_acc_std": c2st_std,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "comparison", "class", "n_a", "n_b", "fid", "mean_l2", "mmd_rbf",
        "c2st_bal_acc_mean", "c2st_bal_acc_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def fmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.4f}"


def write_report(path: Path, rows: list[dict], config: dict) -> None:
    lines = [
        "# Feature Distribution Analysis",
        "",
        "Feature extractor: EfficientNet1DV2 penultimate global pooled feature.",
        "",
        "Interpretation: C2ST balanced accuracy close to 0.5 means two feature sets are hard to separate; higher means stronger distribution gap. FID/MMD are relative distances and should be compared against the real-train vs real-test baseline.",
        "",
        "## Summary",
        "",
        "| comparison | class | n_a | n_b | FID | MMD-RBF | C2ST bal. acc. |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {comparison} | {class_name} | {n_a} | {n_b} | {fid} | {mmd} | {c2st} |".format(
                comparison=row["comparison"],
                class_name=row["class"],
                n_a=row["n_a"],
                n_b=row["n_b"],
                fid=fmt(row["fid"]),
                mmd=fmt(row["mmd_rbf"]),
                c2st=fmt(row["c2st_bal_acc_mean"]),
            )
        )
    lines.extend([
        "",
        "## Config",
        "",
        "```json",
        json.dumps(jsonable(config), indent=2, ensure_ascii=False),
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_pca(
    path: Path,
    real_feats: np.ndarray,
    real_classes: np.ndarray,
    synth_feats: np.ndarray,
    synth_classes: np.ndarray,
) -> None:
    x = np.concatenate([real_feats, synth_feats], axis=0)
    source = np.concatenate([
        np.zeros(len(real_feats), dtype=np.int64),
        np.ones(len(synth_feats), dtype=np.int64),
    ])
    classes = np.concatenate([real_classes, synth_classes], axis=0)
    z = StandardScaler().fit_transform(x)
    xy = PCA(n_components=2, random_state=42).fit_transform(z)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = np.array(["#4C78A8", "#F58518"])
    for src, label in [(0, "real test"), (1, "synthetic")]:
        mask = source == src
        axes[0].scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.55, c=colors[src], label=label)
    axes[0].set_title("Feature PCA by source")
    axes[0].legend(markerscale=2)
    cmap = plt.get_cmap("tab10")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = classes == class_idx
        axes[1].scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.55, color=cmap(class_idx), label=class_name)
    axes[1].set_title("Feature PCA by intended class")
    axes[1].legend(markerscale=2, fontsize=8)
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def copy_to_evidence(out_dir: Path, evidence_dir: Path) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "feature_distribution_metrics.csv",
        "feature_distribution_summary.json",
        "feature_distribution_report.md",
        "feature_pca_real_vs_synth.png",
    ]:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, evidence_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", default=DEFAULT_MODEL)
    parser.add_argument("--real_cache", default=DEFAULT_REAL_CACHE)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--split_json", default=DEFAULT_SPLIT)
    parser.add_argument("--synth_npz", default=DEFAULT_SYNTH)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--evidence_dir", default=DEFAULT_EVIDENCE)
    parser.add_argument("--max_per_class", type=int, default=600)
    parser.add_argument("--pca_per_class", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    labels = np.load(args.labels, mmap_mode="r")
    real_signals = np.load(args.real_cache, mmap_mode="r")
    synth_npz = np.load(args.synth_npz, allow_pickle=True)
    synth_signals = synth_npz["signals"]
    synth_labels = np.asarray(synth_npz["labels"], dtype=np.float32)
    target_class = np.asarray(synth_npz["target_class"], dtype=np.int64) if "target_class" in synth_npz.files else None

    model = build_model(Path(args.model_ckpt), device)

    rows: list[dict] = []
    pca_real_feats: list[np.ndarray] = []
    pca_real_classes: list[np.ndarray] = []
    pca_synth_feats: list[np.ndarray] = []
    pca_synth_classes: list[np.ndarray] = []
    selection: dict[str, dict] = {}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        n_synth_avail = int((target_class == class_idx).sum()) if target_class is not None else int((synth_labels[:, class_idx] > 0.5).sum())
        n_train_avail = int((np.asarray(labels[train_indices, class_idx]) > 0.5).sum())
        n_test_avail = int((np.asarray(labels[test_indices, class_idx]) > 0.5).sum())
        n = min(args.max_per_class, n_synth_avail, n_train_avail, n_test_avail)
        if n < 20:
            continue

        train_sel = select_by_class(labels, train_indices, class_idx, n, rng)
        test_sel = select_by_class(labels, test_indices, class_idx, n, rng)
        synth_sel = select_synth_by_class(synth_labels, target_class, class_idx, n, rng)

        train_feat = extract_features(model, real_signals, train_sel, batch_size=args.batch_size, device=device)
        test_feat = extract_features(model, real_signals, test_sel, batch_size=args.batch_size, device=device)
        synth_feat = extract_features(model, synth_signals, synth_sel, batch_size=args.batch_size, device=device)

        rows.append(compare_sets("real_train2000_vs_real_test", class_name, train_feat, test_feat, args.seed))
        rows.append(compare_sets("synth_vs_real_test", class_name, synth_feat, test_feat, args.seed))
        rows.append(compare_sets("synth_vs_real_train2000", class_name, synth_feat, train_feat, args.seed))

        p = min(args.pca_per_class, n)
        pca_real_feats.append(test_feat[:p])
        pca_real_classes.append(np.full(p, class_idx, dtype=np.int64))
        pca_synth_feats.append(synth_feat[:p])
        pca_synth_classes.append(np.full(p, class_idx, dtype=np.int64))
        selection[class_name] = {
            "n": int(n),
            "train_indices_preview": train_sel[:10].tolist(),
            "test_indices_preview": test_sel[:10].tolist(),
            "synth_indices_preview": synth_sel[:10].tolist(),
            "available": {
                "synth": n_synth_avail,
                "train": n_train_avail,
                "test": n_test_avail,
            },
        }

    if pca_real_feats and pca_synth_feats:
        plot_pca(
            out_dir / "feature_pca_real_vs_synth.png",
            np.concatenate(pca_real_feats, axis=0),
            np.concatenate(pca_real_classes, axis=0),
            np.concatenate(pca_synth_feats, axis=0),
            np.concatenate(pca_synth_classes, axis=0),
        )

    config = {
        "model_ckpt": args.model_ckpt,
        "real_cache": args.real_cache,
        "labels": args.labels,
        "split_json": args.split_json,
        "synth_npz": args.synth_npz,
        "out_dir": str(out_dir),
        "max_per_class": args.max_per_class,
        "pca_per_class": args.pca_per_class,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "class_names": CLASS_NAMES,
        "selection": selection,
    }
    summary = {"config": config, "rows": rows}
    (out_dir / "feature_distribution_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "feature_distribution_metrics.csv", rows)
    write_report(out_dir / "feature_distribution_report.md", rows, config)

    if args.evidence_dir:
        copy_to_evidence(out_dir, Path(args.evidence_dir))

    print(json.dumps({
        "out_dir": str(out_dir),
        "evidence_dir": args.evidence_dir,
        "metrics_csv": str(out_dir / "feature_distribution_metrics.csv"),
        "report": str(out_dir / "feature_distribution_report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
