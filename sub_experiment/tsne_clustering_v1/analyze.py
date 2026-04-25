"""t-SNE + metrics for ECGTwin center-conditional generation clustering.

Metrics rationale:
  - k-NN purity: does a gen with ref center C end up near reals from C?
    Expected random baseline is 1/|centers|; values well above that mean
    reference-latent conditioning does carry center information.
  - Silhouette (reals only): sanity that the embedder separates centers
    at all. If this is low, the whole experiment has no signal and no
    conclusion can be drawn about gens.
  - Silhouette (reals + gens): does adding gens (labeled by their target
    center) degrade separation? If gens spread evenly across the real
    clusters, silhouette drops; if they cleanly augment the target cluster,
    silhouette stays.
  - MMD (RBF): distribution-level check — per target center, is the gen
    distribution closer to the same-center real distribution than to
    other-center real distributions?
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# ── Metrics ───────────────────────────────────────────────────────────────────

def knn_purity(
    real_feats: np.ndarray,
    real_center_ids: np.ndarray,
    gen_feats: np.ndarray,
    gen_target_ids: np.ndarray,
    k: int = 10,
) -> dict:
    """For each gen sample, fraction of its k nearest real neighbors that
    belong to its target center. Returns per-center and overall means.
    """
    nn_ = NearestNeighbors(n_neighbors=k).fit(real_feats)
    _, idx = nn_.kneighbors(gen_feats)                         # (N_gen, k)
    neighbor_centers = real_center_ids[idx]                    # (N_gen, k)
    hit = (neighbor_centers == gen_target_ids[:, None]).mean(axis=1)  # (N_gen,)

    per_center = {}
    for cid in np.unique(gen_target_ids):
        mask = gen_target_ids == cid
        per_center[int(cid)] = float(hit[mask].mean())
    return {"per_center": per_center, "overall": float(hit.mean())}


def _rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: Optional[float] = None) -> float:
    """Squared MMD with RBF kernel; sigma via median heuristic if None."""
    xy = np.vstack([x, y])
    if sigma is None:
        from scipy.spatial.distance import pdist
        d = pdist(xy, metric="euclidean")
        med = float(np.median(d)) if d.size else 1.0
        sigma = med if med > 1e-8 else 1.0
    g = 1.0 / (2.0 * sigma * sigma)

    def k(a, b):
        sqd = (
            (a ** 2).sum(axis=1, keepdims=True)
            + (b ** 2).sum(axis=1, keepdims=True).T
            - 2 * a @ b.T
        )
        return np.exp(-g * np.maximum(sqd, 0.0))

    Kxx = k(x, x)
    Kyy = k(y, y)
    Kxy = k(x, y)
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def mmd_per_center(
    real_feats: np.ndarray,
    real_center_ids: np.ndarray,
    gen_feats: np.ndarray,
    gen_target_ids: np.ndarray,
) -> dict:
    """For each target center C: MMD(gen_C, real_C) vs mean MMD(gen_C, real_other)."""
    out = {}
    for cid in np.unique(gen_target_ids):
        gen_c = gen_feats[gen_target_ids == cid]
        real_c = real_feats[real_center_ids == cid]
        if len(gen_c) < 2 or len(real_c) < 2:
            out[int(cid)] = {"mmd_same": None, "mmd_other_mean": None}
            continue
        mmd_same = _rbf_mmd2(gen_c, real_c)
        mmds_other = []
        for other_cid in np.unique(real_center_ids):
            if other_cid == cid:
                continue
            real_o = real_feats[real_center_ids == other_cid]
            if len(real_o) < 2:
                continue
            mmds_other.append(_rbf_mmd2(gen_c, real_o))
        out[int(cid)] = {
            "mmd_same": mmd_same,
            "mmd_other_mean": float(np.mean(mmds_other)) if mmds_other else None,
        }
    return out


def silhouette_real_and_all(
    real_feats: np.ndarray,
    real_center_ids: np.ndarray,
    gen_feats: np.ndarray,
    gen_target_ids: np.ndarray,
) -> dict:
    s_real = (
        float(silhouette_score(real_feats, real_center_ids))
        if len(np.unique(real_center_ids)) > 1
        else None
    )
    X = np.vstack([real_feats, gen_feats])
    y = np.concatenate([real_center_ids, gen_target_ids])
    s_all = (
        float(silhouette_score(X, y))
        if len(np.unique(y)) > 1
        else None
    )
    return {"silhouette_real": s_real, "silhouette_all": s_all}


# ── t-SNE + plotting ──────────────────────────────────────────────────────────

def run_tsne(
    real_feats: np.ndarray,
    gen_feats: np.ndarray,
    perplexity: float = 30.0,
    random_state: int = 0,
    n_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([real_feats, gen_feats])
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=n_iter,
        init="pca",
        learning_rate="auto",
    )
    Z = tsne.fit_transform(X)
    return Z[: len(real_feats)], Z[len(real_feats) :]


def _center_palette(n: int) -> list:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_tsne_all(
    real_xy: np.ndarray,
    gen_xy: np.ndarray,
    real_cids: np.ndarray,
    gen_cids: np.ndarray,
    center_names: list[str],
    save_path: Path,
) -> None:
    colors = _center_palette(len(center_names))
    fig, ax = plt.subplots(figsize=(10, 8))
    for cid, name in enumerate(center_names):
        m_real = real_cids == cid
        m_gen = gen_cids == cid
        if m_real.any():
            ax.scatter(
                real_xy[m_real, 0], real_xy[m_real, 1],
                s=18, c=[colors[cid]], marker="o", alpha=0.7,
                label=f"{name} (real)", edgecolors="none",
            )
        if m_gen.any():
            ax.scatter(
                gen_xy[m_gen, 0], gen_xy[m_gen, 1],
                s=40, c=[colors[cid]], marker="^", alpha=0.9,
                label=f"{name} (gen)", edgecolors="black", linewidths=0.5,
            )
    ax.set_title("t-SNE: PN2021 real + ECGTwin-gen (colored by center; triangles = gen)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tsne_per_center(
    real_xy: np.ndarray,
    gen_xy: np.ndarray,
    real_cids: np.ndarray,
    gen_cids: np.ndarray,
    center_names: list[str],
    save_dir: Path,
) -> list[Path]:
    out_paths = []
    colors = _center_palette(len(center_names))
    for cid, name in enumerate(center_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        # Gray background: all non-target reals
        m_bg = real_cids != cid
        if m_bg.any():
            ax.scatter(real_xy[m_bg, 0], real_xy[m_bg, 1], s=12, c="lightgray", alpha=0.5, label="other reals")
        # Target center reals
        m_tr = real_cids == cid
        if m_tr.any():
            ax.scatter(
                real_xy[m_tr, 0], real_xy[m_tr, 1],
                s=24, c=[colors[cid]], marker="o", alpha=0.85, label=f"{name} (real)",
            )
        # Target center gens
        m_tg = gen_cids == cid
        if m_tg.any():
            ax.scatter(
                gen_xy[m_tg, 0], gen_xy[m_tg, 1],
                s=50, c=[colors[cid]], marker="^", alpha=1.0,
                label=f"{name} (gen)", edgecolors="black", linewidths=0.7,
            )
        ax.set_title(f"t-SNE — target center: {name}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        p = save_dir / f"tsne_per_center_{name}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out_paths.append(p)
    return out_paths


def plot_purity_bar(
    purity_per_center: dict,
    center_names: list[str],
    k: int,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cids = sorted(purity_per_center.keys())
    names = [center_names[c] for c in cids]
    vals = [purity_per_center[c] for c in cids]
    colors = _center_palette(len(center_names))
    bar_colors = [colors[c] for c in cids]
    ax.bar(names, vals, color=bar_colors, edgecolor="black")
    random_baseline = 1.0 / len(center_names)
    ax.axhline(random_baseline, color="red", linestyle="--", label=f"random baseline={random_baseline:.2f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"k-NN purity (k={k})")
    ax.set_title("gen→real k-NN purity per target center")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Top-level entry ───────────────────────────────────────────────────────────

def run_full_analysis(
    real_feats: np.ndarray,
    real_center_ids: np.ndarray,
    gen_feats: np.ndarray,
    gen_target_ids: np.ndarray,
    center_names: list[str],
    output_root: Path,
    k: int = 10,
    perplexity: float = 30.0,
    random_state: int = 0,
    n_iter: int = 1000,
) -> dict:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    real_xy, gen_xy = run_tsne(
        real_feats, gen_feats,
        perplexity=perplexity, random_state=random_state, n_iter=n_iter,
    )
    np.save(output_root / "tsne_coords.npy",
            np.vstack([real_xy, gen_xy]).astype(np.float32))

    plot_tsne_all(real_xy, gen_xy, real_center_ids, gen_target_ids,
                  center_names, plots_dir / "tsne_all.png")
    plot_tsne_per_center(real_xy, gen_xy, real_center_ids, gen_target_ids,
                         center_names, plots_dir)

    purity = knn_purity(real_feats, real_center_ids, gen_feats, gen_target_ids, k=k)
    plot_purity_bar(purity["per_center"], center_names, k, plots_dir / "purity_bar.png")

    sil = silhouette_real_and_all(real_feats, real_center_ids, gen_feats, gen_target_ids)
    mmd = mmd_per_center(real_feats, real_center_ids, gen_feats, gen_target_ids)

    metrics = {
        "knn_k": k,
        "knn_purity_overall": purity["overall"],
        "knn_purity_per_center": {
            center_names[cid]: purity["per_center"][cid] for cid in purity["per_center"]
        },
        "silhouette_real": sil["silhouette_real"],
        "silhouette_all": sil["silhouette_all"],
        "mmd_per_center": {
            center_names[cid]: mmd[cid] for cid in mmd
        },
        "random_baseline_purity": 1.0 / len(center_names),
    }
    with open(output_root / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
