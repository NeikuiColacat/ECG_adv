"""End-to-end runner for the ECGTwin center-conditional clustering experiment.

Stage 1 — Load PN2021 real samples from N centers, preprocess to (12,1000)@100Hz.
Stage 2 — Embed real via ECGFounder; pick M refs per center and run ECGTwin; embed gens.
Stage 3 — t-SNE + per-center k-NN purity / silhouette / MMD, dump metrics + plots.

Intermediate .pt artifacts land under config.output_root (default
/root/autodl-tmp/sub_experiment/tsne_clustering_v1/). Re-running will reuse
existing artifacts — delete them to force a full rebuild.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sub_experiment.tsne_clustering_v1 import analyze
from sub_experiment.tsne_clustering_v1.data_loader import (
    load_all_centers, samples_to_arrays, RealSample,
)
from sub_experiment.tsne_clustering_v1.embedders import ECGFounderEmbedder
from sub_experiment.tsne_clustering_v1.generation import (
    ECGTwinGenerator, pack_generated,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_refs_per_center(
    samples: list[RealSample],
    centers: list[str],
    n_gen_per_center: int,
    seed: int,
) -> list[RealSample]:
    """Pick M distinct refs per center from the already-loaded real samples."""
    rng = random.Random(seed + 100)
    by_center = {c: [] for c in centers}
    for s in samples:
        by_center[s.center].append(s)
    refs = []
    for c in centers:
        pool = by_center[c]
        if len(pool) < n_gen_per_center:
            raise RuntimeError(
                f"Center '{c}' has only {len(pool)} reals but {n_gen_per_center} refs requested."
            )
        refs.extend(rng.sample(pool, n_gen_per_center))
    return refs


def _stage1_reals(cfg: dict, out_root: Path) -> dict:
    """Load real samples + ECGFounder features. Cache to disk."""
    sig_path = out_root / "real_signals.pt"
    feat_path = out_root / "real_features.pt"
    if sig_path.exists() and feat_path.exists():
        print(f"[stage1] reusing cached {sig_path.name}, {feat_path.name}")
        sig_pkg = torch.load(sig_path, weights_only=False)
        feat_pkg = torch.load(feat_path, weights_only=False)
        return {**sig_pkg, **feat_pkg}

    print("[stage1] loading PN2021 real samples ...")
    t0 = time.time()
    samples = load_all_centers(
        centers=cfg["centers"],
        pn2021_root=cfg["pn2021_root"],
        n_per_center=cfg["n_real_per_center"],
        seed=cfg["seed"],
    )
    arrs = samples_to_arrays(samples, cfg["centers"])
    arrs["samples"] = samples
    print(f"[stage1] loaded {len(samples)} reals in {time.time() - t0:.1f}s")
    torch.save(
        {k: arrs[k] for k in
         ["signals", "center_ids", "center_names", "ages", "sexes", "hrs", "snomed", "record_ids"]},
        sig_path,
    )

    print("[stage1] extracting ECGFounder features ...")
    embedder = ECGFounderEmbedder(
        weights_path=cfg["ecgfounder"]["weights_path"],
        device=cfg["ecgfounder"]["device"],
    )
    signals_t = torch.from_numpy(arrs["signals"]).float()
    feats = embedder.extract_features(signals_t, batch_size=32)
    torch.save({"features": feats}, feat_path)
    arrs["features"] = feats
    return arrs


def _stage2_gens(cfg: dict, reals: dict, out_root: Path) -> dict:
    """Pick refs, run ECGTwin, extract ECGFounder features on gens. Cache to disk."""
    sig_path = out_root / "gen_signals.pt"
    feat_path = out_root / "gen_features.pt"
    if sig_path.exists() and feat_path.exists():
        print(f"[stage2] reusing cached {sig_path.name}, {feat_path.name}")
        sig_pkg = torch.load(sig_path, weights_only=False)
        feat_pkg = torch.load(feat_path, weights_only=False)
        return {**sig_pkg, **feat_pkg}

    refs = _pick_refs_per_center(
        samples=reals["samples"],
        centers=cfg["centers"],
        n_gen_per_center=cfg["n_gen_per_center"],
        seed=cfg["seed"],
    )
    print(f"[stage2] loading ECGTwin and generating {len(refs)} samples ...")
    gen = ECGTwinGenerator(
        device=cfg["ecgtwin"]["device"],
        load_encoder=cfg["ecgtwin"]["load_encoder"],
        load_text_model=cfg["ecgtwin"]["load_text_model"],
        num_inference_steps=cfg["ecgtwin"]["num_inference_steps"],
    )
    t0 = time.time()
    gen_sigs = gen.generate_for_samples(refs)
    print(f"[stage2] generation done in {time.time() - t0:.1f}s")

    pack = pack_generated(gen_sigs, refs, cfg["centers"])
    torch.save(pack, sig_path)

    print("[stage2] extracting ECGFounder features on gens ...")
    embedder = ECGFounderEmbedder(
        weights_path=cfg["ecgfounder"]["weights_path"],
        device=cfg["ecgfounder"]["device"],
    )
    gen_t = torch.from_numpy(pack["signals"]).float()
    feats = embedder.extract_features(gen_t, batch_size=32)
    torch.save({"features": feats}, feat_path)
    pack["features"] = feats
    return pack


def _stage3_analyze(cfg: dict, reals: dict, gens: dict, out_root: Path) -> dict:
    print("[stage3] running t-SNE + metrics ...")
    return analyze.run_full_analysis(
        real_feats=reals["features"],
        real_center_ids=reals["center_ids"],
        gen_feats=gens["features"],
        gen_target_ids=gens["target_center_ids"],
        center_names=cfg["centers"],
        output_root=out_root,
        k=cfg["metrics"]["knn_k"],
        perplexity=cfg["tsne"]["perplexity"],
        random_state=cfg["tsne"]["random_state"],
        n_iter=cfg["tsne"]["n_iter"],
    )


def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    _set_seed(cfg["seed"])

    out_root = Path(cfg["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    reals = _stage1_reals(cfg, out_root)
    gens = _stage2_gens(cfg, reals, out_root)
    metrics = _stage3_analyze(cfg, reals, gens, out_root)

    print("\n=== metrics ===")
    print(f"random baseline purity = {metrics['random_baseline_purity']:.3f}")
    print(f"k-NN purity overall    = {metrics['knn_purity_overall']:.3f}")
    print(f"silhouette (real only) = {metrics['silhouette_real']}")
    print(f"silhouette (real+gen)  = {metrics['silhouette_all']}")
    print("k-NN purity per center:")
    for name, v in metrics["knn_purity_per_center"].items():
        print(f"  {name:25s}  {v:.3f}")
    print("MMD per center:")
    for name, d in metrics["mmd_per_center"].items():
        print(f"  {name:25s}  same={d['mmd_same']}  other_mean={d['mmd_other_mean']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
    )
    args = ap.parse_args()
    main(args.config)
