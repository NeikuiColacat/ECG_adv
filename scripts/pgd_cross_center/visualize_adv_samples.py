"""Regenerate a small batch of adversarial samples from M1 (real-anchored) anchors
and visualize them as ECGTwin-gallery-style 12-lead figures.

Why regenerate? The training script `synth_online_at_super5.py` keeps the adv
buffer in-memory only; nothing is persisted. We replay the exact PGD recipe
(ε=2.0, K_pgd=10, delta_init_scale=0.1, fresh random restart per anchor) on
the K=200 anchor latents that already exist on disk, attacking the **Super5
baseline classifier** — i.e. the very first epoch's perspective, before AT
hardens the model. This is the most interpretable view for a clinician.

Anchor selection: only anchors with baseline p(target) > anchor_min_p are
attacked, so the perturbation actually has somewhere to push the prediction
to. (NORM in cpsc_2018_extra is mostly mislabeled by PTB-XL-trained baseline
under cross-center shift; without filtering, p_adv ≈ p_clean and the figure
shows nothing meaningful.)

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python \\
        scripts/pgd_cross_center/visualize_adv_samples.py \\
        --cell_tag extra_real_k200 --n_per_class 2

Outputs (per cell):
    /root/autodl-tmp/real_anchored_super5/viz_adv_samples/<cell>/
        <CLASS>/sample_<i>_clean.png      ← 12-lead anchor (decode of z0)
        <CLASS>/sample_<i>_adv.png        ← 12-lead adv (decode of z0+δ)
        <CLASS>/sample_<i>_compare.png    ← clean vs adv vs δ (3-row stacked)
        sanity.json                        ← per-sample HR / Einthoven / warnings
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adversarial.efficientnet_victim_tierM import EfficientNetVictimTierM
from adversarial.pgd_advdiff import PGDAdvDiffGenerator
from util.ecg_viz import _import_ecg_plot, sanity_check
from util.ecgtwin_utils import ECGTwinWrapper
from util.lead_utils import PTBXL_LEADS

CLASS_NAMES_SUPER5 = ["CD", "HYP", "MI", "NORM", "STTC"]
DEFAULT_SUPER5_CKPT = "/root/autodl-tmp/triple_labels/super5/best_model.pt"
M1_ROOT = Path("/root/autodl-tmp/real_anchored_super5")


# ──────────────────────────────────────────────────────────────────────────────
# Custom plotters (clean title placement + hidden offset axis labels)
# ──────────────────────────────────────────────────────────────────────────────
def plot_12lead_strip(
    sig: np.ndarray,           # (12, L) z-scored or mV
    save_path: Path,
    *,
    title: str,
    sample_rate: float = 100.0,
    row_height: float = 3.5,
    per_lead_norm: bool = False,
) -> None:
    """ECGTwin-gallery-style single-column 12-lead, z-scored input safe.

    Differences from util.ecg_viz.plot_ecg_ecgtwin_style:
      - Title goes ABOVE the strip via suptitle (no overlap with lead I)
      - Y-axis tick labels (the offset-stack numbers) are hidden — cardiologist
        reads ECG by lead label + grid spacing, not by absolute y-position
      - Global (not per-lead) p95 scaling preserves inter-lead amplitude
        relationships, which matters for HYP / axis assessment.
    """
    ecg_plot = _import_ecg_plot()
    if ecg_plot is None:
        raise RuntimeError("ecg_plot library not importable")

    arr = np.ascontiguousarray(sig.astype(np.float32))

    if per_lead_norm:
        peaks = np.percentile(np.abs(arr), 95, axis=1, keepdims=True)
        peaks = np.where(peaks > 1e-8, peaks, 1.0)
        arr = arr / peaks * (0.4 * row_height)
    else:
        # Global p95 scale: bulk of signal sits at ±0.4·row_half; preserves
        # cross-lead amplitude relationships (low-amplitude AVL stays low).
        global_p95 = float(np.percentile(np.abs(arr), 95))
        if global_p95 > 1e-6:
            arr = arr * (0.4 * row_height / global_p95)
    # Hard clip to row half-height to prevent bleed into adjacent leads.
    arr = np.clip(arr, -0.5 * row_height, 0.5 * row_height)

    ecg_plot.plot(
        arr,
        sample_rate=sample_rate,
        title="",                 # we'll place title via suptitle ourselves
        lead_index=list(PTBXL_LEADS),
        columns=1,
        row_height=row_height,
    )

    # Pretty ticks: 1-second majors, 0.2s minors
    from matplotlib.ticker import AutoMinorLocator
    ax = plt.gca()
    secs = arr.shape[1] / sample_rate
    ax.set_xticks(np.arange(0, secs + 1e-6, 1.0))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_yticklabels([])              # hide noisy offset numbers
    ax.set_ylabel("")
    ax.tick_params(axis="y", which="both", left=False, right=False)

    fig = plt.gcf()
    # Wrap long titles in two lines if needed
    fig.suptitle(title, fontsize=10, y=0.995, x=0.5, ha="center")
    fig.subplots_adjust(top=0.94)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare_overlay(
    clean: np.ndarray,        # (12, L)
    adv: np.ndarray,
    save_path: Path,
    *,
    title: str,
    sample_rate: float = 100.0,
) -> None:
    """4x3 grid, 12 leads, each panel overlays clean (blue) and adv (orange).

    Doctor-readable: each lead gets a sizable subplot, clean+adv share the
    same y-axis so true differences are honest. The δ panel is dropped —
    overlay alone shows the perturbation footprint clearly when ε ≈ 1σ.
    """
    leads = list(PTBXL_LEADS)
    L = clean.shape[1]
    t = np.arange(L) / sample_rate

    fig, axes = plt.subplots(3, 4, figsize=(20, 10), squeeze=False)
    for i, name in enumerate(leads):
        ax = axes[i // 4][i % 4]
        ax.plot(t, clean[i], color="#1f77b4", linewidth=0.9, alpha=0.85,
                label="clean (anchor)" if i == 0 else None)
        ax.plot(t, adv[i],   color="#ff7f0e", linewidth=0.9, alpha=0.85,
                label="adversarial"   if i == 0 else None)
        ax.set_title(name, fontsize=12, pad=3)
        ax.set_xlim(0, t[-1])
        ax.grid(True, alpha=0.35, color="pink")
        ax.tick_params(labelsize=8)
        if i % 4 == 0:
            ax.set_ylabel("amplitude (z-score)", fontsize=9)
        if i // 4 == 2:
            ax.set_xlabel("time (s)", fontsize=9)

    axes[0][0].legend(loc="upper right", fontsize=10, framealpha=0.85)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.subplots_adjust(top=0.93, hspace=0.35, wspace=0.20)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Anchor picking with min-prob filter
# ──────────────────────────────────────────────────────────────────────────────
def select_anchors_above_prob(
    latents: np.ndarray,        # (N, 4, 128)
    labels: np.ndarray,         # (N, C) one-hot
    cls_idx: int,
    n_pick: int,
    victim,
    pgd_gen: PGDAdvDiffGenerator,
    device: str,
    p_min: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (picked_indices, p_clean_for_picked).

    Pre-filters anchors of class `cls_idx` to those whose baseline target
    probability exceeds `p_min`. Falls back to top-K-by-p if not enough
    pass the filter (so the script never crashes on rare classes).
    """
    cls_mask = labels[:, cls_idx] > 0.5
    candidate_idx = np.where(cls_mask)[0]
    if len(candidate_idx) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)

    z = torch.from_numpy(latents[candidate_idx]).float().to(device)
    with torch.no_grad():
        x = pgd_gen._decode_to_ptbxl_1000(z)             # (N, 12, 1000)
        logits = victim.compute_logits_from_ecg(x.to(device))
        p_target = torch.sigmoid(logits)[:, cls_idx].cpu().numpy()

    keep_mask = p_target > p_min
    keep_local = np.where(keep_mask)[0]
    if len(keep_local) >= n_pick:
        # Random sample among the qualifying anchors
        chosen_local = rng.choice(keep_local, size=n_pick, replace=False)
    else:
        # Fall back: take top-K by descending p_target
        order = np.argsort(-p_target)
        chosen_local = order[:min(n_pick, len(p_target))]
        print(f"  [warn] only {len(keep_local)}/{len(candidate_idx)} anchors "
              f"of cls={cls_idx} pass p>{p_min}; fell back to top-{n_pick}-by-p")

    picked = candidate_idx[chosen_local]
    return picked, p_target[chosen_local]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_tag", default="extra_real_k200",
                    help="One of {extra,nin,geo}_real_k200")
    ap.add_argument("--n_per_class", type=int, default=2)
    ap.add_argument("--anchor_min_p", type=float, default=0.5,
                    help="Only attack anchors whose baseline p(target) > this")
    ap.add_argument("--init_ckpt", default=DEFAULT_SUPER5_CKPT)
    ap.add_argument("--pgd_eps", type=float, default=2.0)
    ap.add_argument("--pgd_K", type=int, default=10)
    ap.add_argument("--delta_init_scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_root", default=str(M1_ROOT / "viz_adv_samples"))
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    npz_path = M1_ROOT / "synth_latents" / f"{args.cell_tag}.latent.npz"
    npz = np.load(npz_path, allow_pickle=True)
    latents = npz["latents"]
    labels = npz["labels"]
    center_name = str(npz["center_name"])
    print(f"[load] {npz_path}")
    print(f"       latents={latents.shape}  labels={labels.shape}  center={center_name}")
    print(f"       per-class counts: " + ", ".join(
        f"{c}={int(labels[:, i].sum())}"
        for i, c in enumerate(CLASS_NAMES_SUPER5)))

    print(f"[load] Super5 baseline victim: {args.init_ckpt}")
    ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=False)
    victim = EfficientNetVictimTierM(
        weight_path=args.init_ckpt, num_classes=5, crop_len=250,
        ecgtwin_wrapper=ecgtwin,
    )

    pgd_gen = PGDAdvDiffGenerator(
        ecgtwin_wrapper=ecgtwin, victim=victim,
        epsilon=args.pgd_eps, K_pgd=args.pgd_K,
        alpha=None, delta_init_scale=args.delta_init_scale,
        device=args.device,
    )

    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out_root) / args.cell_tag
    out_root.mkdir(parents=True, exist_ok=True)

    sanity_log: dict = {
        "cell_tag": args.cell_tag,
        "center_name": center_name,
        "init_ckpt": args.init_ckpt,
        "anchor_min_p": args.anchor_min_p,
        "pgd": {"eps": args.pgd_eps, "K_pgd": args.pgd_K,
                "delta_init_scale": args.delta_init_scale, "alpha": pgd_gen.alpha},
        "samples": [],
    }

    for cls_idx, cls_name in enumerate(CLASS_NAMES_SUPER5):
        n_avail = int((labels[:, cls_idx] > 0.5).sum())
        if n_avail == 0:
            print(f"[skip] {cls_name}: 0 anchors in pool (untrusted, filtered out)")
            continue

        picked, p_clean_pre = select_anchors_above_prob(
            latents, labels, cls_idx,
            n_pick=args.n_per_class,
            victim=victim, pgd_gen=pgd_gen,
            device=args.device, p_min=args.anchor_min_p, rng=rng,
        )
        print(f"\n[class={cls_name}] picked {len(picked)}/{n_avail} anchors with "
              f"p_clean>{args.anchor_min_p}: indices={picked.tolist()}, "
              f"p_clean_baseline={[round(float(x), 3) for x in p_clean_pre]}")

        if len(picked) == 0:
            continue

        z_b = torch.from_numpy(latents[picked]).float().to(args.device)
        y_b = torch.from_numpy(labels[picked]).float().to(args.device)
        delta_init = torch.randn_like(z_b) * pgd_gen.delta_init_scale
        x_adv, delta = pgd_gen.attack_from_latent(z_b, y_b, delta_init=delta_init)
        with torch.no_grad():
            x_clean = pgd_gen._decode_to_ptbxl_1000(z_b)
        delta_norms = delta.detach().flatten(1).norm(dim=1).cpu().numpy()

        with torch.no_grad():
            logits_clean = victim.compute_logits_from_ecg(x_clean.to(args.device))
            logits_adv   = victim.compute_logits_from_ecg(x_adv.to(args.device))
            p_clean = torch.sigmoid(logits_clean)[:, cls_idx].cpu().numpy()
            p_adv   = torch.sigmoid(logits_adv)[:, cls_idx].cpu().numpy()

        cls_dir = out_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(picked)):
            anchor_idx = int(picked[i])
            clean_np = x_clean[i].detach().cpu().numpy().astype(np.float32)
            adv_np   = x_adv[i].detach().cpu().numpy().astype(np.float32)

            tag_clean = (
                f"CLEAN  |  center={center_name}  |  class={cls_name}  |  "
                f"anchor #{anchor_idx}  |  victim p({cls_name})={p_clean[i]:.3f}"
            )
            tag_adv = (
                f"ADVERSARIAL  |  center={center_name}  |  class={cls_name}  |  "
                f"anchor #{anchor_idx}  |  victim p({cls_name}): "
                f"{p_clean[i]:.3f} → {p_adv[i]:.3f}  |  "
                f"‖δ‖₂ = {delta_norms[i]:.3f}  (ε = {args.pgd_eps})"
            )
            tag_compare = (
                f"clean vs adversarial vs δ  |  class={cls_name}  |  "
                f"anchor #{anchor_idx}  |  p({cls_name}): "
                f"{p_clean[i]:.3f} → {p_adv[i]:.3f}  |  ‖δ‖₂={delta_norms[i]:.3f}"
            )

            clean_png = cls_dir / f"sample_{i}_clean.png"
            adv_png   = cls_dir / f"sample_{i}_adv.png"
            compare_png = cls_dir / f"sample_{i}_compare.png"

            plot_12lead_strip(clean_np, clean_png, title=tag_clean,
                              sample_rate=100.0, row_height=2.5)
            plot_12lead_strip(adv_np, adv_png, title=tag_adv,
                              sample_rate=100.0, row_height=2.5)
            plot_compare_overlay(clean_np, adv_np, compare_png,
                                 title=tag_compare, sample_rate=100.0)

            rep_clean = sanity_check(clean_np, 100.0, lead_order="ptbxl")
            rep_adv   = sanity_check(adv_np,   100.0, lead_order="ptbxl")

            sanity_log["samples"].append({
                "class": cls_name,
                "anchor_idx": anchor_idx,
                "files": {"clean": str(clean_png), "adv": str(adv_png),
                          "compare": str(compare_png)},
                "victim_p_target": {"clean": float(p_clean[i]),
                                    "adv":   float(p_adv[i])},
                "delta_l2_norm": float(delta_norms[i]),
                "sanity_clean": {
                    "hr_bpm": rep_clean["hr_estimate_bpm"],
                    "einthoven_residual": rep_clean["einthoven_residual"],
                    "n_warnings": len(rep_clean["warnings"]),
                    "warnings": rep_clean["warnings"],
                },
                "sanity_adv": {
                    "hr_bpm": rep_adv["hr_estimate_bpm"],
                    "einthoven_residual": rep_adv["einthoven_residual"],
                    "n_warnings": len(rep_adv["warnings"]),
                    "warnings": rep_adv["warnings"],
                },
            })
            print(f"  [{cls_name} {i}] anchor#{anchor_idx}  "
                  f"p_clean={p_clean[i]:.3f}→p_adv={p_adv[i]:.3f}  "
                  f"‖δ‖={delta_norms[i]:.3f}  HR_clean={rep_clean['hr_estimate_bpm']}  "
                  f"warn={len(rep_clean['warnings'])}/{len(rep_adv['warnings'])}")

    log_path = out_root / "sanity.json"
    with open(log_path, "w") as f:
        json.dump(sanity_log, f, indent=2)
    print(f"\n[done] sanity log → {log_path}")
    print(f"[done] PNGs under  → {out_root}")


if __name__ == "__main__":
    main()
