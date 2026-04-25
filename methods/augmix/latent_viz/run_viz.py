"""End-to-end: load real ECGs -> latent AugMix -> visualize + sanity-check.

For each sampled record, produce:
  - plots/original.png / vae_roundtrip.png / augmix_output.png (single 12-lead)
  - plots/chain_{i}_{op1+op2+...}.png (time-domain chain pre-encode)
  - plots/compare_overlay.png (original vs vae_roundtrip vs augmix_output)
  - report.json with sanity_check on each version + AugMix metadata (ws, m, ops)

We save plots for the VAE-roundtrip as well: it is the best baseline for "what
does the decoder do to a perfect latent with no aug?". Any physiological
violations in the AugMix output that are also present in the roundtrip are
attributable to the VAE, not to the augmentation.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from util.ecgtwin_utils import load_ecgtwin
from util.ecg_viz import plot_ecg_ecgtwin_style, plot_comparison, sanity_check
from sub_experiment.tsne_clustering_v1.data_loader import load_all_centers, RealSample

from methods.augmix.latent_viz.latent_augmix import (
    LatentAugMixResult,
    latent_augmix_on_signal,
)


# canonical preprocessing (PTBXL order) -> plots use "ptbxl" labels
_CANONICAL_LEAD_ORDER = "ptbxl"
_CANONICAL_SR = 100.0


def _save_single_plot(sig_ct: np.ndarray, save_path: Path, title: str) -> None:
    # Ported 1:1 from ECGTwin author (model/ECGTwin/utils/inference_utils.py:116):
    # one-column rhythm-strip via ecg_plot.plot with columns=1, row_height=4.
    #
    # per_lead_norm=True because our signals are per-sample z-scored, not mV:
    # precordial std ≈ 1.5σ, limb std ≈ 0.4σ, so without normalization V1–V6
    # QRS spills into neighboring rows while limbs get squashed into flat
    # lines. Per-lead rescaling to std ≈ 0.4 "mV" lets morphology read in
    # every lead; it does sacrifice the ability to judge inter-lead amplitude
    # relationships visually (QRS axis, precordial progression), but those
    # are captured in report.json (p2p_per_lead) and via the sanity Einthoven
    # check anyway.
    plot_ecg_ecgtwin_style(
        sig_ct, save_path,
        sample_rate=_CANONICAL_SR,
        lead_order=_CANONICAL_LEAD_ORDER,
        title=title,
        per_lead_norm=True,
    )


def _encode_chain_tag(op_seq: List[str]) -> str:
    return "+".join(op_seq) if op_seq else "noop"


def _report_sample(
    res: LatentAugMixResult,
    sample: RealSample,
    out_dir: Path,
) -> dict:
    """Write plots + JSON for one sample; return the JSON payload for aggregation."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{sample.center}/{sample.record_id}"
    _save_single_plot(res.original_ct,         plots_dir / "original.png",       f"{tag} | original")
    _save_single_plot(res.vae_roundtrip_ct,    plots_dir / "vae_roundtrip.png",  f"{tag} | VAE roundtrip")
    _save_single_plot(res.augmix_latent_ct,    plots_dir / "augmix_output.png",  f"{tag} | latent AugMix")

    for i, (chain_sig, op_seq) in enumerate(zip(res.chain_signals_ct, res.chain_op_sequences)):
        chain_name = f"chain_{i}_{_encode_chain_tag(op_seq)}"
        _save_single_plot(chain_sig, plots_dir / f"{chain_name}.png", f"{tag} | {chain_name}")

    plot_comparison(
        signals=[res.original_ct, res.vae_roundtrip_ct, res.augmix_latent_ct],
        labels=["original", "vae_roundtrip", "augmix_output"],
        sample_rate=_CANONICAL_SR,
        save_path=plots_dir / "compare_overlay.png",
        mode="overlay",
        lead_order=_CANONICAL_LEAD_ORDER,
    )

    reports = {
        "original":       sanity_check(res.original_ct,      _CANONICAL_SR, lead_order=_CANONICAL_LEAD_ORDER),
        "vae_roundtrip":  sanity_check(res.vae_roundtrip_ct, _CANONICAL_SR, lead_order=_CANONICAL_LEAD_ORDER),
        "augmix_output":  sanity_check(res.augmix_latent_ct, _CANONICAL_SR, lead_order=_CANONICAL_LEAD_ORDER),
    }
    for i, chain_sig in enumerate(res.chain_signals_ct):
        reports[f"chain_{i}"] = sanity_check(
            chain_sig, _CANONICAL_SR, lead_order=_CANONICAL_LEAD_ORDER,
        )

    summary = {
        "sample": {
            "center": sample.center,
            "record_id": sample.record_id,
            "age": sample.age, "sex": sample.sex,
            "hr_estimate_bpm": sample.hr,
        },
        "augmix": {
            "severity": res.severity, "width": res.width, "depth": res.depth,
            "dirichlet_weights": res.dirichlet_weights.tolist(),
            "beta_m": res.beta_m,
            "chain_op_sequences": res.chain_op_sequences,
        },
        "sanity": reports,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(config_path: str) -> dict:
    cfg_path = Path(config_path).resolve()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_root = Path(cfg["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "config.snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"[stage] loading {cfg['n_samples_per_center']} samples x {len(cfg['centers'])} centers")
    samples = load_all_centers(
        centers=cfg["centers"],
        pn2021_root=cfg["pn2021_root"],
        n_per_center=cfg["n_samples_per_center"],
        seed=cfg.get("seed", 0),
    )
    print(f"[stage] loaded {len(samples)} records")

    print(f"[stage] loading ECGTwin (encoder + decoder; text={cfg['ecgtwin']['load_text_model']})")
    wrapper = load_ecgtwin(
        device=cfg["ecgtwin"]["device"],
        load_encoder=cfg["ecgtwin"]["load_encoder"],
        load_text_model=cfg["ecgtwin"]["load_text_model"],
    )

    aug_cfg = cfg["augmix"]
    seed0 = cfg.get("seed", 0)

    overall: List[dict] = []
    for idx, sample in enumerate(samples):
        print(f"[run] {idx + 1}/{len(samples)}  {sample.center}/{sample.record_id}")
        res = latent_augmix_on_signal(
            wrapper,
            sample.signal,
            severity=aug_cfg["severity"],
            width=aug_cfg["width"],
            depth=aug_cfg["depth"],
            alpha=aug_cfg["alpha"],
            ops=aug_cfg["ops"],
            device=cfg["ecgtwin"]["device"],
            rng_seed=seed0 + idx,
        )
        sample_dir = out_root / sample.center / sample.record_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        summary = _report_sample(res, sample, sample_dir)
        overall.append(summary)

    # aggregate: overall warnings / HR stats
    agg = {
        "n_samples": len(overall),
        "warnings_per_version": {},
        "hr_per_version": {},
    }
    versions = ["original", "vae_roundtrip", "augmix_output"]
    for v in versions:
        warn_counts = [len(s["sanity"][v]["warnings"]) for s in overall]
        hrs = [s["sanity"][v]["hr_estimate_bpm"] for s in overall]
        hrs = [h for h in hrs if h is not None]
        agg["warnings_per_version"][v] = {
            "mean": float(np.mean(warn_counts)) if warn_counts else 0.0,
            "max": int(np.max(warn_counts)) if warn_counts else 0,
            "n_with_warnings": int(sum(1 for c in warn_counts if c > 0)),
        }
        agg["hr_per_version"][v] = {
            "n_estimable": len(hrs),
            "mean_bpm": float(np.mean(hrs)) if hrs else None,
            "std_bpm": float(np.std(hrs)) if hrs else None,
        }
    with open(out_root / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[done] output_root={out_root}")
    print(f"[done] aggregate={json.dumps(agg, indent=2)}")
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
    )
    args = ap.parse_args()
    main(args.config)
