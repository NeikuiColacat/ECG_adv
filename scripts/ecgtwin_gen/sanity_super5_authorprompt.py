"""Sanity #3: vanilla ECGTwin with the AUTHOR'S OWN demo prompts.

The fixed-NORM-ref sanity (#1) showed CD p_target=0.04 with our super5 fallback
prompt `complete left bundle branch block|lbbb`. But the author's demo
gallery uses `left bundle branch block|lbbb` (no "complete"). Other classes
also differ (we use a STTC grouping prompt that has no analogue in the demo
gallery; the author uses NSTEMI as a closer specific-disease anchor).

This script runs the author's *exact* demo prompts (NORM ref → 12 disease
targets) under our victim probe, to separate two confounders:
  (a) prompt-text difference (we vs author)
  (b) super5 victim OOD on ECGTwin synth domain (will affect any prompt)

If author prompts give strong p_target on at least the CD/MI/HYP classes
that have a clear super5 mapping → we were wrong about "vanilla can't do
5 classes", we just had bad prompt for CD. If author prompts also give
p_target ≤ 0.4 → confirms the bottleneck is super5 victim OOD, not prompt.

Run:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/sanity_super5_authorprompt.py \
    --n_per_class 3 --steps 50 \
    --out_dir outputs/sanity_super5_authorprompt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from util.ecgtwin_utils import ECGTwinWrapper            # noqa: E402
from util.ecg_viz import plot_with_report                # noqa: E402
from adversarial.efficientnet_victim_tierM import EfficientNetVictimTierM  # noqa: E402

# Each entry: (super5 super-class, author-style prompt)
# Author prompts taken verbatim from generation_result_by_disease/*/features.json
AUTHOR_PROMPTS = [
    ("NORM",  "sinus rhythm|normal ecg."),
    ("MI",    "myocardial infarction|st elevation|anterior wall"),
    ("MI",    "stemi|st elevation myocardial infarction|acute"),
    ("STTC",  "nstemi|non st elevation|t wave inversion"),     # closest demo to STTC
    ("STTC",  "acute pericarditis|diffuse st elevation"),       # also ST/T pathology
    ("HYP",   "left ventricular hypertrophy|high voltage"),
    ("CD",    "left bundle branch block|lbbb"),                 # ← differs from our fallback
    ("CD",    "right bundle branch block|rbbb"),
    ("CD",    "atrioventricular block|av block"),
    ("NORM",  "sinus bradycardia|slow heart rate"),             # NORM grouping covers SBRAD
    ("NORM",  "sinus tachycardia|fast heart rate"),             # NORM grouping covers STACH
    ("NORM",  "atrial fibrillation|irregular rhythm"),          # AF in PTBXL super5 → NORM-grouping has no AF; this is a probe
]

SUPER5_CLASSES = ["NORM", "MI", "HYP", "CD", "STTC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_pt",
                    default="/root/ECG_adv_Gen/model/ECGTwin/data/prepared_input/normal_1.pt")
    ap.add_argument("--victim_ckpt",
                    default="/root/autodl-tmp/triple_labels/super5/best_model.pt")
    ap.add_argument("--n_per_class", type=int, default=3,
                    help="Samples per (super5_cls, prompt) cell")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="outputs/sanity_super5_authorprompt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base_seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ECGTwin WITH text model (we use raw text → on-the-fly nomic) ────
    print(f"[setup] loading ECGTwin with text model (nomic encode on-the-fly)...")
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=True)

    print(f"[setup] ref: {args.ref_pt}")
    ref = torch.load(args.ref_pt, map_location="cpu", weights_only=False)
    ref_latent = ref["data"]
    ref_label = dict(ref["label"])
    print(f"    ref hr={ref_label.get('hr')}, age={ref_label.get('age')}, sex={ref_label.get('sex')}")
    print(f"    ref text='{ref_label.get('text')}'")

    print(f"[setup] Super5 victim: {args.victim_ckpt}")
    victim = EfficientNetVictimTierM(
        weight_path=args.victim_ckpt,
        device=args.device,
        ecgtwin_wrapper=wrapper,
        num_classes=5,
        crop_len=250,
    )

    results = []
    for cell_i, (cls, prompt) in enumerate(AUTHOR_PROMPTS):
        slug = prompt.split("|")[0].replace(" ", "_")
        cell_dir = out / f"{cls}__{slug}"
        cell_dir.mkdir(exist_ok=True)
        for k in range(args.n_per_class):
            seed = args.base_seed + cell_i * 1000 + k
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            cond = wrapper.prepare_conditions(
                ref_latent=ref_latent,
                ref_label=ref_label,
                batch_size=1,
                target_text=prompt,            # ← raw text, on-the-fly nomic encode
            )
            x_init = torch.randn(1, 4, 128, device=wrapper.device)
            latent = wrapper.ddpm_sample(
                cond, batch_size=1, num_inference_steps=args.steps, x_init=x_init,
            )
            ecg_tc = wrapper.decode_latent(latent)

            with torch.no_grad():
                probs = victim.forward_from_latent(latent, enable_grad=False)[0].cpu().numpy()
            probs_dict = {SUPER5_CLASSES[i]: float(probs[i]) for i in range(5)}
            top1 = SUPER5_CLASSES[int(np.argmax(probs))]

            sig = ecg_tc[0].detach().cpu().numpy()
            png_path = cell_dir / f"seed{seed}.png"
            title = f"{cls}|{prompt} | top1={top1} (p={float(probs.max()):.2f})"
            _, report = plot_with_report(
                sig, sample_rate=102.4, save_path=png_path,
                title_prefix=title, lead_order="ecgtwin", engine="matplotlib",
            )

            entry = {
                "super5_target": cls,
                "prompt": prompt,
                "seed": seed,
                "png": str(png_path),
                "top1": top1,
                "p_target_super5": probs_dict[cls],
                "victim_probs": probs_dict,
                "hr_bpm": report["hr_estimate_bpm"],
                "einthoven_residual": report["einthoven_residual"],
                "warnings": report["warnings"],
            }
            results.append(entry)
            hr_str = f"HR={report['hr_estimate_bpm']:.0f}" if report['hr_estimate_bpm'] else "HR=N/A"
            print(f"  [{cls:<5} '{prompt[:40]:<40}' seed={seed}]  "
                  f"top1={top1:<5} p_super5={probs_dict[cls]:.3f}  "
                  f"{hr_str}  einth={report['einthoven_residual']:.3f}")

    # ── Aggregate per (super5_cls, prompt) cell ──────────────────────────────
    cells = {}
    for r in results:
        key = f"{r['super5_target']}|{r['prompt']}"
        cells.setdefault(key, []).append(r)

    cell_summary = {}
    for key, runs in cells.items():
        cls = runs[0]["super5_target"]
        cell_summary[key] = {
            "super5_target": cls,
            "prompt": runs[0]["prompt"],
            "n_samples": len(runs),
            "p_target_avg": float(np.mean([r["p_target_super5"] for r in runs])),
            "p_target_max": float(np.max([r["p_target_super5"] for r in runs])),
            "top1_hit_rate": sum(r["top1"] == cls for r in runs) / len(runs),
            "einthoven_pass_p20": sum(r["einthoven_residual"] < 0.2 for r in runs) / len(runs),
            "hr_in_range": sum(40 <= (r["hr_bpm"] or 0) <= 180 for r in runs) / len(runs),
        }

    summary = {
        "config": {
            "n_per_class": args.n_per_class,
            "steps": args.steps,
            "ref_pt": args.ref_pt,
            "centertoken_loaded": False,
            "note": "Vanilla ECGTwin + author's exact demo prompts; "
                    "fixed-normal ref (matches the author's demo gallery setup).",
        },
        "per_cell": cell_summary,
        "global": {
            "frac_einthoven_pass_p20": sum(
                r["einthoven_residual"] < 0.2 for r in results
            ) / len(results),
            "frac_hr_in_40_180": sum(
                40 <= (r["hr_bpm"] or 0) <= 180 for r in results
            ) / len(results),
            "n_warnings_total": sum(len(r["warnings"]) for r in results),
            "n_samples": len(results),
        },
        "results": results,
    }

    json_path = out / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print(f"[done] PNGs + summary.json in {out}")
    print()
    print(f"{'super5':<6} {'author prompt':<55} {'p_avg':>6} {'top1_hit':>8}")
    print("-" * 80)
    for key, info in cell_summary.items():
        print(f"{info['super5_target']:<6} {info['prompt'][:53]:<55} "
              f"{info['p_target_avg']:>6.3f} {info['top1_hit_rate']:>8.2f}")
    print()
    print(f"global Einthoven<0.2: {summary['global']['frac_einthoven_pass_p20']:.2f}")
    print(f"global HR ∈ [40,180]:  {summary['global']['frac_hr_in_40_180']:.2f}")


if __name__ == "__main__":
    main()
