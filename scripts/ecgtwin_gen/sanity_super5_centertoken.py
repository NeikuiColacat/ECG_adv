"""Sanity #4: vanilla ECGTwin vs CenterToken-hooked, same prompts/refs.

The author-prompt sanity (#3) showed vanilla ECGTwin produces visually
reasonable ECGs but the super5 victim cannot identify CD/MI classes
(p_target ≈ 0.01 for LBBB/RBBB/AVB, ≈ 0.24 for MI/STEMI). The hypothesis
is that CenterToken hook pulls synth latents toward PTB-XL center style,
moving CD/MI out of the victim's OOD region.

This script does a side-by-side compare:
  - 5 super5 classes × 5 author-style prompts (same set)
  - 3 conditions: vanilla / +extra_k200_ckpt / +nin_k200_ckpt
  - per-cell p_target_super5 + top1_hit + Einthoven + HR + warnings

Output table: per-class avg p_target across 3 conditions = direct evidence
of whether CenterToken closes the OOD gap.

Run:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/sanity_super5_centertoken.py \
    --n_per_class 3 --steps 50 \
    --out_dir outputs/sanity_super5_centertoken
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
from methods.ecgtwin_gen.center_token.model import CenterToken  # noqa: E402

# Same author-style prompts as sanity #3
PROMPTS = [
    ("NORM",  "sinus rhythm|normal ecg."),
    ("MI",    "myocardial infarction|st elevation|anterior wall"),
    ("HYP",   "left ventricular hypertrophy|high voltage"),
    ("CD",    "left bundle branch block|lbbb"),
    ("STTC",  "nstemi|non st elevation|t wave inversion"),
]

SUPER5_CLASSES = ["NORM", "MI", "HYP", "CD", "STTC"]


def register_centertoken_hook(wrapper, center_token):
    """Hook every DiTBlock to add center_token offset to the c (modulation) path."""
    def make_hook():
        def hook_fn(module, args):
            x, c, c2 = args[0], args[1], args[2]
            rest = args[3:]
            c = c + center_token(c.size(0))
            return (x, c, c2) + rest
        return hook_fn
    handles = []
    for block in wrapper.noise_predictor.blocks:
        handles.append(block.register_forward_pre_hook(make_hook()))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def load_centertoken(ckpt_path: str, device: str) -> CenterToken:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ct = CenterToken(dim=256)
    ct.load_state_dict(ck["center_token"])
    ct.to(device)
    ct.eval()
    return ct


def run_condition(label, wrapper, victim, ref_latent, ref_label, args, out_root):
    """Generate n_per_class samples for each (cls, prompt). label is 'vanilla' / 'extra' / 'nin'."""
    out = out_root / label
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for cell_i, (cls, prompt) in enumerate(PROMPTS):
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
                target_text=prompt,
            )
            x_init = torch.randn(1, 4, 128, device=wrapper.device)
            latent = wrapper.ddpm_sample(cond, batch_size=1,
                                         num_inference_steps=args.steps, x_init=x_init)
            ecg_tc = wrapper.decode_latent(latent)

            with torch.no_grad():
                probs = victim.forward_from_latent(latent, enable_grad=False)[0].cpu().numpy()
            probs_dict = {SUPER5_CLASSES[i]: float(probs[i]) for i in range(5)}
            top1 = SUPER5_CLASSES[int(np.argmax(probs))]

            sig = ecg_tc[0].detach().cpu().numpy()
            png_path = cell_dir / f"seed{seed}.png"
            title = f"[{label}] {cls}: {prompt[:50]} | top1={top1} (p={float(probs.max()):.2f})"
            _, report = plot_with_report(
                sig, sample_rate=102.4, save_path=png_path,
                title_prefix=title, lead_order="ecgtwin", engine="matplotlib",
            )

            entry = {
                "condition": label,
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
            print(f"  [{label:<8}{cls:<5} '{prompt[:40]:<40}' seed={seed}]  "
                  f"top1={top1:<5} p_super5={probs_dict[cls]:.3f}  "
                  f"{hr_str}  einth={report['einthoven_residual']:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_pt",
                    default="/root/ECG_adv_Gen/model/ECGTwin/data/prepared_input/normal_1.pt")
    ap.add_argument("--victim_ckpt",
                    default="/root/autodl-tmp/triple_labels/super5/best_model.pt")
    ap.add_argument("--ckpt_extra",
                    default="/root/autodl-tmp/center_token_super5/extra_k200_ckpt/center_token_best.pth")
    ap.add_argument("--ckpt_nin",
                    default="/root/autodl-tmp/center_token_super5/nin_k200_ckpt/center_token_best.pth")
    ap.add_argument("--n_per_class", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="outputs/sanity_super5_centertoken")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base_seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[setup] loading ECGTwin (with text model for on-the-fly nomic encode)...")
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=True)

    print(f"[setup] ref: {args.ref_pt}")
    ref = torch.load(args.ref_pt, map_location="cpu", weights_only=False)
    ref_latent = ref["data"]
    ref_label = dict(ref["label"])
    print(f"    ref hr={ref_label.get('hr')}, age={ref_label.get('age')}, sex={ref_label.get('sex')}")

    print(f"[setup] Super5 victim: {args.victim_ckpt}")
    victim = EfficientNetVictimTierM(
        weight_path=args.victim_ckpt, device=args.device,
        ecgtwin_wrapper=wrapper, num_classes=5, crop_len=250,
    )

    all_results = []

    # Condition A: vanilla
    print("\n" + "=" * 80)
    print("[CONDITION A] VANILLA (no CenterToken)")
    print("=" * 80)
    res_vanilla = run_condition("vanilla", wrapper, victim, ref_latent, ref_label, args, out)
    all_results.extend(res_vanilla)

    # Condition B: + CenterToken extra_k200
    print("\n" + "=" * 80)
    print(f"[CONDITION B] +CenterToken extra_k200 ({args.ckpt_extra})")
    print("=" * 80)
    ct_extra = load_centertoken(args.ckpt_extra, args.device)
    print(f"    center_token norm = {float(ct_extra.norm):.4f}")
    handles = register_centertoken_hook(wrapper, ct_extra)
    try:
        res_extra = run_condition("hook_extra", wrapper, victim, ref_latent, ref_label, args, out)
    finally:
        remove_hooks(handles)
    all_results.extend(res_extra)

    # Condition C: + CenterToken nin_k200
    print("\n" + "=" * 80)
    print(f"[CONDITION C] +CenterToken nin_k200 ({args.ckpt_nin})")
    print("=" * 80)
    ct_nin = load_centertoken(args.ckpt_nin, args.device)
    print(f"    center_token norm = {float(ct_nin.norm):.4f}")
    handles = register_centertoken_hook(wrapper, ct_nin)
    try:
        res_nin = run_condition("hook_nin", wrapper, victim, ref_latent, ref_label, args, out)
    finally:
        remove_hooks(handles)
    all_results.extend(res_nin)

    # ── Aggregate ────────────────────────────────────────────────────────────
    summary = {"per_cell": {}, "results": all_results}
    for r in all_results:
        key = f"{r['condition']}|{r['super5_target']}|{r['prompt']}"
        summary["per_cell"].setdefault(key, []).append(r)

    cell_summary = {}
    for key, runs in summary["per_cell"].items():
        cls = runs[0]["super5_target"]
        cell_summary[key] = {
            "condition": runs[0]["condition"],
            "super5_target": cls,
            "prompt": runs[0]["prompt"],
            "n_samples": len(runs),
            "p_target_avg": float(np.mean([r["p_target_super5"] for r in runs])),
            "p_target_max": float(np.max([r["p_target_super5"] for r in runs])),
            "top1_hit_rate": sum(r["top1"] == cls for r in runs) / len(runs),
            "einthoven_pass_p20": sum(r["einthoven_residual"] < 0.2 for r in runs) / len(runs),
        }
    summary["per_cell"] = cell_summary

    json_path = out / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Pivot table: rows = (cls, prompt), cols = condition
    print()
    print("=" * 100)
    print("Side-by-side avg p_target_super5 (top1_hit_rate in parens):")
    print("=" * 100)
    print(f"{'super5':<6} {'prompt':<55} {'vanilla':>14} {'+extra_k200':>14} {'+nin_k200':>14}")
    print("-" * 100)
    for cls, prompt in PROMPTS:
        row = []
        for cond in ["vanilla", "hook_extra", "hook_nin"]:
            key = f"{cond}|{cls}|{prompt}"
            if key in cell_summary:
                c = cell_summary[key]
                row.append(f"{c['p_target_avg']:.3f}({c['top1_hit_rate']:.0%})")
            else:
                row.append("--")
        print(f"{cls:<6} {prompt[:53]:<55} {row[0]:>14} {row[1]:>14} {row[2]:>14}")
    print()
    print(f"[done] PNGs + summary.json in {out}")


if __name__ == "__main__":
    main()
