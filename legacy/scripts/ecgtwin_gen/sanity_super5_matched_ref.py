"""Sanity check #2: matched-class ref + matched-class prompt.

ECGTwin's intended use is *conditional generation*: a per-class ref provides
patient/morphology base_vector, and the prompt provides class control on
top. The fixed-normal-ref experiment (sanity_super5_generation.py) revealed
that a NORM base_vector locks the victim verdict at CD across all 5 prompts.

This script tests whether ECGTwin can produce 5 distinguishable super5
classes when ref *and* prompt are class-matched — the natural use mode.

For each class, we:
  (1) Pick the first ref of that primary class from extra_k200.pt
  (2) Use the ref's own SNOMED-keyed text_embed (lookup from
      super5_text_embeds.pt) — this is exactly what generate_center_synth.py
      does in production
  (3) Generate `--n_per_class` samples with different DDPM seeds
  (4) Save 12-lead viz + sanity_check + super5 victim probs

Run:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/sanity_super5_matched_ref.py \
    --n_per_class 2 --steps 50 \
    --out_dir outputs/sanity_super5_matched
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

SUPER5_CLASSES = ["NORM", "MI", "HYP", "CD", "STTC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle",
                    default="/root/autodl-tmp/center_token_super5/super5_text_embeds.pt")
    ap.add_argument("--ref_pool",
                    default="/root/autodl-tmp/center_token_super5/extra_k200.pt")
    ap.add_argument("--victim_ckpt",
                    default="/root/autodl-tmp/triple_labels/super5/best_model.pt")
    ap.add_argument("--n_per_class", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="outputs/sanity_super5_matched")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base_seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load bundle (precomputed nomic embeddings) ────────────────────────────
    print(f"[setup] bundle: {args.bundle}")
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    by_class = bundle["by_class"]
    prompts = bundle["prompts_by_class"]
    print("    fallback prompt per class:")
    for c in SUPER5_CLASSES:
        print(f"      {c:<6} '{prompts[c]}'")

    # ── Load ECGTwin (no encoder, no nomic) ──────────────────────────────────
    print(f"[setup] ECGTwin wrapper (decoder only)")
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)

    # ── Pick one ref per super5 primary class from extra_k200 pool ────────────
    print(f"[setup] ref pool: {args.ref_pool}")
    pool = torch.load(args.ref_pool, map_location="cpu", weights_only=False)
    chosen_ref = {}
    for cls in SUPER5_CLASSES:
        for s in pool:
            if s["label"].get("diagnostic_class") == cls:
                chosen_ref[cls] = s
                break
        if cls not in chosen_ref:
            print(f"    [warn] no ref of class {cls} in pool — skipping that cell")
    print(f"    chosen ref record_ids:")
    for cls in SUPER5_CLASSES:
        if cls in chosen_ref:
            ref = chosen_ref[cls]
            print(f"      {cls:<5} record_id={ref['label']['record_id']:<10} "
                  f"hr={ref['label']['hr']:.1f}, age={ref['label']['age']}, "
                  f"sex={ref['label']['sex']}, "
                  f"text_embed shape={tuple(ref['label']['text_embed'].shape)}")

    # ── Load Super5 victim ────────────────────────────────────────────────────
    print(f"[setup] Super5 victim: {args.victim_ckpt}")
    victim = EfficientNetVictimTierM(
        weight_path=args.victim_ckpt,
        device=args.device,
        ecgtwin_wrapper=wrapper,
        num_classes=5,
        crop_len=250,
    )

    # ── Loop: per class, use its own ref + per-class fallback prompt ─────────
    results = []
    for cls_i, cls in enumerate(SUPER5_CLASSES):
        if cls not in chosen_ref:
            continue
        ref = chosen_ref[cls]
        ref_latent = ref["data"]                          # (4, 128)
        ref_label = dict(ref["label"])                    # already has hr/age/sex/text_embed
        # The ref's own SNOMED-keyed text_embed acts as the IBE base_vector text;
        # the *target* text_embed comes from the per-class fallback prompt (bundle).
        target_emb = by_class[cls]
        prompt = prompts[cls]
        (out / cls).mkdir(exist_ok=True)

        for k in range(args.n_per_class):
            seed = args.base_seed + cls_i * 1000 + k
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            cond = wrapper.prepare_conditions(
                ref_latent=ref_latent,
                ref_label=ref_label,
                batch_size=1,
                target_text_embed=target_emb,
            )
            x_init = torch.randn(1, 4, 128, device=wrapper.device)
            latent = wrapper.ddpm_sample(
                cond, batch_size=1, num_inference_steps=args.steps, x_init=x_init,
            )
            ecg_tc = wrapper.decode_latent(latent)        # (1, 1024, 12)

            with torch.no_grad():
                probs = victim.forward_from_latent(latent, enable_grad=False)[0].cpu().numpy()
            probs_dict = {SUPER5_CLASSES[i]: float(probs[i]) for i in range(5)}
            top1 = SUPER5_CLASSES[int(np.argmax(probs))]

            sig = ecg_tc[0].detach().cpu().numpy()
            png_path = out / cls / f"{cls}_seed{seed}.png"
            title = f"target={cls} (matched ref) | top1={top1} (p={float(probs.max()):.2f})"
            _, report = plot_with_report(
                sig, sample_rate=102.4, save_path=png_path,
                title_prefix=title, lead_order="ecgtwin", engine="matplotlib",
            )

            entry = {
                "target_class": cls,
                "ref_record_id": ref["label"]["record_id"],
                "ref_hr": float(ref["label"]["hr"]),
                "prompt": prompt,
                "seed": seed,
                "png": str(png_path),
                "top1": top1,
                "top1_correct": top1 == cls,
                "victim_probs": probs_dict,
                "hr_bpm": report["hr_estimate_bpm"],
                "einthoven_residual": report["einthoven_residual"],
                "avR_residual": report["avR_residual"],
                "p2p_II_mV": report["amplitude_p2p_per_lead"].get("II"),
                "warnings": report["warnings"],
            }
            results.append(entry)
            hr_str = f"HR={report['hr_estimate_bpm']:.0f}bpm" if report['hr_estimate_bpm'] else "HR=N/A"
            print(f"  [{cls:<5} ref={ref['label']['record_id']:<10} seed={seed}] "
                  f"top1={top1:<5} ({'OK ' if entry['top1_correct'] else 'MISS'}) "
                  f"p_target={probs_dict[cls]:.3f}  {hr_str}  "
                  f"einth={report['einthoven_residual']:.3f}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    summary = {
        "config": {
            "n_per_class": args.n_per_class,
            "steps": args.steps,
            "ref_pool": args.ref_pool,
            "bundle": args.bundle,
            "victim_ckpt": args.victim_ckpt,
            "centertoken_loaded": False,
            "base_vector_source": "matched-class ref from extra_k200 (one ref per class)",
        },
        "per_class_top1_acc": {},
        "per_class_avg_target_prob": {},
        "global": {
            "frac_einthoven_pass_p20": sum(
                r["einthoven_residual"] < 0.2 for r in results
            ) / len(results),
            "frac_einthoven_pass_p50": sum(
                r["einthoven_residual"] < 0.5 for r in results
            ) / len(results),
            "frac_hr_in_40_180": sum(
                40 <= (r["hr_bpm"] or 0) <= 180 for r in results
            ) / len(results),
            "n_warnings_total": sum(len(r["warnings"]) for r in results),
            "n_samples": len(results),
        },
        "results": results,
    }
    for cls in SUPER5_CLASSES:
        cls_runs = [r for r in results if r["target_class"] == cls]
        if not cls_runs:
            summary["per_class_top1_acc"][cls] = None
            summary["per_class_avg_target_prob"][cls] = None
            continue
        summary["per_class_top1_acc"][cls] = sum(
            r["top1_correct"] for r in cls_runs
        ) / len(cls_runs)
        summary["per_class_avg_target_prob"][cls] = float(np.mean(
            [r["victim_probs"][cls] for r in cls_runs]
        ))

    json_path = out / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print(f"[done] PNGs + summary.json in {out}")
    print(f"[done] per_class top-1 accuracy: {summary['per_class_top1_acc']}")
    print(f"[done] per_class avg p(target):  {summary['per_class_avg_target_prob']}")
    print(f"[done] Einthoven < 0.2 (clean):  "
          f"{summary['global']['frac_einthoven_pass_p20']:.2f}")
    print(f"[done] HR ∈ [40, 180] bpm:        "
          f"{summary['global']['frac_hr_in_40_180']:.2f}")


if __name__ == "__main__":
    main()
