"""Sanity #4: regenerate the super5 author-prompt suite, save raw tensors.

Same author prompts, base seed, and seed offsets as
`sanity_super5_authorprompt.py` — but we save the (12, 1024) signal tensor
in MIMIC lead order to .npz (no PNG generation). This is the input for
the digital validator (`digital_gt_validate.py`).

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/sanity_super5_dump_tensors.py \
    --n_per_class 3 --steps 50 \
    --out_dir outputs/sanity_super5_tensors
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
from adversarial.efficientnet_victim_tierM import EfficientNetVictimTierM  # noqa: E402

# Author prompts identical to sanity_super5_authorprompt.py (do not edit)
AUTHOR_PROMPTS = [
    ("NORM",  "sinus rhythm|normal ecg."),
    ("MI",    "myocardial infarction|st elevation|anterior wall"),
    ("MI",    "stemi|st elevation myocardial infarction|acute"),
    ("STTC",  "nstemi|non st elevation|t wave inversion"),
    ("STTC",  "acute pericarditis|diffuse st elevation"),
    ("HYP",   "left ventricular hypertrophy|high voltage"),
    ("CD",    "left bundle branch block|lbbb"),
    ("CD",    "right bundle branch block|rbbb"),
    ("CD",    "atrioventricular block|av block"),
    ("NORM",  "sinus bradycardia|slow heart rate"),
    ("NORM",  "sinus tachycardia|fast heart rate"),
    ("NORM",  "atrial fibrillation|irregular rhythm"),
]
SUPER5_CLASSES = ["NORM", "MI", "HYP", "CD", "STTC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_pt",
                    default="/root/ECG_adv_Gen/model/ECGTwin/data/prepared_input/normal_1.pt")
    ap.add_argument("--victim_ckpt",
                    default="/root/autodl-tmp/triple_labels/super5/best_model.pt")
    ap.add_argument("--n_per_class", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="outputs/sanity_super5_tensors")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base_seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[setup] loading ECGTwin (text model on, encoder off)...")
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=True)

    ref = torch.load(args.ref_pt, map_location="cpu", weights_only=False)
    ref_latent = ref["data"]
    ref_label = dict(ref["label"])

    print(f"[setup] Super5 victim: {args.victim_ckpt}")
    victim = EfficientNetVictimTierM(
        weight_path=args.victim_ckpt,
        device=args.device,
        ecgtwin_wrapper=wrapper,
        num_classes=5,
        crop_len=250,
    )

    records = []
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
                target_text=prompt,
            )
            x_init = torch.randn(1, 4, 128, device=wrapper.device)
            latent = wrapper.ddpm_sample(
                cond, batch_size=1, num_inference_steps=args.steps, x_init=x_init,
            )
            ecg_tc = wrapper.decode_latent(latent)  # (1, 1024, 12)

            with torch.no_grad():
                probs = victim.forward_from_latent(latent, enable_grad=False)[0].cpu().numpy()
            probs_dict = {SUPER5_CLASSES[i]: float(probs[i]) for i in range(5)}
            top1 = SUPER5_CLASSES[int(np.argmax(probs))]

            sig_ct = ecg_tc[0].detach().cpu().numpy().T  # (12, 1024) channels-first
            sig_ct = sig_ct.astype(np.float32)
            npz_path = cell_dir / f"seed{seed}.npz"
            np.savez_compressed(
                npz_path,
                signal_ct=sig_ct,
                prompt=prompt,
                seed=np.int64(seed),
                super5_target=cls,
                super5_probs=probs.astype(np.float32),
                super5_top1=top1,
            )

            entry = {
                "super5_target": cls,
                "prompt": prompt,
                "seed": seed,
                "npz": str(npz_path),
                "top1": top1,
                "p_target_super5": probs_dict[cls],
                "victim_probs": probs_dict,
                "signal_min_mv": float(sig_ct.min()),
                "signal_max_mv": float(sig_ct.max()),
                "signal_p2p_lead_ii_mv": float(sig_ct[1].max() - sig_ct[1].min()),
            }
            records.append(entry)
            print(f"  [{cls:<5} '{prompt[:38]:<38}' seed={seed}]  "
                  f"top1={top1:<5} p_target={probs_dict[cls]:.3f}  "
                  f"p2p(II)={entry['signal_p2p_lead_ii_mv']:.2f} mV")

    summary = {
        "config": {
            "n_per_class": args.n_per_class,
            "steps": args.steps,
            "ref_pt": args.ref_pt,
            "lead_order_in_npz": "mimic",
            "fs_hz": 102.4,
            "duration_s": 10.0,
            "n_samples_total": len(records),
        },
        "records": records,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print(f"[done] {len(records)} npz tensors + summary.json in {out}")


if __name__ == "__main__":
    main()
