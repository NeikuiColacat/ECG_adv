#!/usr/bin/env python
"""
准备困难样本合并文件：
1. 生成 MIMIC_MI_inferolateral（如果还没有）
2. 加载全部 4 个生成文件，按 target pattern mean-prob >= 0.5 过滤
3. 构建 77-dim 标签，保存 all_hard_samples.pt

用法：
    python scripts/prepare_hard_samples.py
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adversarial.adv_generate import run_generation, _make_labels_77
from adversarial.label_mapping import PROMPT_TO_PATTERN_INDICES

OUTPUT_DIR = Path("/root/ECG_adv_Gen/outputs/mi_experiment")
PTBXL_500MI = "/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_500MI.pt"

INFEROLATERAL_PLAN = [
    {
        "prompt_key": "MIMIC_MI_inferolateral",
        "text_prompt": "inferolateral myocardial infarction|abnormal ecg",
        "superdiag": "MI",
        "target_count": 400,
    },
]

INFEROLATERAL_HP = {
    "num_inference_steps": 50,
    "adversarial_guidance_scale": 0.03,   # 3% of latent norm per step
    "noise_sampling_guidance_scale": 0.01, # 1% of x_T norm
    "num_noise_sampling_steps": 2,
    "guidance_start_fraction": 0.8,
    "acceptance_range": [0.5, 0.65],   # only accept correctly-classified samples
    "batch_size": 4,
}

# All 4 plan items with their target indices for filtering
PLAN_FILES = [
    ("MIMIC_MI_inferior",       [14, 36]),
    ("MIMIC_MI_acute",          [69]),
    ("MIMIC_MI_acute_inferior", [69, 14, 36]),
    ("MIMIC_MI_inferolateral",  [14, 36]),
]


def generate_inferolateral(device: str = "cuda"):
    save_path = OUTPUT_DIR / "MIMIC_MI_inferolateral.pt"
    if save_path.exists():
        data = torch.load(save_path, map_location="cpu")
        n = data["ecg"].shape[0]
        if n >= 400:
            print(f"[SKIP] MIMIC_MI_inferolateral already has {n} samples")
            return
        else:
            print(f"[RESUME] MIMIC_MI_inferolateral has {n}/400, generating more")
    else:
        print("Generating MIMIC_MI_inferolateral...")

    run_generation(
        output_dir=str(OUTPUT_DIR),
        ptbxl_encoded_path=PTBXL_500MI,
        device=device,
        hyperparams=INFEROLATERAL_HP,
        plan_override=INFEROLATERAL_PLAN,
        resume=True,
    )


def post_filter_and_merge(device: str = "cuda"):
    """Filter each dataset to prob >= 0.5, create 77-dim labels, merge."""
    from adversarial.efficientnet_victim import EfficientNetVictim

    print("\nLoading EfficientNet for re-scoring...")
    victim = EfficientNetVictim(device=device)

    all_ecg = []
    all_probs = []
    all_labels = []
    all_prompt_keys = []

    for prompt_key, target_idx in PLAN_FILES:
        path = OUTPUT_DIR / f"{prompt_key}.pt"
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue

        data = torch.load(path, map_location="cpu")
        ecg = data["ecg"]       # (N, 12, 2500)
        saved_probs = data["probs"]  # (N, 77) — saved at generation time

        # Re-score on GPU for accuracy (acceptance filter used saved probs at generation)
        ecg_gpu = ecg.to(device)
        with torch.no_grad():
            probs = victim.forward_from_ecg(ecg_gpu).cpu()

        target_probs = probs[:, target_idx].mean(dim=1)  # (N,)
        mask = target_probs >= 0.5

        n_total = ecg.shape[0]
        n_kept = mask.sum().item()
        print(f"\n{prompt_key}: {n_total} → {n_kept} samples (target_idx={target_idx}, "
              f"mean_prob: {target_probs.mean():.3f}, kept {100*n_kept/n_total:.1f}%)")

        ecg_filtered = ecg[mask]
        probs_filtered = probs[mask]
        labels = _make_labels_77(probs_filtered, prompt_key)

        all_ecg.append(ecg_filtered)
        all_probs.append(probs_filtered)
        all_labels.append(labels)
        all_prompt_keys.extend([prompt_key] * n_kept)

    if not all_ecg:
        print("ERROR: No samples collected!")
        return

    merged = {
        "ecg":         torch.cat(all_ecg,    dim=0),  # (N, 12, 2500)
        "probs":       torch.cat(all_probs,  dim=0),  # (N, 77)
        "labels_77":   torch.cat(all_labels, dim=0),  # (N, 77)
        "prompt_keys": all_prompt_keys,
    }
    n_total = merged["ecg"].shape[0]
    merge_path = OUTPUT_DIR / "all_hard_samples.pt"
    torch.save(merged, merge_path)
    print(f"\nSaved {n_total} filtered hard samples → {merge_path}")

    # Summary
    for prompt_key, target_idx in PLAN_FILES:
        mask = [pk == prompt_key for pk in merged["prompt_keys"]]
        n = sum(mask)
        if n == 0:
            continue
        idx = [i for i, m in enumerate(mask) if m]
        mean_prob = merged["probs"][idx][:, target_idx].mean().item()
        label_rate = merged["labels_77"][idx][:, target_idx].mean().item()
        print(f"  {prompt_key}: {n} samples, mean_target_prob={mean_prob:.3f}, "
              f"label_pos_rate={label_rate:.3f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_generation", action="store_true")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_generation:
        generate_inferolateral(args.device)

    post_filter_and_merge(args.device)
    print("\nDone. Run finetune with:")
    print("  python scripts/run_mi_experiment.py --skip_generation --n_per_superdiag 50 --epochs 30")
