#!/usr/bin/env python
"""
Step 5: 使用 AdvDiff 边界引导生成困难样本

前置条件：
  - Step 0: scripts/run_prepare_ptbxl.py 已生成 PTBXL VAE 编码数据
  - Step 1: adversarial/label_mapping.py 已就绪
  - Step 2: adversarial/efficientnet_victim.py 已就绪

预计时间：2-4 小时（GPU）

用法:
    conda run -n ECGTwin python scripts/run_adv_generate.py
    conda run -n ECGTwin python scripts/run_adv_generate.py --prompt_key MI_inferior --device cuda
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adversarial.adv_generate import run_generation, GENERATION_PLAN, DEFAULT_HYPERPARAMS

PTBXL_ENCODED_PATH = "/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt"
OUTPUT_DIR = "/root/ECG_adv_Gen/outputs/hard_samples"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial hard samples via AdvDiff")
    parser.add_argument("--ptbxl_encoded", default=PTBXL_ENCODED_PATH,
                        help="Path to PTBXL VAE-encoded data (.pt)")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prompt_key", default=None,
        help="Generate only this prompt key (e.g. MI_inferior). Default: all",
    )
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--adversarial_guidance_scale", type=float, default=1.5)
    parser.add_argument("--noise_sampling_guidance_scale", type=float, default=0.3)
    parser.add_argument("--num_noise_sampling_steps", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_resume", action="store_true",
                        help="Do not resume from existing progress")
    parser.add_argument(
        "--max_per_prompt", type=int, default=None,
        help="Override target_count for every prompt (for quick experiments). "
             "E.g. --max_per_prompt 100 → 1000 total samples across 10 prompts.",
    )
    parser.add_argument(
        "--skip_prompts", nargs="*", default=[],
        help="Prompt keys to skip entirely. E.g. --skip_prompts ISCHEMIA_ant ISCHEMIA_inf",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("AdvDiff Boundary Guidance Hard Sample Generation")
    print("=" * 60)
    print(f"PTBXL encoded: {args.ptbxl_encoded}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Device:        {args.device}")

    # 超参数覆盖
    hp_override = {
        "num_inference_steps": args.num_inference_steps,
        "adversarial_guidance_scale": args.adversarial_guidance_scale,
        "noise_sampling_guidance_scale": args.noise_sampling_guidance_scale,
        "num_noise_sampling_steps": args.num_noise_sampling_steps,
        "batch_size": args.batch_size,
    }
    print(f"Hyperparams:   {hp_override}")

    # 可选：只生成特定 prompt
    plan = GENERATION_PLAN
    if args.prompt_key is not None:
        plan = [item for item in GENERATION_PLAN if item["prompt_key"] == args.prompt_key]
        if not plan:
            print(f"ERROR: Unknown prompt_key '{args.prompt_key}'")
            valid = [item["prompt_key"] for item in GENERATION_PLAN]
            print(f"Valid keys: {valid}")
            sys.exit(1)
        print(f"Generating only: {args.prompt_key}")

    # 覆盖每个 prompt 的 target_count（快速实验模式）
    if args.max_per_prompt is not None:
        plan = [dict(item, target_count=args.max_per_prompt) for item in plan]
        total = args.max_per_prompt * len(plan)
        print(f"max_per_prompt={args.max_per_prompt} → {total} total samples across {len(plan)} prompts")

    # 跳过指定 prompt
    if args.skip_prompts:
        plan = [item for item in plan if item["prompt_key"] not in args.skip_prompts]
        print(f"Skipping: {args.skip_prompts}")

    results = run_generation(
        output_dir=args.output_dir,
        ptbxl_encoded_path=args.ptbxl_encoded,
        device=args.device,
        hyperparams=hp_override,
        plan_override=plan,
        resume=not args.no_resume,
    )

    total = sum(r["ecg"].shape[0] for r in results if "ecg" in r)
    print(f"\nDone. Total generated: {total} hard samples")
    print(f"Merged file: {args.output_dir}/all_hard_samples.pt")


if __name__ == "__main__":
    main()
