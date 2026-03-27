#!/usr/bin/env python
"""
MI 困难样本实验：使用 MIMIC-IV 风格提示词合成心梗困难样本并微调 DeepECG

流程：
  1. 从 PTBXL MI 类别随机选 500 条样本作参考
  2. 用 ECGTwin + AdvDiff 边界引导生成 ~2000 条心梗困难样本
     - 接受条件：target pattern 置信度在 (0.45, 0.65)，即"近边界但仍正确分类"
     - 提示词风格：MIMIC-IV ECG 机器报告格式（小写 + | 分隔）
  3. 微调 EfficientNet Adapter（500 真实 MI + 2000 合成困难样本）
  4. AUPRC 评估对比（baseline vs finetuned）

用法：
    conda run -n ECGTwin python scripts/run_mi_experiment.py
    conda run -n ECGTwin python scripts/run_mi_experiment.py --device cuda --batch_size 4
"""

import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adversarial.adv_generate import run_generation
from adversarial.efficientnet_adapter import EfficientNetAdapter
from adversarial.finetune import load_ptbxl_for_finetune, load_generated_samples, train_adapter, DEFAULT_TRAIN_CONFIG
from adversarial.evaluate import evaluate as run_evaluation

# ——— 路径配置 ———
PTBXL_ENCODED_PATH = "/root/ECG_adv_Gen/datasets/PTBXL/PTBXL_vae_multi_nomic.pt"
PTBXL_ROOT = "/root/ECG_adv_Gen/datasets/PTBXL"
OUTPUT_DIR = "/root/ECG_adv_Gen/outputs/mi_experiment"
EFFICIENTNET_WEIGHT = (
    "/root/ECG_adv_Gen/model/DeepECG/weights"
    "/efficientnetv2_77_classes/efficientnet_deepecg_unscaled.pt"
)

# ——— MIMIC-IV 风格 MI 提示词 ———
# ECGTwin 训练数据来自 MIMIC-IV ECG 报告，文本格式：小写 + | 分隔多诊断
# prompt_process() 将 "a|b|c" 转为：
#   "Most importantly, the 1st diagnosis is {a}. As a supplementary condition, ..."
#
# 目标 pattern 选取原则：EfficientNet 在 MI 样本上的实测置信度必须 > 0.3
#   Acute MI    (69):  0.622  ← EfficientNet 对 MI 最敏感
#   Q wave inf  (14):  0.549  ← 下壁 Q 波
#   ST elev inf (36):  0.476  ← 下壁 ST 抬高
#   inferolateral 混合 (14,36,49,41): 0.287（边缘可用）
#
# 排除的 pattern（EfficientNet 实测 prob < 0.1，无判别能力）：
#   Q wave septal (60): 0.003, Q wave anterior (61): 0.018
#   ST elev septal (25): 0.008, ST elev anterior (24): 0.049
MI_GENERATION_PLAN = [
    {
        # 下壁 MI：已验证有效（500 个样本已生成）
        "prompt_key": "MIMIC_MI_inferior",
        "text_prompt": "inferior myocardial infarction|st elevation|abnormal ecg",
        "superdiag": "MI",
        "target_count": 500,
    },
    {
        # 急性 MI（广泛前壁）：目标 Acute MI (69)，EfficientNet 置信度最高 (0.622)
        "prompt_key": "MIMIC_MI_acute",
        "text_prompt": "consider acute st elevation mi|inferior st elevation, consider acute infarct|abnormal ecg",
        "superdiag": "MI",
        "target_count": 600,
    },
    {
        # 急性下壁 MI 变体：同时引导 Acute MI + Q wave inf + ST elev inf
        "prompt_key": "MIMIC_MI_acute_inferior",
        "text_prompt": "acute myocardial infarction|inferior myocardial infarction|st elevation|abnormal ecg",
        "superdiag": "MI",
        "target_count": 500,
    },
    {
        # 下侧壁 MI：inferolateral，目标混合置信度 0.287（边缘可用）
        "prompt_key": "MIMIC_MI_inferolateral",
        "text_prompt": "inferolateral myocardial infarction|abnormal ecg",
        "superdiag": "MI",
        "target_count": 400,
    },
]

# ——— 超参数 ———
# Latent-relative step sizing: guidance_scale 表示每步扰动占 latent L2 norm 的比例
# 0.03 × 10 guidance steps ≈ 30% total perturbation
# acceptance_range = (0.4, 0.65)：近边界且仍正确分类 (>0.5)
GENERATION_HYPERPARAMS = {
    "num_inference_steps": 50,
    "adversarial_guidance_scale": 0.03,   # 3% of latent norm per step
    "noise_sampling_guidance_scale": 0.01, # 1% of x_T norm
    "num_noise_sampling_steps": 2,
    "guidance_start_fraction": 0.8,     # 只在最后 20% 步引导
    "acceptance_range": [0.4, 0.65],    # 略宽：0.4-0.5 为近边界但低置信，0.5-0.65 为正确分类
    "batch_size": 4,
}


def prepare_500_mi_reference(ptbxl_encoded_path: str, seed: int = 42) -> str:
    """
    从 PTBXL 编码数据中随机选取 500 条 MI 样本，保存为新的 .pt 文件

    Returns:
        path: 保存的 500-MI 样本路径
    """
    save_path = Path(ptbxl_encoded_path).parent / "PTBXL_vae_500MI.pt"
    if save_path.exists():
        print(f"[SKIP] 500-MI reference already at {save_path}")
        return str(save_path)

    print(f"Loading full PTBXL encoded data...")
    all_items = torch.load(ptbxl_encoded_path, map_location="cpu")
    mi_items = [x for x in all_items if x["label"].get("diagnostic_class") == "MI"]
    print(f"Total MI samples: {len(mi_items)}")

    torch.manual_seed(seed)
    perm = torch.randperm(len(mi_items))[:500].tolist()
    selected = [mi_items[i] for i in perm]

    torch.save(selected, str(save_path))
    print(f"Saved 500 random MI samples → {save_path}")
    return str(save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="MI Hard Samples Experiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, use existing outputs")
    parser.add_argument("--skip_finetune", action="store_true",
                        help="Skip finetune, only run evaluation")
    parser.add_argument("--n_per_superdiag", type=int, default=50,
                        help="Few-shot PTBXL samples per superdiagnostic class (default: 50)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MI Hard Samples Experiment")
    print("=" * 60)
    print(f"Device:     {args.device}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Few-shot:   {args.n_per_superdiag} samples/superdiag class")
    print(f"Epochs:     {args.epochs}")

    # ─── Step 1: 准备 500 MI 参考样本 ───────────────────────────────
    mi_ref_path = prepare_500_mi_reference(PTBXL_ENCODED_PATH, seed=args.seed)

    # ─── Step 2: 生成 2000 MI 困难样本 ───────────────────────────────
    generated_path = output_dir / "all_hard_samples.pt"
    hp = {**GENERATION_HYPERPARAMS, "batch_size": args.batch_size}

    if not args.skip_generation:
        print(f"\n{'='*60}")
        print("Step 2: Generating MI hard samples")
        print(f"  Prompts: {[p['prompt_key'] for p in MI_GENERATION_PLAN]}")
        print(f"  Target: {sum(p['target_count'] for p in MI_GENERATION_PLAN)} samples total")
        print(f"  Acceptance range: {hp['acceptance_range']}")

        results = run_generation(
            output_dir=str(output_dir),
            ptbxl_encoded_path=mi_ref_path,
            device=args.device,
            hyperparams=hp,
            plan_override=MI_GENERATION_PLAN,
            resume=not args.no_resume,
        )
        total_generated = sum(r["ecg"].shape[0] for r in results if "ecg" in r)
        print(f"\nGeneration done. Total: {total_generated} hard samples")
    else:
        print(f"\n[SKIP] Generation (--skip_generation)")
        if not generated_path.exists():
            print(f"ERROR: {generated_path} not found. Run without --skip_generation first.")
            sys.exit(1)

    # ─── Step 3: 微调 Adapter ───────────────────────────────────────
    adapter_path = str(output_dir / "adapter_best.pt")
    history_path = str(output_dir / "finetune_history.pt")

    if not args.skip_finetune:
        print(f"\n{'='*60}")
        print("Step 3: Fine-tuning EfficientNet Adapter")
        print(f"  Few-shot: {args.n_per_superdiag} PTBXL samples/superdiag class")

        # 加载 PTBXL 少样本训练集（模拟真实医院部署场景）
        train_ds, val_ds = load_ptbxl_for_finetune(
            ptbxl_data_root=PTBXL_ROOT,
            val_fold=DEFAULT_TRAIN_CONFIG["val_fold"],
            n_per_superdiag=args.n_per_superdiag,
            seed=args.seed,
        )

        # 加载生成的困难样本
        generated_ds = None
        if generated_path.exists():
            generated_ds = load_generated_samples(str(generated_path))
        else:
            print(f"[WARN] No generated samples found at {generated_path}")

        # 创建 Adapter
        adapter = EfficientNetAdapter(
            weight_path=EFFICIENTNET_WEIGHT,
            device=args.device,
            hidden_dim=DEFAULT_TRAIN_CONFIG["hidden_dim"],
            dropout=DEFAULT_TRAIN_CONFIG["dropout"],
        )
        print(f"Adapter parameters: {adapter.count_adapter_params():,}")

        config = {
            "epochs": args.epochs,
            "batch_size": DEFAULT_TRAIN_CONFIG["batch_size"],
            "lr": DEFAULT_TRAIN_CONFIG["lr"],
            "generated_weight": 2.0,
        }

        history = train_adapter(
            adapter=adapter,
            train_ds=train_ds,
            val_ds=val_ds,
            generated_ds=generated_ds,
            config=config,
            save_path=adapter_path,
            device=args.device,
        )
        torch.save(history, history_path)
        print(f"Training history saved → {history_path}")
    else:
        print(f"\n[SKIP] Fine-tuning (--skip_finetune)")

    # ─── Step 4: AUPRC 评估 ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 4: AUPRC Evaluation (baseline vs finetuned)")

    eval_result_path = str(output_dir / "eval_results.json")
    results = run_evaluation(
        ptbxl_data_root=PTBXL_ROOT,
        adapter_path=adapter_path if Path(adapter_path).exists() else None,
        efficientnet_weight_path=EFFICIENTNET_WEIGHT,
        device=args.device,
        output_path=eval_result_path,
    )

    # ─── 打印关键指标 ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Results Summary")
    print("=" * 60)

    import json
    with open(eval_result_path) as f:
        eval_data = json.load(f)

    baseline_cat = eval_data.get("baseline", {}).get("per_category", {})
    finetuned_cat = eval_data.get("finetuned", {}).get("per_category", {})

    print(f"\n{'Category':<30} {'Baseline':>12} {'Finetuned':>12} {'Δ':>8}")
    print("-" * 65)
    for cat in sorted(set(list(baseline_cat.keys()) + list(finetuned_cat.keys()))):
        b_auprc = baseline_cat.get(cat, {}).get("auprc")
        f_auprc = finetuned_cat.get(cat, {}).get("auprc")
        if b_auprc is None and f_auprc is None:
            continue
        b_str = f"{b_auprc:.3f}" if b_auprc is not None else "  N/A"
        f_str = f"{f_auprc:.3f}" if f_auprc is not None else "  N/A"
        if b_auprc is not None and f_auprc is not None:
            delta = f_auprc - b_auprc
            d_str = f"{delta:+.3f}"
        else:
            d_str = "   N/A"
        print(f"{cat:<30} {b_str:>12} {f_str:>12} {d_str:>8}")

    print(f"\nFull results: {eval_result_path}")


if __name__ == "__main__":
    main()
