#!/usr/bin/env python
"""
Step 7: 使用生成的困难样本微调 EfficientNet Adapter

前置条件：
  - Step 5: outputs/hard_samples/all_hard_samples.pt 已生成

预计时间：~1 小时

用法:
    conda run -n ECGTwin python scripts/run_finetune.py
    conda run -n ECGTwin python scripts/run_finetune.py --no_generated  # 只用 PTBXL 真实数据
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adversarial.efficientnet_adapter import EfficientNetAdapter
from adversarial.finetune import (
    load_ptbxl_for_finetune,
    load_generated_samples,
    train_adapter,
    DEFAULT_TRAIN_CONFIG,
)

PTBXL_ROOT = "/root/ECG_adv_Gen/datasets/PTBXL"
GENERATED_PATH = "/root/ECG_adv_Gen/outputs/hard_samples/all_hard_samples.pt"
EFFICIENTNET_WEIGHT = (
    "/root/ECG_adv_Gen/model/DeepECG/weights"
    "/efficientnetv2_77_classes/efficientnet_deepecg_unscaled.pt"
)
ADAPTER_SAVE_PATH = "/root/ECG_adv_Gen/outputs/adapter_best.pt"
HISTORY_SAVE_PATH = "/root/ECG_adv_Gen/outputs/finetune_history.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune EfficientNet Adapter")
    parser.add_argument("--ptbxl_root", default=PTBXL_ROOT)
    parser.add_argument("--generated_path", default=GENERATED_PATH)
    parser.add_argument("--weight_path", default=EFFICIENTNET_WEIGHT)
    parser.add_argument("--adapter_save", default=ADAPTER_SAVE_PATH)
    parser.add_argument("--history_save", default=HISTORY_SAVE_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAIN_CONFIG["lr"])
    parser.add_argument("--generated_weight", type=float,
                        default=DEFAULT_TRAIN_CONFIG["generated_weight"])
    parser.add_argument("--no_generated", action="store_true",
                        help="Train without generated hard samples (ablation)")
    parser.add_argument(
        "--n_per_superdiag", type=int, default=None,
        help="Few-shot: max PTBXL samples per superdiagnostic class for training. "
             "None = all data. E.g. --n_per_superdiag 50 simulates real-hospital scenario.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("EfficientNet Adapter Fine-tuning")
    print("=" * 60)
    print(f"PTBXL root:    {args.ptbxl_root}")
    print(f"Device:        {args.device}")

    # 创建输出目录
    Path(args.adapter_save).parent.mkdir(parents=True, exist_ok=True)

    # 加载 PTBXL 训练/验证集（支持少样本模式）
    if args.n_per_superdiag is not None:
        print(f"Few-shot mode: max {args.n_per_superdiag} samples per superdiagnostic class")
    train_ds, val_ds = load_ptbxl_for_finetune(
        ptbxl_data_root=args.ptbxl_root,
        val_fold=DEFAULT_TRAIN_CONFIG["val_fold"],
        n_per_superdiag=args.n_per_superdiag,
    )

    # 加载生成的困难样本（可选）
    generated_ds = None
    if not args.no_generated and Path(args.generated_path).exists():
        generated_ds = load_generated_samples(args.generated_path)
    elif not args.no_generated:
        print(f"[WARN] Generated samples not found at {args.generated_path}")
        print("       Training with PTBXL real data only.")

    # 创建 Adapter
    adapter = EfficientNetAdapter(
        weight_path=args.weight_path,
        device=args.device,
        hidden_dim=DEFAULT_TRAIN_CONFIG["hidden_dim"],
        dropout=DEFAULT_TRAIN_CONFIG["dropout"],
    )
    print(f"Adapter parameters: {adapter.count_adapter_params():,}")

    # 训练配置
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "generated_weight": args.generated_weight,
    }

    # 训练
    history = train_adapter(
        adapter=adapter,
        train_ds=train_ds,
        val_ds=val_ds,
        generated_ds=generated_ds,
        config=config,
        save_path=args.adapter_save,
        device=args.device,
    )

    # 保存训练历史
    import torch
    torch.save(history, args.history_save)
    print(f"Training history saved to {args.history_save}")

    # 打印最终验证 loss
    best_val = min(history["val_loss"])
    print(f"\nBest val_loss: {best_val:.4f}")
    print(f"Adapter saved to: {args.adapter_save}")
    print("\nNext step: python scripts/run_evaluate_finetuned.py")


if __name__ == "__main__":
    main()
