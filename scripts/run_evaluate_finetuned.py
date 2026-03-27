#!/usr/bin/env python
"""
Step 8: 评估微调后的 EfficientNet Adapter vs Baseline

对比在 PTBXL test set (fold 10) 上的 AUPRC / AUROC。

前置条件：
  - Step 3: outputs/eval_baseline.json 存在（可选，用于验证）
  - Step 7: outputs/adapter_best.pt 存在

用法:
    conda run -n ECGTwin python scripts/run_evaluate_finetuned.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adversarial.evaluate import evaluate

PTBXL_ROOT = "/root/ECG_adv_Gen/datasets/PTBXL"
EFFICIENTNET_WEIGHT = (
    "/root/ECG_adv_Gen/model/DeepECG/weights"
    "/efficientnetv2_77_classes/efficientnet_deepecg_unscaled.pt"
)
ADAPTER_PATH = "/root/ECG_adv_Gen/outputs/adapter_best.pt"
OUTPUT_PATH = "/root/ECG_adv_Gen/outputs/eval_finetuned.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline vs finetuned adapter")
    parser.add_argument("--ptbxl_root", default=PTBXL_ROOT)
    parser.add_argument("--weight_path", default=EFFICIENTNET_WEIGHT)
    parser.add_argument("--adapter_path", default=ADAPTER_PATH)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.adapter_path).exists():
        print(f"ERROR: Adapter weights not found at {args.adapter_path}")
        print("Please run scripts/run_finetune.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Baseline vs Finetuned Adapter Comparison")
    print("=" * 60)
    print(f"PTBXL root:    {args.ptbxl_root}")
    print(f"Adapter:       {args.adapter_path}")
    print(f"Device:        {args.device}")

    results = evaluate(
        ptbxl_data_root=args.ptbxl_root,
        adapter_path=args.adapter_path,
        efficientnet_weight_path=args.weight_path,
        device=args.device,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )

    # 打印关键指标变化
    b = results["baseline"]["per_category"]
    f = results["finetuned"]["per_category"] if results["finetuned"] else {}

    print("\n" + "=" * 60)
    print("Key Results Summary")
    print("=" * 60)
    for cat in ["Infarction or ischemia", "Conduction Disorder",
                "Enlargement of the heart chambers", "Other diagnoses"]:
        b_auprc = b.get(cat, {}).get("macro_auprc", float("nan"))
        f_auprc = f.get(cat, {}).get("macro_auprc", float("nan"))

        import math
        if not math.isnan(f_auprc):
            delta = f_auprc - b_auprc
            sign = "+" if delta >= 0 else ""
            print(f"  {cat:<40} {b_auprc:.3f} → {f_auprc:.3f}  ({sign}{delta:.3f})")
        else:
            print(f"  {cat:<40} {b_auprc:.3f}")


if __name__ == "__main__":
    main()
