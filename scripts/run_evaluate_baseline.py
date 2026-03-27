#!/usr/bin/env python
"""
Step 3: 在 PTBXL test set (fold 10) 上评估 EfficientNet baseline

用法:
    conda run -n ECGTwin python scripts/run_evaluate_baseline.py
    conda run -n ECGTwin python scripts/run_evaluate_baseline.py --device cpu
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
OUTPUT_PATH = "/root/ECG_adv_Gen/outputs/eval_baseline.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet baseline on PTBXL")
    parser.add_argument("--ptbxl_root", default=PTBXL_ROOT)
    parser.add_argument("--weight_path", default=EFFICIENTNET_WEIGHT)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Baseline EfficientNet Evaluation on PTBXL Test Set")
    print("=" * 60)
    print(f"PTBXL root:    {args.ptbxl_root}")
    print(f"Weights:       {args.weight_path}")
    print(f"Device:        {args.device}")

    results = evaluate(
        ptbxl_data_root=args.ptbxl_root,
        adapter_path=None,
        efficientnet_weight_path=args.weight_path,
        device=args.device,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )

    # 打印关键指标
    b = results["baseline"]
    infarct = b["per_category"].get("Infarction or ischemia", {})
    print(f"\nKey result — Infarction or ischemia macro-AUPRC: {infarct.get('macro_auprc', 'N/A'):.3f}")


if __name__ == "__main__":
    main()
