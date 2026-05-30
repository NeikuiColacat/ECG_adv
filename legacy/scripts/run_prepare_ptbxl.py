"""
入口脚本: 准备 PTBXL 数据集用于 center token 训练

Usage:
    python scripts/run_prepare_ptbxl.py
    python scripts/run_prepare_ptbxl.py --device cuda:1 --vae_batch_size 32
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.prepare_ptbxl_for_ecgtwin import prepare_ptbxl
import argparse


def main():
    parser = argparse.ArgumentParser(description="Prepare PTBXL for center token training")
    parser.add_argument("--ptbxl_path", type=str, default="datasets/PTBXL")
    parser.add_argument("--output_path", type=str, default="datasets/PTBXL/PTBXL_vae_multi_nomic.pt")
    parser.add_argument("--vae_path", type=str, default="model/ECGTwin/checkpoints/vae_model.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scp_threshold", type=float, default=50.0)
    parser.add_argument("--vae_batch_size", type=int, default=64)
    args = parser.parse_args()

    prepare_ptbxl(
        ptbxl_path=args.ptbxl_path,
        output_path=args.output_path,
        vae_checkpoint_path=args.vae_path,
        device=args.device,
        scp_threshold=args.scp_threshold,
        vae_batch_size=args.vae_batch_size,
    )


if __name__ == "__main__":
    main()
