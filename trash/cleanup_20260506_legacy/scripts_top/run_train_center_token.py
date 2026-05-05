"""
入口脚本: 训练 center token

Usage:
    python scripts/run_train_center_token.py
    python scripts/run_train_center_token.py --device cuda:1 --epochs 50
"""

import sys
import argparse
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from center_token.trainer import CenterTokenTrainer


def main():
    parser = argparse.ArgumentParser(description="Train center token")
    parser.add_argument(
        "--config",
        type=str,
        default="center_token/config.yaml",
        help="Center token config path",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="checkpoints/center_token")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(config["ecgtwin"]["config_path"], "r") as f:
        ecgtwin_config = yaml.safe_load(f)

    # 命令行覆盖
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["training"]["scheduler"]["T_max"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    # 创建 trainer
    trainer = CenterTokenTrainer(
        config=config,
        ecgtwin_config=ecgtwin_config,
        device=args.device,
    )

    # 训练
    dataset_path = config["data"]["prepared_data_path"]
    trainer.train(dataset_path=dataset_path, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
