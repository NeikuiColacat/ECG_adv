"""Train a single center-style token using CenterTokenTrainer.

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/train_center_token.py \
    --dataset /root/autodl-tmp/center_token/cpsc_2018_extra.pt \
    --save_dir /root/autodl-tmp/center_token/cpsc_2018_extra_ckpt
"""
import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from methods.ecgtwin_gen.center_token.trainer import CenterTokenTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Per-center .pt from prep_center_dataset.py")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--config", default=str(REPO / "methods/ecgtwin_gen/center_token/config.yaml"))
    ap.add_argument("--ecgtwin_config", default=str(REPO / "model/ECGTwin/config/DiT_ECGTwin.yaml"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.ecgtwin_config) as f:
        ecgtwin_config = yaml.safe_load(f)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        config["training"]["scheduler"]["T_max"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    trainer = CenterTokenTrainer(config=config, ecgtwin_config=ecgtwin_config, device=args.device)
    trainer.train(dataset_path=args.dataset, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
