"""Train CenterToken v2 (per-block, per-center INDEPENDENT, step-based).

Plan Rev 13: replaces v1 trainer's 5 bugs (detach, best.pth=epoch1, 1-batch/epoch,
L_inv variance, L_reg pull-to-0). See methods/ecgtwin_gen/center_token/trainer_v2.py
for full design. Key changes vs v1:
  - Per-block CT (CenterTokenPerBlock, 6 × 256-d, TokenVerse paradigm)
  - Step-based 5000 steps, not 30 epochs × 1 batch = 30 grad steps
  - Phase-separated LR (5e-3 → 5e-6 at 80% step mark)
  - L_recon (this center) + L_isolation (50% iters, vs base ECGTwin on PTBXL)
  - NO L_inv (broken), NO L_reg (replaced by sphere projection)
  - EMA-smoothed snapshot (drops broken epoch-1 best.pth selection)
  - NO detach() in loss path

Per-center INDEPENDENT: train one center at a time, save to its own dir.
For 3-center pilot use scripts/ecgtwin_gen/run_train_ct_v2_3centers.sh.

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/train_center_token_v2.py \
      --dataset /root/autodl-tmp/center_token_super5/extra_k200.pt \
      --save_dir /root/autodl-tmp/center_token_super5_v2/extra_k200/ \
      --total_steps 5000 --batch_size 8
"""
import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from methods.ecgtwin_gen.center_token.trainer_v2 import CenterTokenTrainerV2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Per-center .pt from prep_center_dataset_super5.py "
                         "(K=200 records with VAE latents + text_embed + metadata)")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--config", default=str(REPO / "methods/ecgtwin_gen/center_token/config_v2.yaml"))
    ap.add_argument("--ecgtwin_config", default=str(REPO / "model/ECGTwin/config/DiT_ECGTwin.yaml"))
    ap.add_argument("--device", default="cuda:0")
    # Optional CLI overrides
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr_phase1", type=float, default=None)
    ap.add_argument("--lr_phase2", type=float, default=None)
    ap.add_argument("--iso_freq", type=float, default=None)
    ap.add_argument("--iso_weight", type=float, default=None)
    ap.add_argument("--sphere_target_norm", type=float, default=None)
    ap.add_argument("--log_every", type=int, default=None)
    ap.add_argument("--ptbxl_latent_path", default=None,
                    help="Override isolation pool source")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.ecgtwin_config) as f:
        ecgtwin_config = yaml.safe_load(f)

    # Apply CLI overrides
    tcfg = config["training_v2"]
    if args.total_steps is not None:
        tcfg["total_steps"] = args.total_steps
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size
    if args.lr_phase1 is not None:
        tcfg["lr_phase1"] = args.lr_phase1
    if args.lr_phase2 is not None:
        tcfg["lr_phase2"] = args.lr_phase2
    if args.iso_freq is not None:
        tcfg["iso_freq"] = args.iso_freq
    if args.iso_weight is not None:
        tcfg["iso_weight"] = args.iso_weight
    if args.log_every is not None:
        tcfg["log_every"] = args.log_every
    if args.sphere_target_norm is not None:
        config["center_token"]["sphere_target_norm"] = args.sphere_target_norm

    print(f"[cfg] dataset={args.dataset}")
    print(f"[cfg] save_dir={args.save_dir}")
    print(f"[cfg] training_v2={tcfg}")
    print(f"[cfg] center_token={config['center_token']}")

    iso_path = args.ptbxl_latent_path or config["data"]["isolation_pool_path"]
    if not Path(iso_path).is_absolute():
        iso_path = str(REPO / iso_path)

    trainer = CenterTokenTrainerV2(
        config=config, ecgtwin_config=ecgtwin_config, device=args.device,
    )
    trainer.train(
        dataset_path=args.dataset,
        save_dir=args.save_dir,
        ptbxl_latent_path=iso_path,
    )


if __name__ == "__main__":
    main()
