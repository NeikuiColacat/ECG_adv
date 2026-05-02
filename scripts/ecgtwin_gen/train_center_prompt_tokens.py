"""Train ECGTwin 768-d center-class prompt tokens.

This is the active textual-inversion-style path:
  base ECGTwin prompt embeddings + one learnable [center, super5_class] token.

The token is appended directly to ECGTwin's `text_embed` sequence; tokenizer
vocabulary and original ECGTwin weights remain unchanged.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from methods.ecgtwin_gen.prompt_token.trainer import ECGTwinPromptTokenTrainer  # noqa: E402


DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DEFAULT_CACHE_ROOT = "/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1"
DEFAULT_SAVE_DIR = "/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/v1_all4"
DEFAULT_PROMPT_BANK = f"{DEFAULT_CACHE_ROOT}/text_prompt_bank.pt"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--prompt_bank", default=DEFAULT_PROMPT_BANK)
    ap.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    ap.add_argument("--ecgtwin_config", default=str(REPO / "model/ECGTwin/config/DiT_ECGTwin.yaml"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--reg_init_weight", type=float, default=1.0e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp_dtype", choices=["bf16", "fp16", "none"], default="bf16")
    ap.add_argument("--init_noise_std", type=float, default=0.01)
    ap.add_argument("--token_repeat", type=int, default=1,
                    help="Append the same learned center-class token this many times.")
    ap.add_argument("--sample_strategy",
                    choices=["shuffle", "class_balanced", "center_class_balanced"],
                    default="shuffle")
    ap.add_argument("--n_token_vectors", type=int, default=1,
                    help="True multi-vector soft prompt length for direct token mode.")
    ap.add_argument("--token_mode", choices=["direct", "factorized"], default="direct",
                    help="direct=(center,class,vector,dim); factorized=center+class+residual.")
    ap.add_argument("--center_vectors", type=int, default=1)
    ap.add_argument("--class_vectors", type=int, default=1)
    ap.add_argument("--residual_vectors", type=int, default=1)
    ap.add_argument("--residual_init_std", type=float, default=0.0)
    ap.add_argument("--token_orth_weight", type=float, default=0.0,
                    help="Penalty on squared cosine between vectors in one soft prompt.")
    ap.add_argument("--ref_text_mode", choices=["class_fallback", "actual_report", "normal"],
                    default="class_fallback",
                    help="Text embedding used by IBE/base_vector path.")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=0,
                    help="If >0, save intermediate prompt_token_bank_stepXXXXX.pt checkpoints.")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    print(f"[cfg] centers={args.centers}")
    print(f"[cfg] cache_root={args.cache_root}")
    print(f"[cfg] save_dir={args.save_dir}")
    print(
        f"[cfg] steps={args.total_steps} batch={args.batch_size} "
        f"workers={args.num_workers} amp={args.amp_dtype}"
    )

    trainer = ECGTwinPromptTokenTrainer(
        centers=args.centers,
        cache_root=args.cache_root,
        prompt_bank_path=args.prompt_bank,
        ecgtwin_config_path=args.ecgtwin_config,
        device=args.device,
        k=args.K,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        reg_init_weight=args.reg_init_weight,
        grad_clip=args.grad_clip,
        amp_dtype=args.amp_dtype,
        init_noise_std=args.init_noise_std,
        token_repeat=args.token_repeat,
        sample_strategy=args.sample_strategy,
        n_token_vectors=args.n_token_vectors,
        token_mode=args.token_mode,
        center_vectors=args.center_vectors,
        class_vectors=args.class_vectors,
        residual_vectors=args.residual_vectors,
        residual_init_std=args.residual_init_std,
        token_orth_weight=args.token_orth_weight,
        ref_text_mode=args.ref_text_mode,
    )
    trainer.train(
        save_dir=args.save_dir,
        total_steps=args.total_steps,
        log_every=args.log_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
