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

from apps.streamlit_ecg_demo.services.paths import DATA_ROOT  # noqa: E402
from methods.ecgtwin_gen.prompt_token.trainer import ECGTwinPromptTokenTrainer  # noqa: E402


DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DEFAULT_CACHE_ROOT = str(DATA_ROOT / "ecgtwin_prompt_token_super5/cache_v1")
DEFAULT_SAVE_DIR = str(DATA_ROOT / "ecgtwin_prompt_token_super5/prompt_token_runs/v1_all4")
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
    ap.add_argument("--style_ckpt", default=None,
                    help="Optional frozen center-style classifier checkpoint.")
    ap.add_argument("--style_loss_weight", type=float, default=0.0,
                    help="Weight for target-center CE loss on decoded one-step x0.")
    ap.add_argument("--style_loss_mode", choices=["ce", "target_prob"], default="ce",
                    help="ce maximizes target-center classification; target_prob matches --style_target_prob.")
    ap.add_argument("--style_target_prob", type=float, default=0.65,
                    help="Target center probability for style_loss_mode=target_prob.")
    ap.add_argument("--style_crop_len", type=int, default=1000)
    ap.add_argument("--semantic_ckpt", default=None,
                    help="Optional frozen EfficientNet super5 checkpoint.")
    ap.add_argument("--semantic_loss_weight", type=float, default=0.0,
                    help="Weight for super5 BCE preservation loss on decoded one-step x0.")
    ap.add_argument("--semantic_crop_len", type=int, default=250)
    ap.add_argument("--contrast_recon_weight", type=float, default=0.0,
                    help="Weight for paired token-vs-no-token denoise MSE margin loss.")
    ap.add_argument("--contrast_recon_margin", type=float, default=0.0,
                    help="Required token MSE advantage margin; 0 means token should be no worse.")
    ap.add_argument("--contrast_style_delta_weight", type=float, default=0.0,
                    help="Weight for target-center probability margin versus no-token.")
    ap.add_argument("--contrast_style_delta_margin", type=float, default=0.05,
                    help="Required P(target center) token minus no-token margin.")
    ap.add_argument("--contrast_semantic_delta_weight", type=float, default=0.0,
                    help="Weight for primary-class semantic probability margin versus no-token.")
    ap.add_argument("--contrast_semantic_delta_margin", type=float, default=0.02,
                    help="Required P(primary class) token minus no-token margin.")
    ap.add_argument("--feature_loss_weight", type=float, default=0.0,
                    help="Weight for frozen EfficientNet penultimate feature alignment to the real ECG latent.")
    ap.add_argument("--feature_crop_len", type=int, default=1000,
                    help="Crop length used by the frozen EfficientNet feature loss.")
    ap.add_argument("--contrast_feature_weight", type=float, default=0.0,
                    help="Weight for requiring token x0 to be closer than no-token x0 in EfficientNet feature space.")
    ap.add_argument("--contrast_feature_margin", type=float, default=0.0,
                    help="Required no-token minus token feature-distance margin.")
    ap.add_argument("--feature_mmd_weight", type=float, default=0.0,
                    help="Weight for batch-level MMD between token x0 EfficientNet features and real ECG features.")
    ap.add_argument("--contrast_feature_mmd_weight", type=float, default=0.0,
                    help="Weight for requiring token feature-MMD to beat no-token feature-MMD.")
    ap.add_argument("--feature_mmd_margin", type=float, default=0.0,
                    help="Required no-token minus token feature-MMD margin.")
    ap.add_argument("--feature_mmd_sigma", type=float, default=1.0,
                    help="RBF sigma for feature-MMD on normalized EfficientNet features.")
    ap.add_argument("--aux_timestep_max", type=int, default=0,
                    help="If >0 and aux losses are enabled, sample denoise t in [1, aux_timestep_max).")
    ap.add_argument("--aux_amp_clamp", type=float, default=6.0)
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
        style_ckpt=args.style_ckpt,
        style_loss_weight=args.style_loss_weight,
        style_loss_mode=args.style_loss_mode,
        style_target_prob=args.style_target_prob,
        style_crop_len=args.style_crop_len,
        semantic_ckpt=args.semantic_ckpt,
        semantic_loss_weight=args.semantic_loss_weight,
        semantic_crop_len=args.semantic_crop_len,
        contrast_recon_weight=args.contrast_recon_weight,
        contrast_recon_margin=args.contrast_recon_margin,
        contrast_style_delta_weight=args.contrast_style_delta_weight,
        contrast_style_delta_margin=args.contrast_style_delta_margin,
        contrast_semantic_delta_weight=args.contrast_semantic_delta_weight,
        contrast_semantic_delta_margin=args.contrast_semantic_delta_margin,
        feature_loss_weight=args.feature_loss_weight,
        feature_crop_len=args.feature_crop_len,
        contrast_feature_weight=args.contrast_feature_weight,
        contrast_feature_margin=args.contrast_feature_margin,
        feature_mmd_weight=args.feature_mmd_weight,
        contrast_feature_mmd_weight=args.contrast_feature_mmd_weight,
        feature_mmd_margin=args.feature_mmd_margin,
        feature_mmd_sigma=args.feature_mmd_sigma,
        aux_timestep_max=args.aux_timestep_max,
        aux_amp_clamp=args.aux_amp_clamp,
    )
    trainer.train(
        save_dir=args.save_dir,
        total_steps=args.total_steps,
        log_every=args.log_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
