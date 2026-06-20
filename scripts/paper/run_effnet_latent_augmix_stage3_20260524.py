#!/usr/bin/env python3
"""Run one EfficientNet1DV2 stage-3 latent-branch AugMix LHAT test.

This is a narrow pilot for the current migrated host:

  target real anchors -> VAE Latent-Hull x_adv
  -> x_adv as one AugMix branch + ECG corruption chains
  -> EfficientNet1DV2 online AT
  -> full PN2021 ref-excluded evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        str(_MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")),
    )
)

from ecg_adv_gen.data.class_trust import write_real_all_present_trust  # noqa: E402
from ecg_adv_gen.labels.super5 import CLASS_NAMES_SUPER5  # noqa: E402
from ecg_adv_gen.runner.effnet_vae_lhat import (  # noqa: E402
    build_effnet_vae_lhat_eval_cmd,
    build_effnet_vae_lhat_train_cmd,
    resolve_effnet_vae_lhat_paths,
)
from ecg_adv_gen.runner.process import build_process_env, run_stream  # noqa: E402
from methods.augmix.severity import AVAILABLE_OPS  # noqa: E402
from scripts.triple_labels.eval_pn2021_corruptions import (  # noqa: E402
    STRESS_PROFILE_CHOICES as PN2021C_STRESS_PROFILE_CHOICES,
)
from scripts.triple_labels.model_zoo import available_model_names  # noqa: E402

CLASS_NAMES = list(CLASS_NAMES_SUPER5)


def run(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    run_stream(cmd, log_path=log_path, env=env, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", default="cpsc_2018")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260524)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--model_name", default="efficientnet1dv2", choices=available_model_names())
    ap.add_argument("--data_root", default=str(DATA_ROOT))
    ap.add_argument("--out_root", default="")
    ap.add_argument(
        "--anchor_base",
        default="",
        help=(
            "Optional path prefix without suffix for K-shot anchors, e.g. "
            ".../georgia_real_k500_seed20260531. When omitted, uses the "
            "historical real_anchor_selected_v2 seed42 prefix."
        ),
    )
    ap.add_argument("--hull_steps", type=int, default=3)
    ap.add_argument("--hull_M", type=int, default=20)
    ap.add_argument("--hull_lambda", type=float, default=0.15)
    ap.add_argument("--hull_lr", type=float, default=0.25)
    ap.add_argument("--hull_include_anchor", action="store_true")
    ap.add_argument(
        "--hull_label_mode",
        choices=["primary", "exact", "compatible"],
        default="primary",
        help=(
            "Latent-hull neighbor grouping passed to synth_online_at_super5. "
            "compatible allows non-NORM abnormal anchors to use abnormal "
            "neighbors while still avoiding NORM/abnormal contradiction."
        ),
    )
    ap.add_argument(
        "--hull_mix_label_mode",
        choices=["anchor", "anchor_soft"],
        default="anchor",
        help=(
            "Label target construction for latent-hull mixed samples. "
            "anchor keeps the historical anchor label; anchor_soft adds "
            "capped fractional labels from weighted hull neighbors."
        ),
    )
    ap.add_argument("--hull_label_lambda_y", type=float, default=0.5)
    ap.add_argument("--hull_label_positive", type=float, default=0.95)
    ap.add_argument("--hull_label_negative_floor", type=float, default=0.0)
    ap.add_argument("--hull_label_new_class_cap", type=float, default=0.5)
    ap.add_argument(
        "--hull_neighbor_distance_space",
        choices=["raw", "standardized"],
        default="raw",
    )
    ap.add_argument(
        "--hull_neighbor_mode",
        choices=["nearest", "local_random", "random"],
        default="nearest",
    )
    ap.add_argument("--hull_neighbor_pool_size", type=int, default=0)
    ap.add_argument("--hull_neighbor_pool_multiplier", type=int, default=4)
    ap.add_argument("--k_anchor", type=int, default=300)
    ap.add_argument("--pgd_batch", type=int, default=32)
    ap.add_argument(
        "--anchor_class_weights",
        default="",
        help="Optional CLASS=weight comma map passed to the latent anchor sampler.",
    )
    ap.add_argument("--es_metric", choices=["target_macro_auroc", "target_macro_auprc"], default="target_macro_auprc")
    ap.add_argument("--classes_in_scope", nargs="+", default=CLASS_NAMES)
    ap.add_argument(
        "--init_ckpt",
        default="",
        help="Optional EfficientNet checkpoint to initialize from; defaults to the PTB-XL source baseline.",
    )
    ap.add_argument(
        "--synth_npz_override",
        default="",
        help=(
            "Optional latent pool for adversarial anchor sampling. The K500 "
            "target signals/ref ids still come from --anchor_base; use this "
            "for source+target partner-pool diversity tests."
        ),
    )
    ap.add_argument(
        "--target_real_npz_override",
        default="",
        help=(
            "Optional target-real supervised signal artifact. Use a raw1000 "
            "artifact with --target_real_norm_mode per_sample_global for the "
            "locked PN2021-C raw-first protocol."
        ),
    )
    ap.add_argument(
        "--target_real_norm_mode",
        choices=["pre_zscored", "per_sample_global"],
        default="pre_zscored",
        help=(
            "Normalization contract for --target_real_npz. pre_zscored keeps "
            "legacy selected .signals.npz behavior; per_sample_global z-scores "
            "raw1000 target ECGs inside the online-AT dataset before the model."
        ),
    )
    ap.add_argument(
        "--source_weights",
        default="real_anchor=1.0",
        help="Comma map passed to synth_online_at_super5 --source_weights.",
    )
    ap.add_argument(
        "--source_class_weights",
        default="",
        help="Optional class-source map passed to --source_class_weights.",
    )
    ap.add_argument(
        "--source_floor_per_class",
        type=int,
        default=0,
        help="Optional source floor passed to --source_floor_per_class.",
    )
    ap.add_argument(
        "--anchor_class_weight_mode",
        choices=["manual", "inv_freq_kshot"],
        default="manual",
        help="Passed to synth_online_at_super5 --anchor_class_weight_mode.",
    )
    ap.add_argument(
        "--anchor_class_weight_reference_source",
        default="real_anchor",
        help="Passed to synth_online_at_super5 --anchor_class_weight_reference_source.",
    )
    ap.add_argument("--anchor_class_weight_gamma", type=float, default=0.5)
    ap.add_argument("--anchor_class_weight_min", type=float, default=0.35)
    ap.add_argument("--anchor_class_weight_cap", type=float, default=4.0)
    ap.add_argument("--anchor_class_missing_weight", type=float, default=0.35)
    ap.add_argument("--target_real_weight", type=float, default=80.0)
    ap.add_argument("--adv_weight", type=float, default=0.2)
    ap.add_argument("--adv_weight_warmup_epochs", type=int, default=0)
    ap.add_argument(
        "--adv_label_mode",
        choices=[
            "hard",
            "multi_hot_hard",
            "latent_soft",
            "mixed_soft",
            "teacher_soft",
            "latent_mixed_teacher",
        ],
        default="mixed_soft",
    )
    ap.add_argument("--adv_teacher_mix", type=float, default=0.3)
    ap.add_argument("--adv_soft_target_floor", type=float, default=0.0)
    ap.add_argument(
        "--boundary_prob_min",
        type=float,
        default=0.0,
        help=(
            "Only buffer adversarial decoded samples whose post-attack target "
            "probability is at least this value. This keeps attack strength in "
            "a useful range without changing the PGD objective."
        ),
    )
    ap.add_argument(
        "--boundary_prob_max",
        type=float,
        default=1.0,
        help=(
            "Only buffer adversarial decoded samples whose post-attack target "
            "probability is at most this value."
        ),
    )
    ap.add_argument("--ptbxl_weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_batch_size", type=int, default=128)
    ap.add_argument(
        "--source_logit_anchor_weight",
        type=float,
        default=0.0,
        help="Optional PTB-XL source-logit distillation weight to reduce source forgetting.",
    )
    ap.add_argument(
        "--source_logit_anchor_batches",
        type=int,
        default=0,
        help="Max PTB-XL source batches for each source-logit anchor pass; 0 uses full source loader.",
    )
    ap.add_argument(
        "--freeze_backbone_classifier_only",
        action="store_true",
        help=(
            "Freeze EfficientNet feature extractor and train only the final "
            "classifier adapter. This tests whether VAE-LHAT helps under a "
            "source-preserving low-capacity adaptation."
        ),
    )
    ap.add_argument(
        "--classifier_only_train_final_norm",
        action="store_true",
        help=(
            "With --freeze_backbone_classifier_only, also train final_norm "
            "affine parameters while keeping running statistics frozen."
        ),
    )
    ap.add_argument(
        "--classifier_adapter_type",
        choices=["linear", "lora"],
        default="linear",
        help="Classifier adapter used with --freeze_backbone_classifier_only.",
    )
    ap.add_argument("--classifier_lora_rank", type=int, default=16)
    ap.add_argument("--classifier_lora_alpha", type=float, default=16.0)
    ap.add_argument(
        "--unfreeze_last_n_features",
        type=int,
        default=0,
        help=(
            "Train classifier, final_conv/final_norm, and the last N "
            "EfficientNet feature blocks with BatchNorm running statistics "
            "frozen. Mutually exclusive with --freeze_backbone_classifier_only."
        ),
    )
    ap.add_argument("--latent_augmix_latent_weight_cap", type=float, default=0.3)
    ap.add_argument(
        "--latent_augmix_topology",
        choices=["legacy_branch", "locked_three_chain"],
        default="legacy_branch",
        help=(
            "Forwarded to synth_online_at_super5.py. locked_three_chain keeps "
            "the PN2021-C main method as two corruption chains plus one "
            "uncorrupted VAE-LHAT adversarial waveform chain."
        ),
    )
    ap.add_argument(
        "--disable_latent_augmix_branch",
        action="store_true",
        help=(
            "Do not forward --enable_latent_augmix_branch to "
            "synth_online_at_super5.py. This creates a true VAE-LHAT-only "
            "control while keeping this legacy wrapper's defaults unchanged."
        ),
    )
    ap.add_argument("--latent_augmix_copies", type=int, default=1)
    ap.add_argument("--latent_augmix_width", type=int, default=3)
    ap.add_argument("--latent_augmix_depth", type=int, default=-1)
    ap.add_argument("--latent_augmix_alpha", type=float, default=1.0)
    ap.add_argument("--latent_augmix_mixture_mode", choices=["beta", "fixed"], default="beta")
    ap.add_argument("--latent_augmix_mixture_prob", type=float, default=0.5)
    ap.add_argument("--latent_augmix_mixture_beta_a", type=float, default=0.0)
    ap.add_argument("--latent_augmix_mixture_beta_b", type=float, default=0.0)
    ap.add_argument(
        "--latent_augmix_op_schedule",
        choices=["random", "cycle", "per_op"],
        default="random",
        help=(
            "Forwarded to locked three-chain AugMix. per_op maps AugMix copies "
            "onto the configured PN2021-C operators inside the main three-chain graph."
        ),
    )
    ap.add_argument(
        "--latent_augmix_chain_weights",
        default="",
        help="Optional comma-separated locked-chain weights: raw1,raw2,vae_lhat_adv.",
    )
    ap.add_argument(
        "--latent_augmix_signal_space",
        choices=["model_zscore", "raw_pre_zscore"],
        default="model_zscore",
        help="Forwarded to locked three-chain AugMix corruption input space.",
    )
    ap.add_argument(
        "--latent_augmix_corruption_source",
        choices=["vae_decode", "target_real"],
        default="vae_decode",
        help=(
            "Forwarded to locked three-chain AugMix. target_real uses the "
            "picked K500 raw ECG as the two corruption-chain anchors while "
            "keeping the VAE-LH adversarial waveform as the third chain."
        ),
    )
    ap.add_argument("--latent_augmix_severity", type=int, default=2)
    ap.add_argument(
        "--latent_augmix_severity_profile",
        choices=PN2021C_STRESS_PROFILE_CHOICES,
        default="standard",
        help=(
            "Parameter profile forwarded to non-latent latent-AugMix ECG op chains. "
            "Use calibrated_10to20pp to match the strong PN2021-C evaluator."
        ),
    )
    ap.add_argument("--latent_augmix_severity_params_file", default="")
    ap.add_argument("--latent_augmix_severity_params_name", default="")
    ap.add_argument(
        "--latent_augmix_ops",
        nargs="+",
        choices=AVAILABLE_OPS,
        default=["powerline_noise", "emg_noise", "baseline_wander", "baseline_shift"],
        help=(
            "ECG corruption ops used by non-latent AugMix branches. The legacy "
            "default excludes random_leads_masking; robustness sweeps can pass "
            "all five ops explicitly."
        ),
    )
    ap.add_argument(
        "--no_latent_augmix_renorm",
        action="store_true",
        help="Forwarded to latent AugMix so calibrated chains keep their raw corruption scale.",
    )
    ap.add_argument(
        "--enable_latent_augmix_consistency",
        action="store_true",
        help="Forward latent-AugMix generated views through a direct BCE/JSD training phase.",
    )
    ap.add_argument("--latent_augmix_consistency_weight", type=float, default=2.0)
    ap.add_argument(
        "--latent_augmix_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="jsd",
    )
    ap.add_argument("--latent_augmix_bce_weight", type=float, default=1.0)
    ap.add_argument("--latent_augmix_consistency_max_batches", type=int, default=0)
    ap.add_argument(
        "--enable_raw_corrupt_consistency",
        action="store_true",
        help=(
            "Enable a target-real raw corruption consistency phase after each "
            "normal LHAT epoch. This is the direct PN2021-C robustness branch."
        ),
    )
    ap.add_argument("--raw_corrupt_copies", type=int, default=1)
    ap.add_argument("--raw_corrupt_prob", type=float, default=0.5)
    ap.add_argument("--raw_corrupt_severity", type=int, default=4)
    ap.add_argument(
        "--raw_corrupt_severity_profile",
        choices=PN2021C_STRESS_PROFILE_CHOICES,
        default="standard",
        help=(
            "Parameter profile forwarded to the raw corruption consistency branch. "
            "Use calibrated_10to20pp to match the strong PN2021-C evaluator."
        ),
    )
    ap.add_argument("--raw_corrupt_severity_params_file", default="")
    ap.add_argument("--raw_corrupt_severity_params_name", default="")
    ap.add_argument(
        "--raw_corrupt_ops",
        nargs="+",
        choices=AVAILABLE_OPS,
        default=[
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
            "random_leads_masking",
        ],
    )
    ap.add_argument("--raw_corrupt_consistency_weight", type=float, default=0.5)
    ap.add_argument(
        "--raw_corrupt_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="soft_bce",
    )
    ap.add_argument("--raw_corrupt_bce_weight", type=float, default=0.1)
    ap.add_argument("--raw_corrupt_max_batches", type=int, default=0)
    ap.add_argument(
        "--raw_corrupt_scope",
        choices=["target", "source", "source_target"],
        default="target",
    )
    ap.add_argument("--raw_corrupt_no_renorm", action="store_true")
    ap.add_argument("--raw_corrupt_clip_abs", type=float, default=6.0)
    ap.add_argument(
        "--raw_corrupt_view_mode",
        choices=["single_op", "augmix"],
        default="single_op",
        help=(
            "Forwarded to the raw corruption consistency branch. single_op "
            "uses one sampled PN2021-C operator per view; augmix mixes "
            "multi-op corruption chains into each view."
        ),
    )
    ap.add_argument("--raw_augmix_width", type=int, default=3)
    ap.add_argument("--raw_augmix_depth", type=int, default=-1)
    ap.add_argument("--raw_augmix_alpha", type=float, default=1.0)
    ap.add_argument("--raw_augmix_mixture_mode", choices=["beta", "fixed"], default="beta")
    ap.add_argument("--raw_augmix_mixture_prob", type=float, default=0.5)
    ap.add_argument("--raw_augmix_mixture_beta_a", type=float, default=0.0)
    ap.add_argument("--raw_augmix_mixture_beta_b", type=float, default=0.0)
    ap.add_argument("--raw_input_bandpass_low_hz", type=float, default=None)
    ap.add_argument("--raw_input_bandpass_high_hz", type=float, default=None)
    ap.add_argument("--raw_input_repair_flat_leads", action="store_true")
    ap.add_argument("--raw_input_clip_abs", type=float, default=None)
    ap.add_argument("--raw_input_renorm_after_stabilizer", action="store_true")
    ap.add_argument("--raw_input_sample_rate_hz", type=float, default=100.0)
    ap.add_argument(
        "--enable_mask_shift_consistency",
        action="store_true",
        help=(
            "Enable the deterministic random_leads_masking + baseline_shift "
            "consistency phase in synth_online_at_super5.py."
        ),
    )
    ap.add_argument("--mask_shift_copies", type=int, default=1)
    ap.add_argument("--mask_shift_mask_severity", type=int, default=6)
    ap.add_argument("--mask_shift_shift_severity", type=int, default=6)
    ap.add_argument("--mask_shift_consistency_weight", type=float, default=1.0)
    ap.add_argument(
        "--mask_shift_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="jsd",
    )
    ap.add_argument("--mask_shift_bce_weight", type=float, default=0.05)
    ap.add_argument("--mask_shift_max_batches", type=int, default=0)
    ap.add_argument(
        "--mask_shift_scope",
        choices=["target", "source", "source_target"],
        default="target",
    )
    ap.add_argument("--mask_shift_no_renorm", action="store_true")
    ap.add_argument("--mask_shift_clip_abs", type=float, default=6.0)
    ap.add_argument(
        "--quick_eval_source",
        choices=["pn2021", "target_real_val", "none"],
        default="pn2021",
    )
    ap.add_argument("--quick_eval_n_per_center", type=int, default=500)
    ap.add_argument("--target_real_val_fraction", type=float, default=0.2)
    ap.add_argument("--target_real_val_seed", type=int, default=20260531)
    ap.add_argument("--checkpoint_policy", choices=["best", "last"], default="best")
    ap.add_argument("--eval_batch_size", type=int, default=192)
    ap.add_argument("--eval_min_pos", type=int, default=10)
    ap.add_argument("--eval_pn2021_limit", type=int, default=0)
    ap.add_argument(
        "--qab_size",
        type=int,
        default=2048,
        help="Forwarded to synth_online_at_super5.py QualityAwareBuffer max size.",
    )
    ap.add_argument(
        "--rescore_interval",
        type=int,
        default=3,
        help="Forwarded to synth_online_at_super5.py buffer rescore interval in epochs.",
    )
    ap.add_argument(
        "--asr_consec_low_max",
        type=int,
        default=999,
        help="Forwarded to synth_online_at_super5.py low-ASR stop patience.",
    )
    ap.add_argument(
        "--asr_low_threshold",
        type=float,
        default=0.30,
        help="Forwarded to synth_online_at_super5.py low-ASR threshold.",
    )
    ap.add_argument(
        "--run_tag_extra",
        default="",
        help="Optional short suffix appended to the run directory name for parameter sweeps.",
    )
    ap.add_argument(
        "--resume",
        default="",
        help=(
            "Forwarded to synth_online_at_super5.py. Use 'latest' to resume "
            "from the run directory's checkpoints/checkpoint_latest.pt."
        ),
    )
    ap.add_argument(
        "--allow_resume_config_drift",
        action="store_true",
        help="Forwarded to synth_online_at_super5.py for intentional recovery only.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root) if args.out_root else (
        data_root / "paper_effnet_latent_augmix_stage3_20260524"
    )
    center = args.center
    if args.freeze_backbone_classifier_only and args.unfreeze_last_n_features > 0:
        raise ValueError(
            "--freeze_backbone_classifier_only and --unfreeze_last_n_features "
            "are mutually exclusive"
        )
    paths = resolve_effnet_vae_lhat_paths(args, data_root=data_root, out_root=out_root)
    out_dir = paths.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    class_trust = write_real_all_present_trust(
        paths.signal_npz,
        out_root / "config",
        center,
        classes_in_scope=args.classes_in_scope,
        policy=(
            "VAE-only real-anchor stage-3 AugMix test; trust every Super5 "
            "class present in the real target-center K subset."
        ),
    )

    # PyTorch multiprocessing creates AF_UNIX sockets under TMPDIR. The
    # migrated data path is too long for that socket path, so use a short
    # user-owned temp directory while keeping large caches on the data disk.
    env = build_process_env(
        updates={
            "TMPDIR": os.environ.get("TMPDIR", str(Path.home() / "tmp_ecg")),
            "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", str(data_root / "cache")),
            "DEEPECG_NOTEBOOKS": os.environ.get(
                "DEEPECG_NOTEBOOKS",
                str(REPO / "model" / "DeepECG" / "notebooks"),
            ),
        },
    )

    python = sys.executable
    train_cmd = build_effnet_vae_lhat_train_cmd(
        args,
        python=python,
        data_root=data_root,
        paths=paths,
        class_trust=class_trust,
    )
    eval_cmd = build_effnet_vae_lhat_eval_cmd(
        args,
        python=python,
        data_root=data_root,
        paths=paths,
    )

    (out_dir / "launch_config.json").write_text(json.dumps({
        "args": vars(args),
        "class_trust": str(class_trust),
        "train_cmd": train_cmd,
        "eval_cmd": eval_cmd,
    }, indent=2))

    run(train_cmd, out_dir / "train_stdout.log", env)
    run(eval_cmd, out_dir / "eval_full.log", env)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
