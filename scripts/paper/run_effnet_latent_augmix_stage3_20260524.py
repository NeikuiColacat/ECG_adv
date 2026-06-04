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
from ecg_adv_gen.runner.process import build_process_env, run_stream  # noqa: E402
from ecg_adv_gen.run_naming import build_effnet_vae_lhat_run_leaf  # noqa: E402
from methods.augmix.severity import AVAILABLE_OPS  # noqa: E402

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
    ap.add_argument("--latent_augmix_severity", type=int, default=2)
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
        "--quick_eval_source",
        choices=["pn2021", "target_real_val"],
        default="pn2021",
    )
    ap.add_argument("--quick_eval_n_per_center", type=int, default=500)
    ap.add_argument("--target_real_val_fraction", type=float, default=0.2)
    ap.add_argument("--target_real_val_seed", type=int, default=20260531)
    ap.add_argument("--eval_batch_size", type=int, default=192)
    ap.add_argument("--eval_min_pos", type=int, default=10)
    ap.add_argument("--eval_pn2021_limit", type=int, default=0)
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
    out_dir = out_root / build_effnet_vae_lhat_run_leaf(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_base = Path(args.anchor_base) if args.anchor_base else (
        data_root
        / "ecgtwin_prompt_token_super5/real_anchor_selected_v2"
        / center
        / f"{center}_real_k500_seed42"
    )
    signal_npz = anchor_base.with_suffix(".signals.npz")
    latent_npz = (
        Path(args.synth_npz_override)
        if args.synth_npz_override
        else anchor_base.with_suffix(".latent.npz")
    )
    ref_meta = anchor_base.with_suffix(".ref_meta.json")
    class_trust = write_real_all_present_trust(
        signal_npz,
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
    train_cmd = [
        python, "-u", "scripts/pgd_cross_center/synth_online_at_super5.py",
        "--center_name", center,
        "--ref_meta_json", str(ref_meta),
        "--synth_npz", str(latent_npz),
        "--target_real_npz", str(signal_npz),
        "--class_trust", str(class_trust),
        "--init_ckpt",
        str(
            Path(args.init_ckpt)
            if args.init_ckpt
            else data_root / "triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt"
        ),
        "--model_name", "efficientnet1dv2",
        "--output_dir", str(out_dir),
        "--data_dir", str(data_root / "physionet2021/training"),
        "--quick_eval_source", args.quick_eval_source,
        "--quick_eval_centers", center,
        "--quick_eval_n_per_center", str(args.quick_eval_n_per_center),
        "--target_real_val_fraction", str(args.target_real_val_fraction),
        "--target_real_val_seed", str(args.target_real_val_seed),
        "--ptbxl_raw", str(data_root / "ptbxl/raw100.npy"),
        "--ptbxl_csv", str(data_root / "ptbxl/ptbxl_database.csv"),
        "--ptbxl_prep", str(data_root / "crosscenter_v2/ptbxl_preprocessed.npy"),
        "--attack_mode", "latent_hull",
        "--hull_M", str(args.hull_M),
        "--hull_lambda", str(args.hull_lambda),
        "--hull_steps", str(args.hull_steps),
        "--hull_lr", str(args.hull_lr),
        "--hull_label_mode", args.hull_label_mode,
        "--hull_mix_label_mode", args.hull_mix_label_mode,
        "--hull_label_lambda_y", str(args.hull_label_lambda_y),
        "--hull_label_positive", str(args.hull_label_positive),
        "--hull_label_negative_floor", str(args.hull_label_negative_floor),
        "--hull_label_new_class_cap", str(args.hull_label_new_class_cap),
        "--hull_neighbor_distance_space", args.hull_neighbor_distance_space,
        "--hull_neighbor_mode", args.hull_neighbor_mode,
        "--hull_neighbor_pool_size", str(args.hull_neighbor_pool_size),
        "--hull_neighbor_pool_multiplier", str(args.hull_neighbor_pool_multiplier),
        "--source_sampling_strategy", "source_weighted",
        "--source_weights", args.source_weights,
        "--source_floor_per_class", str(args.source_floor_per_class),
        "--anchor_class_weight_mode", args.anchor_class_weight_mode,
        "--anchor_class_weight_reference_source", args.anchor_class_weight_reference_source,
        "--anchor_class_weight_gamma", str(args.anchor_class_weight_gamma),
        "--anchor_class_weight_min", str(args.anchor_class_weight_min),
        "--anchor_class_weight_cap", str(args.anchor_class_weight_cap),
        "--anchor_class_missing_weight", str(args.anchor_class_missing_weight),
        "--K_anchor", str(args.k_anchor),
        "--pgd_batch", str(args.pgd_batch),
        "--classes_in_scope", *args.classes_in_scope,
        "--allow_hyp_cd_trust",
        "--target_real_weight", str(args.target_real_weight),
        "--adv_weight", str(args.adv_weight),
        "--adv_weight_warmup_epochs", str(args.adv_weight_warmup_epochs),
        "--ptbxl_weight", str(args.ptbxl_weight),
        "--roundtrip_weight", "0.0",
        "--roundtrip_anchor_n", "0",
        "--source_logit_anchor_weight", str(args.source_logit_anchor_weight),
        "--source_logit_anchor_batches", str(args.source_logit_anchor_batches),
        "--adv_label_mode", args.adv_label_mode,
        "--adv_teacher_mix", str(args.adv_teacher_mix),
        "--adv_soft_target_floor", str(args.adv_soft_target_floor),
        "--boundary_prob_min", str(args.boundary_prob_min),
        "--boundary_prob_max", str(args.boundary_prob_max),
        "--disable_quality_gate",
        "--lr", str(args.lr),
        "--weight_decay", "1e-4",
        "--batch_size", str(args.train_batch_size),
        "--n_epochs", str(args.epochs),
        "--patience", str(args.epochs),
        "--eval_every", "2",
        "--es_metric", args.es_metric,
        "--ewa_decay", "0.999",
        "--anchor_lambda", "0.05",
        "--asr_consec_low_max", "999",
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--crop_len", "1000",
        "--device", args.device,
    ]
    if args.hull_include_anchor:
        train_cmd.append("--hull_include_anchor")
    if not args.disable_latent_augmix_branch:
        train_cmd.extend([
            "--enable_latent_augmix_branch",
            "--latent_augmix_copies", str(args.latent_augmix_copies),
            "--latent_augmix_width", str(args.latent_augmix_width),
            "--latent_augmix_depth", str(args.latent_augmix_depth),
            "--latent_augmix_alpha", str(args.latent_augmix_alpha),
            "--latent_augmix_severity", str(args.latent_augmix_severity),
            "--latent_augmix_latent_weight_cap", str(args.latent_augmix_latent_weight_cap),
            "--latent_augmix_ops", *args.latent_augmix_ops,
        ])
    if args.enable_raw_corrupt_consistency:
        train_cmd.extend([
            "--enable_raw_corrupt_consistency",
            "--raw_corrupt_copies", str(args.raw_corrupt_copies),
            "--raw_corrupt_prob", str(args.raw_corrupt_prob),
            "--raw_corrupt_severity", str(args.raw_corrupt_severity),
            "--raw_corrupt_ops", *args.raw_corrupt_ops,
            "--raw_corrupt_consistency_weight", str(args.raw_corrupt_consistency_weight),
            "--raw_corrupt_consistency_loss", str(args.raw_corrupt_consistency_loss),
            "--raw_corrupt_bce_weight", str(args.raw_corrupt_bce_weight),
            "--raw_corrupt_max_batches", str(args.raw_corrupt_max_batches),
            "--raw_corrupt_scope", str(args.raw_corrupt_scope),
            "--raw_corrupt_clip_abs", str(args.raw_corrupt_clip_abs),
        ])
        if args.raw_corrupt_no_renorm:
            train_cmd.append("--raw_corrupt_no_renorm")
    if args.resume:
        train_cmd.extend(["--resume", args.resume])
    if args.allow_resume_config_drift:
        train_cmd.append("--allow_resume_config_drift")
    if args.anchor_class_weights:
        train_cmd.extend(["--anchor_class_weights", args.anchor_class_weights])
    if args.source_class_weights:
        train_cmd.extend(["--source_class_weights", args.source_class_weights])
    if args.freeze_backbone_classifier_only:
        train_cmd.extend([
            "--freeze_backbone_classifier_only",
            "--classifier_adapter_type", args.classifier_adapter_type,
            "--classifier_lora_rank", str(args.classifier_lora_rank),
            "--classifier_lora_alpha", str(args.classifier_lora_alpha),
        ])
        if args.classifier_only_train_final_norm:
            train_cmd.append("--classifier_only_train_final_norm")
    if args.unfreeze_last_n_features > 0:
        train_cmd.extend(["--unfreeze_last_n_features", str(args.unfreeze_last_n_features)])
    eval_cmd = [
        python, "-u", "scripts/triple_labels/eval_crosscenter.py",
        "--scheme", "super5",
        "--model_dir", str(out_dir),
        "--model_name", "efficientnet1dv2",
        "--device", "cuda",
        "--crop_len", "1000",
        "--batch_size", str(args.eval_batch_size),
        "--min_pos", str(args.eval_min_pos),
        "--num_workers", str(args.num_workers),
        "--ptbxl_csv", str(data_root / "ptbxl/ptbxl_database.csv"),
        "--ptbxl_cache", str(data_root / "crosscenter_v2/ptbxl_preprocessed.npy"),
        "--preprocess_mode", "minimal_resample",
        "--norm_mode", "per_sample_global",
        "--pn2021_root", str(data_root / "physionet2021"),
        "--pn2021_cache_dir", str(data_root / "triple_labels/pn2021_eval_cache_minresample_perglobal"),
        "--pn2021_mmap_cache_dir", str(data_root / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"),
        "--skip_mimic",
        "--report_drop_all_zero_pn2021",
        "--exclude_ref_ids", str(ref_meta),
        "--output_path", str(out_dir / "eval_result_v7_exclrefs_crop1000.json"),
    ]
    if args.eval_pn2021_limit > 0:
        eval_cmd.extend(["--pn2021_limit", str(args.eval_pn2021_limit)])

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
