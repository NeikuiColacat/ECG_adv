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
from ecg_adv_gen.evaluation.pn2021c import (  # noqa: E402
    STRESS_PROFILE_CHOICES as PN2021C_STRESS_PROFILE_CHOICES,
)
from ecg_adv_gen.models.super5_model_zoo import available_model_names  # noqa: E402
from ecg_adv_gen.matched_effnet import (  # noqa: E402
    MATCHED_EFFNET_ARMS,
    MATCHED_EFFNET_CONTRACT_VERSION,
    MATCHED_EFFNET_THIRD_CHAIN_ROUTES,
    f004_identity,
    is_f004_rho_sweep,
    is_matched_effnet_arm,
    validate_f004_runtime,
    validate_matched_effnet_runtime,
)
from ecg_adv_gen.f005_control import validate_f005_runtime  # noqa: E402

CLASS_NAMES = list(CLASS_NAMES_SUPER5)


def run(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    run_stream(cmd, log_path=log_path, env=env, cwd=REPO)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", default="cpsc_2018")
    ap.add_argument(
        "--comparison_arm",
        choices=["historical_unmatched", *MATCHED_EFFNET_ARMS],
        default="historical_unmatched",
    )
    ap.add_argument("--comparison_protocol", default="")
    ap.add_argument("--comparison_variant", default="")
    ap.add_argument("--comparison_topology_version", default="")
    ap.add_argument("--comparison_topology_sha256", default="")
    ap.add_argument("--study_scope", default="")
    ap.add_argument("--mechanism_variant", default="")
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
    ap.add_argument("--hull_init_logit_gap", type=float, default=4.0)
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
    ap.add_argument("--pgd_eps", type=float, default=2.0)
    ap.add_argument("--pgd_batch", type=int, default=32)
    ap.add_argument("--asr_low_threshold", type=float, default=0.30)
    ap.add_argument("--asr_high_threshold", type=float, default=0.70)
    ap.add_argument("--classes_in_scope", nargs="+", default=CLASS_NAMES)
    ap.add_argument(
        "--init_ckpt",
        default="",
        help="Optional EfficientNet checkpoint to initialize from; defaults to the PTB-XL source baseline.",
    )
    ap.add_argument("--init_checkpoint_sha256", default="")
    ap.add_argument("--init_lineage_stage", default="")
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
    ap.add_argument("--target_real_weight", type=float, default=80.0)
    ap.add_argument("--target_adv_fraction", type=float, choices=[0.0, 0.25, 0.5], default=None)
    ap.add_argument("--adv_weight", type=float, default=0.2)
    ap.add_argument("--vae_adv_stream_sample_scale", type=float, default=1.0)
    ap.add_argument("--vae_adv_consistency_weight", type=float, default=0.0)
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
    ap.add_argument("--ptbxl_weight", type=float, default=1.0)
    ap.add_argument("--target_real_val_fraction", type=float, default=0.2)
    ap.add_argument("--target_real_val_seed", type=int, default=20260531)
    ap.add_argument("--selection_metric", choices=["macro_auroc", "macro_auprc"], default="macro_auprc")
    ap.add_argument("--source_floor_max_drop", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--train_batch_size", type=int, default=128)
    ap.add_argument(
        "--latent_augmix_third_chain_role",
        choices=list(MATCHED_EFFNET_THIRD_CHAIN_ROUTES),
        default="vae_lhat_adversarial_waveform",
    )
    ap.add_argument(
        "--latent_augmix_chain_base_mode",
        choices=["clean_clean_third", "all_clean", "one_adv", "all_adv", "all_clean_plus_vae_adv"],
        default="clean_clean_third",
    )
    ap.add_argument("--latent_augmix_adv_base_mix", type=float, default=1.0)
    ap.add_argument("--latent_augmix_copies", type=int, default=1)
    ap.add_argument("--latent_augmix_width", type=int, default=3)
    ap.add_argument("--latent_augmix_depth", type=int, default=-1)
    ap.add_argument("--latent_augmix_alpha", type=float, default=1.0)
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
    ap.add_argument("--latent_augmix_consistency_weight", type=float, default=2.0)
    ap.add_argument(
        "--latent_augmix_consistency_loss",
        choices=["soft_bce", "jsd"],
        default="jsd",
    )
    ap.add_argument("--latent_augmix_bce_weight", type=float, default=1.0)
    ap.add_argument("--latent_augmix_chain_weights", default="")
    ap.add_argument("--latent_augmix_consistency_max_batches", type=int, default=0)
    consistency = ap.add_mutually_exclusive_group()
    consistency.add_argument(
        "--enable_latent_augmix_consistency",
        dest="enable_latent_augmix_consistency",
        action="store_true",
    )
    consistency.add_argument(
        "--disable_latent_augmix_consistency",
        dest="enable_latent_augmix_consistency",
        action="store_false",
    )
    ap.set_defaults(enable_latent_augmix_consistency=True)
    vae_lhat = ap.add_mutually_exclusive_group()
    vae_lhat.add_argument("--enable_vae_lhat", dest="enable_vae_lhat", action="store_true")
    vae_lhat.add_argument("--disable_vae_lhat", dest="enable_vae_lhat", action="store_false")
    raw_augmix = ap.add_mutually_exclusive_group()
    raw_augmix.add_argument("--enable_raw_augmix", dest="enable_raw_augmix", action="store_true")
    raw_augmix.add_argument("--disable_raw_augmix", dest="enable_raw_augmix", action="store_false")
    ap.set_defaults(enable_vae_lhat=True, enable_raw_augmix=True)
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
    ap.add_argument(
        "--final_checkpoint_only",
        action="store_true",
        help="Forwarded to synth_online_at_super5.py to avoid epoch resume checkpoint writes.",
    )
    args = ap.parse_args(argv)
    if is_matched_effnet_arm(args.comparison_arm):
        try:
            validate_matched_effnet_runtime(
                args.comparison_arm,
                enable_vae_lhat=args.enable_vae_lhat,
                enable_raw_augmix=args.enable_raw_augmix,
                enable_auxiliary_steps=args.enable_latent_augmix_consistency,
                bce_weight=args.latent_augmix_bce_weight,
                jsd_weight=args.latent_augmix_consistency_weight,
                third_chain_route=args.latent_augmix_third_chain_role,
                hull_label_mode=args.hull_label_mode,
                hull_include_anchor=args.hull_include_anchor,
                target_adv_fraction=args.target_adv_fraction,
            )
        except ValueError as exc:
            ap.error(str(exc))
    if (
        args.comparison_protocol or args.comparison_variant
        or args.comparison_topology_version or args.comparison_topology_sha256
    ):
        if not is_f004_rho_sweep(args.comparison_protocol):
            ap.error(f"unsupported --comparison_protocol {args.comparison_protocol!r}")
        if args.comparison_topology_version != MATCHED_EFFNET_CONTRACT_VERSION:
            ap.error("F-004 comparison topology version mismatch")
        try:
            validate_f004_runtime(
                comparison_protocol=args.comparison_protocol,
                comparison_arm=args.comparison_arm,
                comparison_variant=args.comparison_variant,
                comparison_topology_sha256=args.comparison_topology_sha256,
                target_adv_fraction=args.target_adv_fraction,
                kshot_seed=args.seed,
                kshot_path=args.anchor_base,
                enable_vae_lhat=args.enable_vae_lhat,
                enable_raw_augmix=args.enable_raw_augmix,
                enable_latent_augmix_consistency=args.enable_latent_augmix_consistency,
                latent_augmix_bce_weight=args.latent_augmix_bce_weight,
                latent_augmix_consistency_weight=args.latent_augmix_consistency_weight,
                latent_augmix_consistency_loss=args.latent_augmix_consistency_loss,
                latent_augmix_third_chain_role=args.latent_augmix_third_chain_role,
                latent_augmix_width=args.latent_augmix_width,
                latent_augmix_depth=args.latent_augmix_depth,
                latent_augmix_copies=args.latent_augmix_copies,
                latent_augmix_chain_base_mode=args.latent_augmix_chain_base_mode,
                latent_augmix_adv_base_mix=args.latent_augmix_adv_base_mix,
                latent_augmix_alpha=args.latent_augmix_alpha,
                latent_augmix_severity=args.latent_augmix_severity,
                latent_augmix_severity_profile=args.latent_augmix_severity_profile,
                latent_augmix_ops=args.latent_augmix_ops,
                latent_augmix_signal_space="raw_pre_zscore",
                hull_M=args.hull_M,
                hull_lambda=args.hull_lambda,
                hull_steps=args.hull_steps,
                hull_include_anchor=args.hull_include_anchor,
                hull_init_logit_gap=args.hull_init_logit_gap,
                hull_label_mode=args.hull_label_mode,
                hull_mix_label_mode=args.hull_mix_label_mode,
                hull_lr=args.hull_lr,
                hull_neighbor_distance_space=args.hull_neighbor_distance_space,
                hull_neighbor_mode=args.hull_neighbor_mode,
                hull_neighbor_pool_size=args.hull_neighbor_pool_size,
                hull_neighbor_pool_multiplier=args.hull_neighbor_pool_multiplier,
                pgd_eps=args.pgd_eps,
            )
        except (TypeError, ValueError) as exc:
            ap.error(str(exc))
    try:
        validate_f005_runtime(
            study_scope=args.study_scope,
            mechanism_variant=args.mechanism_variant,
            comparison_arm=args.comparison_arm,
            hull_lambda=args.hull_lambda,
            hull_label_mode=args.hull_label_mode,
            hull_include_anchor=args.hull_include_anchor,
        )
    except ValueError as exc:
        ap.error(str(exc))
    return args


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root) if args.out_root else (
        data_root / "paper_effnet_latent_augmix_stage3_20260524"
    )
    center = args.center
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

    launch_payload = {
        "args": vars(args),
        "class_trust": str(class_trust),
        "train_cmd": train_cmd,
        "eval_cmd": eval_cmd,
    }
    if is_f004_rho_sweep(args.comparison_protocol):
        launch_payload["comparison_identity"] = f004_identity(args.target_adv_fraction)
    (out_dir / "launch_config.json").write_text(json.dumps(launch_payload, indent=2))

    run(train_cmd, out_dir / "train_stdout.log", env)
    run(eval_cmd, out_dir / "eval_full.log", env)
    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
