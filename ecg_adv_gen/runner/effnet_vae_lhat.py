"""Command builders for the EfficientNet VAE-LHAT managed runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ecg_adv_gen.evaluation.ref_exclusion import append_target_ref_exclusion_args_from_anchor_base
from ecg_adv_gen.run_naming import build_effnet_vae_lhat_run_leaf


@dataclass(frozen=True)
class EffNetVaeLhatPaths:
    """Derived paths consumed by the EfficientNet VAE-LHAT wrapper."""

    out_dir: Path
    anchor_base: Path
    signal_npz: Path
    latent_npz: Path
    ref_meta: Path


def resolve_effnet_vae_lhat_paths(
    args: Any,
    *,
    data_root: Path,
    out_root: Path,
) -> EffNetVaeLhatPaths:
    """Resolve wrapper output and K-shot anchor paths without touching disk."""

    center = str(args.center)
    out_dir = out_root / build_effnet_vae_lhat_run_leaf(args)
    anchor_base = (
        Path(args.anchor_base)
        if args.anchor_base
        else data_root
        / "ecgtwin_prompt_token_super5/real_anchor_selected_v2"
        / center
        / f"{center}_real_k500_seed42"
    )
    signal_npz = (
        Path(args.target_real_npz_override)
        if getattr(args, "target_real_npz_override", "")
        else anchor_base.with_suffix(".signals.npz")
    )
    latent_npz = (
        Path(args.synth_npz_override)
        if args.synth_npz_override
        else anchor_base.with_suffix(".latent.npz")
    )
    ref_meta = anchor_base.with_suffix(".ref_meta.json")
    return EffNetVaeLhatPaths(
        out_dir=out_dir,
        anchor_base=anchor_base,
        signal_npz=signal_npz,
        latent_npz=latent_npz,
        ref_meta=ref_meta,
    )


def _default_init_ckpt(args: Any, data_root: Path) -> Path:
    return (
        Path(args.init_ckpt)
        if args.init_ckpt
        else data_root / "triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt"
    )


def build_effnet_vae_lhat_train_cmd(
    args: Any,
    *,
    python: str,
    data_root: Path,
    paths: EffNetVaeLhatPaths,
    class_trust: Path,
) -> list[str]:
    """Build the synth-online-AT command used by the managed runner."""

    model_name = str(getattr(args, "model_name", "efficientnet1dv2"))
    train_cmd = [
        python,
        "-u",
        "ecg_adv_gen/runner/synth_online_at_super5.py",
        "--center_name",
        str(args.center),
        "--ref_meta_json",
        str(paths.ref_meta),
        "--synth_npz",
        str(paths.latent_npz),
        "--target_real_npz",
        str(paths.signal_npz),
        "--target_real_norm_mode",
        str(getattr(args, "target_real_norm_mode", "pre_zscored")),
        "--class_trust",
        str(class_trust),
        "--init_ckpt",
        str(_default_init_ckpt(args, data_root)),
        "--model_name",
        model_name,
        "--output_dir",
        str(paths.out_dir),
        "--ptbxl_raw",
        str(data_root / "ptbxl/raw100.npy"),
        "--ptbxl_csv",
        str(data_root / "ptbxl/ptbxl_database.csv"),
        "--ptbxl_prep",
        str(data_root / "crosscenter_v2/ptbxl_preprocessed.npy"),
        "--hull_M",
        str(args.hull_M),
        "--hull_lambda",
        str(args.hull_lambda),
        "--hull_steps",
        str(args.hull_steps),
        "--hull_lr",
        str(args.hull_lr),
        "--hull_label_mode",
        str(args.hull_label_mode),
        "--hull_mix_label_mode",
        str(args.hull_mix_label_mode),
        "--hull_label_lambda_y",
        str(args.hull_label_lambda_y),
        "--hull_label_positive",
        str(args.hull_label_positive),
        "--hull_label_negative_floor",
        str(args.hull_label_negative_floor),
        "--hull_label_new_class_cap",
        str(args.hull_label_new_class_cap),
        "--hull_neighbor_distance_space",
        str(args.hull_neighbor_distance_space),
        "--hull_neighbor_mode",
        str(args.hull_neighbor_mode),
        "--hull_neighbor_pool_size",
        str(args.hull_neighbor_pool_size),
        "--hull_neighbor_pool_multiplier",
        str(args.hull_neighbor_pool_multiplier),
        "--K_anchor",
        str(args.k_anchor),
        "--pgd_batch",
        str(args.pgd_batch),
        "--classes_in_scope",
        *[str(item) for item in args.classes_in_scope],
        "--target_real_weight",
        str(args.target_real_weight),
        "--adv_weight",
        str(args.adv_weight),
        "--adv_weight_warmup_epochs",
        str(args.adv_weight_warmup_epochs),
        "--ptbxl_weight",
        str(args.ptbxl_weight),
        "--adv_label_mode",
        str(args.adv_label_mode),
        "--adv_teacher_mix",
        str(args.adv_teacher_mix),
        "--disable_quality_gate",
        "--lr",
        str(args.lr),
        "--weight_decay",
        "1e-4",
        "--batch_size",
        str(args.train_batch_size),
        "--n_epochs",
        str(args.epochs),
        "--eval_every",
        "2",
        "--ewa_decay",
        "0.999",
        "--anchor_lambda",
        "0.05",
        "--qab_size",
        "2048",
        "--rescore_interval",
        "3",
        "--asr_consec_low_max",
        "999",
        "--asr_low_threshold",
        "0.30",
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--crop_len",
        "1000",
        "--device",
        str(args.device),
    ]
    if args.hull_include_anchor:
        train_cmd.append("--hull_include_anchor")
    train_cmd.extend(
        [
            "--latent_augmix_copies",
            str(args.latent_augmix_copies),
            "--latent_augmix_width",
            str(args.latent_augmix_width),
            "--latent_augmix_depth",
            str(args.latent_augmix_depth),
            "--latent_augmix_alpha",
            str(args.latent_augmix_alpha),
            "--latent_augmix_severity",
            str(args.latent_augmix_severity),
            "--latent_augmix_severity_profile",
            str(getattr(args, "latent_augmix_severity_profile", "standard")),
            "--latent_augmix_latent_weight_cap",
            str(args.latent_augmix_latent_weight_cap),
            "--latent_augmix_ops",
            *[str(item) for item in args.latent_augmix_ops],
        ]
    )
    train_cmd.extend(
        [
            "--latent_augmix_consistency_weight",
            str(args.latent_augmix_consistency_weight),
            "--latent_augmix_consistency_loss",
            str(args.latent_augmix_consistency_loss),
            "--latent_augmix_bce_weight",
            str(args.latent_augmix_bce_weight),
            "--latent_augmix_consistency_max_batches",
            str(args.latent_augmix_consistency_max_batches),
        ]
    )
    if args.resume:
        train_cmd.extend(["--resume", str(args.resume)])
    if args.allow_resume_config_drift:
        train_cmd.append("--allow_resume_config_drift")
    return train_cmd


def build_effnet_vae_lhat_eval_cmd(
    args: Any,
    *,
    python: str,
    data_root: Path,
    paths: EffNetVaeLhatPaths,
) -> list[str]:
    """Build the full PN2021 ref-excluded evaluation command."""

    model_name = str(getattr(args, "model_name", "efficientnet1dv2"))
    eval_cmd = [
        python,
        "-u",
        "ecg_adv_gen/runner/pn2021_clean_eval.py",
        "--scheme",
        "super5",
        "--model_dir",
        str(paths.out_dir),
        "--model_name",
        model_name,
        "--device",
        "cuda",
        "--crop_len",
        "1000",
        "--batch_size",
        str(args.eval_batch_size),
        "--min_pos",
        str(args.eval_min_pos),
        "--num_workers",
        str(args.num_workers),
        "--ptbxl_csv",
        str(data_root / "ptbxl/ptbxl_database.csv"),
        "--ptbxl_cache",
        str(data_root / "crosscenter_v2/ptbxl_preprocessed.npy"),
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--pn2021_root",
        str(data_root / "physionet2021"),
        "--pn2021_cache_dir",
        str(data_root / "triple_labels/pn2021_eval_cache_minresample_perglobal"),
        "--pn2021_mmap_cache_dir",
        str(data_root / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"),
        "--skip_mimic",
        "--report_drop_all_zero_pn2021",
    ]
    eval_cmd.extend(["--checkpoint_name", "last_model.pt"])
    try:
        append_target_ref_exclusion_args_from_anchor_base(eval_cmd, paths.anchor_base)
    except ValueError:
        eval_cmd.extend(["--exclude_ref_ids", str(paths.ref_meta)])
    eval_cmd.extend([
        "--output_path",
        str(paths.out_dir / "eval_result_v7_exclrefs_crop1000.json"),
    ])
    if int(args.eval_pn2021_limit) > 0:
        eval_cmd.extend(["--pn2021_limit", str(args.eval_pn2021_limit)])
    return eval_cmd
