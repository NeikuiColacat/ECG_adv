"""CPU-only tests for the EfficientNet VAE-LHAT wrapper command builder."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ecg_adv_gen.config.adapters.common import argv_option_map, opt_first, opt_list
from ecg_adv_gen.run_naming import build_effnet_vae_lhat_run_leaf
from ecg_adv_gen.runner.effnet_vae_lhat import (
    build_effnet_vae_lhat_eval_cmd,
    build_effnet_vae_lhat_train_cmd,
    resolve_effnet_vae_lhat_paths,
)


def _args(**overrides):
    values = {
        "center": "georgia",
        "epochs": 30,
        "seed": 20260531,
        "device": "cuda:0",
        "num_workers": 2,
        "anchor_base": "",
        "synth_npz_override": "",
        "hull_steps": 3,
        "hull_M": 20,
        "hull_lambda": 0.15,
        "hull_lr": 0.25,
        "hull_include_anchor": False,
        "hull_label_mode": "primary",
        "hull_mix_label_mode": "anchor",
        "hull_label_lambda_y": 0.5,
        "hull_label_positive": 0.95,
        "hull_label_negative_floor": 0.0,
        "hull_label_new_class_cap": 0.5,
        "hull_neighbor_distance_space": "raw",
        "hull_neighbor_mode": "nearest",
        "hull_neighbor_pool_size": 0,
        "hull_neighbor_pool_multiplier": 4,
        "k_anchor": 300,
        "pgd_batch": 32,
        "anchor_class_weights": "",
        "es_metric": "target_macro_auprc",
        "classes_in_scope": ["CD", "MI"],
        "init_ckpt": "",
        "source_weights": "real_anchor=1.0",
        "source_class_weights": "",
        "source_floor_per_class": 0,
        "anchor_class_weight_mode": "manual",
        "anchor_class_weight_reference_source": "real_anchor",
        "anchor_class_weight_gamma": 0.5,
        "anchor_class_weight_min": 0.35,
        "anchor_class_weight_cap": 4.0,
        "anchor_class_missing_weight": 0.35,
        "target_real_weight": 80.0,
        "adv_weight": 0.2,
        "adv_weight_warmup_epochs": 0,
        "adv_label_mode": "mixed_soft",
        "adv_teacher_mix": 0.3,
        "adv_soft_target_floor": 0.0,
        "boundary_prob_min": 0.0,
        "boundary_prob_max": 1.0,
        "ptbxl_weight": 1.0,
        "lr": 5e-5,
        "train_batch_size": 128,
        "source_logit_anchor_weight": 0.0,
        "source_logit_anchor_batches": 0,
        "freeze_backbone_classifier_only": False,
        "classifier_only_train_final_norm": False,
        "classifier_adapter_type": "linear",
        "classifier_lora_rank": 16,
        "classifier_lora_alpha": 16.0,
        "unfreeze_last_n_features": 0,
        "latent_augmix_latent_weight_cap": 0.3,
        "disable_latent_augmix_branch": False,
        "latent_augmix_copies": 1,
        "latent_augmix_width": 3,
        "latent_augmix_depth": -1,
        "latent_augmix_alpha": 1.0,
        "latent_augmix_severity": 2,
        "latent_augmix_ops": ["powerline_noise", "baseline_wander"],
        "enable_raw_corrupt_consistency": False,
        "raw_corrupt_copies": 1,
        "raw_corrupt_prob": 0.5,
        "raw_corrupt_severity": 4,
        "raw_corrupt_severity_profile": "standard",
        "raw_corrupt_ops": ["emg_noise"],
        "raw_corrupt_consistency_weight": 0.5,
        "raw_corrupt_consistency_loss": "soft_bce",
        "raw_corrupt_bce_weight": 0.1,
        "raw_corrupt_max_batches": 0,
        "raw_corrupt_scope": "target",
        "raw_corrupt_no_renorm": False,
        "raw_corrupt_clip_abs": 6.0,
        "enable_mask_shift_consistency": False,
        "mask_shift_copies": 1,
        "mask_shift_mask_severity": 6,
        "mask_shift_shift_severity": 6,
        "mask_shift_consistency_weight": 1.0,
        "mask_shift_consistency_loss": "jsd",
        "mask_shift_bce_weight": 0.05,
        "mask_shift_max_batches": 0,
        "mask_shift_scope": "target",
        "mask_shift_no_renorm": False,
        "mask_shift_clip_abs": 6.0,
        "quick_eval_source": "target_real_val",
        "quick_eval_n_per_center": 500,
        "target_real_val_fraction": 0.2,
        "target_real_val_seed": 20260531,
        "eval_batch_size": 192,
        "eval_min_pos": 10,
        "eval_pn2021_limit": 0,
        "run_tag_extra": "",
        "resume": "",
        "allow_resume_config_drift": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resolve_effnet_vae_lhat_paths_preserves_legacy_defaults_and_overrides():
    data_root = Path("/tmp/ecg-data")
    out_root = data_root / "paper_effnet_latent_augmix_stage3_20260524"
    args = _args()

    paths = resolve_effnet_vae_lhat_paths(args, data_root=data_root, out_root=out_root)

    assert paths.out_dir == out_root / build_effnet_vae_lhat_run_leaf(args)
    assert paths.anchor_base == (
        data_root
        / "ecgtwin_prompt_token_super5/real_anchor_selected_v2"
        / "georgia"
        / "georgia_real_k500_seed42"
    )
    assert paths.signal_npz == paths.anchor_base.with_suffix(".signals.npz")
    assert paths.latent_npz == paths.anchor_base.with_suffix(".latent.npz")
    assert paths.ref_meta == paths.anchor_base.with_suffix(".ref_meta.json")

    override_args = _args(
        anchor_base="/tmp/custom/ningbo_real_k500_seed20260531",
        synth_npz_override="/tmp/pools/source_target.latent.npz",
    )
    override_paths = resolve_effnet_vae_lhat_paths(
        override_args,
        data_root=data_root,
        out_root=out_root,
    )

    assert override_paths.signal_npz == Path("/tmp/custom/ningbo_real_k500_seed20260531.signals.npz")
    assert override_paths.latent_npz == Path("/tmp/pools/source_target.latent.npz")
    assert override_paths.ref_meta == Path("/tmp/custom/ningbo_real_k500_seed20260531.ref_meta.json")


def test_build_effnet_vae_lhat_commands_preserve_wrapper_flags():
    data_root = Path("/tmp/ecg-data")
    out_root = data_root / "runs"
    args = _args(
        hull_include_anchor=True,
        enable_raw_corrupt_consistency=True,
        raw_corrupt_no_renorm=True,
        enable_mask_shift_consistency=True,
        mask_shift_no_renorm=True,
        mask_shift_mask_severity=5,
        mask_shift_shift_severity=5,
        mask_shift_consistency_weight=10.0,
        resume="latest",
        allow_resume_config_drift=True,
        anchor_class_weights="MI=2.0",
        source_class_weights="real_anchor:MI=2.0",
        freeze_backbone_classifier_only=True,
        classifier_only_train_final_norm=True,
        eval_pn2021_limit=25,
    )
    paths = resolve_effnet_vae_lhat_paths(args, data_root=data_root, out_root=out_root)

    train_cmd = build_effnet_vae_lhat_train_cmd(
        args,
        python="/env/bin/python",
        data_root=data_root,
        paths=paths,
        class_trust=Path("/tmp/config/georgia_class_trust.json"),
    )
    train_opts = argv_option_map(train_cmd)

    assert train_cmd[:3] == ["/env/bin/python", "-u", "scripts/pgd_cross_center/synth_online_at_super5.py"]
    assert opt_first(train_opts, "--center_name") == "georgia"
    assert opt_first(train_opts, "--synth_npz") == str(paths.latent_npz)
    assert opt_first(train_opts, "--target_real_npz") == str(paths.signal_npz)
    assert opt_first(train_opts, "--init_ckpt") == str(
        data_root / "triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt"
    )
    assert opt_list(train_opts, "--classes_in_scope") == ["CD", "MI"]
    assert train_opts["--allow_hyp_cd_trust"] is True
    assert train_opts["--hull_include_anchor"] is True
    assert train_opts["--enable_latent_augmix_branch"] is True
    assert train_opts["--enable_raw_corrupt_consistency"] is True
    assert train_opts["--raw_corrupt_no_renorm"] is True
    assert opt_first(train_opts, "--raw_corrupt_severity_profile") == "standard"
    assert train_opts["--enable_mask_shift_consistency"] is True
    assert train_opts["--mask_shift_no_renorm"] is True
    assert opt_first(train_opts, "--mask_shift_mask_severity") == "5"
    assert opt_first(train_opts, "--mask_shift_shift_severity") == "5"
    assert opt_first(train_opts, "--mask_shift_consistency_weight") == "10.0"
    assert opt_first(train_opts, "--resume") == "latest"
    assert train_opts["--allow_resume_config_drift"] is True
    assert opt_first(train_opts, "--anchor_class_weights") == "MI=2.0"
    assert opt_first(train_opts, "--source_class_weights") == "real_anchor:MI=2.0"
    assert train_opts["--freeze_backbone_classifier_only"] is True
    assert train_opts["--classifier_only_train_final_norm"] is True

    eval_cmd = build_effnet_vae_lhat_eval_cmd(
        args,
        python="/env/bin/python",
        data_root=data_root,
        paths=paths,
    )
    eval_opts = argv_option_map(eval_cmd)

    assert eval_cmd[:3] == ["/env/bin/python", "-u", "scripts/triple_labels/eval_crosscenter.py"]
    assert opt_first(eval_opts, "--model_dir") == str(paths.out_dir)
    assert opt_first(eval_opts, "--exclude_ref_ids") == str(paths.ref_meta)
    assert opt_first(eval_opts, "--output_path") == str(
        paths.out_dir / "eval_result_v7_exclrefs_crop1000.json"
    )
    assert opt_first(eval_opts, "--pn2021_limit") == "25"


def test_build_effnet_vae_lhat_eval_cmd_uses_all_target_refs_for_standard_anchor_base():
    data_root = Path("/tmp/ecg-data")
    out_root = data_root / "runs"
    args = _args(
        anchor_base="/tmp/refs/georgia/k500_seed20260531/georgia_real_k500_seed20260531",
    )
    paths = resolve_effnet_vae_lhat_paths(args, data_root=data_root, out_root=out_root)

    eval_cmd = build_effnet_vae_lhat_eval_cmd(
        args,
        python="/env/bin/python",
        data_root=data_root,
        paths=paths,
    )
    eval_opts = argv_option_map(eval_cmd)

    assert opt_list(eval_opts, "--exclude_ref_ids") == [
        "/tmp/refs/ningbo/k500_seed20260531/ningbo_real_k500_seed20260531.ref_meta.json",
        "/tmp/refs/chapman_shaoxing/k500_seed20260531/chapman_shaoxing_real_k500_seed20260531.ref_meta.json",
        "/tmp/refs/cpsc_2018/k500_seed20260531/cpsc_2018_real_k500_seed20260531.ref_meta.json",
        "/tmp/refs/georgia/k500_seed20260531/georgia_real_k500_seed20260531.ref_meta.json",
    ]
