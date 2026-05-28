"""CPU-only tests for legacy-compatible run naming helpers."""

from __future__ import annotations

from argparse import Namespace

from ecg_adv_gen.run_naming import (
    build_ecgfounder_fullft_method_tag,
    build_ecgfounder_fullft_run_leaf,
    build_ecgfounder_fullft_selection_tag,
    build_effnet_direct_run_leaf,
    build_effnet_vae_lhat_run_leaf,
    tag_value,
)


def _base(**overrides):
    params = {
        "center": "ningbo",
        "k": 500,
        "epochs": 10,
        "lr": 0.0001,
        "source_weight": 1.0,
        "target_real_weight": 40.0,
        "enable_vae_adv_stream": False,
        "init_head_path": "",
        "vae_classes_in_scope": None,
        "vae_min_class_count": 1,
        "adv_weight": 20.0,
        "k_anchor": 100,
        "hull_m": 20,
        "hull_lambda": 0.15,
        "hull_steps": 3,
        "hull_lr": 0.25,
        "hull_weight_mode": "optimized",
        "hull_attack_pos_weight_source": "none",
        "hull_attack_pos_weight_clip": 50.0,
        "hull_label_mode": "primary",
        "hull_include_anchor": False,
        "hull_partner_pool": "target",
        "source_partner_limit_per_class": 0,
        "anchor_sample_mode": "stratified",
        "anchor_sample_power": 1.0,
        "anchor_class_sample_weights": "",
        "anchor_class_max_repeat": 0,
        "adv_weight_warmup_epochs": 0,
        "adv_bce_loss_weight": 1.0,
        "adv_clean_logit_anchor_weight": 0.0,
        "run_suffix": "",
        "target_val_count": 0,
        "selection_metric": "source_auprc",
        "target_val_split_mode": "random",
        "target_val_seed": None,
        "run_name": "",
        "seed": 20260531,
    }
    params.update(overrides)
    return Namespace(**params)


def test_tag_value_matches_legacy_filename_format():
    assert tag_value(0.15) == "0p15"
    assert tag_value(-2.0) == "m2p0"
    assert tag_value("1e-05") == "1em05"


def test_ecgfounder_fullft_run_leaf_plain_and_init_head_modes():
    plain = _base()
    init = _base(init_head_path="/tmp/head.pt", run_suffix="smoke")

    assert build_ecgfounder_fullft_method_tag(plain) == "fullft"
    assert build_ecgfounder_fullft_run_leaf(plain) == (
        "ningbo_K500_fullft_ep10_lr0p0001_sw1p0_tw40p0_fullft_seed20260531"
    )
    assert build_ecgfounder_fullft_method_tag(init) == "fullft_inithead_smoke"
    assert build_ecgfounder_fullft_run_leaf(init).endswith(
        "_fullft_inithead_smoke_seed20260531"
    )


def test_ecgfounder_fullft_run_leaf_complex_vae_mode_preserves_legacy_tokens():
    params = _base(
        center="georgia",
        epochs=3,
        enable_vae_adv_stream=True,
        init_head_path="/tmp/head.pt",
        vae_classes_in_scope=["HYP", "MI"],
        vae_min_class_count=2,
        adv_weight=12.5,
        k_anchor=64,
        hull_m=8,
        hull_lambda=0.2,
        hull_steps=2,
        hull_lr=0.15,
        hull_weight_mode="uniform",
        hull_attack_pos_weight_source="source_target",
        hull_attack_pos_weight_clip=25.0,
        hull_label_mode="compatible",
        hull_include_anchor=True,
        hull_partner_pool="target_source",
        source_partner_limit_per_class=7,
        anchor_sample_mode="hard_bce",
        anchor_sample_power=1.5,
        anchor_class_sample_weights="HYP=2,MI=4",
        anchor_class_max_repeat=3,
        adv_weight_warmup_epochs=4,
        adv_bce_loss_weight=0.5,
        adv_clean_logit_anchor_weight=0.1,
        run_suffix="smoke",
        target_val_count=50,
        selection_metric="source_plus_target_val_auprc",
        target_val_split_mode="stratified",
        target_val_seed=123,
    )

    method = build_ecgfounder_fullft_method_tag(params)
    selection = build_ecgfounder_fullft_selection_tag(params)
    leaf = build_ecgfounder_fullft_run_leaf(params)

    assert method == (
        "fullft_vae_inithead_clsHYP-MI_mincnt2_aw12p5_ka64_M8_lam0p2"
        "_hs2_hlr0p15_uniform_apwsource_targetclip25p0_labelcompatible"
        "_includeanchor_partnerstarget_source_splim7_ashard_bce_aspow1p5"
        "_acwHYP2-MI4_acmaxrep3_awarm4_abce0p5_aclean0p1_smoke"
    )
    assert selection == "_tv50_source_plus_target_val_auprc_stratified_tvseed123"
    assert leaf == (
        "georgia_K500_fullft_ep3_lr0p0001_sw1p0_tw40p0_"
        f"{method}{selection}_seed20260531"
    )


def test_ecgfounder_fullft_run_leaf_honors_explicit_run_name_and_mapping_inputs():
    assert build_ecgfounder_fullft_run_leaf({"run_name": "manual_child"}) == "manual_child"
    assert build_ecgfounder_fullft_method_tag(
        {
            "enable_vae_adv_stream": "false",
            "init_head_path": "",
            "run_suffix": "x",
        }
    ) == "fullft_x"


def test_effnet_direct_run_leaf_matches_legacy_child_dir():
    assert build_effnet_direct_run_leaf(
        {
            "center": "ningbo",
            "k": "500",
            "epochs": "30",
            "seed": "20260531",
            "val_fraction": "0.2",
        }
    ) == "ningbo_K500_direct_ft_ep30_seed20260531_val0.2"
    assert build_effnet_direct_run_leaf(
        {
            "center": "georgia",
            "k": 32,
            "epochs": 1,
            "seed": 7,
            "val_fraction": 0.125,
        }
    ) == "georgia_K32_direct_ft_ep1_seed7_val0.125"


def test_effnet_vae_lhat_run_leaf_matches_stage3_legacy_tags():
    assert build_effnet_vae_lhat_run_leaf(
        {
            "center": "cpsc_2018",
            "hull_M": 20,
            "hull_lambda": 0.05,
            "latent_augmix_severity": 2,
            "latent_augmix_latent_weight_cap": 0.25,
            "hull_steps": 3,
            "es_metric": "target_macro_auprc",
            "classes_in_scope": ["CD", "HYP", "MI", "NORM", "STTC"],
            "hull_label_mode": "compatible",
            "hull_mix_label_mode": "anchor_soft",
            "hull_neighbor_distance_space": "standardized",
            "hull_neighbor_mode": "local_random",
            "hull_neighbor_pool_size": 120,
            "hull_neighbor_pool_multiplier": 4,
            "epochs": 20,
            "seed": 20260531,
        }
    ) == (
        "cpsc_2018_realall_targetheavy_M20_lam0p05_augmix_s2_wlat0p25"
        "_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft"
        "_sta_local_random_p120_fullft_ep20_seed20260531"
    )


def test_effnet_vae_lhat_run_leaf_encodes_adapter_modes():
    assert build_effnet_vae_lhat_run_leaf(
        {
            "center": "ningbo",
            "freeze_backbone_classifier_only": True,
            "classifier_only_train_final_norm": True,
            "classifier_adapter_type": "lora",
            "classifier_lora_rank": 8,
            "run_tag_extra": "smoke",
        }
    ).endswith("_headfn_lora8_smoke_ep30_seed20260531")
    assert build_effnet_vae_lhat_run_leaf(
        {"center": "ningbo", "unfreeze_last_n_features": 2}
    ).endswith("_last2_ep30_seed20260531")
