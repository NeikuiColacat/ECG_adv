"""CPU-only tests for the YAML experiment configuration layer."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ecg_adv_gen.config import (
    ALLOWED_CLI_OVERRIDE_KEYS,
    ConfigError,
    LaunchError,
    apply_cli_overrides,
    attach_launch_artifacts,
    build_postprocess_commands,
    build_runner_commands,
    check_nvidia_smi,
    default_run_dir,
    load_experiment_config,
    make_dry_run_manifest,
    prepare_output_dir,
    require_cuda_visible_devices,
    run_legacy_commands,
    run_postprocess_commands,
    validate_experiment_config,
    verify_required_inputs,
    verify_required_artifacts,
    write_k500_ref_ids_artifact,
    write_selection_record_artifact,
)
from ecg_adv_gen.config.loader import audit_runner_commands
from ecg_adv_gen.config.runner_audit import audit_runner_command
from ecg_adv_gen.config.adapters.direct import audit_direct_finetune_command
from ecg_adv_gen.config.adapters.source_training import audit_train_ptbxl_command
from ecg_adv_gen.config.paths import PathSafetyError, translate_legacy_path, validate_local_paths
from ecg_adv_gen.models import ecgfounder_kshot_head_run_dir, ecgfounder_lhat_run_dir
from ecg_adv_gen.run_naming import (
    build_benchmark_direct_run_leaf,
    build_ecgfounder_fullft_run_leaf,
    build_effnet_direct_run_leaf,
    build_effnet_vae_lhat_run_leaf,
)


REPO = Path(__file__).resolve().parents[2]
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"


def _load(name: str) -> dict:
    return load_experiment_config(
        REPO / "configs" / "experiments" / name,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest_run"},
    )


@pytest.mark.parametrize(
    ("config_name", "expected_commands"),
    [
        ("effnet_direct_k500_v6.yaml", 1),
        ("effnet_direct_k500_v6_smoke.yaml", 1),
        ("effnet_direct_k500_v7_sjr_rgq.yaml", 1),
        ("benchmark_resnet1d_direct_k500_v7_sjr_rgq_matrix.yaml", 4),
        ("effnet_direct_percent_v7_sjr_rgq_matrix.yaml", 8),
        ("effnet_vae_lhat_k500_v6.yaml", 4),
        ("effnet_vae_lhat_k500_v6_smoke.yaml", 4),
        ("effnet_vae_lhat_k500_v7_sjr_rgq.yaml", 4),
        ("effnet_vae_lhat_rawjsd_curriculum_k500_v7_sjr_rgq.yaml", 4),
        ("effnet_vae_lhat_rawjsd_curriculum_k500_v7_sjr_rgq_smoke.yaml", 1),
        ("effnet_vae_lhat_rawsupervised_stressor_k500_v7_sjr_rgq.yaml", 4),
        ("effnet_vae_lhat_rawsupervised_stressor_k500_v7_sjr_rgq_ningbo.yaml", 1),
        ("effnet_vae_lhat_rawsupervised_stressor_k500_v7_sjr_rgq_smoke.yaml", 1),
        ("effnet_vae_lhat_calibrated_rawsupervised_k500_v7_sjr_rgq.yaml", 4),
        ("effnet_vae_lhat_calibrated_rawsupervised_k500_v7_sjr_rgq_ningbo.yaml", 1),
        ("effnet_vae_lhat_calibrated_rawsupervised_k500_v7_sjr_rgq_smoke.yaml", 1),
        ("effnet_vae_lhat_calibrated_latent_augmix_k500_v7_sjr_rgq.yaml", 4),
        ("effnet_vae_lhat_calibrated_latent_augmix_k500_v7_sjr_rgq_cpsc_2018.yaml", 1),
        ("effnet_vae_lhat_calibrated_latent_augmix_norenorm_k500_v7_sjr_rgq_cpsc_2018.yaml", 1),
        ("effnet_vae_lhat_calibrated_latent_augmix_directloss_k500_v7_sjr_rgq_cpsc_2018.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_locked_k500.yaml", 4),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_k500_cpsc_base.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cpsc_d1_c2_cw2_bce1_m065.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cpsc_d1_c3_cw4_bce1_m065.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cpsc_d1_c3_cw6_bce1_m075.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cpsc_d1_c4_cw4_bce2_m075.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cpsc_randdepth_c3_cw4_bce1_m065.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cpsc_d1_c3_cw4_bce2_m075_hs5.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cpsc_strong_hlam10_aw1p5_c3_cw6_bce2_m085.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cpsc_strong_hlam15_aw2p0_c4_cw8_bce2_m090.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cpsc_highfreq_c6_cw10_bce3_m095.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cpsc_advheavy_hlam20_aw3p0_c4_cw10_bce3_m090.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cand09_highfreq_missing3.yaml", 3),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cand10_advheavy_missing3.yaml", 3),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cand09_highfreq_ningbo.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cand09_highfreq_chapman_shaoxing.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cand09_highfreq_georgia.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cand10_advheavy_ningbo.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cand10_advheavy_chapman_shaoxing.yaml", 1),
        ("effnet_vae_lhat_augmix_threechain_dualmodel_cand10_advheavy_georgia.yaml", 1),
        ("effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_stabilizer35_k500_v7_sjr_rgq.yaml", 4),
        ("effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_stabilizer35_k500_v7_sjr_rgq_smoke.yaml", 1),
        ("effnet_vae_lhat_maskshift_consistency_k500_v7_sjr_rgq.yaml", 4),
        ("effnet_vae_lhat_maskshift_consistency_k500_v7_sjr_rgq_ningbo.yaml", 1),
        ("effnet_vae_lhat_maskshift_consistency_k500_v7_sjr_rgq_smoke.yaml", 1),
        ("effnet_vae_lhat_percent_v7_sjr_rgq.yaml", 8),
        ("ecgfounder_direct_k500_v7_sjr_rgq_matrix.yaml", 4),
        ("ecgfounder_vae_lhat_augmix_k500_v7_sjr_rgq.yaml", 4),
        ("benchmark_source_v7_sjr_rgq_matrix.yaml", 3),
        ("benchmark_resnet1d_direct_percent_v7_sjr_rgq_matrix.yaml", 8),
        ("benchmark_inception1d_direct_k500_v7_sjr_rgq_matrix.yaml", 4),
        ("benchmark_fcn_wang_direct_k500_v7_sjr_rgq_matrix.yaml", 4),
        ("benchmark_resnet1d_vae_lhat_k500_v7_sjr_rgq.yaml", 4),
        ("benchmark_inception1d_vae_lhat_k500_v7_sjr_rgq.yaml", 4),
        ("benchmark_fcn_wang_vae_lhat_k500_v7_sjr_rgq.yaml", 4),
        ("benchmark_resnet1d_vae_noaug_k500_v7_sjr_rgq.yaml", 4),
        ("benchmark_inception1d_vae_noaug_k500_v7_sjr_rgq.yaml", 4),
        ("benchmark_fcn_wang_vae_noaug_k500_v7_sjr_rgq.yaml", 4),
        ("ecgfounder_direct_k500_v6.yaml", 1),
        ("ecgfounder_inithead_fullft_k500_v6.yaml", 4),
        ("ecgfounder_inithead_fullft_k500_v6_smoke.yaml", 4),
        ("ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml", 4),
        ("ecgfounder_vae_lhat_k500_v6.yaml", 4),
        ("ecgfounder_vae_lhat_k500_v6_smoke.yaml", 4),
        ("ecgtwin_author_ibe_repro.yaml", 1),
        ("ecgtwin_author_dit_repro.yaml", 1),
        ("ecgtwin_prompt_token_train_minimal.yaml", 1),
        ("ecgtwin_prompt_token_generate_minimal.yaml", 1),
        ("ecgtwin_prompt_token_gate_minimal.yaml", 1),
        ("ecgtwin_prompt_token_online_at_minimal.yaml", 1),
        ("ecgtwin_prompt_token_online_at_minimal_pn2021_eval.yaml", 1),
        ("pn2021_eval_v6_refexcluded.yaml", 4),
        ("pn2021_eval_v6_refexcluded_smoke.yaml", 4),
        ("pn2021_eval_v7_sjr_rgq_refexcluded.yaml", 4),
        ("pn2021c_effnet_v7_augmix_vs_noaug.yaml", 8),
        ("pn2021c_official_s5_locked_protocol.yaml", 8),
        ("pn2021c_effnet_threechain_locked_official_s5.yaml", 4),
        ("pn2021c_ecgfounder_threechain_locked_official_s5.yaml", 4),
        ("pn2021c_effnet_v7_strong_10to20pp.yaml", 8),
        ("pn2021c_effnet_v7_strong_10to20pp_smoke.yaml", 2),
        ("pn2021c_effnet_v7_strong_raw_candidates.yaml", 20),
        ("pn2021c_effnet_v7_strong_raw_candidates_smoke.yaml", 5),
        ("pn2021c_effnet_v7_strong_maskshift_ningbo.yaml", 2),
        ("pn2021c_effnet_v7_strong_rawsupervised_ningbo.yaml", 2),
        ("pn2021c_effnet_v7_strong_calibrated_rawsupervised.yaml", 8),
        ("pn2021c_effnet_v7_strong_calibrated_rawsupervised_ningbo.yaml", 2),
        ("pn2021c_effnet_dual3ch_cpsc_dualmodel_smoke.yaml", 11),
        ("pn2021c_effnet_dual3ch_selected_dualmodel_missing3.yaml", 15),
        ("pn2021c_effnet_dual3ch_selected_dualmodel_cpsc.yaml", 5),
        ("pn2021c_effnet_dual3ch_selected_official_s5_missing3.yaml", 15),
        ("pn2021c_effnet_dual3ch_selected_official_s5_cpsc.yaml", 5),
    ],
)
def test_tracked_configs_validate_and_expand_commands(config_name: str, expected_commands: int):
    config = _load(config_name)
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert paths["project_root"].endswith("ECG_adv_Gen")
    assert config["paper_protocol"]["mapping_version"] == "v7_super5_sjr_rgq_review_20260528"
    assert config["paper_protocol"]["mapping_hash"] == "555ec85d5b51"
    assert "pn2021_all_zero_kept_refexcluded" in config["evaluation"]["views"]
    assert "pn2021_drop_all_zero_refexcluded" in config["evaluation"]["views"]
    assert len(commands) == expected_commands
    for command in commands:
        assert command["cwd"] == paths["project_root"]
        assert command["argv"][0].startswith("/home/linbinhao/")
        assert command["argv"][1].endswith(".py")
        assert command["env"]["TMPDIR"] == paths["short_tmp_root"]
        assert len(command["env"]["TMPDIR"]) < len(paths["tmp_root"])


def test_logging_artifacts_are_split_by_lifecycle():
    config = _load("effnet_direct_k500_v6.yaml")
    logging_cfg = config["logging"]

    assert "launch_artifacts" in logging_cfg
    assert "child_artifacts" in logging_cfg
    assert "postprocess_artifacts" in logging_cfg
    assert "k500_ref_ids.json" in logging_cfg["launch_artifacts"]
    assert "data_manifest.json" in logging_cfg["launch_artifacts"]
    assert "selection.json" in logging_cfg["launch_artifacts"]
    assert "run_card.json" in logging_cfg["launch_artifacts"]
    assert "run_file_index.json" in logging_cfg["launch_artifacts"]
    assert "summary.md" in logging_cfg["launch_artifacts"]
    assert "metrics_long.csv" in logging_cfg["postprocess_artifacts"]
    assert "metrics_long.csv" not in logging_cfg["launch_artifacts"]
    assert "selection.json" not in logging_cfg["postprocess_artifacts"]


def test_pipeline_stages_are_recorded_in_dry_run_manifest():
    config = _load("ecgtwin_prompt_token_online_at_minimal.yaml")
    config = copy.deepcopy(config)
    config["stages"] = [
        {
            "name": "train_prompt_token_bank",
            "produces": ["prompt_token_bank"],
            "skip_if_exists": ["${paths.output_root}/prompt_token_bank.pt"],
        },
        {
            "name": "online_at",
            "requires": ["train_prompt_token_bank"],
            "produces": ["online_at_checkpoint", "online_at_metrics"],
        },
    ]
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_pipeline_stages",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert manifest["pipeline_stages"]["schema_version"] == 1
    assert manifest["pipeline_stages"]["stages"] == [
        {
            "index": 0,
            "name": "train_prompt_token_bank",
            "requires": [],
            "produces": ["prompt_token_bank"],
            "skip_if_exists": ["${paths.output_root}/prompt_token_bank.pt"],
        },
        {
            "index": 1,
            "name": "online_at",
            "requires": ["train_prompt_token_bank"],
            "produces": ["online_at_checkpoint", "online_at_metrics"],
            "skip_if_exists": [],
        },
    ]
    assert manifest["artifact_trace"]["pipeline_stages"] == manifest["pipeline_stages"]


def test_pipeline_stages_reject_unknown_dependencies():
    config = _load("ecgtwin_prompt_token_online_at_minimal.yaml")
    config = copy.deepcopy(config)
    config["stages"] = [
        {
            "name": "online_at",
            "requires": ["missing_prompt_token_bank"],
            "produces": ["online_at_checkpoint"],
        }
    ]

    with pytest.raises(ConfigError, match="unknown required stage"):
        validate_experiment_config(config, repo_root=REPO)


def test_pipeline_stages_reject_duplicate_names():
    config = _load("ecgtwin_prompt_token_online_at_minimal.yaml")
    config = copy.deepcopy(config)
    config["stages"] = [
        {"name": "online_at", "produces": ["ckpt"]},
        {"name": "online_at", "produces": ["metrics"]},
    ]

    with pytest.raises(ConfigError, match="duplicate stage name"):
        validate_experiment_config(config, repo_root=REPO)


def test_dry_run_manifest_preserves_yaml_run_record_metadata():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["run_record"] = {
        "purpose": "Verify YAML-managed run-record metadata survives dry-run planning.",
        "result_summary": "The dry-run metadata path recorded the intended summary.",
        "registration_status": "provisional",
    }
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_run_record_metadata",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert manifest["experiment"]["name"] == "effnet_direct_k500_v7_sjr_rgq"
    assert manifest["experiment"]["description"].startswith("EfficientNet1DV2 direct K500")
    assert manifest["run_record"] == {
        "purpose": "Verify YAML-managed run-record metadata survives dry-run planning.",
        "result_summary": "The dry-run metadata path recorded the intended summary.",
        "registration_status": "provisional",
    }


def test_experiment_schema_rejects_missing_logging_block():
    config = _load("effnet_direct_k500_v6.yaml")
    config = copy.deepcopy(config)
    del config["logging"]

    with pytest.raises(ConfigError, match="Schema validation failed"):
        validate_experiment_config(config, repo_root=REPO)


def test_experiment_schema_rejects_bad_postprocess_artifact_shape():
    config = _load("effnet_direct_k500_v6.yaml")
    config = copy.deepcopy(config)
    del config["postprocess"]["commands"][0]["expected_artifacts"][0]["path"]

    with pytest.raises(ConfigError, match="Schema validation failed"):
        validate_experiment_config(config, repo_root=REPO)


def test_local_config_cannot_override_paper_or_runner_sections(tmp_path: Path):
    local = yaml.safe_load(LOCAL_EXAMPLE.read_text(encoding="utf-8"))
    local["paper_protocol"] = {"mapping_hash": "unsafe-local-override"}
    local["runner"] = {"entrypoint": "scripts/run_experiment.py"}
    local["evaluation"] = {"views": ["unsafe_local_view"]}
    local_path = tmp_path / "unsafe_local.yaml"
    local_path.write_text(yaml.safe_dump(local, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="Local config may only define"):
        load_experiment_config(
            REPO / "configs" / "experiments" / "effnet_direct_k500_v6.yaml",
            local_path,
            runtime_context={"run_id": "pytest_run"},
        )


def test_runner_protocol_audit_rejects_unknown_entrypoint_even_if_it_exists():
    config = _load("effnet_direct_k500_v6.yaml")
    config = copy.deepcopy(config)
    config["runner"]["entrypoint"] = "scripts/run_experiment.py"
    config["runner"]["argv"] = []
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="not in the managed runner allowlist"):
        build_runner_commands(config)


def test_vae_configs_declare_required_epoch_metrics():
    effnet = _load("effnet_vae_lhat_k500_v6.yaml")
    ecgfounder = _load("ecgfounder_vae_lhat_k500_v6.yaml")

    effnet_metrics = set(effnet["logging"]["required_epoch_metrics"])
    ecgfounder_metrics = set(ecgfounder["logging"]["required_epoch_metrics"])
    assert {"asr_overall", "sample_any_positive_below_0p5_asr", "latent_augmix_stats"} <= effnet_metrics
    assert {"attack_success.success_rate", "attack_vs_anchor.success_rate", "ptbxl_macro_auprc"} <= ecgfounder_metrics


def _option_value(argv: list[str], option: str) -> str:
    idx = argv.index(option)
    return argv[idx + 1]


def _option_value_or(argv: list[str], option: str, default: str | None = None) -> str | None:
    return _option_value(argv, option) if option in argv else default


def _all_option_values(argv: list[str], option: str) -> list[str]:
    idx = argv.index(option) + 1
    values: list[str] = []
    while idx < len(argv) and not argv[idx].startswith("--"):
        values.append(argv[idx])
        idx += 1
    return values


def _remove_option_pair(argv: list[str], option: str) -> list[str]:
    idx = argv.index(option)
    return argv[:idx] + argv[idx + 2:]


def test_config_loader_import_does_not_load_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import ecg_adv_gen.config.loader; "
                "print('torch_loaded', 'torch' in sys.modules)"
            ),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == "torch_loaded False"


def test_effnet_vae_lhat_command_is_protocol_equivalent():
    config = _load("effnet_vae_lhat_k500_v6.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        assert "--anchor_base" in argv
        assert "seed20260531" in _option_value(argv, "--anchor_base")
        assert "seed42" not in _option_value(argv, "--anchor_base")
        assert "--init_ckpt" in argv
        center = _option_value(argv, "--center")
        assert _option_value(argv, "--init_ckpt").endswith(
            f"/{center}_K500_direct_ft_ep30_seed20260531_val0.2/best_model.pt"
        )
        assert _option_value(argv, "--hull_lambda") == "0.05"
        assert _option_value(argv, "--hull_label_lambda_y") == "0.25"
        assert _option_value(argv, "--hull_label_new_class_cap") == "0.25"
        assert _option_value(argv, "--hull_neighbor_distance_space") == "standardized"
        assert _option_value(argv, "--hull_neighbor_mode") == "local_random"
        assert _option_value(argv, "--hull_neighbor_pool_size") == "120"
        assert _option_value(argv, "--k_anchor") == "300"
        assert _option_value(argv, "--pgd_batch") == "32"
        assert _option_value(argv, "--adv_weight") == "0.3"
        assert _option_value(argv, "--adv_weight_warmup_epochs") == "10"
        assert _option_value(argv, "--adv_teacher_mix") == "0.4"
        assert _option_value(argv, "--train_batch_size") == "128"
        assert _option_value(argv, "--latent_augmix_latent_weight_cap") == "0.25"
        assert _option_value(argv, "--quick_eval_source") == "target_real_val"
        assert _option_value(argv, "--target_real_val_fraction") == "0.2"
        assert _option_value(argv, "--eval_batch_size") == "192"
        assert _option_value(argv, "--eval_min_pos") == "10"
        assert _option_value(argv, "--eval_pn2021_limit") == "0"


def test_effnet_raw_corrupt_input_stabilizer_config_maps_to_legacy_flags():
    config = _load("effnet_vae_lhat_calibrated_rawsupervised_k500_v7_sjr_rgq_smoke.yaml")
    config = copy.deepcopy(config)
    config["adaptation"]["raw_corrupt_consistency"]["input_stabilizer"] = {
        "bandpass_low_hz": 0.5,
        "bandpass_high_hz": 35.0,
        "repair_flat_leads": True,
        "renorm_after_stabilizer": True,
        "sample_rate_hz": 100.0,
    }
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert _option_value(argv, "--raw_input_bandpass_low_hz") == "0.5"
    assert _option_value(argv, "--raw_input_bandpass_high_hz") == "35.0"
    assert "--raw_input_repair_flat_leads" in argv
    assert "--raw_input_renorm_after_stabilizer" in argv
    assert _option_value(argv, "--raw_input_sample_rate_hz") == "100.0"


@pytest.mark.parametrize(
    ("config_name", "output_family"),
    [
        (
            "benchmark_resnet1d_vae_noaug_k500_v7_sjr_rgq.yaml",
            "benchmark_resnet1d_wang_vae_noaug_k500_v7_sjr_rgq",
        ),
        (
            "benchmark_inception1d_vae_noaug_k500_v7_sjr_rgq.yaml",
            "benchmark_inception1d_vae_noaug_k500_v7_sjr_rgq",
        ),
        (
            "benchmark_fcn_wang_vae_noaug_k500_v7_sjr_rgq.yaml",
            "benchmark_fcn_wang_vae_noaug_k500_v7_sjr_rgq",
        ),
    ],
)
def test_benchmark_vae_noaug_configs_disable_latent_augmix_branch(
    config_name: str,
    output_family: str,
):
    config = _load(config_name)
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        model_name = _option_value(argv, "--model_name")
        assert argv[1].endswith("scripts/paper/run_effnet_latent_augmix_stage3_20260524.py")
        assert _option_value(argv, "--center") == center
        assert _option_value(argv, "--out_root").endswith(f"/runs/{output_family}/pytest_run")
        assert f"/{model_name}_direct_k500_v7_sjr_rgq/pytest_run/{center}/runs/" in _option_value(
            argv,
            "--init_ckpt",
        )
        assert "--disable_latent_augmix_branch" in argv
        assert _option_value(argv, "--run_tag_extra") == "noaugmix"
        assert "--latent_augmix_latent_weight_cap" not in argv
        assert "--latent_augmix_width" not in argv
        assert "--latent_augmix_depth" not in argv
        assert "--latent_augmix_alpha" not in argv
        assert "--latent_augmix_severity" not in argv


def test_command_protocol_audit_rejects_missing_vae_init_checkpoint():
    config = _load("effnet_vae_lhat_k500_v6.yaml")
    config = copy.deepcopy(config)
    config["runner"]["argv"] = _remove_option_pair(config["runner"]["argv"], "--init_ckpt")
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="missing required option --init_ckpt"):
        build_runner_commands(config)


def test_runner_audit_dispatches_eval_crosscenter_command():
    command = {
        "argv": [
            "python",
            "scripts/triple_labels/eval_crosscenter.py",
            "--scheme",
            "super5",
            "--model_dir",
            "/home/linbinhao/outputs/model",
            "--model_name",
            "efficientnet1dv2",
            "--device",
            "cuda",
            "--crop_len",
            "1000",
            "--batch_size",
            "192",
            "--num_workers",
            "0",
            "--ptbxl_csv",
            "/home/linbinhao/data/ptbxl_database.csv",
            "--ptbxl_cache",
            "/home/linbinhao/data/ptbxl_cache.npy",
            "--preprocess_mode",
            "minimal_resample",
            "--norm_mode",
            "per_sample_global",
            "--pn2021_root",
            "/home/linbinhao/data/physionet2021",
            "--pn2021_cache_dir",
            "/home/linbinhao/cache/pn2021",
            "--pn2021_mmap_cache_dir",
            "/home/linbinhao/cache/pn2021_mmap",
            "--skip_mimic",
            "--report_drop_all_zero_pn2021",
            "--eval_protocol",
            "paper_refexcluded",
            "--exclude_ref_ids",
            "/home/linbinhao/subsets/ningbo_real_k500_seed1.ref_meta.json",
            "/home/linbinhao/subsets/chapman_shaoxing_real_k500_seed1.ref_meta.json",
            "/home/linbinhao/subsets/cpsc_2018_real_k500_seed1.ref_meta.json",
            "/home/linbinhao/subsets/georgia_real_k500_seed1.ref_meta.json",
            "--output_path",
            "/home/linbinhao/runs/pytest_run/ningbo/eval.json",
        ],
        "matrix": {"center": "ningbo"},
    }
    config = {
        "paper_protocol": {
            "kshot": {"k": 500, "seed": 1},
            "centers": {"target_4": ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]},
        },
        "preprocess": {"crop_len": 1000, "mode": "minimal_resample", "norm_mode": "per_sample_global"},
        "evaluation": {"min_pos": 10},
        "runtime": {"run_id": "pytest_run"},
    }
    report = audit_runner_command(command, config=config)

    assert report["errors"] == []


def test_effnet_vae_lhat_command_audit_is_split_into_adapter():
    from ecg_adv_gen.config.adapters.effnet_vae_lhat import audit_effnet_vae_lhat_command

    source = inspect.getsource(audit_runner_commands)
    assert "audit_effnet_vae_lhat_command(" in source
    assert "anchor_base does not encode K" not in source

    config = _load("effnet_vae_lhat_k500_v6.yaml")
    command = build_runner_commands(config)[0]
    result = audit_effnet_vae_lhat_command(
        command,
        expected_k=500,
        expected_seed=20260531,
        target_centers=set(config["paper_protocol"]["centers"]["target_4"]),
    )
    assert result == {"errors": [], "warnings": []}

    bad = copy.deepcopy(command)
    bad["argv"] = list(bad["argv"])
    bad["argv"][bad["argv"].index("--hull_neighbor_mode") + 1] = "heldout_oracle"
    result = audit_effnet_vae_lhat_command(
        bad,
        expected_k=500,
        expected_seed=20260531,
        target_centers=set(config["paper_protocol"]["centers"]["target_4"]),
    )
    assert "invalid --hull_neighbor_mode" in "\n".join(result["errors"])


def test_effnet_vae_lhat_typed_adapter_matches_legacy_runner_argv():
    legacy = _load("effnet_vae_lhat_k500_v6.yaml")
    adapted = copy.deepcopy(legacy)
    adapted["runner"]["adapter"] = "effnet_vae_lhat"
    adapted["runner"].pop("argv")
    validate_experiment_config(adapted, repo_root=REPO)

    legacy_commands = build_runner_commands(legacy)
    adapted_commands = build_runner_commands(adapted)

    assert adapted_commands == legacy_commands


def test_effnet_vae_lhat_typed_adapter_honors_ptbxl_weight_override():
    config = _load("effnet_vae_lhat_augmix_threechain_dualmodel_cand15_freq_cpsc.yaml")
    config = copy.deepcopy(config)
    config["adaptation"]["loss"]["ptbxl_weight"] = 0.0
    validate_experiment_config(config, repo_root=REPO)

    commands = build_runner_commands(config)

    assert len(commands) == 1
    assert _option_value(commands[0]["argv"], "--ptbxl_weight") == "0.0"


def test_effnet_vae_lhat_v7_config_uses_typed_runner_adapter():
    config = _load("effnet_vae_lhat_k500_v7_sjr_rgq.yaml")

    assert config["runner"]["adapter"] == "effnet_vae_lhat"
    assert "argv" not in config["runner"]
    commands = build_runner_commands(config)
    assert len(commands) == 4
    assert all(command["argv"][1].endswith("scripts/paper/run_effnet_latent_augmix_stage3_20260524.py") for command in commands)


def test_effnet_vae_lhat_rawjsd_curriculum_config_exposes_raw_corruption_flags():
    config = _load("effnet_vae_lhat_rawjsd_curriculum_k500_v7_sjr_rgq.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        assert "--enable_raw_corrupt_consistency" in argv
        assert _option_value(argv, "--raw_corrupt_consistency_loss") == "jsd"
        assert _option_value(argv, "--raw_corrupt_copies") == "2"
        assert _option_value(argv, "--raw_corrupt_prob") == "0.75"
        assert _option_value(argv, "--raw_corrupt_severity") == "5"
        assert _option_value(argv, "--raw_corrupt_severity_profile") == "standard"
        assert _option_value(argv, "--raw_corrupt_consistency_weight") == "8.0"
        assert _option_value(argv, "--raw_corrupt_bce_weight") == "0.05"
        assert _option_value(argv, "--raw_corrupt_max_batches") == "8"
        assert _option_value(argv, "--raw_corrupt_scope") == "target"
        assert "--raw_corrupt_no_renorm" in argv
        assert _all_option_values(argv, "--latent_augmix_ops") == [
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
            "random_leads_masking",
        ]
        assert _all_option_values(argv, "--raw_corrupt_ops") == [
            "baseline_shift",
            "random_leads_masking",
            "baseline_shift",
            "random_leads_masking",
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
        ]
        assert _option_value(argv, "--run_tag_extra") == "k500_rawjsd"


def test_effnet_vae_lhat_rawsupervised_stressor_config_exposes_source_target_flags():
    config = _load("effnet_vae_lhat_rawsupervised_stressor_k500_v7_sjr_rgq.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        assert "--enable_raw_corrupt_consistency" in argv
        assert _option_value(argv, "--raw_corrupt_consistency_loss") == "jsd"
        assert _option_value(argv, "--raw_corrupt_copies") == "2"
        assert _option_value(argv, "--raw_corrupt_prob") == "1.0"
        assert _option_value(argv, "--raw_corrupt_severity") == "5"
        assert _option_value(argv, "--raw_corrupt_severity_profile") == "standard"
        assert _option_value(argv, "--raw_corrupt_consistency_weight") == "2.0"
        assert _option_value(argv, "--raw_corrupt_bce_weight") == "1.0"
        assert _option_value(argv, "--raw_corrupt_max_batches") == "64"
        assert _option_value(argv, "--raw_corrupt_scope") == "source_target"
        assert "--raw_corrupt_no_renorm" in argv
        assert _all_option_values(argv, "--raw_corrupt_ops") == [
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
            "random_leads_masking",
        ]
        assert _option_value(argv, "--run_tag_extra") == "k500_rawsupervised"


def test_effnet_vae_lhat_calibrated_rawsupervised_config_exposes_profile_flag():
    config = _load("effnet_vae_lhat_calibrated_rawsupervised_k500_v7_sjr_rgq.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        assert "--enable_raw_corrupt_consistency" in argv
        assert _option_value(argv, "--raw_corrupt_severity") == "5"
        assert _option_value(argv, "--raw_corrupt_severity_profile") == "calibrated_10to20pp"
        assert _option_value(argv, "--raw_corrupt_scope") == "source_target"
        assert _option_value(argv, "--run_tag_extra") == "k500_calrawsupervised"


def test_effnet_vae_lhat_calibrated_latent_augmix_config_exposes_profile_flag():
    config = _load("effnet_vae_lhat_calibrated_latent_augmix_k500_v7_sjr_rgq_cpsc_2018.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert _option_value(argv, "--center") == "cpsc_2018"
    assert "--enable_raw_corrupt_consistency" not in argv
    assert _option_value(argv, "--seed") == "20260601"
    assert _option_value(argv, "--latent_augmix_topology") == "locked_three_chain"
    assert _option_value(argv, "--latent_augmix_severity") == "5"
    assert _option_value(argv, "--latent_augmix_severity_profile") == "calibrated_10to20pp"
    assert _option_value(argv, "--run_tag_extra") == "k500_callatentaugmix"
    assert _all_option_values(argv, "--latent_augmix_ops") == [
        "powerline_noise",
        "emg_noise",
        "baseline_wander",
        "baseline_shift",
        "random_leads_masking",
    ]


def test_effnet_vae_lhat_calibrated_latent_augmix_norenorm_config_exposes_flag():
    config = _load("effnet_vae_lhat_calibrated_latent_augmix_norenorm_k500_v7_sjr_rgq_cpsc_2018.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert _option_value(argv, "--center") == "cpsc_2018"
    assert _option_value(argv, "--latent_augmix_severity_profile") == "calibrated_10to20pp"
    assert "--no_latent_augmix_renorm" in argv
    assert _option_value(argv, "--run_tag_extra") == "k500_callatentaugmix_norenorm"


def test_effnet_vae_lhat_calibrated_latent_augmix_directloss_config_exposes_flags():
    config = _load("effnet_vae_lhat_calibrated_latent_augmix_directloss_k500_v7_sjr_rgq_cpsc_2018.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert _option_value(argv, "--center") == "cpsc_2018"
    assert "--enable_raw_corrupt_consistency" not in argv
    assert _option_value(argv, "--latent_augmix_copies") == "2"
    assert _option_value(argv, "--latent_augmix_severity_profile") == "calibrated_10to20pp"
    assert "--enable_latent_augmix_consistency" in argv
    assert _option_value(argv, "--latent_augmix_consistency_loss") == "jsd"
    assert _option_value(argv, "--latent_augmix_consistency_weight") == "2.0"
    assert _option_value(argv, "--latent_augmix_bce_weight") == "1.0"
    assert _option_value(argv, "--run_tag_extra") == "k500_callatentaugmix_directloss"


def test_effnet_vae_lhat_threechain_locked_k500_config_uses_official_s5_last_checkpoint():
    config = _load("effnet_vae_lhat_augmix_threechain_locked_k500.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_effnet_threechain_locked",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 4
    refs = manifest["artifact_trace"]["inputs"]["k500_refs"]
    assert len(refs) == 4
    assert all(item["signals_npz"]["role"] == "kshot_raw1000_signals" for item in refs)
    assert all(item["signals_npz"]["path"].endswith(".raw1000.npz") for item in refs)
    for command in commands:
        argv = command["argv"]
        assert "--enable_raw_corrupt_consistency" not in argv
        assert _option_value(argv, "--latent_augmix_topology") == "locked_three_chain"
        assert _option_value(argv, "--latent_augmix_copies") == "2"
        assert _option_value(argv, "--latent_augmix_width") == "3"
        assert _option_value(argv, "--latent_augmix_severity") == "5"
        assert _option_value(argv, "--latent_augmix_severity_profile") == "standard"
        assert _option_value(argv, "--checkpoint_policy") == "last"
        assert _option_value(argv, "--quick_eval_source") == "none"
        assert _option_value(argv, "--target_real_val_fraction") == "0.0"
        assert _option_value(argv, "--target_real_norm_mode") == "per_sample_global"
        assert _option_value(argv, "--target_real_npz_override").endswith(
            f"/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/{command['matrix']['center']}/"
            f"k500_seed20260601/{command['matrix']['center']}_real_k500_seed20260601.raw1000.npz"
        )
        assert _option_value(argv, "--run_tag_extra") == "k500_threechain_s5_locked"
        assert _all_option_values(argv, "--latent_augmix_ops") == [
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
            "random_leads_masking",
        ]


@pytest.mark.parametrize(
    ("config_name", "center", "copies", "mix_prob", "hull_steps", "hull_lambda", "run_tag_extra"),
    [
        (
            "effnet_vae_lhat_augmix_threechain_dualmodel_cand09_highfreq_ningbo.yaml",
            "ningbo",
            "6",
            "0.95",
            "6",
            "0.12",
            "k500_dual3ch_highfreq_hlam12_hs6_lr40_c6_cw10_bce3_m095_aw2p0_ka1600_wlat65_b64",
        ),
        (
            "effnet_vae_lhat_augmix_threechain_dualmodel_cand10_advheavy_georgia.yaml",
            "georgia",
            "4",
            "0.9",
            "8",
            "0.2",
            "k500_dual3ch_advheavy_hlam20_hs8_lr50_c4_cw10_bce3_m090_aw3p0_ka1800_treal40",
        ),
    ],
)
def test_effnet_vae_lhat_dualmodel_selected_configs_preserve_locked_threechain_mainline(
    config_name: str,
    center: str,
    copies: str,
    mix_prob: str,
    hull_steps: str,
    hull_lambda: str,
    run_tag_extra: str,
):
    config = _load(config_name)
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 1
    command = commands[0]
    argv = command["argv"]
    assert command["matrix"]["center"] == center
    assert _option_value(argv, "--center") == center
    assert "--enable_raw_corrupt_consistency" not in argv
    assert "--raw_input_stabilizer" not in argv
    assert "--raw_corrupt_view_mode" not in argv
    assert _option_value(argv, "--latent_augmix_topology") == "locked_three_chain"
    assert _option_value(argv, "--latent_augmix_width") == "3"
    assert _option_value(argv, "--latent_augmix_depth") == "1"
    assert _option_value(argv, "--latent_augmix_copies") == copies
    assert _option_value(argv, "--latent_augmix_mixture_mode") == "fixed"
    assert _option_value(argv, "--latent_augmix_mixture_prob") == mix_prob
    if "opcycle" in config_name:
        assert _option_value(argv, "--latent_augmix_op_schedule") == "per_op"
        assert _option_value(argv, "--latent_augmix_chain_weights") == "0.45,0.45,0.10"
    assert _option_value(argv, "--latent_augmix_severity") == "5"
    assert _option_value(argv, "--latent_augmix_severity_profile") == "custom"
    assert _option_value(argv, "--latent_augmix_severity_params_file") == (
        "configs/corruption_profiles/pn2021c_dual_model_10to15pp_v1.yaml"
    )
    assert _option_value(argv, "--latent_augmix_severity_params_name") == "dual_model_10to15pp_v1"
    assert "--enable_latent_augmix_consistency" in argv
    assert _option_value(argv, "--latent_augmix_consistency_loss") == "jsd"
    assert _option_value(argv, "--latent_augmix_consistency_weight") == "10.0"
    assert _option_value(argv, "--latent_augmix_bce_weight") == "3.0"
    assert _option_value(argv, "--latent_augmix_consistency_max_batches") == "0"
    assert _option_value(argv, "--hull_steps") == hull_steps
    assert _option_value(argv, "--hull_lambda") == hull_lambda
    assert _option_value(argv, "--checkpoint_policy") == "last"
    assert _option_value(argv, "--target_real_val_fraction") == "0.0"
    assert _option_value(argv, "--run_tag_extra") == run_tag_extra
    assert _all_option_values(argv, "--latent_augmix_ops") == [
        "powerline_noise",
        "emg_noise",
        "baseline_wander",
        "baseline_shift",
        "random_leads_masking",
    ]


def test_effnet_vae_lhat_dualmodel_cand11_opcycle_config_preserves_mainline_matrix():
    config = _load("effnet_vae_lhat_augmix_threechain_dualmodel_cand11_opcycle_rawheavy.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    assert {command["matrix"]["center"] for command in commands} == {
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    }
    for command in commands:
        argv = command["argv"]
        assert "--enable_raw_corrupt_consistency" not in argv
        assert "--raw_input_stabilizer" not in argv
        assert "--raw_corrupt_view_mode" not in argv
        assert _option_value(argv, "--latent_augmix_topology") == "locked_three_chain"
        assert _option_value(argv, "--latent_augmix_width") == "3"
        assert _option_value(argv, "--latent_augmix_depth") == "1"
        assert _option_value(argv, "--latent_augmix_copies") == "5"
        assert _option_value(argv, "--latent_augmix_mixture_mode") == "fixed"
        assert _option_value(argv, "--latent_augmix_mixture_prob") == "1.0"
        assert _option_value(argv, "--latent_augmix_op_schedule") == "per_op"
        assert _option_value(argv, "--latent_augmix_chain_weights") == "0.45,0.45,0.10"
        assert _option_value(argv, "--latent_augmix_severity_profile") == "custom"
        assert _option_value(argv, "--latent_augmix_severity_params_name") == "dual_model_10to15pp_v1"
        assert "--enable_latent_augmix_consistency" in argv
        assert _option_value(argv, "--latent_augmix_consistency_weight") == "12.0"
        assert _option_value(argv, "--latent_augmix_bce_weight") == "4.0"
        assert _option_value(argv, "--run_tag_extra") == "opcyraw_c5_w4510_cw12_b4_m1"


def test_effnet_vae_lhat_dualmodel_cand12_rawspace_config_preserves_mainline_matrix():
    config = _load("effnet_vae_lhat_augmix_threechain_dualmodel_cand12_rawspace_advbce.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    assert {command["matrix"]["center"] for command in commands} == {
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    }
    for command in commands:
        argv = command["argv"]
        assert "--enable_raw_corrupt_consistency" not in argv
        assert "--raw_input_stabilizer" not in argv
        assert "--raw_corrupt_view_mode" not in argv
        assert _option_value(argv, "--latent_augmix_topology") == "locked_three_chain"
        assert _option_value(argv, "--latent_augmix_signal_space") == "raw_pre_zscore"
        assert _option_value(argv, "--latent_augmix_op_schedule") == "per_op"
        assert _option_value(argv, "--latent_augmix_chain_weights") == "0.35,0.35,0.30"
        assert _option_value(argv, "--latent_augmix_copies") == "5"
        assert _option_value(argv, "--latent_augmix_mixture_mode") == "fixed"
        assert _option_value(argv, "--latent_augmix_mixture_prob") == "1.0"
        assert _option_value(argv, "--latent_augmix_severity_profile") == "custom"
        assert _option_value(argv, "--latent_augmix_severity_params_name") == "dual_model_10to15pp_v1"
        assert "--enable_latent_augmix_consistency" in argv
        assert _option_value(argv, "--latent_augmix_consistency_weight") == "8.0"
        assert _option_value(argv, "--latent_augmix_bce_weight") == "8.0"
        assert _option_value(argv, "--hull_steps") == "8"
        assert _option_value(argv, "--hull_lambda") == "0.15"
        assert _option_value(argv, "--run_tag_extra") == "rawspace_c5_w353530_cw8_b8_aw3"


def test_effnet_vae_lhat_dualmodel_cand13_targetreal_config_preserves_mainline_matrix():
    config = _load("effnet_vae_lhat_augmix_threechain_dualmodel_cand13_targetrealraw.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    assert {command["matrix"]["center"] for command in commands} == {
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    }
    for command in commands:
        argv = command["argv"]
        assert "--enable_raw_corrupt_consistency" not in argv
        assert "--raw_input_stabilizer" not in argv
        assert "--raw_corrupt_view_mode" not in argv
        assert _option_value(argv, "--latent_augmix_topology") == "locked_three_chain"
        assert _option_value(argv, "--latent_augmix_signal_space") == "raw_pre_zscore"
        assert _option_value(argv, "--latent_augmix_corruption_source") == "target_real"
        assert _option_value(argv, "--latent_augmix_op_schedule") == "per_op"
        assert _option_value(argv, "--latent_augmix_chain_weights") == "0.45,0.45,0.10"
        assert _option_value(argv, "--latent_augmix_copies") == "5"
        assert _option_value(argv, "--latent_augmix_mixture_mode") == "fixed"
        assert _option_value(argv, "--latent_augmix_mixture_prob") == "1.0"
        assert _option_value(argv, "--latent_augmix_severity_profile") == "custom"
        assert _option_value(argv, "--latent_augmix_severity_params_name") == "dual_model_10to15pp_v1"
        assert "--enable_latent_augmix_consistency" in argv
        assert _option_value(argv, "--latent_augmix_consistency_weight") == "12.0"
        assert _option_value(argv, "--latent_augmix_bce_weight") == "8.0"
        assert _option_value(argv, "--hull_steps") == "8"
        assert _option_value(argv, "--hull_lambda") == "0.15"
        assert _option_value(argv, "--run_tag_extra") == "trraw_c5_w4510_cw12_b8_aw3"


def test_effnet_vae_lhat_fullpool_raw_augmix_config_exposes_flags():
    config = _load("effnet_vae_lhat_fullpool_raw_augmix_k500_v7_sjr_rgq_cpsc_2018.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert _option_value(argv, "--center") == "cpsc_2018"
    assert "--enable_raw_corrupt_consistency" in argv
    assert _option_value(argv, "--raw_corrupt_severity_profile") == "calibrated_10to20pp"
    assert _option_value(argv, "--raw_corrupt_scope") == "source_target"
    assert _option_value(argv, "--raw_corrupt_view_mode") == "augmix"
    assert _option_value(argv, "--raw_augmix_width") == "3"
    assert _option_value(argv, "--raw_augmix_depth") == "-1"
    assert _option_value(argv, "--raw_augmix_alpha") == "1.0"
    assert _option_value(argv, "--run_tag_extra") == "k500_fullpool_rawaugmix"


def test_effnet_vae_lhat_raw_augmix_config_exposes_mixture_flags():
    config = _load("effnet_vae_lhat_fullpool_raw_augmix_depth1_k500_v7_sjr_rgq_cpsc_2018.yaml")
    config = copy.deepcopy(config)
    config["adaptation"]["raw_corrupt_consistency"]["augmix"]["mixture_mode"] = "fixed"
    config["adaptation"]["raw_corrupt_consistency"]["augmix"]["mixture_prob"] = 0.75
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    argv = commands[0]["argv"]
    assert _option_value(argv, "--raw_corrupt_view_mode") == "augmix"
    assert _option_value(argv, "--raw_augmix_depth") == "1"
    assert _option_value(argv, "--raw_augmix_mixture_mode") == "fixed"
    assert _option_value(argv, "--raw_augmix_mixture_prob") == "0.75"


def test_effnet_vae_lhat_raw_augmix_w1_m100_remaining3_config_exposes_matrix():
    config = _load("effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_k500_v7_sjr_rgq_remaining3.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert [command["matrix"]["center"] for command in commands] == [
        "ningbo",
        "chapman_shaoxing",
        "georgia",
    ]
    for command in commands:
        argv = command["argv"]
        assert _option_value(argv, "--raw_corrupt_view_mode") == "augmix"
        assert _option_value(argv, "--raw_augmix_width") == "1"
        assert _option_value(argv, "--raw_augmix_depth") == "1"
        assert _option_value(argv, "--raw_augmix_mixture_mode") == "fixed"
        assert _option_value(argv, "--raw_augmix_mixture_prob") == "1.0"
        assert _option_value(argv, "--run_tag_extra") == "k500_fullpool_rawaugmix_d1_w1_m100"


def test_effnet_vae_lhat_weighted_hardops_raw_augmix_chapman_config():
    config = _load("effnet_vae_lhat_fullpool_raw_augmix_depth1_w1_m100_hardops_k500_v7_sjr_rgq_chapman.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert _option_value(argv, "--center") == "chapman_shaoxing"
    assert _option_value(argv, "--raw_corrupt_view_mode") == "augmix"
    assert _option_value(argv, "--raw_augmix_width") == "1"
    assert _option_value(argv, "--raw_augmix_depth") == "1"
    assert _option_value(argv, "--raw_augmix_mixture_mode") == "fixed"
    assert _option_value(argv, "--raw_augmix_mixture_prob") == "1.0"
    assert _all_option_values(argv, "--raw_corrupt_ops") == [
        "powerline_noise",
        "emg_noise",
        "emg_noise",
        "baseline_wander",
        "baseline_wander",
        "baseline_shift",
        "random_leads_masking",
    ]
    assert _option_value(argv, "--run_tag_extra") == "k500_fullpool_rawaugmix_d1_w1_m100_hardops"


def test_effnet_vae_lhat_maskshift_consistency_config_exposes_targeted_flags():
    config = _load("effnet_vae_lhat_maskshift_consistency_k500_v7_sjr_rgq.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        assert "--enable_mask_shift_consistency" in argv
        assert "--enable_raw_corrupt_consistency" not in argv
        assert _option_value(argv, "--mask_shift_consistency_loss") == "jsd"
        assert _option_value(argv, "--mask_shift_copies") == "1"
        assert _option_value(argv, "--mask_shift_mask_severity") == "5"
        assert _option_value(argv, "--mask_shift_shift_severity") == "5"
        assert _option_value(argv, "--mask_shift_consistency_weight") == "10.0"
        assert _option_value(argv, "--mask_shift_bce_weight") == "0.05"
        assert _option_value(argv, "--mask_shift_max_batches") == "0"
        assert _option_value(argv, "--mask_shift_scope") == "target"
        assert "--mask_shift_no_renorm" in argv
        assert _option_value(argv, "--run_tag_extra") == "k500_maskshift"


def test_effnet_vae_lhat_maskshift_smoke_config_targets_ningbo_only():
    config = _load("effnet_vae_lhat_maskshift_consistency_k500_v7_sjr_rgq_smoke.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert _option_value(argv, "--center") == "ningbo"
    assert _option_value(argv, "--epochs") == "1"
    assert _option_value(argv, "--num_workers") == "0"
    assert _option_value(argv, "--mask_shift_max_batches") == "2"


def test_runner_rejects_adapter_and_argv_together():
    config = _load("effnet_vae_lhat_k500_v7_sjr_rgq.yaml")
    config["runner"]["argv"] = ["--center", "${matrix.center}"]

    with pytest.raises(ConfigError, match="runner.*adapter.*argv"):
        build_runner_commands(config)


def test_effnet_direct_command_writes_under_managed_output_root():
    config = _load("effnet_direct_k500_v6.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    argv = commands[0]["argv"]
    assert _option_value(argv, "--k") == "500"
    assert _option_value(argv, "--subset_seed") == "20260531"
    assert _option_value(argv, "--val_fraction") == "0.2"
    assert "--out_root" in argv
    assert _option_value(argv, "--out_root").endswith("/runs/effnet_direct_k500_v6/pytest_run")


def test_effnet_v7_configs_use_v7_subset_root_and_direct_init_dependency():
    direct = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    vae = _load("effnet_vae_lhat_k500_v7_sjr_rgq.yaml")
    direct_paths = validate_experiment_config(direct, repo_root=REPO)
    vae_paths = validate_experiment_config(vae, repo_root=REPO)
    direct_command = build_runner_commands(direct)[0]
    vae_commands = build_runner_commands(vae)

    direct_subset_root = _option_value(direct_command["argv"], "--subset_root")
    assert direct_subset_root.endswith("/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets")
    assert _option_value(direct_command["argv"], "--out_root").endswith(
        "/runs/effnet_direct_k500_v7_sjr_rgq/pytest_run"
    )

    direct_manifest = make_dry_run_manifest(
        direct,
        commands=[direct_command],
        local_paths=direct_paths,
        run_id="pytest_v7_direct",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )
    assert all(
        "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets" in ref["anchor_base"]
        for ref in direct_manifest["artifact_trace"]["inputs"]["k500_refs"]
    )

    for command in vae_commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets" in _option_value(
            argv,
            "--anchor_base",
        )
        assert _option_value(argv, "--init_ckpt").endswith(
            f"/runs/effnet_direct_k500_v7_sjr_rgq_matrix/pytest_run/{center}/runs/"
            f"{center}_K500_direct_ft_ep30_seed20260531_val0.2/best_model.pt"
        )

    vae_manifest = make_dry_run_manifest(
        vae,
        commands=vae_commands,
        local_paths=vae_paths,
        run_id="pytest_v7_lhat",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )
    assert all(
        "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets" in ref["anchor_base"]
        for ref in vae_manifest["artifact_trace"]["inputs"]["k500_refs"]
    )


def test_effnet_direct_k500_v7_matrix_config_generates_per_center_commands():
    config = _load("effnet_direct_k500_v7_sjr_rgq_matrix.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_effnet_direct_v7_matrix",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 4
    assert {command["matrix"]["center"] for command in commands} == {
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    }
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert "CUDA_VISIBLE_DEVICES" not in command["env"]
        assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
        assert "OPENAI_API_KEY" not in command["env"]
        assert argv[1].endswith("scripts/paper/run_direct_finetune_k500_20260516.py")
        assert _option_value(argv, "--centers") == center
        assert _option_value(argv, "--k") == "500"
        assert _option_value(argv, "--subset_seed") == "20260531"
        assert _option_value(argv, "--seed") == "20260531"
        assert _option_value(argv, "--subset_root").endswith(
            "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
        )
        assert _option_value(argv, "--out_root").endswith(
            f"/runs/effnet_direct_k500_v7_sjr_rgq_matrix/pytest_run/{center}"
        )

    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    assert {child["matrix"]["center"] for child in child_runs} == {
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    }


def test_effnet_v7_subset_export_config_generates_percent_and_fixed_k_prep_command():
    config = _load("effnet_v7_sjr_rgq_subset_export.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_effnet_v7_subset_export",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 1
    command = commands[0]
    argv = command["argv"]
    assert argv[1].endswith("scripts/paper/export_percent_kshot_v7_sjr_rgq_20260530.py")
    assert "CUDA_VISIBLE_DEVICES" not in command["env"]
    assert "--force" not in argv
    assert _option_value(argv, "--seed") == "20260531"
    assert _option_value(argv, "--fixed-ks") == "500"
    assert _option_value(argv, "--output-root").endswith(
        "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
    )
    assert _option_value(argv, "--summary-path").endswith(
        "/runs/effnet_v7_sjr_rgq_subset_export/pytest_run/percent_shot_v7_seed20260531_summary.json"
    )
    centers = _all_option_values(argv, "--centers")
    assert centers == ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
    percents = _all_option_values(argv, "--percents")
    assert percents == ["0.1", "0.2"]

    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert child_runs == []


def test_effnet_v7_subset_export_audit_rejects_force():
    config = _load("effnet_v7_sjr_rgq_subset_export.yaml")
    config = copy.deepcopy(config)
    config["runner"]["argv"] = [*config["runner"]["argv"], "--force"]
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="managed subset export must not pass --force"):
        build_runner_commands(config)


def test_effnet_direct_percent_v7_matrix_config_generates_protocol_center_commands():
    config = _load("effnet_direct_percent_v7_sjr_rgq_matrix.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_effnet_direct_percent_v7_matrix",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    expected_cases = {
        ("ningbo", "p10", "1923"),
        ("ningbo", "p20", "3846"),
        ("chapman_shaoxing", "p10", "582"),
        ("chapman_shaoxing", "p20", "1164"),
        ("cpsc_2018", "p10", "475"),
        ("cpsc_2018", "p20", "949"),
        ("georgia", "p10", "871"),
        ("georgia", "p20", "1742"),
    }

    assert len(commands) == 8
    observed_cases = set()
    for command in commands:
        argv = command["argv"]
        case = command["matrix"]["case"]
        center = case["center"]
        protocol = case["protocol"]
        k = str(case["k"])
        observed_cases.add((center, protocol, k))

        assert "CUDA_VISIBLE_DEVICES" not in command["env"]
        assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
        assert "OPENAI_API_KEY" not in command["env"]
        assert argv[1].endswith("scripts/paper/run_direct_finetune_k500_20260516.py")
        assert _option_value(argv, "--centers") == center
        assert _option_value(argv, "--k") == k
        assert _option_value(argv, "--subset_seed") == "20260531"
        assert _option_value(argv, "--seed") == "20260531"
        assert _option_value(argv, "--subset_root").endswith(
            "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
        )
        assert _option_value(argv, "--out_root").endswith(
            f"/runs/effnet_direct_{protocol}_v7_sjr_rgq/pytest_run/{center}"
        )

    assert observed_cases == expected_cases
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 8
    assert {tuple([child["matrix"]["case"][key] for key in ("center", "protocol")]) for child in child_runs} == {
        (center, protocol) for center, protocol, _k in expected_cases
    }


def test_effnet_direct_percent_v7_matrix_audit_rejects_case_k_mismatch():
    config = _load("effnet_direct_percent_v7_sjr_rgq_matrix.yaml")
    config = copy.deepcopy(config)
    idx = config["runner"]["argv"].index("--k")
    config["runner"]["argv"][idx + 1] = "999"
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="--k='999', expected '1923'"):
        build_runner_commands(config)


def test_effnet_vae_lhat_percent_v7_matrix_config_uses_protocol_matched_direct_checkpoints():
    config = _load("effnet_vae_lhat_percent_v7_sjr_rgq.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_effnet_vae_lhat_percent_v7",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    expected_cases = {
        ("ningbo", "p10", "1923"),
        ("ningbo", "p20", "3846"),
        ("chapman_shaoxing", "p10", "582"),
        ("chapman_shaoxing", "p20", "1164"),
        ("cpsc_2018", "p10", "475"),
        ("cpsc_2018", "p20", "949"),
        ("georgia", "p10", "871"),
        ("georgia", "p20", "1742"),
    }

    assert len(commands) == 8
    observed_cases = set()
    for command in commands:
        argv = command["argv"]
        case = command["matrix"]["case"]
        center = case["center"]
        protocol = case["protocol"]
        k = str(case["k"])
        observed_cases.add((center, protocol, k))

        assert "CUDA_VISIBLE_DEVICES" not in command["env"]
        assert argv[1].endswith("scripts/paper/run_effnet_latent_augmix_stage3_20260524.py")
        assert _option_value(argv, "--center") == center
        assert _option_value(argv, "--seed") == "20260531"
        assert _option_value(argv, "--out_root").endswith(
            f"/runs/effnet_vae_lhat_{protocol}_v7_sjr_rgq/pytest_run"
        )
        assert _option_value(argv, "--init_ckpt").endswith(
            f"/runs/effnet_direct_{protocol}_v7_sjr_rgq/pytest_run/{center}/runs/"
            f"{center}_K{k}_direct_ft_ep30_seed20260531_val0.2/best_model.pt"
        )
        assert _option_value(argv, "--anchor_base").endswith(
            f"/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/"
            f"{center}/k{k}_seed20260531/{center}_real_k{k}_seed20260531"
        )
        assert _option_value(argv, "--run_tag_extra") == protocol
        assert _option_value(argv, "--quick_eval_source") == "target_real_val"

    assert observed_cases == expected_cases
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 8
    assert all(
        f"fullft_{child['matrix']['case']['protocol']}_ep30_seed20260531" in child["child_run_dir"]
        for child in child_runs
    )


def test_effnet_vae_lhat_percent_v7_matrix_audit_rejects_case_anchor_k_mismatch():
    config = _load("effnet_vae_lhat_percent_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    idx = config["runner"]["argv"].index("--anchor_base")
    config["runner"]["argv"][idx + 1] = (
        "${data.kshot_subset_root}/${matrix.case.center}/k999_seed20260531/"
        "${matrix.case.center}_real_k999_seed20260531"
    )
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="anchor_base does not encode K1923/seed20260531"):
        build_runner_commands(config)


@pytest.mark.parametrize(
    ("config_name", "model_name"),
    [
        ("benchmark_resnet1d_direct_k500_v7_sjr_rgq_matrix.yaml", "benchmark_resnet1d_wang"),
        ("benchmark_inception1d_direct_k500_v7_sjr_rgq_matrix.yaml", "benchmark_inception1d"),
        ("benchmark_fcn_wang_direct_k500_v7_sjr_rgq_matrix.yaml", "benchmark_fcn_wang"),
    ],
)
def test_benchmark_direct_k500_v7_matrix_config_generates_model_scoped_commands(
    config_name: str,
    model_name: str,
):
    config = _load(config_name)
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id=f"pytest_{model_name}_direct_k500_v7",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 4
    assert {command["matrix"]["center"] for command in commands} == {
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    }
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert "CUDA_VISIBLE_DEVICES" not in command["env"]
        assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
        assert argv[1].endswith("scripts/paper/run_benchmark_direct_finetune_v7_20260530.py")
        assert _option_value(argv, "--centers") == center
        assert _option_value(argv, "--model_name") == model_name
        assert _option_value(argv, "--k") == "500"
        assert _option_value(argv, "--subset_seed") == "20260531"
        assert _option_value(argv, "--seed") == "20260531"
        assert _option_value(argv, "--init_ckpt").endswith(
            "/runs/benchmark_source_v7_sjr_rgq/"
            f"{model_name}_seed20260531_v7_sjr_rgq_pytest_run/best_model.pt"
        )
        assert _option_value(argv, "--subset_root").endswith(
            "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
        )
        assert _option_value(argv, "--out_root").endswith(
            f"/runs/{model_name}_direct_k500_v7_sjr_rgq/pytest_run/{center}"
        )

    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    for command, child_run in zip(commands, child_runs):
        argv = command["argv"]
        center = command["matrix"]["center"]
        expected_leaf = build_benchmark_direct_run_leaf(
            {
                "center": center,
                "k": _option_value(argv, "--k"),
                "model_name": _option_value(argv, "--model_name"),
                "epochs": _option_value(argv, "--epochs"),
                "seed": _option_value(argv, "--seed"),
                "val_fraction": _option_value(argv, "--val_fraction"),
            }
        )
        assert Path(child_run["child_run_dir"]).name == expected_leaf


def test_benchmark_direct_k500_v7_audit_rejects_force():
    config = _load("benchmark_resnet1d_direct_k500_v7_sjr_rgq_matrix.yaml")
    config = copy.deepcopy(config)
    config["runner"]["argv"] = [*config["runner"]["argv"], "--force"]
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="benchmark direct managed config must not pass --force"):
        build_runner_commands(config)


def test_benchmark_source_v7_matrix_config_generates_model_commands():
    config = _load("benchmark_source_v7_sjr_rgq_matrix.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_benchmark_source_v7",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    expected_models = {
        "benchmark_resnet1d_wang",
        "benchmark_inception1d",
        "benchmark_fcn_wang",
    }

    assert len(commands) == 3
    observed_models = set()
    for command in commands:
        argv = command["argv"]
        model_name = command["matrix"]["model"]
        observed_models.add(model_name)

        assert "CUDA_VISIBLE_DEVICES" not in command["env"]
        assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
        assert "OPENAI_API_KEY" not in command["env"]
        assert argv[1].endswith("scripts/triple_labels/train_ptbxl.py")
        assert _option_value(argv, "--scheme") == "super5"
        assert _option_value(argv, "--model_name") == model_name
        assert _option_value(argv, "--output_dir").endswith(
            f"/runs/benchmark_source_v7_sjr_rgq/{model_name}_seed20260531_v7_sjr_rgq_pytest_run"
        )
        assert _option_value(argv, "--data_path").endswith("/ptbxl/raw100.npy")
        assert _option_value(argv, "--csv_path").endswith("/ptbxl/ptbxl_database.csv")
        assert _option_value(argv, "--cache_path").endswith(
            "/triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"
        )
        assert _option_value(argv, "--preprocess_mode") == "minimal_resample"
        assert _option_value(argv, "--norm_mode") == "per_sample_global"
        assert _option_value(argv, "--device") == "cuda"
        assert _option_value(argv, "--crop_len") == "1000"
        assert _option_value(argv, "--batch_size") == "128"
        assert _option_value(argv, "--epochs") == "30"
        assert _option_value(argv, "--lr") == "0.001"
        assert _option_value(argv, "--weight_decay") == "0.01"
        assert _option_value(argv, "--cosine_tmax") == "30"
        assert _option_value(argv, "--patience") == "8"
        assert _option_value(argv, "--seed") == "20260531"
        assert _option_value(argv, "--checkpoint_metric") == "auprc"
        assert _option_value(argv, "--pos_weight_clip_max") == "50.0"

    assert observed_models == expected_models
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 3
    assert {child["matrix"]["model"] for child in child_runs} == expected_models
    for child in child_runs:
        model_name = child["matrix"]["model"]
        assert child["child_run_dir"].endswith(
            f"/runs/benchmark_source_v7_sjr_rgq/{model_name}_seed20260531_v7_sjr_rgq_pytest_run"
        )
        artifact_roles = {item["role"] for item in child["expected_artifacts"]}
        assert artifact_roles >= {"best_model", "best_model_auprc", "training_log", "train_result"}


def test_benchmark_source_v7_matrix_audit_rejects_non_auprc_checkpoint_metric():
    config = _load("benchmark_source_v7_sjr_rgq_matrix.yaml")
    config = copy.deepcopy(config)
    idx = config["runner"]["argv"].index("--checkpoint_metric")
    config["runner"]["argv"][idx + 1] = "auroc"
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="benchmark source managed config must use --checkpoint_metric auprc"):
        build_runner_commands(config)


def test_ecgtwin_author_ibe_repro_config_generates_run_scoped_stage_command():
    config = _load("ecgtwin_author_ibe_repro.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_ecgtwin_author_ibe",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 1
    command = commands[0]
    argv = command["argv"]
    assert "CUDA_VISIBLE_DEVICES" not in command["env"]
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
    assert argv[1].endswith("scripts/ecgtwin_author_repro/train_ibe_repro.py")
    assert _option_value(argv, "--output_dir").endswith(
        "/runs/ecgtwin_author_repro/pytest_run/ibe_stage1"
    )
    assert _option_value(argv, "--train_path").endswith(
        "/ECGTwin_Data/paired_Mimic_vae_multi_nomic.pt"
    )
    assert _option_value(argv, "--val_path").endswith(
        "/ECGTwin_Data/paired_Mimic_vae_multi_nomic_test.pt"
    )
    assert _option_value(argv, "--epochs") == "40"
    assert _option_value(argv, "--batch_size") == "65536"
    assert _option_value(argv, "--mini_batch_size") == "512"
    assert _option_value(argv, "--lr") == "0.001"
    assert _option_value(argv, "--weight_decay") == "0.001"
    assert _option_value(argv, "--device") == "cuda"
    assert _option_value(argv, "--seed") == "20260531"
    assert "--pin_memory" in argv
    assert "--drop_last" in argv
    assert "--amp" in argv
    assert "--persistent_workers" not in argv

    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 1
    assert child_runs[0]["stage"] == "ibe_stage1"
    artifact_roles = {item["role"] for item in child_runs[0]["expected_artifacts"]}
    assert artifact_roles >= {
        "ecgtwin_author.ibe_best",
        "ecgtwin_author.latest_checkpoint",
        "legacy_run_config",
        "metrics_jsonl",
        "loss_curve_png",
    }


def test_ecgtwin_author_dit_repro_config_consumes_same_run_ibe_stage():
    config = _load("ecgtwin_author_dit_repro.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_ecgtwin_author_dit",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 1
    command = commands[0]
    argv = command["argv"]
    assert "CUDA_VISIBLE_DEVICES" not in command["env"]
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
    assert argv[1].endswith("scripts/ecgtwin_author_repro/train_dit_repro.py")
    assert _option_value(argv, "--output_dir").endswith(
        "/runs/ecgtwin_author_repro/pytest_run/dit_stage2"
    )
    assert _option_value(argv, "--ibe_path").endswith(
        "/runs/ecgtwin_author_repro/pytest_run/ibe_stage1/checkpoints/IBE_best.pth"
    )
    assert _option_value(argv, "--epochs") == "30"
    assert _option_value(argv, "--batch_size") == "512"
    assert _option_value(argv, "--val_batch_size") == "512"
    assert _option_value(argv, "--device") == "cuda"
    assert _option_value(argv, "--seed") == "20260531"
    assert "--pin_memory" in argv
    assert "--amp" in argv
    assert "--persistent_workers" not in argv
    assert "--use_pretrained_author_ibe" not in argv

    trace = manifest["artifact_trace"]
    assert any(
        item["role"] == "ecgtwin_author.ibe_stage1_best"
        and item["path"].endswith("/runs/ecgtwin_author_repro/pytest_run/ibe_stage1/checkpoints/IBE_best.pth")
        for item in trace["inputs"]["checkpoints"]
    )
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 1
    assert child_runs[0]["stage"] == "dit_stage2"
    artifact_roles = {item["role"] for item in child_runs[0]["expected_artifacts"]}
    assert artifact_roles >= {
        "ecgtwin_author.dit_best_val",
        "ecgtwin_author.dit_latest",
        "legacy_run_config",
        "metrics_jsonl",
        "loss_curve_png",
    }


def test_ecgtwin_author_dit_audit_rejects_pretrained_author_ibe_shortcut():
    config = _load("ecgtwin_author_dit_repro.yaml")
    config = copy.deepcopy(config)
    config["runner"]["argv"] = [*config["runner"]["argv"], "--use_pretrained_author_ibe"]
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="must consume the managed IBE stage"):
        build_runner_commands(config)


def test_ecgtwin_prompt_token_train_config_generates_run_scoped_command():
    config = _load("ecgtwin_prompt_token_train_minimal.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_prompt_token_train",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 1
    command = commands[0]
    argv = command["argv"]
    assert "CUDA_VISIBLE_DEVICES" not in command["env"]
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
    assert argv[1].endswith("scripts/ecgtwin_gen/train_center_prompt_tokens.py")
    assert _option_value(argv, "--save_dir").endswith(
        "/runs/ecgtwin_prompt_token_minimal/pytest_run/prompt_token_train"
    )
    assert _option_value(argv, "--cache_root").endswith("/ecgtwin_prompt_token_super5/cache_v1")
    assert _option_value(argv, "--prompt_bank").endswith(
        "/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt"
    )
    assert _option_value(argv, "--ecgtwin_config").endswith("model/ECGTwin/config/DiT_ECGTwin.yaml")
    assert _all_option_values(argv, "--centers") == [
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    ]
    assert _option_value(argv, "--K") == "500"
    assert _option_value(argv, "--seed") == "20260531"
    assert _option_value(argv, "--total_steps") == "2500"
    assert _option_value(argv, "--batch_size") == "16"
    assert _option_value(argv, "--token_mode") == "direct"
    assert _option_value(argv, "--n_token_vectors") == "4"
    assert _option_value(argv, "--ref_text_mode") == "actual_report"
    assert _option_value(argv, "--sample_strategy") == "center_class_balanced"
    assert _option_value(argv, "--device") == "cuda"

    trace = manifest["artifact_trace"]
    assert any(
        item["role"] == "prompt_token.text_prompt_bank"
        and item["path"].endswith("/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt")
        for item in trace["inputs"]["data_caches"]
    )
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 1
    assert child_runs[0]["stage"] == "prompt_token_train"
    artifact_roles = {item["role"] for item in child_runs[0]["expected_artifacts"]}
    assert artifact_roles >= {
        "prompt_token.prompt_token_bank",
        "prompt_token.run_config",
        "prompt_token.metrics_jsonl",
    }


def test_ecgtwin_prompt_token_generate_config_consumes_same_run_token_bank():
    config = _load("ecgtwin_prompt_token_generate_minimal.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_prompt_token_generate",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 1
    command = commands[0]
    argv = command["argv"]
    assert "CUDA_VISIBLE_DEVICES" not in command["env"]
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
    assert argv[1].endswith("scripts/ecgtwin_gen/generate_center_prompt_token_synth.py")
    assert _option_value(argv, "--center") == "ningbo"
    assert _all_option_values(argv, "--classes") == ["NORM", "MI", "STTC"]
    assert _option_value(argv, "--out_dir").endswith(
        "/runs/ecgtwin_prompt_token_minimal/pytest_run/generated/target_token"
    )
    assert _option_value(argv, "--token_bank").endswith(
        "/runs/ecgtwin_prompt_token_minimal/pytest_run/prompt_token_train/prompt_token_bank.pt"
    )
    assert _option_value(argv, "--cache_root").endswith("/ecgtwin_prompt_token_super5/cache_v1")
    assert _option_value(argv, "--prompt_bank").endswith(
        "/ecgtwin_prompt_token_super5/cache_v1/text_prompt_bank.pt"
    )
    assert _option_value(argv, "--victim_ckpt").endswith("/triple_labels/super5/best_model.pt")
    assert _option_value(argv, "--arm") == "target_token"
    assert _option_value(argv, "--K") == "500"
    assert _option_value(argv, "--selection_seed") == "42"
    assert _option_value(argv, "--seed") == "20260531"
    assert _option_value(argv, "--device") == "cuda"
    assert "--no_token" not in argv

    trace = manifest["artifact_trace"]
    assert any(
        item["role"] == "prompt_token.prompt_token_bank"
        and item["path"].endswith("/runs/ecgtwin_prompt_token_minimal/pytest_run/prompt_token_train/prompt_token_bank.pt")
        for item in trace["inputs"]["init_heads"]
    )
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 1
    assert child_runs[0]["stage"] == "prompt_token_generate"
    assert child_runs[0]["center"] == "ningbo"
    artifact_roles = {item["role"] for item in child_runs[0]["expected_artifacts"]}
    assert artifact_roles >= {"prompt_token.samples_npz", "prompt_token.summary_json"}


def test_ecgtwin_prompt_token_gate_config_consumes_generated_outputs():
    config = _load("ecgtwin_prompt_token_gate_minimal.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_prompt_token_gate",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 1
    command = commands[0]
    argv = command["argv"]
    assert "CUDA_VISIBLE_DEVICES" not in command["env"]
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
    assert argv[1].endswith("scripts/ecgtwin_gen/gate_prompt_token_synth.py")
    assert _option_value(argv, "--input_dir").endswith(
        "/runs/ecgtwin_prompt_token_minimal/pytest_run/generated/target_token/ningbo"
    )
    assert _option_value(argv, "--out_dir").endswith(
        "/runs/ecgtwin_prompt_token_minimal/pytest_run/generated/target_token/ningbo/gated"
    )
    assert _all_option_values(argv, "--classes") == ["NORM", "MI", "STTC"]
    assert _option_value(argv, "--min_target_prob") == "0.30"
    assert _option_value(argv, "--cache_root").endswith("/ecgtwin_prompt_token_super5/cache_v1")
    assert _option_value(argv, "--K") == "500"
    assert _option_value(argv, "--selection_seed") == "42"

    trace = manifest["artifact_trace"]
    input_roles = {item["role"] for item in trace["inputs"]["data_caches"]}
    assert {"prompt_token.generated_samples", "prompt_token.generated_summary"} <= input_roles
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 1
    assert child_runs[0]["stage"] == "prompt_token_gate"
    artifact_roles = {item["role"] for item in child_runs[0]["expected_artifacts"]}
    assert artifact_roles >= {
        "prompt_token.gated_samples_npz",
        "prompt_token.gated_latent_npz",
        "prompt_token.class_trust_json",
        "prompt_token.ref_meta_json",
        "prompt_token.gate_report_json",
    }


def test_ecgtwin_prompt_token_generate_audit_rejects_no_token_shortcut():
    config = _load("ecgtwin_prompt_token_generate_minimal.yaml")
    config = copy.deepcopy(config)
    config["runner"]["argv"] = [*config["runner"]["argv"], "--no_token"]
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="target-token managed config must not pass --no_token"):
        build_runner_commands(config)


def test_ecgtwin_prompt_token_online_at_config_consumes_gated_pool_and_real_k500_refs():
    config = _load("ecgtwin_prompt_token_online_at_minimal.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_prompt_token_online_at",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 1
    command = commands[0]
    argv = command["argv"]
    assert "CUDA_VISIBLE_DEVICES" not in command["env"]
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
    assert argv[1].endswith("scripts/pgd_cross_center/synth_online_at_super5.py")
    assert _option_value(argv, "--center_name") == "ningbo"
    assert _option_value(argv, "--model_name") == "efficientnet1dv2"
    assert _option_value(argv, "--device") == "cuda"
    assert _option_value(argv, "--quick_eval_source") == "target_real_val"
    assert _all_option_values(argv, "--quick_eval_centers") == ["ningbo"]
    assert _option_value(argv, "--target_real_val_seed") == "20260531"
    assert _option_value(argv, "--attack_mode") == "latent_hull"
    assert _option_value(argv, "--hull_label_mode") == "compatible"
    assert _option_value(argv, "--hull_mix_label_mode") == "anchor_soft"
    assert _all_option_values(argv, "--classes_in_scope") == ["NORM", "MI", "STTC"]
    assert _option_value(argv, "--adv_label_mode") == "latent_mixed_teacher"
    assert _option_value(argv, "--adv_teacher_mix") == "0.4"
    assert _option_value(argv, "--n_epochs") == "20"
    assert _option_value(argv, "--num_workers") == "4"

    gated_root = "/runs/ecgtwin_prompt_token_minimal/pytest_run/generated/target_token/ningbo/gated"
    assert _option_value(argv, "--synth_npz").endswith(f"{gated_root}/gated_samples.latent.npz")
    assert _option_value(argv, "--class_trust").endswith(f"{gated_root}/gated_samples.class_trust.json")
    assert _option_value(argv, "--ref_meta_json").endswith(f"{gated_root}/gated_samples.ref_meta.json")
    assert _option_value(argv, "--target_real_npz").endswith(
        "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/"
        "ningbo/k500_seed20260531/ningbo_real_k500_seed20260531.signals.npz"
    )
    assert _option_value(argv, "--init_ckpt").endswith(
        "/runs/effnet_direct_k500_v7_sjr_rgq/pytest_run/runs/"
        "ningbo_K500_direct_ft_ep30_seed20260531_val0.2/best_model.pt"
    )
    assert _option_value(argv, "--output_dir").endswith(
        "/runs/ecgtwin_prompt_token_online_at_minimal/pytest_run/ningbo"
    )

    trace = manifest["artifact_trace"]
    input_roles = {item["role"] for item in trace["inputs"]["data_caches"]}
    assert {
        "prompt_token.gated_latent_npz",
        "prompt_token.gated_class_trust_json",
        "prompt_token.gated_ref_meta_json",
        "target_k500.signals_npz",
    } <= input_roles
    assert any(item["role"] == "efficientnet.direct_init_checkpoint" for item in trace["inputs"]["checkpoints"])
    assert any(item["center"] == "ningbo" for item in trace["inputs"]["k500_refs"])
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 1
    assert child_runs[0]["stage"] == "prompt_token_online_at"
    artifact_roles = {item["role"] for item in child_runs[0]["expected_artifacts"]}
    assert artifact_roles >= {
        "best_model",
        "training_log",
        "train_result",
        "checkpoint_latest",
    }


def test_ecgtwin_prompt_token_online_at_audit_rejects_pn2021_quick_eval():
    config = _load("ecgtwin_prompt_token_online_at_minimal.yaml")
    config = copy.deepcopy(config)
    validate_experiment_config(config, repo_root=REPO)
    command = copy.deepcopy(build_runner_commands(config)[0])
    argv = command["argv"]
    argv[argv.index("--quick_eval_source") + 1] = "pn2021"

    with pytest.raises(
        ConfigError,
        match=r"--quick_eval_source='pn2021', expected 'target_real_val'",
    ):
        audit_runner_commands(config, [command])


def test_ecgtwin_prompt_token_online_at_eval_config_uses_same_run_model_and_ref_exclusion():
    config = _load("ecgtwin_prompt_token_online_at_minimal_pn2021_eval.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    postprocess_commands = build_postprocess_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_prompt_token_online_at_eval",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
        postprocess_commands=postprocess_commands,
    )

    assert len(commands) == 1
    command = commands[0]
    argv = command["argv"]
    assert command["matrix"]["center"] == "ningbo"
    assert "CUDA_VISIBLE_DEVICES" not in command["env"]
    assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
    assert argv[1].endswith("scripts/triple_labels/eval_crosscenter.py")
    assert _option_value(argv, "--scheme") == "super5"
    assert _option_value(argv, "--model_name") == "efficientnet1dv2"
    assert _option_value(argv, "--device") == "cuda"
    assert _option_value(argv, "--model_dir").endswith(
        "/runs/ecgtwin_prompt_token_online_at_minimal/pytest_run/ningbo"
    )
    assert _option_value(argv, "--output_path").endswith(
        "/runs/ecgtwin_prompt_token_online_at_minimal_eval/pytest_run/ningbo/"
        "eval_result_v7_super5_sjr_rgq_refexcluded.json"
    )
    assert "--skip_mimic" in argv
    assert "--report_drop_all_zero_pn2021" in argv
    ref_start = argv.index("--exclude_ref_ids") + 1
    ref_end = argv.index("--output_path")
    ref_metas = argv[ref_start:ref_end]
    assert len(ref_metas) == 4
    assert {Path(path).name for path in ref_metas} == {
        "ningbo_real_k500_seed20260531.ref_meta.json",
        "chapman_shaoxing_real_k500_seed20260531.ref_meta.json",
        "cpsc_2018_real_k500_seed20260531.ref_meta.json",
        "georgia_real_k500_seed20260531.ref_meta.json",
    }

    trace = manifest["artifact_trace"]
    assert any(
        item["role"] == "command.model_dir.best_model"
        and item["path"].endswith("/ecgtwin_prompt_token_online_at_minimal/pytest_run/ningbo/best_model.pt")
        for item in trace["inputs"]["checkpoints"]
    )
    assert len(trace["inputs"]["k500_refs"]) == 4
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 1
    assert child_runs[0]["center"] == "ningbo"
    assert child_runs[0]["model_dir"].endswith(
        "/runs/ecgtwin_prompt_token_online_at_minimal/pytest_run/ningbo"
    )
    assert child_runs[0]["expected_artifacts"][0]["role"] == "eval_result"
    assert len(postprocess_commands) == 3
    assert postprocess_commands[0]["argv"][1].endswith("scripts/export_metrics_long.py")
    assert "ecgtwin_prompt_token_online_at_minimal" in postprocess_commands[0]["argv"]
    assert postprocess_commands[1]["argv"][1].endswith("scripts/export_paper_table.py")
    assert "pn2021_all_zero_kept_refexcluded" in postprocess_commands[1]["argv"]
    assert "pn2021_drop_all_zero_refexcluded" in postprocess_commands[2]["argv"]
    postprocess_runs = trace["expected_outputs"]["postprocess_runs"]
    assert len(postprocess_runs) == 3


def test_ecgtwin_prompt_token_online_at_eval_audit_rejects_wrong_model_dir():
    config = _load("ecgtwin_prompt_token_online_at_minimal_pn2021_eval.yaml")
    config = copy.deepcopy(config)
    argv = config["runner"]["argv"]
    argv[argv.index("--model_dir") + 1] = (
        "${paths.output_root}/effnet_direct_k500_v7_sjr_rgq/${runtime.run_id}/runs/"
        "ningbo_K500_direct_ft_ep30_seed20260531_val0.2"
    )
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="model_dir must point at same-run prompt-token online-AT output"):
        build_runner_commands(config)


def test_benchmark_direct_percent_v7_matrix_config_generates_protocol_center_commands():
    config = _load("benchmark_resnet1d_direct_percent_v7_sjr_rgq_matrix.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_benchmark_direct_percent_v7",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    expected_cases = {
        ("ningbo", "p10", "1923"),
        ("ningbo", "p20", "3846"),
        ("chapman_shaoxing", "p10", "582"),
        ("chapman_shaoxing", "p20", "1164"),
        ("cpsc_2018", "p10", "475"),
        ("cpsc_2018", "p20", "949"),
        ("georgia", "p10", "871"),
        ("georgia", "p20", "1742"),
    }

    assert len(commands) == 8
    observed_cases = set()
    for command in commands:
        argv = command["argv"]
        case = command["matrix"]["case"]
        center = case["center"]
        protocol = case["protocol"]
        k = str(case["k"])
        observed_cases.add((center, protocol, k))

        assert "CUDA_VISIBLE_DEVICES" not in command["env"]
        assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
        assert "OPENAI_API_KEY" not in command["env"]
        assert argv[1].endswith("scripts/paper/run_benchmark_direct_finetune_v7_20260530.py")
        assert _option_value(argv, "--centers") == center
        assert _option_value(argv, "--model_name") == "benchmark_resnet1d_wang"
        assert _option_value(argv, "--k") == k
        assert _option_value(argv, "--subset_seed") == "20260531"
        assert _option_value(argv, "--seed") == "20260531"
        assert _option_value(argv, "--init_ckpt").endswith(
            "/runs/benchmark_source_v7_sjr_rgq/"
            "benchmark_resnet1d_wang_seed20260531_v7_sjr_rgq_pytest_run/best_model.pt"
        )
        assert _option_value(argv, "--subset_root").endswith(
            "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
        )
        assert _option_value(argv, "--out_root").endswith(
            f"/runs/benchmark_resnet1d_wang_direct_{protocol}_v7_sjr_rgq/pytest_run/{center}"
        )

    assert observed_cases == expected_cases
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 8
    assert {tuple([child["matrix"]["case"][key] for key in ("center", "protocol")]) for child in child_runs} == {
        (center, protocol) for center, protocol, _k in expected_cases
    }


def test_benchmark_direct_percent_v7_matrix_audit_rejects_case_k_mismatch():
    config = _load("benchmark_resnet1d_direct_percent_v7_sjr_rgq_matrix.yaml")
    config = copy.deepcopy(config)
    idx = config["runner"]["argv"].index("--k")
    config["runner"]["argv"][idx + 1] = "999"
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="--k='999', expected '1923'"):
        build_runner_commands(config)


def test_ecgfounder_direct_k500_v7_matrix_config_generates_center_commands():
    config = _load("ecgfounder_direct_k500_v7_sjr_rgq_matrix.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_ecgfounder_direct_k500_v7",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 4
    assert {command["matrix"]["center"] for command in commands} == {
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    }
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert "CUDA_VISIBLE_DEVICES" not in command["env"]
        assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
        assert argv[1].endswith("scripts/paper/run_ecgfounder_kshot_head_ft_20260517.py")
        assert _option_value(argv, "--centers") == center
        assert _option_value(argv, "--linear_probe_dir").endswith(
            "/paper_foundation_baselines_20260530/ecgfounder_linear_probe_v7_from_v6_cache"
        )
        assert _option_value(argv, "--preprocess_policy") == "official_ptbxl_eval"
        assert _option_value(argv, "--out_dir").endswith(
            f"/runs/ecgfounder_direct_k500_v7_sjr_rgq/pytest_run/{center}"
        )
        assert _option_value(argv, "--ref_root").endswith(
            "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
        )
        assert _option_value(argv, "--k") == "500"
        assert _option_value(argv, "--source_k") == "500"
        assert _option_value(argv, "--subset_seed") == "20260531"
        assert _option_value(argv, "--seed") == "20260531"
        assert _option_value(argv, "--epochs") == "50"
        assert _option_value(argv, "--val_fraction") == "0.2"
        assert _option_value(argv, "--device") == "cuda"
        assert "--reset_head" not in argv
        assert "--force" not in argv

    refs = manifest["artifact_trace"]["inputs"]["k500_refs"]
    assert len(refs) == 4
    assert all(
        "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/"
        in item["ref_meta_json"]["path"]
        for item in refs
    )
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    for command, child_run in zip(commands, child_runs):
        argv = command["argv"]
        center = command["matrix"]["center"]
        expected = ecgfounder_kshot_head_run_dir(
            Path(_option_value(argv, "--out_dir")),
            center=center,
            k=500,
            source_k=500,
            epochs=50,
            seed=20260531,
        )
        assert Path(child_run["child_run_dir"]) == expected


def test_ecgfounder_direct_k500_v7_matrix_audit_rejects_missing_ref_root():
    config = _load("ecgfounder_direct_k500_v7_sjr_rgq_matrix.yaml")
    config = copy.deepcopy(config)
    idx = config["runner"]["argv"].index("--ref_root")
    del config["runner"]["argv"][idx : idx + 2]
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="matrix command must pass --ref_root"):
        build_runner_commands(config)


def test_ecgfounder_vae_lhat_k500_v7_config_uses_v7_direct_heads_and_refs():
    config = _load("ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_ecgfounder_vae_lhat_k500_v7",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert config["runner"]["adapter"] == "ecgfounder_vae_lhat"
    assert "argv" not in config["runner"]
    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert "CUDA_VISIBLE_DEVICES" not in command["env"]
        assert "CUDA_VISIBLE_DEVICES" not in " ".join(argv)
        assert argv[1].endswith("scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py")
        assert _option_value(argv, "--centers") == center
        assert _option_value(argv, "--out_dir").endswith(
            f"/runs/ecgfounder_vae_lhat_k500_v7_sjr_rgq/pytest_run/{center}"
        )
        assert _option_value(argv, "--linear_probe_dir").endswith(
            "/paper_foundation_baselines_20260530/ecgfounder_linear_probe_v7_from_v6_cache"
        )
        assert _option_value(argv, "--init_base_head_from_k500_root").endswith(
            f"/runs/ecgfounder_direct_k500_v7_sjr_rgq/pytest_run/{center}/runs"
        )
        assert _option_value(argv, "--anchor_base_root").endswith(
            "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
        )
        assert _option_value(argv, "--ref_root").endswith(
            "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
        )
        assert _option_value(argv, "--k") == "500"
        assert _option_value(argv, "--k_anchor") == "300"
        assert _option_value(argv, "--hull_neighbor_distance_space") == "standardized"
        assert _option_value(argv, "--hull_neighbor_mode") == "local_random"
        assert _option_value(argv, "--hull_neighbor_pool_size") == "120"
        assert _option_value(argv, "--hull_neighbor_pool_multiplier") == "4"
        assert _option_value(argv, "--selection_source") == "target_real_val"
        assert _option_value(argv, "--target_real_val_seed") == "20260531"
        assert _option_value(argv, "--head_type") == "residual_adapter"
        assert "--freeze_base_head" in argv
        assert "--report_drop_all_zero_pn2021" in argv
        assert "--enable_latent_augmix_branch" not in argv
        assert "--force" not in argv

    trace = manifest["artifact_trace"]
    refs = trace["inputs"]["k500_refs"]
    assert len(refs) == 4
    assert all(
        "/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/"
        in item["ref_meta_json"]["path"]
        for item in refs
    )
    assert sum(item["role"] == "ecgfounder.k500_base_head.best_head"
               for item in trace["inputs"]["init_heads"]) == 4
    assert all(
        "/runs/ecgfounder_direct_k500_v7_sjr_rgq/pytest_run/"
        in item["path"]
        for item in trace["inputs"]["init_heads"]
        if item["role"] == "ecgfounder.k500_base_head.best_head"
    )
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    for command, child_run in zip(commands, child_runs):
        argv = command["argv"]
        center = command["matrix"]["center"]
        expected = ecgfounder_lhat_run_dir(
            Path(_option_value(argv, "--out_dir")),
            center=center,
            k=500,
            hull_m=20,
            hull_lambda="0.15",
            epochs=20,
            seed=20260531,
        )
        assert Path(child_run["child_run_dir"]) == expected
        assert any(
            item["role"] == "checkpoint_index"
            and item["path"].endswith("/checkpoint_index.jsonl")
            for item in child_run["expected_artifacts"]
        )


def test_ecgfounder_vae_lhat_augmix_config_enables_latent_augmix_branch():
    config = _load("ecgfounder_vae_lhat_augmix_k500_v7_sjr_rgq.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert config["experiment"]["name"] == "ecgfounder_vae_lhat_augmix_k500_v7_sjr_rgq"
    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        assert "--enable_latent_augmix_branch" in argv
        assert _option_value(argv, "--latent_augmix_latent_weight_cap") == "0.25"
        assert _option_value(argv, "--latent_augmix_width") == "3"
        assert _option_value(argv, "--latent_augmix_depth") == "-1"
        assert _option_value(argv, "--latent_augmix_alpha") == "1.0"
        assert _option_value(argv, "--latent_augmix_severity") == "2"
        assert _all_option_values(argv, "--latent_augmix_ops") == [
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
        ]
        assert _option_value(argv, "--out_dir").endswith(
            "/runs/ecgfounder_vae_lhat_augmix_k500_v7_sjr_rgq/pytest_run/"
            f"{command['matrix']['center']}"
        )


def test_ecgfounder_vae_lhat_raw_augmix_config_exposes_mixture_flags():
    config = _load("ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["adaptation"]["raw_corrupt_consistency"] = {
        "enabled": True,
        "batch_size": 128,
        "copies": 2,
        "prob": 1.0,
        "severity": 5,
        "severity_profile": "calibrated_10to20pp",
        "consistency_weight": 2.0,
        "consistency_loss": "jsd",
        "bce_weight": 1.0,
        "max_batches": 64,
        "scope": "source_target",
        "clip_abs": 6.0,
        "grad_clip": 1.0,
        "source_signal_cache_dir": "${paths.data_root}/paper_effnet_source_signal_cache",
        "ops": [
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
            "random_leads_masking",
        ],
        "no_renorm": True,
        "view_mode": "augmix",
        "augmix": {
            "width": 1,
            "depth": 1,
            "alpha": 1.0,
            "mixture_mode": "fixed",
            "mixture_prob": 1.0,
            "mixture_beta_a": 0.0,
            "mixture_beta_b": 0.0,
        },
    }
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        assert "--enable_raw_corrupt_consistency" in argv
        assert _option_value(argv, "--raw_corrupt_severity_profile") == "calibrated_10to20pp"
        assert _option_value(argv, "--raw_corrupt_scope") == "source_target"
        assert "--raw_corrupt_no_renorm" in argv
        assert _option_value(argv, "--raw_corrupt_view_mode") == "augmix"
        assert _option_value(argv, "--raw_augmix_width") == "1"
        assert _option_value(argv, "--raw_augmix_depth") == "1"
        assert _option_value(argv, "--raw_augmix_alpha") == "1.0"
        assert _option_value(argv, "--raw_augmix_mixture_mode") == "fixed"
        assert _option_value(argv, "--raw_augmix_mixture_prob") == "1.0"
        assert _option_value(argv, "--raw_augmix_mixture_beta_a") == "0.0"
        assert _option_value(argv, "--raw_augmix_mixture_beta_b") == "0.0"


def test_ecgfounder_vae_lhat_diagnostic_last_epoch_config_is_explicitly_allowed():
    config = _load("ecgfounder_vae_lhat_fullpool_raw_augmix_w1_m100_last_epoch_k500_v7_sjr_rgq_cpsc_2018.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert config["run_record"]["registration_status"] == "exploratory"
    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert _option_value(argv, "--selection_metric") == "last_epoch"
    assert _option_value(argv, "--raw_corrupt_view_mode") == "augmix"
    assert _option_value(argv, "--raw_augmix_width") == "1"
    assert _option_value(argv, "--raw_augmix_mixture_mode") == "fixed"


def test_ecgfounder_vae_lhat_adapter_is_registered():
    from ecg_adv_gen.config.adapters.registry import runner_adapter_names

    assert "ecgfounder_vae_lhat" in runner_adapter_names()


def test_high_risk_long_argv_adapters_are_registered():
    from ecg_adv_gen.config.adapters.registry import runner_adapter_names

    names = set(runner_adapter_names())
    assert {
        "pn2021_eval",
        "pn2021c_eval",
        "prompt_token_online_at",
        "direct_finetune",
    } <= names


@pytest.mark.parametrize(
    ("config_name", "adapter_name", "expected_commands"),
    [
        ("pn2021_eval_v7_sjr_rgq_refexcluded.yaml", "pn2021_eval", 4),
        ("pn2021c_effnet_v7_augmix_vs_noaug.yaml", "pn2021c_eval", 8),
        ("ecgtwin_prompt_token_online_at_minimal.yaml", "prompt_token_online_at", 1),
        ("effnet_direct_k500_v7_sjr_rgq_matrix.yaml", "direct_finetune", 4),
    ],
)
def test_high_risk_long_argv_configs_use_typed_adapters(config_name: str, adapter_name: str, expected_commands: int):
    config = _load(config_name)

    assert config["runner"]["adapter"] == adapter_name
    assert "argv" not in config["runner"]
    commands = build_runner_commands(config)
    assert len(commands) == expected_commands


def test_ecgfounder_vae_lhat_k500_v7_audit_rejects_missing_ref_root():
    config = _load("ecgfounder_vae_lhat_k500_v7_sjr_rgq.yaml")
    validate_experiment_config(config, repo_root=REPO)
    command = copy.deepcopy(build_runner_commands(config)[0])
    idx = command["argv"].index("--ref_root")
    del command["argv"][idx : idx + 2]

    with pytest.raises(ConfigError, match="matrix command must pass --ref_root"):
        audit_runner_commands(config, [command])


def test_effnet_direct_manifest_child_dir_uses_shared_run_naming_helper():
    config = _load("effnet_direct_k500_v6.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    command = build_runner_commands(config)[0]
    manifest = make_dry_run_manifest(
        config,
        commands=[command],
        local_paths=paths,
        run_id="pytest_effnet_direct_naming",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )
    argv = command["argv"]
    centers = []
    idx = argv.index("--centers") + 1
    while idx < len(argv) and not argv[idx].startswith("--"):
        centers.append(argv[idx])
        idx += 1
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]

    assert len(child_runs) == len(centers) == 4
    for center, child_run in zip(centers, child_runs):
        expected_leaf = build_effnet_direct_run_leaf(
            {
                "center": center,
                "k": _option_value(argv, "--k"),
                "epochs": _option_value(argv, "--epochs"),
                "seed": _option_value(argv, "--seed"),
                "val_fraction": _option_value(argv, "--val_fraction"),
            }
        )
        assert Path(child_run["child_run_dir"]).name == expected_leaf


def test_runner_audit_rejects_unscoped_output_root():
    config = _load("effnet_direct_k500_v6.yaml")
    config = copy.deepcopy(config)
    idx = config["runner"]["argv"].index("--out_root")
    config["runner"]["argv"][idx + 1] = f"{config['paths']['output_root']}/effnet_direct_k500_v6"
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="must include runtime.run_id"):
        build_runner_commands(config)


@pytest.mark.parametrize(
    ("option", "script"),
    [
        ("--input_dirs", "merge_gated_prompt_token_pools.py"),
        ("--gated_dirs", "select_quality_prompt_token_pool.py"),
    ],
)
def test_runner_audit_checks_multi_input_directory_options(option: str, script: str):
    config = _load("effnet_direct_k500_v6.yaml")
    outside_path = "/tmp/outside_gated_pool"
    command = {
        "name": "pytest_multi_input_dir_audit",
        "matrix": {},
        "cwd": str(REPO),
        "env": {},
        "argv": [
            "/home/linbinhao/micromamba/envs/ECGTwin/bin/python",
            str(REPO / "scripts" / "ecgtwin_gen" / script),
            option,
            str(REPO / "safe_gated_pool"),
            outside_path,
            "--out_dir",
            str(REPO / "runs" / "pytest_run" / "merged_gated_pool"),
        ],
    }

    with pytest.raises(ConfigError, match=option):
        audit_runner_commands(config, [command])


def test_runner_audit_records_managed_entrypoint_profiles():
    config = _load("ecgtwin_prompt_token_online_at_minimal.yaml")
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=validate_experiment_config(config, repo_root=REPO),
        run_id=config["runtime"]["run_id"],
        cli_args=argparse.Namespace(
            config="configs/experiments/ecgtwin_prompt_token_online_at_minimal.yaml",
            local_config=str(LOCAL_EXAMPLE),
            run_id=config["runtime"]["run_id"],
            dry_run=True,
            execute=False,
            write_plan=False,
            resume=False,
            force=False,
            set_overrides=[],
        ),
    )

    command_profiles = manifest["artifact_trace"]["protocol_audit"]["command_audit"]["managed_entrypoints"]

    assert command_profiles
    assert {
        "script_name": "synth_online_at_super5.py",
        "relative_path": "scripts/pgd_cross_center/synth_online_at_super5.py",
        "wrapper_root": "scripts/pgd_cross_center",
        "family": "synthetic_online_adversarial_training",
    } in command_profiles


def test_train_ptbxl_command_audit_adapter_rejects_non_auprc_checkpoint_metric():
    config = _load("benchmark_source_v7_sjr_rgq_matrix.yaml")
    command = build_runner_commands(config)[0]

    assert audit_train_ptbxl_command(command, expected_seed=20260531) == []

    bad = copy.deepcopy(command)
    idx = bad["argv"].index("--checkpoint_metric")
    bad["argv"][idx + 1] = "auroc"

    errors = audit_train_ptbxl_command(bad, expected_seed=20260531)

    assert any("--checkpoint_metric" in error for error in errors)


def test_direct_finetune_command_audit_adapter_rejects_wrong_matrix_center():
    config = _load("effnet_direct_k500_v7_sjr_rgq_matrix.yaml")
    command = build_runner_commands(config)[0]

    assert audit_direct_finetune_command(
        command,
        expected_k=500,
        expected_seed=20260531,
        target_centers=set(config["paper_protocol"]["centers"]["target_4"]),
    ) == []

    bad = copy.deepcopy(command)
    bad["matrix"]["center"] = "georgia"

    errors = audit_direct_finetune_command(
        bad,
        expected_k=500,
        expected_seed=20260531,
        target_centers=set(config["paper_protocol"]["centers"]["target_4"]),
    )

    assert any("matrix center" in error for error in errors)


def test_postprocess_audit_rejects_unscoped_output_dir():
    config = _load("effnet_direct_k500_v6.yaml")
    config = copy.deepcopy(config)
    idx = config["postprocess"]["commands"][0]["argv"].index("--output-dir")
    config["postprocess"]["commands"][0]["argv"][idx + 1] = f"{config['paths']['output_root']}/metrics_export"
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="must include runtime.run_id"):
        build_postprocess_commands(config)


def test_cli_set_overrides_are_whitelisted_and_applied_before_interpolation():
    config = load_experiment_config(
        REPO / "configs" / "experiments" / "effnet_direct_k500_v6.yaml",
        LOCAL_EXAMPLE,
        overrides=[
            "training.epochs=2",
            "training.optimizer.lr=0.001",
            "resources.default_num_workers=1",
        ],
        runtime_context={"run_id": "pytest_override"},
    )
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    argv = commands[0]["argv"]
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_cli_override",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert "training.epochs" in ALLOWED_CLI_OVERRIDE_KEYS
    assert config["_cli_overrides"][0]["key"] == "training.epochs"
    assert _option_value(argv, "--epochs") == "2"
    assert _option_value(argv, "--lr") == "0.001"
    assert _option_value(argv, "--num_workers") == "1"
    assert manifest["cli_overrides"] == config["_cli_overrides"]


def test_cli_set_rejects_protocol_and_type_changes():
    config = _load("effnet_direct_k500_v6.yaml")
    with pytest.raises(ConfigError, match="not allowed"):
        apply_cli_overrides(config, ["paper_protocol.kshot.k=100"])
    with pytest.raises(ConfigError, match="expects int"):
        apply_cli_overrides(config, ["training.epochs=two"])
    with pytest.raises(ConfigError, match="key=value"):
        apply_cli_overrides(config, ["training.epochs"])


def test_ecgfounder_inithead_command_has_required_k500_inputs():
    config = _load("ecgfounder_inithead_fullft_k500_v6.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        assert "--ref_meta_json" in argv
        assert _option_value(argv, "--ref_meta_json").endswith("_real_k500_seed20260531.ref_meta.json")
        assert _option_value(argv, "--k") == "500"
        assert "--init_head_path" in argv
        assert _option_value(argv, "--init_head_path").endswith("best_head.pt")
        assert _option_value(argv, "--eval_batch_size") == "128"
        assert _option_value(argv, "--source_train_limit") == "0"
        assert _option_value(argv, "--cache_dir") == ""
        assert _option_value(argv, "--selection_metric") == "source_plus_target_val_auprc"
        assert _option_value(argv, "--target_val_count") == "100"


def test_ecgfounder_locked_fullft_adapter_is_registered():
    from ecg_adv_gen.config.adapters.registry import runner_adapter_names

    assert "ecgfounder_fullft" in runner_adapter_names()


def test_ecgfounder_locked_k500_fullft_command_uses_last_checkpoint_no_head_route():
    config = _load("ecgfounder_k500_fullft_locked.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert config["runner"]["adapter"] == "ecgfounder_fullft"
    assert config["paper_protocol"]["selection"]["policy"] == "last_checkpoint_only"
    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert argv[1].endswith("scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py")
        assert _option_value(argv, "--center") == center
        assert _option_value(argv, "--ref_meta_json").endswith(
            f"/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/{center}/"
            f"k500_seed20260531/{center}_real_k500_seed20260531.ref_meta.json"
        )
        assert _option_value(argv, "--k") == "500"
        assert _option_value(argv, "--seed") == "20260531"
        assert _option_value(argv, "--target_val_count") == "0"
        assert "--target_val_seed" not in argv
        assert _option_value(argv, "--selection_metric") == "source_auprc"
        assert _option_value(argv, "--checkpoint_policy") == "last"
        assert _option_value(argv, "--init_model_path").endswith(
            "/ecgfounder_ptbxl_super5_fullft_locked/pytest_run/runs/ptbxl_super5_fullft_locked/last_model.pt"
        )
        assert _option_value(argv, "--supervised_input_mode") == "raw1000"
        assert _option_value(argv, "--target_raw1000_npz_override").endswith(
            f"/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/{center}/"
            f"k500_seed20260531/{center}_real_k500_seed20260531.raw1000.npz"
        )
        assert "--init_head_path" not in argv
        assert "--enable_raw_corrupt_consistency" not in argv
        assert "--ecgfounder_input_repair_flat_leads" not in argv
        assert "--ecgfounder_input_bandpass_low_hz" not in argv
        assert "--ecgfounder_input_bandpass_high_hz" not in argv
        assert "best_head.pt" not in " ".join(argv)
        assert "residual_adapter" not in " ".join(argv)


def test_ecgfounder_locked_threechain_augmix_command_uses_fullft_last_checkpoint_mainline():
    config = _load("ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert config["runner"]["adapter"] == "ecgfounder_fullft"
    assert config["paper_protocol"]["selection"]["policy"] == "last_checkpoint_only"
    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        joined = " ".join(argv)
        assert argv[1].endswith("scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py")
        assert _option_value(argv, "--center") == center
        assert _option_value(argv, "--checkpoint_policy") == "last"
        assert _option_value(argv, "--target_val_count") == "0"
        assert "--target_val_seed" not in argv
        assert _option_value(argv, "--selection_metric") == "source_auprc"
        assert _option_value(argv, "--supervised_input_mode") == "raw1000"
        assert _option_value(argv, "--target_raw1000_npz_override").endswith(
            f"/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/{center}/"
            f"k500_seed20260531/{center}_real_k500_seed20260531.raw1000.npz"
        )
        assert _option_value(argv, "--init_model_path").endswith(
            f"/ecgfounder_k500_fullft_locked/pytest_run/runs/{center}_k500_fullft_locked/last_model.pt"
        )
        assert "--enable_vae_adv_stream" in argv
        assert "--enable_latent_augmix_branch" in argv
        assert _option_value(argv, "--latent_augmix_topology") == "locked_three_chain"
        assert _option_value(argv, "--latent_augmix_width") == "3"
        assert _option_value(argv, "--latent_augmix_severity") == "5"
        assert _option_value(argv, "--latent_augmix_severity_profile") == "standard"
        assert _all_option_values(argv, "--latent_augmix_ops") == [
            "powerline_noise",
            "emg_noise",
            "baseline_wander",
            "baseline_shift",
            "random_leads_masking",
        ]
        assert "--enable_raw_corrupt_consistency" not in argv
        assert "--enable_raw_corrupt_aux_consistency" not in argv
        assert "--ecgfounder_input_repair_flat_leads" not in argv
        assert "--ecgfounder_input_bandpass_low_hz" not in argv
        assert "--init_head_path" not in argv
        assert "best_head.pt" not in joined
        assert "residual_adapter" not in joined


def test_ecgfounder_locked_threechain_augmix_manifest_expects_last_model():
    config = _load("ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_ecgfounder_threechain_locked",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    checkpoint_inputs = manifest["artifact_trace"]["inputs"]["checkpoints"]
    refs = manifest["artifact_trace"]["inputs"]["k500_refs"]

    assert len(child_runs) == len(commands) == 4
    assert len(refs) == 4
    assert all(item["signals_npz"]["role"] == "kshot_raw1000_signals" for item in refs)
    assert all(item["signals_npz"]["path"].endswith(".raw1000.npz") for item in refs)
    assert all(
        any(item["role"] == "last_model" and item["path"].endswith("/last_model.pt")
            for item in child["expected_artifacts"])
        for child in child_runs
    )
    assert all(
        not any(item["path"].endswith("best_head.pt") for item in child["expected_artifacts"])
        for child in child_runs
    )
    assert any(
        item["role"] == "command.init_model_path"
        and "/ecgfounder_k500_fullft_locked/" in item["path"]
        and item["path"].endswith("/last_model.pt")
        for item in checkpoint_inputs
    )


def test_ecgfounder_locked_ptbxl_fullft_command_is_source_only_last_checkpoint():
    config = _load("ecgfounder_ptbxl_super5_fullft_locked.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert config["runner"]["adapter"] == "ecgfounder_fullft"
    assert config["adaptation"]["stage"] == "ptbxl_source"
    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert argv[1].endswith("scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py")
    assert _option_value(argv, "--stage") == "ptbxl_source"
    assert _option_value(argv, "--checkpoint_policy") == "last"
    assert _option_value(argv, "--selection_metric") == "source_auprc"
    assert _option_value(argv, "--run_name") == "ptbxl_super5_fullft_locked"
    assert "--ref_meta_json" not in argv
    assert "--center" not in argv
    assert "--init_model_path" not in argv
    assert "--init_head_path" not in argv
    assert "--target_val_seed" not in argv
    assert "--enable_raw_corrupt_consistency" not in argv
    assert "best_head.pt" not in " ".join(argv)
    assert "residual_adapter" not in " ".join(argv)


def test_ecgfounder_locked_ptbxl_fullft_manifest_has_no_k500_refs():
    config = _load("ecgfounder_ptbxl_super5_fullft_locked.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_ecgfounder_ptbxl_fullft",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    launch_artifacts = manifest["artifact_trace"]["expected_outputs"]["launch_artifacts_declared"]

    assert manifest["artifact_trace"]["inputs"]["k500_refs"] == []
    assert "k500_ref_ids.json" not in launch_artifacts
    assert len(child_runs) == 1
    assert child_runs[0]["child_run_dir"].endswith("/runs/ptbxl_super5_fullft_locked")
    paths_by_role = {item["role"]: item["path"] for item in child_runs[0]["expected_artifacts"]}
    assert paths_by_role["last_model"].endswith("/runs/ptbxl_super5_fullft_locked/last_model.pt")


def test_ecgfounder_locked_k500_fullft_manifest_expects_last_model():
    config = _load("ecgfounder_k500_fullft_locked.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_ecgfounder_locked_fullft",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    checkpoint_inputs = manifest["artifact_trace"]["inputs"]["checkpoints"]
    refs = manifest["artifact_trace"]["inputs"]["k500_refs"]

    assert len(child_runs) == len(commands) == 4
    assert len(refs) == 4
    assert all(item["signals_npz"]["role"] == "kshot_raw1000_signals" for item in refs)
    assert all(item["signals_npz"]["path"].endswith(".raw1000.npz") for item in refs)
    assert any(
        item["role"] == "command.init_model_path" and item["path"].endswith("/last_model.pt")
        for item in checkpoint_inputs
    )
    for child in child_runs:
        roles = {item["role"] for item in child["expected_artifacts"]}
        paths_by_role = {item["role"]: item["path"] for item in child["expected_artifacts"]}
        assert "last_model" in roles
        assert paths_by_role["last_model"].endswith("/last_model.pt")
        assert "best_head" not in roles
        assert all(not item["path"].endswith("best_head.pt") for item in child["expected_artifacts"])


def test_ecgfounder_inithead_manifest_child_dirs_use_shared_fullft_naming_helper():
    config = _load("ecgfounder_inithead_fullft_k500_v6.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_ecgfounder_fullft_naming",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]

    assert len(child_runs) == len(commands) == 4
    for command, child in zip(commands, child_runs):
        argv = command["argv"]
        center = command["matrix"]["center"]
        expected_leaf = build_ecgfounder_fullft_run_leaf(
            {
                "center": center,
                "k": _option_value(argv, "--k"),
                "epochs": _option_value(argv, "--epochs"),
                "lr": _option_value(argv, "--lr"),
                "source_weight": _option_value_or(argv, "--source_weight", "1.0"),
                "target_real_weight": _option_value_or(argv, "--target_real_weight", "40.0"),
                "init_head_path": _option_value_or(argv, "--init_head_path", ""),
                "run_suffix": _option_value_or(argv, "--run_suffix", ""),
                "target_val_count": _option_value_or(argv, "--target_val_count", "0"),
                "target_val_seed": _option_value_or(argv, "--target_val_seed"),
                "selection_metric": _option_value_or(argv, "--selection_metric", "source_auprc"),
                "seed": _option_value(argv, "--seed"),
            }
        )

        assert Path(child["child_run_dir"]).name == expected_leaf


def test_ecgfounder_direct_headft_command_has_required_k500_inputs():
    config = _load("ecgfounder_direct_k500_v6.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert argv[1].endswith("scripts/paper/run_ecgfounder_kshot_head_ft_20260517.py")
    centers_start = argv.index("--centers") + 1
    centers_end = argv.index("--linear_probe_dir")
    assert set(argv[centers_start:centers_end]) == {"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"}
    assert _option_value(argv, "--linear_probe_dir").endswith(
        "/paper_foundation_baselines_20260524/ecgfounder_linear_probe_v6_from_legacy_cache"
    )
    assert _option_value(argv, "--preprocess_policy") == "official_ptbxl_eval"
    assert _option_value(argv, "--out_dir").endswith("/runs/ecgfounder_direct_k500_v6/pytest_run")
    assert _option_value(argv, "--k") == "500"
    assert _option_value(argv, "--source_k") == "500"
    assert _option_value(argv, "--subset_seed") == "20260531"
    assert _option_value(argv, "--seed") == "20260531"
    assert _option_value(argv, "--epochs") == "50"
    assert _option_value(argv, "--val_fraction") == "0.2"
    assert _option_value(argv, "--device") == "cuda"
    assert "--reset_head" not in argv
    assert "--force" not in argv


def test_ecgfounder_inithead_smoke_command_reuses_signal_cache_and_limits_runtime():
    config = _load("ecgfounder_inithead_fullft_k500_v6_smoke.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    postprocess_commands = build_postprocess_commands(config)
    paths = validate_experiment_config(config, repo_root=REPO)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_ecgfounder_inithead_smoke",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
        postprocess_commands=postprocess_commands,
    )

    assert config["experiment"]["status"] == "smoke-only"
    assert config["paper_protocol"]["centers"]["target_4"] == [
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    ]
    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert argv[1].endswith("scripts/paper/run_ecgfounder_fullft_super5_pilot_20260523.py")
        assert _option_value(argv, "--center") == center
        assert _option_value(argv, "--epochs") == "1"
        assert _option_value(argv, "--num_workers") == "0"
        assert _option_value(argv, "--batch_size") == "32"
        assert _option_value(argv, "--eval_batch_size") == "64"
        assert _option_value(argv, "--source_train_limit") == "64"
        assert _option_value(argv, "--target_val_count") == "20"
        assert _option_value(argv, "--run_name") == f"{center}_smoke_inithead_fullft_ep1"
        assert _option_value(argv, "--cache_dir").endswith(
            "/paper_ecgfounder_fullft_signal_cache_union_20260528/cache"
        )
    assert len(postprocess_commands) == 3
    assert any(
        item["role"] == "ecgfounder.fullft.signal_cache_dir"
        for item in manifest["artifact_trace"]["inputs"]["data_caches"]
    )


def test_ecgfounder_vae_lhat_command_has_paper_safe_k500_inputs():
    config = _load("ecgfounder_vae_lhat_k500_v6.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert argv[1].endswith("scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py")
        assert _option_value(argv, "--centers") == center
        assert _option_value(argv, "--out_dir").endswith(
            f"/runs/ecgfounder_vae_lhat_k500_v6/pytest_run/{center}"
        )
        assert _option_value(argv, "--linear_probe_dir").endswith(
            "/paper_foundation_baselines_20260524/ecgfounder_linear_probe_v6_from_legacy_cache"
        )
        assert _option_value(argv, "--checkpoint").endswith("/ecgfounder/checkpoint/12_lead_ECGFounder.pth")
        assert _option_value(argv, "--preprocess_policy") == "official_ptbxl_eval"
        assert _option_value(argv, "--k") == "500"
        assert _option_value(argv, "--k_anchor") == "300"
        assert _option_value(argv, "--anchor_sample_mode") == "hard_bce"
        assert _option_value(argv, "--hull_m") == "20"
        assert _option_value(argv, "--hull_lambda") == "0.15"
        assert _option_value(argv, "--hull_steps") == "5"
        assert _option_value(argv, "--head_type") == "residual_adapter"
        assert _option_value(argv, "--source_train_limit") == "0"
        assert "--freeze_base_head" in argv
        assert _option_value(argv, "--init_base_head_from_k500_root").endswith(
            "/paper_foundation_baselines_20260524/ecgfounder_kshot_head_ft_v6_from_legacy_cache/runs"
        )
        assert _option_value(argv, "--selection_metric") == "target_auprc"
        assert _option_value(argv, "--selection_source") == "target_real_val"
        assert _option_value(argv, "--target_real_val_seed") == "20260531"
        assert _option_value(argv, "--anchor_base_root").endswith("/paper_vae_only_lhat_kcurve_20260518/subsets")
        assert _option_value(argv, "--device") == "cuda"
        assert "--report_drop_all_zero_pn2021" in argv
        assert "--force" not in argv


def test_pn2021_eval_command_has_refexcluded_v6_inputs():
    config = _load("pn2021_eval_v6_refexcluded.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    postprocess_commands = build_postprocess_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_pn2021_eval_v6",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
        postprocess_commands=postprocess_commands,
    )

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert argv[1].endswith("scripts/triple_labels/eval_crosscenter.py")
        assert _option_value(argv, "--scheme") == "super5"
        assert _option_value(argv, "--model_name") == "efficientnet1dv2"
        assert _option_value(argv, "--model_dir").endswith(
            f"/{center}_K500_direct_ft_ep30_seed20260531_val0.2"
        )
        assert _option_value(argv, "--crop_len") == "1000"
        assert _option_value(argv, "--preprocess_mode") == "minimal_resample"
        assert _option_value(argv, "--norm_mode") == "per_sample_global"
        assert "--skip_mimic" in argv
        assert "--report_drop_all_zero_pn2021" in argv
        ref_start = argv.index("--exclude_ref_ids") + 1
        ref_end = argv.index("--output_path")
        ref_metas = argv[ref_start:ref_end]
        assert len(ref_metas) == 4
        assert all("k500_seed20260531" in path for path in ref_metas)
        assert _option_value(argv, "--output_path").endswith(
            f"/pn2021_eval_v6_refexcluded/pytest_run/{center}/eval_result_v7_super5_sjr_rgq_refexcluded.json"
        )

    traced_checkpoints = manifest["artifact_trace"]["inputs"]["checkpoints"]
    assert any(item["role"] == "command.model_dir.best_model" for item in traced_checkpoints)
    assert len(manifest["artifact_trace"]["inputs"]["k500_refs"]) == 16
    assert len(postprocess_commands) == 3
    assert postprocess_commands[0]["argv"][1].endswith("scripts/export_metrics_long.py")
    assert "--expected-mapping-version" in postprocess_commands[0]["argv"]
    assert "v7_super5_sjr_rgq_review_20260528" in postprocess_commands[0]["argv"]
    assert postprocess_commands[1]["argv"][1].endswith("scripts/export_paper_table.py")
    assert "pn2021_all_zero_kept_refexcluded" in postprocess_commands[1]["argv"]
    assert "pn2021_drop_all_zero_refexcluded" in postprocess_commands[2]["argv"]
    postprocess_runs = manifest["artifact_trace"]["expected_outputs"]["postprocess_runs"]
    assert len(postprocess_runs) == 3
    assert any(
        artifact["role"] == "metrics_long"
        for run in postprocess_runs
        for artifact in run["expected_artifacts"]
    )


def test_pn2021_eval_smoke_command_limits_runtime_samples():
    config = _load("pn2021_eval_v6_refexcluded_smoke.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert config["experiment"]["status"] == "smoke-only"
    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert argv[1].endswith("scripts/triple_labels/eval_crosscenter.py")
        assert _option_value(argv, "--pn2021_limit") == "64"
        assert _option_value(argv, "--num_workers") == "0"
        assert _option_value(argv, "--min_pos") == "2"
        assert _option_value(argv, "--output_path").endswith(
            f"/pn2021_eval_v6_refexcluded_smoke/pytest_run/{center}/eval_result_v7_super5_sjr_rgq_refexcluded_smoke.json"
        )


def test_pn2021c_effnet_v7_augmix_config_generates_refexcluded_corruption_commands():
    config = _load("pn2021c_effnet_v7_augmix_vs_noaug.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_pn2021c_effnet_v7",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 8
    seen = {(cmd["matrix"]["center"], cmd["matrix"]["method"]["name"]) for cmd in commands}
    assert seen == {
        (center, method)
        for center in ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
        for method in ["vae_noaug", "vae_lhat"]
    }
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        method = command["matrix"]["method"]["name"]
        assert argv[1].endswith("scripts/triple_labels/eval_pn2021_corruptions.py")
        assert _option_value(argv, "--mode") == "stream"
        assert _option_value(argv, "--scheme") == "super5"
        assert _option_value(argv, "--centers") == center
        assert _option_value(argv, "--device") == "cuda"
        assert _option_value(argv, "--crop_len") == "1000"
        assert _option_value(argv, "--severity_profile") == "standard"
        assert _option_value(argv, "--seed") == "20260501"
        model_dir = _option_value(argv, "--model_dir")
        assert f"/effnet_{method}_k500_v7_sjr_rgq/" in model_dir
        assert f"/{center}_realall_" in model_dir
        assert _option_value(argv, "--clean_eval_json") == f"{model_dir}/eval_result_v7_exclrefs_crop1000.json"
        assert _option_value(argv, "--exclude_ref_ids").endswith(
            f"/{center}/k500_seed20260531/{center}_real_k500_seed20260531.ref_meta.json"
        )
        assert _option_value(argv, "--output_path").endswith(
            f"/pn2021c_effnet_v7_augmix_vs_noaug/pytest_run/{center}/{method}/"
            "eval_pn2021_c_v7_refexcluded_stream_standard.json"
        )
        if method == "vae_noaug":
            assert "_noaugmix_" in model_dir
        else:
            assert "_noaugmix_" not in model_dir

    traced_refs = manifest["artifact_trace"]["inputs"]["k500_refs"]
    assert len(traced_refs) == 8
    assert {ref["center"] for ref in traced_refs} == {"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"}
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 8
    assert all(child["expected_artifacts"][0]["role"] == "eval_result" for child in child_runs)


def test_pn2021c_official_s5_locked_protocol_overrides_selection_and_raw_order():
    config = _load("pn2021c_official_s5_locked_protocol.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_pn2021c_official_s5_locked",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 8
    assert config["paper_protocol"]["selection"]["policy"] == "last_checkpoint_only"
    assert config["paper_protocol"]["selection"]["forbid_k500_validation_split"] is True
    assert config["paper_protocol"]["selection"]["forbid_best_checkpoint_selection"] is True
    assert manifest["artifact_trace"]["selection_policy"]["policy"] == "last_checkpoint_only"
    for command in commands:
        argv = command["argv"]
        assert _option_value(argv, "--corruption_input") == "raw_first"
        assert _option_value(argv, "--severity_profile") == "standard"
        assert _option_value(argv, "--severities") == "5"
        assert _option_value(argv, "--output_path").endswith(
            "eval_pn2021_c_v7_refexcluded_stream_standard_official_s5_locked.json"
        )


def test_pn2021c_effnet_threechain_locked_official_s5_targets_locked_run_dir():
    config = _load("pn2021c_effnet_threechain_locked_official_s5.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id=config["runtime"]["run_id"],
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        model_dir = _option_value(argv, "--model_dir")
        assert model_dir == (
            f"/home/linbinhao/ECG_adv_data/runs/"
            f"effnet_vae_lhat_augmix_threechain_locked_k500/"
            f"{config['runtime']['run_id']}/"
            f"{center}_realall_targetheavy_M20_lam0p05_augmix_s5_wlat0p25_hs3_target_macro_auprc_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_threechain_s5_locked_ep30_seed20260601"
        )
        assert _option_value(argv, "--clean_eval_json") == (
            f"{model_dir}/eval_result_v7_exclrefs_crop1000.json"
        )
        assert _option_value(argv, "--corruption_input") == "raw_first"
        assert _option_value(argv, "--severity_profile") == "standard"
        assert _option_value(argv, "--severities") == "5"
        assert _option_value(argv, "--checkpoint_name") == "last_model.pt"

    traced_checkpoints = manifest["artifact_trace"]["inputs"]["checkpoints"]
    assert any(
        item["role"] == "command.model_dir.last_model"
        and item["path"].endswith("/last_model.pt")
        for item in traced_checkpoints
    )
    assert not any(item["role"] == "command.model_dir.best_model" for item in traced_checkpoints)


def test_pn2021c_ecgfounder_threechain_locked_official_s5_targets_locked_run_dir():
    config = _load("pn2021c_ecgfounder_threechain_locked_official_s5.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id=config["runtime"]["run_id"],
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        run_dir = _option_value(argv, "--run_dir")
        assert argv[1].endswith("scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py")
        assert run_dir == (
            f"/home/linbinhao/ECG_adv_data/runs/"
            f"ecgfounder_vae_lhat_augmix_threechain_locked_k500/"
            f"{config['runtime']['run_id']}/runs/{center}_vae_lhat_augmix_threechain_locked"
        )
        assert _option_value(argv, "--corruption_input") == "bottleneck5000"
        assert _option_value(argv, "--severity_profile") == "standard"
        assert _option_value(argv, "--severities") == "5"
        assert _option_value(argv, "--output_path").endswith(
            f"/pn2021c_ecgfounder_threechain_locked_official_s5/{config['runtime']['run_id']}/"
            f"{center}/ecgfounder_threechain_locked/"
            "eval_pn2021_c_ecgfounder_v7_refexcluded_stream_standard_official_s5_locked.json"
        )

    traced_checkpoints = manifest["artifact_trace"]["inputs"]["checkpoints"]
    assert any(
        item["role"] == "command.run_dir.last_model"
        and item["path"].endswith("/last_model.pt")
        for item in traced_checkpoints
    )
    assert any(item["role"] == "model.checkpoint" for item in traced_checkpoints)
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    assert all(child["expected_artifacts"][0]["role"] == "eval_result" for child in child_runs)


def test_pn2021c_effnet_v7_strong_config_uses_calibrated_profile():
    config = _load("pn2021c_effnet_v7_strong_10to20pp.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 8
    assert config["experiment"]["name"] == "pn2021c_effnet_v7_strong_10to20pp"
    assert config["evaluation"]["severity_profile"] == "calibrated_10to20pp"
    assert config["evaluation"]["severities"] == [5]
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        method = command["matrix"]["method"]["name"]
        assert _option_value(argv, "--severity_profile") == "calibrated_10to20pp"
        assert _option_value(argv, "--severities") == "5"
        assert _option_value(argv, "--output_path").endswith(
            f"/pn2021c_effnet_v7_strong_10to20pp/pytest_run/{center}/{method}/"
            "eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp.json"
        )


def test_pn2021c_effnet_v7_strong_smoke_config_targets_ningbo_only(tmp_path: Path):
    config = _load("pn2021c_effnet_v7_strong_10to20pp_smoke.yaml")
    local_paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=local_paths,
        run_id="pytest_run",
        postprocess_commands=[],
        cli_args=argparse.Namespace(
            config="configs/experiments/pn2021c_effnet_v7_strong_10to20pp_smoke.yaml",
            local_config="configs/local/linbinhao_server.example.yaml",
            run_id="pytest_run",
            dry_run=True,
            write_plan=True,
        ),
    )

    assert len(commands) == 2
    assert config["experiment"]["name"] == "pn2021c_effnet_v7_strong_10to20pp_smoke"
    assert config["paper_protocol"]["centers"]["target_4"] == [
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
    ]
    assert {command["matrix"]["center"] for command in commands} == {"ningbo"}
    for command in commands:
        argv = command["argv"]
        assert _option_value(argv, "--severity_profile") == "calibrated_10to20pp"

    traced_refs = manifest["artifact_trace"]["inputs"]["k500_refs"]
    assert len(traced_refs) == 2
    assert {ref["center"] for ref in traced_refs} == {"ningbo"}

    artifact_path = tmp_path / "k500_ref_ids.json"
    write_k500_ref_ids_artifact(manifest, artifact_path)
    k500_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(k500_payload["centers"]) == {"ningbo"}


def test_pn2021c_strong_raw_candidates_use_seed20260601_ref_exclusion():
    config = _load("pn2021c_effnet_v7_strong_raw_candidates_smoke.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert config["paper_protocol"]["kshot"]["seed"] == 20260601
    assert config["paper_protocol"]["kshot"]["subset_seed"] == 20260601
    assert len(commands) == 5
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert _option_value(argv, "--exclude_ref_ids").endswith(
            f"/{center}/k500_seed20260601/{center}_real_k500_seed20260601.ref_meta.json"
        )
        assert _option_value(argv, "--severities") == "5"
        assert _option_value(argv, "--num_workers") == "0"
        assert _option_value(argv, "--centers") == "ningbo"


def test_effnet_direct_smoke_command_limits_runtime_eval_samples():
    config = _load("effnet_direct_k500_v6_smoke.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    postprocess_commands = build_postprocess_commands(config)

    assert config["experiment"]["status"] == "smoke-only"
    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert argv[1].endswith("scripts/paper/run_direct_finetune_k500_20260516.py")
    assert _option_value(argv, "--epochs") == "1"
    assert _option_value(argv, "--num_workers") == "0"
    assert _option_value(argv, "--eval_min_pos") == "2"
    assert _option_value(argv, "--eval_pn2021_limit") == "64"
    assert _option_value(argv, "--out_root").endswith("/runs/effnet_direct_k500_v6_smoke/pytest_run")
    assert len(postprocess_commands) == 3


def test_effnet_vae_lhat_smoke_command_limits_runtime_work():
    config = _load("effnet_vae_lhat_k500_v6_smoke.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    postprocess_commands = build_postprocess_commands(config)

    assert config["experiment"]["status"] == "smoke-only"
    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert argv[1].endswith("scripts/paper/run_effnet_latent_augmix_stage3_20260524.py")
        assert _option_value(argv, "--center") == center
        assert _option_value(argv, "--epochs") == "1"
        assert _option_value(argv, "--num_workers") == "0"
        assert _option_value(argv, "--hull_M") == "4"
        assert _option_value(argv, "--hull_steps") == "1"
        assert _option_value(argv, "--k_anchor") == "32"
        assert _option_value(argv, "--pgd_batch") == "8"
        assert _option_value(argv, "--train_batch_size") == "32"
        assert _option_value(argv, "--eval_batch_size") == "64"
        assert _option_value(argv, "--eval_min_pos") == "2"
        assert _option_value(argv, "--eval_pn2021_limit") == "64"
        assert _option_value(argv, "--latent_augmix_width") == "2"
        assert _option_value(argv, "--latent_augmix_depth") == "1"
        assert _option_value(argv, "--latent_augmix_severity") == "1"
        assert _option_value(argv, "--out_root").endswith("/runs/effnet_vae_lhat_k500_v6_smoke/pytest_run")
    assert len(postprocess_commands) == 3


def test_ecgfounder_vae_lhat_smoke_command_limits_runtime_work():
    config = _load("ecgfounder_vae_lhat_k500_v6_smoke.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    postprocess_commands = build_postprocess_commands(config)

    assert config["experiment"]["status"] == "smoke-only"
    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        assert argv[1].endswith("scripts/paper/run_ecgfounder_vae_only_lhat_head_ft_20260523.py")
        assert _option_value(argv, "--centers") == center
        assert _option_value(argv, "--out_dir").endswith(
            f"/runs/ecgfounder_vae_lhat_k500_v6_smoke/pytest_run/{center}"
        )
        assert _option_value(argv, "--epochs") == "1"
        assert _option_value(argv, "--eval_every") == "1"
        assert _option_value(argv, "--source_train_limit") == "64"
        assert _option_value(argv, "--batch_size") == "64"
        assert _option_value(argv, "--eval_batch_size") == "512"
        assert _option_value(argv, "--k") == "500"
        assert _option_value(argv, "--k_anchor") == "16"
        assert _option_value(argv, "--hull_m") == "4"
        assert _option_value(argv, "--hull_lambda") == "0.05"
        assert _option_value(argv, "--hull_steps") == "1"
        assert _option_value(argv, "--pgd_batch") == "8"
        assert _option_value(argv, "--target_real_weight") == "20.0"
        assert _option_value(argv, "--adv_weight") == "5.0"
        assert "--report_drop_all_zero_pn2021" in argv
    assert len(postprocess_commands) == 3


def test_pn2021_eval_command_audit_rejects_missing_ref_exclusion():
    config = _load("pn2021_eval_v6_refexcluded.yaml")
    config = copy.deepcopy(config)
    config["runner"]["argv"] = _remove_option_pair(config["runner"]["argv"], "--exclude_ref_ids")
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="missing required option --exclude_ref_ids"):
        build_runner_commands(config)


def test_ecgfounder_checkpoint_matches_legacy_runtime_path():
    config = _load("ecgfounder_inithead_fullft_k500_v6.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    manifest = make_dry_run_manifest(
        config,
        commands=build_runner_commands(config),
        local_paths=paths,
        run_id="pytest_ecgfounder_checkpoint",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )
    checkpoint = next(
        item for item in manifest["artifact_trace"]["inputs"]["checkpoints"]
        if item["role"] == "model.checkpoint"
    )

    assert checkpoint["path"].endswith("/ecgfounder/checkpoint/12_lead_ECGFounder.pth")
    assert checkpoint["exists"] is True


def test_mapping_hash_mismatch_is_rejected():
    config = _load("effnet_direct_k500_v6.yaml")
    config = copy.deepcopy(config)
    config["paper_protocol"]["mapping_hash"] = "bad_hash"

    with pytest.raises(ConfigError, match="Mapping hash mismatch"):
        validate_experiment_config(config, repo_root=REPO)


def test_missing_ref_exclusion_is_rejected():
    config = _load("effnet_direct_k500_v6.yaml")
    config = copy.deepcopy(config)
    config["paper_protocol"]["kshot"]["exclude_refs_from_eval"] = False

    with pytest.raises(ConfigError, match="exclude_refs_from_eval"):
        validate_experiment_config(config, repo_root=REPO)


def test_selection_allowed_data_drift_is_rejected():
    config = _load("effnet_direct_k500_v6.yaml")
    config = copy.deepcopy(config)
    config["paper_protocol"]["selection"]["allowed_data"] = ["target_k500_internal_val"]

    with pytest.raises(ConfigError, match="selection.allowed_data"):
        validate_experiment_config(config, repo_root=REPO)


def test_runner_audit_rejects_heldout_target_selection_source():
    config = _load("ecgfounder_vae_lhat_k500_v6.yaml")
    config = copy.deepcopy(config)
    argv = config["runner"]["argv"]
    idx = argv.index("--selection_source")
    argv[idx + 1] = "pn2021_heldout"
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="held-out target selection data"):
        build_runner_commands(config)


def test_tracked_config_rejects_local_absolute_paths(tmp_path: Path):
    bad_config = tmp_path / "bad_experiment.yaml"
    bad_config.write_text(
        yaml.safe_dump({"experiment": {"name": "bad"}, "bad_path": "/home/linbinhao/private"}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="local/root absolute paths"):
        load_experiment_config(bad_config, LOCAL_EXAMPLE)


def test_known_legacy_root_alias_translates():
    assert translate_legacy_path("/root/autodl-tmp/ptbxl").startswith("/home/linbinhao/")
    assert (
        translate_legacy_path("/root/miniforge3/envs/ECGTwin/bin/python")
        == "/home/linbinhao/micromamba/envs/ECGTwin/bin/python"
    )


def test_local_config_accepts_known_root_aliases(tmp_path: Path):
    local = yaml.safe_load(LOCAL_EXAMPLE.read_text(encoding="utf-8"))
    local["paths"]["data_root"] = "/root/autodl-tmp"
    local["paths"]["model_root"] = "/root/autodl-tmp/models"
    local["paths"]["output_root"] = "/root/autodl-tmp/runs"
    local["paths"]["cache_root"] = "/root/autodl-tmp/cache"
    local["paths"]["tmp_root"] = "/root/autodl-tmp/tmp"
    local["python"]["executable"] = "/root/miniforge3/envs/ECGTwin/bin/python"
    local_path = tmp_path / "local_root_alias.yaml"
    local_path.write_text(yaml.safe_dump(local), encoding="utf-8")

    config = load_experiment_config(
        REPO / "configs" / "experiments" / "effnet_direct_k500_v6.yaml",
        local_path,
        runtime_context={"run_id": "pytest_root_alias"},
    )
    paths = validate_local_paths(config)
    assert paths["data_root"].startswith("/home/linbinhao/")
    assert paths["python_executable"].startswith("/home/linbinhao/")


def test_local_config_rejects_unknown_root_path(tmp_path: Path):
    local = yaml.safe_load(LOCAL_EXAMPLE.read_text(encoding="utf-8"))
    local["paths"]["data_root"] = "/root/unknown-dataset-root"
    local_path = tmp_path / "local_bad_root.yaml"
    local_path.write_text(yaml.safe_dump(local), encoding="utf-8")

    config = load_experiment_config(
        REPO / "configs" / "experiments" / "effnet_direct_k500_v6.yaml",
        local_path,
        runtime_context={"run_id": "pytest_bad_root"},
    )
    with pytest.raises(PathSafetyError, match="root-era path"):
        validate_local_paths(config)


def test_dry_run_manifest_never_marks_child_scripts_invoked():
    config = _load("effnet_direct_k500_v6.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    cli_args = argparse.Namespace(dry_run=True, write_plan=False)

    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_dry_run",
        cli_args=cli_args,
    )

    assert manifest["status"] == "dry_run"
    assert manifest["manifest_schema_version"] == 2
    assert manifest["safety"]["legacy_child_scripts_invoked"] is False
    assert manifest["safety"]["gpu_launch_requires_cuda_visible_devices"] is True
    assert manifest["paper_protocol"]["selection"]["forbid_heldout_target_labels"] is True
    assert manifest["artifact_trace"]["protocol_audit"]["passed"] is True


def _dry_manifest(config_name: str) -> dict:
    config = _load(config_name)
    paths = validate_experiment_config(config, repo_root=REPO)
    return make_dry_run_manifest(
        config,
        commands=build_runner_commands(config),
        local_paths=paths,
        run_id=f"pytest_{config_name}",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )


def test_manifest_contains_metric_views_and_mapping_metadata():
    manifest = _dry_manifest("effnet_direct_k500_v6.yaml")
    trace = manifest["artifact_trace"]

    assert trace["metrics"]["mapping_version"] == "v7_super5_sjr_rgq_review_20260528"
    assert trace["metrics"]["mapping_hash"] == "555ec85d5b51"
    assert trace["metrics"]["class_order"] == ["CD", "HYP", "MI", "NORM", "STTC"]
    assert "pn2021_all_zero_kept_refexcluded" in trace["metrics"]["views"]
    assert "pn2021_drop_all_zero_refexcluded" in trace["metrics"]["views"]


def test_manifest_records_direct_k500_checkpoint_refs_and_outputs():
    manifest = _dry_manifest("effnet_direct_k500_v6.yaml")
    trace = manifest["artifact_trace"]

    checkpoints = trace["inputs"]["checkpoints"]
    assert any(item["role"] == "model.init_checkpoint" for item in checkpoints)
    refs = trace["inputs"]["k500_refs"]
    assert len(refs) == 4
    assert {item["center"] for item in refs} == {"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"}
    for item in refs:
        assert item["k"] == 500
        assert item["seed"] == 20260531
        assert item["ref_meta_json"]["path"].endswith("_real_k500_seed20260531.ref_meta.json")
        assert item["signals_npz"]["path"].endswith("_real_k500_seed20260531.signals.npz")
        assert item["latent_npz"] is None
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    assert all(any(art["role"] == "eval_result" for art in run["expected_artifacts"]) for run in child_runs)
    assert any("eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json" in art["path"]
               for run in child_runs for art in run["expected_artifacts"])


def test_manifest_records_vae_direct_init_anchor_refs_and_outputs():
    manifest = _dry_manifest("effnet_vae_lhat_k500_v6.yaml")
    trace = manifest["artifact_trace"]

    checkpoints = trace["inputs"]["checkpoints"]
    assert any(item["role"] == "command.init_ckpt" and "paper_direct_finetune_k500_20260516" in item["path"]
               for item in checkpoints)
    refs = trace["inputs"]["k500_refs"]
    assert len(refs) == 4
    for item in refs:
        assert "seed42" not in item["anchor_base"]
        assert item["latent_npz"] is not None
        assert item["latent_npz"]["path"].endswith("_real_k500_seed20260531.latent.npz")
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    assert all("lam0p05" in run["child_run_dir"] for run in child_runs)
    assert all("wlat0p25" in run["child_run_dir"] for run in child_runs)
    for run in child_runs:
        expected_leaf = build_effnet_vae_lhat_run_leaf(
            {
                "center": run["center"],
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
                "epochs": 30,
                "seed": 20260531,
            }
        )
        assert Path(run["child_run_dir"]).name == expected_leaf
    assert any("eval_result_v7_exclrefs_crop1000.json" in art["path"]
               for run in child_runs for art in run["expected_artifacts"])


def test_manifest_records_ecgfounder_init_head_refs_and_outputs():
    manifest = _dry_manifest("ecgfounder_inithead_fullft_k500_v6.yaml")
    trace = manifest["artifact_trace"]

    assert any(item["role"] == "model.checkpoint" for item in trace["inputs"]["checkpoints"])
    assert len(trace["inputs"]["init_heads"]) == 4
    assert all(item["path"].endswith("best_head.pt") for item in trace["inputs"]["init_heads"])
    refs = trace["inputs"]["k500_refs"]
    assert len(refs) == 4
    assert all(item["ref_meta_json"]["path"].endswith("_real_k500_seed20260531.ref_meta.json") for item in refs)
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    assert any("source_plus_target_val_auprc" in run["child_run_dir"] for run in child_runs)
    assert any(art["path"].endswith("/eval_result.json")
               for run in child_runs for art in run["expected_artifacts"])


def test_manifest_records_ecgfounder_direct_headft_refs_features_and_outputs():
    manifest = _dry_manifest("ecgfounder_direct_k500_v6.yaml")
    trace = manifest["artifact_trace"]

    assert trace["inputs"]["checkpoints"] == []
    assert any(item["role"] == "ecgfounder.linear_probe.best_head" for item in trace["inputs"]["init_heads"])
    assert any(
        item["role"] == "ecgfounder.linear_probe.pn2021_features"
        and item["path"].endswith("pn2021_ecgfounder_features_official_ptbxl_eval.npz")
        for item in trace["inputs"]["data_caches"]
    )
    refs = trace["inputs"]["k500_refs"]
    assert len(refs) == 4
    assert {item["center"] for item in refs} == {"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"}
    assert all(item["ref_meta_json"]["path"].endswith("_real_k500_seed20260531.ref_meta.json") for item in refs)
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    assert all("fromK500_headft_ep50_seed20260531" in run["child_run_dir"] for run in child_runs)
    for run in child_runs:
        expected = ecgfounder_kshot_head_run_dir(
            Path(run["output_root"]),
            center=run["center"],
            k=500,
            source_k=500,
            epochs=50,
            seed=20260531,
        )
        assert Path(run["child_run_dir"]) == expected
    assert all(any(art["role"] == "best_head" for art in run["expected_artifacts"]) for run in child_runs)
    assert all(any(art["role"] == "eval_result" for art in run["expected_artifacts"]) for run in child_runs)


def test_manifest_records_ecgfounder_vae_lhat_refs_heads_features_and_outputs():
    manifest = _dry_manifest("ecgfounder_vae_lhat_k500_v6.yaml")
    trace = manifest["artifact_trace"]

    assert any(item["role"] == "model.checkpoint" for item in trace["inputs"]["checkpoints"])
    assert any(item["role"] == "command.checkpoint" for item in trace["inputs"]["checkpoints"])
    assert any(item["role"] == "ecgfounder.linear_probe.best_head" for item in trace["inputs"]["init_heads"])
    assert sum(item["role"] == "ecgfounder.k500_base_head.best_head" for item in trace["inputs"]["init_heads"]) == 4
    assert any(
        item["role"] == "ecgfounder.linear_probe.ptbxl_features"
        and item["path"].endswith("ptbxl_ecgfounder_features_official_ptbxl_eval.npz")
        for item in trace["inputs"]["data_caches"]
    )
    assert any(
        item["role"] == "ecgfounder.linear_probe.pn2021_features"
        and item["path"].endswith("pn2021_ecgfounder_features_official_ptbxl_eval.npz")
        for item in trace["inputs"]["data_caches"]
    )
    refs = trace["inputs"]["k500_refs"]
    assert len(refs) == 4
    assert {item["center"] for item in refs} == {"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"}
    assert all(item["latent_npz"] is not None for item in refs)
    assert all("paper_vae_only_lhat_kcurve_20260518/subsets" in item["anchor_base"] for item in refs)
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    assert all("lam0p15" in run["child_run_dir"] for run in child_runs)
    for run in child_runs:
        expected = ecgfounder_lhat_run_dir(
            Path(run["output_root"]),
            center=run["center"],
            k=500,
            hull_m=20,
            hull_lambda="0.15",
            epochs=20,
            seed=20260531,
        )
        assert Path(run["child_run_dir"]) == expected
    assert all(any(art["role"] == "best_head" for art in run["expected_artifacts"]) for run in child_runs)
    assert all(any(art["role"] == "eval_result" for art in run["expected_artifacts"]) for run in child_runs)


def _small_kshot_manifest(tmp_path: Path, *, bad_count: bool = False) -> dict:
    centers = ["ningbo", "georgia"]
    refs = []
    for center in centers:
        center_dir = tmp_path / center
        center_dir.mkdir()
        ids = [f"{center}_A", f"{center}_B"]
        if bad_count and center == "georgia":
            ids = [f"{center}_A"]
        meta_path = center_dir / f"{center}_real_k2_seed20260531.ref_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "center": center,
                    "K": len(ids),
                    "selection_seed": 20260531,
                    "ref_record_ids": ids,
                    "source_indices": list(range(len(ids))),
                    "policy": "pytest fixed subset",
                    "parent": "/root/autodl-tmp/old/provenance-only.json",
                }
            ),
            encoding="utf-8",
        )
        signals_path = center_dir / f"{center}_real_k2_seed20260531.signals.npz"
        signals_path.write_bytes(f"signals-{center}".encode("utf-8"))
        latent_path = center_dir / f"{center}_real_k2_seed20260531.latent.npz"
        latent_path.write_bytes(f"latent-{center}".encode("utf-8"))
        refs.append(
            {
                "center": center,
                "k": 2,
                "seed": 20260531,
                "anchor_base": str(center_dir / f"{center}_real_k2_seed20260531"),
                "ref_meta_json": {"role": "kshot_ref_meta", "path": str(meta_path), "required": True},
                "signals_npz": {"role": "kshot_signals", "path": str(signals_path), "required": True},
                "latent_npz": {"role": "kshot_latents", "path": str(latent_path), "required": True},
            }
        )
    return {
        "run_id": "pytest_kshot",
        "paper_protocol": {
            "centers": {"target_4": centers},
            "kshot": {
                "k": 2,
                "seed": 20260531,
                "subset_seed": 20260531,
                "exclude_refs_from_eval": True,
            },
            "selection": {"policy": "pytest fixed subset"},
        },
        "artifact_trace": {
            "metrics": {
                "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                "mapping_hash": "555ec85d5b51",
                "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            },
            "selection_policy": {
                "policy": "k500_internal_val_plus_source_floor",
                "allowed_data": [
                    "target_k500_train_split",
                    "target_k500_internal_val",
                    "ptbxl_source_floor",
                ],
                "forbid_heldout_target_labels": True,
                "forbid_full_target_distribution_tuning": True,
            },
            "inputs": {"k500_refs": refs},
            "expected_outputs": {"child_runs": []},
        },
    }


def test_write_k500_ref_ids_artifact_and_verify_launch_artifact(tmp_path: Path):
    manifest = _small_kshot_manifest(tmp_path)
    artifact_path = tmp_path / "run" / "k500_ref_ids.json"
    write_k500_ref_ids_artifact(manifest, artifact_path)
    write_selection_record_artifact(manifest, artifact_path.parent / "selection.json")
    manifest = attach_launch_artifacts(manifest, run_dir=artifact_path.parent)

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["paper_protocol"]["k"] == 2
    assert set(payload["centers"]) == {"ningbo", "georgia"}
    assert len(payload["centers"]["ningbo"]["ref_record_ids_ordered"]) == 2
    assert payload["centers"]["ningbo"]["parent_provenance"].startswith("/root/autodl-tmp")

    report = verify_required_artifacts(manifest)
    missing_roles = {item["role"] for item in report["missing"]}
    assert report["content_errors"] == []
    assert "k500_ref_ids" not in missing_roles
    assert "selection_record" not in missing_roles


def test_write_selection_record_artifact_denies_target_heldout_selection(tmp_path: Path):
    manifest = _small_kshot_manifest(tmp_path)
    output_path = tmp_path / "run" / "selection.json"
    write_selection_record_artifact(manifest, output_path)
    manifest = attach_launch_artifacts(manifest, run_dir=output_path.parent)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "selection_record"
    assert payload["selection_policy"]["policy"] == "k500_internal_val_plus_source_floor"
    assert payload["selection_safety"]["heldout_target_labels_used_for_selection"] is False
    assert payload["selection_safety"]["full_target_distribution_used_for_tuning"] is False
    assert payload["selection_safety"]["forbidden_reference_found"] is False

    report = verify_required_artifacts(manifest)
    assert report["content_errors"] == []
    assert any(item["role"] == "selection_record" and item.get("content_verified") for item in report["verified_artifacts"])


def test_k500_ref_ids_generation_rejects_bad_count(tmp_path: Path):
    manifest = _small_kshot_manifest(tmp_path, bad_count=True)
    with pytest.raises(LaunchError, match="expected 2"):
        write_k500_ref_ids_artifact(manifest, tmp_path / "k500_ref_ids.json")


def _artifact_manifest(eval_path: Path, *, mapping_hash: str = "555ec85d5b51") -> dict:
    return {
        "artifact_trace": {
            "metrics": {
                "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                "mapping_hash": "555ec85d5b51",
                "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
            },
            "expected_outputs": {
                "child_runs": [
                    {
                        "command_index": 0,
                        "center": "ningbo",
                        "expected_artifacts": [
                            {"role": "eval_result", "path": str(eval_path), "required": True},
                        ],
                    }
                ]
            },
        },
        "label_mapping_for_test": mapping_hash,
    }


def _training_log_manifest(path: Path, required_metrics: list[str]) -> dict:
    return {
        "artifact_trace": {
            "expected_outputs": {
                "required_epoch_metrics": required_metrics,
                "child_runs": [
                    {
                        "command_index": 0,
                        "center": "ningbo",
                        "expected_artifacts": [
                            {"role": "training_log", "path": str(path), "required": True},
                        ],
                    }
                ],
            },
        },
    }


def test_verify_required_artifacts_checks_eval_mapping_metadata(tmp_path: Path):
    eval_path = tmp_path / "eval_result.json"
    eval_path.write_text(
        json.dumps(
            {
                "label_mapping": {
                    "pn2021_super5": {
                        "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                        "mapping_hash": "555ec85d5b51",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report = verify_required_artifacts(_artifact_manifest(eval_path))

    assert report["passed"] is True
    assert report["n_checked"] == 1
    assert report["verified_artifacts"][0]["mapping_verified"] is True


def test_verify_required_artifacts_checks_training_log_epoch_metrics(tmp_path: Path):
    log_path = tmp_path / "training_log.json"
    log_path.write_text(
        json.dumps(
            {
                "epochs": [
                    {
                        "epoch": 1,
                        "train_loss": 0.7,
                        "attack_success": {"success_rate": 0.4, "loss_gain_mean": 0.01},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = verify_required_artifacts(
        _training_log_manifest(
            log_path,
            ["epoch", "train_loss", "attack_success.success_rate", "attack_success.loss_gain_mean"],
        )
    )

    assert report["passed"] is True
    assert report["content_errors"] == []
    assert report["verified_artifacts"][0]["epoch_metric_verified"] is True


def test_verify_required_artifacts_rejects_missing_training_log_epoch_metric(tmp_path: Path):
    log_path = tmp_path / "training_log.json"
    log_path.write_text(json.dumps([{"epoch": 1, "train_loss": 0.7}]), encoding="utf-8")

    report = verify_required_artifacts(
        _training_log_manifest(log_path, ["epoch", "train_loss", "attack_success.success_rate"])
    )

    assert report["passed"] is False
    assert report["content_errors"]
    assert report["content_errors"][0]["missing_metrics"] == ["attack_success.success_rate"]


def test_verify_required_artifacts_rejects_mapping_mismatch(tmp_path: Path):
    eval_path = tmp_path / "eval_result.json"
    eval_path.write_text(
        json.dumps(
            {
                "label_mapping": {
                    "pn2021_super5": {
                        "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                        "mapping_hash": "bad_hash",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report = verify_required_artifacts(_artifact_manifest(eval_path))

    assert report["passed"] is False
    assert report["mapping_errors"]


def test_verify_required_artifacts_rejects_missing_mapping_metadata(tmp_path: Path):
    eval_path = tmp_path / "eval_result.json"
    eval_path.write_text(json.dumps({"target_view": {"avg_macro_auroc": 0.8}}), encoding="utf-8")

    report = verify_required_artifacts(_artifact_manifest(eval_path))

    assert report["passed"] is False
    assert report["missing"] == []
    assert report["mapping_errors"]
    assert report["mapping_errors"][0]["mapping_verified"] is False


def test_run_legacy_commands_creates_short_runtime_env_dirs(tmp_path: Path):
    eval_path = tmp_path / "missing_outputs" / "child_eval.json"
    tmp_env = tmp_path / "missing_short_tmp"
    cache_env = tmp_path / "missing_cache"
    script = tmp_path / "write_eval.py"
    script.write_text(
        """
import json
import os
import sys
from pathlib import Path

assert Path(os.environ["TMPDIR"]).is_dir()
assert Path(os.environ["XDG_CACHE_HOME"]).is_dir()
print("child stdout marker")
print("child stderr marker", file=sys.stderr)
Path(sys.argv[1]).write_text(json.dumps({
    "label_mapping": {
        "pn2021_super5": {
            "mapping_version": "v7_super5_sjr_rgq_review_20260528",
            "mapping_hash": "555ec85d5b51",
        }
    }
}))
""".strip(),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest = _artifact_manifest(eval_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    command = {
        "name": "pytest_env_dirs",
        "argv": [sys.executable, str(script), str(eval_path)],
        "cwd": str(REPO),
        "env": {
            "TMPDIR": str(tmp_env),
            "XDG_CACHE_HOME": str(cache_env),
        },
    }

    run_legacy_commands([command], run_dir=tmp_path / "run", manifest_path=manifest_path)

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert tmp_env.is_dir()
    assert cache_env.is_dir()
    assert updated["status"] == "succeeded"
    assert updated["execution_started_at_utc"]
    assert updated["child_commands_finished_at_utc"]
    assert updated["artifact_verification"]["passed"] is True
    assert len(updated["command_runs"]) == 1
    child_run = updated["command_runs"][0]
    assert child_run["command_index"] == 0
    assert child_run["name"] == "pytest_env_dirs"
    assert child_run["returncode"] == 0
    assert child_run["status"] == "succeeded"
    assert Path(child_run["log_path"]).name == "command_00.log"
    assert Path(child_run["stdout_log_path"]).name == "command_00.stdout.log"
    assert Path(child_run["stderr_log_path"]).name == "command_00.stderr.log"
    assert Path(child_run["stdout_log_path"]).read_text(encoding="utf-8").strip() == "child stdout marker"
    assert Path(child_run["stderr_log_path"]).read_text(encoding="utf-8").strip() == "child stderr marker"
    assert "child stdout marker" in Path(child_run["log_path"]).read_text(encoding="utf-8")
    assert "child stderr marker" in Path(child_run["log_path"]).read_text(encoding="utf-8")
    assert "child stdout marker" in (tmp_path / "run" / "stdout.log").read_text(encoding="utf-8")
    assert "child stderr marker" in (tmp_path / "run" / "stderr.log").read_text(encoding="utf-8")
    assert child_run["started_at_utc"]
    assert child_run["finished_at_utc"]


def test_run_legacy_commands_records_failed_child_run(tmp_path: Path):
    eval_path = tmp_path / "missing_eval.json"
    script = tmp_path / "fail.py"
    script.write_text(
        "import sys\nprint('failed stdout marker')\nprint('failed stderr marker', file=sys.stderr)\nsys.exit(7)\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(_artifact_manifest(eval_path)), encoding="utf-8")
    command = {
        "name": "pytest_failed_child",
        "argv": [sys.executable, str(script)],
        "cwd": str(REPO),
        "env": {},
    }

    with pytest.raises(LaunchError, match="return code 7"):
        run_legacy_commands([command], run_dir=tmp_path / "run", manifest_path=manifest_path, base_env={})

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["status"] == "failed"
    assert updated["failed_command_index"] == 0
    assert len(updated["command_runs"]) == 1
    child_run = updated["command_runs"][0]
    assert child_run["name"] == "pytest_failed_child"
    assert child_run["returncode"] == 7
    assert child_run["status"] == "failed"
    assert Path(child_run["log_path"]).name == "command_00.log"
    assert Path(child_run["stdout_log_path"]).read_text(encoding="utf-8").strip() == "failed stdout marker"
    assert Path(child_run["stderr_log_path"]).read_text(encoding="utf-8").strip() == "failed stderr marker"


def test_run_postprocess_commands_verifies_expected_artifacts(tmp_path: Path):
    eval_path = tmp_path / "eval_result.json"
    eval_path.write_text(
        json.dumps(
            {
                "label_mapping": {
                    "pn2021_super5": {
                        "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                        "mapping_hash": "555ec85d5b51",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "postprocess" / "metrics_long.csv"
    script = tmp_path / "write_metrics.py"
    script.write_text(
        """
import sys
from pathlib import Path

Path(sys.argv[1]).write_text("run_id,metric,value\\npytest,macro_auroc,0.9\\n", encoding="utf-8")
print("postprocess stdout marker")
print("postprocess stderr marker", file=sys.stderr)
""".strip(),
        encoding="utf-8",
    )
    manifest = _artifact_manifest(eval_path)
    manifest["execution_started_at_utc"] = "2026-05-28T00:00:00+00:00"
    manifest["artifact_trace"]["expected_outputs"]["postprocess_runs"] = [
        {
            "command_index": 0,
            "name": "pytest_postprocess",
            "expected_artifacts": [
                {"role": "metrics_long", "path": str(metrics_path), "required": True},
            ],
        }
    ]
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    commands = [
        {
            "name": "pytest_postprocess",
            "cwd": str(REPO),
            "env": {},
            "argv": [sys.executable, str(script), str(metrics_path)],
        }
    ]

    run_postprocess_commands(commands, run_dir=tmp_path / "run", manifest_path=manifest_path, base_env={})

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["status"] == "succeeded"
    assert updated["finished_at_utc"]
    assert updated["duration_seconds"] >= 0
    assert updated["artifact_verification"]["passed"] is True
    assert len(updated["postprocess_runs"]) == 1
    postprocess_run = updated["postprocess_runs"][0]
    assert postprocess_run["command_index"] == 0
    assert postprocess_run["name"] == "pytest_postprocess"
    assert postprocess_run["returncode"] == 0
    assert postprocess_run["status"] == "succeeded"
    assert Path(postprocess_run["log_path"]).name == "postprocess_00.log"
    assert Path(postprocess_run["stdout_log_path"]).name == "postprocess_00.stdout.log"
    assert Path(postprocess_run["stderr_log_path"]).name == "postprocess_00.stderr.log"
    assert "postprocess stdout marker" in Path(postprocess_run["stdout_log_path"]).read_text(encoding="utf-8")
    assert "postprocess stderr marker" in Path(postprocess_run["stderr_log_path"]).read_text(encoding="utf-8")
    assert "postprocess stdout marker" in (tmp_path / "run" / "stdout.log").read_text(encoding="utf-8")
    assert "postprocess stderr marker" in (tmp_path / "run" / "stderr.log").read_text(encoding="utf-8")
    assert postprocess_run["started_at_utc"]
    assert postprocess_run["finished_at_utc"]
    assert metrics_path.exists()
    assert any(
        item.get("postprocess") and item["role"] == "metrics_long"
        for item in updated["artifact_verification"]["verified_artifacts"]
    )


def test_run_postprocess_commands_finalizes_no_postprocess_run(tmp_path: Path):
    eval_path = tmp_path / "eval_result.json"
    script = tmp_path / "write_eval.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "label_mapping": {
        "pn2021_super5": {
            "mapping_version": "v7_super5_sjr_rgq_review_20260528",
            "mapping_hash": "555ec85d5b51",
        }
    }
}), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(_artifact_manifest(eval_path)), encoding="utf-8")
    command = {
        "name": "pytest_no_postprocess_child",
        "argv": [sys.executable, str(script), str(eval_path)],
        "cwd": str(REPO),
        "env": {},
    }

    run_legacy_commands([command], run_dir=tmp_path / "run", manifest_path=manifest_path, base_env={})
    run_postprocess_commands([], run_dir=tmp_path / "run", manifest_path=manifest_path, base_env={})

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["status"] == "succeeded"
    assert updated["finished_at_utc"]
    assert updated["duration_seconds"] >= 0
    assert updated["artifact_verification"]["passed"] is True


def test_verify_required_inputs_rejects_missing_traced_checkpoint(tmp_path: Path):
    manifest = {
        "artifact_trace": {
            "inputs": {
                "checkpoints": [
                    {"role": "model.init_checkpoint", "path": str(tmp_path / "missing.pt"), "required": True}
                ]
            }
        }
    }
    report = verify_required_inputs(manifest)

    assert report["passed"] is False
    assert report["missing"][0]["role"] == "model.init_checkpoint"


def test_run_legacy_commands_fails_if_required_artifact_missing(tmp_path: Path):
    missing_eval = tmp_path / "missing_eval.json"
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(_artifact_manifest(missing_eval)), encoding="utf-8")
    commands = [
        {
            "name": "noop",
            "cwd": str(REPO),
            "env": {},
            "argv": [sys.executable, "-c", "print('child returned zero without artifacts')"],
        }
    ]

    with pytest.raises(LaunchError, match="artifact verification failed"):
        run_legacy_commands(commands, run_dir=tmp_path, manifest_path=manifest_path, base_env={})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["artifact_verification"]["passed"] is False
    assert manifest["artifact_verification"]["missing"][0]["path"] == str(missing_eval)


def test_require_cuda_visible_devices_rejects_missing_or_broad_values():
    with pytest.raises(LaunchError, match="CUDA_VISIBLE_DEVICES"):
        require_cuda_visible_devices({})
    with pytest.raises(LaunchError, match="too broad"):
        require_cuda_visible_devices({"CUDA_VISIBLE_DEVICES": "all"})
    with pytest.raises(LaunchError, match="multiple GPUs"):
        require_cuda_visible_devices({"CUDA_VISIBLE_DEVICES": "0,1"})
    assert require_cuda_visible_devices({"CUDA_VISIBLE_DEVICES": "3"}) == "3"


def test_runner_env_rejects_cuda_visible_devices_override():
    config = _load("effnet_direct_k500_v6.yaml")
    config = copy.deepcopy(config)
    config["runner"].setdefault("env", {})["CUDA_VISIBLE_DEVICES"] = "0"
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="CUDA_VISIBLE_DEVICES"):
        build_runner_commands(config)


def test_check_nvidia_smi_parses_fake_runner():
    def fake_runner(cmd, check, text, capture_output):
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="0, NVIDIA RTX 4090, 100, 24564, 2\n",
            stderr="",
        )

    snapshot = check_nvidia_smi(runner=fake_runner)
    assert snapshot["gpus"][0]["index"] == "0"
    assert snapshot["gpus"][0]["memory_total_mb"] == "24564"


def test_prepare_output_dir_rejects_nonempty_without_resume_or_force(tmp_path: Path):
    config = _load("effnet_direct_k500_v6.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "leftover.txt").write_text("old", encoding="utf-8")
    paths = {**paths, "write_boundary": str(tmp_path)}

    with pytest.raises(LaunchError, match="non-empty output directory"):
        prepare_output_dir(
            out_dir,
            local_paths=paths,
            run_id="run1",
            config_hash="abc",
        )


def test_prepare_output_dir_allows_matching_resume_manifest(tmp_path: Path):
    config = _load("effnet_direct_k500_v6.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    manifest = make_dry_run_manifest(
        config,
        commands=build_runner_commands(config),
        local_paths=paths,
        run_id="run1",
        cli_args=argparse.Namespace(dry_run=True, write_plan=True),
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    paths = {**paths, "write_boundary": str(tmp_path)}

    resolved = prepare_output_dir(
        out_dir,
        local_paths=paths,
        run_id="run1",
        config_hash=manifest["config_hash_sha256"],
        resume=True,
    )
    assert resolved == out_dir.resolve()


def test_default_run_dir_uses_dry_and_dated_roots():
    config = _load("effnet_direct_k500_v6.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)

    dry = default_run_dir(paths, "abc", dry_run=True)
    real = default_run_dir(paths, "abc", dry_run=False)
    assert dry.name == "abc"
    assert dry.parent.name == "dry_runs"
    assert real.name == "abc"
    assert real.parent.name.isdigit()


def test_execute_mode_requires_cuda_before_writing_plan(tmp_path: Path):
    out_dir = tmp_path / "execute_should_not_write"
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_experiment.py"),
            "--config",
            str(REPO / "configs" / "experiments" / "effnet_direct_k500_v6.yaml"),
            "--local-config",
            str(LOCAL_EXAMPLE),
            "--run-id",
            "pytest_execute_missing_cuda",
            "--output-dir",
            str(out_dir),
            "--execute",
        ],
        cwd=str(REPO),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 3
    assert "CUDA_VISIBLE_DEVICES" in proc.stderr
    assert not out_dir.exists()


def test_run_experiment_execute_path_has_finalizer_binding():
    import scripts.run_experiment as run_experiment

    assert callable(run_experiment.finalize_run_record)


def test_write_plan_command_sh_includes_env_prefixes(tmp_path: Path):
    out_dir = Path.home() / ".cache" / "ecg_adv_gen_pytest" / f"{tmp_path.name}_dry_plan"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_experiment.py"),
            "--config",
            str(REPO / "configs" / "experiments" / "effnet_direct_k500_v7_sjr_rgq.yaml"),
            "--local-config",
            str(LOCAL_EXAMPLE),
            "--run-id",
            "pytest_dry_plan",
            "--output-dir",
            str(out_dir),
            "--dry-run",
            "--write-plan",
        ],
        cwd=str(REPO),
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    command_text = (out_dir / "command.sh").read_text(encoding="utf-8")
    assert command_text.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert "ECG_ADV_GEN_DATA_ROOT=" in command_text
    assert "scripts/paper/run_direct_finetune_k500_20260516.py" in command_text
    k500_ref_ids = out_dir / "k500_ref_ids.json"
    assert k500_ref_ids.exists()
    selection_json = out_dir / "selection.json"
    assert selection_json.exists()
    data_manifest = out_dir / "data_manifest.json"
    assert data_manifest.exists()
    data_payload = json.loads(data_manifest.read_text(encoding="utf-8"))
    assert data_payload["contract"]["classifier_fs"] == 100
    assert data_payload["contract"]["target_dataset"] == "pn2021"
    k500_payload = json.loads(k500_ref_ids.read_text(encoding="utf-8"))
    assert k500_payload["paper_protocol"]["k"] == 500
    assert set(k500_payload["centers"]) == {"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"}
    selection_payload = json.loads(selection_json.read_text(encoding="utf-8"))
    assert selection_payload["selection_policy"]["policy"] == "k500_internal_val_plus_source_floor"
    assert selection_payload["selection_safety"]["heldout_target_labels_used_for_selection"] is False
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    launch_roles = {
        item["role"]
        for item in manifest["artifact_trace"]["expected_outputs"]["launch_artifacts"]
    }
    assert "k500_ref_ids" in launch_roles
    assert "data_manifest" in launch_roles
    assert "selection_record" in launch_roles
    shutil.rmtree(out_dir)
