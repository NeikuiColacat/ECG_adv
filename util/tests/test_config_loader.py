"""CPU-only tests for the YAML experiment configuration layer."""

from __future__ import annotations

import argparse
import copy
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
from ecg_adv_gen.config.paths import PathSafetyError, translate_legacy_path, validate_local_paths
from ecg_adv_gen.models import ecgfounder_kshot_head_run_dir, ecgfounder_lhat_run_dir
from ecg_adv_gen.run_naming import (
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
        ("effnet_vae_lhat_k500_v6.yaml", 4),
        ("effnet_vae_lhat_k500_v6_smoke.yaml", 4),
        ("ecgfounder_direct_k500_v6.yaml", 1),
        ("ecgfounder_inithead_fullft_k500_v6.yaml", 4),
        ("ecgfounder_inithead_fullft_k500_v6_smoke.yaml", 4),
        ("ecgfounder_vae_lhat_k500_v6.yaml", 4),
        ("ecgfounder_vae_lhat_k500_v6_smoke.yaml", 4),
        ("pn2021_eval_v6_refexcluded.yaml", 4),
        ("pn2021_eval_v6_refexcluded_smoke.yaml", 4),
    ],
)
def test_tracked_configs_validate_and_expand_commands(config_name: str, expected_commands: int):
    config = _load(config_name)
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert paths["project_root"].endswith("ECG_adv_Gen")
    assert config["paper_protocol"]["mapping_version"] == "v6_super5_clinician_review_20260524"
    assert config["paper_protocol"]["mapping_hash"] == "3adc673a60ad"
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
    assert "metrics_long.csv" in logging_cfg["postprocess_artifacts"]
    assert "metrics_long.csv" not in logging_cfg["launch_artifacts"]
    assert "selection.json" not in logging_cfg["postprocess_artifacts"]


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


def test_command_protocol_audit_rejects_missing_vae_init_checkpoint():
    config = _load("effnet_vae_lhat_k500_v6.yaml")
    config = copy.deepcopy(config)
    config["runner"]["argv"] = _remove_option_pair(config["runner"]["argv"], "--init_ckpt")
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="missing required option --init_ckpt"):
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
            f"/pn2021_eval_v6_refexcluded/pytest_run/{center}/eval_result_v6_super5_refexcluded.json"
        )

    traced_checkpoints = manifest["artifact_trace"]["inputs"]["checkpoints"]
    assert any(item["role"] == "command.model_dir.best_model" for item in traced_checkpoints)
    assert len(manifest["artifact_trace"]["inputs"]["k500_refs"]) == 16
    assert len(postprocess_commands) == 3
    assert postprocess_commands[0]["argv"][1].endswith("scripts/export_metrics_long.py")
    assert "--expected-mapping-version" in postprocess_commands[0]["argv"]
    assert "v6_super5_clinician_review_20260524" in postprocess_commands[0]["argv"]
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
            f"/pn2021_eval_v6_refexcluded_smoke/pytest_run/{center}/eval_result_v6_super5_refexcluded_smoke.json"
        )


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

    assert trace["metrics"]["mapping_version"] == "v6_super5_clinician_review_20260524"
    assert trace["metrics"]["mapping_hash"] == "3adc673a60ad"
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
    assert any("eval_result_v6_super5_clinician_review_exclrefs_crop1000.json" in art["path"]
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
    assert any("eval_result_v6_exclrefs_crop1000.json" in art["path"]
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
                "mapping_version": "v6_super5_clinician_review_20260524",
                "mapping_hash": "3adc673a60ad",
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


def _artifact_manifest(eval_path: Path, *, mapping_hash: str = "3adc673a60ad") -> dict:
    return {
        "artifact_trace": {
            "metrics": {
                "mapping_version": "v6_super5_clinician_review_20260524",
                "mapping_hash": "3adc673a60ad",
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
                        "mapping_version": "v6_super5_clinician_review_20260524",
                        "mapping_hash": "3adc673a60ad",
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
                        "mapping_version": "v6_super5_clinician_review_20260524",
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
            "mapping_version": "v6_super5_clinician_review_20260524",
            "mapping_hash": "3adc673a60ad",
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
                        "mapping_version": "v6_super5_clinician_review_20260524",
                        "mapping_hash": "3adc673a60ad",
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
            "mapping_version": "v6_super5_clinician_review_20260524",
            "mapping_hash": "3adc673a60ad",
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


def test_write_plan_command_sh_includes_env_prefixes(tmp_path: Path):
    out_dir = Path.home() / ".cache" / "ecg_adv_gen_pytest" / f"{tmp_path.name}_dry_plan"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_experiment.py"),
            "--config",
            str(REPO / "configs" / "experiments" / "effnet_direct_k500_v6.yaml"),
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
