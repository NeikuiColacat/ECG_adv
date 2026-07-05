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
    run_managed_commands,
    run_postprocess_commands,
    validate_experiment_config,
    verify_required_inputs,
    verify_required_artifacts,
    write_k500_ref_ids_artifact,
    write_selection_record_artifact,
)
from ecg_adv_gen.config.entrypoints import managed_runner_script_names
from ecg_adv_gen.config.loader import _audit_runtime_path, audit_runner_commands, build_artifact_trace
from ecg_adv_gen.config.runner_audit import DISPATCHED_RUNNER_AUDIT_SCRIPT_NAMES, audit_runner_command
from ecg_adv_gen.config.paths import PathSafetyError, validate_local_paths
from ecg_adv_gen.evaluation.pn2021c_protocol import official_s5_depth23_composites
from ecg_adv_gen.run_naming import build_effnet_direct_run_leaf


REPO = Path(__file__).resolve().parents[2]
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"


def _load(name: str) -> dict:
    return load_experiment_config(
        REPO / "configs" / "experiments" / name,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest_run"},
    )


def _latest_mainline_config_names() -> list[str]:
    index = yaml.safe_load((REPO / "configs" / "active_scripts.yaml").read_text(encoding="utf-8"))
    return [Path(stage["config"]).name for stage in index["latest_mainline"]["stages"]]


def test_command_expansion_contract_covers_latest_mainline_stages_only():
    index = yaml.safe_load((REPO / "configs" / "active_scripts.yaml").read_text(encoding="utf-8"))
    latest_mainline_names = {Path(stage["config"]).name for stage in index["latest_mainline"]["stages"]}

    assert set(_latest_mainline_config_names()) == latest_mainline_names


@pytest.mark.parametrize("config_name", _latest_mainline_config_names())
def test_latest_mainline_configs_validate_and_expand_commands(config_name: str):
    config = _load(config_name)
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert paths["project_root"].endswith("ECG_adv_Gen")
    assert config["paper_protocol"]["mapping_version"] == "v7_super5_sjr_rgq_review_20260528"
    assert config["paper_protocol"]["mapping_hash"] == "555ec85d5b51"
    assert "pn2021_all_zero_kept_refexcluded" in config["evaluation"]["views"]
    assert "pn2021_drop_all_zero_refexcluded" in config["evaluation"]["views"]
    if config_name.startswith("pn2021c_"):
        assert "pn2021c_all_zero_kept_corrupted_refexcluded" in config["evaluation"]["views"]
        assert "pn2021c_drop_all_zero_corrupted_refexcluded" in config["evaluation"]["views"]
    assert commands
    for command in commands:
        assert command["cwd"] == paths["project_root"]
        assert command["argv"][0].startswith("/home/linbinhao/")
        assert command["argv"][1].endswith(".py")
        assert command["env"]["TMPDIR"] == paths["short_tmp_root"]
        assert len(command["env"]["TMPDIR"]) < len(paths["tmp_root"])


@pytest.mark.parametrize(
    "config_name",
    [
        "pn2021_eval_v7_sjr_rgq_refexcluded.yaml",
        "pn2021c_effnet_threechain_locked_official_s5.yaml",
        "pn2021c_effnet_official_s5_depth23_composite.yaml",
    ],
)
def test_effnet_latest_eval_refs_use_locked_k500_seed(config_name: str):
    config = _load(config_name)
    assert config["paper_protocol"]["kshot"]["seed"] == 20260601
    assert config["paper_protocol"]["kshot"]["subset_seed"] == 20260601

    argv = " ".join(str(part) for command in build_runner_commands(config) for part in command["argv"])
    assert "seed20260601" in argv
    assert "seed20260531" not in argv


def test_public_experiment_configs_are_latest_mainline_only():
    index = yaml.safe_load((REPO / "configs" / "active_scripts.yaml").read_text(encoding="utf-8"))
    latest_configs = {Path(stage["config"]).name for stage in index["latest_mainline"]["stages"]}
    actual_configs = {path.name for path in (REPO / "configs" / "experiments").glob("*.yaml")}

    assert len(latest_configs) == 10
    assert "inactive_experiment_configs" not in index
    assert actual_configs == latest_configs


def test_generic_config_framework_tests_use_active_fixtures():
    index = yaml.safe_load((REPO / "configs" / "active_scripts.yaml").read_text(encoding="utf-8"))
    public_config_names = {path.name for path in (REPO / "configs" / "experiments").glob("*.yaml")}
    latest_config_names = {Path(stage["config"]).name for stage in index["latest_mainline"]["stages"]}
    generic_tests = [
        test_logging_artifacts_are_split_by_lifecycle,
        test_pipeline_stages_are_recorded_in_dry_run_manifest,
        test_pipeline_stages_reject_unknown_dependencies,
        test_pipeline_stages_reject_duplicate_names,
        test_experiment_schema_rejects_missing_logging_block,
        test_experiment_schema_rejects_bad_postprocess_artifact_shape,
        test_local_config_cannot_override_paper_or_runner_sections,
        test_vae_configs_declare_required_epoch_metrics,
    ]
    offenders = []
    for test_func in generic_tests:
        source = inspect.getsource(test_func)
        offenders.extend(
            f"{test_func.__name__}: {config_name}"
            for config_name in sorted(public_config_names - latest_config_names)
            if config_name in source
        )

    assert public_config_names == latest_config_names
    assert offenders == []


def test_logging_artifacts_are_split_by_lifecycle():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
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
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["stages"] = [
        {
            "name": "direct_k500_train",
            "produces": ["direct_k500_checkpoint"],
            "skip_if_exists": ["${paths.output_root}/direct_k500_checkpoint.pt"],
        },
        {
            "name": "pn2021_clean_eval",
            "requires": ["direct_k500_train"],
            "produces": ["pn2021_clean_metrics"],
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
            "name": "direct_k500_train",
            "requires": [],
            "produces": ["direct_k500_checkpoint"],
            "skip_if_exists": ["${paths.output_root}/direct_k500_checkpoint.pt"],
        },
        {
            "index": 1,
            "name": "pn2021_clean_eval",
            "requires": ["direct_k500_train"],
            "produces": ["pn2021_clean_metrics"],
            "skip_if_exists": [],
        },
    ]
    assert manifest["artifact_trace"]["pipeline_stages"] == manifest["pipeline_stages"]


def test_pipeline_stages_reject_unknown_dependencies():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["stages"] = [
        {
            "name": "pn2021_clean_eval",
            "requires": ["missing_direct_k500_train"],
            "produces": ["pn2021_clean_metrics"],
        }
    ]

    with pytest.raises(ConfigError, match="unknown required stage"):
        validate_experiment_config(config, repo_root=REPO)


def test_pipeline_stages_reject_duplicate_names():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["stages"] = [
        {"name": "pn2021_clean_eval", "produces": ["ckpt"]},
        {"name": "pn2021_clean_eval", "produces": ["metrics"]},
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
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    del config["logging"]

    with pytest.raises(ConfigError, match="Schema validation failed"):
        validate_experiment_config(config, repo_root=REPO)


def test_experiment_schema_rejects_bad_postprocess_artifact_shape():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
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
            REPO / "configs" / "experiments" / "effnet_direct_k500_v7_sjr_rgq.yaml",
            local_path,
            runtime_context={"run_id": "pytest_run"},
        )


def test_runner_protocol_audit_rejects_unknown_entrypoint_even_if_it_exists():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["runner"] = {
        "entrypoint": "scripts/run_experiment.py",
        "adapter": "direct_finetune",
        "env": config["runner"]["env"],
    }
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="not in the managed runner allowlist"):
        build_runner_commands(config)


def test_vae_configs_declare_required_epoch_metrics():
    effnet = _load("effnet_vae_lhat_augmix_threechain_locked_k500.yaml")

    effnet_metrics = set(effnet["logging"]["required_epoch_metrics"])
    assert {
        "asr_overall",
        "sample_any_positive_below_0p5_asr",
        "attack_vs_anchor",
        "loss_gain",
        "latent_augmix_stats",
    } <= effnet_metrics
    assert "quick_eval" not in effnet_metrics


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


def test_command_protocol_audit_rejects_missing_vae_init_checkpoint():
    config = _load("effnet_vae_lhat_augmix_threechain_locked_k500.yaml")
    validate_experiment_config(config, repo_root=REPO)
    command = copy.deepcopy(build_runner_commands(config)[0])
    command["argv"] = _remove_option_pair(command["argv"], "--init_ckpt")

    with pytest.raises(ConfigError, match="missing required option --init_ckpt"):
        audit_runner_commands(config, [command])


def test_runner_audit_dispatches_eval_crosscenter_command():
    command = {
        "argv": [
            "python",
            "ecg_adv_gen/runner/pn2021_clean_eval.py",
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

    source = inspect.getsource(audit_runner_command)
    assert "audit_effnet_vae_lhat_command(" in source
    assert "anchor_base does not encode K" not in source

    config = _load("effnet_vae_lhat_augmix_threechain_locked_k500.yaml")
    command = build_runner_commands(config)[0]
    seed = config["paper_protocol"]["kshot"]["seed"]
    result = audit_effnet_vae_lhat_command(
        command,
        expected_k=500,
        expected_seed=seed,
        target_centers=set(config["paper_protocol"]["centers"]["target_4"]),
    )
    assert result["errors"] == []

    bad = copy.deepcopy(command)
    bad["argv"] = list(bad["argv"])
    bad["argv"][bad["argv"].index("--hull_neighbor_mode") + 1] = "heldout_oracle"
    result = audit_effnet_vae_lhat_command(
        bad,
        expected_k=500,
        expected_seed=seed,
        target_centers=set(config["paper_protocol"]["centers"]["target_4"]),
    )
    assert "invalid --hull_neighbor_mode" in "\n".join(result["errors"])


def test_runner_command_audit_does_not_keep_unmanaged_runner_branches():
    unmanaged_names = {
        "ptbxl_source_train.py",
        "train_ibe_repro.py",
        "train_dit_repro.py",
        "train_center_prompt_tokens.py",
        "generate_center_prompt_token_synth.py",
        "gate_prompt_token_synth.py",
        "synth_online_at_super5.py",
        "benchmark_direct_finetune.py",
        "ecgfounder_kshot_head_ft.py",
        "ecgfounder_vae_lhat.py",
        "kshot_subset_export.py",
    }
    source = inspect.getsource(audit_runner_commands)

    assert managed_runner_script_names().isdisjoint(unmanaged_names)
    assert sorted(name for name in unmanaged_names if name in source) == []


def test_runner_command_dispatcher_does_not_keep_unmanaged_runner_branches():
    unmanaged_names = {
        "synth_online_at_super5.py",
        "benchmark_direct_finetune.py",
    }
    source = inspect.getsource(audit_runner_command)

    assert DISPATCHED_RUNNER_AUDIT_SCRIPT_NAMES <= managed_runner_script_names()
    assert sorted(name for name in unmanaged_names if name in source) == []


def test_artifact_trace_does_not_keep_unmanaged_runner_branches():
    unmanaged_names = {
        "ptbxl_source_train.py",
        "train_ibe_repro.py",
        "train_dit_repro.py",
        "train_center_prompt_tokens.py",
        "generate_center_prompt_token_synth.py",
        "gate_prompt_token_synth.py",
        "synth_online_at_super5.py",
        "benchmark_direct_finetune.py",
        "ecgfounder_kshot_head_ft.py",
        "ecgfounder_vae_lhat.py",
        "kshot_subset_export.py",
    }
    source = inspect.getsource(build_artifact_trace)

    assert managed_runner_script_names().isdisjoint(unmanaged_names)
    assert sorted(name for name in unmanaged_names if name in source) == []


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
    assert config["adaptation"]["latent_augmix"]["chain_roles"] == [
        {"name": "chain1", "role": "official_corruption_chain"},
        {"name": "chain2", "role": "official_corruption_chain"},
        {"name": "chain3", "role": "vae_lhat_adversarial_waveform"},
    ]
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
        assert "--checkpoint_policy" not in argv
        assert "--quick_eval_source" not in argv
        assert "--target_real_val_fraction" not in argv
        assert "--target_real_val_seed" not in argv
        assert "--attack_mode" not in argv
        assert "--pgd_K" not in argv
        assert "--pgd_alpha" not in argv
        assert "--delta_init_scale" not in argv
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


def test_runner_rejects_adapter_and_argv_together():
    config = _load("effnet_vae_lhat_augmix_threechain_locked_k500.yaml")
    config["runner"]["argv"] = ["--center", "${matrix.center}"]

    with pytest.raises(ConfigError, match="runner.*adapter.*argv"):
        build_runner_commands(config)


def test_runner_rejects_yaml_argv_launch_surface_without_adapter():
    config = _load("effnet_vae_lhat_augmix_threechain_locked_k500.yaml")
    config["runner"].pop("adapter")
    config["runner"]["argv"] = []

    with pytest.raises(ConfigError, match="runner.adapter is required"):
        build_runner_commands(config)


def test_runner_adapter_registry_only_exposes_latest_mainline_adapters():
    from ecg_adv_gen.config.adapters.registry import runner_adapter_names

    assert set(runner_adapter_names()) == {
        "direct_finetune",
        "ecgfounder_fullft",
        "ecgfounder_pn2021c_eval",
        "effnet_vae_lhat",
        "pn2021_eval",
        "pn2021c_eval",
    }


def test_experiment_schema_adapter_enum_matches_runtime_registry():
    from ecg_adv_gen.config.adapters.registry import runner_adapter_names

    schema = json.loads((REPO / "configs/schemas/experiment_config.schema.json").read_text(encoding="utf-8"))
    adapter_enum = set(schema["properties"]["runner"]["properties"]["adapter"]["enum"])

    assert adapter_enum == set(runner_adapter_names())


def test_direct_finetune_adapter_does_not_keep_benchmark_entrypoint_branch():
    from ecg_adv_gen.config.adapters import direct_finetune

    source = inspect.getsource(direct_finetune)

    assert "benchmark_direct_finetune.py" not in source
    assert "build_benchmark_direct_finetune_argv" not in source


def test_pn2021_eval_adapter_does_not_keep_prompt_token_entrypoint_branch():
    from ecg_adv_gen.config.adapters import pn2021_eval

    source = inspect.getsource(pn2021_eval)

    assert "ecgtwin_prompt_token_online_at_minimal_eval" not in source
    assert "prompt-token online-AT" not in source


def test_unmanaged_adapter_files_are_removed_from_runtime_tree():
    from ecg_adv_gen.config.runner_audit import audit_runner_command

    removed_paths = {
        "ecg_adv_gen/config/adapters/benchmark_vae_lhat.py",
        "ecg_adv_gen/config/adapters/ecgfounder_vae_lhat.py",
        "ecg_adv_gen/config/adapters/prompt_token_online_at.py",
        "ecg_adv_gen/config/adapters/source_training.py",
        "util/tests/test_benchmark_vae_lhat_adapter.py",
        "util/tests/test_prompt_token_adapter.py",
    }
    active_index = yaml.safe_load((REPO / "configs" / "active_scripts.yaml").read_text(encoding="utf-8"))
    source_of_truth_paths = set(active_index["source_of_truth"].values())

    assert sorted(path for path in removed_paths if (REPO / path).exists()) == []
    assert sorted(removed_paths & source_of_truth_paths) == []
    assert "benchmark_vae_lhat" not in inspect.getsource(audit_runner_command)


def test_high_risk_long_argv_adapters_are_registered():
    from ecg_adv_gen.config.adapters.registry import runner_adapter_names

    names = set(runner_adapter_names())
    assert {
        "pn2021_eval",
        "pn2021c_eval",
        "direct_finetune",
    } <= names


@pytest.mark.parametrize(
    ("config_name", "adapter_name", "expected_commands"),
    [
        ("pn2021_eval_v7_sjr_rgq_refexcluded.yaml", "pn2021_eval", 4),
        ("pn2021c_effnet_threechain_locked_official_s5.yaml", "pn2021c_eval", 4),
        ("effnet_direct_k500_v7_sjr_rgq.yaml", "direct_finetune", 1),
    ],
)
def test_high_risk_long_argv_configs_use_typed_adapters(config_name: str, adapter_name: str, expected_commands: int):
    config = _load(config_name)

    assert config["runner"]["adapter"] == adapter_name
    assert "argv" not in config["runner"]
    commands = build_runner_commands(config)
    assert len(commands) == expected_commands


def test_effnet_direct_manifest_child_dir_uses_shared_run_naming_helper():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
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
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    validate_experiment_config(config, repo_root=REPO)
    command = copy.deepcopy(build_runner_commands(config)[0])
    idx = command["argv"].index("--out_root")
    command["argv"][idx + 1] = f"{config['paths']['output_root']}/effnet_direct_k500_v7_sjr_rgq"

    with pytest.raises(ConfigError, match="must include runtime.run_id"):
        audit_runner_commands(config, [command])


@pytest.mark.parametrize(
    ("option", "script_path"),
    [
        ("--input_dirs", "ecg_adv_gen/runner/merge_gated_prompt_token_pools.py"),
        ("--gated_dirs", "ecg_adv_gen/runner/select_quality_prompt_token_pool.py"),
    ],
)
def test_runner_audit_rejects_historical_prompt_pool_tools(option: str, script_path: str):
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    outside_path = "/tmp/outside_gated_pool"
    command = {
        "name": "pytest_multi_input_dir_audit",
        "matrix": {},
        "cwd": str(REPO),
        "env": {},
        "argv": [
            "/home/linbinhao/micromamba/envs/ECGTwin/bin/python",
            str(REPO / script_path),
            option,
            str(REPO / "safe_gated_pool"),
            outside_path,
            "--out_dir",
            str(REPO / "runs" / "pytest_run" / "merged_gated_pool"),
        ],
    }

    with pytest.raises(ConfigError, match="not in the managed runner allowlist"):
        audit_runner_commands(config, [command])


def test_runner_audit_records_managed_entrypoint_profiles():
    config = _load("effnet_vae_lhat_augmix_threechain_locked_k500.yaml")
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=validate_experiment_config(config, repo_root=REPO),
        run_id=config["runtime"]["run_id"],
        cli_args=argparse.Namespace(
            config="configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml",
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
        "script_name": "effnet_vae_lhat_augmix.py",
        "relative_path": "ecg_adv_gen/runner/effnet_vae_lhat_augmix.py",
        "runner_root": "ecg_adv_gen/runner",
        "family": "effnet_vae_lhat",
    } in command_profiles


def test_postprocess_audit_rejects_unscoped_output_dir():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    idx = config["postprocess"]["commands"][0]["argv"].index("--output-dir")
    config["postprocess"]["commands"][0]["argv"][idx + 1] = f"{config['paths']['output_root']}/metrics_export"
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="must include runtime.run_id"):
        build_postprocess_commands(config)


def test_cli_set_overrides_are_whitelisted_and_applied_before_interpolation():
    config = load_experiment_config(
        REPO / "configs" / "experiments" / "effnet_direct_k500_v7_sjr_rgq.yaml",
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
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    with pytest.raises(ConfigError, match="not allowed"):
        apply_cli_overrides(config, ["paper_protocol.kshot.k=100"])
    with pytest.raises(ConfigError, match="expects int"):
        apply_cli_overrides(config, ["training.epochs=two"])
    with pytest.raises(ConfigError, match="key=value"):
        apply_cli_overrides(config, ["training.epochs"])


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
        assert argv[1].endswith("ecg_adv_gen/runner/ecgfounder_fullft.py")
        assert _option_value(argv, "--center") == center
        assert _option_value(argv, "--ref_meta_json").endswith(
            f"/paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets/{center}/"
            f"k500_seed20260531/{center}_real_k500_seed20260531.ref_meta.json"
        )
        assert _option_value(argv, "--k") == "500"
        assert _option_value(argv, "--seed") == "20260531"
        assert "--target_val_count" not in argv
        assert "--target_val_seed" not in argv
        assert "--selection_metric" not in argv
        assert "--checkpoint_policy" not in argv
        assert "--source_train_limit" not in argv
        assert "--cache_dir" not in argv
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
    assert config["adaptation"]["latent_augmix"]["chain_roles"] == [
        {"name": "chain1", "role": "official_corruption_chain"},
        {"name": "chain2", "role": "official_corruption_chain"},
        {"name": "chain3", "role": "vae_lhat_adversarial_waveform"},
    ]
    assert "partner_pool" not in config["adaptation"]["hull"]
    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        center = command["matrix"]["center"]
        joined = " ".join(argv)
        assert argv[1].endswith("ecg_adv_gen/runner/ecgfounder_fullft.py")
        assert _option_value(argv, "--center") == center
        assert "--checkpoint_policy" not in argv
        assert "--target_val_count" not in argv
        assert "--target_val_seed" not in argv
        assert "--selection_metric" not in argv
        assert "--source_train_limit" not in argv
        assert "--cache_dir" not in argv
        assert "--hull_partner_pool" not in argv
        assert "--source_partner_limit_per_class" not in argv
        assert "--ptbxl_vae_cache" not in argv
        assert "target_source" not in joined
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


def test_ecgfounder_locked_threechain_augmix_can_reuse_locked_upstream_run_id():
    config = _load("ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml")
    config = copy.deepcopy(config)
    config["runner"]["matrix"]["center"] = ["cpsc_2018"]
    config["model"]["upstream_run_id"] = "locked_rawfirst_20260618"
    config["model"]["init_model_path"] = (
        "${paths.output_root}/ecgfounder_k500_fullft_locked/"
        "locked_rawfirst_20260618/runs/${matrix.center}_k500_fullft_locked/last_model.pt"
    )
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 1
    argv = commands[0]["argv"]
    assert _option_value(argv, "--out_dir").endswith(
        "/runs/ecgfounder_vae_lhat_augmix_threechain_locked_k500/pytest_run"
    )
    assert _option_value(argv, "--init_model_path").endswith(
        "/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs/"
        "cpsc_2018_k500_fullft_locked/last_model.pt"
    )
    audit = audit_runner_command(commands[0], config=config)
    assert audit["errors"] == []


def test_ecgfounder_fullft_adapter_passes_hull_init_logit_gap():
    config = _load("ecgfounder_vae_lhat_augmix_threechain_locked_k500.yaml")
    config = copy.deepcopy(config)
    config["runner"]["matrix"]["center"] = ["cpsc_2018"]
    config["adaptation"]["hull"]["init_logit_gap"] = 0.0

    commands = build_runner_commands(config)

    argv = commands[0]["argv"]
    assert _option_value(argv, "--hull_init_logit_gap") == "0.0"


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
    assert argv[1].endswith("ecg_adv_gen/runner/ecgfounder_fullft.py")
    assert _option_value(argv, "--stage") == "ptbxl_source"
    assert "--checkpoint_policy" not in argv
    assert "--selection_metric" not in argv
    assert "--source_train_limit" not in argv
    assert "--cache_dir" not in argv
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


def test_pn2021c_effnet_threechain_locked_protocol_overrides_selection_and_raw_order():
    config = _load("pn2021c_effnet_threechain_locked_official_s5.yaml")
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_pn2021c_official_s5_locked",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert len(commands) == 4
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
            f"{center}_realall_targetheavy_M20_lam0p05_augmix_s5_wlat0p25_hs3_cdhypminormsttc_hlabelcom_anchor_soft_sta_local_random_p120_fullft_k500_threechain_s5_locked_ep30_seed20260601"
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
        assert argv[1].endswith("ecg_adv_gen/runner/ecgfounder_pn2021c_eval.py")
        assert run_dir == (
            f"/home/linbinhao/ECG_adv_data/runs/"
            f"ecgfounder_vae_lhat_augmix_threechain_locked_k500/"
            f"{config['runtime']['run_id']}/runs/{center}_vae_lhat_augmix_threechain_locked"
        )
        assert _option_value(argv, "--corruption_input") == "bottleneck5000"
        assert _option_value(argv, "--crop_len") == "1000"
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


def test_pn2021c_ecgfounder_depth23_composite_config_uses_official_locked_surface():
    config = _load("pn2021c_ecgfounder_official_s5_depth23_composite.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        assert _option_value(argv, "--corruption_input") == "bottleneck5000"
        assert _option_value(argv, "--crop_len") == "1000"
        assert _option_value(argv, "--severity_profile") == "standard"
        assert _option_value(argv, "--severities") == "5"
        assert _all_option_values(argv, "--corruptions") == official_s5_depth23_composites()
        assert _option_value(argv, "--output_path").endswith(
            "/ecgfounder_threechain_locked/"
            "eval_pn2021_c_ecgfounder_v7_refexcluded_stream_standard_official_s5_depth23_composite.json"
        )


def test_pn2021c_effnet_depth23_composite_config_uses_official_locked_surface():
    config = _load("pn2021c_effnet_official_s5_depth23_composite.yaml")
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 4
    for command in commands:
        argv = command["argv"]
        assert _option_value(argv, "--corruption_input") == "raw_first"
        assert _option_value(argv, "--crop_len") == "1000"
        assert _option_value(argv, "--checkpoint_name") == "last_model.pt"
        assert _option_value(argv, "--severity_profile") == "standard"
        assert _option_value(argv, "--severities") == "5"
        assert _all_option_values(argv, "--corruptions") == official_s5_depth23_composites()
        assert _option_value(argv, "--output_path").endswith(
            "/effnet_threechain_locked/"
            "eval_pn2021_c_v7_refexcluded_stream_standard_official_s5_depth23_composite.json"
        )


def test_pn2021_eval_command_audit_rejects_missing_ref_exclusion():
    config = _load("pn2021_eval_v7_sjr_rgq_refexcluded.yaml")
    validate_experiment_config(config, repo_root=REPO)
    command = copy.deepcopy(build_runner_commands(config)[0])
    start = command["argv"].index("--exclude_ref_ids")
    end = command["argv"].index("--output_path")
    del command["argv"][start:end]

    with pytest.raises(ConfigError, match="missing required option --exclude_ref_ids"):
        audit_runner_commands(config, [command])


def test_mapping_hash_mismatch_is_rejected():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["paper_protocol"]["mapping_hash"] = "bad_hash"

    with pytest.raises(ConfigError, match="Mapping hash mismatch"):
        validate_experiment_config(config, repo_root=REPO)


def test_missing_ref_exclusion_is_rejected():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["paper_protocol"]["kshot"]["exclude_refs_from_eval"] = False

    with pytest.raises(ConfigError, match="exclude_refs_from_eval"):
        validate_experiment_config(config, repo_root=REPO)


def test_selection_allowed_data_drift_is_rejected():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["paper_protocol"]["selection"]["allowed_data"] = ["target_k500_internal_val"]

    with pytest.raises(ConfigError, match="selection.allowed_data"):
        validate_experiment_config(config, repo_root=REPO)


def test_tracked_config_rejects_local_absolute_paths(tmp_path: Path):
    bad_config = tmp_path / "bad_experiment.yaml"
    bad_config.write_text(
        yaml.safe_dump({"experiment": {"name": "bad"}, "bad_path": "/home/linbinhao/private"}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="local/root absolute paths"):
        load_experiment_config(bad_config, LOCAL_EXAMPLE)


def test_local_config_rejects_known_root_aliases(tmp_path: Path):
    local = yaml.safe_load(LOCAL_EXAMPLE.read_text(encoding="utf-8"))
    local["paths"]["data_root"] = "/root/autodl-tmp"
    local["paths"]["model_root"] = "/root/autodl-tmp/models"
    local["paths"]["output_root"] = "/root/autodl-tmp/runs"
    local["paths"]["cache_root"] = "/root/autodl-tmp/cache"
    local["paths"]["tmp_root"] = "/root/autodl-tmp/tmp"
    local["python"]["executable"] = "/root/miniforge3/envs/ECGTwin/bin/python"
    local_path = tmp_path / "local_root_alias.yaml"
    local_path.write_text(yaml.safe_dump(local), encoding="utf-8")

    with pytest.raises(PathSafetyError, match="root-era path"):
        config = load_experiment_config(
            REPO / "configs" / "experiments" / "effnet_direct_k500_v7_sjr_rgq.yaml",
            local_path,
            runtime_context={"run_id": "pytest_root_alias"},
        )
        validate_local_paths(config)


def test_local_config_rejects_unknown_root_path(tmp_path: Path):
    local = yaml.safe_load(LOCAL_EXAMPLE.read_text(encoding="utf-8"))
    local["paths"]["data_root"] = "/root/unknown-dataset-root"
    local_path = tmp_path / "local_bad_root.yaml"
    local_path.write_text(yaml.safe_dump(local), encoding="utf-8")

    config = load_experiment_config(
        REPO / "configs" / "experiments" / "effnet_direct_k500_v7_sjr_rgq.yaml",
        local_path,
        runtime_context={"run_id": "pytest_bad_root"},
    )
    with pytest.raises(PathSafetyError, match="root-era path"):
        validate_local_paths(config)


def test_dry_run_manifest_never_marks_child_scripts_invoked():
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
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
    assert manifest["safety"]["managed_child_commands_invoked"] is False
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
    manifest = _dry_manifest("effnet_direct_k500_v7_sjr_rgq.yaml")
    trace = manifest["artifact_trace"]

    assert trace["metrics"]["mapping_version"] == "v7_super5_sjr_rgq_review_20260528"
    assert trace["metrics"]["mapping_hash"] == "555ec85d5b51"
    assert trace["metrics"]["class_order"] == ["CD", "HYP", "MI", "NORM", "STTC"]
    assert "pn2021_all_zero_kept_refexcluded" in trace["metrics"]["views"]
    assert "pn2021_drop_all_zero_refexcluded" in trace["metrics"]["views"]


def test_manifest_records_direct_k500_checkpoint_refs_and_outputs():
    manifest = _dry_manifest("effnet_direct_k500_v7_sjr_rgq.yaml")
    trace = manifest["artifact_trace"]

    checkpoints = trace["inputs"]["checkpoints"]
    assert any(item["role"] == "model.init_checkpoint" for item in checkpoints)
    refs = trace["inputs"]["k500_refs"]
    assert len(refs) == 4
    assert {item["center"] for item in refs} == {"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"}
    for item in refs:
        assert item["k"] == 500
        assert item["seed"] == 20260601
        assert item["ref_meta_json"]["path"].endswith("_real_k500_seed20260601.ref_meta.json")
        assert item["signals_npz"]["path"].endswith("_real_k500_seed20260601.signals.npz")
        assert item["latent_npz"] is None
    child_runs = trace["expected_outputs"]["child_runs"]
    assert len(child_runs) == 4
    assert all(any(art["role"] == "eval_result" for art in run["expected_artifacts"]) for run in child_runs)
    assert any("eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json" in art["path"]
               for run in child_runs for art in run["expected_artifacts"])


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


def test_run_managed_commands_creates_short_runtime_env_dirs(tmp_path: Path):
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

    run_managed_commands([command], run_dir=tmp_path / "run", manifest_path=manifest_path)

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


def test_run_managed_commands_records_failed_child_run(tmp_path: Path):
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
        run_managed_commands([command], run_dir=tmp_path / "run", manifest_path=manifest_path, base_env={})

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

    run_managed_commands([command], run_dir=tmp_path / "run", manifest_path=manifest_path, base_env={})
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


def test_run_managed_commands_fails_if_required_artifact_missing(tmp_path: Path):
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
        run_managed_commands(commands, run_dir=tmp_path, manifest_path=manifest_path, base_env={})

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
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
    config = copy.deepcopy(config)
    config["runner"].setdefault("env", {})["CUDA_VISIBLE_DEVICES"] = "0"
    validate_experiment_config(config, repo_root=REPO)

    with pytest.raises(ConfigError, match="CUDA_VISIBLE_DEVICES"):
        build_runner_commands(config)


def test_runtime_path_audit_keeps_home_entrypoint_for_symlinked_inputs(tmp_path: Path):
    boundary = tmp_path / "home"
    external = tmp_path / "external_data"
    boundary.mkdir()
    external.mkdir()
    input_link = boundary / "kshot_subsets"
    input_link.symlink_to(external, target_is_directory=True)

    errors: list[str] = []
    _audit_runtime_path(
        errors,
        script="pn2021_clean_eval.py",
        label="--exclude_ref_ids",
        value=str(input_link / "ref_meta.json"),
        boundary=boundary,
    )

    assert errors == []


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
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
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
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
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
    config = _load("effnet_direct_k500_v7_sjr_rgq.yaml")
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
            str(REPO / "configs" / "experiments" / "effnet_direct_k500_v7_sjr_rgq.yaml"),
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


def test_latest_mainline_configs_dry_run_through_run_experiment_cli():
    index = yaml.safe_load((REPO / "configs" / "active_scripts.yaml").read_text(encoding="utf-8"))
    stages = index["latest_mainline"]["stages"]

    for stage in stages:
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "run_experiment.py"),
                "--config",
                str(REPO / stage["config"]),
                "--local-config",
                str(LOCAL_EXAMPLE),
                "--run-id",
                f"pytest_latest_cli_{stage['name']}",
                "--dry-run",
            ],
            cwd=str(REPO),
            text=True,
            capture_output=True,
            check=False,
        )

        assert proc.returncode == 0, f"{stage['name']}: {proc.stderr}"
        manifest_text = proc.stdout.split("\n\n# Managed commands", 1)[0]
        manifest = json.loads(manifest_text)
        assert manifest["status"] == "dry_run"
        assert manifest["launcher"]["script"] == "scripts/run_experiment.py"
        assert manifest["safety"]["managed_child_commands_invoked"] is False
        assert manifest["experiment"]["name"]
        assert manifest["commands"]


def test_latest_mainline_dry_run_manifests_do_not_expose_legacy_artifact_roles():
    index = yaml.safe_load((REPO / "configs" / "active_scripts.yaml").read_text(encoding="utf-8"))

    for stage in index["latest_mainline"]["stages"]:
        config = load_experiment_config(
            REPO / stage["config"],
            LOCAL_EXAMPLE,
            runtime_context={"run_id": f"pytest_latest_roles_{stage['name']}"},
        )
        paths = validate_experiment_config(config, repo_root=REPO)
        commands = build_runner_commands(config)
        manifest = make_dry_run_manifest(
            config,
            commands=commands,
            local_paths=paths,
            run_id=f"pytest_latest_roles_{stage['name']}",
            cli_args=argparse.Namespace(dry_run=True, write_plan=False),
        )

        roles = [
            artifact["role"]
            for child in manifest["artifact_trace"]["expected_outputs"]["child_runs"]
            for artifact in child.get("expected_artifacts", [])
        ]
        assert not [role for role in roles if str(role).startswith("legacy_")]


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
    assert "ecg_adv_gen/runner/effnet_direct_finetune.py" in command_text
    k500_ref_ids = out_dir / "k500_ref_ids.json"
    assert k500_ref_ids.exists()
    selection_json = out_dir / "selection.json"
    assert selection_json.exists()
    data_manifest = out_dir / "data_manifest.json"
    assert data_manifest.exists()
    env_json = out_dir / "env.json"
    assert env_json.exists()
    env_payload = json.loads(env_json.read_text(encoding="utf-8"))
    assert env_payload["python_executable"]
    assert env_payload["python_version"]
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
