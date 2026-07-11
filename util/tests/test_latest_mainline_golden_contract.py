"""CPU-only tests for the versioned latest-mainline golden contract."""

from __future__ import annotations

import json
import copy
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "configs" / "active_scripts.yaml"
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"
GOLDEN = REPO / "configs" / "golden" / "latest_mainline_contract_v1.json"
GENERATOR = REPO / "scripts" / "agent" / "build_latest_mainline_golden.py"

def test_latest_mainline_golden_identity_order_and_content():
    assert GOLDEN.exists(), f"missing golden contract: {GOLDEN.relative_to(REPO)}"

    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    contract = json.loads(GOLDEN.read_text(encoding="utf-8"))
    latest = index["latest_mainline"]

    assert contract["schema_version"] == 1
    assert contract["run_id"] == "GOLDEN_LATEST_MAINLINE_V1"
    assert contract["latest_mainline"] == {
        "class_order": latest["class_order"],
        "launcher": latest["launcher"],
        "mapping_hash": latest["mapping_hash"],
        "mapping_version": latest["mapping_version"],
        "method": latest["method"],
        "claim_scope": latest["claim_scope"],
        "comparison_surface": latest["comparison_surface"],
        "stage_count": len(latest["stages"]),
    }
    assert [
        (stage["name"], stage["role"], stage["config"])
        for stage in contract["stages"]
    ] == [
        (stage["name"], stage["role"], stage["config"])
        for stage in latest["stages"]
    ]

    for stage in contract["stages"]:
        assert stage["resolved_experiment"]
        assert stage["method_family"]
        assert stage["runner_adapter"]
        assert stage["runner_entrypoint"].startswith("ecg_adv_gen/runner/")
        assert stage["evaluation_views"]
        assert stage["paper_protocol"]["kshot"]["mode"] == "fixed_k"
        assert stage["paper_protocol"]["kshot"]["exclude_refs_from_eval"] is True
        assert stage["paper_protocol"]["selection"]["forbid_heldout_target_labels"] is True
        assert stage["runner_commands"]
        assert stage["checkpoint_artifacts"]
        if stage["name"] == "ecgfounder_ptbxl_locked_fullft":
            assert stage["k500_refs"] == []
        else:
            assert stage["k500_refs"]
        assert len(stage["matrix_cases"]) == len(stage["runner_commands"])
        assert stage["expected_artifact_roles"]
        for record in stage["checkpoint_artifacts"]:
            assert isinstance(record, str) and record.count("|") >= 4
        for record in stage["k500_refs"]:
            assert isinstance(record, str) and record.count("|") >= 7
        for command in stage["runner_commands"]:
            assert "\t" in command
        for command in stage["postprocess_commands"]:
            assert "\t" in command
        for command in [*stage["runner_commands"], *stage["postprocess_commands"]]:
            argv = shlex.split(command.split("\t", 1)[1])
            assert argv[0] == "<PYTHON_EXECUTABLE>"
            assert argv[1].startswith("<PROJECT_ROOT>/")

    rendered = GOLDEN.read_text(encoding="utf-8")
    assert rendered.endswith("\n")
    assert len(rendered.splitlines()) <= 2000
    assert len(GENERATOR.read_text(encoding="utf-8").splitlines()) <= 200
    assert "/home/" not in rendered
    assert "/root/" not in rendered
    for option in (
        "--k",
        "--seed",
        "--latent_augmix_copies",
        "--latent_augmix_width",
        "--latent_augmix_depth",
        "--checkpoint",
        "--exclude_ref_ids",
    ):
        assert option in rendered

    matched = contract["matched_effnet"]
    assert list(matched["arm_claims"]) == ["a0", "a2", "a3", "a4", "a5"]
    assert len(matched["producer_dirs"]) == 20
    assert contract["latest_mainline"]["comparison_surface"]["source_reference"]["checkpoint_sha256"] == (
        "f7a4b05d85da8352013b67a0d56d8d47378ab2abc196d1f3a6ba8bbf15aa40ea"
    )
    by_config = {stage["config"]: stage for stage in contract["stages"]}
    train = by_config["configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml"]
    expected_cases = [
        {"center": center, "arm": arm}
        for center in ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
        for arm in ("a0", "a2", "a3", "a4", "a5")
    ]
    assert train["matrix_cases"] == expected_cases
    for config_rel in (
        "configs/experiments/pn2021_eval_v7_sjr_rgq_refexcluded.yaml",
        "configs/experiments/pn2021c_effnet_threechain_locked_official_s5.yaml",
        "configs/experiments/pn2021c_effnet_official_s5_depth23_composite.yaml",
    ):
        stage = by_config[config_rel]
        assert stage["matrix_cases"] == expected_cases
        for case, command in zip(stage["matrix_cases"], stage["runner_commands"]):
            argv = shlex.split(command.split("\t", 1)[1])
            key = f"{case['center']}/{case['arm']}"
            assert argv[argv.index("--model_dir") + 1] == matched["producer_dirs"][key]
            assert argv[argv.index("--checkpoint_name") + 1] == "best_model.pt"
            assert "--exclude_ref_ids" in argv


def test_latest_mainline_golden_checker_accepts_tracked_snapshot():
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_normalizer_drops_only_explicit_machine_state_and_keeps_unknown_fields():
    from scripts.agent.build_latest_mainline_golden import normalize_value

    value = {
        "created_at_utc": "volatile",
        "exists": True,
        "git": {"commit": "volatile", "status_short": "volatile"},
        "mtime_ns": 123,
        "size_bytes": 456,
        "unknown": {
            "custom": 7,
            "path": "/home/example/data/artifact.json",
            "sha256": "abc123",
        },
    }

    assert normalize_value(
        value,
        replacements=[
            ("/home/example", "<HOME_ROOT>"),
            ("/home/example/data", "<DATA_ROOT>"),
        ],
    ) == {
        "unknown": {
            "custom": 7,
            "path": "<DATA_ROOT>/artifact.json",
            "sha256": "abc123",
        }
    }


def test_latest_mainline_golden_is_host_neutral_for_configured_data_roots(tmp_path: Path):
    from scripts.agent.build_latest_mainline_golden import build_contract

    alternate = yaml.safe_load(LOCAL_EXAMPLE.read_text(encoding="utf-8"))
    alternate["host"]["name"] = "alternate_host"
    alternate["paths"]["data_root"] = "/home/linbinhao/golden_alt/data"
    alternate["paths"]["model_root"] = "${paths.data_root}/models"
    alternate["paths"]["output_root"] = "${paths.data_root}/runs"
    alternate["paths"]["cache_root"] = "${paths.data_root}/cache"
    alternate["paths"]["tmp_root"] = "${paths.data_root}/tmp"
    alternate["paths"]["short_tmp_root"] = "/home/linbinhao/golden_alt/short_tmp"
    alternate_path = tmp_path / "alternate_local.yaml"
    alternate_path.write_text(yaml.safe_dump(alternate, sort_keys=False), encoding="utf-8")

    assert build_contract(index_path=INDEX, local_config_path=alternate_path) == build_contract(
        index_path=INDEX,
        local_config_path=LOCAL_EXAMPLE,
    )


def test_latest_mainline_builder_rejects_duplicate_and_missing_stage_config(tmp_path: Path):
    from scripts.agent.build_latest_mainline_golden import build_contract

    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    stages = index["latest_mainline"]["stages"]
    stages[-1]["config"] = stages[0]["config"]
    mutated_index = tmp_path / "active_scripts_duplicate_missing.yaml"
    mutated_index.write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unique.*managed"):
        build_contract(index_path=mutated_index, local_config_path=LOCAL_EXAMPLE)


def test_latest_mainline_builder_rejects_cross_stage_producer_consumer_drift(monkeypatch):
    import scripts.agent.build_latest_mainline_golden as builder

    real_load = builder.load_experiment_config

    def drifting_load(config_path, *args, **kwargs):
        config = real_load(config_path, *args, **kwargs)
        if Path(config_path).name == "pn2021c_effnet_threechain_locked_official_s5.yaml":
            config = copy.deepcopy(config)
            config["adaptation"]["run_tag_extra"] = "intentional_drift"
        return config

    monkeypatch.setattr(builder, "load_experiment_config", drifting_load)
    with pytest.raises(ValueError, match="producer/consumer path mismatch"):
        builder.build_contract(index_path=INDEX, local_config_path=LOCAL_EXAMPLE)
