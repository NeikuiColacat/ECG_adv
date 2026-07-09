"""CPU-only tests for the versioned latest-mainline golden contract."""

from __future__ import annotations

import json
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
            assert {"path", "required", "role"} <= set(record)
        for record in stage["k500_refs"]:
            assert {"center", "k", "path", "required", "role", "seed"} <= set(record)
        for command in stage["runner_commands"]:
            assert set(command) == {"argv", "name"}
        for command in stage["postprocess_commands"]:
            assert set(command) == {"argv", "name"}
        for command in [*stage["runner_commands"], *stage["postprocess_commands"]]:
            argv = shlex.split(command["argv"])
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
