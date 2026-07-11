"""CPU-only contract tests for the non-canonical matched EffNet replications."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from ecg_adv_gen.config import (
    ConfigError,
    build_runner_commands,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
)


REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "configs" / "active_scripts.yaml"
LOCAL = REPO / "configs" / "local" / "linbinhao_server.example.yaml"
GOLDEN = REPO / "configs" / "golden" / "latest_mainline_contract_v1.json"
GENERATOR = REPO / "scripts" / "agent" / "build_latest_mainline_golden.py"
REPLICATION_ROOT = REPO / "configs" / "replications"
SEEDS = (20260531, 20260601, 20260611)
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
ARMS = ("a0", "a2", "a3", "a4", "a5")
STAGES = ("train", "clean", "s5", "depth23")
CANONICAL_CONFIGS = {
    "train": "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml",
    "clean": "configs/experiments/pn2021_eval_v7_sjr_rgq_refexcluded.yaml",
    "s5": "configs/experiments/pn2021c_effnet_threechain_locked_official_s5.yaml",
    "depth23": "configs/experiments/pn2021c_effnet_official_s5_depth23_composite.yaml",
}
SOURCE_SHA256 = "f7a4b05d85da8352013b67a0d56d8d47378ab2abc196d1f3a6ba8bbf15aa40ea"


def _index() -> dict[str, Any]:
    return yaml.safe_load(INDEX.read_text(encoding="utf-8"))


def _surface() -> dict[str, Any]:
    surfaces = _index()["replication_surfaces"]
    return next(item for item in surfaces if item["name"] == "effnet_matched_three_seed")


def _records_by_seed() -> dict[int, dict[str, Any]]:
    return {int(item["seed"]): item for item in _surface()["replicates"]}


def _load(path: str, *, seed: int, run_id: str | None = None) -> dict[str, Any]:
    config = load_experiment_config(
        REPO / path,
        LOCAL,
        runtime_context={"run_id": run_id or f"pytest_effnet_matched_seed{seed}"},
    )
    validate_experiment_config(config, repo_root=REPO)
    return config


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _key(command: dict[str, Any]) -> tuple[str, str]:
    matrix = command["matrix"]
    case = matrix.get("case") or {}
    return str(matrix["center"]), str(case.get("arm", matrix.get("arm")))


def _assert_exact_grid(commands: list[dict[str, Any]]) -> None:
    expected = Counter((center, arm) for center in CENTERS for arm in ARMS)
    assert len(commands) == 20
    assert Counter(_key(command) for command in commands) == expected
    assert all(_key(command)[1] != "a1" for command in commands)


def _manifest(config: dict[str, Any], commands: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    return make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=validate_experiment_config(config, repo_root=REPO),
        run_id=run_id,
        cli_args=Namespace(dry_run=True, write_plan=False),
    )


def test_replication_index_and_wrapper_inventory_are_exact_and_tracked():
    surface = _surface()
    records = _records_by_seed()
    assert surface["status"] == "launch_only_evidence_pending"
    assert surface["seeds"] == list(SEEDS)
    assert surface["canonical_seed"] == 20260601
    assert surface["command_count_per_stage"] == 20
    assert surface["pipeline_contract"] == {
        "stage_order": list(STAGES),
        "producer_stage": "train",
        "consumer_stages": ["clean", "s5", "depth23"],
        "matrix": {"centers": list(CENTERS), "arms": list(ARMS)},
    }
    assert surface["same_run_id_across_stages"] is True
    assert surface["stage_specific_plan_output_dirs"] is True
    assert surface["cli_seed_override_allowed"] is False
    assert surface["corruption_seed"] == 20260501
    assert surface["materialization_artifact_suffixes"] == [
        "ref_meta.json",
        "signals.npz",
        "latent.npz",
        "raw1000.npz",
        "class_trust.json",
    ]
    assert surface["runtime_consumed_artifact_suffixes"] == [
        "ref_meta.json",
        "latent.npz",
        "raw1000.npz",
    ]
    assert surface["provenance_companion_suffixes"] == ["signals.npz", "class_trust.json"]
    assert surface["f005_pre_split_exact_partner_eligibility"]["scope"] == (
        "full_K500_pre_split_descriptive_only"
    )
    assert surface["f005_pre_split_exact_partner_eligibility"]["post_split_promotion_evidence"] is False
    assert surface["execute_preflight"] == "require_all_four_centers_x_five_k500_artifacts"
    assert set(records) == set(SEEDS)
    assert all(tuple(record["stages"]) == STAGES for record in records.values())
    assert all(record["canonical"] is False for record in records.values())
    assert records[20260601]["canonical_reference_seed"] is True
    assert records[20260601]["selection_ids_expected_equal_to_latest_canonical"] is True
    assert records[20260601]["latent_realization_expected_bitwise_equal_to_latest_canonical"] is False
    assert all(record["k500_input_status"] == "present_validated_2026-07-11" for record in records.values())
    assert all(record["materialization_required_before_execute"] is False for record in records.values())
    assert records[20260611]["copy_or_rename_from_other_seed_allowed"] is False

    indexed_wrappers = {
        path
        for record in records.values()
        for path in record["stages"].values()
    }
    actual_wrappers = {
        path.relative_to(REPO).as_posix()
        for path in REPLICATION_ROOT.glob("effnet_matched_*.yaml")
    }
    assert actual_wrappers == indexed_wrappers
    assert len(actual_wrappers) == 12
    for path in sorted(actual_wrappers):
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, f"untracked replication config: {path}"


@pytest.mark.parametrize("seed", SEEDS)
def test_wrappers_only_override_seed_fields(seed: int):
    record = _records_by_seed()[seed]
    for stage, path in record["stages"].items():
        raw = yaml.safe_load((REPO / path).read_text(encoding="utf-8"))
        expected_keys = {"extends", "paper_protocol", "data"}
        assert raw["data"] == {
            "kshot_subset_root": (
                "${paths.data_root}/"
                "paper_matched_effnet_k500_v7_fixedk_three_seed_20260711/subsets"
            )
        }
        if stage != "train":
            expected_keys.add("experiment")
            assert raw["experiment"]["name"].endswith(f"_seed{seed}")
        if stage in {"s5", "depth23"}:
            expected_keys.add("model")
            assert raw["model"] == {"eval_seed": seed}
        assert set(raw) == expected_keys
        assert raw["paper_protocol"] == {
            "kshot": {"seed": seed, "subset_seed": seed}
        }
        text = (REPO / path).read_text(encoding="utf-8")
        assert not any(token in text for token in ("/home/", "/root/", "CUDA_VISIBLE_DEVICES", "runner:", "--set"))


def test_each_seed_resolves_to_exact_twenty_matched_commands_per_stage():
    records = _records_by_seed()
    for seed in SEEDS:
        for path in records[seed]["stages"].values():
            config = _load(path, seed=seed)
            assert config["paper_protocol"]["kshot"]["seed"] == seed
            assert config["paper_protocol"]["kshot"]["subset_seed"] == seed
            _assert_exact_grid(build_runner_commands(config))


def test_three_seed_training_contract_is_matched_except_for_seed_identity():
    records = _records_by_seed()
    invariant_snapshots = []
    for seed in SEEDS:
        run_id = f"pytest_effnet_matched_seed{seed}"
        config = _load(records[seed]["stages"]["train"], seed=seed, run_id=run_id)
        commands = build_runner_commands(config)
        _assert_exact_grid(commands)
        manifest = _manifest(config, commands, run_id)
        for command in commands:
            assert _option(command["argv"], "--seed") == str(seed)
            assert _option(command["argv"], "--target_real_val_seed") == str(seed)
            assert f"seed{seed}" in _option(command["argv"], "--anchor_base")
        k500_refs = manifest["artifact_trace"]["inputs"]["k500_refs"]
        assert {int(item["seed"]) for item in k500_refs} == {seed}
        assert all(f"seed{seed}" in item["anchor_base"] for item in k500_refs)
        children = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
        assert len({item["child_run_dir"] for item in children}) == 20
        invariant_snapshots.append(
            (
                config["model"]["init_lineage"]["checkpoint_sha256"],
                config["training"]["epochs"],
                config["paper_protocol"]["selection"]["policy"],
                config["paper_protocol"]["selection"]["source_floor"],
                config["adaptation"]["hull"]["M"],
                config["adaptation"]["hull"]["lambda"],
            )
        )
    assert all(snapshot == invariant_snapshots[0] for snapshot in invariant_snapshots)
    assert invariant_snapshots[0][0] == SOURCE_SHA256


def test_consumers_match_each_seed_producer_and_keep_corruption_seed_fixed():
    records = _records_by_seed()
    for seed in SEEDS:
        run_id = f"pytest_effnet_matched_seed{seed}"
        train_config = _load(records[seed]["stages"]["train"], seed=seed, run_id=run_id)
        train_commands = build_runner_commands(train_config)
        train_manifest = _manifest(train_config, train_commands, run_id)
        producers = {
            (child["center"], child["protocol_claim"]["arm"]): child["child_run_dir"]
            for child in train_manifest["artifact_trace"]["expected_outputs"]["child_runs"]
        }
        assert len(producers) == 20

        for stage in ("clean", "s5", "depth23"):
            config = _load(records[seed]["stages"][stage], seed=seed, run_id=run_id)
            commands = build_runner_commands(config)
            _assert_exact_grid(commands)
            if stage in {"s5", "depth23"}:
                assert config["model"]["eval_seed"] == seed
                assert config["evaluation"]["corruption_seed"] == 20260501
            for command in commands:
                key = _key(command)
                model_dir = _option(command["argv"], "--model_dir")
                assert model_dir == producers[key]
                assert _option(command["argv"], "--checkpoint_name") == "best_model.pt"
                assert f"seed{seed}" in _option(command["argv"], "--exclude_ref_ids")
                if stage in {"s5", "depth23"}:
                    assert _option(command["argv"], "--seed") == "20260501"
                    assert _option(command["argv"], "--clean_eval_json") == (
                        f"{model_dir}/eval_result_v7_exclrefs_crop1000.json"
                    )


def test_same_run_id_is_path_isolated_across_all_three_seeds():
    records = _records_by_seed()
    run_id = "pytest_effnet_matched_shared_run_id"
    producer_dirs_by_seed: dict[int, set[str]] = {}
    consumer_outputs_by_seed: dict[int, set[str]] = {}
    for seed in SEEDS:
        train = _load(records[seed]["stages"]["train"], seed=seed, run_id=run_id)
        train_manifest = _manifest(train, build_runner_commands(train), run_id)
        producer_dirs_by_seed[seed] = {
            child["child_run_dir"]
            for child in train_manifest["artifact_trace"]["expected_outputs"]["child_runs"]
        }
        consumer_outputs_by_seed[seed] = set()
        for stage in ("clean", "s5", "depth23"):
            config = _load(records[seed]["stages"][stage], seed=seed, run_id=run_id)
            commands = build_runner_commands(config)
            consumer_outputs_by_seed[seed].update(
                _option(command["argv"], "--output_path") for command in commands
            )

    for left_index, left_seed in enumerate(SEEDS):
        for right_seed in SEEDS[left_index + 1 :]:
            assert producer_dirs_by_seed[left_seed].isdisjoint(producer_dirs_by_seed[right_seed])
            assert consumer_outputs_by_seed[left_seed].isdisjoint(consumer_outputs_by_seed[right_seed])


def test_cross_seed_or_run_id_consumer_cannot_equal_the_producer_path():
    records = _records_by_seed()
    producer_config = _load(records[20260531]["stages"]["train"], seed=20260531, run_id="shared_run")
    producer_commands = build_runner_commands(producer_config)
    producer_manifest = _manifest(producer_config, producer_commands, "shared_run")
    producer_dirs = {
        (child["center"], child["protocol_claim"]["arm"]): child["child_run_dir"]
        for child in producer_manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    }
    mismatched_consumers = (
        _load(records[20260531]["stages"]["clean"], seed=20260531, run_id="different_run"),
        _load(records[20260611]["stages"]["clean"], seed=20260611, run_id="shared_run"),
    )
    for consumer_config in mismatched_consumers:
        consumer_commands = {
            _key(command): command
            for command in build_runner_commands(consumer_config)
        }
        assert all(
            _option(consumer_commands[key]["argv"], "--model_dir") != producer_dirs[key]
            for key in producer_dirs
        )


def test_seed_matrix_and_missing_grid_cell_are_rejected_by_contract():
    config = _load(CANONICAL_CONFIGS["train"], seed=20260601)
    with_seed_axis = copy.deepcopy(config)
    with_seed_axis["runner"]["matrix"]["seed"] = list(SEEDS)
    with pytest.raises(ConfigError, match="keys must be exactly center and case"):
        build_runner_commands(with_seed_axis)

    missing = build_runner_commands(config)[:-1]
    with pytest.raises(AssertionError):
        _assert_exact_grid(missing)


def test_replications_do_not_expand_latest_mainline_or_canonical_golden():
    index = _index()
    contract = json.loads(GOLDEN.read_text(encoding="utf-8"))
    rendered = GOLDEN.read_text(encoding="utf-8")
    assert len(index["latest_mainline"]["stages"]) == 10
    assert len(index["managed_experiments"]) == 10
    assert len(contract["matched_effnet"]["producer_dirs"]) == 20
    assert all("seed20260601" in path for path in contract["matched_effnet"]["producer_dirs"].values())
    assert "configs/replications/" not in rendered
    assert len(rendered.splitlines()) <= 2000
    assert len(GENERATOR.read_text(encoding="utf-8").splitlines()) <= 200
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
