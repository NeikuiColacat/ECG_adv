"""Task 6B contracts for the matched EfficientNet matrix and honest claim scope."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from ecg_adv_gen.config import (
    ConfigError,
    build_postprocess_commands,
    build_runner_commands,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
)
from ecg_adv_gen.matched_effnet import MATCHED_EFFNET_ARMS, matched_effnet_arm
from ecg_adv_gen.evaluation import PN2021_ALL_ZERO_KEPT_REFEXCLUDED
from ecg_adv_gen.reporting import export_metrics, export_paper_table


REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / "configs/local/linbinhao_server.example.yaml"
TRAIN = REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml"
CLEAN = REPO / "configs/experiments/pn2021_eval_v7_sjr_rgq_refexcluded.yaml"
S5 = REPO / "configs/experiments/pn2021c_effnet_threechain_locked_official_s5.yaml"
DEPTH23 = REPO / "configs/experiments/pn2021c_effnet_official_s5_depth23_composite.yaml"
HISTORICAL_DIRECT = REPO / "configs/experiments/effnet_direct_k500_v7_sjr_rgq.yaml"
INDEX = REPO / "configs/active_scripts.yaml"
LOCKED_PROTOCOL_DOC = REPO / "docs/pipelines/pn2021c_vae_lhat_augmix_locked_protocol_20260618.md"
CLAIM_SCOPE = "known_family_corruption_robustness"
ATOMIC_OPERATORS = (
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
)
SOURCE_SHA256 = "f7a4b05d85da8352013b67a0d56d8d47378ab2abc196d1f3a6ba8bbf15aa40ea"


def _load_train(run_id: str = "pytest_task6b") -> dict:
    return load_experiment_config(TRAIN, LOCAL, runtime_context={"run_id": run_id})


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _option_values(argv: list[str], name: str) -> list[str]:
    start = argv.index(name) + 1
    values: list[str] = []
    for item in argv[start:]:
        if item.startswith("--"):
            break
        values.append(item)
    return values


def _manifest(path: Path, run_id: str) -> tuple[dict, list[dict]]:
    config = load_experiment_config(path, LOCAL, runtime_context={"run_id": run_id})
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    return make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id=run_id,
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    ), commands


def test_operator_resolver_rejects_empty_duplicate_unknown_and_expands_composites():
    from ecg_adv_gen.config.protocol_claim import ProtocolClaimError, resolve_operator_set

    assert resolve_operator_set(
        ["powerline_noise+emg_noise", "baseline_wander+baseline_shift+random_leads_masking"],
        allowed=ATOMIC_OPERATORS,
        allow_composites=True,
        field="evaluation.corruptions",
    ) == ATOMIC_OPERATORS
    for values, match in [([], "non-empty"), (["emg_noise", "emg_noise"], "duplicate"), (["unknown"], "unknown")]:
        with pytest.raises(ProtocolClaimError, match=match):
            resolve_operator_set(
                values,
                allowed=ATOMIC_OPERATORS,
                allow_composites=True,
                field="operators",
            )
    for values, allow_composites, match in [
        ("powerline_noise", True, "non-empty"),
        ([None], True, "strings"),
        (["powerline_noise+powerline_noise"], True, "duplicate atomic"),
        (["powerline_noise++emg_noise"], True, "unknown"),
        (["powerline_noise+emg_noise"], False, "atomic operators"),
    ]:
        with pytest.raises(ProtocolClaimError, match=match):
            resolve_operator_set(
                values,
                allowed=ATOMIC_OPERATORS,
                allow_composites=allow_composites,
                field="operators",
            )


def test_protocol_claim_rejects_explicitly_empty_executable_eval_operators():
    from ecg_adv_gen.config.protocol_claim import ProtocolClaimError, resolve_protocol_claim

    paper = _load_train()["paper_protocol"]
    with pytest.raises(ProtocolClaimError, match="evaluation.corruptions must be non-empty"):
        resolve_protocol_claim(paper, arm="a5", evaluation_operators=[])

    config = load_experiment_config(
        S5,
        LOCAL,
        runtime_context={"run_id": "pytest_task6b_empty_eval"},
    )
    config["evaluation"].pop("corruption_set", None)
    config["evaluation"]["corruptions"] = []
    with pytest.raises(ConfigError, match="evaluation.corruptions must be non-empty"):
        build_runner_commands(config)


@pytest.mark.parametrize("operator_kind", ["training", "evaluation"])
def test_protocol_claim_rejects_reordered_executable_operators(operator_kind: str):
    from ecg_adv_gen.config.protocol_claim import ProtocolClaimError, resolve_protocol_claim

    kwargs = {f"{operator_kind}_operators": list(reversed(ATOMIC_OPERATORS))}
    expected = (
        "training command operators do not match train_atomic"
        if operator_kind == "training"
        else "evaluation operators must resolve"
    )
    with pytest.raises(ProtocolClaimError, match=expected):
        resolve_protocol_claim(_load_train()["paper_protocol"], arm="a5", **kwargs)

    if operator_kind == "training":
        config = _load_train("pytest_task6b_reversed_train")
        config["adaptation"]["latent_augmix"]["ops"] = list(reversed(ATOMIC_OPERATORS))
    else:
        config = load_experiment_config(
            S5,
            LOCAL,
            runtime_context={"run_id": "pytest_task6b_reversed_s5"},
        )
        config["evaluation"]["corruptions"] = list(reversed(ATOMIC_OPERATORS))
    with pytest.raises(ConfigError, match=expected):
        build_runner_commands(config)


def test_clean_postprocess_preserves_arm_identity():
    config = load_experiment_config(
        CLEAN,
        LOCAL,
        runtime_context={"run_id": "pytest_task6b_postprocess"},
    )
    postprocess = build_postprocess_commands(config)
    metrics_export = next(
        command
        for command in postprocess
        if Path(command["argv"][1]).name == "export_metrics_long.py"
    )
    assert "--run-id" not in metrics_export["argv"]
    for command in build_runner_commands(config):
        output_path = Path(_option(command["argv"], "--output_path"))
        assert output_path.parent.name == command["matrix"]["arm"]
        assert output_path.parent.parent.name == command["matrix"]["center"]

    invalid = copy.deepcopy(config)
    invalid["postprocess"]["commands"][0]["argv"].extend(
        ["--run-id", "effnet_matched_a0_a2_a3_a4_a5"]
    )
    with pytest.raises(ConfigError, match="preserve per-arm run identity"):
        build_postprocess_commands(invalid)


def test_matched_train_delegates_reporting_to_clean_arm_matrix_and_guard_recognizes_case_arms():
    train = _load_train("pytest_task6b_train_postprocess")
    assert build_postprocess_commands(train) == []

    clean = load_experiment_config(
        CLEAN,
        LOCAL,
        runtime_context={"run_id": "pytest_task6b_invalid_case_postprocess"},
    )
    invalid = copy.deepcopy(train)
    invalid["postprocess"] = {
        "commands": [copy.deepcopy(clean["postprocess"]["commands"][0])]
    }
    invalid["postprocess"]["commands"][0]["argv"].extend(
        ["--run-id", "collapsed_case_matrix"]
    )
    with pytest.raises(ConfigError, match="preserve per-arm run identity"):
        build_postprocess_commands(invalid)


def test_clean_reporting_sentinel_produces_exactly_five_arm_rows_with_four_centers_each(tmp_path: Path):
    centers = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
    for arm_index, arm in enumerate(MATCHED_EFFNET_ARMS):
        for center_index, center in enumerate(centers):
            artifact = tmp_path / "clean" / center / arm / "eval_result_v7_super5_sjr_rgq_refexcluded.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            value = 0.5 + 0.01 * arm_index + 0.001 * center_index
            artifact.write_text(
                json.dumps(
                    {
                        "mapping": {
                            "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                            "mapping_hash": "555ec85d5b51",
                        },
                        "class_names": ["CD", "HYP", "MI", "NORM", "STTC"],
                        "pn2021": {
                            "per_center": {
                                center: {
                                    "macro_auroc": value + 0.1,
                                    "macro_auprc": value,
                                    "drop_all_zero_macro_auroc": value + 0.1,
                                    "drop_all_zero_macro_auprc": value,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

    metrics = export_metrics(
        [tmp_path / "clean"],
        tmp_path / "metrics",
        expected_mapping_version="v7_super5_sjr_rgq_review_20260528",
        expected_mapping_hash="555ec85d5b51",
        filter_to_target_center=True,
        target_centers=centers,
    )
    table = export_paper_table(
        Path(metrics["metrics_long"]),
        tmp_path / "paper_table",
        view=PN2021_ALL_ZERO_KEPT_REFEXCLUDED,
        dataset="pn2021",
        centers=centers,
        include_dataset_rows=False,
    )
    with Path(table["paper_table"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert {row["run_id"] for row in rows} == set(MATCHED_EFFNET_ARMS)
    assert all(row["scope"] == "center_mean" and row["n_centers"] == "4" for row in rows)


def test_locked_doc_forbids_last_checkpoint_for_matched_matrix():
    text = LOCKED_PROTOCOL_DOC.read_text(encoding="utf-8")
    assert "fixed last-checkpoint evaluation is allowed" not in text
    assert "Select `best_model.pt` by K500-internal macro AUPRC" in text


def test_overlap_is_resolved_per_arm_and_rejected_before_command_build_for_other_scope(monkeypatch):
    from ecg_adv_gen.config.protocol_claim import resolve_protocol_claim

    config = _load_train()
    paper = config["paper_protocol"]
    a2 = resolve_protocol_claim(paper, arm="a2")
    a3 = resolve_protocol_claim(paper, arm="a3")
    assert a2 == {
        "arm": "a2",
        "claim_scope": CLAIM_SCOPE,
        "train_operator_set_resolved": list(ATOMIC_OPERATORS),
        "eval_operator_set_resolved": list(ATOMIC_OPERATORS),
        "overlap_operator_set_resolved": list(ATOMIC_OPERATORS),
        "overlap_allowed_by_claim_scope": True,
    }
    assert a3["train_operator_set_resolved"] == []
    assert a3["overlap_operator_set_resolved"] == []

    config["paper_protocol"]["claim_scope"] = "unseen_family_generalization"
    with pytest.raises(ConfigError, match="overlap.*known_family_corruption_robustness"):
        validate_experiment_config(config, repo_root=REPO)
    called = False

    def forbidden_adapter(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("adapter must not run before the claim-scope gate")

    monkeypatch.setattr("ecg_adv_gen.config.loader.build_runner_adapter_argv", forbidden_adapter)
    with pytest.raises(ConfigError, match="overlap.*known_family_corruption_robustness"):
        build_runner_commands(config)
    assert called is False


def test_training_operator_drift_is_rejected_before_adapter_build(monkeypatch):
    config = _load_train()
    config["adaptation"]["latent_augmix"]["ops"][-1] = "not_an_official_operator"
    called = False

    def forbidden_adapter(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("adapter must not run before executable operator validation")

    monkeypatch.setattr("ecg_adv_gen.config.loader.build_runner_adapter_argv", forbidden_adapter)
    with pytest.raises(ConfigError, match="unknown operators"):
        build_runner_commands(config)
    assert called is False


def test_locked_effnet_config_expands_only_center_by_canonical_case_to_twenty_commands():
    config = _load_train()
    validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)

    assert len(commands) == 20
    assert list(config["paper_protocol"]["operator_sets"]["train_atomic"]) == list(ATOMIC_OPERATORS)
    assert list(config["paper_protocol"]["operator_sets"]["eval_atomic"]) == list(ATOMIC_OPERATORS)
    assert set(config["runner"]["matrix"]) == {"center", "case"}
    assert [case["arm"] for case in config["runner"]["matrix"]["case"]] == list(MATCHED_EFFNET_ARMS)
    assert Counter((cmd["matrix"]["center"], cmd["matrix"]["case"]["arm"]) for cmd in commands) == Counter(
        (center, arm)
        for center in config["paper_protocol"]["centers"]["target_4"]
        for arm in MATCHED_EFFNET_ARMS
    )
    assert all(cmd["matrix"]["case"]["arm"] != "a1" for cmd in commands)
    for command in commands:
        case = command["matrix"]["case"]
        row = matched_effnet_arm(case["arm"])
        assert case == {
            "arm": case["arm"],
            "role": row.role,
            "vae_lhat": row.vae_lhat,
            "raw_augmix": row.raw_augmix,
            "augmix_view_bce": row.augmix_view_bce,
            "jsd": row.jsd,
            "rho": row.target_adv_fraction,
            "third_chain_route": row.third_chain_route,
        }
        assert _option(command["argv"], "--comparison_arm") == case["arm"]


def test_matched_case_expansion_is_atomic_and_rejects_duplicate_or_independent_axes(monkeypatch):
    called = False

    def forbidden_adapter(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid matrix must fail before adapter expansion")

    monkeypatch.setattr("ecg_adv_gen.config.loader.build_runner_adapter_argv", forbidden_adapter)
    mutations = []
    extra_axis = _load_train()
    extra_axis["runner"]["matrix"]["rho"] = [0.0, 0.5]
    mutations.append((extra_axis, "exactly center and case"))
    duplicate_center = _load_train()
    duplicate_center["runner"]["matrix"]["center"].append("ningbo")
    mutations.append((duplicate_center, "duplicate centers"))
    duplicate_arm = _load_train()
    duplicate_arm["runner"]["matrix"]["case"].append(copy.deepcopy(duplicate_arm["runner"]["matrix"]["case"][0]))
    mutations.append((duplicate_arm, "duplicate arm"))
    drift = _load_train()
    drift["runner"]["matrix"]["case"][-1]["jsd"] = False
    mutations.append((drift, "canonical matched EffNet arm"))
    for config, match in mutations:
        with pytest.raises(ConfigError, match=match):
            build_runner_commands(config)
    assert called is False


def test_all_matched_arms_share_initialization_split_selection_budget_and_reference_contracts():
    manifest, commands = _manifest(TRAIN, "pytest_task6b_shared_contract")
    for option in (
        "--init_ckpt", "--init_checkpoint_sha256", "--epochs", "--seed",
        "--train_batch_size", "--lr", "--target_real_val_fraction",
        "--target_real_val_seed", "--selection_metric", "--source_floor_max_drop",
    ):
        assert len({_option(command["argv"], option) for command in commands}) == 1
    refs = manifest["artifact_trace"]["inputs"]["k500_refs"]
    assert {(item["k"], item["seed"]) for item in refs} == {(500, 20260601)}
    assert len({item["ref_meta_json"]["path"] for item in refs}) == 4
    for center in ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"):
        assert len({item["ref_meta_json"]["path"] for item in refs if item["center"] == center}) == 1
    for child in manifest["artifact_trace"]["expected_outputs"]["child_runs"]:
        assert any(item["role"] == "best_model" for item in child["expected_artifacts"])


def test_manifest_records_global_and_child_claims_with_per_arm_required_metrics():
    config = _load_train()
    paths = validate_experiment_config(config, repo_root=REPO)
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=paths,
        run_id="pytest_task6b",
        cli_args=argparse.Namespace(dry_run=True, write_plan=False),
    )

    assert manifest["paper_protocol"]["claim_scope"] == CLAIM_SCOPE
    global_claim = manifest["artifact_trace"]["protocol_claim"]
    assert global_claim["claim_scope"] == CLAIM_SCOPE
    assert global_claim["train_operator_set_resolved"] == list(ATOMIC_OPERATORS)
    assert global_claim["eval_operator_set_resolved"] == list(ATOMIC_OPERATORS)
    children = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(children) == 20
    for child in children:
        arm = child["matrix"]["case"]["arm"]
        claim = child["protocol_claim"]
        assert claim["arm"] == arm
        assert claim["overlap_allowed_by_claim_scope"] is True
        if matched_effnet_arm(arm).raw_augmix:
            assert claim["train_operator_set_resolved"] == list(ATOMIC_OPERATORS)
        else:
            assert claim["train_operator_set_resolved"] == []
        required = set(child["required_epoch_metrics"])
        if matched_effnet_arm(arm).vae_lhat:
            assert {"atk_init", "atk_anchor", "latent_hull_diagnostics"} <= required
        else:
            assert {"atk_init", "atk_anchor", "latent_hull_diagnostics"}.isdisjoint(required)


def test_active_comparison_surface_has_five_roles_reference_only_source_and_optional_ecgfounder():
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    surface = index["latest_mainline"]["comparison_surface"]
    arms = surface["efficientnet_arms"]

    assert list(arms) == list(MATCHED_EFFNET_ARMS)
    assert [arms[arm]["role"] for arm in MATCHED_EFFNET_ARMS] == [matched_effnet_arm(arm).role for arm in MATCHED_EFFNET_ARMS]
    assert all(arms[arm]["evidence_status"] == "EVIDENCE_PENDING" for arm in MATCHED_EFFNET_ARMS)
    source = surface["source_reference"]
    assert source == {
        "role": "reference_only_ptbxl_source_initialization",
        "evidence_path": "configs/evidence/effnet1dv2_ptbxl_source_super5_20260503.json",
        "checkpoint_sha256": SOURCE_SHA256,
        "training_arm": False,
        "evaluation_ref_exclusion": "same_k500_ref_exclusion_as_matched_arms",
    }
    assert surface["historical_direct_k500"]["comparison_arm"] is None
    assert surface["historical_direct_k500"]["matched_baseline"] is False
    optional = surface["optional_ecgfounder_replication"]
    assert optional["scope"] == "optional_backbone_replication"
    assert optional["arms"] == ["a0", "a3", "a5"]
    assert optional["included_in_default_commands"] is False
    assert "model-agnostic" not in str(surface).lower()


def test_historical_direct_manifest_is_not_a_trained_matched_arm():
    manifest, _ = _manifest(HISTORICAL_DIRECT, "pytest_task6b_historical_direct")
    for child in manifest["artifact_trace"]["expected_outputs"]["child_runs"]:
        claim = child["protocol_claim"]
        assert claim["arm"] == "not_applicable"
        assert claim["train_operator_set_resolved"] == []
        assert claim["overlap_operator_set_resolved"] == []


def test_clean_s5_and_depth23_expand_center_by_arm_and_consume_exact_training_producer_dirs():
    run_id = "pytest_task6b_cross_stage"
    train_manifest, _ = _manifest(TRAIN, run_id)
    producer_dirs = {
        (child["center"], child["protocol_claim"]["arm"]): child["child_run_dir"]
        for child in train_manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    }
    assert len(producer_dirs) == 20

    for eval_path in (CLEAN, S5, DEPTH23):
        eval_manifest, commands = _manifest(eval_path, run_id)
        assert len(commands) == 20
        assert Counter((cmd["matrix"]["center"], cmd["matrix"]["arm"]) for cmd in commands) == Counter(producer_dirs.keys())
        for command in commands:
            key = (command["matrix"]["center"], command["matrix"]["arm"])
            assert _option(command["argv"], "--model_dir") == producer_dirs[key]
            assert _option(command["argv"], "--checkpoint_name") == "best_model.pt"
            joined = " ".join(command["argv"])
            assert "lam0p05" not in joined
            assert f"_arm{key[1]}_" in _option(command["argv"], "--model_dir")
        claims = [
            child["protocol_claim"]
            for child in eval_manifest["artifact_trace"]["expected_outputs"]["child_runs"]
        ]
        assert [claim["arm"] for claim in claims] == [cmd["matrix"]["arm"] for cmd in commands]
        assert all(claim["eval_operator_set_resolved"] == list(ATOMIC_OPERATORS) for claim in claims)


def test_depth23_child_claim_expands_composites_to_same_five_atomic_families():
    from ecg_adv_gen.evaluation.pn2021c_protocol import official_s5_depth23_composites
    from ecg_adv_gen.config.protocol_claim import resolve_operator_set

    manifest, commands = _manifest(DEPTH23, "pytest_task6b_depth23_claim")
    composite_tokens = official_s5_depth23_composites()
    assert any("+" in item for item in composite_tokens)
    assert resolve_operator_set(
        composite_tokens,
        allowed=ATOMIC_OPERATORS,
        allow_composites=True,
        field="evaluation.corruptions",
    ) == (
        "baseline_shift",
        "baseline_wander",
        "emg_noise",
        "powerline_noise",
        "random_leads_masking",
    )
    assert all(
        _option_values(command["argv"], "--corruptions") == composite_tokens
        for command in commands
    )
    global_claim = manifest["artifact_trace"]["protocol_claim"]
    # The argv/golden retain ordered composite tokens; the set-valued claim is
    # projected into the declared canonical atomic-family order.
    assert global_claim["eval_operator_set_resolved"] == list(ATOMIC_OPERATORS)
    assert all(
        child["protocol_claim"]["eval_operator_set_resolved"] == list(ATOMIC_OPERATORS)
        for child in manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    )
