"""Contracts for replication path isolation, K500 preflight, and audit reporting."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from ecg_adv_gen.config import (
    attach_replication_preflight,
    audit_active_managed_configs,
    audit_replication_command_grid,
    audit_replication_path_isolation,
    audit_replication_producer_consumers,
    audit_replication_surfaces,
    build_runner_commands,
    build_replication_k500_groups,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
    verify_replication_k500_groups,
    verify_replication_validation_report,
    verify_required_inputs,
)
from ecg_adv_gen.data.kshot_artifacts import canonical_kshot_base


REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "configs" / "active_scripts.yaml"
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"
MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
MAPPING_HASH = "555ec85d5b51"
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
SUFFIXES = ("ref_meta.json", "signals.npz", "latent.npz", "raw1000.npz", "class_trust.json")
STUDY_EXPECTATIONS = {
    "effnet_f004_rho_sweep_onecenter_smoke.yaml": ([20260601], 3),
    "effnet_f004_rho_sweep_k500.yaml": ([20260601], 12),
    "pn2021c_effnet_f004_rho_sweep_official_s5.yaml": ([20260601], 12),
    "pn2021c_effnet_f004_rho_sweep_depth23.yaml": ([20260601], 12),
    "f005_anchor_geometry_control_smoke.yaml": ([20260601], 2),
    "f005_anchor_geometry_control_train.yaml": ([20260531, 20260601, 20260611], 24),
    "f005_anchor_geometry_control_pn2021_clean.yaml": ([20260531, 20260601, 20260611], 24),
    "f005_anchor_geometry_control_pn2021c_s5.yaml": ([20260531, 20260601, 20260611], 24),
    "f005_anchor_geometry_control_pn2021c_depth23.yaml": ([20260531, 20260601, 20260611], 24),
}


def _write_local_config(tmp_path: Path) -> Path:
    local = yaml.safe_load(LOCAL_EXAMPLE.read_text(encoding="utf-8"))
    data_root = tmp_path / "data"
    local["host"]["home_root"] = str(tmp_path)
    local["paths"].update(
        {
            "data_root": str(data_root),
            "model_root": str(data_root / "models"),
            "output_root": str(data_root / "runs"),
            "cache_root": str(data_root / "cache"),
            "tmp_root": str(data_root / "tmp"),
            "short_tmp_root": str(tmp_path / "short_tmp"),
        }
    )
    local["python"]["executable"] = sys.executable
    # The test checkout and pytest's temporary data root have different
    # ancestors; '/' is used only inside this hermetic config fixture.
    local["safety"]["write_boundary"] = "/"
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(local, sort_keys=False), encoding="utf-8")
    return path


def _subset_root(tmp_path: Path, *, canonical: bool) -> Path:
    family = (
        "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq"
        if canonical
        else "paper_matched_effnet_k500_v7_fixedk_three_seed_20260711"
    )
    return (
        tmp_path
        / "data"
        / family
        / "subsets"
    )


def _write_group(root: Path, *, center: str, k: int, seed: int) -> Path:
    base = canonical_kshot_base(root, center, k=k, seed=seed)
    base.parent.mkdir(parents=True, exist_ok=True)
    record_ids = np.asarray([f"{center}_{idx:04d}" for idx in range(k)])
    labels = np.zeros((k, 5), dtype=np.float32)
    labels[:, 3] = 1.0
    waveform = np.broadcast_to(
        np.linspace(-1.0, 1.0, 1000, dtype=np.float32)[None, :, None],
        (k, 1000, 12),
    ).copy()
    latents = np.broadcast_to(
        np.linspace(-0.5, 0.5, 128, dtype=np.float32)[None, None, :],
        (k, 4, 128),
    ).copy()
    class_names = np.asarray(["CD", "HYP", "MI", "NORM", "STTC"])
    shared = {
        "labels": labels,
        "record_ids": record_ids,
        "center_name": np.asarray(center),
        "class_names": class_names,
        "mapping_version": np.asarray(MAPPING_VERSION),
        "mapping_hash": np.asarray(MAPPING_HASH),
    }
    np.savez_compressed(base.with_suffix(".signals.npz"), signals=waveform, **shared)
    np.savez_compressed(base.with_suffix(".latent.npz"), latents=latents, **shared)
    np.savez_compressed(
        base.with_suffix(".raw1000.npz"),
        signals=waveform,
        labels=labels,
        record_ids=record_ids,
        metadata=np.asarray(
            [
                {
                    "center": center,
                    "source_ref_meta_json": str(base.with_suffix(".ref_meta.json")),
                    "n_records": k,
                    "class_names": class_names.tolist(),
                    "record_ids_source": "ref_meta_order",
                    "signals_are_pre_zscore": True,
                    "preprocess": {
                        "target_fs": 100,
                        "target_len": 1000,
                        "preprocess_mode": "minimal_resample",
                        "norm_mode": "none",
                    },
                    "model_input_preprocess": {"norm_mode": "per_sample_global", "crop_len": 1000},
                }
            ],
            dtype=object,
        ),
    )
    label_counts = dict(zip(class_names.tolist(), labels.sum(axis=0).astype(int).tolist()))
    base.with_suffix(".ref_meta.json").write_text(
        json.dumps(
            {
                "center": center,
                "K": k,
                "selection_seed": seed,
                "fixed_k": k,
                "protocol": "fixed_k",
                "protocol_tag": f"k{k}",
                "sampling_policy": "deterministic_random_from_v7_nonzero_eligible_pool",
                "raw_v7_nonzero_count": k,
                "eligible_v7_nonzero_with_latents": k,
                "ref_record_ids": record_ids.tolist(),
                "mapping_version": MAPPING_VERSION,
                "mapping_hash": MAPPING_HASH,
                "class_names": class_names.tolist(),
                "label_counts": label_counts,
            }
        ),
        encoding="utf-8",
    )
    base.with_suffix(".class_trust.json").write_text(
        json.dumps(
            {
                "center": center,
                "mapping_version": MAPPING_VERSION,
                "mapping_hash": MAPPING_HASH,
                "label_counts": label_counts,
                "class_trust": {name: float(count > 0) for name, count in label_counts.items()},
            }
        ),
        encoding="utf-8",
    )
    return base


def _write_validation_report(root: Path, *, seeds: tuple[int, ...]) -> tuple[Path, str]:
    path = root.parent / "validation_report.json"
    groups = [
        {
            "seed": seed,
            "center": center,
            "checks": {"identity": True, "protocol": True, "mapping": True, "shapes": True},
        }
        for seed in seeds
        for center in CENTERS
    ]
    path.write_text(
        json.dumps({"passed": True, "errors": [], "root": str(root.parent), "groups": groups}),
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_bound_report(
    root: Path, *, seeds: tuple[int, ...], k: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    key_map = {
        "ref_meta.json": "ref_meta",
        "signals.npz": "signals",
        "latent.npz": "latent",
        "raw1000.npz": "raw1000",
        "class_trust.json": "class_trust",
    }
    groups = [
        group
        for seed in seeds
        for group in build_replication_k500_groups(
            subset_root=root,
            centers=CENTERS,
            k=k,
            seed=seed,
            artifact_suffixes=SUFFIXES,
        )
    ]
    rows = []
    for group in groups:
        artifacts = group["artifacts"]
        rows.append(
            {
                "seed": group["seed"],
                "center": group["center"],
                "checks": {"identity": True, "numeric": True},
                "artifact_sha256": {
                    key_map[suffix]: hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
                    for suffix, record in artifacts.items()
                },
                "artifact_sizes": {
                    key_map[suffix]: Path(record["path"]).stat().st_size
                    for suffix, record in artifacts.items()
                },
            }
        )
    path = root.parent / "validation_report.json"
    path.write_text(json.dumps({"passed": True, "errors": [], "groups": rows}), encoding="utf-8")
    return {
        "path": str(path),
        "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_group_count": len(groups),
    }, groups


def _rewrite_npz(path: Path, *, allow_pickle: bool, updates: dict[str, Any]) -> None:
    with np.load(path, allow_pickle=allow_pickle) as payload:
        values = {key: payload[key] for key in payload.files}
    values.update(updates)
    np.savez_compressed(path, **values)


def _surface() -> dict[str, Any]:
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    return index["replication_surfaces"][0]


def _study_entries(index: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    index = index or yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    return {
        Path(entry["config"]).name: {"surface": surface["name"], **entry}
        for surface in index.get("study_surfaces") or []
        for entry in surface.get("entries") or []
    }


def _manifest(config: dict[str, Any], *, run_id: str = "pytest-study-surface") -> dict[str, Any]:
    commands = build_runner_commands(config)
    return make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=validate_experiment_config(config, repo_root=REPO),
        run_id=run_id,
        cli_args=Namespace(dry_run=True, write_plan=False),
    )


def _attached_manifest(config_path: Path, *, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_experiment_config(
        config_path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": run_id},
    )
    return config, attach_replication_preflight(
        _manifest(config, run_id=run_id),
        config,
        repo_root=REPO,
        index_path=INDEX,
    )


def test_study_surface_inventory_and_input_contract_are_exact() -> None:
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    entries = _study_entries(index)
    actual = {path.name for path in (REPO / "configs/studies").glob("*.yaml")}

    assert set(entries) == actual == set(STUDY_EXPECTATIONS)
    assert len(index["latest_mainline"]["stages"]) == 10
    assert len(index["managed_experiments"]) == 10
    assert "configs/studies" in index["launch_surface_policy"]["config_roots"]
    assert {surface["input_contract_ref"] for surface in index["study_surfaces"]} == {
        "effnet_matched_three_seed"
    }
    for name, (seeds, command_count) in STUDY_EXPECTATIONS.items():
        assert entries[name]["input_seeds"] == seeds
        assert entries[name]["expected_command_count"] == command_count


@pytest.mark.parametrize("name", sorted(STUDY_EXPECTATIONS))
def test_indexed_study_entry_binds_observed_seeds_and_full_validation_scope(name: str) -> None:
    expected_seeds, expected_commands = STUDY_EXPECTATIONS[name]
    path = REPO / "configs/studies" / name
    config = load_experiment_config(
        path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": f"pytest-study-{path.stem}"},
    )
    manifest = _manifest(config)
    assert len(manifest["commands"]) == expected_commands

    attached = attach_replication_preflight(
        manifest,
        config,
        repo_root=REPO,
        index_path=INDEX,
    )
    trace = attached["artifact_trace"]
    current = trace["inputs"]["replication_k500_groups"]
    validation = trace["inputs"]["replication_validation_groups"]
    preflight = trace["replication_preflight"]
    input_contract = _surface()
    matrix_centers = {
        str(command["matrix"]["center"])
        for command in manifest["commands"]
    }
    assert sorted({int(group["seed"]) for group in current}) == expected_seeds
    assert len(current) == len(matrix_centers) * len(expected_seeds)
    assert sorted({int(group["seed"]) for group in validation}) == [
        20260531,
        20260601,
        20260611,
    ]
    assert len(validation) == 12
    assert preflight["seeds"] == expected_seeds
    assert preflight["validation_group_count"] == 12
    assert preflight["validation_artifact_count"] == 60
    assert preflight["materialization_artifact_suffixes"] == input_contract[
        "materialization_artifact_suffixes"
    ]
    assert preflight["validation_report"]["expected_sha256"] == input_contract[
        "validation_report"
    ]["sha256"]


@pytest.mark.parametrize("drift", ("duplicate", "seeds", "root", "sources", "command_count"))
def test_study_surface_binding_rejects_index_and_config_drift(
    drift: str, tmp_path: Path
) -> None:
    path = REPO / "configs/studies/f005_anchor_geometry_control_smoke.yaml"
    config = load_experiment_config(
        path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest-study-drift"},
    )
    manifest = _manifest(config, run_id="pytest-study-drift")
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    entries = [
        entry
        for surface in index["study_surfaces"]
        for entry in surface["entries"]
        if entry["config"] == "configs/studies/f005_anchor_geometry_control_smoke.yaml"
    ]
    assert len(entries) == 1
    if drift == "duplicate":
        index["study_surfaces"][0]["entries"].append(copy.deepcopy(entries[0]))
    elif drift == "seeds":
        entries[0]["input_seeds"] = [20260531]
    elif drift == "root":
        config["data"]["kshot_subset_root"] = str(
            Path(config["data"]["kshot_subset_root"]).parent.parent / "wrong-family" / "subsets"
        )
    elif drift == "sources":
        config["_config_sources"] = [
            str(REPO / "configs/studies/f005_anchor_geometry_control_train.yaml")
        ]
    else:
        entries[0]["expected_command_count"] = 3
    fixture = tmp_path / "active_scripts.yaml"
    fixture.write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=drift.replace("command_count", "command count")):
        attach_replication_preflight(
            manifest,
            config,
            repo_root=REPO,
            index_path=fixture,
        )


def test_indexed_study_requires_manifest_k500_refs_even_when_top_seed_matches() -> None:
    path = REPO / "configs/studies/effnet_f004_rho_sweep_k500.yaml"
    config = load_experiment_config(
        path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest-study-missing-refs"},
    )
    manifest = _manifest(config, run_id="pytest-study-missing-refs")
    manifest["artifact_trace"]["inputs"]["k500_refs"] = []
    with pytest.raises(ValueError, match="manifest K500 refs"):
        attach_replication_preflight(
            manifest,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


@pytest.mark.parametrize(
    ("override_key", "suffix"),
    [
        ("synth_npz_override", ".latent.npz"),
        ("target_real_npz_override", ".raw1000.npz"),
    ],
)
def test_study_rejects_foreign_wrapper_input_override(
    override_key: str,
    suffix: str,
    tmp_path: Path,
) -> None:
    path = REPO / "configs/studies/effnet_f004_rho_sweep_k500.yaml"
    config = load_experiment_config(
        path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": f"pytest-foreign-{override_key}"},
    )
    config["data"][override_key] = str(tmp_path / f"foreign{suffix}")
    manifest = _manifest(config, run_id=f"pytest-foreign-{override_key}")

    with pytest.raises(ValueError, match="actual child|canonical.*input|override"):
        attach_replication_preflight(
            manifest,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


@pytest.mark.parametrize(
    ("nested_key", "mutation"),
    [
        ("ref_meta_json", "foreign.ref_meta.json"),
        ("signals_npz", "foreign.raw1000.npz"),
        ("latent_npz", "foreign.latent.npz"),
    ],
)
def test_study_rejects_nested_manifest_path_tamper(
    nested_key: str,
    mutation: str,
    tmp_path: Path,
) -> None:
    path = REPO / "configs/studies/effnet_f004_rho_sweep_k500.yaml"
    config = load_experiment_config(
        path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": f"pytest-nested-{nested_key}"},
    )
    manifest = _manifest(config, run_id=f"pytest-nested-{nested_key}")
    nested = manifest["artifact_trace"]["inputs"]["k500_refs"][0][nested_key]
    assert isinstance(nested, dict)
    nested["path"] = str(tmp_path / mutation)

    with pytest.raises(ValueError, match=f"manifest.*{nested_key}|{nested_key}.*canonical"):
        attach_replication_preflight(
            manifest,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


def test_study_rejects_missing_required_manifest_latent() -> None:
    path = REPO / "configs/studies/f005_anchor_geometry_control_train.yaml"
    config = load_experiment_config(
        path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest-missing-latent"},
    )
    manifest = _manifest(config, run_id="pytest-missing-latent")
    manifest["artifact_trace"]["inputs"]["k500_refs"][0]["latent_npz"] = None

    with pytest.raises(ValueError, match="manifest.*latent_npz|latent_npz.*required"):
        attach_replication_preflight(
            manifest,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


def test_study_rejects_final_child_argv_manifest_divergence(tmp_path: Path) -> None:
    path = REPO / "configs/studies/effnet_f004_rho_sweep_onecenter_smoke.yaml"
    config = load_experiment_config(
        path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest-argv-manifest-divergence"},
    )
    manifest = _manifest(config, run_id="pytest-argv-manifest-divergence")
    argv = manifest["commands"][0]["argv"]
    option_index = argv.index("--target_real_npz_override")
    argv[option_index + 1] = str(tmp_path / "foreign.raw1000.npz")

    with pytest.raises(ValueError, match="actual child|manifest.*input|canonical"):
        attach_replication_preflight(
            manifest,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


def test_canonical_train_bindings_follow_actual_arm_consumption() -> None:
    path = REPO / "configs/replications/effnet_matched_train_seed20260601.yaml"
    _config, manifest = _attached_manifest(path, run_id="pytest-canonical-arm-inputs")
    bindings = manifest["artifact_trace"]["inputs"][
        "replication_command_input_bindings"
    ]

    assert len(bindings) == len(manifest["commands"]) == 20
    for binding in bindings:
        arm = binding["arm"]
        consumed = binding["actual_consumed"]
        companions = binding["manifest_companions"]
        assert consumed["ref_meta_json"].endswith(".ref_meta.json")
        assert consumed["raw1000_npz"].endswith(".raw1000.npz")
        assert companions["signals_npz"] == consumed["raw1000_npz"]
        assert consumed["class_trust"]["provenance_only"] is True
        assert consumed["class_trust"]["source_raw1000"] == consumed["raw1000_npz"]
        if arm in {"a3", "a4", "a5"}:
            assert consumed["latent_npz"].endswith(".latent.npz")
            assert companions["latent_npz"] == consumed["latent_npz"]
        else:
            assert arm in {"a0", "a2"}
            assert consumed["latent_npz"] is None
            assert companions["latent_npz"] is None


@pytest.mark.parametrize(
    "name",
    [
        "effnet_f004_rho_sweep_onecenter_smoke.yaml",
        "effnet_f004_rho_sweep_k500.yaml",
        "f005_anchor_geometry_control_smoke.yaml",
        "f005_anchor_geometry_control_train.yaml",
    ],
)
def test_f004_f005_a5_training_bindings_consume_canonical_triplet(name: str) -> None:
    _config, manifest = _attached_manifest(
        REPO / "configs/studies" / name,
        run_id=f"pytest-training-binding-{Path(name).stem}",
    )
    bindings = manifest["artifact_trace"]["inputs"][
        "replication_command_input_bindings"
    ]
    assert len(bindings) == len(manifest["commands"])
    assert all(item["actual_consumed"]["ref_meta_json"].endswith(".ref_meta.json") for item in bindings)
    assert all(item["actual_consumed"]["raw1000_npz"].endswith(".raw1000.npz") for item in bindings)
    assert all(item["actual_consumed"]["latent_npz"].endswith(".latent.npz") for item in bindings)


@pytest.mark.parametrize("stage", ("clean", "s5", "depth23"))
def test_canonical_eval_bindings_use_only_ref_ids_and_provenance_signals(
    stage: str,
) -> None:
    path = REPO / f"configs/replications/effnet_matched_{stage}_seed20260601.yaml"
    _config, manifest = _attached_manifest(path, run_id=f"pytest-canonical-{stage}-inputs")
    bindings = manifest["artifact_trace"]["inputs"][
        "replication_command_input_bindings"
    ]
    expected_per_command = 4 if stage == "clean" else 1
    assert len(bindings) == len(manifest["commands"]) * expected_per_command
    for binding in bindings:
        consumed = binding["actual_consumed"]
        companions = binding["manifest_companions"]
        assert consumed == {
            "ref_meta_json": binding["ref_meta_json"],
            "raw1000_npz": None,
            "latent_npz": None,
            "class_trust": None,
        }
        assert companions["signals_npz"].endswith(".signals.npz")
        assert companions["latent_npz"] is None


def test_command_input_contract_compares_all_four_cell_surfaces_without_dedup() -> None:
    _config, manifest = _attached_manifest(
        REPO / "configs/studies/effnet_f004_rho_sweep_onecenter_smoke.yaml",
        run_id="pytest-four-cell-surfaces",
    )
    contract = manifest["artifact_trace"]["replication_preflight"][
        "command_input_contract"
    ]
    assert contract["passed"] is True
    assert contract["per_command_passed"] is True
    assert contract["expected_cell_count"] == 3
    assert contract["actual_argv_cell_count"] == 3
    assert contract["manifest_ref_cell_count"] == 3
    assert contract["attached_group_cell_count"] == 3


@pytest.mark.parametrize("layout", ("foreign_parent", "prefixed_data_root"))
def test_indexed_study_rejects_same_family_under_noncanonical_data_root(
    layout: str, tmp_path: Path
) -> None:
    path = REPO / "configs/studies/effnet_f004_rho_sweep_k500.yaml"
    config = load_experiment_config(
        path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": f"pytest-study-{layout}"},
    )
    manifest = _manifest(config, run_id=f"pytest-study-{layout}")
    family = _surface()["replicates"][0]["input_root_family"]
    if layout == "foreign_parent":
        foreign_root = tmp_path / "foreign_data" / family / "subsets"
    else:
        foreign_root = Path(f"{config['paths']['data_root']}_shadow") / family / "subsets"
    config["data"]["kshot_subset_root"] = str(foreign_root)
    for ref in manifest["artifact_trace"]["inputs"]["k500_refs"]:
        ref["anchor_base"] = str(
            canonical_kshot_base(
                foreign_root,
                str(ref["center"]),
                k=int(ref["k"]),
                seed=int(ref["seed"]),
            )
        )

    with pytest.raises(ValueError, match="canonical.*input root|input root.*canonical"):
        attach_replication_preflight(
            manifest,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


@pytest.mark.parametrize("drift", ("foreign_child_base", "missing_child_group"))
def test_indexed_study_rejects_manifest_ref_and_attached_group_divergence(
    drift: str,
) -> None:
    path = REPO / "configs/studies/effnet_f004_rho_sweep_k500.yaml"
    config = load_experiment_config(
        path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": f"pytest-study-{drift}"},
    )
    manifest = _manifest(config, run_id=f"pytest-study-{drift}")
    refs = manifest["artifact_trace"]["inputs"]["k500_refs"]
    if drift == "foreign_child_base":
        target = next(ref for ref in refs if ref["center"] == "ningbo")
        target["anchor_base"] = str(
            canonical_kshot_base(
                Path(config["data"]["kshot_subset_root"]),
                "georgia",
                k=int(target["k"]),
                seed=int(target["seed"]),
            )
        )
    else:
        manifest["artifact_trace"]["inputs"]["k500_refs"] = [
            ref for ref in refs if ref["center"] != "ningbo"
        ]

    with pytest.raises(ValueError, match="manifest K500 refs|input cell"):
        attach_replication_preflight(
            manifest,
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


def test_auxiliary_surface_pure_audits_are_reusable_and_fail_closed():
    commands = [
        {"matrix": {"center": center, "arm": arm}}
        for center in ("c1", "c2")
        for arm in ("a", "b")
    ]
    assert audit_replication_command_grid(
        commands,
        centers=["c1", "c2"],
        arms=["a", "b"],
    )["passed"] is True
    missing_grid = audit_replication_command_grid(
        commands[:-1],
        centers=["c1", "c2"],
        arms=["a", "b"],
    )
    assert missing_grid["passed"] is False
    assert missing_grid["missing"] == [{"center": "c2", "arm": "b"}]

    children = [
        {
            "center": "c1",
            "matrix": {"case": {"arm": "a"}},
            "child_run_dir": "/outputs/producer/c1/a",
        }
    ]
    consumers = {
        "eval": [
            {
                "matrix": {"center": "c1", "arm": "a"},
                "argv": [
                    "python",
                    "runner.py",
                    "--model_dir",
                    "/outputs/producer/c1/a",
                    "--output_path",
                    "/outputs/eval/c1/a.json",
                ],
            }
        ]
    }
    binding = audit_replication_producer_consumers(children, consumers)
    assert binding["passed"] is True
    isolation = audit_replication_path_isolation(
        {1: {"/outputs/producer/1"}, 2: {"/outputs/producer/2"}},
        {1: {"/outputs/eval/1"}, 2: {"/outputs/eval/2"}},
    )
    assert isolation["passed"] is True
    overlap = audit_replication_path_isolation(
        {1: {"/same"}, 2: {"/same"}},
        {1: {"/eval/1"}, 2: {"/eval/2"}},
    )
    assert overlap["passed"] is False


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_replication_preflight_fails_closed_for_each_missing_companion(tmp_path: Path, suffix: str):
    root = tmp_path / "subsets"
    base = _write_group(root, center="ningbo", k=2, seed=17)
    groups = build_replication_k500_groups(
        subset_root=root,
        centers=["ningbo"],
        k=2,
        seed=17,
        artifact_suffixes=list(SUFFIXES),
    )
    base.with_suffix(f".{suffix}").unlink()
    report = verify_replication_k500_groups(
        groups,
        mapping_version=MAPPING_VERSION,
        mapping_hash=MAPPING_HASH,
    )
    assert report["passed"] is False
    assert [item["suffix"] for item in report["missing"]] == [suffix]


def test_replication_preflight_rejects_wrong_seed_identity_and_blocks_children(tmp_path: Path):
    root = tmp_path / "subsets"
    base = _write_group(root, center="ningbo", k=2, seed=17)
    meta = json.loads(base.with_suffix(".ref_meta.json").read_text(encoding="utf-8"))
    meta["selection_seed"] = 99
    base.with_suffix(".ref_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    groups = build_replication_k500_groups(
        subset_root=root,
        centers=["ningbo"],
        k=2,
        seed=17,
        artifact_suffixes=list(SUFFIXES),
    )
    manifest = {
        "artifact_trace": {
            "inputs": {
                "checkpoints": [],
                "data_caches": [],
                "k500_refs": [],
                "replication_k500_groups": groups,
            },
            "initialization": {},
        },
        "commands": [],
        "safety": {"managed_child_commands_invoked": False},
    }
    report = verify_required_inputs(manifest)
    assert report["passed"] is False
    assert report["replication_k500_preflight"]["passed"] is False
    assert any("seed" in item["error"] for item in report["replication_k500_preflight"]["identity_errors"])
    assert manifest["safety"]["managed_child_commands_invoked"] is False


@pytest.mark.parametrize("artifact", ("signals", "latent", "raw1000"))
def test_replication_preflight_rejects_nonfinite_or_wrong_shape_arrays(tmp_path: Path, artifact: str):
    root = tmp_path / "subsets"
    base = _write_group(root, center="ningbo", k=2, seed=17)
    path = base.with_suffix(f".{artifact}.npz")
    allow_pickle = artifact == "raw1000"
    with np.load(path, allow_pickle=allow_pickle) as payload:
        values = {key: payload[key] for key in payload.files}
    payload_key = "latents" if artifact == "latent" else "signals"
    values[payload_key] = np.zeros((2, 2, 1), dtype=np.float32)
    values[payload_key].reshape(-1)[0] = np.nan
    np.savez_compressed(path, **values)
    groups = build_replication_k500_groups(
        subset_root=root,
        centers=["ningbo"],
        k=2,
        seed=17,
        artifact_suffixes=list(SUFFIXES),
    )
    report = verify_replication_k500_groups(
        groups,
        mapping_version=MAPPING_VERSION,
        mapping_hash=MAPPING_HASH,
    )
    assert report["passed"] is False
    assert report["identity_errors"]


def test_replication_preflight_rejects_cross_artifact_label_drift(tmp_path: Path):
    root = tmp_path / "subsets"
    base = _write_group(root, center="ningbo", k=2, seed=17)
    path = base.with_suffix(".latent.npz")
    with np.load(path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    values["labels"] = np.zeros((2, 5), dtype=np.float32)
    values["labels"][:, 0] = 1.0
    np.savez_compressed(path, **values)
    groups = build_replication_k500_groups(
        subset_root=root,
        centers=["ningbo"],
        k=2,
        seed=17,
        artifact_suffixes=list(SUFFIXES),
    )
    report = verify_replication_k500_groups(
        groups,
        mapping_version=MAPPING_VERSION,
        mapping_hash=MAPPING_HASH,
    )
    assert report["passed"] is False
    assert any("labels" in item["error"] for item in report["identity_errors"])


def test_replication_preflight_rejects_incorrect_class_trust_value(tmp_path: Path):
    root = tmp_path / "subsets"
    base = _write_group(root, center="ningbo", k=2, seed=17)
    path = base.with_suffix(".class_trust.json")
    trust = json.loads(path.read_text(encoding="utf-8"))
    trust["class_trust"]["NORM"] = 0.0
    path.write_text(json.dumps(trust), encoding="utf-8")
    groups = build_replication_k500_groups(
        subset_root=root,
        centers=["ningbo"],
        k=2,
        seed=17,
        artifact_suffixes=list(SUFFIXES),
    )
    report = verify_replication_k500_groups(
        groups,
        mapping_version=MAPPING_VERSION,
        mapping_hash=MAPPING_HASH,
    )
    assert report["passed"] is False
    assert any("class_trust" in item["error"] for item in report["identity_errors"])


def test_validation_report_binds_each_artifact_hash_and_blocks_tamper(tmp_path: Path):
    root = tmp_path / "subsets"
    base = _write_group(root, center="ningbo", k=2, seed=17)
    groups = build_replication_k500_groups(
        subset_root=root, centers=["ningbo"], k=2, seed=17, artifact_suffixes=list(SUFFIXES)
    )
    key_map = {
        "ref_meta.json": "ref_meta", "signals.npz": "signals", "latent.npz": "latent",
        "raw1000.npz": "raw1000", "class_trust.json": "class_trust",
    }
    artifacts = groups[0]["artifacts"]
    hashes = {
        key_map[suffix]: hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        for suffix, record in artifacts.items()
    }
    sizes = {key_map[suffix]: Path(record["path"]).stat().st_size for suffix, record in artifacts.items()}
    report_path = tmp_path / "validation_report.json"
    report_path.write_text(
        json.dumps({
            "passed": True, "errors": [],
            "groups": [{"seed": 17, "center": "ningbo", "checks": {"identity": True},
                        "artifact_sha256": hashes, "artifact_sizes": sizes}],
        }),
        encoding="utf-8",
    )
    contract = {
        "path": str(report_path),
        "expected_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "expected_group_count": 1,
    }
    assert verify_replication_validation_report(contract, groups)["passed"] is True
    latent_path = base.with_suffix(".latent.npz")
    with np.load(latent_path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    values["latents"] = values["latents"].copy()
    values["latents"][0, 0, 0] += 0.125
    np.savez_compressed(latent_path, **values)
    assert verify_replication_validation_report(contract, groups)["passed"] is False
    manifest = {
        "artifact_trace": {
            "inputs": {"checkpoints": [], "data_caches": [], "k500_refs": [],
                       "replication_k500_groups": groups},
            "initialization": {},
            "metrics": {"mapping_version": MAPPING_VERSION, "mapping_hash": MAPPING_HASH},
            "replication_preflight": {"validation_report": contract},
        },
        "commands": [],
        "safety": {"managed_child_commands_invoked": False},
    }
    assert verify_required_inputs(manifest)["passed"] is False
    assert manifest["safety"]["managed_child_commands_invoked"] is False


def test_single_seed_numeric_preflight_uses_four_groups_but_report_binds_all_twelve(
    tmp_path: Path,
):
    root = tmp_path / "family" / "subsets"
    seeds = (20260531, 20260601, 20260611)
    for seed in seeds:
        for center in CENTERS:
            _write_group(root, center=center, k=2, seed=seed)
    contract, all_groups = _artifact_bound_report(root, seeds=seeds, k=2)

    for seed in seeds:
        current_groups = [group for group in all_groups if group["seed"] == seed]
        manifest = {
            "artifact_trace": {
                "inputs": {
                    "checkpoints": [],
                    "data_caches": [],
                    "k500_refs": [],
                    "replication_k500_groups": current_groups,
                    "replication_validation_groups": all_groups,
                },
                "initialization": {},
                "metrics": {"mapping_version": MAPPING_VERSION, "mapping_hash": MAPPING_HASH},
                "replication_preflight": {"validation_report": contract},
            },
            "commands": [],
            "safety": {"managed_child_commands_invoked": False},
        }
        report = verify_required_inputs(manifest)
        assert report["passed"] is True, report
        preflight = report["replication_k500_preflight"]
        assert preflight["group_count"] == 4
        assert preflight["validation_report"]["bound_group_count"] == 12
        assert preflight["validation_report"]["bound_artifact_count"] == 60

    other_seed = next(group for group in all_groups if group["seed"] == 20260611)
    tampered = Path(other_seed["artifacts"]["latent.npz"]["path"])
    tampered.write_bytes(tampered.read_bytes() + b"tamper")
    current_groups = [group for group in all_groups if group["seed"] == 20260531]
    manifest["artifact_trace"]["inputs"]["replication_k500_groups"] = current_groups
    assert verify_required_inputs(manifest)["passed"] is False
    assert manifest["safety"]["managed_child_commands_invoked"] is False


@pytest.mark.parametrize("seed", (20260531, 20260601, 20260611))
def test_indexed_entry_attaches_current_and_full_surface_groups(seed: int):
    config_path = REPO / f"configs/replications/effnet_matched_train_seed{seed}.yaml"
    config = load_experiment_config(
        config_path,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest-replication-binding"},
    )
    manifest = attach_replication_preflight(
        _manifest(config, run_id="pytest-replication-binding"),
        config,
        repo_root=REPO,
        index_path=INDEX,
    )
    inputs = manifest["artifact_trace"]["inputs"]
    assert len(inputs["replication_k500_groups"]) == 4
    assert {group["seed"] for group in inputs["replication_k500_groups"]} == {seed}
    assert len(inputs["replication_validation_groups"]) == 12
    assert {group["seed"] for group in inputs["replication_validation_groups"]} == {
        20260531,
        20260601,
        20260611,
    }


def test_loader_preserves_replication_experiment_source_chain():
    entry = REPO / "configs/replications/effnet_matched_train_seed20260531.yaml"
    config = load_experiment_config(
        entry,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest-replication-sources"},
    )
    sources = {Path(path).resolve() for path in config["_config_sources"]}
    assert entry.resolve() in sources
    assert (REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml").resolve() in sources
    assert LOCAL_EXAMPLE.resolve() not in sources


def test_unindexed_derived_replication_entry_is_rejected(tmp_path: Path):
    indexed = REPO / "configs/replications/effnet_matched_train_seed20260531.yaml"
    config = load_experiment_config(
        indexed,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest-derived-replication"},
    )
    derived = tmp_path / "derived_replication.yaml"
    config["_entry_config"] = str(derived)
    config["_config_sources"] = [*config.get("_config_sources", []), str(derived)]
    with pytest.raises(ValueError, match="indexed.*replication/study surface"):
        attach_replication_preflight(
            {"artifact_trace": {"inputs": {}}},
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


def test_unindexed_f005_derived_study_using_replication_family_is_explicitly_blocked(
    tmp_path: Path,
):
    entry = REPO / "configs/studies/f005_anchor_geometry_control_train.yaml"
    config = load_experiment_config(
        entry,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest-unindexed-f005-study"},
    )
    derived = tmp_path / "unindexed_f005_study.yaml"
    config["_entry_config"] = str(derived)
    config["_config_sources"] = [*config["_config_sources"], str(derived)]
    with pytest.raises(ValueError, match="indexed.*replication/study surface"):
        attach_replication_preflight(
            {"artifact_trace": {"inputs": {}}},
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


@pytest.mark.parametrize("drift", ("seed", "root", "sources"))
def test_indexed_replication_binding_rejects_seed_root_or_source_drift(drift: str):
    entry = REPO / "configs/replications/effnet_matched_train_seed20260531.yaml"
    config = load_experiment_config(
        entry,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest-replication-drift"},
    )
    if drift == "seed":
        config["paper_protocol"]["kshot"]["seed"] = 20260611
        config["paper_protocol"]["kshot"]["subset_seed"] = 20260611
    elif drift == "root":
        config["data"]["kshot_subset_root"] = str(
            Path(config["data"]["kshot_subset_root"]).parent.parent / "other_family" / "subsets"
        )
    else:
        config["_config_sources"] = [str(REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml")]
    with pytest.raises(ValueError, match=drift):
        attach_replication_preflight(
            {"artifact_trace": {"inputs": {}}},
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


def test_indexed_replication_entry_must_be_git_tracked(monkeypatch: pytest.MonkeyPatch):
    from ecg_adv_gen.config import replication as replication_module

    entry = REPO / "configs/replications/effnet_matched_train_seed20260531.yaml"
    config = load_experiment_config(
        entry,
        LOCAL_EXAMPLE,
        runtime_context={"run_id": "pytest-replication-untracked"},
    )
    monkeypatch.setattr(replication_module, "_git_tracked", lambda *_args, **_kwargs: False)
    with pytest.raises(ValueError, match="tracked"):
        attach_replication_preflight(
            {"artifact_trace": {"inputs": {}}},
            config,
            repo_root=REPO,
            index_path=INDEX,
        )


def test_replication_rejects_coordinated_noncanonical_class_order(tmp_path: Path):
    root = tmp_path / "subsets"
    base = _write_group(root, center="ningbo", k=2, seed=17)
    reverse = np.asarray(["STTC", "NORM", "MI", "HYP", "CD"])
    for suffix in ("signals.npz", "latent.npz"):
        path = base.with_suffix(f".{suffix}")
        with np.load(path, allow_pickle=False) as payload:
            labels = payload["labels"][:, ::-1]
        _rewrite_npz(path, allow_pickle=False, updates={"labels": labels, "class_names": reverse})
    raw_path = base.with_suffix(".raw1000.npz")
    with np.load(raw_path, allow_pickle=True) as payload:
        raw_labels = payload["labels"][:, ::-1]
        metadata = dict(payload["metadata"].reshape(-1)[0])
    metadata["class_names"] = reverse.tolist()
    _rewrite_npz(
        raw_path,
        allow_pickle=True,
        updates={"labels": raw_labels, "metadata": np.asarray([metadata], dtype=object)},
    )
    ref_path = base.with_suffix(".ref_meta.json")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref["class_names"] = reverse.tolist()
    ref_path.write_text(json.dumps(ref), encoding="utf-8")

    groups = build_replication_k500_groups(
        subset_root=root, centers=["ningbo"], k=2, seed=17, artifact_suffixes=SUFFIXES
    )
    report = verify_replication_k500_groups(
        groups, mapping_version=MAPPING_VERSION, mapping_hash=MAPPING_HASH
    )
    assert report["passed"] is False
    assert any("class order" in item["error"] for item in report["identity_errors"])


def test_replication_rejects_consistent_nonbinary_multihot_labels(tmp_path: Path):
    root = tmp_path / "subsets"
    base = _write_group(root, center="ningbo", k=2, seed=17)
    for suffix, allow_pickle in (("signals.npz", False), ("latent.npz", False), ("raw1000.npz", True)):
        path = base.with_suffix(f".{suffix}")
        with np.load(path, allow_pickle=allow_pickle) as payload:
            labels = payload["labels"].astype(np.float32) * 0.5
        _rewrite_npz(path, allow_pickle=allow_pickle, updates={"labels": labels})
    ref_path = base.with_suffix(".ref_meta.json")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref["label_counts"]["NORM"] = 1
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    trust_path = base.with_suffix(".class_trust.json")
    trust = json.loads(trust_path.read_text(encoding="utf-8"))
    trust["label_counts"]["NORM"] = 1
    trust_path.write_text(json.dumps(trust), encoding="utf-8")

    report = verify_replication_k500_groups(
        build_replication_k500_groups(
            subset_root=root, centers=["ningbo"], k=2, seed=17, artifact_suffixes=SUFFIXES
        ),
        mapping_version=MAPPING_VERSION,
        mapping_hash=MAPPING_HASH,
    )
    assert report["passed"] is False
    assert any("binary multi-hot" in item["error"] for item in report["identity_errors"])


def test_replication_rejects_one_flat_latent_sample(tmp_path: Path):
    root = tmp_path / "subsets"
    base = _write_group(root, center="ningbo", k=2, seed=17)
    path = base.with_suffix(".latent.npz")
    with np.load(path, allow_pickle=False) as payload:
        latents = payload["latents"].copy()
    latents[0] = 0.0
    _rewrite_npz(path, allow_pickle=False, updates={"latents": latents})
    report = verify_replication_k500_groups(
        build_replication_k500_groups(
            subset_root=root, centers=["ningbo"], k=2, seed=17, artifact_suffixes=SUFFIXES
        ),
        mapping_version=MAPPING_VERSION,
        mapping_hash=MAPPING_HASH,
    )
    assert report["passed"] is False
    assert any("latent" in item["error"] and "non-flat" in item["error"] for item in report["identity_errors"])


def test_replication_preflight_rejects_old_id_relabel_protocol(tmp_path: Path):
    root = tmp_path / "subsets"
    base = _write_group(root, center="ningbo", k=2, seed=17)
    meta = json.loads(base.with_suffix(".ref_meta.json").read_text(encoding="utf-8"))
    for key in (
        "fixed_k",
        "protocol",
        "protocol_tag",
        "sampling_policy",
        "raw_v7_nonzero_count",
        "eligible_v7_nonzero_with_latents",
    ):
        meta.pop(key)
    meta.update({"parent": "old_seed.ref_meta.json", "policy": "same K-shot record ids relabeled"})
    base.with_suffix(".ref_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    groups = build_replication_k500_groups(
        subset_root=root,
        centers=["ningbo"],
        k=2,
        seed=17,
        artifact_suffixes=list(SUFFIXES),
    )
    report = verify_replication_k500_groups(
        groups,
        mapping_version=MAPPING_VERSION,
        mapping_hash=MAPPING_HASH,
    )
    assert report["passed"] is False
    assert any(
        "fixed_k" in item["error"] or "deterministic_random" in item["error"]
        for item in report["identity_errors"]
    )


def test_replication_audit_reports_contract_and_execution_readiness_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import ecg_adv_gen.config.replication as replication_module

    monkeypatch.setattr(
        replication_module,
        "verify_replication_validation_report",
        lambda contract, groups: {"passed": True, "path": contract["path"], "errors": []},
    )
    local = _write_local_config(tmp_path)
    for center in CENTERS:
        _write_group(_subset_root(tmp_path, canonical=False), center=center, k=500, seed=20260531)
        _write_group(_subset_root(tmp_path, canonical=False), center=center, k=500, seed=20260601)
    _, report_sha = _write_validation_report(
        _subset_root(tmp_path, canonical=False),
        seeds=(20260531, 20260601, 20260611),
    )

    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    for replicate in index["replication_surfaces"][0]["replicates"]:
        replicate["k500_input_status"] = (
            "present_in_test_fixture"
            if int(replicate["seed"]) in {20260531, 20260601}
            else "materialization_required"
        )
    index["replication_surfaces"][0]["validation_report"]["sha256"] = report_sha
    fixture_index = tmp_path / "active_scripts_fixture.yaml"
    fixture_index.write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")
    report = audit_replication_surfaces(
        repo_root=REPO,
        index_path=fixture_index,
        local_config_path=local,
    )
    assert report["contract_passed"] is True
    assert report["execution_ready"] is False
    assert report["surface_count"] == 2
    surface = next(
        item for item in report["surfaces"] if item["name"] == "effnet_matched_three_seed"
    )
    assert surface["path_isolation_passed"] is True
    seeds = {item["seed"]: item for item in surface["replicates"]}
    assert seeds[20260531]["execution_ready"] is True
    assert seeds[20260601]["execution_ready"] is True
    assert seeds[20260611]["execution_ready"] is False
    assert seeds[20260611]["observed_k500_input_status"] == "materialization_required"
    for seed_report in seeds.values():
        assert len(seed_report["stages"]) == 4
        assert all(stage["command_count"] == 20 for stage in seed_report["stages"])
        assert all(
            stage["command_input_contract"]["passed"] is True
            and stage["command_input_contract"]["per_command_passed"] is True
            for stage in seed_report["stages"]
        )
        assert seed_report["producer_consumer_paths_match"] is True


def test_replication_audit_reports_hard_coded_status_drift(tmp_path: Path):
    local = _write_local_config(tmp_path)
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    replicate = index["replication_surfaces"][0]["replicates"][-1]
    replicate["k500_input_status"] = "present_as_of_2026-07-11"
    drifted_index = tmp_path / "active_scripts.yaml"
    drifted_index.write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")
    report = audit_replication_surfaces(
        repo_root=REPO,
        index_path=drifted_index,
        local_config_path=local,
    )
    assert report["contract_passed"] is False
    seed_report = report["surfaces"][0]["replicates"][-1]
    assert seed_report["status_consistent"] is False
    assert any("declared" in error and "observed" in error for error in seed_report["status_errors"])


def test_launcher_dry_run_exposes_replication_preflight_without_touching_golden(tmp_path: Path):
    local = _write_local_config(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_experiment.py"),
            "--config",
            str(REPO / "configs/replications/effnet_matched_train_seed20260531.yaml"),
            "--local-config",
            str(local),
            "--run-id",
            "shared_run_id",
            "--dry-run",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads(proc.stdout.split("\n\n# Managed commands", 1)[0])
    replication = manifest["artifact_trace"]["replication_preflight"]
    assert replication["seed"] == 20260531
    assert len(manifest["artifact_trace"]["inputs"]["replication_k500_groups"]) == 4


def test_workspace_handoff_contract_exposes_replication_audit(tmp_path: Path):
    from ecg_adv_gen.config import audit_active_managed_configs
    from scripts.agent.audit_agent_workspace import build_handoff_contract

    local = _write_local_config(tmp_path)
    active = audit_active_managed_configs(
        repo_root=REPO,
        index_path=INDEX,
        local_config_path=local,
        require_existing_inputs=False,
    )
    handoff = build_handoff_contract({"active_scripts": active})
    assert handoff["replication_surfaces"] == active["replication_surfaces"]
    assert handoff["replication_surfaces"]["contract_passed"] is False
    assert handoff["replication_surfaces"]["surfaces"][0]["replicates"]
    assert handoff["study_surfaces"] == active["study_surfaces"]
    assert handoff["study_surfaces"]["contract_passed"] is True
    assert handoff["study_surfaces"]["surface_count"] == 2
    assert handoff["study_surfaces"]["entry_count"] == 9


def test_study_surface_audit_is_part_of_active_total_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ecg_adv_gen.config import audit as audit_module

    monkeypatch.setattr(
        audit_module,
        "audit_replication_surfaces",
        lambda **_kwargs: {
            "contract_passed": True,
            "execution_ready": True,
            "failed_count": 0,
            "surfaces": [],
        },
    )
    report = audit_module.audit_active_managed_configs(
        repo_root=REPO,
        index_path=INDEX,
        local_config_path=LOCAL_EXAMPLE,
        require_existing_inputs=False,
    )
    assert report["study_surfaces"]["contract_passed"] is True
    assert report["study_surfaces"]["entry_count"] == 9

    monkeypatch.setattr(
        audit_module,
        "audit_study_surfaces",
        lambda **_kwargs: {
            "contract_passed": False,
            "failed_count": 1,
            "surface_count": 1,
            "entry_count": 1,
            "surfaces": [],
        },
        raising=False,
    )
    failed = audit_module.audit_active_managed_configs(
        repo_root=REPO,
        index_path=INDEX,
        local_config_path=LOCAL_EXAMPLE,
        require_existing_inputs=False,
    )
    assert failed["study_surfaces"]["contract_passed"] is False
    assert failed["passed"] is False


def test_tracked_yaml_attention_fails_the_active_audit_total_verdict(
    monkeypatch: pytest.MonkeyPatch,
):
    from ecg_adv_gen.config import audit as audit_module

    original = audit_module._summarize_config_git_status

    def attention_summary(items, *, tracked_yaml_required):
        summary = original(items, tracked_yaml_required=tracked_yaml_required)
        return {
            **summary,
            "requires_attention": True,
            "untracked_count": 1,
            "untracked_paths": ["configs/replications/unindexed.yaml"],
        }

    monkeypatch.setattr(audit_module, "_summarize_config_git_status", attention_summary)
    monkeypatch.setattr(
        audit_module,
        "audit_replication_surfaces",
        lambda **_kwargs: {
            "contract_passed": True,
            "execution_ready": True,
            "failed_count": 0,
            "surfaces": [],
        },
    )
    report = audit_module.audit_active_managed_configs(
        repo_root=REPO,
        index_path=INDEX,
        local_config_path=LOCAL_EXAMPLE,
        require_existing_inputs=False,
    )
    assert report["launch_surface_policy"]["tracked_yaml_required"] is True
    assert report["config_git_summary"]["requires_attention"] is True
    assert report["passed"] is False
    assert report["audit_failure_count"] >= 1
