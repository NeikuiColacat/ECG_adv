"""Contracts for replication path isolation, K500 preflight, and audit reporting."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from ecg_adv_gen.config import (
    audit_replication_command_grid,
    audit_replication_path_isolation,
    audit_replication_producer_consumers,
    audit_replication_surfaces,
    build_replication_k500_groups,
    verify_replication_k500_groups,
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
    class_names = np.asarray(["CD", "HYP", "MI", "NORM", "STTC"])
    shared = {
        "labels": labels,
        "record_ids": record_ids,
        "center_name": np.asarray(center),
        "class_names": class_names,
        "mapping_version": np.asarray(MAPPING_VERSION),
        "mapping_hash": np.asarray(MAPPING_HASH),
    }
    np.savez(base.with_suffix(".signals.npz"), signals=np.zeros((k, 2, 1), dtype=np.float32), **shared)
    np.savez(base.with_suffix(".latent.npz"), latents=np.zeros((k, 1, 2), dtype=np.float32), **shared)
    np.savez(
        base.with_suffix(".raw1000.npz"),
        signals=np.zeros((k, 2, 1), dtype=np.float32),
        labels=labels,
        record_ids=record_ids,
        metadata=np.asarray(
            [
                {
                    "center": center,
                    "source_ref_meta_json": str(base.with_suffix(".ref_meta.json")),
                    "n_records": k,
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


def _surface() -> dict[str, Any]:
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    return index["replication_surfaces"][0]


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


def test_replication_audit_reports_contract_and_execution_readiness_separately(tmp_path: Path):
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
    assert report["surface_count"] == 1
    surface = report["surfaces"][0]
    assert surface["path_isolation_passed"] is True
    seeds = {item["seed"]: item for item in surface["replicates"]}
    assert seeds[20260531]["execution_ready"] is True
    assert seeds[20260601]["execution_ready"] is True
    assert seeds[20260611]["execution_ready"] is False
    assert seeds[20260611]["observed_k500_input_status"] == "materialization_required"
    for seed_report in seeds.values():
        assert len(seed_report["stages"]) == 4
        assert all(stage["command_count"] == 20 for stage in seed_report["stages"])
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
