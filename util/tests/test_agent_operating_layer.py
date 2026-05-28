"""CPU-only tests for the agent operating layer registry and bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml
import pytest

from ecg_adv_gen.evidence import (
    audit_active_evidence_registry,
    build_comparison_bundle,
    build_legacy_vae_lhat_manifest,
    EvidenceAuditError,
    load_evidence_registry,
)
from ecg_adv_gen.evidence.registry import _audit_manifest
from ecg_adv_gen.reporting.metrics_export import METRICS_FIELDNAMES


REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "configs" / "active_evidence_registry.yaml"
LOCAL_EXAMPLE = REPO / "configs" / "local" / "linbinhao_server.example.yaml"


def _walk_strings(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_strings(value)
    elif isinstance(obj, str):
        yield obj


def _write_metrics(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            full = {key: "" for key in METRICS_FIELDNAMES}
            full.update(row)
            writer.writerow(full)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_rows(run_id: str, auroc: float, auprc: float) -> list[dict[str, str]]:
    rows = []
    for center in ["ningbo", "georgia"]:
        for metric, value in [("macro_auroc", auroc), ("macro_auprc", auprc)]:
            rows.append(
                {
                    "run_id": run_id,
                    "source_file": f"/tmp/{run_id}_{center}.json",
                    "artifact_type": "eval_result",
                    "dataset": "pn2021",
                    "view": "pn2021_all_zero_kept_refexcluded",
                    "canonical_view": "pn2021_all_zero_kept_refexcluded",
                    "scope": "center",
                    "center": center,
                    "metric": metric,
                    "value": str(value),
                    "mapping_version": "vtest",
                    "mapping_hash": "htest",
                    "class_order": "CD|HYP|MI|NORM|STTC",
                }
            )
        for metric, value in [("drop_all_zero_macro_auroc", auroc + 0.1), ("drop_all_zero_macro_auprc", auprc + 0.1)]:
            rows.append(
                {
                    "run_id": run_id,
                    "source_file": f"/tmp/{run_id}_{center}.json",
                    "artifact_type": "eval_result",
                    "dataset": "pn2021",
                    "view": "pn2021_drop_all_zero_refexcluded",
                    "canonical_view": "pn2021_drop_all_zero_refexcluded",
                    "scope": "center",
                    "center": center,
                    "metric": metric,
                    "value": str(value),
                    "mapping_version": "vtest",
                    "mapping_hash": "htest",
                    "class_order": "CD|HYP|MI|NORM|STTC",
                }
            )
    return rows


def test_active_evidence_registry_is_tracked_and_path_safe():
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    host_paths = [
        value for value in _walk_strings(raw)
        if value.startswith("/home/") or value.startswith("/root/")
    ]
    assert host_paths == []
    assert raw["active_claims"][0]["protocol"]["mapping_version"] == "v7_super5_sjr_rgq_review_20260528"
    assert raw["active_claims"][0]["comparison_bundle"]["status"] == "built"


def test_registry_resolves_with_local_config():
    resolved = load_evidence_registry(REGISTRY, LOCAL_EXAMPLE)
    claim = resolved["active_claims"][0]
    out_dir = claim["comparison_bundle"]["output_dir"]
    assert out_dir.startswith("/home/linbinhao/")
    assert claim["protocol"]["kshot"]["ref_meta_files"]["ningbo"].endswith(".ref_meta.json")


def test_agent_workspace_audit_cpu_only_without_artifact_scan():
    report = audit_active_evidence_registry(
        repo_root=REPO,
        registry_path=REGISTRY,
        local_config_path=LOCAL_EXAMPLE,
        require_existing_artifacts=False,
        check_git=False,
    )
    assert report["passed"] is True
    assert report["claim_count"] == 1


def test_trusted_mainline_missing_manifest_is_an_error():
    issues = []
    summary = _audit_manifest(
        "vae_lhat",
        {"status": "trusted", "manifest": ""},
        {"claim_id": "claim", "status": "trusted", "paper_use": "mainline"},
        issues,
    )
    assert summary == {"declared": False, "path": ""}
    assert issues[0]["level"] == "error"
    assert issues[0]["code"] == "manifest_not_declared"


def test_legacy_backfilled_manifest_contract(tmp_path: Path):
    artifact_records = {}
    for name in [
        "launch_config.json",
        "train_result.json",
        "early_stop_info.json",
        "eval_result_v7_exclrefs_crop1000.json",
        "best_model.pt",
    ]:
        path = tmp_path / name
        path.write_bytes(f"{name}\n".encode("utf-8"))
        artifact_records[name] = {
            "exists": True,
            "path": str(path),
            "sha256": _sha256(path),
        }
    manifest_path = tmp_path / "run_manifest.backfilled.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_kind": "legacy_backfilled_manifest",
                "status": "succeeded",
                "claim_id": "claim",
                "method_key": "vae_lhat",
                "run_id": "run",
                "protocol": {
                    "mapping_version": "vtest",
                    "mapping_hash": "htest",
                    "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
                    "target_centers": ["ningbo"],
                },
                "centers": [
                    {
                        "center": "ningbo",
                        "final_eval": {
                            "mapping_version": "vtest",
                            "mapping_hash": "htest",
                        },
                        "artifacts": artifact_records,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    issues = []
    summary = _audit_manifest(
        "vae_lhat",
        {
            "status": "trusted",
            "traceability": "legacy_backfilled_manifest",
            "run_id": "run",
            "manifest": str(manifest_path),
        },
        {
            "claim_id": "claim",
            "status": "trusted",
            "paper_use": "mainline",
            "protocol": {
                "mapping_version": "vtest",
                "mapping_hash": "htest",
                "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
                "target_centers": ["ningbo"],
            },
        },
        issues,
    )
    assert summary["manifest_kind"] == "legacy_backfilled_manifest"
    assert issues == []


def test_build_legacy_vae_lhat_manifest_from_fake_run(tmp_path: Path):
    output_root = tmp_path / "out"
    run_root = output_root / "effnet_vae_lhat_k500_v7_sjr_rgq" / "run"
    center_dir = run_root / "ningbo_run"
    center_dir.mkdir(parents=True)
    launch = {
        "args": {"center": "ningbo", "seed": 7, "init_ckpt": "/tmp/init.pt"},
        "train_cmd": ["python", "train.py"],
        "eval_cmd": ["python", "eval.py"],
    }
    train_result = {
        "args": {
            "ref_meta_json": "/tmp/ref.json",
            "target_real_npz": "/tmp/signals.npz",
            "synth_npz": "/tmp/latent.npz",
        }
    }
    early_stop = {
        "es_metric": "target_macro_auprc",
        "best_epoch": 2,
        "best_metric": 0.7,
        "stopped_epoch": 4,
        "early_stopped": False,
    }
    eval_result = {
        "pn2021": {
            "avg_macro_auroc": 0.8,
            "avg_macro_auprc": 0.6,
            "avg_drop_all_zero_macro_auroc": 0.9,
            "avg_drop_all_zero_macro_auprc": 0.7,
        },
        "label_mapping": {
            "pn2021_super5": {
                "mapping_version": "vtest",
                "mapping_hash": "htest",
            }
        },
    }
    for name, payload in [
        ("launch_config.json", launch),
        ("train_result.json", train_result),
        ("early_stop_info.json", early_stop),
        ("eval_result_v7_exclrefs_crop1000.json", eval_result),
    ]:
        (center_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    (center_dir / "best_model.pt").write_bytes(b"fake checkpoint")
    for extra in ["metrics_long.csv", "artifact_manifest.json", "paper_table.csv"]:
        (run_root / extra).write_text("x\n", encoding="utf-8")

    local_config = tmp_path / "local.yaml"
    local_config.write_text(f"paths:\n  output_root: {output_root}\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        f"""
schema_version: 1
active_claims:
  - claim_id: claim
    protocol:
      mapping_version: vtest
      mapping_hash: htest
      class_order: [CD, HYP, MI, NORM, STTC]
      target_centers: [ningbo]
      kshot: {{k: 500, seed: 7, ref_excluded: true}}
      selection: {{policy: fake}}
    methods:
      vae_lhat:
        status: trusted
        traceability: legacy_backfilled_manifest
        method_family: vae_latent_hull_online_at
        run_id: run
        config: configs/experiments/fake.yaml
        run_root: ${{paths.output_root}}/effnet_vae_lhat_k500_v7_sjr_rgq/run
        manifest: ${{paths.output_root}}/effnet_vae_lhat_k500_v7_sjr_rgq/run/run_manifest.backfilled.json
        metrics_long: {run_root / 'metrics_long.csv'}
        artifact_manifest: {run_root / 'artifact_manifest.json'}
        paper_tables:
          pn2021_all_zero_kept_refexcluded: {run_root / 'paper_table.csv'}
""",
        encoding="utf-8",
    )
    manifest = build_legacy_vae_lhat_manifest(
        repo_root=REPO,
        registry_path=registry,
        local_config_path=local_config,
        claim_id="claim",
        method_key="vae_lhat",
        write_boundary=tmp_path,
    )
    assert manifest["manifest_kind"] == "legacy_backfilled_manifest"
    assert manifest["centers"][0]["center"] == "ningbo"
    assert manifest["centers"][0]["artifacts"]["best_model.pt"]["sha256"]
    assert manifest["centers"][0]["final_eval"]["mapping_version"] == "vtest"
    assert (run_root / "run_manifest.backfilled.json").exists()


def test_backfill_manifest_respects_write_boundary(tmp_path: Path):
    local_config = tmp_path / "local.yaml"
    local_config.write_text(f"paths:\n  output_root: {tmp_path / 'out'}\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
schema_version: 1
active_claims:
  - claim_id: claim
    protocol:
      mapping_version: vtest
      mapping_hash: htest
      class_order: [CD, HYP, MI, NORM, STTC]
      target_centers: [ningbo]
      kshot: {k: 500, seed: 7, ref_excluded: true}
      selection: {policy: fake}
    methods:
      vae_lhat:
        status: trusted
        traceability: legacy_backfilled_manifest
        method_family: vae_latent_hull_online_at
        run_id: run
        config: configs/experiments/fake.yaml
        run_root: ${paths.output_root}/missing
        manifest: ${paths.output_root}/missing/run_manifest.backfilled.json
        metrics_long: ${paths.output_root}/metrics.csv
        artifact_manifest: ${paths.output_root}/artifact_manifest.json
        paper_tables: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(EvidenceAuditError, match="outside write boundary"):
        build_legacy_vae_lhat_manifest(
            repo_root=REPO,
            registry_path=registry,
            local_config_path=local_config,
            claim_id="claim",
            method_key="vae_lhat",
            write_boundary=tmp_path / "other",
        )


def test_build_comparison_bundle_from_registry(tmp_path: Path):
    direct_metrics = _write_metrics(tmp_path / "direct.csv", _metric_rows("direct", 0.8, 0.5))
    vae_metrics = _write_metrics(tmp_path / "vae.csv", _metric_rows("vae", 0.83, 0.56))
    local_config = tmp_path / "local.yaml"
    local_config.write_text(f"paths:\n  output_root: {tmp_path / 'out'}\n", encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        f"""
schema_version: 1
active_claims:
  - claim_id: synthetic_claim
    protocol:
      mapping_version: vtest
      mapping_hash: htest
      class_order: [CD, HYP, MI, NORM, STTC]
      target_centers: [ningbo, georgia]
      evaluation_views:
        - pn2021_all_zero_kept_refexcluded
        - pn2021_drop_all_zero_refexcluded
    methods:
      direct:
        metrics_long: {direct_metrics}
      vae:
        metrics_long: {vae_metrics}
    comparison_bundle:
      comparison_id: synthetic_comparison
      baseline_method: direct
      candidate_method: vae
      baseline_run_id: direct
      candidate_run_id: vae
      output_dir: ${{paths.output_root}}/synthetic_comparison
""",
        encoding="utf-8",
    )

    manifest = build_comparison_bundle(
        registry_path=registry,
        local_config_path=local_config,
        force=True,
    )

    assert Path(manifest["artifacts"]["comparison_delta"]).exists()
    mean = [
        row for row in manifest["delta_rows"]
        if row["view"] == "pn2021_all_zero_kept_refexcluded"
        and row["scope"] == "center_mean"
    ][0]
    assert abs(mean["delta_macro_auroc_pp"] - 3.0) < 1e-9
    assert abs(mean["delta_macro_auprc_pp"] - 6.0) < 1e-9
