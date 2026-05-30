import json
import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_feature_distribution_artifacts_are_optional_extra():
    manifest = json.loads((REPO_ROOT / "docs/artifact_manifest.json").read_text(encoding="utf-8"))
    entries = {
        item["id"]: item
        for item in manifest["artifacts"]
        if item["id"].startswith("feature_distribution_")
    }

    assert entries
    assert all(item["required"] is False for item in entries.values())


def test_all_stage_does_not_run_optional_feature_distribution():
    script = (REPO_ROOT / "scripts/final_round/run_thesis_reproduction.sh").read_text(encoding="utf-8")
    all_block = script.split("all)", 1)[1].split(";;", 1)[0]

    assert "stage_feature_dist" not in all_block


def test_evidence_pack_manifest_does_not_embed_host_absolute_paths():
    text = (REPO_ROOT / "artifacts/evidence_pack/manifest.json").read_text(encoding="utf-8")

    forbidden_prefixes = [
        "/root/autodl-tmp",
        "/home/neiku/autodl-tmp",
        "/root/ECG_adv",
        "/home/neiku/graduate_project",
    ]
    for prefix in forbidden_prefixes:
        assert prefix not in text


def test_table68_runner_writes_summary_outputs():
    script = (REPO_ROOT / "scripts/final_round/run_table_6_8_ablations.sh").read_text(encoding="utf-8")

    assert "summarize_table_6_8_results.py" in script
    assert "--run_root \"$OUT_ROOT\"" in script


def test_archived_table68_summary_does_not_embed_host_absolute_paths():
    text = (REPO_ROOT / "artifacts/evidence_pack/tables/low_sample_ablation_results.csv").read_text(
        encoding="utf-8"
    )

    assert "/root/autodl-tmp" not in text
    assert "/home/neiku/autodl-tmp" not in text


def test_mainline_sources_do_not_import_top_level_adversarial():
    mainline_roots = [
        REPO_ROOT / "apps",
        REPO_ROOT / "scripts",
        REPO_ROOT / "methods",
        REPO_ROOT / "util",
    ]
    offenders = []
    for root in mainline_roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts or "legacy" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports_adversarial = any(
                (
                    isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and node.module.split(".")[0] == "adversarial"
                )
                or (
                    isinstance(node, ast.Import)
                    and any(alias.name.split(".")[0] == "adversarial" for alias in node.names)
                )
                for node in ast.walk(tree)
            )
            if imports_adversarial:
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []


def test_tensorrt_dependency_is_optional_deploy_extra():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    deps_block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    deploy_block = pyproject.split("[project.optional-dependencies]", 1)[1]
    assert "tensorrt-cu12" not in deps_block
    assert "deploy = [" in deploy_block
    assert "tensorrt-cu12" in deploy_block


def test_table63_author_repro_has_small_metric_summary():
    manifest = json.loads((REPO_ROOT / "docs/thesis_repro_manifest.json").read_text(encoding="utf-8"))
    table63 = next(item for item in manifest["experiments"] if item["paper_item"].startswith("Table 6.3"))

    assert "artifacts/evidence_pack/raw/ecgtwin_author_repro_summary.json" in table63["artifacts"]
    assert table63["reported_result"]["ibe_best_validation_score"] == 0.7189
    assert table63["reported_result"]["dit_best_validation_loss"] == 15.0806


def test_reproduction_runner_exposes_restore_artifacts_stage():
    script = (REPO_ROOT / "scripts/final_round/run_thesis_reproduction.sh").read_text(encoding="utf-8")

    assert "restore_artifacts" in script
    assert "restore_migrate_artifacts.py" in script
    assert "ecg_grad_repro_no_pn2021_*.tar.gz" in script


def test_reproduction_runner_separates_archive_and_full_preflight():
    script = (REPO_ROOT / "scripts/final_round/run_thesis_reproduction.sh").read_text(encoding="utf-8")

    assert "--scope archive" in script
    assert "--scope full" in script
    assert "preflight_full" in script
    assert "full_preflight_missing_artifacts.json" in script
    assert "full_preflight_missing_artifacts.md" in script


def test_reproduction_runner_exposes_synthetic_pretrain_init_stage():
    script = (REPO_ROOT / "scripts/final_round/run_thesis_reproduction.sh").read_text(encoding="utf-8")

    assert "synthetic_pretrain_init" in script
    assert "run_synthetic_pretrain_init_checkpoints.sh" in script


def test_synthetic_pretrain_init_runner_rebuilds_both_missing_init_checkpoints():
    script = (REPO_ROOT / "scripts/final_round/run_synthetic_pretrain_init_checkpoints.sh").read_text(
        encoding="utf-8"
    )

    assert "self_distill_v2_e21_v46_no_token_hardlabel_r10_seed8042_auroc" in script
    assert "self_distill_v2_e18_v46_class_oracle_hardlabel_r10_seed42_auroc" in script
    assert "--synthetic_only" in script
    assert "NO_TOKEN_PRETRAIN_SYNTH_NPZ" in script
    assert "CENTER_TOKEN_PRETRAIN_SYNTH_NPZ" in script
