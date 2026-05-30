import json
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


def test_reproduction_runner_exposes_restore_artifacts_stage():
    script = (REPO_ROOT / "scripts/final_round/run_thesis_reproduction.sh").read_text(encoding="utf-8")

    assert "restore_artifacts" in script
    assert "restore_migrate_artifacts.py" in script


def test_reproduction_runner_separates_archive_and_full_preflight():
    script = (REPO_ROOT / "scripts/final_round/run_thesis_reproduction.sh").read_text(encoding="utf-8")

    assert "--scope archive" in script
    assert "--scope full" in script
    assert "preflight_full" in script
