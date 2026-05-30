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
