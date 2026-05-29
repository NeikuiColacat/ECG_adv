from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "agent" / "audit_retrospective_inputs.py"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    write(root / "AGENTS.md", "# Agent notes\n")
    write(root / ".codex/skills/example/SKILL.md", "---\nname: example\ndescription: test\n---\n")
    write(root / "docs/codex-handoffs/2026-05-29-test.md", "# Handoff\n")
    write(root / "docs/pipelines/refactor_handoff.md", "# Refactor\n")
    write(root / "docs/labeling/pn2021_review.md", "# Label Review\n")
    write(root / "configs/label_mappings/pn2021_review.jsonl", "{}\n")
    write(root / "docs/tmp_md/run_summary.md", "# Summary\n")
    write(root / "configs/active_scripts.yaml", "version: 1\n")
    write(root / "configs/active_evidence_registry.yaml", "schema_version: 1\n")
    return root


def test_audit_retrospective_inputs_json(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(root), "--format", "json"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    records = payload["records"]
    by_path = {record["path"]: record for record in records}

    assert "AGENTS.md" in by_path
    assert ".codex/skills/example/SKILL.md" in by_path
    assert "docs/codex-handoffs/2026-05-29-test.md" in by_path
    assert "docs/labeling/pn2021_review.md" in by_path
    assert "configs/label_mappings/pn2021_review.jsonl" in by_path
    assert "configs/active_scripts.yaml" in by_path
    assert "configs/active_evidence_registry.yaml" in by_path
    assert by_path["AGENTS.md"]["category"] == "core_memory"
    assert by_path["AGENTS.md"]["sha256"]


def test_audit_retrospective_inputs_markdown(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(root), "--format", "markdown"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "# Retrospective Input Inventory" in result.stdout
    assert "`AGENTS.md`" in result.stdout
    assert "| category | path | size | lines | note |" in result.stdout


def test_explicit_session_file_must_stay_under_home(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    outside = tmp_path / "outside-session.jsonl"
    outside.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(root),
            "--session-file",
            str(outside),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "outside home" in result.stderr or "outside home" in result.stdout
