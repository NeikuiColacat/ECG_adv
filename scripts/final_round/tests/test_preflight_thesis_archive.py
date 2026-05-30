import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.final_round import preflight_thesis_archive as preflight


def test_archive_scope_checks_only_archive_required_package_items(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    archive_file = data_root / "archive-required.txt"
    archive_file.write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(preflight, "DATA_ROOT", data_root)

    manifest = {
        "artifacts": [
            {
                "id": "archive_required",
                "path": "${ECG_ADV_DATA_ROOT}/archive-required.txt",
                "required": True,
                "package": True,
            },
            {
                "id": "full_rerun_only",
                "path": "${ECG_ADV_DATA_ROOT}/missing-rerun.txt",
                "required": True,
                "package": True,
                "archive_required": False,
            },
            {
                "id": "external_dataset",
                "path": "${ECG_ADV_DATA_ROOT}/missing-dataset.txt",
                "required": True,
                "package": False,
            },
        ]
    }

    statuses = preflight.check_manifest(manifest, verify_sha=False, scope="archive", include_optional=False)

    assert statuses == {"ok": 1, "missing-required": 0, "missing-optional": 0, "bad": 0, "skipped": 2}


def test_full_scope_checks_required_rerun_and_nonpackage_items(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    archive_file = data_root / "archive-required.txt"
    archive_file.write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(preflight, "DATA_ROOT", data_root)

    manifest = {
        "artifacts": [
            {
                "id": "archive_required",
                "path": "${ECG_ADV_DATA_ROOT}/archive-required.txt",
                "required": True,
                "package": True,
            },
            {
                "id": "full_rerun_only",
                "path": "${ECG_ADV_DATA_ROOT}/missing-rerun.txt",
                "required": True,
                "package": True,
                "archive_required": False,
            },
            {
                "id": "external_dataset",
                "path": "${ECG_ADV_DATA_ROOT}/missing-dataset.txt",
                "required": True,
                "package": False,
            },
        ]
    }

    statuses = preflight.check_manifest(manifest, verify_sha=False, scope="full", include_optional=False)

    assert statuses == {"ok": 1, "missing-required": 2, "missing-optional": 0, "bad": 0, "skipped": 0}


def test_write_missing_reports_records_full_scope_gaps(tmp_path):
    records = [
        {
            "status": "missing-required",
            "id": "full_rerun_init",
            "path": "${ECG_ADV_DATA_ROOT}/missing.pt",
            "resolved_path": "/tmp/data/missing.pt",
            "kind": "checkpoint",
            "required": True,
            "package": True,
            "archive_required": False,
            "used_by": ["Table 6.6"],
            "description": "Synthetic-pretrain init checkpoint.",
            "regenerate_command": "bash scripts/final_round/run_thesis_reproduction.sh low_sample_rerun",
        },
        {
            "status": "ok",
            "id": "present",
            "path": "${ECG_ADV_DATA_ROOT}/present.pt",
            "resolved_path": "/tmp/data/present.pt",
            "kind": "checkpoint",
            "required": True,
            "package": True,
            "archive_required": True,
        },
    ]

    json_path = tmp_path / "missing.json"
    md_path = tmp_path / "missing.md"
    preflight.write_missing_reports(records, json_path=json_path, md_path=md_path, scope="full")

    assert '"full_rerun_init"' in json_path.read_text(encoding="utf-8")
    assert '"present"' not in json_path.read_text(encoding="utf-8")
    md = md_path.read_text(encoding="utf-8")
    assert "# Missing Full Preflight Artifacts" in md
    assert "`full_rerun_init`" in md
    assert "Table 6.6" in md
    assert "bash scripts/final_round/run_thesis_reproduction.sh low_sample_rerun" in md
