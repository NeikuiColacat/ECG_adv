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
