import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.final_round import package_thesis_artifacts as pkg


def test_copy_artifacts_reports_missing_when_skip_missing(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    present = data_root / "present.txt"
    present.write_text("archive me\n", encoding="utf-8")
    monkeypatch.setattr(pkg, "DATA_ROOT", data_root)

    manifest = {
        "artifacts": [
            {
                "id": "present_required",
                "path": str(present),
                "kind": "text",
                "required": True,
                "package": True,
            },
            {
                "id": "missing_required",
                "path": str(data_root / "missing-required.txt"),
                "kind": "text",
                "required": True,
                "package": True,
            },
            {
                "id": "missing_optional",
                "path": str(data_root / "missing-optional.txt"),
                "kind": "text",
                "required": False,
                "package": True,
            },
            {
                "id": "missing_nonpackage",
                "path": str(data_root / "missing-nonpackage.txt"),
                "kind": "text",
                "required": True,
                "package": False,
            },
        ]
    }

    copied, missing = pkg.copy_artifacts(
        manifest,
        tmp_path / "out",
        include_optional=True,
        include_nonpackage=False,
        skip_missing=True,
    )

    assert [item["id"] for item in copied] == ["present_required"]
    assert [item["id"] for item in missing] == ["missing_required", "missing_optional"]
    assert missing[0]["required"] is True
    assert missing[1]["required"] is False
