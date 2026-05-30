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
                "path": "${ECG_ADV_DATA_ROOT}/present.txt",
                "kind": "text",
                "required": True,
                "package": True,
            },
            {
                "id": "missing_required",
                "path": "${ECG_ADV_DATA_ROOT}/missing-required.txt",
                "kind": "text",
                "required": True,
                "package": True,
            },
            {
                "id": "missing_optional",
                "path": "${ECG_ADV_DATA_ROOT}/missing-optional.txt",
                "kind": "text",
                "required": False,
                "package": True,
            },
            {
                "id": "missing_nonpackage",
                "path": "${ECG_ADV_DATA_ROOT}/missing-nonpackage.txt",
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
    assert copied[0]["source_path"] == "${ECG_ADV_DATA_ROOT}/present.txt"
    assert [item["id"] for item in missing] == ["missing_required", "missing_optional"]
    assert missing[0]["source_path"] == "${ECG_ADV_DATA_ROOT}/missing-required.txt"
    assert missing[0]["required"] is True
    assert missing[1]["required"] is False


def test_write_missing_markdown_lists_required_artifacts(tmp_path):
    missing = [
        {
            "id": "table68_pool",
            "kind": "generated_pool",
            "required": True,
            "package": True,
            "source_path": "${ECG_ADV_GRAD_ROOT}/missing/table68.npz",
            "used_by": ["Table 6.8"],
            "description": "Table 6.8 input pool.",
            "regenerate_command": "bash scripts/final_round/run_thesis_reproduction.sh ablation_6_8",
        }
    ]
    out_path = tmp_path / "missing_artifacts.md"

    pkg.write_missing_markdown(out_path, missing)

    text = out_path.read_text(encoding="utf-8")
    assert "# Missing Thesis Archive Artifacts" in text
    assert "`table68_pool`" in text
    assert "`${ECG_ADV_GRAD_ROOT}/missing/table68.npz`" in text
    assert "Table 6.8" in text
    assert "bash scripts/final_round/run_thesis_reproduction.sh ablation_6_8" in text
