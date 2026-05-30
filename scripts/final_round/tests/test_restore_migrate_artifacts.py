import hashlib
import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.final_round import restore_migrate_artifacts as restore


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _write_checksums(migrate_dir: Path, *paths: Path) -> None:
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (migrate_dir / "SHA256SUMS_ecg_grad_artifacts_20260507.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def test_restore_archives_extracts_checksum_verified_tar(tmp_path):
    migrate_dir = tmp_path / "migrate_files"
    out_dir = tmp_path / "data"
    migrate_dir.mkdir()
    archive = migrate_dir / "ecg_grad_min_runtime_artifacts_20260507.tar.gz"
    _write_tar(archive, {"streamlit_ecg_demo/models/model.txt": b"model\n"})
    _write_checksums(migrate_dir, archive)

    restored = restore.restore_archives(migrate_dir=migrate_dir, out_dir=out_dir)

    assert restored == [
        {
            "archive": str(archive),
            "members": 1,
            "dry_run": False,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
    ]
    assert (out_dir / "streamlit_ecg_demo/models/model.txt").read_text(encoding="utf-8") == "model\n"


def test_restore_archives_dry_run_does_not_extract(tmp_path):
    migrate_dir = tmp_path / "migrate_files"
    out_dir = tmp_path / "data"
    migrate_dir.mkdir()
    archive = migrate_dir / "ecg_grad_repro_addons_20260507.tar.gz"
    _write_tar(archive, {"graduate_project/run/train_result.json": b"{}\n"})
    _write_checksums(migrate_dir, archive)

    restored = restore.restore_archives(migrate_dir=migrate_dir, out_dir=out_dir, dry_run=True)

    assert restored[0]["members"] == 1
    assert restored[0]["dry_run"] is True
    assert not (out_dir / "graduate_project/run/train_result.json").exists()


def test_restore_archives_supports_external_repro_bundle_pattern(tmp_path):
    bundle_dir = tmp_path / "bundle"
    out_dir = tmp_path / "data"
    bundle_dir.mkdir()
    archive = bundle_dir / "ecg_grad_repro_no_pn2021_20260523.tar.gz"
    _write_tar(archive, {"graduate_project/run/best_model.pt": b"weights\n"})
    _write_checksums(bundle_dir, archive)

    restored = restore.restore_archives(
        migrate_dir=bundle_dir,
        out_dir=out_dir,
        archive_patterns=("ecg_grad_repro_no_pn2021_*.tar.gz",),
    )

    assert restored[0]["archive"] == str(archive)
    assert (out_dir / "graduate_project/run/best_model.pt").read_text(encoding="utf-8") == "weights\n"


def test_restore_archives_rejects_path_traversal(tmp_path):
    migrate_dir = tmp_path / "migrate_files"
    out_dir = tmp_path / "data"
    migrate_dir.mkdir()
    archive = migrate_dir / "ecg_grad_min_runtime_artifacts_20260507.tar.gz"
    _write_tar(archive, {"../escape.txt": b"bad\n"})
    _write_checksums(migrate_dir, archive)

    with pytest.raises(ValueError, match="unsafe tar member"):
        restore.restore_archives(migrate_dir=migrate_dir, out_dir=out_dir)
