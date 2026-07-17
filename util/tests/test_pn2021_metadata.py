from __future__ import annotations

from pathlib import Path

from data_preprocess.pn2021_metadata import (
    parse_header_snomeds,
    parse_pn2021_header_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_parse_header_snomeds_accepts_spacing_and_case(tmp_path: Path) -> None:
    header = tmp_path / "record.hea"
    header.write_text(
        "record 12 500 5000\n# dx : 164889003, 426783006\n",
        encoding="utf-8",
    )

    assert parse_header_snomeds(header) == [164889003, 426783006]


def test_parse_header_snomeds_preserves_empty_on_malformed_codes(
    tmp_path: Path,
) -> None:
    header = tmp_path / "record.hea"
    header.write_text("record 12 500 5000\n#Dx: 164889003,invalid\n", encoding="utf-8")

    assert parse_header_snomeds(header) == []


def test_parse_pn2021_header_metadata_preserves_demographic_contract(
    tmp_path: Path,
) -> None:
    header = tmp_path / "record.hea"
    header.write_text(
        "record 12 500 5000\n# Age: 63\n#Sex: Female\n#Dx: 426783006\n",
        encoding="utf-8",
    )

    assert parse_pn2021_header_metadata(header) == {
        "age": 63.0,
        "sex": "F",
        "hr": None,
    }


def test_parse_pn2021_header_metadata_preserves_missing_file_defaults(
    tmp_path: Path,
) -> None:
    assert parse_pn2021_header_metadata(tmp_path / "missing.hea") == {
        "age": None,
        "sex": None,
        "hr": None,
    }


def test_pn2021_preprocess_does_not_import_legacy_data_layer() -> None:
    source = (REPO_ROOT / "data_preprocess" / "PN2021_preprocess.py").read_text(
        encoding="utf-8"
    )

    assert "ecg_adv_gen.data" not in source
    assert "from data_preprocess.pn2021_metadata import" in source
