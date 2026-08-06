"""CPU-only Super5 tests against the rebuilt preprocessing source of truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from data_preprocess.PN2021_preprocess import (
    EXPECTED_CLASS_ORDER,
    _load_config,
    _load_super5_mapping,
    _snomed_list_to_super5,
)
from data_preprocess.PTBXL_preprocess import (
    CLASS_ORDER,
    _build_labels,
    _load_super5_map,
)


REPO = Path(__file__).resolve().parents[2]
PN2021_CONFIG = REPO / "configs" / "data" / "PN2021.yaml"
PN2021_MAPPING = REPO / "configs" / "data" / "PN2021_super5_v7.yaml"
MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
MAPPING_HASH = "555ec85d5b51"


def _mapping() -> dict:
    config = _load_config(PN2021_CONFIG)
    return _load_super5_mapping(PN2021_CONFIG, config)


def _mapped_classes(mapping: dict, *codes: int) -> set[str]:
    label = _snomed_list_to_super5([*codes, 426783006], mapping)
    return {
        name
        for name, value in zip(EXPECTED_CLASS_ORDER, label, strict=True)
        if value == 1
    }


def test_pn2021_mapping_identity_and_class_order_are_locked() -> None:
    mapping = _mapping()

    assert CLASS_ORDER == EXPECTED_CLASS_ORDER
    assert mapping["version"] == MAPPING_VERSION
    assert mapping["hash"] == MAPPING_HASH
    assert mapping["class_order"] == EXPECTED_CLASS_ORDER
    assert mapping["all_zero_policy"] == "kept"
    assert mapping["path"] == PN2021_MAPPING.resolve()


def test_pn2021_super5_v7_review_policy() -> None:
    mapping = _mapping()

    assert _mapped_classes(mapping, 418818005) == {"CD"}  # Brugada
    assert _mapped_classes(mapping, 49578007) == {"CD"}  # shortened PR
    assert _mapped_classes(mapping, 55827005) == {"HYP"}  # LV high voltage
    assert _mapped_classes(mapping, 67751000119106) == {"HYP"}  # RA high voltage
    assert _mapped_classes(mapping, 164912004) == {"HYP"}  # P-wave change
    assert _mapped_classes(mapping, 251223006) == {"HYP"}  # tall P wave
    assert _mapped_classes(mapping, 251259000) == {"STTC"}  # high T voltage

    # Suppress-only findings remove sinus-rhythm NORM without inventing a class.
    assert _mapped_classes(mapping, 5609005) == set()
    assert _mapped_classes(mapping, 60423000) == set()
    assert _mapped_classes(mapping, 10370003) == set()

    # Reviewed ignored codes do not alter the Super5 vector.
    assert _mapped_classes(mapping, 251198002) == {"NORM"}
    assert _mapped_classes(mapping, 251199005) == {"NORM"}
    assert _mapped_classes(mapping, 428417006) == {"NORM"}
    assert _mapped_classes(mapping, 61721007) == {"NORM"}
    assert _mapped_classes(mapping, 251139008) == {"NORM"}
    assert _mapped_classes(mapping, 53741008) == {"NORM"}


def test_abnormal_labels_suppress_norm_and_unknown_codes_are_all_zero() -> None:
    mapping = _mapping()
    norm_with_mi = _snomed_list_to_super5([426783006, 22298006], mapping)
    unknown = _snomed_list_to_super5([999999999], mapping)

    np.testing.assert_array_equal(norm_with_mi, [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(unknown, np.zeros(5, dtype=np.uint8))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda payload: payload.__setitem__("mapping_hash", "bad"),
            "mapping hash mismatch",
        ),
        (
            lambda payload: payload["norm_suppress"].append(426783006),
            "overlap",
        ),
        (
            lambda payload: payload.__setitem__("all_zero_policy", "drop"),
            "all_zero_policy",
        ),
    ],
)
def test_mapping_loader_rejects_identity_or_policy_drift(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    mapping_payload = yaml.safe_load(PN2021_MAPPING.read_text(encoding="utf-8"))
    mutation(mapping_payload)
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        yaml.safe_dump(mapping_payload, sort_keys=False),
        encoding="utf-8",
    )
    config_payload = yaml.safe_load(PN2021_CONFIG.read_text(encoding="utf-8"))
    config_payload["label_mapping_file"] = mapping_path.name
    config_path = tmp_path / "PN2021.yaml"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        _load_super5_mapping(config_path, config_payload)


def test_ptbxl_super5_conversion_uses_diagnostic_classes(tmp_path: Path) -> None:
    statements = tmp_path / "scp_statements.csv"
    statements.write_text(
        ",diagnostic,diagnostic_class\n"
        "NORM,1,NORM\n"
        "IMI,1,MI\n"
        "NONDIAG,0,STTC\n",
        encoding="utf-8",
    )
    mapping = _load_super5_map(statements)
    metadata = pd.DataFrame(
        {
            "scp_codes": [
                {"NORM": 100.0},
                "{'IMI': 80.0, 'NONDIAG': 100.0}",
            ]
        }
    )

    labels = _build_labels(metadata, mapping)

    assert labels.dtype == np.uint8
    np.testing.assert_array_equal(labels[0], [0, 0, 0, 1, 0])
    np.testing.assert_array_equal(labels[1], [0, 0, 1, 0, 0])
