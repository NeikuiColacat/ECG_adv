"""CPU-only tests for Super5 metadata extraction."""

from __future__ import annotations

import pytest

from ecg_adv_gen.labels import (
    Super5MetadataError,
    default_class_order,
    get_super5_metadata,
    pn2021_super5_label_mapping_payload,
    validate_super5_metadata,
)
from scripts.triple_labels import label_schemes


def test_super5_metadata_matches_legacy_label_scheme_literals():
    metadata = get_super5_metadata()

    assert list(metadata.class_order) == label_schemes.CLASS_NAMES_SUPER5
    assert metadata.mapping_version == label_schemes.SUPER5_PN2021_MAPPING_VERSION
    assert metadata.mapping_hash == label_schemes.SUPER5_PN2021_MAPPING_HASH
    assert metadata.num_classes == len(metadata.class_order)


def test_super5_metadata_validation_accepts_current_protocol():
    metadata = get_super5_metadata()

    validate_super5_metadata(
        mapping_version=metadata.mapping_version,
        mapping_hash=metadata.mapping_hash,
        class_order=metadata.class_order,
        num_classes=metadata.num_classes,
    )
    assert default_class_order() == list(metadata.class_order)
    assert pn2021_super5_label_mapping_payload() == {
        "pn2021_super5": {
            "mapping_version": metadata.mapping_version,
            "mapping_hash": metadata.mapping_hash,
        }
    }


def test_super5_metadata_validation_rejects_drift():
    metadata = get_super5_metadata()

    with pytest.raises(Super5MetadataError, match="Mapping hash mismatch"):
        validate_super5_metadata(
            mapping_version=metadata.mapping_version,
            mapping_hash="bad",
            class_order=metadata.class_order,
            num_classes=metadata.num_classes,
        )


def test_pn2021_super5_v7_sjr_rgq_policy_deltas():
    assert label_schemes.SUPER5_PN2021_MAPPING_VERSION == (
        "v7_super5_sjr_rgq_review_20260528"
    )

    def mapped_classes(*codes: int) -> set[str]:
        label = label_schemes.snomed_list_to_super5([*codes, 426783006])
        return {
            name
            for name, value in zip(label_schemes.CLASS_NAMES_SUPER5, label)
            if value == 1.0
        }

    assert mapped_classes(418818005) == {"CD"}  # Brugada
    assert mapped_classes(49578007) == {"CD"}  # shortened PR interval
    assert mapped_classes(55827005) == {"HYP"}  # left ventricular high voltage
    assert mapped_classes(67751000119106) == {"HYP"}  # right atrial high voltage
    assert mapped_classes(164912004) == {"HYP"}  # P wave change
    assert mapped_classes(251223006) == {"HYP"}  # tall P wave
    assert mapped_classes(251259000) == {"STTC"}  # high T-voltage

    # SJR/RGQ keeps these outside Super5 but still suppresses sinus-rhythm NORM.
    assert mapped_classes(5609005) == set()  # sinus arrest
    assert mapped_classes(60423000) == set()  # sinus node dysfunction
    assert mapped_classes(10370003) == set()  # pacing rhythm

    # These are fully ignored by the SJR/RGQ strategy, so sinus rhythm remains NORM.
    assert mapped_classes(251198002) == {"NORM"}  # clockwise rotation
    assert mapped_classes(251199005) == {"NORM"}  # counterclockwise rotation
    assert mapped_classes(428417006) == {"NORM"}  # early repolarization
    assert mapped_classes(61721007) == {"NORM"}  # vectorcardiographic loop
    assert mapped_classes(251139008) == {"NORM"}  # suspect arm leads reversed
    assert mapped_classes(53741008) == {"NORM"}  # coronary heart disease
