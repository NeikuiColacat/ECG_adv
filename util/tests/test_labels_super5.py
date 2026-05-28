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
