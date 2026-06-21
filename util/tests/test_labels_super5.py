"""CPU-only tests for Super5 metadata extraction."""

from __future__ import annotations

import pytest
import numpy as np

from ecg_adv_gen.labels import (
    Super5MetadataError,
    default_class_order,
    get_super5_metadata,
    pn2021_super5_label_mapping_payload,
    validate_super5_metadata,
)
from ecg_adv_gen.labels import super5_mapping
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


def test_package_super5_mapping_owns_conversion_policy(tmp_path, monkeypatch, request):
    assert super5_mapping.SUPER5_PN2021_MAPPING_VERSION == (
        "v7_super5_sjr_rgq_review_20260528"
    )
    assert super5_mapping.SUPER5_PN2021_MAPPING_HASH == "555ec85d5b51"
    assert tuple(super5_mapping.CLASS_NAMES_SUPER5) == get_super5_metadata().class_order

    pn2021_label = super5_mapping.snomed_list_to_super5([426783006, 55827005])
    legacy_pn2021_label = label_schemes.snomed_list_to_super5([426783006, 55827005])
    np.testing.assert_array_equal(pn2021_label, legacy_pn2021_label)
    assert {
        name
        for name, value in zip(super5_mapping.CLASS_NAMES_SUPER5, pn2021_label)
        if value == 1.0
    } == {"HYP"}

    mimic_label = super5_mapping.mimic_report_to_super5(
        "normal sinus rhythm with left bundle branch block"
    )
    legacy_mimic_label = label_schemes.mimic_report_to_super5(
        "normal sinus rhythm with left bundle branch block"
    )
    np.testing.assert_array_equal(mimic_label, legacy_mimic_label)
    assert {
        name
        for name, value in zip(super5_mapping.CLASS_NAMES_SUPER5, mimic_label)
        if value == 1.0
    } == {"CD"}

    scp_statements = tmp_path / "scp_statements.csv"
    scp_statements.write_text(
        ",diagnostic,diagnostic_class\n"
        "NORM,1.0,NORM\n"
        "IMI,1.0,MI\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(super5_mapping, "SCP_STATEMENTS_PATH", scp_statements)
    super5_mapping._load_scp_super5_map.cache_clear()
    request.addfinalizer(super5_mapping._load_scp_super5_map.cache_clear)

    ptbxl_label = super5_mapping.ptbxl_scp_to_super5({"NORM": 100.0, "IMI": 80.0})
    legacy_ptbxl_label = label_schemes.ptbxl_scp_to_super5({"NORM": 100.0, "IMI": 80.0})
    np.testing.assert_array_equal(ptbxl_label, legacy_ptbxl_label)


def test_legacy_label_schemes_reexports_package_super5_functions():
    assert label_schemes.ptbxl_scp_to_super5 is super5_mapping.ptbxl_scp_to_super5
    assert label_schemes.snomed_list_to_super5 is super5_mapping.snomed_list_to_super5
    assert label_schemes.mimic_report_to_super5 is super5_mapping.mimic_report_to_super5
    assert label_schemes.SNOMED_TO_SUPER5_POSITIVE == super5_mapping.SNOMED_TO_SUPER5_POSITIVE
    assert label_schemes.NORM_SUPPRESS_SNOMEDS == super5_mapping.NORM_SUPPRESS_SNOMEDS
