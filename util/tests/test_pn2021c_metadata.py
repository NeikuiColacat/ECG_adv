from copy import deepcopy

import pytest

from ecg_adv_gen.evaluation.pn2021c_metadata import (
    PN2021CMetadataError,
    build_center_scoped_clean_eval_payload,
    build_pn2021c_metadata_payload,
    validate_pn2021c_metadata_compatibility,
)


CENTER = "ningbo"
CACHE_VERSION = "pn2021c_v1"
REF_HASH = "a" * 64


def _compatible_metadata() -> tuple[dict, dict]:
    source = {
        "label_mapping": {
            "version": "v7_super5_sjr_rgq_review_20260528",
            "hash": "555ec85d5b51",
        },
        "preprocess": {
            "contract_id": "pn2021_100hz_global_zscore_v1",
            "preprocess_mode": "resample_100hz",
            "norm_mode": "global_zscore",
            "crop_len": 1000,
            "target_len": 1000,
        },
    }
    clean = build_center_scoped_clean_eval_payload(
        source,
        center=CENTER,
        n_excluded_ref=500,
        ref_record_ids_sha256=REF_HASH,
        preprocess_contract_id=source["preprocess"]["contract_id"],
        preprocess_mode=source["preprocess"]["preprocess_mode"],
        norm_mode=source["preprocess"]["norm_mode"],
        crop_len=source["preprocess"]["crop_len"],
    )
    corrupt = build_pn2021c_metadata_payload(
        clean_metadata=clean,
        center=CENTER,
        corruption="baseline_wander",
        public_severity=2,
        internal_severity=1,
        cache_version=CACHE_VERSION,
        n_excluded_ref=500,
        preprocess_contract_id=source["preprocess"]["contract_id"],
        ref_record_ids_sha256=REF_HASH,
    )
    return clean, corrupt


def _validate(clean: dict, corrupt: dict) -> dict:
    return validate_pn2021c_metadata_compatibility(
        clean_eval=clean,
        corrupt_metadata=corrupt,
        center=CENTER,
        required_cache_version=CACHE_VERSION,
    )


def test_clean_and_corrupt_metadata_are_compatible():
    clean, corrupt = _compatible_metadata()

    compatibility = _validate(clean, corrupt)

    assert compatibility["compatible"] is True
    assert compatibility["n_excluded_ref"] == 500
    assert compatibility["ref_record_ids_sha256"] == REF_HASH


def test_clean_and_corrupt_metadata_reject_n_excluded_ref_mismatch():
    clean, corrupt = _compatible_metadata()
    corrupt = deepcopy(corrupt)
    corrupt["pn2021c"]["n_excluded_ref"] = 499

    with pytest.raises(PN2021CMetadataError, match="n_excluded_ref mismatch for ningbo"):
        _validate(clean, corrupt)


def test_clean_and_corrupt_metadata_reject_ref_record_ids_sha256_mismatch():
    clean, corrupt = _compatible_metadata()
    corrupt = deepcopy(corrupt)
    corrupt["pn2021c"]["ref_record_ids_sha256"] = "b" * 64

    with pytest.raises(PN2021CMetadataError, match="ref_record_ids_sha256 mismatch for ningbo"):
        _validate(clean, corrupt)
