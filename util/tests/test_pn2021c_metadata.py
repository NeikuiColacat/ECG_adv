import pytest

from ecg_adv_gen.evaluation.pn2021c_metadata import (
    PN2021CMetadataError,
    build_center_scoped_clean_eval_payload,
    build_pn2021c_metadata_payload,
    validate_pn2021c_metadata_compatibility,
)
from ecg_adv_gen.evaluation.pn2021c_protocol import (
    LOCKED_MAIN_INPUT_ORDER_ID,
    locked_protocol_metadata,
)


def _clean():
    return {
        "label_mapping": {"version": "v7", "hash": "hash-v7"},
        "preprocess": {
            "contract_id": "ptbxl_pn2021_super5_ecgtwin_decode_v1",
            "preprocess_mode": "minimal_resample",
            "norm_mode": "per_sample_global",
            "crop_len": 1000,
            "target_len": 1000,
        },
        "pn2021": {
            "eval_protocol": {
                "status": "paper_safe",
                "eval_protocol": "paper_refexcluded",
                "target_ref_id_hashes": {"ningbo": "refhash"},
            },
            "per_center": {"ningbo": {"n_excluded_ref": 500}},
        },
    }


def _corrupt():
    return {
        "label_mapping": {"version": "v7", "hash": "hash-v7"},
        "preprocess": {
            "contract_id": "ptbxl_pn2021_super5_ecgtwin_decode_v1",
            "preprocess_mode": "minimal_resample",
            "norm_mode": "per_sample_global",
            "crop_len": 1000,
            "target_len": 1000,
        },
        "pn2021c": {
            "cache_version": "v7_refexcluded_100hz1000",
            "center": "ningbo",
            "n_excluded_ref": 500,
            "ref_record_ids_sha256": "refhash",
        },
    }


def test_pn2021c_metadata_accepts_matching_clean_and_corrupt():
    result = validate_pn2021c_metadata_compatibility(
        clean_eval=_clean(),
        corrupt_metadata=_corrupt(),
        center="ningbo",
        required_cache_version="v7_refexcluded_100hz1000",
    )
    assert result["compatible"] is True


def test_pn2021c_metadata_rejects_mapping_hash_drift():
    corrupt = _corrupt()
    corrupt["label_mapping"]["hash"] = "old-hash"
    with pytest.raises(PN2021CMetadataError, match="label_mapping.hash"):
        validate_pn2021c_metadata_compatibility(
            clean_eval=_clean(),
            corrupt_metadata=corrupt,
            center="ningbo",
            required_cache_version="v7_refexcluded_100hz1000",
        )


def test_pn2021c_metadata_rejects_ref_exclusion_drift():
    corrupt = _corrupt()
    corrupt["pn2021c"]["n_excluded_ref"] = 0
    with pytest.raises(PN2021CMetadataError, match="n_excluded_ref"):
        validate_pn2021c_metadata_compatibility(
            clean_eval=_clean(),
            corrupt_metadata=corrupt,
            center="ningbo",
            required_cache_version="v7_refexcluded_100hz1000",
        )


def test_pn2021c_metadata_rejects_diagnostic_clean_eval():
    clean = _clean()
    clean["pn2021"]["eval_protocol"]["status"] = "diagnostic"
    with pytest.raises(PN2021CMetadataError, match="paper_safe"):
        validate_pn2021c_metadata_compatibility(
            clean_eval=clean,
            corrupt_metadata=_corrupt(),
            center="ningbo",
            required_cache_version="v7_refexcluded_100hz1000",
        )


def test_pn2021c_metadata_rejects_preprocess_drift():
    corrupt = _corrupt()
    corrupt["preprocess"]["crop_len"] = 250
    with pytest.raises(PN2021CMetadataError, match="preprocess.crop_len"):
        validate_pn2021c_metadata_compatibility(
            clean_eval=_clean(),
            corrupt_metadata=corrupt,
            center="ningbo",
            required_cache_version="v7_refexcluded_100hz1000",
        )


def test_pn2021c_metadata_rejects_ref_id_hash_drift():
    corrupt = _corrupt()
    corrupt["pn2021c"]["ref_record_ids_sha256"] = "otherhash"
    with pytest.raises(PN2021CMetadataError, match="ref_record_ids_sha256"):
        validate_pn2021c_metadata_compatibility(
            clean_eval=_clean(),
            corrupt_metadata=corrupt,
            center="ningbo",
            required_cache_version="v7_refexcluded_100hz1000",
        )


def test_pn2021c_metadata_supports_clean_eval_nested_mapping_shape():
    clean = _clean()
    clean["label_mapping"] = {
        "pn2021_super5": {"mapping_version": "v7", "mapping_hash": "hash-v7"}
    }
    assert validate_pn2021c_metadata_compatibility(
        clean_eval=clean,
        corrupt_metadata=_corrupt(),
        center="ningbo",
        required_cache_version="v7_refexcluded_100hz1000",
    )["label_mapping_hash"] == "hash-v7"


def test_pn2021c_metadata_builds_center_scoped_clean_payload_for_legacy_clean_eval():
    legacy = {
        "label_mapping": {
            "pn2021_super5": {"mapping_version": "v7", "mapping_hash": "hash-v7"}
        },
        "preprocess_config": {
            "preprocess_mode": "minimal_resample",
            "norm_mode": "per_sample_global",
            "target_len": 1000,
        },
        "pn2021": {
            "per_center": {
                "ningbo": {"macro_auroc": 0.1, "macro_auprc": 0.2},
            },
        },
    }
    clean = build_center_scoped_clean_eval_payload(
        legacy,
        center="ningbo",
        n_excluded_ref=500,
        ref_record_ids_sha256="refhash",
        preprocess_contract_id="ptbxl_pn2021_super5_ecgtwin_decode_v1",
        preprocess_mode="minimal_resample",
        norm_mode="per_sample_global",
        crop_len=1000,
        clean_metrics={"macro_auroc": 0.9, "macro_auprc": 0.5},
    )

    assert clean["pn2021"]["eval_protocol"]["center_scoped_for_pn2021c"] is True
    assert clean["pn2021"]["per_center"]["ningbo"]["macro_auroc"] == 0.9
    assert validate_pn2021c_metadata_compatibility(
        clean_eval=clean,
        corrupt_metadata=_corrupt(),
        center="ningbo",
        required_cache_version="v7_refexcluded_100hz1000",
    )["compatible"] is True


def test_build_pn2021c_metadata_payload_canonicalizes_clean_cache_metadata():
    payload = build_pn2021c_metadata_payload(
        clean_metadata={
            "pn2021_mapping": {"mapping_version": "v7", "mapping_hash": "hash-v7"},
            "preprocess_config": {
                "preprocess_mode": "minimal_resample",
                "norm_mode": "per_sample_global",
                "crop_len": 1000,
                "target_len": 1000,
            },
        },
        center="ningbo",
        corruption="emg_noise",
        public_severity=3,
        internal_severity=6,
        cache_version="v7_refexcluded_100hz1000",
        n_excluded_ref=500,
        preprocess_contract_id="ptbxl_pn2021_super5_ecgtwin_decode_v1",
        ref_record_ids_sha256="refhash",
        source_clean_cache="/cache/clean.npz",
        seed=20260501,
        limit=None,
    )

    assert payload["label_mapping"] == {"version": "v7", "hash": "hash-v7"}
    assert payload["preprocess"] == {
        "contract_id": "ptbxl_pn2021_super5_ecgtwin_decode_v1",
        "preprocess_mode": "minimal_resample",
        "norm_mode": "per_sample_global",
        "crop_len": 1000,
        "target_len": 1000,
    }
    assert payload["pn2021c"]["center"] == "ningbo"
    assert payload["pn2021c"]["n_excluded_ref"] == 500
    assert payload["pn2021c"]["cache_version"] == "v7_refexcluded_100hz1000"
    assert payload["pn2021c"]["ref_record_ids_sha256"] == "refhash"
    assert payload["pn2021_c"] == payload["pn2021c"]


def test_build_pn2021c_metadata_payload_records_locked_input_order():
    payload = build_pn2021c_metadata_payload(
        clean_metadata=_clean(),
        center="ningbo",
        corruption="emg_noise",
        public_severity=5,
        internal_severity=10,
        cache_version="v7_refexcluded_100hz1000",
        n_excluded_ref=500,
        preprocess_contract_id="ptbxl_pn2021_super5_ecgtwin_decode_v1",
        ref_record_ids_sha256="refhash",
        protocol_metadata=locked_protocol_metadata("efficientnet1dv2"),
    )

    assert payload["pn2021c_protocol"]["input_order_id"] == LOCKED_MAIN_INPUT_ORDER_ID
    assert payload["pn2021c_protocol"]["zscore_timing"] == "after_corruption_before_model"
    assert payload["pn2021c"]["input_order_id"] == LOCKED_MAIN_INPUT_ORDER_ID


def test_pn2021c_metadata_rejects_input_order_drift_when_required():
    corrupt = _corrupt()
    corrupt["pn2021c_protocol"] = locked_protocol_metadata("efficientnet1dv2")
    corrupt["pn2021c_protocol"]["input_order_id"] = "pre_zscored_cache_then_corrupt"

    with pytest.raises(PN2021CMetadataError, match="input_order_id"):
        validate_pn2021c_metadata_compatibility(
            clean_eval=_clean(),
            corrupt_metadata=corrupt,
            center="ningbo",
            required_cache_version="v7_refexcluded_100hz1000",
            required_input_order_id=LOCKED_MAIN_INPUT_ORDER_ID,
        )
