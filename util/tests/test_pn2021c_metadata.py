from copy import deepcopy

import pytest

from ecg_adv_gen.evaluation.pn2021c_metadata import (
    PN2021CMetadataError,
    build_center_scoped_clean_eval_payload,
    build_pn2021c_metadata_payload,
    evaluation_k500_identities,
    validate_pn2021c_metadata_compatibility,
    validate_target_init_k500_identity,
)


CENTER = "ningbo"
CACHE_VERSION = "pn2021c_v1"
REF_HASH = "a" * 64


def _target_k500_run() -> dict:
    record_ids = [f"r{i}" for i in range(500)]
    return {
        "stage": "k500",
        "center": CENTER,
        "K": 500,
        "target_train_K": 500,
        "selected_ref_record_ids": record_ids,
        "target_train_record_ids": list(record_ids),
        "config": {"stage": "k500", "seed": 20260531},
    }


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


def test_target_adapted_init_must_use_current_k500_identity():
    current = _target_k500_run()
    init = _target_k500_run()
    init["selected_ref_record_ids"][-1] = "other"
    init["target_train_record_ids"] = list(init["selected_ref_record_ids"])
    init["config"]["seed"] = 20260601

    with pytest.raises(PN2021CMetadataError, match="target-adapted initialization K500 identity mismatch"):
        validate_target_init_k500_identity(current, init)


@pytest.mark.parametrize(
    "case",
    [
        "missing_stage",
        "missing_config_stage",
        "empty_center",
        "bool_k",
        "wrong_k",
        "wrong_target_train_k",
        "selected_tuple",
        "selected_duplicate",
        "selected_short",
        "missing_target_train_ids",
        "target_train_tuple",
    ],
)
def test_target_identity_rejects_incomplete_k500_contract(case: str):
    current = _target_k500_run()
    if case == "missing_stage":
        current.pop("stage")
    elif case == "missing_config_stage":
        current["config"].pop("stage")
    elif case == "empty_center":
        current["center"] = ""
    elif case == "bool_k":
        current["K"] = True
    elif case == "wrong_k":
        current["K"] = 499
    elif case == "wrong_target_train_k":
        current["target_train_K"] = 499
    elif case == "selected_tuple":
        current["selected_ref_record_ids"] = tuple(current["selected_ref_record_ids"])
    elif case == "selected_duplicate":
        current["selected_ref_record_ids"][-1] = current["selected_ref_record_ids"][0]
        current["target_train_record_ids"] = list(current["selected_ref_record_ids"])
    elif case == "selected_short":
        current["selected_ref_record_ids"].pop()
        current["target_train_record_ids"].pop()
    elif case == "missing_target_train_ids":
        current.pop("target_train_record_ids")
    elif case == "target_train_tuple":
        current["target_train_record_ids"] = tuple(current["target_train_record_ids"])

    with pytest.raises(PN2021CMetadataError):
        validate_target_init_k500_identity(current, deepcopy(current))


def test_source_only_init_is_allowed_without_k500_identity():
    current = _target_k500_run()
    source_only = {
        "stage": "ptbxl_source",
        "center": None,
        "K": 0,
        "target_train_K": 0,
        "selected_ref_record_ids": [],
        "target_train_record_ids": [],
        "config": {"stage": "ptbxl_source"},
    }

    assert validate_target_init_k500_identity(current, source_only)["source_only"] is True


def test_source_only_identity_rejects_conflicting_config_stage():
    current = _target_k500_run()
    contradictory_source = {
        "stage": "ptbxl_source",
        "center": None,
        "K": 0,
        "target_train_K": 0,
        "selected_ref_record_ids": [],
        "target_train_record_ids": [],
        "config": {"stage": "k500"},
    }

    with pytest.raises(PN2021CMetadataError, match="stage"):
        validate_target_init_k500_identity(current, contradictory_source)


@pytest.mark.parametrize(
    ("field", "value"),
    [("center", CENTER), ("K", 1), ("target_train_K", 1)],
)
def test_source_only_identity_rejects_target_state(field: str, value: object):
    current = _target_k500_run()
    source_only = {
        "stage": "ptbxl_source",
        "center": None,
        "K": 0,
        "target_train_K": 0,
        "selected_ref_record_ids": [],
        "target_train_record_ids": [],
        "config": {"stage": "ptbxl_source"},
    }
    source_only[field] = value

    with pytest.raises(PN2021CMetadataError, match=field):
        validate_target_init_k500_identity(current, source_only)


@pytest.mark.parametrize(
    "field",
    ["center", "K", "target_train_K", "selected_ref_record_ids", "target_train_record_ids", "config.stage"],
)
def test_source_only_identity_rejects_missing_contract_field(field: str):
    current = _target_k500_run()
    source_only = {
        "stage": "ptbxl_source",
        "center": None,
        "K": 0,
        "target_train_K": 0,
        "selected_ref_record_ids": [],
        "target_train_record_ids": [],
        "config": {"stage": "ptbxl_source"},
    }
    if field == "config.stage":
        source_only["config"].pop("stage")
    else:
        source_only.pop(field)

    with pytest.raises(PN2021CMetadataError):
        validate_target_init_k500_identity(current, source_only)


def test_target_identity_rejects_source_only_config_stage():
    current = _target_k500_run()
    contradictory_target = _target_k500_run()
    contradictory_target["config"]["stage"] = "ptbxl_source"

    with pytest.raises(PN2021CMetadataError, match="stage"):
        validate_target_init_k500_identity(current, contradictory_target)


def test_evaluation_k500_identity_rejects_multiple_hashes_for_one_center():
    payload = {
        "per_center": {
            CENTER: {
                "powerline_noise": {
                    "5": {"metadata_compatibility": {"n_excluded_ref": 500, "ref_record_ids_sha256": "a" * 64}}
                },
                "emg_noise": {
                    "5": {"metadata_compatibility": {"n_excluded_ref": 500, "ref_record_ids_sha256": "b" * 64}}
                },
            }
        }
    }

    with pytest.raises(PN2021CMetadataError, match="multiple evaluation K500 identities"):
        evaluation_k500_identities(payload)


def test_evaluation_k500_identity_ignores_unrelated_nested_hashes():
    payload = {
        "per_center": {
            CENTER: {
                "unrelated": {
                    "deep": {
                        "n_excluded_ref": 500,
                        "ref_record_ids_sha256": REF_HASH,
                    }
                }
            }
        }
    }

    assert evaluation_k500_identities(payload) == {}


def test_evaluation_k500_identity_wraps_malformed_count():
    payload = {
        "per_center": {
            CENTER: {
                "emg_noise": {
                    "5": {
                        "metadata_compatibility": {
                            "n_excluded_ref": "not-an-int",
                            "ref_record_ids_sha256": REF_HASH,
                        }
                    }
                }
            }
        }
    }

    with pytest.raises(PN2021CMetadataError, match="count"):
        evaluation_k500_identities(payload)


@pytest.mark.parametrize("count", [True, 500.9, "500"])
def test_evaluation_k500_identity_rejects_non_integer_count_types(count: object):
    payload = {
        "per_center": {
            CENTER: {
                "emg_noise": {
                    "5": {
                        "metadata_compatibility": {
                            "n_excluded_ref": count,
                            "ref_record_ids_sha256": REF_HASH,
                        }
                    }
                }
            }
        }
    }

    with pytest.raises(PN2021CMetadataError, match="count"):
        evaluation_k500_identities(payload)


def test_evaluation_k500_identity_rejects_embedded_center_mismatch():
    payload = {
        "per_center": {
            CENTER: {
                "emg_noise": {
                    "5": {
                        "metadata": {
                            "pn2021c": {
                                "center": "georgia",
                                "n_excluded_ref": 500,
                                "ref_record_ids_sha256": REF_HASH,
                            }
                        }
                    }
                }
            }
        }
    }

    with pytest.raises(PN2021CMetadataError, match="center"):
        evaluation_k500_identities(payload)


def test_stream_metadata_keeps_clean_evaluation_ref_identity():
    from ecg_adv_gen.runner.pn2021c_eval import _merge_clean_metadata

    clean_eval = {
        "label_mapping": {"version": "v7", "hash": "555ec85d5b51"},
        "preprocess": {"contract_id": "contract"},
        "pn2021": {
            "eval_protocol": {
                "status": "paper_safe",
                "eval_protocol": "paper_refexcluded",
                "target_ref_id_hashes": {CENTER: REF_HASH},
            }
        },
    }

    merged = _merge_clean_metadata(
        {"pn2021": {"cache_version": "v7", "records": 123}},
        clean_eval,
    )

    assert merged["pn2021"]["eval_protocol"]["target_ref_id_hashes"][CENTER] == REF_HASH
    assert merged["pn2021"]["cache_version"] == "v7"
    assert merged["pn2021"]["records"] == 123
