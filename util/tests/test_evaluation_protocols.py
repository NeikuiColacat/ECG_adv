import pytest

from ecg_adv_gen.evaluation.protocols import (
    PN2021ProtocolError,
    normalize_paper_preprocess_mode,
    validate_pn2021_eval_protocol,
)


def _per_center(n_excluded_ref=500):
    return {
        "ningbo": {"n_excluded_ref": n_excluded_ref, "effective_n": 10},
        "chapman_shaoxing": {"n_excluded_ref": n_excluded_ref, "effective_n": 10},
        "cpsc_2018": {"n_excluded_ref": n_excluded_ref, "effective_n": 10},
        "cpsc_2018_extra": {"n_excluded_ref": 0, "effective_n": 10},
        "georgia": {"n_excluded_ref": n_excluded_ref, "effective_n": 10},
        "ptb": {"n_excluded_ref": 0, "effective_n": 10},
        "st_petersburg_incart": {"n_excluded_ref": 0, "effective_n": 10},
    }


EVAL_CENTERS = [
    "chapman_shaoxing",
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "ningbo",
    "ptb",
    "st_petersburg_incart",
]

TARGET_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]


def test_paper_refexcluded_requires_target_ref_exclusion():
    with pytest.raises(PN2021ProtocolError, match="ningbo"):
        validate_pn2021_eval_protocol(
            eval_protocol="paper_refexcluded",
            per_center=_per_center(n_excluded_ref=0),
            target_centers=TARGET_CENTERS,
            eval_centers=EVAL_CENTERS,
            preprocess_mode="paper",
            crop_len=1000,
        )


def test_paper_refexcluded_defaults_to_k500_ref_exclusion():
    with pytest.raises(PN2021ProtocolError, match="n_excluded_ref=499"):
        validate_pn2021_eval_protocol(
            eval_protocol="paper_refexcluded",
            per_center=_per_center(n_excluded_ref=499),
            target_centers=TARGET_CENTERS,
            eval_centers=EVAL_CENTERS,
            preprocess_mode="paper",
            crop_len=1000,
        )


def test_paper_refexcluded_requires_all_eval_centers():
    per_center = _per_center()
    per_center.pop("ptb")
    with pytest.raises(PN2021ProtocolError, match="missing eval centers"):
        validate_pn2021_eval_protocol(
            eval_protocol="paper_refexcluded",
            per_center=per_center,
            target_centers=TARGET_CENTERS,
            eval_centers=EVAL_CENTERS,
            preprocess_mode="paper",
            crop_len=1000,
        )


def test_paper_refexcluded_disallows_missing_centers_escape_hatch():
    per_center = _per_center()
    per_center.pop("ptb")
    with pytest.raises(PN2021ProtocolError, match="missing eval centers"):
        validate_pn2021_eval_protocol(
            eval_protocol="paper_refexcluded",
            per_center=per_center,
            target_centers=TARGET_CENTERS,
            eval_centers=EVAL_CENTERS,
            preprocess_mode="paper",
            crop_len=1000,
            allow_missing_centers=True,
        )


def test_diagnostic_unrefexcluded_allows_zero_ref_exclusion():
    result = validate_pn2021_eval_protocol(
        eval_protocol="diagnostic_unrefexcluded",
        per_center=_per_center(n_excluded_ref=0),
        target_centers=TARGET_CENTERS,
        eval_centers=EVAL_CENTERS,
        preprocess_mode="legacy_ecgfounder_filter",
        crop_len=250,
        diagnostic_legacy_eval=True,
        allow_missing_centers=False,
    )
    assert result["status"] == "diagnostic"


def test_minimal_resample_1000_contract_normalizes_to_paper_mode():
    assert (
        normalize_paper_preprocess_mode(
            preprocess_mode="minimal_resample",
            norm_mode="per_sample_global",
            crop_len=1000,
        )
        == "paper"
    )
