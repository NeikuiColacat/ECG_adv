import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_adv_gen.evaluation.pn2021_corruptions import (
    aggregate_corruption_summary,
    clean_mmap_cache_path,
    clean_npz_cache_path,
    corruption_cache_path,
    filter_record_indices,
    require_clean_eval_json,
    load_clean_metric_lookup,
    load_npz_metadata,
    ref_ids_sha256,
    stable_corruption_seed,
)
from ecg_adv_gen.evaluation.pn2021c import (
    PN2021C_CORRUPTS_PRE_ZSCORE,
    PN2021C_OFFICIAL_OPERATORS,
    build_corruption_op,
    load_custom_severity_profile,
)
from methods.augmix.ecg_ops import BaselineShift, RandomLeadsMask


class _NpzLike:
    def __init__(self, payload):
        self.files = list(payload)
        self._payload = payload

    def __getitem__(self, key):
        return self._payload[key]


def test_pn2021_corruption_cache_paths_are_versioned_and_stable():
    assert corruption_cache_path(
        "/cache", "super5", "ningbo", "baseline_wander", 3, "vtest"
    ) == "/cache/super5_ningbo_baseline_wander_s3_100hz1000_vtest.npz"
    assert clean_mmap_cache_path(
        "/mmap", "super5", "georgia", "vclean"
    ) == "/mmap/super5_georgia_100hz1000_vclean"
    assert clean_npz_cache_path(
        "/npz", "super5", "georgia", "vclean"
    ) == "/npz/super5_georgia_100hz1000_vclean.npz"


def test_pn2021c_official_single_operator_contract_is_package_owned():
    assert PN2021C_OFFICIAL_OPERATORS == (
        "powerline_noise",
        "emg_noise",
        "baseline_wander",
        "baseline_shift",
        "random_leads_masking",
    )
    assert PN2021C_CORRUPTS_PRE_ZSCORE is True


def test_paper_anchored_operator_profile_loads_and_builds():
    profile_path = (
        Path(__file__).resolve().parents[2] / "configs" / "augmentation" / "operators.yaml"
    )
    profile = load_custom_severity_profile(profile_path, "pn2021c_paper_anchored_s5_v1")
    assert profile["powerline_noise"][5]["max_amplitude"] == 0.5
    assert profile["baseline_shift"][5]["amplitude_mode"] == "signed_shared_uniform"
    assert profile["random_leads_masking"][5]["ensure_at_least_one_lead"] is True
    op = build_corruption_op(
        "powerline_noise",
        5,
        "custom",
        sample_rate_hz=500,
        severity_profile_params=profile,
    )
    assert op.freq == 500


def test_baseline_shift_signed_shared_uniform_respects_configured_amplitude():
    np.random.seed(7)
    shifted = BaselineShift(
        max_amplitude=0.5,
        shift_ratio=0.3,
        num_segment=1,
        p=1.0,
        amplitude_mode="signed_shared_uniform",
    )(torch.zeros(12, 1000))
    assert float(shifted.abs().max()) <= 0.5
    assert float(shifted.abs().max()) > 0.0


def test_random_leads_mask_can_guarantee_one_survivor():
    np.random.seed(11)
    sample = torch.ones(12, 100)
    masked = RandomLeadsMask(
        p=1.0,
        mask_leads_prob=1.0,
        ensure_at_least_one_lead=True,
    )(sample)
    surviving_leads = (masked.abs().sum(dim=1) > 0).sum().item()
    assert surviving_leads == 1


def test_load_npz_metadata_decodes_json_scalars_and_falls_back_to_empty_dict():
    assert load_npz_metadata(_NpzLike({})) == {}
    assert load_npz_metadata(
        _NpzLike({"metadata_json": np.array(json.dumps({"version": "v1"}))})
    ) == {"version": "v1"}
    assert load_npz_metadata(
        _NpzLike({"metadata_json": np.array(b'{"center":"cpsc_2018"}')})
    ) == {"center": "cpsc_2018"}
    assert load_npz_metadata(_NpzLike({"metadata_json": np.array("not json")})) == {}


def test_stable_corruption_seed_is_reproducible_and_sensitive_to_parts():
    first = stable_corruption_seed(20260501, "emg_noise", 5, 12)
    assert first == stable_corruption_seed(20260501, "emg_noise", 5, 12)
    assert first != stable_corruption_seed(20260501, "emg_noise", 5, 13)
    assert 0 <= first < 2**32


def test_filter_record_indices_excludes_string_ids_before_limit():
    indices = filter_record_indices(
        np.array([101, 102, "103", "104"]),
        exclude_ids={"102", "104"},
        limit=1,
    )
    np.testing.assert_array_equal(indices, np.array([0], dtype=np.int64))


def test_ref_ids_sha256_sorts_ids_and_keeps_trailing_newline_contract():
    assert ref_ids_sha256({"b", "a"}) == (
        "911169ddaaf146aff539f58c26c489af3b892dff0fe283c1c264c65ae5aa59a2"
    )
    assert ref_ids_sha256(set()) == (
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
    )


def test_load_clean_metric_lookup_reads_per_center_metrics(tmp_path):
    path = tmp_path / "clean.json"
    path.write_text(
        json.dumps(
            {
                "pn2021": {
                    "per_center": {
                        "ningbo": {"macro_auroc": 0.9, "macro_auprc": 0.5},
                        "georgia": {"macro_auroc": 0.8},
                    }
                }
            }
        )
    )

    assert load_clean_metric_lookup(path) == {
        "ningbo": {"macro_auroc": 0.9, "macro_auprc": 0.5},
        "georgia": {"macro_auroc": 0.8, "macro_auprc": None},
    }
    assert load_clean_metric_lookup(None) == {}


def test_require_clean_eval_json_blocks_paper_mode_without_clean_baseline():
    with pytest.raises(ValueError, match="--clean_eval_json is required"):
        require_clean_eval_json(None, diagnostic_without_clean=False)

    require_clean_eval_json(None, diagnostic_without_clean=True)
    require_clean_eval_json("/clean/eval.json", diagnostic_without_clean=False)


def test_aggregate_corruption_summary_skips_nan_metrics_and_missing_clean_drops():
    results = {
        "per_center": {
            "ningbo": {
                "emg_noise": {
                    "1": {
                        "macro_auroc": 0.7,
                        "macro_auprc": 0.4,
                        "auroc_drop_vs_clean": 0.1,
                        "auprc_drop_vs_clean": None,
                    }
                }
            },
            "georgia": {
                "emg_noise": {
                    "1": {
                        "macro_auroc": float("nan"),
                        "macro_auprc": 0.2,
                        "auroc_drop_vs_clean": None,
                        "auprc_drop_vs_clean": 0.3,
                    }
                }
            },
        }
    }

    out = aggregate_corruption_summary(results)
    row = out["emg_noise"]["1"]
    assert row["n_centers"] == 2
    assert row["mean_macro_auroc"] == 0.7
    assert row["mean_macro_auprc"] == 0.30000000000000004
    assert row["mean_auroc_drop_vs_clean"] == 0.1
    assert row["mean_auprc_drop_vs_clean"] == 0.3

    empty = aggregate_corruption_summary(
        {
            "per_center": {
                "ningbo": {
                    "powerline_noise": {
                        "5": {
                            "macro_auroc": float("nan"),
                            "macro_auprc": float("nan"),
                            "auroc_drop_vs_clean": None,
                            "auprc_drop_vs_clean": None,
                        }
                    }
                }
            }
        }
    )
    assert math.isnan(empty["powerline_noise"]["5"]["mean_macro_auroc"])
    assert math.isnan(empty["powerline_noise"]["5"]["mean_macro_auprc"])
    assert empty["powerline_noise"]["5"]["mean_auroc_drop_vs_clean"] is None
    assert empty["powerline_noise"]["5"]["mean_auprc_drop_vs_clean"] is None
