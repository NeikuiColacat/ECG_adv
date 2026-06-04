import json
import math

import numpy as np

from ecg_adv_gen.evaluation.pn2021_corruptions import (
    aggregate_corruption_summary,
    clean_mmap_cache_path,
    clean_npz_cache_path,
    corruption_cache_path,
    filter_record_indices,
    load_clean_metric_lookup,
    load_npz_metadata,
    stable_corruption_seed,
)


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
