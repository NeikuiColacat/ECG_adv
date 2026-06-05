from ecg_adv_gen.config.adapters.pn2021c_eval import build_pn2021c_eval_argv


def _config():
    return {
        "paths": {"output_root": "/out"},
        "paper_protocol": {"kshot": {"k": 500, "seed": 20260531}},
        "preprocess": {"crop_len": 1000},
        "model": {
            "eval_seed": 20260601,
            "epochs": 30,
            "run_stamp": "20260530_rep1",
            "clean_eval_name": "eval_result_v7_exclrefs_crop1000.json",
            "run_leaf_stem": "leaf",
        },
        "training": {"eval_batch_size": 192, "num_workers": 4},
        "data": {
            "cache": {
                "pn2021_clean_cache_dir": "/cache/clean_npz",
                "pn2021_clean_mmap_cache_dir": "/cache/clean_mmap",
            },
            "kshot_subset_root": "/refs",
        },
        "evaluation": {
            "min_pos": 10,
            "corruption_seed": 20260501,
            "severity_profile": "standard",
            "corruptions": ["emg_noise"],
            "severities": [1, 3],
        },
        "runtime": {"run_id": "run1"},
        "experiment": {"name": "pn2021c_eval"},
    }


def test_pn2021c_eval_adapter_requires_clean_eval_and_cache_version():
    argv = [
        str(x)
        for x in build_pn2021c_eval_argv(
            _config(),
            {"matrix": {"center": "ningbo", "method": {"name": "vae_lhat", "family": "family", "noaug_suffix": ""}}},
        )
    ]

    assert "--clean_eval_json" in argv
    assert "--required_cache_version" in argv
    assert argv[argv.index("--required_cache_version") + 1] == "v7_refexcluded_100hz1000"
    assert "--diagnostic_without_clean" not in argv
    assert argv[argv.index("--exclude_ref_ids") + 1].endswith(
        "/ningbo/k500_seed20260531/ningbo_real_k500_seed20260531.ref_meta.json"
    )
