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


def test_pn2021c_eval_adapter_uses_configured_severity_profile_in_output_name():
    config = _config()
    config["evaluation"]["severity_profile"] = "calibrated_10to20pp"
    config["evaluation"]["severities"] = [5]
    argv = [
        str(x)
        for x in build_pn2021c_eval_argv(
            config,
            {"matrix": {"center": "georgia", "method": {"name": "vae_lhat", "family": "family", "noaug_suffix": ""}}},
        )
    ]

    assert argv[argv.index("--severity_profile") + 1] == "calibrated_10to20pp"
    assert argv[argv.index("--severities") + 1] == "5"
    assert argv[argv.index("--output_path") + 1].endswith(
        "/pn2021c_eval/run1/georgia/vae_lhat/"
        "eval_pn2021_c_v7_refexcluded_stream_calibrated_10to20pp.json"
    )


def test_pn2021c_eval_adapter_allows_method_level_run_overrides():
    config = _config()
    method = {
        "name": "rawjsd",
        "family": "effnet_vae_lhat_k500_v7_sjr_rgq",
        "run_stamp": "20260530_rawjsd_s5c2_w8_lowbce",
        "run_leaf_stem": "raw_leaf",
        "eval_seed": 20260601,
        "epochs": 30,
        "clean_eval_name": "clean.json",
    }
    argv = [
        str(x)
        for x in build_pn2021c_eval_argv(
            config,
            {"matrix": {"center": "ningbo", "method": method}},
        )
    ]
    model_dir = argv[argv.index("--model_dir") + 1]

    assert model_dir == (
        "/out/effnet_vae_lhat_k500_v7_sjr_rgq/"
        "seed20260601_k500_v7_sjr_rgq_20260530_rawjsd_s5c2_w8_lowbce/"
        "ningbo_raw_leaf_fullft_k500_ep30_seed20260601"
    )
    assert argv[argv.index("--clean_eval_json") + 1] == f"{model_dir}/clean.json"
