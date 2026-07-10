from ecg_adv_gen.config.adapters.pn2021c_eval import (
    audit_pn2021c_eval_command,
    build_pn2021c_eval_argv,
)
from ecg_adv_gen.config.adapters.ecgfounder_pn2021c_eval import (
    audit_ecgfounder_pn2021c_eval_command,
    build_ecgfounder_pn2021c_eval_argv,
)
from ecg_adv_gen.evaluation.pn2021c_protocol import (
    OFFICIAL_S5_COMPOSITE_CORRUPTION_SET,
    official_s5_depth23_composites,
)


def _config():
    return {
        "paths": {"output_root": "/out", "data_root": "/data"},
        "paper_protocol": {
            "kshot": {"k": 500, "seed": 20260531},
            "centers": {"target_4": ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]},
        },
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


def _ecgfounder_config():
    config = _config()
    config["model"]["checkpoint"] = "/ecgfounder/checkpoint.pt"
    config["training"]["eval_batch_size"] = 96
    config["training"]["num_workers"] = 0
    config["evaluation"]["corruption_input"] = "bottleneck5000"
    return config


def test_ecgfounder_pn2021c_eval_adapter_requires_k500_ref_exclusion_gate():
    config = _ecgfounder_config()
    config["evaluation"]["output_stem"] = "eval_pn2021_c_ecgfounder_official_s5_locked"
    method = {
        "name": "threechain_locked",
        "family": "ecgfounder_vae_lhat_augmix_threechain_locked_k500",
        "run_dir_template": "{output_root}/{family}/{run_id}/runs/{center}_locked",
    }
    command = {
        "argv": build_ecgfounder_pn2021c_eval_argv(
            config,
            {"matrix": {"center": "ningbo", "method": method}},
        ),
        "matrix": {"center": "ningbo", "method": method},
    }
    argv = [str(x) for x in command["argv"]]

    assert argv[argv.index("--min_target_ref_excluded") + 1] == "500"
    assert argv[argv.index("--exclude_ref_ids") + 1].endswith(
        "/ningbo/k500_seed20260531/ningbo_real_k500_seed20260531.ref_meta.json"
    )
    assert audit_ecgfounder_pn2021c_eval_command(command, config=config)["errors"] == []

    start = command["argv"].index("--exclude_ref_ids")
    del command["argv"][start : start + 2]
    assert any(
        "--exclude_ref_ids" in error
        for error in audit_ecgfounder_pn2021c_eval_command(command, config=config)["errors"]
    )


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


def test_pn2021c_eval_adapter_emits_custom_severity_profile_file_and_name():
    config = _config()
    config["evaluation"]["severity_profile"] = "custom"
    config["evaluation"]["severity_params_file"] = "configs/corruption_profiles/candidates.yaml"
    config["evaluation"]["severity_params_name"] = "emg_amp2p3"
    argv = [
        str(x)
        for x in build_pn2021c_eval_argv(
            config,
            {"matrix": {"center": "cpsc_2018", "method": {"name": "vae_lhat", "family": "family", "noaug_suffix": ""}}},
        )
    ]

    assert argv[argv.index("--severity_profile") + 1] == "custom"
    assert argv[argv.index("--severity_params_file") + 1] == "configs/corruption_profiles/candidates.yaml"
    assert argv[argv.index("--severity_params_name") + 1] == "emg_amp2p3"


def test_pn2021c_eval_adapter_expands_official_s5_depth23_composite_set():
    config = _config()
    config["evaluation"]["corruption_set"] = OFFICIAL_S5_COMPOSITE_CORRUPTION_SET
    config["evaluation"]["severities"] = [5]
    argv = [
        str(x)
        for x in build_pn2021c_eval_argv(
            config,
            {"matrix": {"center": "ningbo", "method": {"name": "vae_lhat", "family": "family"}}},
        )
    ]

    start = argv.index("--corruptions") + 1
    end = argv.index("--severities")
    assert argv[start:end] == official_s5_depth23_composites()


def test_ecgfounder_pn2021c_eval_adapter_emits_custom_severity_profile_file_and_name():
    config = _ecgfounder_config()
    config["evaluation"]["severity_profile"] = "custom"
    config["evaluation"]["severity_params_file"] = "configs/corruption_profiles/candidates.yaml"
    config["evaluation"]["severity_params_name"] = "power_native_amp8"
    method = {
        "name": "ecgfounder_k500_fullft_locked",
        "family": "ecgfounder_k500_fullft_locked",
        "run_dir_template": "{output_root}/{family}/{run_id}/runs/{center}_k500_fullft_locked",
    }
    argv = [
        str(x)
        for x in build_ecgfounder_pn2021c_eval_argv(
            config,
            {"matrix": {"center": "cpsc_2018", "method": method}},
        )
    ]

    assert argv[argv.index("--severity_profile") + 1] == "custom"
    assert argv[argv.index("--severity_params_file") + 1] == "configs/corruption_profiles/candidates.yaml"
    assert argv[argv.index("--severity_params_name") + 1] == "power_native_amp8"


def test_ecgfounder_pn2021c_eval_adapter_expands_official_s5_depth23_composite_set():
    config = _ecgfounder_config()
    config["evaluation"]["corruption_set"] = OFFICIAL_S5_COMPOSITE_CORRUPTION_SET
    config["evaluation"]["severities"] = [5]
    method = {
        "name": "ecgfounder_k500_fullft_locked",
        "family": "ecgfounder_k500_fullft_locked",
        "run_dir_template": "{output_root}/{family}/{run_id}/runs/{center}_k500_fullft_locked",
    }
    argv = [
        str(x)
        for x in build_ecgfounder_pn2021c_eval_argv(
            config,
            {"matrix": {"center": "georgia", "method": method}},
        )
    ]

    start = argv.index("--corruptions") + 1
    end = argv.index("--severities")
    assert argv[start:end] == official_s5_depth23_composites()


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


def test_pn2021c_eval_adapter_allows_locked_model_dir_template():
    config = _config()
    config["paper_protocol"]["selection"] = {"policy": "last_checkpoint_only"}
    config["runtime"] = {"run_id": "locked_rawfirst_20260618_effnet_copies2"}
    method = {
        "name": "threechain_locked",
        "family": "effnet_vae_lhat_augmix_threechain_locked_k500",
        "model_dir_template": (
            "{output_root}/{family}/{run_id}/"
            "{center}_realall_targetheavy_M20_lam0p05_augmix_s5_locked_ep30_seed{eval_seed}"
        ),
        "clean_eval_name": "eval_result_v7_exclrefs_crop1000.json",
    }
    argv = [
        str(x)
        for x in build_pn2021c_eval_argv(
            config,
            {"matrix": {"center": "georgia", "method": method}},
        )
    ]
    model_dir = argv[argv.index("--model_dir") + 1]

    assert model_dir == (
        "/out/effnet_vae_lhat_augmix_threechain_locked_k500/"
        "locked_rawfirst_20260618_effnet_copies2/"
        "georgia_realall_targetheavy_M20_lam0p05_augmix_s5_locked_ep30_seed20260601"
    )
    assert argv[argv.index("--clean_eval_json") + 1] == (
        f"{model_dir}/eval_result_v7_exclrefs_crop1000.json"
    )
    assert argv[argv.index("--checkpoint_name") + 1] == "last_model.pt"


def test_pn2021c_eval_adapter_emits_locked_raw_first_protocol_when_configured():
    config = _config()
    config["evaluation"]["corruption_input"] = "raw_first"
    command = {
        "argv": build_pn2021c_eval_argv(
            config,
            {"matrix": {"center": "ningbo", "method": {"name": "vae_lhat", "family": "family", "noaug_suffix": ""}}},
        ),
        "matrix": {"center": "ningbo", "method": {"name": "vae_lhat", "family": "family", "noaug_suffix": ""}},
    }
    argv = [str(x) for x in command["argv"]]

    assert argv[argv.index("--corruption_input") + 1] == "raw_first"
    audit = audit_pn2021c_eval_command(command, config=config)
    assert audit["errors"] == []
