from ecg_adv_gen.config.adapters.pn2021_eval import build_pn2021_eval_argv


def _config():
    return {
        "paths": {"data_root": "/data", "output_root": "/out"},
        "paper_protocol": {
            "kshot": {"k": 500, "seed": 20260531},
            "centers": {"target_4": ["ningbo", "georgia"]},
        },
        "preprocess": {"crop_len": 1000, "mode": "minimal_resample", "norm_mode": "per_sample_global"},
        "model": {"backbone": "efficientnet1dv2", "direct_init_root": "/out/direct/pytest/runs"},
        "training": {"eval_batch_size": 192, "num_workers": 4},
        "evaluation": {
            "min_pos": 10,
            "ref_exclusion_meta": ["/refs/ningbo.ref_meta.json", "/refs/georgia.ref_meta.json"],
        },
        "data": {
            "cache": {
                "pn2021_cache_dir": "/cache/npz",
                "pn2021_mmap_cache_dir": "/cache/mmap",
            }
        },
        "runtime": {"run_id": "run1"},
        "experiment": {"name": "pn2021_eval"},
    }


def test_pn2021_eval_adapter_forces_paper_refexcluded_flags():
    argv = [str(x) for x in build_pn2021_eval_argv(_config(), {"matrix": {"center": "ningbo"}})]

    assert "--eval_protocol" in argv
    assert argv[argv.index("--eval_protocol") + 1] == "paper_refexcluded"
    assert argv[argv.index("--min_target_ref_excluded") + 1] == "500"
    assert "--exclude_ref_ids" in argv
    assert "--report_drop_all_zero_pn2021" in argv
    assert argv[argv.index("--output_path") + 1] == (
        "/out/pn2021_eval/run1/ningbo/eval_result_v7_super5_sjr_rgq_refexcluded.json"
    )
