from ecg_adv_gen.config.adapters.direct_finetune import build_direct_finetune_argv


def _config():
    return {
        "paths": {"output_root": "/out"},
        "paper_protocol": {"kshot": {"k": 500, "seed": 20260531, "subset_seed": 20260531}},
        "preprocess": {"crop_len": 1000},
        "training": {
            "epochs": 30,
            "batch_size": 64,
            "eval_batch_size": 192,
            "num_workers": 4,
            "grad_clip": 1.0,
            "pos_weight_clip_max": 10,
            "optimizer": {"lr": 0.001, "weight_decay": 0.0001},
        },
        "data": {"kshot_subset_root": "/refs"},
        "runtime": {"run_id": "run1"},
        "experiment": {"name": "direct_matrix"},
    }


def test_direct_finetune_adapter_emits_k_seed_center_and_output_root():
    argv = [str(x) for x in build_direct_finetune_argv(_config(), {"matrix": {"center": "ningbo"}})]

    assert argv[argv.index("--centers") + 1] == "ningbo"
    assert argv[argv.index("--k") + 1] == "500"
    assert argv[argv.index("--subset_seed") + 1] == "20260531"
    assert argv[argv.index("--seed") + 1] == "20260531"
    assert argv[argv.index("--subset_root") + 1] == "/refs"
    assert argv[argv.index("--out_root") + 1] == "/out/direct_matrix/run1/ningbo"
