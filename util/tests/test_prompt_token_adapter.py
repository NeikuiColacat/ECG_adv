from ecg_adv_gen.config.adapters.prompt_token_online_at import build_prompt_token_online_at_argv


def _config():
    return {
        "paths": {"data_root": "/data", "output_root": "/out"},
        "paper_protocol": {"kshot": {"k": 500, "seed": 20260531}},
        "data": {"kshot_subset_root": "/refs"},
        "model": {"name": "efficientnet1dv2", "direct_init_root": "/out/direct/run1/runs"},
        "training": {
            "epochs": 20,
            "batch_size": 64,
            "num_workers": 4,
            "grad_clip": 1.0,
            "optimizer": {"lr": 0.001, "weight_decay": 0.0001},
        },
        "adaptation": {
            "prompt_token": {
                "online_at_root": "/out/online/run1",
                "gated_dir": "/out/prompt/run1/generated/target_token/ningbo/gated",
                "center": "ningbo",
                "classes": ["NORM", "MI", "STTC"],
            },
            "hull": {
                "M": 20,
                "lambda": 0.05,
                "steps": 3,
                "lr": 0.25,
                "label_mode": "compatible",
                "mix_label_mode": "anchor_soft",
                "label_lambda_y": 0.25,
                "label_new_class_cap": 0.25,
                "neighbor_distance_space": "standardized",
                "neighbor_mode": "local_random",
                "neighbor_pool_size": 120,
                "neighbor_pool_multiplier": 4,
            },
            "anchors": {"k_anchor": 300},
            "attack": {"pgd_batch": 32},
            "loss": {
                "clean_weight": 1.0,
                "target_real_weight": 40.0,
                "adv_weight": 0.6,
                "adv_weight_warmup_epochs": 10,
                "label_mode": "latent_mixed_teacher",
                "teacher_mix": 0.4,
            },
            "latent_augmix": {"latent_weight_cap": 0.25, "width": 3, "depth": -1, "alpha": 1.0, "severity": 2},
        },
    }


def test_prompt_token_online_at_adapter_uses_target_real_val_and_run_scoped_outputs():
    argv = [str(x) for x in build_prompt_token_online_at_argv(_config(), {"matrix": {}})]

    assert argv[argv.index("--quick_eval_source") + 1] == "target_real_val"
    assert argv[argv.index("--output_dir") + 1] == "/out/online/run1/ningbo"
    assert argv[argv.index("--ref_meta_json") + 1].endswith("/gated_samples.ref_meta.json")
    assert "--build_class_trust" not in argv
    assert argv[argv.index("--classes_in_scope") + 1 : argv.index("--enable_latent_augmix_branch")] == [
        "NORM",
        "MI",
        "STTC",
    ]
