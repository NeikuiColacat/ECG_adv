import pytest

from ecg_adv_gen.config.adapters.benchmark_vae_lhat import (
    audit_benchmark_vae_lhat_command,
    build_benchmark_vae_lhat_argv,
)


def _config():
    return {
        "paths": {"output_root": "/out", "data_root": "/data"},
        "paper_protocol": {
            "kshot": {"k": 500, "seed": 20260531, "subset_seed": 20260531},
            "centers": {"target_4": ["ningbo", "georgia"]},
        },
        "model": {"name": "benchmark_resnet1d_wang"},
        "training": {
            "epochs": 30,
            "batch_size": 128,
            "eval_batch_size": 192,
            "num_workers": 4,
            "optimizer": {"lr": 0.00005},
        },
        "data": {"kshot_subset_root": "/refs"},
        "runtime": {"run_id": "run1"},
        "experiment": {"name": "benchmark_resnet1d_vae_lhat_k500_v7_sjr_rgq"},
        "adaptation": {
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
                "target_real_weight": 80.0,
                "adv_weight": 0.3,
                "adv_weight_warmup_epochs": 10,
                "label_mode": "latent_mixed_teacher",
                "teacher_mix": 0.4,
            },
            "latent_augmix": {
                "latent_weight_cap": 0.25,
                "width": 3,
                "depth": -1,
                "alpha": 1.0,
                "severity": 2,
            },
        },
        "evaluation": {"min_pos": 10, "pn2021_limit": 0},
    }


def test_benchmark_vae_lhat_adapter_emits_model_specific_paths():
    argv = [str(x) for x in build_benchmark_vae_lhat_argv(_config(), {"matrix": {"center": "ningbo"}})]

    assert argv[argv.index("--center") + 1] == "ningbo"
    assert argv[argv.index("--model_name") + 1] == "benchmark_resnet1d_wang"
    assert (
        argv[argv.index("--init_ckpt") + 1]
        == "/out/benchmark_resnet1d_wang_direct_k500_v7_sjr_rgq/run1/ningbo/runs/"
        "ningbo_K500_direct_ft_benchmark_resnet1d_wang_ep30_seed20260531_val0p2/best_model.pt"
    )
    assert (
        argv[argv.index("--anchor_base") + 1]
        == "/refs/ningbo/k500_seed20260531/ningbo_real_k500_seed20260531"
    )
    assert argv[argv.index("--out_root") + 1] == "/out/benchmark_resnet1d_wang_vae_lhat_k500_v7_sjr_rgq/run1"
    assert argv[argv.index("--quick_eval_source") + 1] == "target_real_val"


def test_benchmark_vae_lhat_adapter_requires_center():
    with pytest.raises(ValueError, match="requires runner.matrix.center"):
        build_benchmark_vae_lhat_argv(_config(), {"matrix": {}})


def test_benchmark_vae_lhat_audit_accepts_model_specific_command():
    config = _config()
    command = {
        "argv": [
            "/python",
            "scripts/paper/run_effnet_latent_augmix_stage3_20260524.py",
            *build_benchmark_vae_lhat_argv(config, {"matrix": {"center": "ningbo"}}),
        ],
        "matrix": {"center": "ningbo"},
    }

    report = audit_benchmark_vae_lhat_command(command, config=config)

    assert report == {"errors": [], "warnings": []}
