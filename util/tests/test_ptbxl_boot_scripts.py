from __future__ import annotations

import json

from boot_scripts.train_ptbxl_ecgfounder import main as ecgfounder_main
from boot_scripts.train_ptbxl_effnet import main as effnet_main


def test_effnet_boot_script_dry_run_resolves_100hz_profile(capsys):
    assert effnet_main(["--dry-run", "--epochs", "2", "--num-workers", "0"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["model"]["name"] == "efficientnet1dv2"
    assert payload["model"]["sampling_rate_hz"] == 100
    assert payload["training_parameters"]["epochs"] == 2
    assert payload["training_parameters"]["learning_rate"] == 0.001
    assert payload["training_parameters"]["weight_decay"] == 0.0001
    assert payload["training_parameters"]["minimum_learning_rate_ratio"] == 0.01
    assert payload["training_parameters"]["gradient_clip_norm"] == 1.0
    assert payload["training_parameters"]["amp_enabled"] is True
    assert payload["training_parameters"]["amp_dtype"] == "bfloat16"
    assert payload["dataloader_parameters"]["train_batch_size"] == 128
    assert payload["dataloader_parameters"]["eval_batch_size"] == 256
    assert payload["dataloader_parameters"]["num_workers"] == 0


def test_ecgfounder_boot_script_dry_run_resolves_500hz_fullft_profile(capsys):
    assert ecgfounder_main(["--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["model"]["name"] == "ecgfounder"
    assert payload["model"]["sampling_rate_hz"] == 500
    assert payload["trainable_scope"] == "full"
    assert payload["training_parameters"]["epochs"] == 5
    assert payload["training_parameters"]["learning_rate"] == 0.0001
    assert payload["training_parameters"]["weight_decay"] == 0.00001
    assert payload["training_parameters"]["minimum_learning_rate_ratio"] == 1.0
    assert payload["training_parameters"]["gradient_clip_norm"] == 1.0
    assert payload["training_parameters"]["amp_enabled"] is True
    assert payload["training_parameters"]["amp_dtype"] == "bfloat16"
    assert payload["dataloader_parameters"]["train_batch_size"] == 64
    assert payload["dataloader_parameters"]["eval_batch_size"] == 128
