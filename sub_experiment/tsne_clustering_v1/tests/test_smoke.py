"""Smoke test for the ECGTwin clustering experiment.

Runs the full 3-stage pipeline with tiny sample counts (5 real / 2 gen per
center over 2 centers) to verify end-to-end plumbing: PN2021 loading ->
ECGFounder embed -> ECGTwin generation -> t-SNE + metrics -> disk output.

Does NOT validate numerical results — just that artifacts are produced and
shapes line up. The full experiment must be run via run_experiment.py.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _EXP_DIR.parents[1]
for p in (str(_PROJECT_ROOT), str(_EXP_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def smoke_config(tmp_path_factory):
    """Build a minimal config writing into a tmp output root."""
    out_root = tmp_path_factory.mktemp("tsne_smoke_out")
    cfg = {
        "centers": ["chapman_shaoxing", "georgia"],
        "pn2021_root": "/root/autodl-tmp/physionet2021/training",
        "n_real_per_center": 5,
        "n_gen_per_center": 2,
        "seed": 0,
        "ecgtwin": {
            "device": "cuda:0",
            "load_encoder": True,
            "load_text_model": True,
            "num_inference_steps": 10,
            "batch_size": 1,
        },
        "ecgfounder": {
            "weights_path": "/root/autodl-tmp/ecgfounder/checkpoint/12_lead_ECGFounder.pth",
            "device": "cuda:0",
        },
        "tsne": {"perplexity": 5.0, "random_state": 0, "n_iter": 250},
        "metrics": {"knn_k": 3},
        "output_root": str(out_root),
    }
    cfg_path = out_root / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path, out_root


def test_end_to_end(smoke_config):
    cfg_path, out_root = smoke_config
    from sub_experiment.tsne_clustering_v1.run_experiment import main

    main(str(cfg_path))

    # Expected artifacts
    assert (out_root / "real_signals.pt").exists()
    assert (out_root / "real_features.pt").exists()
    assert (out_root / "gen_signals.pt").exists()
    assert (out_root / "gen_features.pt").exists()
    assert (out_root / "tsne_coords.npy").exists()
    assert (out_root / "metrics.json").exists()

    # Plots
    plots = out_root / "plots"
    assert (plots / "tsne_all.png").exists()
    assert (plots / "purity_bar.png").exists()
    # Per-center plots
    for center in ["chapman_shaoxing", "georgia"]:
        assert (plots / f"tsne_per_center_{center}.png").exists()

    # Metrics shape
    metrics = json.loads((out_root / "metrics.json").read_text())
    assert set(metrics["knn_purity_per_center"].keys()) == {"chapman_shaoxing", "georgia"}
    assert 0.0 <= metrics["knn_purity_overall"] <= 1.0
    assert metrics["random_baseline_purity"] == pytest.approx(0.5, rel=1e-6)
