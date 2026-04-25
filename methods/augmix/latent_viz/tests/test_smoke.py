"""Smoke test: end-to-end latent-AugMix viz on 1 center x 1 sample.

Verifies plumbing only (artifacts produced, shapes align, JSON well-formed).
Does NOT validate numerical properties or physiological legality — those are
judged by inspecting the plots/report.json by a human after a real run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _EXP_DIR.parents[2]
for p in (str(_PROJECT_ROOT), str(_EXP_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def smoke_config(tmp_path_factory):
    out_root = tmp_path_factory.mktemp("latent_viz_smoke_out")
    cfg = {
        "centers": ["chapman_shaoxing"],
        "pn2021_root": "/root/autodl-tmp/physionet2021/training",
        "n_samples_per_center": 1,
        "seed": 0,
        "ecgtwin": {
            "device": "cuda:0",
            "load_encoder": True,
            "load_text_model": False,
        },
        "augmix": {
            "severity": 5,
            "width": 2,
            "depth": 2,
            "alpha": 1.0,
            "ops": ["powerline_noise", "baseline_wander"],
        },
        "output_root": str(out_root),
    }
    cfg_path = out_root / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path, out_root


def test_end_to_end(smoke_config):
    cfg_path, out_root = smoke_config
    from methods.augmix.latent_viz.run_viz import main

    agg = main(str(cfg_path))
    assert agg["n_samples"] == 1

    # Exactly one sample dir under the single center.
    center_dir = out_root / "chapman_shaoxing"
    sample_dirs = [p for p in center_dir.iterdir() if p.is_dir()]
    assert len(sample_dirs) == 1
    sdir = sample_dirs[0]

    for f in ("original.png", "vae_roundtrip.png",
              "augmix_output.png", "compare_overlay.png"):
        assert (sdir / "plots" / f).exists(), f"missing {f}"
    # width=2 chain plots
    chain_files = list((sdir / "plots").glob("chain_*.png"))
    assert len(chain_files) == 2

    report = json.loads((sdir / "report.json").read_text())
    assert "augmix_output" in report["sanity"]
    assert len(report["augmix"]["dirichlet_weights"]) == 2
    assert 0.0 <= report["augmix"]["beta_m"] <= 1.0
    assert len(report["augmix"]["chain_op_sequences"]) == 2

    # aggregate.json should have all three versions
    agg_disk = json.loads((out_root / "aggregate.json").read_text())
    for v in ("original", "vae_roundtrip", "augmix_output"):
        assert v in agg_disk["warnings_per_version"]
