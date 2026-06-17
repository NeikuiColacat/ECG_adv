from __future__ import annotations

from pathlib import Path

import torch

from scripts.paper import select_effnet_robust_checkpoint as selector
from scripts.paper.select_effnet_robust_checkpoint import (
    CandidateScore,
    discover_checkpoint_candidates,
    select_best_candidate,
)


def test_discover_checkpoint_candidates_includes_selected_and_latest(tmp_path: Path):
    selected = tmp_path / "best_model.pt"
    selected.write_bytes(b"selected")
    latest_dir = tmp_path / "checkpoints"
    latest_dir.mkdir()
    torch.save({"model_state_dict": {"weight": torch.ones(1)}, "epoch": 30}, latest_dir / "checkpoint_latest.pt")

    candidates = discover_checkpoint_candidates(tmp_path)

    assert [candidate.name for candidate in candidates] == ["selected", "latest"]
    assert candidates[0].path == selected
    assert candidates[1].epoch == 30


def test_select_best_candidate_uses_corrupted_score_after_clean_floor():
    selected = CandidateScore(
        name="selected",
        clean_macro_auprc=0.78,
        corrupted_macro_auprc=0.40,
        clean_macro_auroc=0.90,
        corrupted_macro_auroc=0.80,
        n_val=100,
    )
    latest = CandidateScore(
        name="latest",
        clean_macro_auprc=0.74,
        corrupted_macro_auprc=0.55,
        clean_macro_auroc=0.89,
        corrupted_macro_auroc=0.82,
        n_val=100,
    )
    bad_clean = CandidateScore(
        name="bad_clean",
        clean_macro_auprc=0.50,
        corrupted_macro_auprc=0.70,
        clean_macro_auroc=0.80,
        corrupted_macro_auroc=0.83,
        n_val=100,
    )

    winner = select_best_candidate(
        [selected, latest, bad_clean],
        clean_auprc_floor=0.70,
        clean_auroc_floor=0.85,
    )

    assert winner.name == "latest"


def test_prepare_eval_signals_applies_input_stabilizer_at_requested_stage(monkeypatch):
    signals_tc = torch.arange(12, dtype=torch.float32).view(1, 6, 2).numpy()
    calls: list[tuple[tuple[int, ...], dict]] = []

    def fake_stabilizer(ecg_ct, config):
        assert isinstance(ecg_ct, torch.Tensor)
        calls.append((tuple(ecg_ct.shape), dict(config)))
        return ecg_ct + 100.0

    monkeypatch.setattr(selector, "apply_effnet_input_stabilizer", fake_stabilizer)

    post_crop = selector._prepare_eval_signals_ct(
        signals_tc,
        crop_len=4,
        input_stabilizer_config={"bandpass_high_hz": 35.0, "stage": "post_crop"},
    )
    pre_crop = selector._prepare_eval_signals_ct(
        signals_tc,
        crop_len=4,
        input_stabilizer_config={"bandpass_high_hz": 35.0, "stage": "pre_crop"},
    )

    assert calls[0][0] == (2, 4)
    assert calls[1][0] == (2, 6)
    assert post_crop.shape == (1, 2, 4)
    assert pre_crop.shape == (1, 2, 4)
    assert float(post_crop.min()) >= 100.0
    assert float(pre_crop.min()) >= 100.0
