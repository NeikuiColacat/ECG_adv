"""CPU-only regressions for non-degenerate latent-hull diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

import adversarial.latent_hull_pgd as latent_hull_pgd_module
from adversarial.latent_hull_pgd import LatentHullPGDGenerator
from ecg_adv_gen.adaptation import latent_hull_torch
from ecg_adv_gen.runner import synth_online_at_super5
from ecg_adv_gen.training.losses import attack_success_stats


def test_balanced_initial_hull_has_uniform_weights_and_0p4_original_share():
    weights = latent_hull_torch.initial_hull_weights(
        1,
        20,
        weight_mode="optimized",
        init_logit_gap=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    z0 = torch.zeros(1, 4, 128)
    candidates = torch.arange(1, 21, dtype=torch.float32).view(1, 20, 1, 1).expand(-1, -1, 4, 128)
    geometry = latent_hull_torch.latent_hull_geometry(
        z0,
        candidates,
        weights,
        hull_lambda=0.6,
        epsilon=None,
        candidate_is_anchor=torch.zeros(1, 20, dtype=torch.bool),
    )

    torch.testing.assert_close(weights, torch.full((1, 20), 0.05))
    assert weights.max().item() == pytest.approx(0.05)
    assert geometry["candidate_anchor_weight"].item() == pytest.approx(0.0)
    assert geometry["effective_lambda"].item() == pytest.approx(0.6)
    assert geometry["effective_original_share"].item() == pytest.approx(0.4)


def test_projection_diagnostics_reconstruct_final_latent():
    z0 = torch.zeros(1, 1, 1)
    candidates = torch.tensor([[[[0.0]], [[2.0]]]])
    weights = torch.tensor([[0.25, 0.75]])
    geometry = latent_hull_torch.latent_hull_geometry(
        z0,
        candidates,
        weights,
        hull_lambda=0.6,
        epsilon=0.3,
        candidate_is_anchor=torch.tensor([[True, False]]),
    )

    assert geometry["pre_projection_norm"].item() == pytest.approx(0.9)
    assert geometry["post_projection_norm"].item() == pytest.approx(0.3)
    assert geometry["projection_scale"].item() == pytest.approx(1.0 / 3.0)
    assert geometry["effective_lambda"].item() == pytest.approx(0.2)
    assert geometry["candidate_anchor_weight"].item() == pytest.approx(0.25)
    assert geometry["effective_original_share"].item() == pytest.approx(0.85)
    torch.testing.assert_close(geometry["latent"], z0 + geometry["delta_post"])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_latent_hull_geometry_rejects_nonfinite_diagnostic_rows(bad_value):
    z0 = torch.zeros(2, 1, 1)
    candidates = torch.zeros(2, 2, 1, 1)
    candidates[1, 0, 0, 0] = bad_value

    with pytest.raises(
        RuntimeError,
        match=r"pre_projection_norm.*sample 1",
    ):
        latent_hull_torch.latent_hull_geometry(
            z0,
            candidates,
            torch.full((2, 2), 0.5),
            hull_lambda=0.6,
            epsilon=None,
        )


def test_candidate_anchor_weight_is_identity_based_and_sums_duplicates():
    z0 = torch.zeros(1, 1, 1)
    candidates = torch.zeros(1, 4, 1, 1)
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    geometry = latent_hull_torch.latent_hull_geometry(
        z0,
        candidates,
        weights,
        hull_lambda=0.6,
        epsilon=None,
        candidate_is_anchor=torch.tensor([[False, True, False, True]]),
    )

    assert geometry["candidate_anchor_weight"].item() == pytest.approx(0.6)
    assert geometry["effective_original_share"].item() == pytest.approx(0.76)


class _ToyVictim(nn.Module):
    num_classes = 1

    def __init__(self):
        super().__init__()
        self.model = nn.Identity()

    def forward_from_latent_to_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.flatten(1).mean(dim=1, keepdim=True)


class _ToyGenerator(LatentHullPGDGenerator):
    def _decode_to_ptbxl_1000(self, latent: torch.Tensor) -> torch.Tensor:
        value = latent.flatten(1).mean(dim=1).view(-1, 1, 1)
        return value.expand(-1, 12, 1000)


def test_generator_snapshots_initial_and_final_weights_without_changing_return_api():
    generator = _ToyGenerator(
        ecgtwin_wrapper=object(),
        victim=_ToyVictim(),
        epsilon=None,
        hull_lambda=0.6,
        hull_steps=1,
        hull_lr=0.5,
        init_logit_gap=0.0,
        device="cpu",
    )
    z0 = torch.zeros(1, 4, 128)
    labels = torch.zeros(1, 1)
    candidates = torch.stack([torch.zeros_like(z0), torch.ones_like(z0)], dim=1)

    result = generator.attack_from_latent(
        z0,
        labels,
        candidates,
        anchor_pool_indices=torch.tensor([7]),
        candidate_pool_indices=torch.tensor([[9, 7]]),
        candidate_is_anchor=torch.tensor([[False, True]]),
    )

    assert isinstance(result, tuple) and len(result) == 2
    info = generator.last_info
    np.testing.assert_allclose(info["initial_weights"], [[0.5, 0.5]])
    assert info["final_weights"] != info["initial_weights"]
    torch.testing.assert_close(generator.last_weights, torch.tensor(info["final_weights"]))
    assert info["candidate_is_anchor"] == [[False, True]]
    assert info["initial_top1_candidate_pool_indices"] == [9]
    assert generator.last_initial_signals.device.type == "cpu"
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_generator_rejects_nonfinite_diagnostics_with_field_and_sample(monkeypatch, bad_value):
    real_geometry = latent_hull_pgd_module.latent_hull_geometry

    def nonfinite_geometry(*args, **kwargs):
        geometry = real_geometry(*args, **kwargs)
        geometry["effective_original_share"] = geometry[
            "effective_original_share"
        ].clone()
        geometry["effective_original_share"][0] = bad_value
        return geometry

    monkeypatch.setattr(latent_hull_pgd_module, "latent_hull_geometry", nonfinite_geometry)
    generator = _ToyGenerator(
        ecgtwin_wrapper=object(),
        victim=_ToyVictim(),
        epsilon=None,
        hull_lambda=0.6,
        hull_steps=1,
        hull_lr=0.5,
        init_logit_gap=0.0,
        device="cpu",
    )
    z0 = torch.zeros(1, 4, 128)

    with pytest.raises(
        RuntimeError,
        match=r"initial_effective_original_share.*sample 0",
    ):
        generator.attack_from_latent(
            z0,
            torch.zeros(1, 1),
            torch.stack([torch.zeros_like(z0), torch.ones_like(z0)], dim=1),
        )


class _SignalLogitModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x[:, 0, 0]
        return torch.stack([value, -value], dim=1)


def test_three_logit_diagnostic_populates_atk_init(monkeypatch):
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def recording_stats(clean_logits, adv_logits, labels, *, margin):
        calls.append((clean_logits.clone(), adv_logits.clone()))
        return attack_success_stats(clean_logits, adv_logits, labels, margin=margin)

    monkeypatch.setattr(synth_online_at_super5, "attack_success_stats", recording_stats)
    shape = (2, 12, 4)
    clean = np.zeros(shape, dtype=np.float32)
    initial = np.ones(shape, dtype=np.float32)
    final = np.full(shape, 2.0, dtype=np.float32)
    labels = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    diagnostics = synth_online_at_super5.attack_bce_diagnostics(
        _SignalLogitModel(),
        clean,
        initial,
        final,
        labels,
        device="cpu",
        crop_len=4,
    )

    assert len(calls) == 2
    assert diagnostics["atk_init"] == pytest.approx(1.0)
    assert diagnostics["atk_anchor"] == pytest.approx(1.0)
    assert diagnostics["attack_vs_init"]["success_rate"] == pytest.approx(1.0)
    assert diagnostics["attack_vs_anchor"]["success_rate"] == pytest.approx(1.0)
    assert diagnostics["initial_bce"] < diagnostics["adv_bce"]


class _BatchIndex:
    def candidates_for(self, anchors: np.ndarray, m: int) -> np.ndarray:
        assert m == 2
        rows = []
        for anchor in anchors.tolist():
            if anchor == 0:
                rows.append([10, 0])
            elif anchor == 1:
                rows.append([1, 1])
            else:
                rows.append([12, 22])
        self.last_candidate_indices = np.asarray(rows, dtype=np.int64)
        return np.zeros((len(rows), 2, 4, 128), dtype=np.float32)


class _BatchGenerator:
    def attack_from_latent(
        self,
        z0,
        labels,
        candidate_latents,
        *,
        anchor_pool_indices,
        candidate_pool_indices,
        candidate_is_anchor,
    ):
        anchors = anchor_pool_indices.detach().cpu().tolist()
        candidates = candidate_pool_indices.detach().cpu()
        mask = candidate_is_anchor.detach().cpu()
        torch.testing.assert_close(mask, candidates == anchor_pool_indices.detach().cpu()[:, None])
        initial_weights = torch.full((len(anchors), 2), 0.5)
        final_weights = torch.tensor([[0.75, 0.25]]).expand(len(anchors), -1).clone()
        self.last_weights = final_weights
        self.last_initial_signals = torch.stack(
            [torch.full((12, 1000), float(anchor)) for anchor in anchors]
        )
        self.last_info = {
            "schema_version": 1,
            "anchor_pool_indices": anchors,
            "candidate_pool_indices": candidates.tolist(),
            "candidate_is_anchor": mask.tolist(),
            "initial_weights": initial_weights.tolist(),
            "final_weights": final_weights.tolist(),
            "initial_top1_candidate_pool_indices": candidates[:, 0].tolist(),
            "final_top1_candidate_pool_indices": candidates[:, 0].tolist(),
            "initial_top1_weights": [0.5] * len(anchors),
            "final_top1_weights": [0.75] * len(anchors),
            "hull_weight_entropy_mean": 0.5,
            "hull_weight_top1_mean": 0.75,
        }
        for prefix, offset in (("initial", 0.0), ("final", 0.1)):
            self.last_info.update({
                f"{prefix}_pre_projection_norm": [float(anchor) + 1.0 + offset for anchor in anchors],
                f"{prefix}_post_projection_norm": [0.5 + offset] * len(anchors),
                f"{prefix}_projection_scale": [0.5] * len(anchors),
                f"{prefix}_effective_lambda": [0.3] * len(anchors),
                f"{prefix}_candidate_anchor_weight": mask.float().mul(
                    initial_weights if prefix == "initial" else final_weights
                ).sum(dim=1).tolist(),
                f"{prefix}_effective_original_share": [0.7 + offset] * len(anchors),
            })
        final = torch.stack([torch.full((12, 1000), 100.0 + anchor) for anchor in anchors])
        return final, torch.zeros_like(z0)

    def _decode_to_ptbxl_1000(self, z0):
        return torch.stack([torch.full((12, 1000), float(row[0, 0])) for row in z0])


def test_runner_logs_candidate_ids_and_complete_geometry_across_batches():
    latents = np.zeros((30, 4, 128), dtype=np.float32)
    for idx in range(len(latents)):
        latents[idx, 0, 0] = idx
    labels = np.ones((30, 1), dtype=np.float32)
    generator = _BatchGenerator()

    _, _, _, stats = synth_online_at_super5.run_pgd_on_synth_pool(
        pgd_gen=generator,
        synth_latents=latents,
        synth_labels=labels,
        pgd_batch=2,
        device="cpu",
        latent_hull_index=_BatchIndex(),
        picked_indices=np.array([0, 1, 2]),
        hull_M=2,
        pool_record_ids=np.array([f"record-{idx}" for idx in range(30)]),
    )

    diagnostics = stats["latent_hull_diagnostics"]
    assert diagnostics["anchor_pool_indices"] == [0, 1, 2]
    assert diagnostics["candidate_pool_indices"] == [[10, 0], [1, 1], [12, 22]]
    assert diagnostics["candidate_is_anchor"] == [[False, True], [True, True], [False, False]]
    assert diagnostics["anchor_record_ids"] == ["record-0", "record-1", "record-2"]
    assert diagnostics["initial_weights"] == [[0.5, 0.5]] * 3
    assert diagnostics["final_weights"] == [[0.75, 0.25]] * 3
    assert diagnostics["final_pre_projection_norm"] == pytest.approx([1.1, 2.1, 3.1])
    assert diagnostics["final_effective_original_share"] == pytest.approx([0.8, 0.8, 0.8])
    assert generator.last_initial_signals_ptbxl_1000.shape == (3, 12, 1000)
    json.dumps(diagnostics, allow_nan=False)
    json.dumps(stats["latent_hull_diagnostics_summary"], allow_nan=False)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_summary_rejects_nonfinite_rows_with_field_and_sample(bad_value):
    diagnostics = {
        "schema_version": 1,
        "n": 2,
        "initial_pre_projection_norm": [1.0, bad_value],
    }

    with pytest.raises(
        RuntimeError,
        match=r"initial_pre_projection_norm.*sample 1",
    ):
        synth_online_at_super5.summarize_latent_hull_diagnostics(diagnostics)


def test_summary_rejects_nonfinite_aggregate_from_finite_rows():
    diagnostics = {
        "schema_version": 1,
        "n": 2,
        "initial_pre_projection_norm": [
            np.finfo(np.float64).max,
            np.finfo(np.float64).max,
        ],
    }

    with pytest.raises(
        RuntimeError,
        match=r"initial_pre_projection_norm.*mean.*non-finite",
    ):
        synth_online_at_super5.summarize_latent_hull_diagnostics(diagnostics)


def test_merge_rejects_batch_field_row_count_mismatch():
    batch = {
        field: [0, 1]
        for field in synth_online_at_super5._LATENT_HULL_ARRAY_FIELDS
    }
    batch["final_weights"] = [[0.5, 0.5]]

    with pytest.raises(
        RuntimeError,
        match=r"batch 0.*final_weights.*1 rows.*2 anchor rows",
    ):
        synth_online_at_super5.merge_latent_hull_batch_diagnostics([batch])
