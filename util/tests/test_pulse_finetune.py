"""CPU admission tests for native500 PULSE chain-count SFT."""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from boot_scripts.run_experiment import load_experiment_plan
from core.augmix import native500_augmix_one
from core.pulse_finetune import MODEL_HASHES, MODEL_CONFIG_HASHES, answer_only_loss, sample_schedule
from util.augmentations.profile import load_augmentation_profile
from util.pulse_training_contract import label_text, validate_config

REPO = Path(__file__).resolve().parents[2]


def config():
    return yaml.safe_load((REPO / "configs/train/pulse_augmix_single_smoke.yaml").read_text())


def test_config_chain_pair_differs_only_in_width():
    single = config()
    two = yaml.safe_load((REPO / "configs/train/pulse_augmix_two_smoke.yaml").read_text())
    validate_config(single)
    validate_config(two)
    single["width"] = 2
    assert single == two


def test_resume_config_preserves_the_uninterrupted_training_protocol():
    original = config()
    resumed = yaml.safe_load((REPO / "configs/train/pulse_augmix_resume_smoke.yaml").read_text())
    validate_config(resumed)
    assert resumed.pop("resume_from").endswith("/training/resume_probe.pt")
    original.pop("resume_from")
    assert resumed == original


def test_formal_center_arms_have_identical_update_and_exposure_budgets():
    reference = None
    for center in ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"):
        for width in (1, 2):
            path = REPO / f"configs/train/pulse_augmix_w{width}_{center}.yaml"
            payload = yaml.safe_load(path.read_text())
            validate_config(payload)
            assert payload.pop("center") == center
            assert payload.pop("width") == width
            assert payload["mode"] == "train" and payload["resume_from"] is None
            assert payload["training"]["optimizer_steps"] == 100
            assert payload["training"]["effective_batch"] == 16
            reference = reference or payload
            assert payload == reference


def test_source_model_hashes_are_sha256_values():
    assert len(MODEL_HASHES) == 3
    assert len(MODEL_CONFIG_HASHES) == 6
    assert all(len(value) == 64 and set(value) <= set("0123456789abcdef")
               for value in {**MODEL_HASHES, **MODEL_CONFIG_HASHES}.values())


@pytest.mark.parametrize("change", [
    {"width": 3}, {"center": "ptbxl"}, {"mode": "jsd"}, {"extra": True},
])
def test_invalid_training_contract_is_rejected(change):
    payload = config()
    payload.update(change)
    with pytest.raises(ValueError):
        validate_config(payload)


def test_label_text_preserves_super5_and_rejects_zero():
    assert label_text([1, 1, 0, 0, 1]) == "CD;HYP;STTC"
    with pytest.raises(ValueError):
        label_text([0] * 5)


def test_answer_only_logits_matches_full_causal_loss_and_gradients():
    torch.manual_seed(421)
    hidden = torch.randn(2, 13, 8, requires_grad=True)
    head = torch.nn.Linear(8, 19)
    labels = torch.full((2, 13), -100, dtype=torch.long)
    labels[0, -3:] = torch.tensor([1, 4, 8])
    labels[1, -2:] = torch.tensor([2, 7])
    reference = F.cross_entropy(head(hidden)[:, :-1].reshape(-1, 19).float(), labels[:, 1:].reshape(-1))
    reference_grads = torch.autograd.grad(reference, (hidden, head.weight, head.bias))
    optimized = answer_only_loss(hidden, labels, head)
    optimized_grads = torch.autograd.grad(optimized, (hidden, head.weight, head.bias))
    torch.testing.assert_close(reference, optimized, atol=1e-7, rtol=1e-6)
    for expected, actual in zip(reference_grads, optimized_grads, strict=True):
        torch.testing.assert_close(expected, actual, atol=1e-7, rtol=1e-6)


def test_native500_pair_preserves_input_and_global_rng_and_first_chain(monkeypatch):
    profile = load_augmentation_profile(REPO / "configs/augmentation/operators.yaml")
    recorded = []
    def operator(name, waveform, *, params, sampling_rate_hz, rng):
        assert sampling_rate_hz == 500
        result = waveform + torch.rand(1, generator=rng)
        recorded.append(result.clone())
        return result
    monkeypatch.setattr("util.augmentations.torch_operators.apply_operator_batch_prevalidated", operator)
    wave = torch.zeros(1, 5000, 12)
    rng_before = torch.get_rng_state().clone()
    single, trace1 = native500_augmix_one(wave, width=1, profile=profile, seed=1, identity="same")
    first_calls = len(recorded)
    recorded.clear()
    two, trace2 = native500_augmix_one(wave, width=2, profile=profile, seed=1, identity="same")
    assert torch.equal(recorded[first_calls - 1], single)
    assert trace1["compositions"][0] == trace2["compositions"][0]
    expected = trace2["weights"][0] * single + trace2["weights"][1] * recorded[-1]
    assert torch.equal(two, expected)
    assert abs(sum(trace2["weights"]) - 1) < 1e-6
    assert torch.equal(wave, torch.zeros_like(wave))
    assert torch.equal(torch.get_rng_state(), rng_before)
    replay, replay_trace = native500_augmix_one(wave, width=2, profile=profile, seed=1, identity="same")
    assert torch.equal(two, replay) and trace2 == replay_trace


def test_native500_rejects_canonical100():
    with pytest.raises(ValueError, match="native500"):
        native500_augmix_one(torch.zeros(1, 1000, 12), width=1, profile=None, seed=1, identity="x")


def test_exposure_schedule_is_balanced_and_independent_of_torch_rng():
    before = torch.get_rng_state().clone()
    schedule = sample_schedule(500, 1600, 20260909)
    assert len(schedule) == 1600
    assert set(schedule[:500]) == set(range(500))
    assert schedule == sample_schedule(500, 1600, 20260909)
    assert torch.equal(before, torch.get_rng_state())


def test_pulse_yaml_dry_run_is_non_mutating(tmp_path):
    configs = sorted((REPO / "configs/experiments").glob("pulse_augmix_*.yaml"))
    assert len(configs) == 17
    for path in configs:
        plan = load_experiment_plan(path, run_dir=tmp_path / path.stem)
        assert plan.expected_result_type == "pulse_train_result"
        assert plan.data_ledger.required_roots == ("pn2021_cache", "split_artifacts")
        description = plan.describe()
        assert not description["loads_model_or_gpu"] and not description["loads_data"]
        assert not plan.run_dir.exists()
