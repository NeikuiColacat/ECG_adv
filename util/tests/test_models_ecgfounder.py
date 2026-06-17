"""Tests for CPU-only ECGFounder model contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from ecg_adv_gen.models import (
    ecgfounder_feature_cache_path,
    ecgfounder_feature_cache_paths,
    ecgfounder_k500_head_path,
    ecgfounder_kshot_head_run_dir,
    ecgfounder_linear_probe_head_path,
    ecgfounder_lhat_run_dir,
)
from ecg_adv_gen.models.ecgfounder_heads import (
    FeatureAdapterHead,
    ResidualAdapterHead,
    clone_linear_head,
    init_dense_from_head_path,
    unwrap_linear_head,
)
from ecg_adv_gen.models.ecgfounder_inference import (
    evaluate_feature_head,
    evaluate_pn2021_feature_head,
    evaluate_signal_split,
    evaluate_ptbxl_fold_head,
    predict_feature_head,
    predict_signal_dataset,
    sigmoid_clipped,
)
from ecg_adv_gen.models.ecgfounder_torch import (
    ecg1000_to_ecgfounder_input,
    fft_bandpass_torch,
    global_zscore_torch,
    repair_flat_ecg_leads_torch,
)
from ecg_adv_gen.training import CachedSignalDataset


def test_ecgfounder_linear_probe_feature_paths():
    root = Path("/tmp/linear_probe")
    assert ecgfounder_linear_probe_head_path(root) == root / "best_head.pt"
    assert ecgfounder_feature_cache_path(
        root,
        dataset="pn2021",
        preprocess_policy="official_ptbxl_eval",
    ) == root / "pn2021_ecgfounder_features_official_ptbxl_eval.npz"
    paths = ecgfounder_feature_cache_paths(root, preprocess_policy="official_ptbxl_eval")
    assert paths["ptbxl"] == root / "ptbxl_ecgfounder_features_official_ptbxl_eval.npz"
    assert paths["pn2021"] == root / "pn2021_ecgfounder_features_official_ptbxl_eval.npz"


def test_ecgfounder_feature_path_rejects_unknown_inputs():
    with pytest.raises(ValueError, match="Unsupported ECGFounder feature dataset"):
        ecgfounder_feature_cache_path(
            "/tmp/linear_probe",
            dataset="mimic",  # type: ignore[arg-type]
            preprocess_policy="official_ptbxl_eval",
        )
    with pytest.raises(ValueError, match="Unsupported ECGFounder preprocess policy"):
        ecgfounder_feature_cache_path(
            "/tmp/linear_probe",
            dataset="pn2021",
            preprocess_policy="bad_policy",
        )


def test_ecgfounder_run_dir_contracts():
    assert ecgfounder_kshot_head_run_dir(
        "/tmp/out",
        center="ningbo",
        k=500,
        source_k=500,
        epochs=50,
        seed=20260531,
    ) == Path("/tmp/out/runs/ningbo_K500_fromK500_headft_ep50_seed20260531")
    assert ecgfounder_lhat_run_dir(
        "/tmp/out",
        center="georgia",
        k=500,
        hull_m=20,
        hull_lambda=0.15,
        epochs=20,
        seed=20260531,
    ) == Path("/tmp/out/runs/georgia_K500_M20_lam0p15_ep20_seed20260531")


def test_ecgfounder_k500_head_path_uses_exact_match_or_fallback(tmp_path: Path):
    expected = (
        tmp_path
        / "ningbo_K500_fromK500_headft_ep50_seed20260531"
        / "best_head.pt"
    )
    assert ecgfounder_k500_head_path(tmp_path, center="ningbo", k=500, seed=20260531) == expected

    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"head")
    assert ecgfounder_k500_head_path(tmp_path, center="ningbo", k=500, seed=20260531) == expected


def test_residual_adapter_head_starts_as_base_head_and_freezes_base():
    torch.manual_seed(0)
    base = nn.Linear(4, 2)
    head = ResidualAdapterHead(base, hidden_dim=3, dropout=0.0, scale=0.7, freeze_base=True)
    x = torch.randn(5, 4)

    torch.testing.assert_close(head(x), base(x))
    assert all(not p.requires_grad for p in head.base_head.parameters())
    assert unwrap_linear_head(head) is base


def test_feature_adapter_head_starts_as_base_head_and_can_keep_base_trainable():
    torch.manual_seed(1)
    base = nn.Linear(4, 2)
    head = FeatureAdapterHead(base, hidden_dim=3, dropout=0.0, scale=0.5, freeze_base=False)
    x = torch.randn(5, 4)

    torch.testing.assert_close(head(x), base(x))
    assert all(p.requires_grad for p in head.base_head.parameters())


def test_clone_linear_head_unwraps_adapter_and_freezes_copy():
    torch.manual_seed(2)
    base = nn.Linear(4, 2)
    adapter = ResidualAdapterHead(base, hidden_dim=3)

    clone = clone_linear_head(adapter, feature_dim=4, device="cpu")

    assert clone is not base
    torch.testing.assert_close(clone.weight, base.weight)
    torch.testing.assert_close(clone.bias, base.bias)
    assert all(not p.requires_grad for p in clone.parameters())
    with pytest.raises(TypeError, match="expected nn.Linear"):
        unwrap_linear_head(nn.Sequential(nn.Linear(4, 2)))


def test_init_dense_from_head_path_loads_linear_state(tmp_path: Path):
    model = nn.Module()
    model.dense = nn.Linear(3, 2)
    weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = torch.tensor([0.25, -0.5])
    head_path = tmp_path / "best_head.pt"
    torch.save({"weight": weight, "bias": bias}, head_path)

    info = init_dense_from_head_path(model, head_path)

    torch.testing.assert_close(model.dense.weight, weight)
    torch.testing.assert_close(model.dense.bias, bias)
    assert info == {
        "path": str(head_path),
        "weight_shape": [2, 3],
        "bias_shape": [2],
    }


def test_init_dense_from_head_path_rejects_bad_state_and_model(tmp_path: Path):
    head_path = tmp_path / "bad_head.pt"
    torch.save({"weight": torch.zeros((2, 3))}, head_path)
    model = nn.Module()
    model.dense = nn.Linear(3, 2)
    with pytest.raises(RuntimeError, match="weight/bias"):
        init_dense_from_head_path(model, head_path)

    good_path = tmp_path / "good_head.pt"
    torch.save({"weight": torch.zeros((2, 3)), "bias": torch.zeros((2,))}, good_path)
    with pytest.raises(RuntimeError, match="no dense attribute"):
        init_dense_from_head_path(nn.Module(), good_path)

    model.dense = nn.Sequential(nn.Linear(3, 2))
    with pytest.raises(RuntimeError, match="expected nn.Linear"):
        init_dense_from_head_path(model, good_path)

    model.dense = nn.Linear(4, 2)
    with pytest.raises(RuntimeError, match="head shape mismatch"):
        init_dense_from_head_path(model, good_path)


def test_ecgfounder_torch_input_conversion_resamples_and_normalizes():
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    z = global_zscore_torch(x)
    flat = x.reshape(2, -1)
    expected = (x - flat.mean(dim=1).view(2, 1, 1)) / flat.std(dim=1).view(2, 1, 1)

    torch.testing.assert_close(z, expected)
    y = ecg1000_to_ecgfounder_input(x, target_points=8)
    assert tuple(y.shape) == (2, 3, 8)
    torch.testing.assert_close(y.reshape(2, -1).mean(dim=1), torch.zeros(2), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(y.reshape(2, -1).std(dim=1), torch.ones(2), atol=1e-6, rtol=0.0)


def test_fft_bandpass_torch_suppresses_out_of_band_power():
    sample_rate_hz = 100.0
    t = torch.arange(1000, dtype=torch.float32) / sample_rate_hz
    low = torch.sin(2 * torch.pi * 5.0 * t)
    high = 0.8 * torch.sin(2 * torch.pi * 40.0 * t)
    x = (low + high).reshape(1, 1, -1).repeat(1, 12, 1)

    y = fft_bandpass_torch(x, sample_rate_hz=sample_rate_hz, low_hz=0.5, high_hz=35.0)

    freq = torch.fft.rfftfreq(y.shape[-1], d=1.0 / sample_rate_hz)
    spectrum = torch.fft.rfft(y[0, 0])
    amp_5 = spectrum[torch.argmin(torch.abs(freq - 5.0))].abs()
    amp_40 = spectrum[torch.argmin(torch.abs(freq - 40.0))].abs()
    assert amp_5 > 100 * amp_40


def test_repair_flat_ecg_leads_recovers_limb_relation_when_anchors_present():
    t = torch.linspace(0.0, 1.0, 16)
    lead_i = torch.sin(2 * torch.pi * t)
    lead_ii = torch.cos(2 * torch.pi * t)
    x = torch.zeros((1, 12, 16), dtype=torch.float32)
    x[:, 0] = lead_i
    x[:, 1] = lead_ii
    x[:, 2] = 0.0

    repaired = repair_flat_ecg_leads_torch(x)

    torch.testing.assert_close(repaired[0, 2], lead_ii - lead_i, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(repaired[0, 0], lead_i, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(repaired[0, 1], lead_ii, atol=1e-6, rtol=0.0)


def test_ecg1000_to_ecgfounder_input_accepts_input_stabilizer_options():
    sample_rate_hz = 100.0
    t = torch.arange(1000, dtype=torch.float32) / sample_rate_hz
    low = torch.sin(2 * torch.pi * 5.0 * t)
    high = 0.8 * torch.sin(2 * torch.pi * 40.0 * t)
    x = (low + high).reshape(1, 1, -1).repeat(1, 12, 1)

    y = ecg1000_to_ecgfounder_input(
        x,
        target_points=1000,
        bandpass_low_hz=0.5,
        bandpass_high_hz=35.0,
        input_sample_rate_hz=sample_rate_hz,
    )

    assert tuple(y.shape) == (1, 12, 1000)
    expected = global_zscore_torch(low.reshape(1, 1, -1).repeat(1, 12, 1))
    corr = torch.corrcoef(torch.stack([y[0, 0], expected[0, 0]]))[0, 1]
    assert float(corr) > 0.99


def test_predict_feature_head_matches_clipped_sigmoid():
    head = nn.Linear(3, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]))
        head.bias.copy_(torch.tensor([0.25, -0.5]))
    features = np.asarray([[2.0, 0.0, 1.0], [100.0, -100.0, 0.0]], dtype=np.float32)

    scores = predict_feature_head(head, features, batch_size=1, device="cpu")
    logits = head(torch.from_numpy(features)).detach().numpy()

    np.testing.assert_allclose(scores, sigmoid_clipped(logits), rtol=1e-6)
    assert predict_feature_head(head, np.empty((0, 3), dtype=np.float32), batch_size=2, device="cpu").shape == (0, 0)
    with pytest.raises(ValueError, match="2D feature matrix"):
        predict_feature_head(head, np.zeros((2, 3, 1), dtype=np.float32), batch_size=2, device="cpu")


def test_evaluate_feature_head_and_ptbxl_fold_use_metric_callback():
    head = nn.Linear(2, 1)
    with torch.no_grad():
        head.weight.fill_(1.0)
        head.bias.zero_()
    labels = np.asarray([[0.0], [1.0], [1.0]], dtype=np.float32)
    features = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    def metric_fn(y_true: np.ndarray, scores: np.ndarray, min_pos: int) -> dict:
        return {"n": len(y_true), "min_pos": min_pos, "score_mean": float(scores.mean())}

    direct = evaluate_feature_head(
        head,
        features,
        labels,
        metric_fn,
        batch_size=2,
        device="cpu",
        min_pos=2,
    )
    assert direct["n"] == 3
    assert direct["min_pos"] == 2
    assert 0.0 < direct["score_mean"] < 1.0

    payload = {
        "features": features,
        "labels": labels,
        "folds": np.asarray([9, 10, 10], dtype=np.int64),
    }
    fold = evaluate_ptbxl_fold_head(
        head,
        payload,
        metric_fn,
        fold=10,
        batch_size=2,
        device="cpu",
    )
    assert fold["n"] == 2
    assert fold["min_pos"] == 1


def test_evaluate_pn2021_feature_head_uses_view_callback():
    head = nn.Linear(2, 1)
    with torch.no_grad():
        head.weight.fill_(1.0)
        head.bias.zero_()
    payload = {
        "features": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "labels": np.asarray([[0.0], [1.0]], dtype=np.float32),
        "centers": np.asarray(["ningbo", "georgia"]),
        "record_ids": np.asarray(["a", "b"]),
    }

    def view_fn(labels, scores, centers, record_ids, ref_ids, report_drop_all_zero):
        return {
            "n": int(len(labels)),
            "centers": centers.tolist(),
            "record_ids": record_ids.tolist(),
            "ref_ids": {k: sorted(v) for k, v in ref_ids.items()},
            "drop": bool(report_drop_all_zero),
            "score_shape": tuple(scores.shape),
        }

    out = evaluate_pn2021_feature_head(
        head,
        payload,
        {"ningbo": {"a"}},
        view_fn,
        batch_size=1,
        device="cpu",
        report_drop_all_zero=True,
    )

    assert out["n"] == 2
    assert out["centers"] == ["ningbo", "georgia"]
    assert out["record_ids"] == ["a", "b"]
    assert out["ref_ids"] == {"ningbo": ["a"]}
    assert out["drop"] is True
    assert out["score_shape"] == (2, 1)


def test_predict_signal_dataset_and_evaluate_signal_split_use_metric_callback():
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -1.0]]))
        model.bias.copy_(torch.tensor([0.25]))
    signals = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]],
        dtype=np.float32,
    )
    labels = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    indices = np.asarray([3, 1], dtype=np.int64)
    ds = CachedSignalDataset(signals, labels, indices)

    y_true, scores = predict_signal_dataset(model, ds, batch_size=1, device="cpu")
    expected_logits = model(torch.from_numpy(signals[indices])).detach()

    np.testing.assert_allclose(y_true, labels[indices])
    np.testing.assert_allclose(scores, torch.sigmoid(expected_logits).numpy(), rtol=1e-6)

    def metric_fn(y: np.ndarray, p: np.ndarray, min_pos: int) -> dict:
        return {"n": len(y), "min_pos": min_pos, "score_sum": float(p.sum())}

    out = evaluate_signal_split(
        model,
        signals,
        labels,
        indices,
        metric_fn,
        batch_size=2,
        device="cpu",
        min_pos=2,
    )
    assert out["n"] == 2
    assert out["min_pos"] == 2
    assert out["score_sum"] == pytest.approx(float(scores.sum()))
