"""CPU-only tests for ECGFounder fullFT raw-AugMix plumbing."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import scripts.paper.run_ecgfounder_fullft_super5_pilot_20260523 as fullft


class _TinyFullFT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(12 * 1000, 8)
        self.dense = nn.Linear(8, 5)
        self.return_features = False

    def forward(self, x: torch.Tensor):
        features = torch.relu(self.encoder(x.reshape(x.shape[0], -1)))
        logits = self.dense(features)
        if self.return_features:
            return logits, features
        return logits


class _NaNFullFT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 5), float("nan"), device=x.device) * self.weight


class _TrackingTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((5,), 0.25), requires_grad=False)
        self.calls = 0
        self.last_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_shape = tuple(x.shape)
        return self.bias.unsqueeze(0).expand(x.shape[0], -1)


def test_fullft_parser_accepts_calibrated_raw_augmix_flags():
    parser = fullft.build_arg_parser()

    args = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--enable_raw_corrupt_consistency",
            "--raw_corrupt_scope",
            "source_target",
            "--raw_corrupt_severity",
            "5",
            "--raw_corrupt_severity_profile",
            "calibrated_10to20pp",
            "--raw_corrupt_view_mode",
            "augmix",
            "--raw_corrupt_augmix_width",
            "1",
            "--raw_corrupt_augmix_depth",
            "1",
            "--raw_corrupt_augmix_mixture_mode",
            "fixed",
            "--raw_corrupt_augmix_mixture_prob",
            "1.0",
            "--raw_corrupt_no_renorm",
        ]
    )

    assert args.enable_raw_corrupt_consistency is True
    assert args.raw_corrupt_scope == "source_target"
    assert args.raw_corrupt_severity_profile == "calibrated_10to20pp"
    assert args.raw_corrupt_view_mode == "augmix"
    assert args.raw_corrupt_augmix_width == 1
    assert args.raw_corrupt_augmix_depth == 1
    assert args.raw_corrupt_augmix_mixture_mode == "fixed"
    assert args.raw_corrupt_augmix_mixture_prob == 1.0
    assert args.raw_corrupt_no_renorm is True


def test_fullft_parser_accepts_aux_raw_corruption_branch_flags():
    parser = fullft.build_arg_parser()

    args = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--enable_raw_corrupt_consistency",
            "--raw_corrupt_ops",
            "powerline_noise",
            "emg_noise",
            "random_leads_masking",
            "--enable_raw_corrupt_aux_consistency",
            "--raw_corrupt_aux_ops",
            "baseline_wander",
            "baseline_shift",
            "--raw_corrupt_aux_consistency_weight",
            "0.25",
            "--raw_corrupt_aux_consistency_loss",
            "soft_bce",
            "--raw_corrupt_aux_bce_weight",
            "0.0",
            "--raw_corrupt_aux_max_batches",
            "16",
        ]
    )

    assert args.enable_raw_corrupt_aux_consistency is True
    assert args.raw_corrupt_aux_ops == ["baseline_wander", "baseline_shift"]
    assert args.raw_corrupt_aux_consistency_weight == 0.25
    assert args.raw_corrupt_aux_consistency_loss == "soft_bce"
    assert args.raw_corrupt_aux_bce_weight == 0.0
    assert args.raw_corrupt_aux_max_batches == 16


def test_fullft_parser_accepts_aux_raw_corruption_teacher_flags():
    parser = fullft.build_arg_parser()

    args = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--enable_raw_corrupt_aux_consistency",
            "--raw_corrupt_aux_ops",
            "baseline_wander",
            "baseline_shift",
            "--raw_corrupt_aux_teacher_type",
            "efficientnet1dv2",
            "--raw_corrupt_aux_teacher_model_name",
            "efficientnet1dv2",
            "--raw_corrupt_aux_teacher_model_path",
            "/tmp/teacher.pt",
            "--raw_corrupt_aux_teacher_weight",
            "0.75",
            "--raw_corrupt_aux_teacher_loss",
            "mse_logits",
            "--raw_corrupt_aux_teacher_view",
            "corrupt",
        ]
    )

    assert args.raw_corrupt_aux_teacher_type == "efficientnet1dv2"
    assert args.raw_corrupt_aux_teacher_model_name == "efficientnet1dv2"
    assert args.raw_corrupt_aux_teacher_model_path == "/tmp/teacher.pt"
    assert args.raw_corrupt_aux_teacher_weight == 0.75
    assert args.raw_corrupt_aux_teacher_loss == "mse_logits"
    assert args.raw_corrupt_aux_teacher_view == "corrupt"


def test_prepare_raw_corruption_teacher_input_uses_raw1000_for_effnet_teacher(monkeypatch):
    teacher = _TrackingTeacher()
    teacher.raw_corruption_input_mode = "raw1000"
    raw = torch.zeros((3, 12, 1000), dtype=torch.float32)

    def fail_if_called(x):
        raise AssertionError("ECGFounder conversion should not run for raw1000 teachers")

    monkeypatch.setattr(fullft, "ecg1000_to_ecgfounder_input", fail_if_called)

    prepared = fullft.prepare_raw_corruption_teacher_input(teacher, raw)

    assert prepared is raw


def test_prepare_raw_corruption_teacher_input_defaults_to_ecgfounder_conversion(monkeypatch):
    teacher = _TrackingTeacher()
    raw = torch.zeros((3, 12, 1000), dtype=torch.float32)

    def fake_convert(x):
        return x + 1.0

    monkeypatch.setattr(fullft, "ecg1000_to_ecgfounder_input", fake_convert)

    prepared = fullft.prepare_raw_corruption_teacher_input(teacher, raw)

    assert torch.allclose(prepared, raw + 1.0)


def test_load_frozen_raw_corruption_teacher_builds_effnet_raw_teacher(tmp_path, monkeypatch):
    class TinyTeacher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(12 * 1000, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x.reshape(x.shape[0], -1))

    built = TinyTeacher()
    ckpt_path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": built.state_dict()}, ckpt_path)

    monkeypatch.setattr(fullft, "build_super5_model", lambda *args, **kwargs: TinyTeacher())

    teacher = fullft.load_frozen_raw_corruption_teacher(
        ckpt_path,
        torch.device("cpu"),
        teacher_type="efficientnet1dv2",
        teacher_model_name="efficientnet1dv2",
    )

    assert getattr(teacher, "raw_corruption_input_mode") == "raw1000"
    assert teacher.training is False
    assert all(not param.requires_grad for param in teacher.parameters())
    logits = teacher(torch.zeros((2, 12, 1000), dtype=torch.float32))
    assert logits.shape == (2, 5)


def test_load_frozen_raw_corruption_teacher_builds_ecgfounder_feature_head_teacher(tmp_path, monkeypatch):
    class TinyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = nn.Linear(8, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = x.reshape(x.shape[0], -1)[:, :8]
            return self.dense(features)

    head_path = tmp_path / "best_head.pt"
    torch.save(
        {
            "weight": torch.ones((5, 8), dtype=torch.float32),
            "bias": torch.arange(5, dtype=torch.float32),
        },
        head_path,
    )

    monkeypatch.setattr(fullft, "ft_12lead_ECGFounder", lambda *args, **kwargs: TinyBackbone())

    teacher = fullft.load_frozen_raw_corruption_teacher(
        head_path,
        torch.device("cpu"),
        teacher_type="ecgfounder_feature_head",
        teacher_model_name="unused",
    )

    assert getattr(teacher, "raw_corruption_input_mode") == "ecgfounder"
    assert teacher.training is False
    assert all(not param.requires_grad for param in teacher.parameters())
    logits = teacher(torch.ones((2, 12, 5000), dtype=torch.float32))
    assert logits.shape == (2, 5)
    torch.testing.assert_close(logits[0], torch.full((5,), 8.0) + torch.arange(5, dtype=torch.float32))


def test_fullft_parser_accepts_raw_feature_consistency_flags():
    parser = fullft.build_arg_parser()

    args = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--enable_raw_corrupt_consistency",
            "--raw_corrupt_feature_consistency_weight",
            "0.15",
            "--enable_raw_corrupt_aux_consistency",
            "--raw_corrupt_aux_ops",
            "baseline_wander",
            "baseline_shift",
            "--raw_corrupt_aux_feature_consistency_weight",
            "0.4",
            "--raw_corrupt_feature_consistency_normalize",
        ]
    )

    assert args.raw_corrupt_feature_consistency_weight == 0.15
    assert args.raw_corrupt_aux_feature_consistency_weight == 0.4
    assert args.raw_corrupt_feature_consistency_normalize is True


def test_fullft_parser_accepts_raw_corruption_op_conditioning_flags():
    parser = fullft.build_arg_parser()

    args = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--enable_raw_corrupt_consistency",
            "--raw_corrupt_op_conditioning",
            "explicit_adapter",
            "--raw_corrupt_op_adapter_hidden",
            "32",
            "--raw_corrupt_op_adapter_dropout",
            "0.1",
            "--raw_corrupt_op_adapter_scale",
            "0.5",
        ]
    )

    assert args.raw_corrupt_op_conditioning == "explicit_adapter"
    assert args.raw_corrupt_op_adapter_hidden == 32
    assert args.raw_corrupt_op_adapter_dropout == 0.1
    assert args.raw_corrupt_op_adapter_scale == 0.5


def test_fullft_parser_accepts_input_stabilizer_flags():
    parser = fullft.build_arg_parser()

    args = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--ecgfounder_input_bandpass_low_hz",
            "0.5",
            "--ecgfounder_input_bandpass_high_hz",
            "35.0",
            "--ecgfounder_input_repair_flat_leads",
        ]
    )

    assert args.ecgfounder_input_bandpass_low_hz == 0.5
    assert args.ecgfounder_input_bandpass_high_hz == 35.0
    assert args.ecgfounder_input_repair_flat_leads is True


def test_input_stabilizer_kwargs_from_args_omits_disabled_values():
    parser = fullft.build_arg_parser()
    disabled = parser.parse_args(["--ref_meta_json", "/tmp/ref.json"])
    assert fullft.input_stabilizer_kwargs_from_args(disabled) == {}

    enabled = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--ecgfounder_input_bandpass_low_hz",
            "0.5",
            "--ecgfounder_input_bandpass_high_hz",
            "35.0",
            "--ecgfounder_input_repair_flat_leads",
        ]
    )

    assert fullft.input_stabilizer_kwargs_from_args(enabled) == {
        "bandpass_low_hz": 0.5,
        "bandpass_high_hz": 35.0,
        "repair_flat_leads": True,
    }


def test_fullft_parser_defaults_to_cached5000_supervised_input_mode():
    parser = fullft.build_arg_parser()
    args = parser.parse_args(["--ref_meta_json", "/tmp/ref.json"])
    assert args.supervised_input_mode == "cached5000"


def test_fullft_parser_accepts_raw1000_supervised_input_mode():
    parser = fullft.build_arg_parser()
    args = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--supervised_input_mode",
            "raw1000",
            "--source_raw1000_signal_cache",
            "/tmp/source.npy",
            "--source_raw1000_label_cache",
            "/tmp/labels.npy",
        ]
    )
    assert args.supervised_input_mode == "raw1000"
    assert args.source_raw1000_signal_cache == "/tmp/source.npy"
    assert args.source_raw1000_label_cache == "/tmp/labels.npy"


def test_fullft_parser_accepts_source_cached_target_raw1000_hybrid_mode():
    parser = fullft.build_arg_parser()
    args = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--supervised_input_mode",
            "source_cached_target_raw1000",
        ]
    )
    assert args.supervised_input_mode == "source_cached_target_raw1000"


def test_prepare_supervised_ecgfounder_input_dispatches_by_mode(monkeypatch):
    raw = torch.zeros((2, 12, 1000), dtype=torch.float32)
    cached = torch.zeros((2, 12, 5000), dtype=torch.float32)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_raw(x, kwargs):
        calls.append(("raw", dict(kwargs)))
        return x + 1.0

    def fake_cached(x, kwargs):
        calls.append(("cached", dict(kwargs)))
        return x + 2.0

    monkeypatch.setattr(fullft, "prepare_raw_ecgfounder_input", fake_raw)
    monkeypatch.setattr(fullft, "prepare_cached_ecgfounder_input", fake_cached)

    stabilizer = {"bandpass_low_hz": 0.5, "bandpass_high_hz": 35.0}
    raw_out = fullft.prepare_supervised_ecgfounder_input(raw, "raw1000", stabilizer)
    cached_out = fullft.prepare_supervised_ecgfounder_input(cached, "cached5000", stabilizer)

    torch.testing.assert_close(raw_out, raw + 1.0)
    torch.testing.assert_close(cached_out, cached + 2.0)
    assert calls == [("raw", stabilizer), ("cached", stabilizer)]


def test_prepare_supervised_ecgfounder_input_hybrid_filters_source_only(monkeypatch):
    x = torch.zeros((3, 12, 5000), dtype=torch.float32)
    stream = torch.tensor([0, 1, 2], dtype=torch.long)
    calls: list[tuple[int, dict[str, object]]] = []

    def fake_cached(batch, kwargs):
        calls.append((int(batch.shape[0]), dict(kwargs)))
        return batch + 2.0

    def fail_raw(batch, kwargs):
        raise AssertionError("hybrid supervised batches should already contain prepared target/adv tensors")

    monkeypatch.setattr(fullft, "prepare_cached_ecgfounder_input", fake_cached)
    monkeypatch.setattr(fullft, "prepare_raw_ecgfounder_input", fail_raw)

    stabilizer = {"bandpass_low_hz": 0.5}
    out = fullft.prepare_supervised_ecgfounder_input(
        x,
        "source_cached_target_raw1000",
        stabilizer,
        stream=stream,
    )

    torch.testing.assert_close(out[0], x[0] + 2.0)
    torch.testing.assert_close(out[1:], x[1:])
    assert calls == [(1, stabilizer)]


def test_prepare_adv_stream_signal_matches_supervised_mode(monkeypatch):
    raw = torch.zeros((2, 12, 1000), dtype=torch.float32)
    captured: list[dict[str, object]] = []

    def fake_raw(x, kwargs):
        captured.append(dict(kwargs))
        return x + 5.0

    monkeypatch.setattr(fullft, "prepare_raw_ecgfounder_input", fake_raw)

    stabilizer = {"bandpass_low_hz": 0.5}
    raw_out = fullft.prepare_adv_stream_signal_for_supervised_mode(raw, "raw1000", stabilizer)
    cached_out = fullft.prepare_adv_stream_signal_for_supervised_mode(raw, "cached5000", stabilizer)
    hybrid_out = fullft.prepare_adv_stream_signal_for_supervised_mode(
        raw,
        "source_cached_target_raw1000",
        stabilizer,
    )

    assert raw_out is raw
    torch.testing.assert_close(cached_out, raw + 5.0)
    torch.testing.assert_close(hybrid_out, raw + 5.0)
    assert captured == [stabilizer, stabilizer]


def test_make_train_loader_raw1000_uses_source_cache_and_target_anchor(monkeypatch, tmp_path):
    source_signals = np.zeros((3, 1000, 12), dtype=np.float32)
    source_labels = np.eye(5, dtype=np.float32)[:3]
    source_signal_path = tmp_path / "ptbxl_raw1000.npy"
    source_label_path = tmp_path / "ptbxl_labels.npy"
    np.save(source_signal_path, source_signals)
    np.save(source_label_path, source_labels)

    target_signal_path = tmp_path / "cpsc_2018_real_k500_seed20260531.signals.npz"
    np.savez_compressed(
        target_signal_path,
        signals=np.ones((2, 1000, 12), dtype=np.float32),
        labels=np.ones((2, 5), dtype=np.float32),
        record_ids=np.asarray(["r0", "r1"], dtype=str),
    )

    def fake_anchor_signal_npz_path(center, args):
        assert center == "cpsc_2018"
        return target_signal_path

    captured = {}

    def fake_build_loader(**kwargs):
        captured.update(kwargs)
        return "loader"

    monkeypatch.setattr(fullft, "anchor_signal_npz_path", fake_anchor_signal_npz_path)
    monkeypatch.setattr(fullft, "build_weighted_signal_stream_loader_from_datasets", fake_build_loader)

    args = fullft.build_arg_parser().parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--center",
            "cpsc_2018",
            "--supervised_input_mode",
            "raw1000",
            "--source_raw1000_signal_cache",
            str(source_signal_path),
            "--source_raw1000_label_cache",
            str(source_label_path),
            "--batch_size",
            "4",
            "--num_workers",
            "0",
        ]
    )
    ptbxl_payload = {
        "signals": np.zeros((3, 12, 5000), dtype=np.float32),
        "labels": source_labels,
        "folds": np.asarray([1, 2, 9], dtype=np.int64),
    }
    pn_payload = {
        "signals": np.zeros((2, 12, 5000), dtype=np.float32),
        "labels": np.ones((2, 5), dtype=np.float32),
    }

    loader = fullft.make_train_loader(
        ptbxl_payload,
        pn_payload,
        np.asarray([0], dtype=np.int64),
        args,
        target_train_record_ids={"r1"},
    )

    assert loader == "loader"
    assert len(captured["source_dataset"]) == 2
    assert len(captured["target_dataset"]) == 1
    src_x, _ = captured["source_dataset"][0]
    tgt_x, _ = captured["target_dataset"][0]
    assert tuple(src_x.shape) == (12, 1000)
    assert tuple(tgt_x.shape) == (12, 1000)


def test_make_train_loader_hybrid_keeps_source_cached_and_prepares_target_adv(monkeypatch, tmp_path):
    target_signal_path = tmp_path / "cpsc_2018_real_k500_seed20260531.signals.npz"
    np.savez_compressed(
        target_signal_path,
        signals=np.ones((2, 1000, 12), dtype=np.float32),
        labels=np.ones((2, 5), dtype=np.float32),
        record_ids=np.asarray(["r0", "r1"], dtype=str),
    )

    def fake_anchor_signal_npz_path(center, args):
        assert center == "cpsc_2018"
        return target_signal_path

    def fake_raw_prepare(x, kwargs):
        assert tuple(x.shape[-2:]) == (12, 1000)
        return torch.nn.functional.interpolate(x, size=5000, mode="linear", align_corners=True) + 7.0

    captured = {}

    def fake_build_loader(**kwargs):
        captured.update(kwargs)
        return "loader"

    monkeypatch.setattr(fullft, "anchor_signal_npz_path", fake_anchor_signal_npz_path)
    monkeypatch.setattr(fullft, "prepare_raw_ecgfounder_input", fake_raw_prepare)
    monkeypatch.setattr(fullft, "build_weighted_signal_stream_loader_from_datasets", fake_build_loader)

    args = fullft.build_arg_parser().parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--center",
            "cpsc_2018",
            "--supervised_input_mode",
            "source_cached_target_raw1000",
            "--batch_size",
            "4",
            "--num_workers",
            "0",
        ]
    )
    ptbxl_payload = {
        "signals": np.zeros((3, 12, 5000), dtype=np.float32),
        "labels": np.eye(5, dtype=np.float32)[:3],
        "folds": np.asarray([1, 2, 9], dtype=np.int64),
    }
    pn_payload = {
        "signals": np.zeros((2, 12, 5000), dtype=np.float32),
        "labels": np.ones((2, 5), dtype=np.float32),
    }
    adv_signals = np.full((1, 12, 1000), 3.0, dtype=np.float32)
    adv_labels = np.ones((1, 5), dtype=np.float32)

    loader = fullft.make_train_loader(
        ptbxl_payload,
        pn_payload,
        np.asarray([0], dtype=np.int64),
        args,
        adv_signals=adv_signals,
        adv_labels=adv_labels,
        target_train_record_ids={"r1"},
    )

    assert loader == "loader"
    src_x, _ = captured["source_dataset"][0]
    tgt_x, _ = captured["target_dataset"][0]
    assert tuple(src_x.shape) == (12, 5000)
    assert tuple(tgt_x.shape) == (12, 5000)
    np.testing.assert_allclose(captured["adv_signals"].shape, (1, 12, 5000))
    assert float(tgt_x.mean()) > 7.0
    assert float(captured["adv_signals"].mean()) > 7.0


def test_prepare_raw_corruption_teacher_input_passes_stabilizer_kwargs(monkeypatch):
    teacher = _TrackingTeacher()
    raw = torch.zeros((3, 12, 1000), dtype=torch.float32)
    captured: list[dict[str, object]] = []

    def fake_convert(x, **kwargs):
        captured.append(dict(kwargs))
        return x + 1.0

    monkeypatch.setattr(fullft, "ecg1000_to_ecgfounder_input", fake_convert)

    prepared = fullft.prepare_raw_corruption_teacher_input(
        teacher,
        raw,
        {"bandpass_low_hz": 0.5, "bandpass_high_hz": 35.0},
    )

    assert torch.allclose(prepared, raw + 1.0)
    assert captured == [{"bandpass_low_hz": 0.5, "bandpass_high_hz": 35.0}]


def test_prepare_cached_ecgfounder_input_applies_500hz_stabilizer():
    sample_rate_hz = 500.0
    t = torch.arange(5000, dtype=torch.float32) / sample_rate_hz
    low = torch.sin(2 * torch.pi * 5.0 * t)
    high = 0.8 * torch.sin(2 * torch.pi * 60.0 * t)
    x = (low + high).reshape(1, 1, -1).repeat(1, 12, 1)

    same = fullft.prepare_cached_ecgfounder_input(x, {})
    assert same is x

    y = fullft.prepare_cached_ecgfounder_input(
        x,
        {"bandpass_low_hz": 0.5, "bandpass_high_hz": 35.0},
    )
    expected = fullft.global_zscore_torch(low.reshape(1, 1, -1).repeat(1, 12, 1))
    corr = torch.corrcoef(torch.stack([y[0, 0], expected[0, 0]]))[0, 1]
    assert float(corr) > 0.99
    torch.testing.assert_close(y.reshape(1, -1).mean(dim=1), torch.zeros(1), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(y.reshape(1, -1).std(dim=1), torch.ones(1), atol=1e-5, rtol=0.0)


def test_fullft_raw_corruption_epoch_updates_encoder(monkeypatch):
    torch.manual_seed(19)
    signals = torch.zeros((2, 12, 1000), dtype=torch.float32)
    labels = torch.zeros((2, 5), dtype=torch.float32)
    labels[:, 0] = 1.0
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2)
    calls: list[dict[str, object]] = []

    def fake_views(clean_np, **kwargs):
        calls.append(dict(kwargs))
        views = np.concatenate([clean_np + 0.1, clean_np + 0.2], axis=0).astype(np.float32)
        return views, {
            "enabled": True,
            "view_mode": "augmix",
            "n_generated": int(views.shape[0]),
            "n_corrupted": int(views.shape[0]),
            "op_counts": {"powerline_noise": int(views.shape[0])},
            "copies": int(kwargs["copies"]),
            "severity": int(kwargs["severity"]),
            "severity_profile": str(kwargs["severity_profile"]),
            "width": int(kwargs["augmix_width"]),
            "depth": int(kwargs["augmix_depth"]),
            "alpha": float(kwargs["augmix_alpha"]),
            "mixture_mode": str(kwargs["augmix_mixture_mode"]),
            "mixture_prob": float(kwargs["augmix_mixture_prob"]),
            "ops": list(kwargs["ops"]),
            "renorm": bool(kwargs["renorm"]),
            "clip_abs": float(kwargs["clip_abs"]),
        }

    monkeypatch.setattr(fullft, "build_profiled_raw_corruption_views", fake_views)
    monkeypatch.setattr(fullft, "ecg1000_to_ecgfounder_input", lambda x: x)

    model = _TinyFullFT()
    before = [p.detach().clone() for p in model.encoder.parameters()]
    optimizer = AdamW(model.parameters(), lr=1e-2)

    stats = fullft.train_fullft_raw_corruption_consistency_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        pos_weight=torch.ones(5),
        device=torch.device("cpu"),
        copies=2,
        severity=5,
        severity_profile="calibrated_10to20pp",
        ops=["powerline_noise", "emg_noise"],
        prob=1.0,
        consistency_weight=0.5,
        bce_weight=1.0,
        consistency_loss="jsd",
        rng=np.random.default_rng(19),
        trainable_params=list(model.parameters()),
        grad_clip=1.0,
        max_batches=1,
        renorm=False,
        clip_abs=6.0,
        view_mode="augmix",
        augmix_width=1,
        augmix_depth=1,
        augmix_alpha=1.0,
        augmix_mixture_mode="fixed",
        augmix_mixture_prob=1.0,
        augmix_mixture_beta_a=0.0,
        augmix_mixture_beta_b=0.0,
    )

    assert stats["enabled"] is True
    assert stats["view_mode"] == "augmix"
    assert stats["n_batches"] == 1
    assert stats["n_generated"] == 4
    assert calls[0]["severity_profile"] == "calibrated_10to20pp"
    assert calls[0]["augmix_width"] == 1
    assert calls[0]["augmix_depth"] == 1
    assert calls[0]["augmix_mixture_mode"] == "fixed"
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, model.encoder.parameters())
    )


def test_fullft_raw_corruption_epoch_passes_input_stabilizer_kwargs(monkeypatch):
    torch.manual_seed(29)
    signals = torch.zeros((2, 12, 1000), dtype=torch.float32)
    labels = torch.zeros((2, 5), dtype=torch.float32)
    labels[:, 0] = 1.0
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2)
    captured: list[dict[str, object]] = []

    def fake_views(clean_np, **kwargs):
        views = (clean_np + 0.1).astype(np.float32)
        return views, {
            "enabled": True,
            "n_generated": int(views.shape[0]),
            "n_corrupted": int(views.shape[0]),
            "op_counts": {"powerline_noise": int(views.shape[0])},
        }

    def fake_convert(x, **kwargs):
        captured.append(dict(kwargs))
        return x

    monkeypatch.setattr(fullft, "build_profiled_raw_corruption_views", fake_views)
    monkeypatch.setattr(fullft, "ecg1000_to_ecgfounder_input", fake_convert)

    model = _TinyFullFT()
    optimizer = AdamW(model.parameters(), lr=1e-2)
    stabilizer = {"bandpass_low_hz": 0.5, "bandpass_high_hz": 35.0}

    stats = fullft.train_fullft_raw_corruption_consistency_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        pos_weight=torch.ones(5),
        device=torch.device("cpu"),
        copies=1,
        severity=5,
        severity_profile="calibrated_10to20pp",
        ops=["powerline_noise"],
        prob=1.0,
        consistency_weight=0.5,
        bce_weight=1.0,
        consistency_loss="soft_bce",
        rng=np.random.default_rng(29),
        trainable_params=list(model.parameters()),
        grad_clip=0.0,
        max_batches=1,
        renorm=False,
        clip_abs=6.0,
        input_stabilizer_kwargs=stabilizer,
    )

    assert stats["input_stabilizer"] == stabilizer
    assert captured == [stabilizer, stabilizer]


def test_fullft_raw_corruption_epoch_updates_op_conditioned_adapter(monkeypatch):
    torch.manual_seed(23)
    signals = torch.zeros((2, 12, 1000), dtype=torch.float32)
    labels = torch.zeros((2, 5), dtype=torch.float32)
    labels[:, 0] = 1.0
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2)

    def fake_views(clean_np, **kwargs):
        views = np.concatenate([clean_np + 0.1, clean_np + 0.2], axis=0).astype(np.float32)
        return views, {
            "enabled": True,
            "view_mode": "augmix",
            "n_generated": int(views.shape[0]),
            "n_corrupted": int(views.shape[0]),
            "op_counts": {"powerline_noise": 2, "emg_noise": 2},
            "view_ops": ["powerline_noise", "emg_noise", "powerline_noise", "emg_noise"],
            "copies": int(kwargs["copies"]),
            "severity": int(kwargs["severity"]),
            "severity_profile": str(kwargs["severity_profile"]),
            "width": int(kwargs["augmix_width"]),
            "depth": int(kwargs["augmix_depth"]),
            "alpha": float(kwargs["augmix_alpha"]),
            "mixture_mode": str(kwargs["augmix_mixture_mode"]),
            "mixture_prob": float(kwargs["augmix_mixture_prob"]),
            "ops": list(kwargs["ops"]),
            "renorm": bool(kwargs["renorm"]),
            "clip_abs": float(kwargs["clip_abs"]),
        }

    monkeypatch.setattr(fullft, "build_profiled_raw_corruption_views", fake_views)
    monkeypatch.setattr(fullft, "ecg1000_to_ecgfounder_input", lambda x: x)

    model = _TinyFullFT()
    op_adapter = fullft.OperatorConditionedLogitAdapter(
        feature_dim=8,
        num_classes=5,
        op_names=["powerline_noise", "emg_noise"],
        hidden_dim=4,
        dropout=0.0,
        scale=1.0,
    )
    before = [p.detach().clone() for p in op_adapter.parameters()]
    optimizer = AdamW(list(model.parameters()) + list(op_adapter.parameters()), lr=1e-2)

    stats = fullft.train_fullft_raw_corruption_consistency_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        pos_weight=torch.ones(5),
        device=torch.device("cpu"),
        copies=2,
        severity=5,
        severity_profile="calibrated_10to20pp",
        ops=["powerline_noise", "emg_noise"],
        prob=1.0,
        consistency_weight=0.0,
        bce_weight=1.0,
        consistency_loss="soft_bce",
        rng=np.random.default_rng(23),
        trainable_params=list(model.parameters()) + list(op_adapter.parameters()),
        grad_clip=0.0,
        max_batches=1,
        renorm=False,
        clip_abs=6.0,
        view_mode="augmix",
        augmix_width=1,
        augmix_depth=1,
        augmix_alpha=1.0,
        augmix_mixture_mode="fixed",
        augmix_mixture_prob=1.0,
        augmix_mixture_beta_a=0.0,
        augmix_mixture_beta_b=0.0,
        op_adapter=op_adapter,
    )

    after = [p.detach().clone() for p in op_adapter.parameters()]
    assert stats["op_conditioning_enabled"] is True
    assert stats["op_conditioned_views"] == 4
    assert any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_fullft_raw_corruption_epoch_can_train_feature_consistency_only(monkeypatch):
    torch.manual_seed(37)
    signals = torch.zeros((2, 12, 1000), dtype=torch.float32)
    labels = torch.zeros((2, 5), dtype=torch.float32)
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2)

    def fake_views(clean_np, **kwargs):
        views = np.concatenate([clean_np + 0.3, clean_np - 0.2], axis=0).astype(np.float32)
        return views, {
            "enabled": True,
            "view_mode": "augmix",
            "n_generated": int(views.shape[0]),
            "n_corrupted": int(views.shape[0]),
            "op_counts": {"baseline_shift": int(views.shape[0])},
        }

    monkeypatch.setattr(fullft, "build_profiled_raw_corruption_views", fake_views)
    monkeypatch.setattr(fullft, "ecg1000_to_ecgfounder_input", lambda x: x)

    model = _TinyFullFT()
    before = [p.detach().clone() for p in model.encoder.parameters()]
    optimizer = AdamW(model.parameters(), lr=1e-2)

    stats = fullft.train_fullft_raw_corruption_consistency_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        pos_weight=torch.ones(5),
        device=torch.device("cpu"),
        copies=2,
        severity=5,
        severity_profile="calibrated_10to20pp",
        ops=["baseline_wander", "baseline_shift"],
        prob=1.0,
        consistency_weight=0.0,
        bce_weight=0.0,
        consistency_loss="soft_bce",
        feature_consistency_weight=0.5,
        feature_consistency_normalize=True,
        rng=np.random.default_rng(37),
        trainable_params=list(model.parameters()),
        grad_clip=1.0,
        max_batches=1,
        renorm=False,
        clip_abs=6.0,
        view_mode="augmix",
        augmix_width=1,
        augmix_depth=1,
        augmix_alpha=1.0,
        augmix_mixture_mode="fixed",
        augmix_mixture_prob=1.0,
        augmix_mixture_beta_a=0.0,
        augmix_mixture_beta_b=0.0,
    )

    assert stats["feature_consistency_enabled"] is True
    assert stats["feature_consistency_weight"] == 0.5
    assert stats["feature_consistency_normalize"] is True
    assert stats["feature_loss"] > 0
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, model.encoder.parameters())
    )


def test_fullft_raw_corruption_epoch_can_train_from_frozen_teacher(monkeypatch):
    torch.manual_seed(31)
    signals = torch.zeros((2, 12, 1000), dtype=torch.float32)
    labels = torch.zeros((2, 5), dtype=torch.float32)
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2)

    def fake_views(clean_np, **kwargs):
        views = np.concatenate([clean_np + 0.1, clean_np + 0.2], axis=0).astype(np.float32)
        return views, {
            "enabled": True,
            "view_mode": "augmix",
            "n_generated": int(views.shape[0]),
            "n_corrupted": int(views.shape[0]),
            "op_counts": {"baseline_wander": int(views.shape[0])},
        }

    monkeypatch.setattr(fullft, "build_profiled_raw_corruption_views", fake_views)
    monkeypatch.setattr(fullft, "ecg1000_to_ecgfounder_input", lambda x: x)

    model = _TinyFullFT()
    teacher = _TrackingTeacher()
    before = [p.detach().clone() for p in model.encoder.parameters()]
    optimizer = AdamW(model.parameters(), lr=1e-2)

    stats = fullft.train_fullft_raw_corruption_consistency_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        pos_weight=torch.ones(5),
        device=torch.device("cpu"),
        copies=2,
        severity=5,
        severity_profile="calibrated_10to20pp",
        ops=["baseline_wander", "baseline_shift"],
        prob=1.0,
        consistency_weight=0.0,
        bce_weight=0.0,
        consistency_loss="soft_bce",
        teacher_model=teacher,
        teacher_weight=0.75,
        teacher_loss="mse_logits",
        teacher_view="corrupt",
        rng=np.random.default_rng(31),
        trainable_params=list(model.parameters()),
        grad_clip=1.0,
        max_batches=1,
        renorm=False,
        clip_abs=6.0,
        view_mode="augmix",
        augmix_width=1,
        augmix_depth=1,
        augmix_alpha=1.0,
        augmix_mixture_mode="fixed",
        augmix_mixture_prob=1.0,
        augmix_mixture_beta_a=0.0,
        augmix_mixture_beta_b=0.0,
    )

    assert teacher.calls == 1
    assert teacher.last_shape == (4, 12, 1000)
    assert stats["teacher_enabled"] is True
    assert stats["teacher_loss"] > 0
    assert stats["teacher_weight"] == 0.75
    assert stats["teacher_objective"] == "mse_logits"
    assert stats["teacher_view"] == "corrupt"
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, model.encoder.parameters())
    )


def test_fullft_raw_corruption_branches_run_primary_and_aux(monkeypatch):
    parser = fullft.build_arg_parser()
    args = parser.parse_args(
        [
            "--ref_meta_json",
            "/tmp/ref.json",
            "--enable_raw_corrupt_consistency",
            "--raw_corrupt_ops",
            "powerline_noise",
            "emg_noise",
            "random_leads_masking",
            "--raw_corrupt_consistency_weight",
            "2.0",
            "--raw_corrupt_consistency_loss",
            "jsd",
            "--raw_corrupt_bce_weight",
            "1.0",
            "--raw_corrupt_max_batches",
            "64",
            "--enable_raw_corrupt_aux_consistency",
            "--raw_corrupt_aux_ops",
            "baseline_wander",
            "baseline_shift",
            "--raw_corrupt_aux_consistency_weight",
            "0.25",
            "--raw_corrupt_aux_consistency_loss",
            "soft_bce",
            "--raw_corrupt_aux_bce_weight",
            "0.0",
            "--raw_corrupt_aux_max_batches",
            "16",
        ]
    )
    calls: list[dict[str, object]] = []

    def fake_epoch(**kwargs):
        calls.append(dict(kwargs))
        return {
            "enabled": True,
            "ops": list(kwargs["ops"]),
            "consistency_weight": float(kwargs["consistency_weight"]),
            "bce_weight": float(kwargs["bce_weight"]),
            "consistency_objective": str(kwargs["consistency_loss"]),
            "max_batches": int(kwargs["max_batches"]),
        }

    monkeypatch.setattr(fullft, "train_fullft_raw_corruption_consistency_epoch", fake_epoch)

    model = _TinyFullFT()
    loader = DataLoader(
        TensorDataset(torch.zeros((2, 12, 1000)), torch.zeros((2, 5))),
        batch_size=2,
    )
    optimizer = AdamW(model.parameters(), lr=1e-2)

    primary, aux = fullft.train_fullft_raw_corruption_branches_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        pos_weight=torch.ones(5),
        device=torch.device("cpu"),
        args=args,
        epoch=3,
        trainable_params=list(model.parameters()),
    )

    assert len(calls) == 2
    assert primary["ops"] == ["powerline_noise", "emg_noise", "random_leads_masking"]
    assert primary["consistency_weight"] == 2.0
    assert primary["bce_weight"] == 1.0
    assert primary["consistency_objective"] == "jsd"
    assert primary["max_batches"] == 64
    assert aux["ops"] == ["baseline_wander", "baseline_shift"]
    assert aux["consistency_weight"] == 0.25
    assert aux["bce_weight"] == 0.0
    assert aux["consistency_objective"] == "soft_bce"
    assert aux["max_batches"] == 16


def test_fullft_raw_corruption_epoch_rejects_nonfinite_loss(monkeypatch):
    signals = torch.zeros((2, 12, 1000), dtype=torch.float32)
    labels = torch.zeros((2, 5), dtype=torch.float32)
    loader = DataLoader(TensorDataset(signals, labels), batch_size=2)

    def fake_views(clean_np, **kwargs):
        return np.concatenate([clean_np, clean_np], axis=0).astype(np.float32), {
            "enabled": True,
            "view_mode": "augmix",
            "n_generated": int(clean_np.shape[0] * 2),
            "n_corrupted": int(clean_np.shape[0] * 2),
            "op_counts": {"powerline_noise": int(clean_np.shape[0] * 2)},
        }

    monkeypatch.setattr(fullft, "build_profiled_raw_corruption_views", fake_views)
    monkeypatch.setattr(fullft, "ecg1000_to_ecgfounder_input", lambda x: x)
    model = _NaNFullFT()

    with pytest.raises(RuntimeError, match="non-finite raw corruption"):
        fullft.train_fullft_raw_corruption_consistency_epoch(
            model=model,
            loader=loader,
            optimizer=AdamW(model.parameters(), lr=1e-2),
            pos_weight=torch.ones(5),
            device=torch.device("cpu"),
            copies=2,
            severity=5,
            severity_profile="calibrated_10to20pp",
            ops=["powerline_noise"],
            prob=1.0,
            consistency_weight=0.5,
            bce_weight=1.0,
            consistency_loss="jsd",
            rng=np.random.default_rng(23),
            trainable_params=list(model.parameters()),
            grad_clip=1.0,
            max_batches=1,
            renorm=False,
            clip_abs=6.0,
            view_mode="augmix",
            augmix_width=1,
            augmix_depth=1,
            augmix_alpha=1.0,
            augmix_mixture_mode="fixed",
            augmix_mixture_prob=1.0,
            augmix_mixture_beta_a=0.0,
            augmix_mixture_beta_b=0.0,
        )
