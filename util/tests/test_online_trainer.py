from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import core.methods.runtime as runtime_module
import core.online_trainer as online_module
from core.online_trainer import load_online_train_config, train_online_model
from models.contracts import ECGFOUNDER_SPEC, EFFICIENTNET1DV2_SPEC


REPO = Path(__file__).resolve().parents[2]


class _RawDataset(Dataset):
    def __init__(
        self,
        *,
        hash_ids: tuple[str, str] = ("hash-0", "hash-1"),
        labels: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.waveform = (
            torch.linspace(-1.0, 1.0, 1000).view(1000, 1).repeat(1, 12)
        )
        self.labels = labels or (
            torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32),
            torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32),
        )
        self.hash_ids = hash_ids

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return {
            "waveform": self.waveform + float(index) * 0.1,
            "label": self.labels[index],
            "hash_id": self.hash_ids[index],
        }


class _TinyManagedModel(torch.nn.Module):
    model_spec = EFFICIENTNET1DV2_SPEC

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(12, 5)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(waveform.mean(dim=2))


class _TinyBatchNormManagedModel(_TinyManagedModel):
    def __init__(self) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(12, momentum=0.1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(self.bn(waveform).mean(dim=2))


class _TinyECGFounderModel(_TinyManagedModel):
    model_spec = ECGFOUNDER_SPEC


class _CaptureMonitor:
    def __init__(self) -> None:
        self.steps: list[dict[str, object]] = []
        self.epochs: list[dict[str, object]] = []

    def describe(self):
        return {"enabled": True, "backend": "capture"}

    def log_train_step(self, **kwargs):
        self.steps.append(dict(kwargs))

    def log_epoch(self, record, *, is_selected):
        self.epochs.append(
            {
                "epoch": int(record["epoch"]),
                "is_selected": bool(is_selected),
            }
        )

    def close(self):
        return None


def test_unknown_nonempty_batch_norm_policy_is_rejected() -> None:
    method = SimpleNamespace(
        contracts={"batch_norm_running_stats_policy": "typo_policy"},
        objective=SimpleNamespace(terms=()),
    )
    with pytest.raises(ValueError, match="unsupported BatchNorm"):
        online_module._build_batch_norm_momentum_plan(
            _TinyBatchNormManagedModel(),
            method,
            fixed20_exposure=False,
        )


def test_pcgrad_auxiliary_policy_keeps_fixed20_group_semantics() -> None:
    assert (
        online_module.FIXED20_PCGRAD_EXPOSURE_POLICY
        in online_module.FIXED20_GROUPED_EXPOSURE_POLICIES
    )


class _FakePool:
    hash_ids = ("hash-0", "hash-1")
    eligible_hash_ids = ("hash-0",)
    eligible_hash_id_set = frozenset(eligible_hash_ids)
    labels = torch.tensor(
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.float32
    )
    standardizer = object()

    def to(self, device):
        assert torch.device(device).type == "cpu"
        return self

    def describe(self):
        return {
            "identity": {"identity_sha256": "a" * 64},
            "eligibility": {
                "eligible_hash_ids": ["hash-0"],
                "ineligible": [{"hash_id": "hash-1", "candidate_count": 0}],
            },
        }

    def indices_for_hashes(self, hashes):
        mapping = {value: index for index, value in enumerate(self.hash_ids)}
        missing = [value for value in hashes if value not in mapping]
        if missing:
            raise KeyError(missing)
        return torch.tensor([mapping[value] for value in hashes], dtype=torch.long)

    def get_attack_batch_by_hashes(self, hashes, **kwargs):
        del kwargs
        assert tuple(hashes) == ("hash-0",)
        return SimpleNamespace(
            labels=torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32),
            anchor_standardized=torch.zeros(1, 4, 128),
            candidates_standardized=torch.zeros(1, 20, 4, 128),
            candidate_pool_indices=torch.arange(20).view(1, 20),
        )


class _TwoEligibleFakePool(_FakePool):
    eligible_hash_ids = _FakePool.hash_ids
    eligible_hash_id_set = frozenset(eligible_hash_ids)

    def describe(self):
        return {
            "identity": {"identity_sha256": "b" * 64},
            "eligibility": {
                "eligible_hash_ids": list(self.eligible_hash_ids),
                "ineligible": [],
            },
        }

    def get_attack_batch_by_hashes(self, hashes, **kwargs):
        del kwargs
        assert tuple(hashes) == self.hash_ids
        return SimpleNamespace(
            labels=self.labels.clone(),
            anchor_standardized=torch.zeros(2, 4, 128),
            candidates_standardized=torch.zeros(2, 20, 4, 128),
            candidate_pool_indices=torch.arange(40).view(2, 20),
        )


class _FakeDiagnostics:
    def __init__(self, values=None) -> None:
        self.values = torch.as_tensor(
            [0.2] if values is None else values, dtype=torch.float32
        )

    def select(self, mask):
        return _FakeDiagnostics(self.values[mask])

    def sample_tensor_dict(self):
        return {
            "adversarial_delta": self.values,
            "sample_anyflip_eligible": torch.ones_like(self.values),
            "sample_anyflip_success": torch.zeros_like(self.values),
        }

    def mean_tensor_dict(self):
        return {
            "loss_gain": self.values.mean(),
            "sample_anyflip_numerator": torch.tensor(0.0),
            "sample_anyflip_denominator": torch.tensor(float(self.values.numel())),
        }


class _FakeCheckpointIdentity:
    def __init__(self, token: str) -> None:
        self.token = token

    def describe(self):
        return {"component": "test_fixture", "sha256": self.token * 64}


def _managed_component(module: torch.nn.Module, token: str = "d"):
    module.checkpoint_identity = _FakeCheckpointIdentity(token)
    return module


def _config_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "configs"
    shutil.copytree(REPO / "configs", root)
    path = root / "train" / "PN2021.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["logging"]["tensorboard"]["enabled"] = False
    payload["logging"]["tensorboard"]["scalars"][
        "batch_loss_interval_steps"
    ] = 1
    payload["diagnostics"]["performance_timing"]["enabled"] = True
    payload["diagnostics"]["performance_timing"]["interval_steps"] = 1
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _method(path: Path, name: str) -> Path:
    return path.parent / "methods" / f"{name}.yaml"


def _fixed20_config_bundle(tmp_path: Path) -> Path:
    matched = _config_bundle(tmp_path)
    path = matched.with_name("PN2021_fixed20.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["logging"]["tensorboard"]["enabled"] = False
    payload["diagnostics"]["performance_timing"]["enabled"] = False
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _fake_lhat(*, flatline: bool = False):
    def generate(**kwargs):
        assert kwargs["targets"].shape == (1, 5)
        assert kwargs["raw_clean_waveform"].shape == (1, 1000, 12)
        waveform = (
            torch.zeros(1, 1000, 12)
            if flatline
            else torch.linspace(-0.1, 0.1, 1000)
            .view(1, 1000, 1)
            .repeat(1, 1, 12)
        )
        return SimpleNamespace(
            waveform_raw=waveform,
            anchor_waveform_raw=waveform.clone(),
            weights=torch.full((1, 20), 0.05),
            diagnostics=_FakeDiagnostics(),
        )

    return generate


def _fake_lhat_one_rejected():
    def generate(**kwargs):
        assert kwargs["targets"].shape == (2, 5)
        valid = (
            torch.linspace(-0.1, 0.1, 1000)
            .view(1, 1000, 1)
            .repeat(1, 1, 12)
        )
        waveform = torch.cat((valid, torch.zeros_like(valid)), dim=0)
        return SimpleNamespace(
            waveform_raw=waveform,
            anchor_waveform_raw=waveform.clone(),
            weights=torch.full((2, 20), 0.05),
            diagnostics=_FakeDiagnostics([0.2, 99.0]),
        )

    return generate


def _fake_lhat_two_valid():
    def generate(**kwargs):
        assert kwargs["targets"].shape == (2, 5)
        first = (
            torch.linspace(-0.1, 0.1, 1000)
            .view(1, 1000, 1)
            .repeat(1, 1, 12)
        )
        waveform = torch.cat((first, first + 0.01), dim=0)
        return SimpleNamespace(
            waveform_raw=waveform,
            anchor_waveform_raw=waveform.clone(),
            weights=torch.full((2, 20), 0.05),
            diagnostics=_FakeDiagnostics([0.2, 0.3]),
        )

    return generate


def _fake_augmix(observed=None):
    def generate(clean, hard, **kwargs):
        batch = clean.shape[0]
        if observed is not None:
            observed["clean_shape"] = tuple(clean.shape)
            observed["hard_shape"] = tuple(hard.shape)
            observed["sampling_rate_hz"] = int(kwargs["sampling_rate_hz"])
        return SimpleNamespace(
            mixed_raw=clean * 0.9 + hard * 0.1,
            chain1_depth=torch.full((batch,), 2, dtype=torch.int64),
            chain2_depth=torch.full((batch,), 3, dtype=torch.int64),
            chain1_composition_index=torch.zeros(batch, dtype=torch.int64),
            chain2_composition_index=torch.full((batch,), 10, dtype=torch.int64),
            chain1_operator_mask=torch.tensor(
                [[True, True, False, False, False]] * batch
            ),
            chain2_operator_mask=torch.tensor(
                [[True, True, True, False, False]] * batch
            ),
            mixture_weights=torch.tensor([[0.2, 0.3, 0.5]] * batch),
            augmented_strength=torch.full((batch,), 0.4),
        )

    return generate


def _fake_augmix_random_trace():
    def generate(clean, hard, **kwargs):
        batch = clean.shape[0]
        generator = kwargs["generator"]
        chain1 = torch.randint(0, 20, (batch,), generator=generator)
        chain2 = torch.randint(0, 20, (batch,), generator=generator)
        weights = torch.rand(batch, 3, generator=generator)
        weights = weights / weights.sum(dim=1, keepdim=True)
        strength = torch.rand(batch, generator=generator)
        return SimpleNamespace(
            mixed_raw=clean * 0.9 + hard * 0.1,
            chain1_depth=torch.where(chain1 < 10, 2, 3),
            chain2_depth=torch.where(chain2 < 10, 2, 3),
            chain1_composition_index=chain1,
            chain2_composition_index=chain2,
            chain1_operator_mask=torch.stack(
                tuple(chain1 == index for index in range(5)), dim=1
            ),
            chain2_operator_mask=torch.stack(
                tuple(chain2 == index for index in range(5)), dim=1
            ),
            mixture_weights=weights,
            augmented_strength=strength,
        )

    return generate


def _fake_latent_augmix(observed=None):
    def generate(clean, **kwargs):
        if observed is not None:
            observed.append(
                {
                    "shape": tuple(clean.shape),
                    "sampling_rate_hz": int(kwargs["sampling_rate_hz"]),
                }
            )
        offset = torch.rand(
            clean.shape[0],
            device=clean.device,
            generator=kwargs["generator"],
        ).view(-1, 1, 1)
        return SimpleNamespace(
            mixed_raw=clean + 0.05 * offset,
            chain_depths=torch.tensor(
                [[1, 2, 3]] * clean.shape[0], device=clean.device
            ),
            mixture_weights=torch.tensor(
                [[0.2, 0.3, 0.5]] * clean.shape[0], device=clean.device
            ),
            augmented_strength=torch.ones(clean.shape[0], device=clean.device),
            residual_rms_ratio=torch.zeros(clean.shape[0], device=clean.device),
            chain_output_nonfinite_count=torch.zeros(
                clean.shape[0], 3, dtype=torch.int64, device=clean.device
            ),
        )

    return generate


def test_online_config_resolves_method_independent_copied_bundle(tmp_path):
    path = _config_bundle(tmp_path)
    config = load_online_train_config(path)

    assert config.config_root == path.parents[1]
    assert set(config.references) == {
        "split_config",
        "data_load_config",
        "random_seed_config",
        "source_baseline_registry",
    }
    assert config.payload["fairness"]["loss_normalization"] == (
        "per_origin_record_mean"
    )


def test_a0_uses_one_outer_step_and_rejects_vae_state(tmp_path):
    path = _config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    result = train_online_model(
        _TinyManagedModel(),
        loader,
        center="ningbo",
        method_config_path=_method(path, "a0_clean_v1"),
        config_path=path,
        output_dir=tmp_path / "a0-run",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )

    assert result.method_id == "a0_clean_v1"
    assert result.optimizer_steps == 1
    assert result.history[0]["train"]["sample_count"] == 2
    assert result.history[0]["train"]["objective_valid_counts"] == {
        "clean_bce": 2
    }

    with pytest.raises(ValueError, match="decoder presence"):
        train_online_model(
            _TinyManagedModel(),
            loader,
            center="ningbo",
            method_config_path=_method(path, "a0_clean_v1"),
            decoder=torch.nn.Identity(),
            config_path=path,
            output_dir=tmp_path / "a0-invalid",
            device="cpu",
            training_parameters={"epochs": 1, "batch_size": 2},
        )


def test_tensorboard_marks_only_final_online_epoch_as_selected(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    monitor = _CaptureMonitor()
    monkeypatch.setattr(
        online_module,
        "build_tensorboard_monitor",
        lambda logging, output: monitor,
    )
    checkpoint_writes: list[Path] = []
    real_torch_save = online_module.torch.save

    def counted_torch_save(payload, path):
        checkpoint_writes.append(Path(path))
        return real_torch_save(payload, path)

    monkeypatch.setattr(online_module.torch, "save", counted_torch_save)

    result = train_online_model(
        _TinyManagedModel(),
        loader,
        center="ningbo",
        method_config_path=_method(path, "a0_clean_v1"),
        config_path=path,
        output_dir=tmp_path / "a0-two-epochs",
        device="cpu",
        training_parameters={"epochs": 2, "batch_size": 2, "amp_enabled": False},
    )

    assert result.epochs_completed == 2
    assert monitor.epochs == [
        {"epoch": 1, "is_selected": False},
        {"epoch": 2, "is_selected": True},
    ]
    assert len(checkpoint_writes) == 1


def test_online_result_reuses_checkpoint_digest(monkeypatch, tmp_path):
    path = _config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    sha_calls: list[Path] = []
    real_sha256_file = online_module.sha256_file

    def counted_sha256_file(path):
        sha_calls.append(Path(path))
        return real_sha256_file(path)

    monkeypatch.setattr(online_module, "sha256_file", counted_sha256_file)
    result = train_online_model(
        _TinyManagedModel(),
        loader,
        center="ningbo",
        method_config_path=_method(path, "a0_clean_v1"),
        config_path=path,
        output_dir=tmp_path / "a0-cached-checkpoint-sha",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )

    first = result.describe()
    second = result.describe()

    assert first["last_checkpoint"] == second["last_checkpoint"]
    assert len(sha_calls) == 1
    assert sha_calls[0] == result.last_checkpoint_path


def test_a3c_is_executable_without_vae_and_uses_canonical_100hz(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    observed = {}

    def fake_corruption(clean, **kwargs):
        del kwargs
        observed["shape"] = tuple(clean.shape)
        return SimpleNamespace(
            waveform_raw_100hz=clean + 0.05,
            diagnostics=SimpleNamespace(
                output_nonfinite_count=torch.zeros(2, dtype=torch.long),
                depth=torch.tensor([2, 3]),
                composition_index=torch.tensor([0, 10]),
                operator_mask=torch.tensor(
                    [
                        [True, True, False, False, False],
                        [True, True, True, False, False],
                    ]
                ),
            ),
        )

    monkeypatch.setattr(
        runtime_module, "generate_canonical_corruption", fake_corruption
    )
    result = train_online_model(
        _TinyManagedModel(),
        loader,
        center="ningbo",
        method_config_path=_method(path, "a3c_depth23_v1"),
        config_path=path,
        output_dir=tmp_path / "a3c-run",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )

    assert observed["shape"] == (2, 1000, 12)
    assert result.optimizer_steps == 1
    assert result.history[0]["train"]["objective_valid_counts"] == {
        "clean_bce": 2,
        "corrupted_bce": 2,
    }


def test_sampled_timer_preserves_observed_wall_and_excludes_observer_time(
    monkeypatch,
):
    ticks = iter((0.000, 0.001, 0.011, 0.012, 0.022, 0.032))
    monkeypatch.setattr(online_module.time, "perf_counter", lambda: next(ticks))
    timer = online_module._SampledStepTimer(
        device=torch.device("cpu"),
        data_wait_ms=0.0,
        cuda_events=False,
    )
    timer.start("h2d")
    timer.stop("h2d")
    with online_module._excluded_observer_work(timer):
        pass
    result = timer.finish(batch_size=4, grad_norm=0.0, view_executions=2)

    assert result["base_group_observed_wall_ms"] == pytest.approx(32.0)
    assert result["base_group_excluded_logging_ms"] == pytest.approx(10.0)
    assert result["base_group_wall_ms"] == pytest.approx(22.0)
    assert result["base_group_gpu_plus_data_ms"] == pytest.approx(10.0)
    assert result["base_group_host_gap_ms"] == pytest.approx(12.0)


def test_diagnostic_sample_summary_uses_record_level_quantiles_and_exact_rates():
    distributions, flat, rates = online_module._diagnostic_sample_summary(
        {
            "lhat/adversarial_delta": (
                torch.tensor([0.1, 0.2]),
                torch.tensor([0.3, 0.4]),
            ),
            "lhat/sample_anyflip_success": (torch.tensor([1.0, 0.0, 0.0]),),
            "lhat/sample_anyflip_eligible": (torch.tensor([1.0, 0.0, 1.0]),),
        }
    )

    attack = distributions["lhat/adversarial_delta"]
    assert attack["count"] == 4
    assert attack["mean"] == pytest.approx(0.25)
    assert attack["median"] == pytest.approx(0.25)
    assert attack["p90"] == pytest.approx(0.37)
    assert flat["lhat/adversarial_delta_median"] == pytest.approx(0.25)
    assert flat["lhat/sample_anyflip_numerator"] == pytest.approx(1.0)
    assert flat["lhat/sample_anyflip_denominator"] == pytest.approx(2.0)
    assert flat["lhat/decoded_anchor_sample_anyflip_asr"] == pytest.approx(0.5)
    assert rates["lhat/decoded_anchor_sample_anyflip_asr"] == {
        "numerator": pytest.approx(1.0),
        "denominator": pytest.approx(2.0),
        "rate": pytest.approx(0.5),
    }


def test_stochastic_trace_summary_is_stable_and_content_sensitive():
    chunks = {
        "augmix/chain1_operator_mask": (
            torch.tensor([[True, False], [False, True]]),
        ),
        "augmix/mixture_weights": (
            torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]]),
        ),
    }
    first = online_module._stochastic_trace_summary(
        input_identity_sha256="a" * 64,
        input_record_count=2,
        chunks=chunks,
    )
    replay = online_module._stochastic_trace_summary(
        input_identity_sha256="a" * 64,
        input_record_count=2,
        chunks={name: tuple(value.clone() for value in values) for name, values in chunks.items()},
    )
    changed = online_module._stochastic_trace_summary(
        input_identity_sha256="a" * 64,
        input_record_count=2,
        chunks={
            **chunks,
            "augmix/mixture_weights": (
                torch.tensor([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]),
            ),
        },
    )

    assert first == replay
    assert first["aggregate_sha256"] != changed["aggregate_sha256"]
    assert first["tensors"]["augmix/chain1_operator_mask"]["shape"] == [2, 2]


def test_fixed20_accumulates_family_balanced_views_before_one_optimizer_step(
    monkeypatch, tmp_path
):
    path = _fixed20_config_bundle(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["diagnostics"]["performance_timing"].update(
        {
            "enabled": True,
            "interval_steps": 1,
            "scope": (
                "fixed20_base_group_excludes_logging_otherwise_sampled_step"
            ),
        }
    )
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    observed_indices: list[int] = []
    observer_events: list[str] = []
    timer_start = online_module._SampledStepTimer.start_excluded_host
    timer_stop = online_module._SampledStepTimer.stop_excluded_host

    def tracked_observer_start(timer):
        observer_events.append("start")
        return timer_start(timer)

    def tracked_observer_stop(timer):
        observer_events.append("stop")
        return timer_stop(timer)

    monkeypatch.setattr(
        online_module._SampledStepTimer,
        "start_excluded_host",
        tracked_observer_start,
    )
    monkeypatch.setattr(
        online_module._SampledStepTimer,
        "stop_excluded_host",
        tracked_observer_stop,
    )

    def fake_corruption(clean, *, composition_indices, **kwargs):
        del kwargs
        assert composition_indices is not None
        unique = torch.unique(composition_indices).tolist()
        assert len(unique) == 1
        index = int(unique[0])
        observed_indices.append(index)
        return SimpleNamespace(
            waveform_raw_100hz=clean + float(index + 1) * 1.0e-3,
            diagnostics=SimpleNamespace(
                output_nonfinite_count=torch.zeros(
                    clean.shape[0], dtype=torch.int64, device=clean.device
                ),
                depth=torch.full(
                    (clean.shape[0],),
                    2 if index < 10 else 3,
                    dtype=torch.int64,
                    device=clean.device,
                ),
                composition_index=composition_indices,
                operator_mask=torch.tensor(
                    [[True, True, False, False, False]] * clean.shape[0],
                    device=clean.device,
                ),
            ),
        )

    monkeypatch.setattr(runtime_module, "generate_canonical_corruption", fake_corruption)
    model = _TinyBatchNormManagedModel()
    observed_bn_momenta: list[float] = []
    hook = model.bn.register_forward_pre_hook(
        lambda module, inputs: observed_bn_momenta.append(float(module.momentum))
    )
    try:
        result = train_online_model(
            model,
            loader,
            center="ningbo",
            method_config_path=_method(path, "direct_depth23_fixed20"),
            config_path=path,
            output_dir=tmp_path / "fixed20-run",
            device="cpu",
            training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
        )
    finally:
        hook.remove()

    assert observed_indices == list(range(20))
    assert result.optimizer_steps == 1
    train = result.history[0]["train"]
    assert train["sample_count"] == 2
    assert train["view_execution_sample_count"] == 42
    assert train["objective_valid_counts"] == {
        "clean_bce": 2,
        "corrupted_bce": 40,
    }
    assert train["exposure"]["base_record_count"] == 2
    assert train["exposure"]["clean_count"] == 2
    assert train["exposure"]["corrupted_count"] == 40
    assert train["exposure"]["total_count"] == 42
    assert train["exposure"]["optimizer_steps_per_base_batch"] == 1
    assert train["exposure"]["optimizer_steps_this_epoch"] == 1
    assert train["exposure"]["view_executions_per_base_batch"] == 21
    assert train["exposure"]["family_loss_weights"] == {
        "clean": 0.5,
        "corrupted_total": 0.5,
        "corrupted_per_composition": 0.025,
    }
    assert train["exposure"]["materialized_corruption_cache"] is False
    performance = result.history[0]["performance"]
    assert performance["timed_step_count"] == 1
    assert performance["view_executions"] == pytest.approx(21.0)
    assert performance["base_group_wall_ms"] >= 0.0
    assert performance["base_group_augmentation_ms"] >= 0.0
    assert performance["base_group_forward_backward_ms"] >= 0.0
    assert performance["base_group_optimizer_ms"] >= 0.0
    assert performance["base_records_per_sec"] > 0.0
    assert performance["base_group_observed_wall_ms"] >= performance[
        "base_group_wall_ms"
    ]
    assert performance["base_group_excluded_logging_ms"] >= 0.0
    assert observer_events
    assert observer_events == ["start", "stop"] * (len(observer_events) // 2)
    assert tuple(train["exposure"]["per_exposure_counts"]) == (
        "clean",
        *(f"corruption_{index:02d}" for index in range(20)),
    )
    assert set(train["exposure"]["per_exposure_counts"].values()) == {2}
    expected_bn_momenta = online_module._family_balanced_batch_norm_momenta(
        0.1, (0.5, *(0.025 for _ in range(20)))
    )
    assert observed_bn_momenta == pytest.approx(expected_bn_momenta)
    assert model.bn.momentum == pytest.approx(0.1)


def test_fixed20_raw_aux_adds_one_normalized_exposure_without_extra_step(
    monkeypatch, tmp_path
):
    path = _fixed20_config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    observed_indices: list[int] = []

    def fake_corruption(clean, *, composition_indices, **kwargs):
        del kwargs
        assert composition_indices is not None
        index = int(torch.unique(composition_indices).item())
        observed_indices.append(index)
        return SimpleNamespace(
            waveform_raw_100hz=clean + float(index + 1) * 1.0e-3,
            diagnostics=SimpleNamespace(
                output_nonfinite_count=torch.zeros(
                    clean.shape[0], dtype=torch.int64, device=clean.device
                ),
                depth=torch.full(
                    (clean.shape[0],),
                    2 if index < 10 else 3,
                    dtype=torch.int64,
                    device=clean.device,
                ),
                composition_index=composition_indices,
                operator_mask=torch.tensor(
                    [[True, True, False, False, False]] * clean.shape[0],
                    device=clean.device,
                ),
            ),
        )

    monkeypatch.setattr(runtime_module, "generate_canonical_corruption", fake_corruption)
    model = _TinyBatchNormManagedModel()
    observed_bn_momenta: list[float] = []
    hook = model.bn.register_forward_pre_hook(
        lambda module, inputs: observed_bn_momenta.append(float(module.momentum))
    )
    try:
        result = train_online_model(
            model,
            loader,
            center="ningbo",
            method_config_path=_method(
                path, "direct_depth23_fixed20_raw_aux"
            ),
            config_path=path,
            output_dir=tmp_path / "fixed20-raw-aux-run",
            device="cpu",
            training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
        )
    finally:
        hook.remove()

    assert observed_indices == list(range(20))
    assert result.optimizer_steps == 1
    train = result.history[0]["train"]
    assert train["sample_count"] == 2
    assert train["view_execution_sample_count"] == 44
    assert train["objective_valid_counts"] == {
        "clean_bce": 2,
        "corrupted_bce": 40,
        "auxiliary_bce": 2,
    }
    assert train["objective_effective_loss_mass"] == pytest.approx(
        {"clean_bce": 0.45, "corrupted_bce": 0.45, "auxiliary_bce": 0.1}
    )
    assert train["exposure"]["view_executions_per_base_batch"] == 22
    assert train["exposure"]["family_loss_weights"] == {
        "clean": 0.45,
        "corrupted_total": 0.45,
        "corrupted_per_composition": 0.0225,
        "auxiliary": 0.1,
    }
    assert train["exposure"]["auxiliary_nominal_loss_mass"] == pytest.approx(0.1)
    assert train["exposure"]["auxiliary_effective_loss_mass"] == pytest.approx(0.1)
    assert train["exposure"][
        "auxiliary_effective_fraction_of_nominal"
    ] == pytest.approx(1.0)
    assert tuple(train["exposure"]["per_exposure_counts"]) == (
        "clean",
        *(f"corruption_{index:02d}" for index in range(20)),
        "auxiliary",
    )
    expected_bn_momenta = online_module._family_balanced_batch_norm_momenta(
        0.1,
        (0.45, *(0.0225 for _ in range(20)), 0.1),
    )
    assert observed_bn_momenta == pytest.approx(expected_bn_momenta)
    assert model.bn.momentum == pytest.approx(0.1)


def test_fixed20_lhat_aux_reports_effective_mass_over_origin_records(
    monkeypatch, tmp_path
):
    path = _fixed20_config_bundle(tmp_path)

    def fake_corruption(clean, *, composition_indices, **kwargs):
        del kwargs
        assert composition_indices is not None
        return SimpleNamespace(
            waveform_raw_100hz=clean + 1.0e-3,
            diagnostics=SimpleNamespace(
                output_nonfinite_count=torch.zeros(
                    clean.shape[0], dtype=torch.int64, device=clean.device
                ),
                depth=torch.full(
                    (clean.shape[0],),
                    2,
                    dtype=torch.int64,
                    device=clean.device,
                ),
                composition_index=composition_indices,
                operator_mask=torch.tensor(
                    [[True, True, False, False, False]] * clean.shape[0],
                    device=clean.device,
                ),
            ),
        )

    monkeypatch.setattr(runtime_module, "generate_canonical_corruption", fake_corruption)
    monkeypatch.setattr(runtime_module, "generate_lhat_adversarial", _fake_lhat())
    result = train_online_model(
        _TinyManagedModel(),
        DataLoader(_RawDataset(), batch_size=2, shuffle=False),
        center="ningbo",
        method_config_path=_method(path, "direct_depth23_fixed20_lhat_aux"),
        latent_pool=_FakePool(),
        decoder=_managed_component(torch.nn.Identity()),
        config_path=path,
        output_dir=tmp_path / "fixed20-lhat-aux-mass",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )

    train = result.history[0]["train"]
    assert train["objective_valid_counts"] == {
        "clean_bce": 2,
        "corrupted_bce": 40,
        "auxiliary_bce": 1,
    }
    assert train["objective_effective_loss_mass"] == pytest.approx(
        {"clean_bce": 0.45, "corrupted_bce": 0.45, "auxiliary_bce": 0.05}
    )
    assert train["candidate_eligible_fraction"] == pytest.approx(0.5)
    assert train["quality_accepted_fraction"] == pytest.approx(0.5)
    assert train["exposure"]["auxiliary_nominal_loss_mass"] == pytest.approx(0.1)
    assert train["exposure"]["auxiliary_effective_loss_mass"] == pytest.approx(0.05)
    assert train["exposure"][
        "auxiliary_effective_fraction_of_nominal"
    ] == pytest.approx(0.5)


def test_a5_preserves_eligible_over_batch_weighting_and_chain3_reuse(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    observed = {}
    monkeypatch.setattr(
        runtime_module, "generate_lhat_adversarial", _fake_lhat()
    )
    monkeypatch.setattr(
        runtime_module,
        "generate_three_chain_augmix",
        _fake_augmix(observed),
    )
    result = train_online_model(
        _TinyManagedModel(),
        loader,
        center="ningbo",
        method_config_path=_method(path, "a5_lhat_threechain_v1"),
        latent_pool=_FakePool(),
        decoder=_managed_component(torch.nn.Identity()),
        config_path=path,
        output_dir=tmp_path / "a5-run",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )

    assert observed == {
        "clean_shape": (2, 1000, 12),
        "hard_shape": (2, 1000, 12),
        "sampling_rate_hz": 100,
    }
    train = result.history[0]["train"]
    assert train["sample_count"] == 2
    assert train["candidate_eligible_count"] == 1
    assert train["quality_accepted_count"] == 1
    assert train["ineligible_count"] == 1
    assert train["objective_valid_counts"] == {
        "clean_bce": 2,
        "lhat_direct_bce": 1,
        "augmix_bce": 1,
        "clean_lhat_augmix_jsd": 1,
    }
    assert train["weighted_lhat_contribution"] == pytest.approx(
        0.5 * train["lhat_hard_bce"]
    )
    assert train["weighted_augmix_contribution"] == pytest.approx(
        0.5 * train["augmix_bce"]
    )
    assert train["loss"] == pytest.approx(
        sum(train["weighted_objective_terms"].values())
    )
    checkpoint = torch.load(
        result.last_checkpoint_path, map_location="cpu", weights_only=True
    )
    assert checkpoint["method_id"] == "a5_lhat_threechain_v1"
    assert checkpoint["scientific_arm"] == "A5"
    assert "arm" not in checkpoint
    assert checkpoint["optimizer_steps"] == 1
    assert checkpoint["run_identity"]["method"]["method"][
        "profile_sha256"
    ]
    train_result = json.loads(
        (result.output_dir / "train_result.json").read_text(encoding="utf-8")
    )
    assert train_result["selection"]["policy"] == "last"
    assert train_result["selection"]["selected_epoch"] == 1
    assert train_result["selection"][
        "heldout_evaluation_used_for_selection"
    ] is False


def test_a5_diagnostics_are_filtered_by_the_final_quality_mask(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    monkeypatch.setattr(
        runtime_module,
        "generate_lhat_adversarial",
        _fake_lhat_one_rejected(),
    )
    monkeypatch.setattr(runtime_module, "generate_three_chain_augmix", _fake_augmix())
    result = train_online_model(
        _TinyManagedModel(),
        DataLoader(_RawDataset(), batch_size=2, shuffle=False),
        center="ningbo",
        method_config_path=_method(path, "a5_lhat_threechain_v1"),
        latent_pool=_TwoEligibleFakePool(),
        decoder=_managed_component(torch.nn.Identity()),
        config_path=path,
        output_dir=tmp_path / "a5-filtered-diagnostics",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )

    epoch = result.history[0]
    assert epoch["train"]["candidate_eligible_count"] == 2
    assert epoch["train"]["quality_accepted_count"] == 1
    assert epoch["diagnostics"]["lhat/loss_gain"] == pytest.approx(0.2)
    assert epoch["diagnostics"]["lhat/sample_anyflip_denominator"] == pytest.approx(
        1.0
    )
    distribution = epoch["diagnostic_distributions"]["lhat/adversarial_delta"]
    assert distribution == {
        "count": 1,
        "sum": pytest.approx(0.2),
        "mean": pytest.approx(0.2),
        "median": pytest.approx(0.2),
        "p90": pytest.approx(0.2),
    }
    assert epoch["diagnostic_rates"][
        "lhat/decoded_anchor_sample_anyflip_asr"
    ] == {
        "numerator": pytest.approx(0.0),
        "denominator": pytest.approx(1.0),
        "rate": pytest.approx(0.0),
    }
    trace = epoch["stochastic_trace"]
    assert len(trace["aggregate_sha256"]) == 64
    assert trace["tensors"]["lhat/candidate_pool_indices"]["shape"] == [2, 20]
    assert trace["tensors"][
        "threechain_augmix/chain1_composition_index"
    ]["shape"] == [2]
    assert epoch["quality_rejected"] == [
        {"node_id": "lhat", "hash_id": "hash-1", "reason": "flatline"}
    ]


def test_augmix_random_trace_is_independent_of_lhat_quality_acceptance(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    monkeypatch.setattr(
        runtime_module,
        "generate_three_chain_augmix",
        _fake_augmix_random_trace(),
    )
    monkeypatch.setattr(
        runtime_module,
        "generate_lhat_adversarial",
        _fake_lhat_two_valid(),
    )
    accepted = train_online_model(
        _TinyManagedModel(),
        DataLoader(_RawDataset(), batch_size=2, shuffle=False),
        center="ningbo",
        method_config_path=_method(path, "a5_lhat_threechain_v1"),
        latent_pool=_TwoEligibleFakePool(),
        decoder=_managed_component(torch.nn.Identity()),
        config_path=path,
        output_dir=tmp_path / "a5-trace-all-accepted",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )
    monkeypatch.setattr(
        runtime_module,
        "generate_lhat_adversarial",
        _fake_lhat_one_rejected(),
    )
    rejected = train_online_model(
        _TinyManagedModel(),
        DataLoader(_RawDataset(), batch_size=2, shuffle=False),
        center="ningbo",
        method_config_path=_method(path, "a5_lhat_threechain_v1"),
        latent_pool=_TwoEligibleFakePool(),
        decoder=_managed_component(torch.nn.Identity()),
        config_path=path,
        output_dir=tmp_path / "a5-trace-one-rejected",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )

    accepted_trace = accepted.history[0]["stochastic_trace"]
    rejected_trace = rejected.history[0]["stochastic_trace"]
    assert accepted_trace["input_identity_sha256"] == rejected_trace[
        "input_identity_sha256"
    ]
    shared_names = (
        "threechain_augmix/chain1_composition_index",
        "threechain_augmix/chain2_composition_index",
        "threechain_augmix/chain1_operator_mask",
        "threechain_augmix/chain2_operator_mask",
        "threechain_augmix/mixture_weights",
        "threechain_augmix/augmented_strength",
    )
    assert all(
        accepted_trace["tensors"][name]["sha256"]
        == rejected_trace["tensors"][name]["sha256"]
        for name in shared_names
    )
    assert accepted_trace["tensors"]["lhat/quality_accepted_mask"]["sha256"] != (
        rejected_trace["tensors"]["lhat/quality_accepted_mask"]["sha256"]
    )


def test_latent_threechain_online_method_uses_encoder_without_latent_pool(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    observed = []
    monkeypatch.setattr(
        runtime_module,
        "generate_latent_three_chain_augmix",
        _fake_latent_augmix(observed),
    )
    encoder = _managed_component(torch.nn.Identity(), "e")
    decoder = _managed_component(torch.nn.Identity(), "d")
    method_path = (
        path.parent / "methods" / "exp_paired_augmix_latent_bridge_v1.yaml"
    )

    model = _TinyBatchNormManagedModel()
    observed_bn_momenta = []
    hook = model.bn.register_forward_pre_hook(
        lambda module, inputs: observed_bn_momenta.append(float(module.momentum))
    )
    try:
        result = train_online_model(
            model,
            loader,
            center="ningbo",
            method_config_path=method_path,
            encoder=encoder,
            decoder=decoder,
            config_path=path,
            output_dir=tmp_path / "latent-threechain-run",
            device="cpu",
            training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
        )
    finally:
        hook.remove()

    assert result.method_id == "latent_threechain_augmix_residual_depth23_aug075"
    assert result.optimizer_steps == 1
    assert observed == [
        {"shape": (2, 1000, 12), "sampling_rate_hz": 100},
        {"shape": (2, 1000, 12), "sampling_rate_hz": 100},
    ]
    history = result.history[0]["train"]
    assert history["objective_valid_counts"] == {
        "clean_bce": 2,
        "clean_augmix_jsd": 2,
        "augmix_view_1_bce": 2,
        "augmix_view_2_bce": 2,
    }
    resources = json.loads(
        (result.output_dir / "method_resources.json").read_text(encoding="utf-8")
    )
    assert resources["latent_pool"] is None
    assert resources["vae_encoder_checkpoint"]["sha256"] == "e" * 64
    assert resources["vae_decoder_checkpoint"]["sha256"] == "d" * 64
    assert resources["rng_seed_configs"]["latent_augmix_rng"]["path"] == str(
        path.parents[1] / "random_seed.yaml"
    )
    assert history["quality_view_total_count"] == 4
    assert history["quality_view_accepted_count"] == 4
    assert history["quality_view_accepted_fraction"] == pytest.approx(1.0)
    assert history["both_views_accepted_record_count"] == 2
    assert history["both_views_accepted_fraction"] == pytest.approx(1.0)
    assert observed_bn_momenta == pytest.approx(
        online_module._family_balanced_batch_norm_momenta(
            0.1,
            (0.5, 0.25, 0.25),
        )
    )
    assert model.bn.momentum == pytest.approx(0.1)


def test_latent_threechain_rejected_view_keeps_bn_schedule_and_reports_intersection(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    calls = 0

    def fake_latent(clean, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        waveform = torch.zeros_like(clean) if calls == 1 else clean + 0.05
        return SimpleNamespace(
            mixed_raw=waveform,
            chain_depths=torch.tensor([[1, 2, 3]] * clean.shape[0]),
            mixture_weights=torch.tensor([[0.2, 0.3, 0.5]] * clean.shape[0]),
            augmented_strength=torch.ones(clean.shape[0]),
            residual_rms_ratio=torch.zeros(clean.shape[0]),
            chain_output_nonfinite_count=torch.zeros(
                clean.shape[0], 3, dtype=torch.int64
            ),
        )

    monkeypatch.setattr(
        runtime_module, "generate_latent_three_chain_augmix", fake_latent
    )
    model = _TinyBatchNormManagedModel()
    observed_bn_momenta = []
    hook = model.bn.register_forward_pre_hook(
        lambda module, inputs: observed_bn_momenta.append(float(module.momentum))
    )
    try:
        result = train_online_model(
            model,
            loader,
            center="ningbo",
            method_config_path=(
                path.parent / "methods" / "exp_paired_augmix_latent_bridge_v1.yaml"
            ),
            encoder=_managed_component(torch.nn.Identity(), "e"),
            decoder=_managed_component(torch.nn.Identity(), "d"),
            config_path=path,
            output_dir=tmp_path / "latent-one-view-rejected",
            device="cpu",
            training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
        )
    finally:
        hook.remove()

    train = result.history[0]["train"]
    assert train["objective_valid_counts"] == {
        "clean_bce": 2,
        "clean_augmix_jsd": 0,
        "augmix_view_1_bce": 0,
        "augmix_view_2_bce": 2,
    }
    assert train["quality_accepted_count"] == 0
    assert train["quality_view_total_count"] == 4
    assert train["quality_view_accepted_count"] == 2
    assert train["quality_view_accepted_fraction"] == pytest.approx(0.5)
    assert train["both_views_accepted_record_count"] == 0
    assert train["both_views_accepted_fraction"] == pytest.approx(0.0)
    assert observed_bn_momenta == pytest.approx(
        online_module._family_balanced_batch_norm_momenta(
            0.1,
            (0.5, 0.25, 0.25),
        )
    )
    assert {item["node_id"] for item in result.history[0]["quality_rejected"]} == {
        "augmix_view_1"
    }


def test_latent_threechain_rejects_method_rng_seed_config_drift(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    method_path = (
        path.parent / "methods" / "exp_paired_augmix_latent_bridge_v1.yaml"
    )
    payload = yaml.safe_load(method_path.read_text(encoding="utf-8"))
    payload["resources"]["latent_augmix_rng"]["seed_config"] = "train/augmix.yaml"
    method_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        runtime_module,
        "generate_latent_three_chain_augmix",
        _fake_latent_augmix(),
    )

    with pytest.raises(ValueError, match="method RNG resources"):
        train_online_model(
            _TinyManagedModel(),
            DataLoader(_RawDataset(), batch_size=2, shuffle=False),
            center="ningbo",
            method_config_path=method_path,
            encoder=_managed_component(torch.nn.Identity(), "e"),
            decoder=_managed_component(torch.nn.Identity(), "d"),
            config_path=path,
            output_dir=tmp_path / "latent-rng-drift",
            device="cpu",
            training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
        )


def test_a5_rejected_waveform_falls_back_to_clean_only(monkeypatch, tmp_path):
    path = _config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    observed = {}
    monkeypatch.setattr(
        runtime_module, "generate_lhat_adversarial", _fake_lhat(flatline=True)
    )
    monkeypatch.setattr(
        runtime_module,
        "generate_three_chain_augmix",
        _fake_augmix(observed),
    )
    result = train_online_model(
        _TinyManagedModel(),
        loader,
        center="ningbo",
        method_config_path=_method(path, "a5_lhat_threechain_v1"),
        latent_pool=_FakePool(),
        decoder=_managed_component(torch.nn.Identity()),
        config_path=path,
        output_dir=tmp_path / "a5-rejected",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )

    train = result.history[0]["train"]
    assert train["quality_accepted_count"] == 0
    assert train["weighted_lhat_contribution"] == 0.0
    assert train["weighted_augmix_contribution"] == 0.0
    assert observed["clean_shape"] == (2, 1000, 12)
    assert observed["hard_shape"] == (2, 1000, 12)
    assert result.history[0]["quality_rejected"] == [
        {"node_id": "lhat", "hash_id": "hash-0", "reason": "flatline"}
    ]


def test_ecgfounder_a5_generates_at_100hz_before_model_domain_adapter(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    observed = {}
    monkeypatch.setattr(
        runtime_module, "generate_lhat_adversarial", _fake_lhat()
    )
    monkeypatch.setattr(
        runtime_module,
        "generate_three_chain_augmix",
        _fake_augmix(observed),
    )
    result = train_online_model(
        _TinyECGFounderModel(),
        loader,
        center="ningbo",
        method_config_path=_method(path, "a5_lhat_threechain_v1"),
        latent_pool=_FakePool(),
        decoder=_managed_component(torch.nn.Identity()),
        config_path=path,
        output_dir=tmp_path / "a5-ecgfounder",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )

    assert result.optimizer_steps == 1
    assert observed["clean_shape"] == (2, 1000, 12)
    assert observed["sampling_rate_hz"] == 100


def test_train_step_logs_generic_terms_and_locked_a5_aliases(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    loader = DataLoader(_RawDataset(), batch_size=2, shuffle=False)
    monitor = _CaptureMonitor()
    monkeypatch.setattr(
        online_module, "build_tensorboard_monitor", lambda logging, output: monitor
    )
    monkeypatch.setattr(
        runtime_module, "generate_lhat_adversarial", _fake_lhat()
    )
    monkeypatch.setattr(
        runtime_module, "generate_three_chain_augmix", _fake_augmix()
    )
    train_online_model(
        _TinyManagedModel(),
        loader,
        center="ningbo",
        method_config_path=_method(path, "a5_lhat_threechain_v1"),
        latent_pool=_FakePool(),
        decoder=_managed_component(torch.nn.Identity()),
        config_path=path,
        output_dir=tmp_path / "a5-logging",
        device="cpu",
        training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
    )

    components = monitor.steps[0]["loss_components"]
    assert {
        "clean_bce",
        "lhat_direct_bce",
        "augmix_bce",
        "clean_lhat_augmix_jsd",
        "lhat_hard_bce",
        "augmix_jsd",
        "weighted_clean",
        "weighted_lhat",
        "weighted_augmix",
        "weighted_jsd",
    } <= set(components)


def test_unknown_loader_hash_fails_instead_of_becoming_clean_only(
    monkeypatch, tmp_path
):
    path = _config_bundle(tmp_path)
    loader = DataLoader(
        _RawDataset(hash_ids=("hash-0", "outside-pool")),
        batch_size=2,
        shuffle=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "generate_lhat_adversarial",
        lambda **kwargs: pytest.fail("attack must not run for unbound hashes"),
    )
    with pytest.raises(RuntimeError, match="outside the bound latent pool"):
        train_online_model(
            _TinyManagedModel(),
            loader,
            center="ningbo",
            method_config_path=_method(path, "a5_lhat_threechain_v1"),
            latent_pool=_FakePool(),
            decoder=_managed_component(torch.nn.Identity()),
            config_path=path,
            output_dir=tmp_path / "a5-unknown-hash",
            device="cpu",
            training_parameters={"epochs": 1, "batch_size": 2, "amp_enabled": False},
        )
