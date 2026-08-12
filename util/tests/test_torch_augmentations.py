"""Parity and device-contract tests for Torch ECG augmentations."""

from __future__ import annotations

import inspect
import os
import random
from collections.abc import Sequence

import numpy as np
import pytest
import torch

from util.augmentations.operators import (
    baseline_shift,
    baseline_wander,
    emg_noise,
    powerline_noise,
    random_leads_masking,
)
from util.augmentations.torch_operators import (
    baseline_shift as torch_baseline_shift, baseline_wander as torch_baseline_wander,
    emg_noise as torch_emg_noise, powerline_noise as torch_powerline_noise,
    random_leads_masking as torch_random_leads_masking,
)
from util.random_seed import (
    derive_seed,
    load_random_seed_config,
    make_numpy_rng,
    make_python_rng,
    make_torch_generator,
)


CPU_TORCH_PAIRS = (
    (powerline_noise, torch_powerline_noise),
    (emg_noise, torch_emg_noise),
    (baseline_wander, torch_baseline_wander),
    (baseline_shift, torch_baseline_shift),
    (random_leads_masking, torch_random_leads_masking),
)


def test_project_random_seed_is_loaded_and_factories_are_reproducible():
    assert load_random_seed_config().base_seed == 20260501
    assert derive_seed("record", "operator") == derive_seed("record", "operator")
    assert np.array_equal(
        make_numpy_rng("sample").normal(size=16),
        make_numpy_rng("sample").normal(size=16),
    )
    python_first = make_python_rng("sample")
    python_second = make_python_rng("sample")
    assert [python_first.random() for _ in range(3)] == [
        python_second.random() for _ in range(3)
    ]
    first = torch.rand(16, generator=make_torch_generator("cpu", "sample"))
    second = torch.rand(16, generator=make_torch_generator("cpu", "sample"))
    assert torch.equal(first, second)


def _signal(dtype=np.float32, time_size: int = 1000) -> np.ndarray:
    time = np.linspace(0.0, 10.0, time_size, endpoint=False, dtype=dtype)
    return np.stack(
        [np.sin(2.0 * np.pi * (1.0 + lead / 20.0) * time) for lead in range(12)],
        axis=1,
    ).astype(dtype)


def _shape(size: int | Sequence[int] | None) -> tuple[int, ...]:
    if size is None:
        return ()
    if isinstance(size, int):
        return (size,)
    return tuple(size)


class TorchBackedNumpyRNG:
    """NumPy-style adapter around a CPU ``torch.Generator`` for parity tests."""

    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    @staticmethod
    def _return(value: torch.Tensor, size: object) -> float | np.ndarray:
        array = value.numpy()
        return float(array.item()) if size is None else array

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | Sequence[int] | None = None,
    ) -> float | np.ndarray:
        value = torch.rand(
            _shape(size),
            dtype=torch.float32,
            generator=self.generator,
        )
        value = value * (float(high) - float(low)) + float(low)
        return self._return(value, size)

    def normal(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        size: int | Sequence[int] | None = None,
    ) -> float | np.ndarray:
        value = torch.randn(
            _shape(size),
            dtype=torch.float32,
            generator=self.generator,
        )
        value = value * float(std) + float(mean)
        return self._return(value, size)

    def choice(
        self,
        values: Sequence[int] | np.ndarray,
        size: int | Sequence[int] | None = None,
    ) -> int | np.ndarray:
        candidates = np.asarray(values)
        indices = torch.randint(
            0,
            len(candidates),
            _shape(size),
            generator=self.generator,
        ).numpy()
        selected = candidates[indices]
        return selected.item() if size is None else selected


@pytest.mark.parametrize(("cpu_function", "torch_function"), CPU_TORCH_PAIRS)
def test_torch_parameter_interface_matches_cpu(cpu_function, torch_function):
    cpu_parameters = inspect.signature(cpu_function).parameters
    torch_parameters = inspect.signature(torch_function).parameters

    assert tuple(torch_parameters) == tuple(cpu_parameters)
    for name, cpu_parameter in cpu_parameters.items():
        torch_parameter = torch_parameters[name]
        assert torch_parameter.kind == cpu_parameter.kind
        assert torch_parameter.default == cpu_parameter.default


@pytest.mark.parametrize(
    ("cpu_function", "torch_function", "kwargs", "atol"),
    [
        (
            powerline_noise,
            torch_powerline_noise,
            {"max_amplitude": 0.3, "freq": 100, "dependency": True},
            2e-4,
        ),
        (
            emg_noise,
            torch_emg_noise,
            {"max_amplitude": 0.2, "dependency": True},
            2e-5,
        ),
        (
            baseline_wander,
            torch_baseline_wander,
            {
                "max_amplitude": 0.4,
                "max_freq": 0.2,
                "min_freq": 0.01,
                "k": 3,
                "freq": 100,
                "dependency": True,
            },
            2e-5,
        ),
        (
            baseline_shift,
            torch_baseline_shift,
            {
                "max_amplitude": 0.4,
                "shift_ratio": 0.3,
                "num_segment": 2,
                "freq": 100,
                "dependency": False,
            },
            2e-6,
        ),
        (
            baseline_shift,
            torch_baseline_shift,
            {
                "max_amplitude": 0.4,
                "shift_ratio": 0.3,
                "num_segment": 2,
                "freq": 100,
                "dependency": False,
                "amplitude_mode": "signed_shared_uniform",
            },
            2e-6,
        ),
        (
            random_leads_masking,
            torch_random_leads_masking,
            {"mask_leads_prob": 0.4},
            0.0,
        ),
        (
            random_leads_masking,
            torch_random_leads_masking,
            {"mask_leads_prob": 1.0, "ensure_at_least_one_lead": True},
            0.0,
        ),
    ],
)
def test_torch_cpu_equation_parity(
    cpu_function,
    torch_function,
    kwargs,
    atol,
):
    signal = _signal()
    seed = 20260716
    expected = cpu_function(
        signal,
        p=1.0,
        rng=TorchBackedNumpyRNG(seed),
        **kwargs,
    )
    actual = torch_function(
        torch.from_numpy(signal),
        p=1.0,
        rng=torch.Generator(device="cpu").manual_seed(seed),
        **kwargs,
    )

    np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=atol)


def test_torch_conditional_masking_matches_cpu():
    signal = np.ones((1000, 12), dtype=np.float32)
    seed = 29
    kwargs = {
        "mask_leads_selection": "conditional",
        "mask_leads_condition": (2, 3),
    }
    expected = random_leads_masking(
        signal,
        rng=TorchBackedNumpyRNG(seed),
        python_rng=random.Random(seed),
        **kwargs,
    )
    actual = torch_random_leads_masking(
        torch.from_numpy(signal),
        rng=torch.Generator(device="cpu").manual_seed(seed),
        python_rng=random.Random(seed),
        **kwargs,
    )
    assert np.array_equal(actual.numpy(), expected)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (torch_powerline_noise, {"freq": 100}),
        (torch_emg_noise, {}),
        (torch_baseline_wander, {"freq": 100}),
        (torch_baseline_shift, {"freq": 100}),
        (torch_random_leads_masking, {}),
    ],
)
def test_torch_contract_reproducibility_and_no_in_place_change(function, kwargs):
    signal = torch.from_numpy(_signal(np.float64))
    original = signal.clone()
    first = function(
        signal,
        rng=torch.Generator(device="cpu").manual_seed(7),
        **kwargs,
    )
    second = function(
        signal,
        rng=torch.Generator(device="cpu").manual_seed(7),
        **kwargs,
    )

    assert first.shape == signal.shape
    assert first.dtype == torch.float32
    assert first.device == signal.device
    assert first.is_contiguous()
    assert torch.isfinite(first).all()
    assert torch.equal(signal, original)
    assert first.data_ptr() != signal.data_ptr()
    assert torch.equal(first, second)


def test_torch_random_mask_can_keep_one_lead():
    signal = torch.ones((1000, 12), dtype=torch.float32)
    output = torch_random_leads_masking(
        signal,
        mask_leads_prob=1.0,
        ensure_at_least_one_lead=True,
        rng=torch.Generator(device="cpu").manual_seed(3),
    )
    surviving_leads = torch.any(output != 0.0, dim=0)
    assert int(surviving_leads.sum().item()) == 1


def test_torch_wrong_layout_is_rejected():
    channel_first = torch.zeros((12, 1000), dtype=torch.float32)
    with pytest.raises(ValueError, match=r"shape \(time, 12\)"):
        torch_emg_noise(channel_first)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (torch_powerline_noise, {"freq": 500}),
        (torch_emg_noise, {}),
        (torch_baseline_wander, {"freq": 500}),
        (
            torch_baseline_shift,
            {"freq": 500, "amplitude_mode": "signed_shared_uniform"},
        ),
        (
            torch_random_leads_masking,
            {"mask_leads_prob": 0.5, "ensure_at_least_one_lead": True},
        ),
    ],
)
def test_torch_batch_contract_is_reproducible_and_non_inplace(function, kwargs):
    single = torch.from_numpy(_signal(time_size=500))
    signal = single.unsqueeze(0).repeat(8, 1, 1)
    original = signal.clone()

    first = function(
        signal,
        rng=torch.Generator(device="cpu").manual_seed(20260716),
        **kwargs,
    )
    second = function(
        signal,
        rng=torch.Generator(device="cpu").manual_seed(20260716),
        **kwargs,
    )

    assert first.shape == (8, 500, 12)
    assert first.dtype == torch.float32
    assert first.device == signal.device
    assert first.is_contiguous()
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.equal(signal, original)
    assert first.data_ptr() != signal.data_ptr()
    assert not torch.equal(first[0], first[1])


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (torch_powerline_noise, {"freq": 500}),
        (torch_emg_noise, {}),
        (torch_baseline_wander, {"freq": 500}),
        (torch_baseline_shift, {"freq": 500}),
        (torch_random_leads_masking, {}),
    ],
)
def test_torch_batch_probability_zero_returns_float32_copy(function, kwargs):
    signal = torch.from_numpy(_signal(np.float64, time_size=128)).repeat(4, 1, 1)

    output = function(
        signal,
        p=0.0,
        rng=torch.Generator(device="cpu").manual_seed(9),
        **kwargs,
    )

    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert output.data_ptr() != signal.data_ptr()
    torch.testing.assert_close(output, signal.float(), rtol=0.0, atol=0.0)


def test_torch_batch_dependency_reconstructs_limb_leads_per_sample():
    signal = torch.from_numpy(_signal(time_size=256)).repeat(4, 1, 1)

    output = torch_emg_noise(
        signal,
        dependency=True,
        rng=torch.Generator(device="cpu").manual_seed(77),
    )

    torch.testing.assert_close(output[:, :, 2], output[:, :, 1] - output[:, :, 0])
    torch.testing.assert_close(
        output[:, :, 3], -(output[:, :, 1] + output[:, :, 0]) / 2.0
    )
    torch.testing.assert_close(
        output[:, :, 4], output[:, :, 0] - output[:, :, 1] / 2.0
    )
    torch.testing.assert_close(
        output[:, :, 5], output[:, :, 1] - output[:, :, 0] / 2.0
    )


def test_torch_batch_conditional_masking_has_requested_survivor_counts():
    signal = torch.ones((16, 100, 12), dtype=torch.float32)

    output = torch_random_leads_masking(
        signal,
        mask_leads_selection="conditional",
        mask_leads_condition=(2, 3),
        rng=torch.Generator(device="cpu").manual_seed(91),
    )

    survivors = torch.any(output != 0.0, dim=1)
    assert torch.all(survivors[:, :6].sum(dim=1) == 4)
    assert torch.all(survivors[:, 6:].sum(dim=1) == 3)


def test_torch_wrong_batch_layout_is_rejected():
    channel_first_batch = torch.zeros((4, 12, 1000), dtype=torch.float32)
    with pytest.raises(ValueError, match=r"\(batch, time, 12\)"):
        torch_emg_noise(channel_first_batch)


RUN_CUDA_TESTS = os.environ.get("ECG_RUN_CUDA_AUG_TESTS") == "1"


@pytest.mark.skipif(
    not RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="set ECG_RUN_CUDA_AUG_TESTS=1 after selecting a free shared-server GPU",
)
@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (torch_powerline_noise, {"freq": 100}),
        (torch_emg_noise, {}),
        (torch_baseline_wander, {"freq": 100}),
        (torch_baseline_shift, {"freq": 100}),
        (
            torch_random_leads_masking,
            {"mask_leads_prob": 1.0, "ensure_at_least_one_lead": True},
        ),
    ],
)
def test_torch_operator_runs_reproducibly_on_selected_cuda(function, kwargs):
    device = torch.device("cuda:0")
    signal = torch.from_numpy(_signal()).to(device)
    original = signal.clone()
    first = function(
        signal,
        rng=torch.Generator(device=device).manual_seed(11),
        **kwargs,
    )
    second = function(
        signal,
        rng=torch.Generator(device=device).manual_seed(11),
        **kwargs,
    )

    assert first.device == device
    assert first.dtype == torch.float32
    assert first.shape == signal.shape
    assert first.is_contiguous()
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.equal(signal, original)
    assert first.data_ptr() != signal.data_ptr()


@pytest.mark.skipif(
    not RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="set ECG_RUN_CUDA_AUG_TESTS=1 after selecting a free shared-server GPU",
)
@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (torch_powerline_noise, {"freq": 500}),
        (torch_emg_noise, {}),
        (torch_baseline_wander, {"freq": 500}),
        (
            torch_baseline_shift,
            {"freq": 500, "amplitude_mode": "signed_shared_uniform"},
        ),
        (
            torch_random_leads_masking,
            {"mask_leads_prob": 0.5, "ensure_at_least_one_lead": True},
        ),
    ],
)
def test_torch_batch_operator_runs_on_selected_cuda(function, kwargs):
    device = torch.device("cuda:0")
    signal = torch.from_numpy(_signal(time_size=500)).to(device)
    signal = signal.unsqueeze(0).repeat(32, 1, 1)

    output = function(
        signal,
        rng=torch.Generator(device=device).manual_seed(37),
        **kwargs,
    )

    assert output.shape == signal.shape
    assert output.device == device
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert torch.isfinite(output).all()
