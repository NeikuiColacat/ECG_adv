"""Goldens for the live batched Torch ECG corruption kernels."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from core.augmix import generate_two_chain_augmix_strong_view, load_augmix_config
from util.augmentations import torch_operators
from util.random_seed import (
    derive_seed,
    load_random_seed_config,
    make_numpy_rng,
    make_python_rng,
    make_torch_generator,
)


RETIRED_WRAPPERS = {
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
}
KERNEL_GOLDENS = (
    (
        "powerline_noise",
        990,
        "22077a779450798bef7d1ac04fc4e70d3ab2718ad2e3eb205e165066cd3ad784",
        "c95ba75ebc9f1902da424d8f2cec7f1605b33e33cee31c9e2721617872b09ec0",
    ),
    (
        "emg_noise",
        991,
        "01217632a8d7e758d6e3a04f7568b665c362c10e39a650181fc552cc98495531",
        "71c6cd8714899663428de0d6df4e554bc5f9940f19caa21fd66e848c9bd42f59",
    ),
    (
        "baseline_wander",
        992,
        "d9bf6bbabb4dfce7aee0a7900b29c0bf913be5dda150d5b4e7914a9248eb8366",
        "368ab6c7b80df3d69d4747e3679c81e5ee2142efa22b51398376db0fe98f36a8",
    ),
    (
        "baseline_shift",
        993,
        "9eba66b379b340a44071deb833be0d0d4342a28f599f2cfc252a95eb88bb7abc",
        "9003c2af028892b54ab8148e83ead24a35b353f6e8b95d7c4512e1f30409e6ca",
    ),
    (
        "random_leads_masking",
        994,
        "e3ec6c83ef3fb95580685b051017018b7bc180aac13cda4b61ad4b506c5e4822",
        "5a4c2ffdd498baae6a8369ee18f59499912928def6b6ff673e43b35aeea60ca1",
    ),
)


def _sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _raw_batch() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, 2 * 1000 * 12, dtype=torch.float32).reshape(
        2, 1000, 12
    )


def _operator_batch() -> torch.Tensor:
    return F.interpolate(
        _raw_batch().transpose(1, 2),
        size=5000,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2).contiguous()


def test_project_random_seed_is_loaded_and_factories_are_reproducible() -> None:
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


def test_torch_operator_owner_exposes_only_the_live_batch_hook() -> None:
    assert torch_operators.__all__ == []
    assert callable(torch_operators.apply_operator_batch_prevalidated)
    assert RETIRED_WRAPPERS.isdisjoint(vars(torch_operators))
    assert {"_validate_ecg", "_adjust_channel_dependency", "_output_tc"}.isdisjoint(
        vars(torch_operators)
    )


@pytest.mark.parametrize("operator,seed,output_sha,rng_sha", KERNEL_GOLDENS)
def test_live_cpu_batch_kernels_match_output_and_rng_goldens(
    operator: str,
    seed: int,
    output_sha: str,
    rng_sha: str,
) -> None:
    signal = _operator_batch()
    original = signal.clone()
    config = load_augmix_config()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = torch_operators.apply_operator_batch_prevalidated(
        operator,
        signal,
        params=config.operator_profile_config.parameters_for(operator),
        sampling_rate_hz=500,
        rng=generator,
    )
    assert output.shape == signal.shape
    assert output.dtype == torch.float32
    assert output.device == signal.device
    assert output.is_contiguous()
    assert torch.isfinite(output).all()
    assert output.data_ptr() != signal.data_ptr()
    assert torch.equal(signal, original)
    assert _sha256_tensor(output) == output_sha
    assert _sha256_tensor(generator.get_state()) == rng_sha


def test_live_batch_hook_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="unknown Torch ECG operator: 'unknown'"):
        torch_operators.apply_operator_batch_prevalidated(
            "unknown",
            _operator_batch(),
            params={},
            sampling_rate_hz=500,
            rng=torch.Generator(device="cpu").manual_seed(1),
        )


@pytest.mark.parametrize("operator,seed,output_sha,rng_sha", KERNEL_GOLDENS)
def test_live_batch_probability_zero_returns_a_float32_copy(
    operator: str,
    seed: int,
    output_sha: str,
    rng_sha: str,
) -> None:
    del output_sha, rng_sha
    signal = _operator_batch().to(dtype=torch.float64)
    params = load_augmix_config().operator_profile_config.parameters_for(operator)
    params["p"] = 0.0
    output = torch_operators.apply_operator_batch_prevalidated(
        operator,
        signal,
        params=params,
        sampling_rate_hz=500,
        rng=torch.Generator(device="cpu").manual_seed(seed),
    )
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert output.data_ptr() != signal.data_ptr()
    torch.testing.assert_close(output, signal.float(), rtol=0.0, atol=0.0)


def test_live_batch_dependency_reconstructs_limb_leads() -> None:
    signal = _operator_batch()
    params = load_augmix_config().operator_profile_config.parameters_for("emg_noise")
    params["dependency"] = True
    output = torch_operators.apply_operator_batch_prevalidated(
        "emg_noise",
        signal,
        params=params,
        sampling_rate_hz=500,
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


def test_live_batch_conditional_masking_has_requested_survivor_counts() -> None:
    signal = torch.ones((16, 5000, 12), dtype=torch.float32)
    params = load_augmix_config().operator_profile_config.parameters_for(
        "random_leads_masking"
    )
    params.update(mask_leads_selection="conditional", mask_leads_condition=(2, 3))
    output = torch_operators.apply_operator_batch_prevalidated(
        "random_leads_masking",
        signal,
        params=params,
        sampling_rate_hz=500,
        rng=torch.Generator(device="cpu").manual_seed(91),
    )
    survivors = torch.any(output != 0.0, dim=1)
    assert torch.all(survivors[:, :6].sum(dim=1) == 4)
    assert torch.all(survivors[:, 6:].sum(dim=1) == 3)


def test_stage1_two_chain_output_and_rng_identity_are_locked() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260813)
    output = generate_two_chain_augmix_strong_view(
        _raw_batch(),
        sampling_rate_hz=100,
        config=load_augmix_config(),
        generator=generator,
    )
    assert _sha256_tensor(output.mixed_raw) == (
        "170fd30f7d43b6be2494fbcafd0fc0a2b6963f8369c27630d4bac570697967cb"
    )
    assert _sha256_tensor(generator.get_state()) == (
        "5157c481374a20f9dff27809fb939452417caa7e819eb42f5d93351e21d7b82a"
    )


RUN_CUDA_TESTS = os.environ.get("ECG_RUN_CUDA_AUG_TESTS") == "1"


@pytest.mark.skipif(
    not RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="set ECG_RUN_CUDA_AUG_TESTS=1 after selecting a free shared-server GPU",
)
@pytest.mark.parametrize("operator,seed,output_sha,rng_sha", KERNEL_GOLDENS)
def test_live_batch_kernel_runs_on_selected_cuda(
    operator: str,
    seed: int,
    output_sha: str,
    rng_sha: str,
) -> None:
    del output_sha, rng_sha
    device = torch.device("cuda:0")
    signal = _operator_batch().to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    config = load_augmix_config()
    output = torch_operators.apply_operator_batch_prevalidated(
        operator,
        signal,
        params=config.operator_profile_config.parameters_for(operator),
        sampling_rate_hz=500,
        rng=generator,
    )
    assert output.shape == signal.shape
    assert output.device == device
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert torch.isfinite(output).all()
