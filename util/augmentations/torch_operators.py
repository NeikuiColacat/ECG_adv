"""Device-native batched Torch kernels for the five ECG augmentations.

The canonical corruption pipeline validates its full batch once, then applies
these kernels directly on the input device with an isolated ``torch.Generator``.
Outputs are new contiguous ``torch.float32`` tensors on the same device.
"""

from __future__ import annotations

import torch

TorchGenerator = torch.Generator | None


def _validate_ecg_batch(signal: torch.Tensor) -> torch.Tensor:
    """Validate a channel-last ECG batch without modifying it."""

    if not isinstance(signal, torch.Tensor):
        raise TypeError(f"signal must be a torch.Tensor; got {type(signal).__name__}")
    if signal.ndim != 3 or signal.shape[0] < 1 or signal.shape[2] != 12:
        raise ValueError(
            "signal must have shape (batch, time, 12) in PTB-XL lead order; "
            f"got {tuple(signal.shape)}"
        )
    if signal.shape[1] < 2:
        raise ValueError("signal must contain at least two time samples")
    if not signal.is_floating_point():
        raise TypeError(f"signal must use a floating-point dtype; got {signal.dtype}")
    if not bool(torch.isfinite(signal).all().item()):
        raise ValueError("signal contains NaN or infinity")
    return signal


def _prepare_ecg_batch(
    signal: torch.Tensor,
    *,
    prevalidated: bool,
) -> torch.Tensor:
    """Avoid repeated CUDA finite-value synchronizations inside one pipeline.

    Standalone batch kernels use the full validator.  The canonical corruption
    kernel validates its complete batch once, then calls the private batched
    implementations with ``prevalidated=True`` while it applies multiple
    operators to the same tensor.
    """

    return signal if prevalidated else _validate_ecg_batch(signal)


def _validate_common_params(
    *,
    min_amplitude: float,
    max_amplitude: float,
    p: float,
) -> None:
    if float(min_amplitude) < 0.0:
        raise ValueError("min_amplitude must be non-negative")
    if float(max_amplitude) < float(min_amplitude):
        raise ValueError("max_amplitude must be >= min_amplitude")
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("p must be in [0, 1]")


def _resolve_generator(
    rng: TorchGenerator,
    signal: torch.Tensor,
) -> torch.Generator:
    if rng is None:
        raise ValueError(
            "rng is required; create an isolated stream with util.random_seed"
        )
    if not isinstance(rng, torch.Generator):
        raise TypeError("rng must be a torch.Generator or None")
    generator_device = torch.device(rng.device)
    if generator_device.type != signal.device.type:
        raise ValueError(
            f"rng device {generator_device} must match signal device {signal.device}"
        )
    if generator_device.type == "cuda":
        generator_index = (
            torch.cuda.current_device()
            if generator_device.index is None
            else generator_device.index
        )
        signal_index = (
            torch.cuda.current_device()
            if signal.device.index is None
            else signal.device.index
        )
        if generator_index != signal_index:
            raise ValueError(
                f"rng device {generator_device} must match signal device {signal.device}"
            )
    return rng


def _rand(
    shape: tuple[int, ...] | list[int] | torch.Size,
    *,
    signal: torch.Tensor,
    rng: TorchGenerator,
) -> torch.Tensor:
    return torch.rand(
        shape,
        device=signal.device,
        dtype=torch.float32,
        generator=rng,
    )


def _randn(
    shape: tuple[int, ...] | list[int] | torch.Size,
    *,
    signal: torch.Tensor,
    rng: TorchGenerator,
) -> torch.Tensor:
    return torch.randn(
        shape,
        device=signal.device,
        dtype=torch.float32,
        generator=rng,
    )


def _uniform(
    low: float,
    high: float,
    shape: tuple[int, ...] | list[int] | torch.Size,
    *,
    signal: torch.Tensor,
    rng: TorchGenerator,
) -> torch.Tensor:
    return _rand(shape, signal=signal, rng=rng) * (float(high) - float(low)) + float(low)


def _normal(
    mean: float,
    std: float,
    shape: tuple[int, ...] | list[int] | torch.Size,
    *,
    signal: torch.Tensor,
    rng: TorchGenerator,
) -> torch.Tensor:
    return _randn(shape, signal=signal, rng=rng) * float(std) + float(mean)


def _adjust_channel_dependency_batch(ecg_bct: torch.Tensor) -> torch.Tensor:
    """Functionally reconstruct III/aVR/aVL/aVF for every batch item."""

    lead_i = ecg_bct[:, 0:1]
    lead_ii = ecg_bct[:, 1:2]
    return torch.cat(
        (
            lead_i,
            lead_ii,
            lead_ii - lead_i,
            -(lead_ii + lead_i) / 2.0,
            lead_i - lead_ii / 2.0,
            lead_ii - lead_i / 2.0,
            ecg_bct[:, 6:],
        ),
        dim=1,
    )


def _output_btc(output_bct: torch.Tensor) -> torch.Tensor:
    return output_bct.transpose(1, 2).contiguous().to(dtype=torch.float32)


def _powerline_noise_batch(
    signal: torch.Tensor,
    *,
    max_amplitude: float,
    min_amplitude: float,
    p: float,
    freq: float,
    dependency: bool,
    rng: TorchGenerator,
    _prevalidated: bool = False,
) -> torch.Tensor:
    tensor = _prepare_ecg_batch(signal, prevalidated=_prevalidated)
    _validate_common_params(
        min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p
    )
    if float(freq) <= 0.0:
        raise ValueError("freq must be positive")
    rng = _resolve_generator(rng, tensor)
    output_bct = tensor.to(dtype=torch.float32).transpose(1, 2).contiguous().clone()
    batch_size, _, time_size = (int(value) for value in output_bct.shape)
    apply_mask = _uniform(
        0.0, 1.0, (batch_size, 1, 1), signal=tensor, rng=rng
    ) < float(p)
    amplitude = _uniform(
        min_amplitude,
        max_amplitude,
        (batch_size, 1, 1),
        signal=tensor,
        rng=rng,
    )
    powerline_hz = torch.where(
        _uniform(0.0, 1.0, (batch_size, 1, 1), signal=tensor, rng=rng) > 0.5,
        50.0,
        60.0,
    )
    phase = _uniform(
        0.0,
        2.0 * torch.pi,
        (batch_size, 1, 1),
        signal=tensor,
        rng=rng,
    )
    time = torch.arange(
        time_size, device=tensor.device, dtype=torch.float32
    ).view(1, 1, time_size)
    noise = torch.cos(
        2.0 * torch.pi * powerline_hz * (time / float(freq)) + phase
    )
    candidate = output_bct + noise * amplitude
    if dependency:
        candidate = _adjust_channel_dependency_batch(candidate)
    return _output_btc(torch.where(apply_mask, candidate, output_bct))


def _emg_noise_batch(
    signal: torch.Tensor,
    *,
    max_amplitude: float,
    min_amplitude: float,
    dependency: bool,
    p: float,
    rng: TorchGenerator,
    _prevalidated: bool = False,
) -> torch.Tensor:
    tensor = _prepare_ecg_batch(signal, prevalidated=_prevalidated)
    _validate_common_params(
        min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p
    )
    rng = _resolve_generator(rng, tensor)
    output_bct = tensor.to(dtype=torch.float32).transpose(1, 2).contiguous().clone()
    batch_size, channel_size, time_size = (
        int(value) for value in output_bct.shape
    )
    apply_mask = _uniform(
        0.0, 1.0, (batch_size, 1, 1), signal=tensor, rng=rng
    ) < float(p)
    amplitude = _uniform(
        min_amplitude,
        max_amplitude,
        (batch_size, channel_size, 1),
        signal=tensor,
        rng=rng,
    )
    noise = _normal(
        0.0,
        1.0,
        (batch_size, channel_size, time_size),
        signal=tensor,
        rng=rng,
    )
    candidate = output_bct + noise * amplitude
    if dependency:
        candidate = _adjust_channel_dependency_batch(candidate)
    return _output_btc(torch.where(apply_mask, candidate, output_bct))


def _baseline_shift_batch(
    signal: torch.Tensor,
    *,
    max_amplitude: float,
    min_amplitude: float,
    shift_ratio: float,
    num_segment: int,
    freq: float,
    dependency: bool,
    p: float,
    amplitude_mode: str,
    rng: TorchGenerator,
    _prevalidated: bool = False,
) -> torch.Tensor:
    tensor = _prepare_ecg_batch(signal, prevalidated=_prevalidated)
    _validate_common_params(
        min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p
    )
    if not 0.0 < float(shift_ratio) <= 1.0:
        raise ValueError("shift_ratio must be in (0, 1]")
    if int(num_segment) < 1:
        raise ValueError("num_segment must be >= 1")
    if float(freq) <= 0.0:
        raise ValueError("freq must be positive")
    if amplitude_mode not in {"legacy_upstream", "signed_shared_uniform"}:
        raise ValueError(f"unknown baseline-shift amplitude_mode: {amplitude_mode!r}")
    rng = _resolve_generator(rng, tensor)
    output_bct = tensor.to(dtype=torch.float32).transpose(1, 2).contiguous().clone()
    batch_size, channel_size, time_size = (
        int(value) for value in output_bct.shape
    )
    apply_mask = _uniform(
        0.0, 1.0, (batch_size, 1, 1), signal=tensor, rng=rng
    ) < float(p)
    sign_indices = torch.randint(
        0,
        2,
        (batch_size, channel_size, 1),
        device=tensor.device,
        generator=rng,
    )
    amp_channel = 1.0 - 2.0 * sign_indices.to(dtype=torch.float32)
    amp_general = _uniform(
        min_amplitude,
        max_amplitude,
        (batch_size, 1, 1),
        signal=tensor,
        rng=rng,
    )
    amplitude = (
        amp_channel * amp_general
        if amplitude_mode == "signed_shared_uniform"
        else amp_channel - amp_general
    )
    shift_length = time_size * float(shift_ratio)
    segment_lengths = _normal(
        shift_length,
        shift_length * 0.2,
        (batch_size, int(num_segment)),
        signal=tensor,
        rng=rng,
    ).clamp(min=1.0, max=float(time_size))
    starts = (
        _rand((batch_size, int(num_segment)), signal=tensor, rng=rng)
        * (float(time_size) - segment_lengths)
    ).trunc().to(dtype=torch.long)
    stops = (starts.to(dtype=torch.float32) + segment_lengths).trunc().to(
        dtype=torch.long
    )
    positions = torch.arange(time_size, device=tensor.device).view(1, 1, time_size)
    segment_mask = (
        (positions >= starts.unsqueeze(-1)) & (positions < stops.unsqueeze(-1))
    ).any(dim=1, keepdim=True)
    candidate = output_bct + segment_mask.to(dtype=torch.float32) * amplitude
    if dependency:
        candidate = _adjust_channel_dependency_batch(candidate)
    return _output_btc(torch.where(apply_mask, candidate, output_bct))


def _baseline_wander_batch(
    signal: torch.Tensor,
    *,
    max_amplitude: float,
    min_amplitude: float,
    p: float,
    max_freq: float,
    min_freq: float,
    k: int,
    freq: float,
    dependency: bool,
    rng: TorchGenerator,
    _prevalidated: bool = False,
) -> torch.Tensor:
    tensor = _prepare_ecg_batch(signal, prevalidated=_prevalidated)
    _validate_common_params(
        min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p
    )
    if float(min_freq) <= 0.0 or float(max_freq) < float(min_freq):
        raise ValueError("frequency bounds must satisfy 0 < min_freq <= max_freq")
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    if float(freq) <= 0.0:
        raise ValueError("freq must be positive")
    rng = _resolve_generator(rng, tensor)
    output_bct = tensor.to(dtype=torch.float32).transpose(1, 2).contiguous().clone()
    batch_size, channel_size, time_size = (
        int(value) for value in output_bct.shape
    )
    apply_mask = _uniform(
        0.0, 1.0, (batch_size, 1, 1), signal=tensor, rng=rng
    ) < float(p)
    amp_channel = _normal(
        1.0,
        0.5,
        (batch_size, channel_size, 1),
        signal=tensor,
        rng=rng,
    )
    amp_general = _uniform(
        min_amplitude,
        max_amplitude,
        (batch_size, int(k), 1),
        signal=tensor,
        rng=rng,
    )
    component_freq = _uniform(
        min_freq,
        max_freq,
        (batch_size, int(k), 1),
        signal=tensor,
        rng=rng,
    )
    phase = _uniform(
        0.0,
        2.0 * torch.pi,
        (batch_size, int(k), 1),
        signal=tensor,
        rng=rng,
    )
    time = torch.arange(
        time_size, device=tensor.device, dtype=torch.float32
    ).view(1, 1, time_size)
    noise = (
        torch.cos(
            2.0 * torch.pi * component_freq * (time / float(freq)) + phase
        )
        * amp_general
    ).sum(dim=1, keepdim=True)
    candidate = output_bct + noise * amp_channel
    if dependency:
        candidate = _adjust_channel_dependency_batch(candidate)
    return _output_btc(torch.where(apply_mask, candidate, output_bct))


def _random_leads_masking_batch(
    signal: torch.Tensor,
    *,
    p: float,
    mask_leads_selection: str,
    mask_leads_prob: float,
    mask_leads_condition: tuple[int, int] | None,
    ensure_at_least_one_lead: bool,
    rng: TorchGenerator,
    _prevalidated: bool = False,
) -> torch.Tensor:
    tensor = _prepare_ecg_batch(signal, prevalidated=_prevalidated)
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if not 0.0 <= float(mask_leads_prob) <= 1.0:
        raise ValueError("mask_leads_prob must be in [0, 1]")
    if mask_leads_selection not in {"random", "conditional"}:
        raise ValueError(
            "mask_leads_selection must be either 'random' or 'conditional'"
        )
    rng = _resolve_generator(rng, tensor)
    output_bct = tensor.to(dtype=torch.float32).transpose(1, 2).contiguous().clone()
    batch_size = int(output_bct.shape[0])
    apply_mask = _uniform(
        0.0, 1.0, (batch_size, 1, 1), signal=tensor, rng=rng
    ) <= float(p)
    if mask_leads_selection == "random":
        survivors = _uniform(
            0.0, 1.0, (batch_size, 12), signal=tensor, rng=rng
        ) >= float(mask_leads_prob)
        if ensure_at_least_one_lead:
            empty = ~survivors.any(dim=1, keepdim=True)
            fallback_indices = torch.randint(
                0,
                12,
                (batch_size, 1),
                device=tensor.device,
                generator=rng,
            )
            fallback = torch.zeros_like(survivors).scatter(
                1, fallback_indices, True
            )
            survivors = torch.where(empty, fallback, survivors)
    else:
        if mask_leads_condition is None:
            raise ValueError("mask_leads_condition is required for conditional masking")
        n_limb, n_chest = (int(value) for value in mask_leads_condition)
        if not (0 <= n_limb <= 6 and 0 <= n_chest <= 6):
            raise ValueError("conditional masked-lead counts must both be in [0, 6]")
        limb_order = _rand((batch_size, 6), signal=tensor, rng=rng).argsort(dim=1)
        chest_order = _rand((batch_size, 6), signal=tensor, rng=rng).argsort(dim=1)
        limb = torch.zeros(
            (batch_size, 6), device=tensor.device, dtype=torch.bool
        ).scatter(1, limb_order[:, : 6 - n_limb], True)
        chest = torch.zeros(
            (batch_size, 6), device=tensor.device, dtype=torch.bool
        ).scatter(1, chest_order[:, : 6 - n_chest], True)
        survivors = torch.cat((limb, chest), dim=1)
    candidate = output_bct * survivors.to(dtype=torch.float32).unsqueeze(-1)
    return _output_btc(torch.where(apply_mask, candidate, output_bct))


def apply_operator_batch_prevalidated(
    operator: str,
    signal: torch.Tensor,
    *,
    params: dict[str, object],
    sampling_rate_hz: int,
    rng: TorchGenerator,
) -> torch.Tensor:
    """Apply one named batched operator after an outer contract validation.

    This is an internal pipeline hook, not an alternative public operator API.
    It intentionally skips only the tensor-wide finite check; parameter and RNG
    device validation remain active.
    """

    kwargs = dict(params)
    if operator in {"powerline_noise", "baseline_wander", "baseline_shift"}:
        kwargs["freq"] = float(sampling_rate_hz)
    kwargs["rng"] = rng
    kwargs["_prevalidated"] = True
    if operator == "powerline_noise":
        return _powerline_noise_batch(signal, **kwargs)
    if operator == "emg_noise":
        return _emg_noise_batch(signal, **kwargs)
    if operator == "baseline_wander":
        return _baseline_wander_batch(signal, **kwargs)
    if operator == "baseline_shift":
        return _baseline_shift_batch(signal, **kwargs)
    if operator == "random_leads_masking":
        kwargs.setdefault("mask_leads_condition", None)
        return _random_leads_masking_batch(signal, **kwargs)
    raise ValueError(f"unknown Torch ECG operator: {operator!r}")


__all__: list[str] = []
