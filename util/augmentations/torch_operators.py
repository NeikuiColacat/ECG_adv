"""Device-native Torch implementations of the five ECG augmentations.

The public parameter names match ``operators.py``. Input and output may be one
ECG ``(time, 12)`` or a true vectorized batch ``(batch, time, 12)`` in PTB-XL
lead order. Random tensors are created directly on the input device, so CUDA
inputs never make a NumPy/CPU round trip. Outputs are new contiguous
``torch.float32`` tensors on the same device as the input.

``rng`` is a required ``torch.Generator`` whose device must match the input
tensor. Create it through ``util.random_seed``. Numeric seeds are reproducible
within a device/backend. CPU and CUDA generators do not promise identical
random streams, while operator equations and parameter semantics remain equal.
"""

from __future__ import annotations

import random

import torch

TorchGenerator = torch.Generator | None


def _validate_ecg(signal: torch.Tensor) -> torch.Tensor:
    """Validate one channel-last 12-lead Torch ECG without modifying it."""

    if not isinstance(signal, torch.Tensor):
        raise TypeError(f"signal must be a torch.Tensor; got {type(signal).__name__}")
    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(
            "signal must have shape (time, 12) in PTB-XL lead order; "
            f"got {tuple(signal.shape)}"
        )
    if signal.shape[0] < 2:
        raise ValueError("signal must contain at least two time samples")
    if not signal.is_floating_point():
        raise TypeError(f"signal must use a floating-point dtype; got {signal.dtype}")
    if not bool(torch.isfinite(signal).all().item()):
        raise ValueError("signal contains NaN or infinity")
    return signal


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

    Public operators always use the full validator.  The canonical corruption
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


def _adjust_channel_dependency(ecg_ct: torch.Tensor) -> torch.Tensor:
    """Functionally reconstruct III/aVR/aVL/aVF from I and II."""

    lead_i = ecg_ct[0:1]
    lead_ii = ecg_ct[1:2]
    derived = torch.cat(
        (
            lead_i,
            lead_ii,
            lead_ii - lead_i,
            -(lead_ii + lead_i) / 2.0,
            lead_i - lead_ii / 2.0,
            lead_ii - lead_i / 2.0,
            ecg_ct[6:],
        ),
        dim=0,
    )
    return derived


def _output_tc(output_ct: torch.Tensor) -> torch.Tensor:
    return output_ct.transpose(0, 1).contiguous().to(dtype=torch.float32)


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
    device validation remain identical to the public functions.
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


def powerline_noise(
    signal: torch.Tensor,
    *,
    max_amplitude: float = 0.5,
    min_amplitude: float = 0.0,
    p: float = 1.0,
    freq: float = 500.0,
    dependency: bool = True,
    rng: TorchGenerator = None,
) -> torch.Tensor:
    """Apply random 50/60 Hz cosine noise directly on the input device."""

    if isinstance(signal, torch.Tensor) and signal.ndim == 3:
        return _powerline_noise_batch(
            signal,
            max_amplitude=max_amplitude,
            min_amplitude=min_amplitude,
            p=p,
            freq=freq,
            dependency=dependency,
            rng=rng,
        )
    tensor = _validate_ecg(signal)
    _validate_common_params(min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p)
    if float(freq) <= 0.0:
        raise ValueError("freq must be positive")
    rng = _resolve_generator(rng, tensor)
    output_ct = tensor.to(dtype=torch.float32).transpose(0, 1).contiguous().clone()

    if float(p) > float(_uniform(0.0, 1.0, (), signal=tensor, rng=rng).item()):
        time_size = int(output_ct.shape[1])
        amplitude = _uniform(
            min_amplitude, max_amplitude, (1, 1), signal=tensor, rng=rng
        )
        powerline_hz = (
            50
            if float(_uniform(0.0, 1.0, (), signal=tensor, rng=rng).item()) > 0.5
            else 60
        )
        time = torch.linspace(
            0.0,
            float(time_size - 1),
            time_size,
            device=tensor.device,
            dtype=torch.float32,
        )
        phase = _uniform(0.0, 2.0 * torch.pi, (), signal=tensor, rng=rng)
        noise = torch.cos(2.0 * torch.pi * powerline_hz * (time / float(freq)) + phase)
        output_ct = output_ct + noise.unsqueeze(0) * amplitude
        if dependency:
            output_ct = _adjust_channel_dependency(output_ct)
    return _output_tc(output_ct)


def emg_noise(
    signal: torch.Tensor,
    *,
    max_amplitude: float = 0.5,
    min_amplitude: float = 0.0,
    dependency: bool = True,
    p: float = 1.0,
    rng: TorchGenerator = None,
) -> torch.Tensor:
    """Apply per-lead Gaussian EMG noise directly on the input device."""

    if isinstance(signal, torch.Tensor) and signal.ndim == 3:
        return _emg_noise_batch(
            signal,
            max_amplitude=max_amplitude,
            min_amplitude=min_amplitude,
            dependency=dependency,
            p=p,
            rng=rng,
        )
    tensor = _validate_ecg(signal)
    _validate_common_params(min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p)
    rng = _resolve_generator(rng, tensor)
    output_ct = tensor.to(dtype=torch.float32).transpose(0, 1).contiguous().clone()

    if float(p) > float(_uniform(0.0, 1.0, (), signal=tensor, rng=rng).item()):
        channel_size, time_size = (int(v) for v in output_ct.shape)
        amplitude = _uniform(
            min_amplitude,
            max_amplitude,
            (channel_size, 1),
            signal=tensor,
            rng=rng,
        )
        noise = _normal(
            0.0,
            1.0,
            (channel_size, time_size),
            signal=tensor,
            rng=rng,
        )
        output_ct = output_ct + noise * amplitude
        if dependency:
            output_ct = _adjust_channel_dependency(output_ct)
    return _output_tc(output_ct)


def baseline_shift(
    signal: torch.Tensor,
    *,
    max_amplitude: float = 0.25,
    min_amplitude: float = 0.0,
    shift_ratio: float = 0.2,
    num_segment: int = 1,
    freq: float = 500.0,
    dependency: bool = False,
    p: float = 1.0,
    amplitude_mode: str = "legacy_upstream",
    rng: TorchGenerator = None,
) -> torch.Tensor:
    """Apply step-segment baseline shift directly on the input device."""

    if isinstance(signal, torch.Tensor) and signal.ndim == 3:
        return _baseline_shift_batch(
            signal,
            max_amplitude=max_amplitude,
            min_amplitude=min_amplitude,
            shift_ratio=shift_ratio,
            num_segment=num_segment,
            freq=freq,
            dependency=dependency,
            p=p,
            amplitude_mode=amplitude_mode,
            rng=rng,
        )
    tensor = _validate_ecg(signal)
    _validate_common_params(min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p)
    if not 0.0 < float(shift_ratio) <= 1.0:
        raise ValueError("shift_ratio must be in (0, 1]")
    if int(num_segment) < 1:
        raise ValueError("num_segment must be >= 1")
    if float(freq) <= 0.0:
        raise ValueError("freq must be positive")
    if amplitude_mode not in {"legacy_upstream", "signed_shared_uniform"}:
        raise ValueError(f"unknown baseline-shift amplitude_mode: {amplitude_mode!r}")
    rng = _resolve_generator(rng, tensor)
    output_ct = tensor.to(dtype=torch.float32).transpose(0, 1).contiguous().clone()

    if float(p) > float(_uniform(0.0, 1.0, (), signal=tensor, rng=rng).item()):
        channel_size, time_size = (int(v) for v in output_ct.shape)
        shift_length = time_size * float(shift_ratio)
        sign_indices = torch.randint(
            0,
            2,
            (channel_size, 1),
            device=tensor.device,
            generator=rng,
        )
        amp_channel = 1.0 - 2.0 * sign_indices.to(dtype=torch.float32)
        amp_general = _uniform(
            min_amplitude, max_amplitude, (1, 1), signal=tensor, rng=rng
        )
        amplitude = (
            amp_channel * amp_general
            if amplitude_mode == "signed_shared_uniform"
            else amp_channel - amp_general
        )
        noise = torch.zeros_like(output_ct)
        for _ in range(int(num_segment)):
            segment_len = float(
                _normal(
                    shift_length,
                    shift_length * 0.2,
                    (),
                    signal=tensor,
                    rng=rng,
                ).item()
            )
            start = int(
                float(
                    _uniform(
                        0.0,
                        time_size - segment_len,
                        (),
                        signal=tensor,
                        rng=rng,
                    ).item()
                )
            )
            stop = int(start + segment_len)
            noise[:, start:stop] = 1.0
        output_ct = output_ct + noise * amplitude
        if dependency:
            output_ct = _adjust_channel_dependency(output_ct)
    return _output_tc(output_ct)


def baseline_wander(
    signal: torch.Tensor,
    *,
    max_amplitude: float = 0.5,
    min_amplitude: float = 0.0,
    p: float = 1.0,
    max_freq: float = 0.2,
    min_freq: float = 0.01,
    k: int = 3,
    freq: float = 500.0,
    dependency: bool = True,
    rng: TorchGenerator = None,
) -> torch.Tensor:
    """Apply summed-cosine baseline wander directly on the input device."""

    if isinstance(signal, torch.Tensor) and signal.ndim == 3:
        return _baseline_wander_batch(
            signal,
            max_amplitude=max_amplitude,
            min_amplitude=min_amplitude,
            p=p,
            max_freq=max_freq,
            min_freq=min_freq,
            k=k,
            freq=freq,
            dependency=dependency,
            rng=rng,
        )
    tensor = _validate_ecg(signal)
    _validate_common_params(min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p)
    if float(min_freq) <= 0.0 or float(max_freq) < float(min_freq):
        raise ValueError("frequency bounds must satisfy 0 < min_freq <= max_freq")
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    if float(freq) <= 0.0:
        raise ValueError("freq must be positive")
    rng = _resolve_generator(rng, tensor)
    output_ct = tensor.to(dtype=torch.float32).transpose(0, 1).contiguous().clone()

    if float(p) > float(_uniform(0.0, 1.0, (), signal=tensor, rng=rng).item()):
        channel_size, time_size = (int(v) for v in output_ct.shape)
        amp_channel = _normal(
            1.0, 0.5, (channel_size, 1), signal=tensor, rng=rng
        )
        amp_general = _uniform(
            min_amplitude, max_amplitude, (int(k),), signal=tensor, rng=rng
        )
        noise = torch.zeros((1, time_size), device=tensor.device, dtype=torch.float32)
        time = torch.linspace(
            0.0,
            float(time_size - 1),
            time_size,
            device=tensor.device,
            dtype=torch.float32,
        )
        for component in range(int(k)):
            component_freq = _uniform(
                min_freq, max_freq, (), signal=tensor, rng=rng
            )
            phase = _uniform(0.0, 2.0 * torch.pi, (), signal=tensor, rng=rng)
            noise = noise + (
                torch.cos(
                    2.0 * torch.pi * component_freq * (time / float(freq)) + phase
                ).unsqueeze(0)
                * amp_general[component]
            )
        output_ct = output_ct + noise * amp_channel
        if dependency:
            output_ct = _adjust_channel_dependency(output_ct)
    return _output_tc(output_ct)


def random_leads_masking(
    signal: torch.Tensor,
    *,
    p: float = 1.0,
    mask_leads_selection: str = "random",
    mask_leads_prob: float = 0.5,
    mask_leads_condition: tuple[int, int] | None = None,
    ensure_at_least_one_lead: bool = False,
    rng: TorchGenerator = None,
    python_rng: random.Random | None = None,
) -> torch.Tensor:
    """Mask random or conditional leads directly on the input device."""

    if isinstance(signal, torch.Tensor) and signal.ndim == 3:
        return _random_leads_masking_batch(
            signal,
            p=p,
            mask_leads_selection=mask_leads_selection,
            mask_leads_prob=mask_leads_prob,
            mask_leads_condition=mask_leads_condition,
            ensure_at_least_one_lead=ensure_at_least_one_lead,
            rng=rng,
        )
    tensor = _validate_ecg(signal)
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if not 0.0 <= float(mask_leads_prob) <= 1.0:
        raise ValueError("mask_leads_prob must be in [0, 1]")
    if mask_leads_selection not in {"random", "conditional"}:
        raise ValueError(
            "mask_leads_selection must be either 'random' or 'conditional'"
        )
    rng = _resolve_generator(rng, tensor)
    python_random = python_rng
    output_ct = tensor.to(dtype=torch.float32).transpose(0, 1).contiguous().clone()

    if float(p) >= float(_uniform(0.0, 1.0, (), signal=tensor, rng=rng).item()):
        if mask_leads_selection == "random":
            survivors = _uniform(0.0, 1.0, (12,), signal=tensor, rng=rng) >= float(
                mask_leads_prob
            )
            if bool(ensure_at_least_one_lead) and not bool(survivors.any().item()):
                survivor = int(
                    torch.randint(
                        0,
                        12,
                        (),
                        device=tensor.device,
                        generator=rng,
                    ).item()
                )
                survivors[survivor] = True
        else:
            if python_random is None:
                raise ValueError(
                    "python_rng is required for conditional lead masking"
                )
            if mask_leads_condition is None:
                raise ValueError(
                    "mask_leads_condition is required for conditional masking"
                )
            n_limb, n_chest = (int(value) for value in mask_leads_condition)
            if not (0 <= n_limb <= 6 and 0 <= n_chest <= 6):
                raise ValueError(
                    "conditional masked-lead counts must both be in [0, 6]"
                )
            limb_survivors = python_random.sample(list(range(6)), 6 - n_limb)
            chest_survivors = [
                lead + 6
                for lead in python_random.sample(list(range(6)), 6 - n_chest)
            ]
            survivors = torch.zeros(12, device=tensor.device, dtype=torch.bool)
            survivors[limb_survivors + chest_survivors] = True
        output_ct = output_ct * survivors.to(dtype=output_ct.dtype).unsqueeze(1)
    return _output_tc(output_ct)


__all__ = [
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
]
