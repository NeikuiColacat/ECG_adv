"""Torch ECGFounder preprocessing helpers.

Keep these helpers out of ``ecg_adv_gen.models.__init__`` so lightweight config
and path-contract imports do not load torch unless a runner explicitly needs
model execution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_LIMB_LEAD_COEFFS = torch.tensor(
    [
        [1.0, 0.0],      # I
        [0.0, 1.0],      # II
        [-1.0, 1.0],     # III = II - I
        [-0.5, -0.5],    # aVR = -(I + II) / 2
        [1.0, -0.5],     # aVL = I - II / 2
        [-0.5, 1.0],     # aVF = II - I / 2
    ],
    dtype=torch.float32,
)


def global_zscore_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply per-sample global z-score normalization over all non-batch values."""
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp(min=eps)
    return (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)


def fft_bandpass_torch(
    x: torch.Tensor,
    *,
    sample_rate_hz: float,
    low_hz: float | None = None,
    high_hz: float | None = None,
) -> torch.Tensor:
    """Apply an FFT-domain bandpass filter along the time axis."""
    if x.ndim != 3:
        raise ValueError(f"expected ECG tensor with shape (B, C, T), got {tuple(x.shape)}")
    if low_hz is None and high_hz is None:
        return x
    sample_rate_hz = float(sample_rate_hz)
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    low = 0.0 if low_hz is None else float(low_hz)
    high = sample_rate_hz / 2.0 if high_hz is None else float(high_hz)
    if low < 0 or high <= 0 or low >= high:
        raise ValueError(f"invalid bandpass range: low_hz={low_hz}, high_hz={high_hz}")

    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sample_rate_hz).to(x.device)
    mask = (freqs >= low) & (freqs <= high)
    spectrum = torch.fft.rfft(x, dim=-1)
    spectrum = spectrum * mask.to(dtype=spectrum.dtype).view(1, 1, -1)
    return torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1).to(dtype=x.dtype)


def _flat_lead_mask(x: torch.Tensor, eps: float) -> torch.Tensor:
    p2p = x.amax(dim=-1) - x.amin(dim=-1)
    std = x.std(dim=-1)
    return (p2p <= float(eps)) | (std <= float(eps))


def repair_flat_ecg_leads_torch(x: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Repair flat/masked 12-lead ECG channels from redundant lead relations.

    Limb leads are reconstructed by least-squares from the available
    Einthoven/Goldberger relations. Precordial flat leads are filled from the
    nearest non-flat chest lead, or the average of nearest left/right neighbors.
    Only leads detected as flat are overwritten.
    """
    if x.ndim != 3:
        raise ValueError(f"expected ECG tensor with shape (B, 12, T), got {tuple(x.shape)}")
    if x.shape[1] != 12:
        raise ValueError(f"expected 12 ECG leads, got {x.shape[1]}")

    out = x.clone()
    flat = _flat_lead_mask(out, eps)
    coeffs = _LIMB_LEAD_COEFFS.to(device=out.device, dtype=out.dtype)
    for b in range(out.shape[0]):
        limb_flat = flat[b, :6]
        limb_obs = ~limb_flat
        if bool(limb_flat.any()) and int(limb_obs.sum()) >= 2:
            a_obs = coeffs[limb_obs]
            if int(torch.linalg.matrix_rank(a_obs).item()) >= 2:
                base = torch.linalg.lstsq(a_obs, out[b, :6][limb_obs]).solution
                recon = (coeffs @ base).to(dtype=out.dtype)
                out[b, :6][limb_flat] = recon[limb_flat]

        chest_flat = flat[b, 6:12]
        if bool(chest_flat.any()):
            chest = out[b, 6:12]
            nonflat_idx = torch.nonzero(~chest_flat, as_tuple=False).flatten()
            if int(nonflat_idx.numel()) == 0:
                continue
            for local_idx in torch.nonzero(chest_flat, as_tuple=False).flatten():
                left = nonflat_idx[nonflat_idx < local_idx]
                right = nonflat_idx[nonflat_idx > local_idx]
                if int(left.numel()) and int(right.numel()):
                    li = left[-1]
                    ri = right[0]
                    chest[local_idx] = 0.5 * (chest[li] + chest[ri])
                elif int(left.numel()):
                    chest[local_idx] = chest[left[-1]]
                else:
                    chest[local_idx] = chest[right[0]]
    return out


def stabilize_ecg_torch(
    x: torch.Tensor,
    *,
    sample_rate_hz: float,
    bandpass_low_hz: float | None = None,
    bandpass_high_hz: float | None = None,
    repair_flat_leads: bool = False,
    clip_abs: float | None = None,
) -> torch.Tensor:
    """Apply optional ECGFounder input stabilizers before model normalization."""
    y = fft_bandpass_torch(
        x,
        sample_rate_hz=sample_rate_hz,
        low_hz=bandpass_low_hz,
        high_hz=bandpass_high_hz,
    )
    if repair_flat_leads:
        y = repair_flat_ecg_leads_torch(y)
    if clip_abs is not None and float(clip_abs) > 0:
        y = y.clamp(min=-float(clip_abs), max=float(clip_abs))
    return y


def ecg1000_to_ecgfounder_input(
    ecg_ct_1000: torch.Tensor,
    *,
    target_points: int = 5000,
    eps: float = 1e-8,
    bandpass_low_hz: float | None = None,
    bandpass_high_hz: float | None = None,
    input_sample_rate_hz: float = 100.0,
    repair_flat_leads: bool = False,
    clip_abs: float | None = None,
    apply_global_zscore: bool = True,
) -> torch.Tensor:
    """Convert channel-time 100Hz ECG tensors to ECGFounder 500Hz normalized input."""
    ecg_ct_1000 = stabilize_ecg_torch(
        ecg_ct_1000,
        sample_rate_hz=float(input_sample_rate_hz),
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        repair_flat_leads=repair_flat_leads,
        clip_abs=clip_abs,
    )
    x = F.interpolate(
        ecg_ct_1000,
        size=int(target_points),
        mode="linear",
        align_corners=True,
    )
    if apply_global_zscore:
        return global_zscore_torch(x, eps=eps)
    return x
