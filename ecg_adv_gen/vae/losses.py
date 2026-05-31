"""Loss and signal-consistency helpers for the 500 Hz ECG VAE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VAE500LossOutput:
    loss: torch.Tensor
    recon_huber: torch.Tensor
    kl: torch.Tensor
    first_diff_huber: torch.Tensor
    lead_consistency_huber: torch.Tensor


def lead_consistency_residual(x: torch.Tensor) -> torch.Tensor:
    """Return simple limb-lead consistency residuals for PTB-XL lead order.

    Lead order is ``I, II, III, aVR, aVL, aVF, V1..V6``. These residuals are
    not a full clinical validator; they are a cheap training/audit signal that
    catches obvious lead-order or decode corruption.
    """
    if x.ndim != 3 or x.shape[-1] < 6:
        raise ValueError(f"expected ECG shape (B, L, >=6), got {tuple(x.shape)}")
    lead_i = x[..., 0]
    lead_ii = x[..., 1]
    lead_iii = x[..., 2]
    avr = x[..., 3]
    avl = x[..., 4]
    avf = x[..., 5]
    residuals = (
        lead_iii - (lead_ii - lead_i),
        avr + 0.5 * (lead_i + lead_ii),
        avl - (lead_i - 0.5 * lead_ii),
        avf - (lead_ii - 0.5 * lead_i),
    )
    return torch.stack(residuals, dim=-1)


def _huber(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(input_tensor, target_tensor, reduction="mean")


def vae500_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    *,
    beta: float = 1e-4,
    first_diff_weight: float = 0.05,
    lead_consistency_weight: float = 0.02,
) -> VAE500LossOutput:
    """Compute the planned VAE500 objective.

    The reconstruction terms operate on normalized ECG tensors. Raw-mV audit
    statistics should still be computed separately from saved metadata.
    """
    if recon.shape != target.shape:
        raise ValueError(f"recon shape {tuple(recon.shape)} does not match target shape {tuple(target.shape)}")
    recon_huber = _huber(recon, target)
    kl = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())
    first_diff_huber = _huber(recon[:, 1:] - recon[:, :-1], target[:, 1:] - target[:, :-1])
    lead_consistency_huber = _huber(lead_consistency_residual(recon), lead_consistency_residual(target))
    loss = (
        recon_huber
        + float(beta) * kl
        + float(first_diff_weight) * first_diff_huber
        + float(lead_consistency_weight) * lead_consistency_huber
    )
    return VAE500LossOutput(
        loss=loss,
        recon_huber=recon_huber.detach(),
        kl=kl.detach(),
        first_diff_huber=first_diff_huber.detach(),
        lead_consistency_huber=lead_consistency_huber.detach(),
    )


__all__ = ["VAE500LossOutput", "lead_consistency_residual", "vae500_loss"]
