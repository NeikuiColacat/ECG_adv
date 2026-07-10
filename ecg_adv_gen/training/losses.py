"""Loss helpers for multi-label ECG classification.

These functions preserve the legacy Super5/PTB-XL trainer behavior while
making the implementation importable without depending on dated scripts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels_mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    """BCE-with-logits that ignores label entries marked as ``-1``."""
    mask = (labels_mask >= 0).float()
    labels_safe = torch.where(mask.bool(), labels_mask, torch.zeros_like(labels_mask))
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels_safe,
        pos_weight=pos_weight,
        reduction="none",
    )
    denom = mask.sum().clamp_min(1.0)
    return (bce * mask).sum() / denom


def masked_bce_per_sample(
    logits: torch.Tensor,
    labels_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample masked BCE-with-logits for labels where ``-1`` means unknown."""
    mask = (labels_mask >= 0).float()
    labels_safe = torch.where(mask.bool(), labels_mask, torch.zeros_like(labels_mask))
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels_safe,
        pos_weight=pos_weight,
        reduction="none",
    )
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (bce * mask).sum(dim=1) / denom


def stream_weighted_masked_bce(
    logits: torch.Tensor,
    labels_mask: torch.Tensor,
    stream_ids: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    *,
    source_weight: float = 1.0,
    target_real_weight: float = 1.0,
    adv_weight: float = 1.0,
) -> torch.Tensor:
    """Masked BCE loss with per-row weights selected by stream id.

    Stream ids follow the active ECGFounder full-FT convention:
    0 = PTB-XL source, 1 = target real K-shot, 2 = adversarial/augmented.
    Unknown stream ids receive weight 1.0, matching the legacy inline default.
    """
    per_sample = masked_bce_per_sample(logits, labels_mask, pos_weight)
    stream_weights = torch.ones_like(per_sample)
    stream_weights = torch.where(
        stream_ids == 0,
        torch.full_like(stream_weights, float(source_weight)),
        stream_weights,
    )
    stream_weights = torch.where(
        stream_ids == 1,
        torch.full_like(stream_weights, float(target_real_weight)),
        stream_weights,
    )
    stream_weights = torch.where(
        stream_ids == 2,
        torch.full_like(stream_weights, float(adv_weight)),
        stream_weights,
    )
    return (per_sample * stream_weights).sum() / stream_weights.sum().clamp_min(1e-6)


def multilabel_bce_per_sample(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample mean BCE-with-logits for ordinary 0/1 multi-label targets."""
    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    ).mean(dim=1)


def target_clean_adv_objective(
    clean_losses: torch.Tensor,
    adv_losses: torch.Tensor,
    rho: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Combine clean and VAE-adversarial target losses with an explicit coefficient."""
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if clean_losses.numel() == 0 or adv_losses.numel() == 0:
        raise ValueError("clean and adversarial losses must be nonempty")

    clean_mean = clean_losses.mean()
    adv_mean = adv_losses.mean()
    clean_weighted = (1.0 - rho) * clean_mean
    adv_weighted = rho * adv_mean
    objective = clean_weighted + adv_weighted
    total = float(objective.detach().item())
    clean_value = float(clean_weighted.detach().item())
    adv_value = float(adv_weighted.detach().item())
    return objective, {
        "target_clean_count": int(clean_losses.numel()),
        "target_adv_count": int(adv_losses.numel()),
        "target_clean_loss_mean": float(clean_mean.detach().item()),
        "target_adv_loss_mean": float(adv_mean.detach().item()),
        "target_clean_weighted_loss": clean_value,
        "target_adv_weighted_loss": adv_value,
        "target_clean_nominal_fraction": 1.0 - rho,
        "target_adv_nominal_fraction": rho,
        "target_clean_contribution_fraction": clean_value / total if total else 0.0,
        "target_adv_contribution_fraction": adv_value / total if total else 0.0,
    }


@torch.no_grad()
def attack_success_stats(
    clean_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
) -> dict[str, float]:
    """Untargeted multi-label attack success against the BCE objective.

    This mirrors the ECGFounder VAE-LHAT legacy diagnostic: an adversarial
    sample succeeds when its per-sample BCE loss gain exceeds ``margin``.
    """
    clean_loss = multilabel_bce_per_sample(clean_logits, labels)
    adv_loss = multilabel_bce_per_sample(adv_logits, labels)
    gain = adv_loss - clean_loss
    success = gain > float(margin)

    clean_prob = torch.sigmoid(clean_logits)
    adv_prob = torch.sigmoid(adv_logits)
    pos_mask = labels > 0.5
    neg_mask = labels <= 0.0
    pos_drop = (clean_prob - adv_prob)[pos_mask]
    neg_rise = (adv_prob - clean_prob)[neg_mask]
    return {
        "n": float(labels.shape[0]),
        "success_count": float(success.float().sum().item()),
        "success_rate": float(success.float().mean().item()),
        "clean_loss_mean": float(clean_loss.mean().item()),
        "adv_loss_mean": float(adv_loss.mean().item()),
        "loss_gain_mean": float(gain.mean().item()),
        "loss_gain_median": float(gain.median().item()),
        "loss_gain_p10": float(torch.quantile(gain, 0.10).item()),
        "loss_gain_p90": float(torch.quantile(gain, 0.90).item()),
        "pos_prob_drop_mean": float(pos_drop.mean().item()) if pos_drop.numel() else float("nan"),
        "neg_prob_rise_mean": float(neg_rise.mean().item()) if neg_rise.numel() else float("nan"),
    }


def merge_attack_success_stats(stats: list[dict[str, float]]) -> dict[str, float | None]:
    """Merge per-batch attack-success diagnostics with sample-count weights."""
    if not stats:
        return {
            "success_rate": None,
            "loss_gain_mean": None,
            "loss_gain_median_mean": None,
            "clean_loss_mean": None,
            "adv_loss_mean": None,
            "pos_prob_drop_mean": None,
            "neg_prob_rise_mean": None,
        }
    n_total = sum(float(s["n"]) for s in stats)
    if n_total <= 0:
        return {}

    def weighted(key: str) -> float:
        vals = [(float(s[key]), float(s["n"])) for s in stats if np.isfinite(float(s[key]))]
        if not vals:
            return float("nan")
        denom = sum(w for _, w in vals)
        return float(sum(v * w for v, w in vals) / max(denom, 1e-12))

    return {
        "success_rate": float(sum(float(s["success_count"]) for s in stats) / n_total),
        "loss_gain_mean": weighted("loss_gain_mean"),
        "loss_gain_median_mean": weighted("loss_gain_median"),
        "loss_gain_p10_mean": weighted("loss_gain_p10"),
        "loss_gain_p90_mean": weighted("loss_gain_p90"),
        "clean_loss_mean": weighted("clean_loss_mean"),
        "adv_loss_mean": weighted("adv_loss_mean"),
        "pos_prob_drop_mean": weighted("pos_prob_drop_mean"),
        "neg_prob_rise_mean": weighted("neg_prob_rise_mean"),
    }


@torch.no_grad()
def fullft_adv_batch_diagnostics(
    clean_logits: torch.Tensor,
    init_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, np.ndarray | int]:
    """Return ECGFounder full-FT VAE adversarial batch diagnostics.

    The metrics preserve the legacy runner semantics:
    positive labels are attacked by lowering positive probability, negative
    labels by raising false positives, and sample-level ASR only counts anchors
    that were cleanly correct on at least one label.
    """

    clean_bce = F.binary_cross_entropy_with_logits(clean_logits, labels, reduction="none").mean(dim=1)
    init_bce = F.binary_cross_entropy_with_logits(init_logits, labels, reduction="none").mean(dim=1)
    adv_bce = F.binary_cross_entropy_with_logits(adv_logits, labels, reduction="none").mean(dim=1)
    loss_gain = adv_bce - clean_bce
    init_loss_gain = adv_bce - init_bce

    clean_prob = torch.sigmoid(clean_logits)
    adv_prob = torch.sigmoid(adv_logits)
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    signed = torch.where(pos_mask, torch.ones_like(labels), -torch.ones_like(labels))
    margin_drop = signed * (clean_logits - adv_logits)

    clean_pos_ok = pos_mask & (clean_prob >= 0.5)
    clean_neg_ok = neg_mask & (clean_prob < 0.5)
    pos_flips = clean_pos_ok & (adv_prob < 0.5)
    neg_flips = clean_neg_ok & (adv_prob >= 0.5)
    sample_ok = (clean_pos_ok | clean_neg_ok).any(dim=1)
    sample_flip = (pos_flips | neg_flips).any(dim=1) & sample_ok

    def arr(value: torch.Tensor) -> np.ndarray:
        return value.detach().float().cpu().numpy().astype(np.float32)

    return {
        "clean_bce": arr(clean_bce),
        "init_bce": arr(init_bce),
        "adv_bce": arr(adv_bce),
        "loss_gain": arr(loss_gain),
        "init_loss_gain": arr(init_loss_gain),
        "signed_margin_drop": arr(margin_drop.flatten()),
        "pos_signed_margin_drop": arr(margin_drop[pos_mask]),
        "neg_signed_margin_drop": arr(margin_drop[neg_mask]),
        "pos_correct_count": int(clean_pos_ok.sum().item()),
        "pos_hide_count": int(pos_flips.sum().item()),
        "neg_correct_count": int(clean_neg_ok.sum().item()),
        "neg_add_count": int(neg_flips.sum().item()),
        "sample_clean_correct_count": int(sample_ok.sum().item()),
        "sample_anyflip_count": int(sample_flip.sum().item()),
    }


def summarize_fullft_adv_epoch_diagnostics(
    batch_stats: list[dict[str, np.ndarray | int]],
    *,
    delta_norms: list[float] | np.ndarray,
    n_adv: int,
    anchor_sample_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    """Aggregate full-FT VAE adversarial diagnostics across one epoch."""

    def concat(key: str) -> np.ndarray:
        pieces = [np.asarray(row[key], dtype=np.float32) for row in batch_stats if len(np.asarray(row[key])) > 0]
        if not pieces:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32)

    def count(key: str) -> int:
        return int(sum(int(row.get(key, 0)) for row in batch_stats))

    def mean(values: np.ndarray) -> float | None:
        return float(np.mean(values)) if values.size else None

    def percentile(values: np.ndarray, q: float) -> float | None:
        return float(np.percentile(values, q)) if values.size else None

    delta = np.asarray(delta_norms, dtype=np.float32)
    clean_bce = concat("clean_bce")
    init_bce = concat("init_bce")
    adv_bce = concat("adv_bce")
    loss_gain = concat("loss_gain")
    init_loss_gain = concat("init_loss_gain")
    signed_margin_drop = concat("signed_margin_drop")
    pos_signed_margin_drop = concat("pos_signed_margin_drop")
    neg_signed_margin_drop = concat("neg_signed_margin_drop")
    pos_correct = count("pos_correct_count")
    pos_hide = count("pos_hide_count")
    neg_correct = count("neg_correct_count")
    neg_add = count("neg_add_count")
    sample_clean_correct = count("sample_clean_correct_count")
    sample_anyflip = count("sample_anyflip_count")

    return {
        "n_adv": int(n_adv),
        "delta_mean": mean(delta),
        "delta_max": float(np.max(delta)) if delta.size else None,
        "clean_bce_mean": mean(clean_bce),
        "init_bce_mean": mean(init_bce),
        "adv_bce_mean": mean(adv_bce),
        "loss_gain_mean": mean(loss_gain),
        "loss_gain_p50": percentile(loss_gain, 50),
        "loss_gain_p90": percentile(loss_gain, 90),
        "init_loss_gain_mean": mean(init_loss_gain),
        "init_loss_gain_p50": percentile(init_loss_gain, 50),
        "init_loss_gain_p90": percentile(init_loss_gain, 90),
        "signed_margin_drop_mean": mean(signed_margin_drop),
        "signed_margin_drop_p90": percentile(signed_margin_drop, 90),
        "pos_signed_margin_drop_mean": mean(pos_signed_margin_drop),
        "neg_signed_margin_drop_mean": mean(neg_signed_margin_drop),
        "pos_hide_asr": float(pos_hide / pos_correct) if pos_correct else None,
        "neg_add_asr": float(neg_add / neg_correct) if neg_correct else None,
        "sample_anyflip_asr": (
            float(sample_anyflip / sample_clean_correct) if sample_clean_correct else None
        ),
        "pos_correct_count": int(pos_correct),
        "neg_correct_count": int(neg_correct),
        "sample_clean_correct_count": int(sample_clean_correct),
        **(anchor_sample_stats or {}),
    }


def pairwise_rank_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_indices: list[int],
    *,
    class_weights: dict[int, float] | None = None,
    temperature: float,
) -> torch.Tensor:
    """Pairwise positive-vs-negative ranking loss for selected classes."""
    losses = []
    weights = []
    temp = max(float(temperature), 1e-6)
    for cls_idx in class_indices:
        y = labels[:, cls_idx]
        pos = logits[y > 0.5, cls_idx]
        neg = logits[y <= 0.0, cls_idx]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        margins = (pos[:, None] - neg[None, :]) / temp
        losses.append(F.softplus(-margins).mean())
        weights.append(float((class_weights or {}).get(cls_idx, 1.0)))
    if not losses:
        return logits.new_zeros(())
    loss_t = torch.stack(losses)
    weight_t = torch.tensor(weights, dtype=loss_t.dtype, device=loss_t.device)
    return (loss_t * weight_t).sum() / weight_t.sum().clamp_min(1e-6)


def compute_pos_weight(
    labels_mat: np.ndarray,
    num_classes: int,
    clip_max: float = 50.0,
) -> np.ndarray:
    """Compute clipped ``N_neg / N_pos`` weights from valid 0/1 labels only."""
    labels = np.asarray(labels_mat)
    if labels.ndim != 2:
        raise ValueError(f"expected labels_mat to be 2D, got shape {labels.shape}")
    if int(num_classes) > labels.shape[1]:
        raise ValueError(
            f"num_classes={num_classes} exceeds labels_mat columns={labels.shape[1]}"
        )

    pw = np.zeros(int(num_classes), dtype=np.float32)
    for i in range(int(num_classes)):
        col = labels[:, i]
        n_pos = int((col == 1.0).sum())
        n_neg = int((col == 0.0).sum())
        if n_pos == 0:
            pw[i] = float(clip_max)
        else:
            pw[i] = float(np.clip(n_neg / max(n_pos, 1), 1.0, clip_max))
    return pw
