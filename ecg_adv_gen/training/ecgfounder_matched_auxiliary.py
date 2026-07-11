"""Equal-budget train-only auxiliary objective for matched ECGFounder arms."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecg_adv_gen.models.ecgfounder_torch import ecg1000_to_ecgfounder_input
from methods.augmix.jsd_loss import jsd_multilabel


def select_matched_auxiliary_clean_examples(
    *,
    signals: np.ndarray,
    labels: np.ndarray,
    record_ids: np.ndarray,
    train_record_ids: set[str],
    val_record_ids: set[str],
    n_samples: int,
    seed: int,
    epoch: int,
) -> dict[str, Any]:
    """Select an arm-invariant clean exposure from K500 train IDs only."""

    ids = np.asarray(record_ids).astype(str)
    train, val = map(lambda values: {str(value) for value in values}, (train_record_ids, val_record_ids))
    if train.intersection(val):
        raise ValueError("matched auxiliary train and internal-val IDs overlap")
    train_indices = np.asarray(
        sorted(
            (index for index, record_id in enumerate(ids) if record_id in train),
            key=lambda index: ids[index],
        ),
        dtype=np.int64,
    )
    requested = int(n_samples)
    if len(train_indices) != len(train) or requested <= 0 or requested > len(train_indices):
        raise ValueError(
            f"matched auxiliary n_samples={requested} is invalid for train pool={len(train_indices)}"
        )
    rng = np.random.default_rng(int(seed) + int(epoch) * 100003 + 1709)
    selected = train_indices[rng.permutation(len(train_indices))[:requested]]
    selected_ids = ids[selected].tolist()
    val_overlap = len(set(selected_ids).intersection(val))
    if val_overlap:
        raise AssertionError("matched auxiliary clean exposure leaked internal-val IDs")
    digest = hashlib.sha256(("\n".join(selected_ids) + "\n").encode("utf-8")).hexdigest()
    return {
        "signals": np.asarray(signals)[selected].astype(np.float32, copy=False),
        "labels": np.asarray(labels)[selected].astype(np.float32, copy=False),
        "record_ids": selected_ids,
        "record_ids_sha256": digest,
        "n_samples": requested,
        "val_overlap_count": val_overlap,
        "selection_seed": int(seed),
        "epoch": int(epoch),
    }


def optimizer_parameter_step(
    optimizer: torch.optim.Optimizer, parameter: nn.Parameter
) -> int:
    """Return Adam-style per-parameter step state, or zero before first update."""

    step = optimizer.state.get(parameter, {}).get("step", 0)
    return int(step.item()) if torch.is_tensor(step) else int(step)


def _masked_bce(
    logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    mask = (labels >= 0).float()
    raw = F.binary_cross_entropy_with_logits(
        logits,
        labels.clamp(min=0.0),
        pos_weight=pos_weight,
        reduction="none",
    )
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def _validate_views(
    clean: np.ndarray,
    view_clean: np.ndarray | None,
    augmented: np.ndarray | None,
    view_labels: np.ndarray | None,
    *,
    copies: int,
    view_bce_weight: float,
    consistency_weight: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    use_views = any(value is not None for value in (view_clean, augmented, view_labels))
    if not use_views:
        if int(copies) or float(view_bce_weight) or float(consistency_weight):
            raise ValueError("matched A0/A3 auxiliary cannot enable raw-view objectives")
        return None, None, None
    if any(value is None for value in (view_clean, augmented, view_labels)):
        raise ValueError("matched A5 auxiliary views must provide clean, augmented, and labels")
    if int(copies) != 2 or float(view_bce_weight) != 1.0 or float(consistency_weight) != 2.0:
        raise ValueError("matched A5 auxiliary requires copies=2, view BCE=1, and JSD=2")
    view_clean_np = np.asarray(view_clean, dtype=np.float32)
    view_labels_np = np.asarray(view_labels, dtype=np.float32)
    augmented_np = np.asarray(augmented, dtype=np.float32)
    n = len(clean)
    if len(view_clean_np) != n or len(view_labels_np) != n or len(augmented_np) != copies * n:
        raise ValueError("matched A5 view exposure must be copies x common clean exposure")
    return view_clean_np, augmented_np.reshape(copies, n, *augmented_np.shape[1:]), view_labels_np


def train_matched_auxiliary_epoch(
    *,
    model: nn.Module,
    clean_signals_ct: np.ndarray,
    labels_np: np.ndarray,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    device: torch.device,
    trainable_params: list[nn.Parameter],
    batch_size: int,
    max_batches: int = 0,
    input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    view_clean_signals_ct: np.ndarray | None = None,
    augmix_signals_ct: np.ndarray | None = None,
    view_labels_np: np.ndarray | None = None,
    copies: int = 0,
    view_bce_weight: float = 0.0,
    consistency_weight: float = 0.0,
) -> dict[str, Any]:
    """Apply common clean BCE; only A5 adds its declared view BCE/JSD terms."""

    transform = input_transform or ecg1000_to_ecgfounder_input
    clean = np.asarray(clean_signals_ct, dtype=np.float32)
    labels = np.asarray(labels_np, dtype=np.float32)
    if not len(clean) or len(labels) != len(clean):
        raise ValueError("matched auxiliary clean signals/labels must be non-empty and aligned")
    view_clean, views, view_labels = _validate_views(
        clean,
        view_clean_signals_ct,
        augmix_signals_ct,
        view_labels_np,
        copies=copies,
        view_bce_weight=view_bce_weight,
        consistency_weight=consistency_weight,
    )
    labels_t = torch.from_numpy(labels).float()
    view_labels_t = torch.from_numpy(view_labels).float() if view_labels is not None else None
    losses: list[float] = []
    common_bces: list[float] = []
    view_bces: list[float] = []
    consistency_losses: list[float] = []
    parameter_advances = value_change_steps = clean_exposed = view_exposed = 0

    for batch_i, lo in enumerate(range(0, len(clean), max(1, int(batch_size))), start=1):
        hi = min(lo + max(1, int(batch_size)), len(clean))
        common_x = torch.from_numpy(clean[lo:hi]).float().to(device)
        common_y = labels_t[lo:hi].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            common_bce = _masked_bce(model(transform(common_x)), common_y, pos_weight)
            loss = common_bce
            view_bce = consistency = None
            if views is not None and view_clean is not None and view_labels_t is not None:
                view_x = torch.from_numpy(view_clean[lo:hi]).float().to(device)
                view_y = view_labels_t[lo:hi].to(device)
                augmented_x = torch.from_numpy(
                    views[:, lo:hi].reshape(-1, *views.shape[2:])
                ).float().to(device)
                clean_logits = model(transform(view_x))
                augmented_logits = model(transform(augmented_x))
                by_copy = augmented_logits.view(copies, hi - lo, -1)
                augmented_bce = _masked_bce(
                    augmented_logits, view_y.repeat((copies, 1)), pos_weight
                )
                view_bce = 0.5 * (_masked_bce(clean_logits, view_y, pos_weight) + augmented_bce)
                consistency = torch.stack(
                    [
                        jsd_multilabel(clean_logits, by_copy[i], by_copy[(i + 1) % copies])
                        for i in range(copies)
                    ]
                ).mean()
                loss = loss + view_bce_weight * view_bce + consistency_weight * consistency
        loss.backward()
        gradient_params = [
            parameter
            for parameter in trainable_params
            if parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(torch.count_nonzero(parameter.grad).item())
        ]
        if not gradient_params:
            raise RuntimeError("matched auxiliary BCE produced no finite nonzero parameter gradient")
        nn.utils.clip_grad_norm_(trainable_params, 1.0)
        sentinel = gradient_params[0]
        sentinel_before = sentinel.detach().clone()
        steps_before = {id(p): optimizer_parameter_step(optimizer, p) for p in gradient_params}
        optimizer.step()
        advances = sum(optimizer_parameter_step(optimizer, p) > steps_before[id(p)] for p in gradient_params)
        if not advances or torch.equal(sentinel.detach(), sentinel_before):
            raise RuntimeError("matched auxiliary optimizer step did not update parameters")
        parameter_advances += advances
        value_change_steps += 1
        losses.append(float(loss.detach().item()))
        common_bces.append(float(common_bce.detach().item()))
        if view_bce is not None:
            view_bces.append(float(view_bce.detach().item()))
        if consistency is not None:
            consistency_losses.append(float(consistency.detach().item()))
        clean_exposed += hi - lo
        view_exposed += copies * (hi - lo) if views is not None else 0
        if int(max_batches) > 0 and batch_i >= int(max_batches):
            break

    return {
        "enabled": True,
        "control": "train_only_clean_bce",
        "loss": float(np.mean(losses)),
        "bce_loss": float(np.mean(common_bces)),
        "view_bce_loss": float(np.mean(view_bces)) if view_bces else None,
        "consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else None,
        "n_batches": len(losses),
        "actual_param_update_steps": value_change_steps,
        "optimizer_parameter_step_advances": parameter_advances,
        "clean_samples_exposed": clean_exposed,
        "view_samples_exposed": view_exposed,
        "copies": int(copies),
        "view_bce_weight": float(view_bce_weight),
        "consistency_weight": float(consistency_weight),
        "max_batches": int(max_batches),
    }


__all__ = [
    "optimizer_parameter_step",
    "select_matched_auxiliary_clean_examples",
    "train_matched_auxiliary_epoch",
]
