"""Signal-level dataset and weighted stream loader helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler


class CachedSignalDataset(Dataset):
    """Index-backed ECG signal dataset over cached numpy arrays."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        self.signals = signals
        self.labels = labels.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_i = int(self.indices[idx])
        x = np.asarray(self.signals[src_i], dtype=np.float32).copy()
        y = np.asarray(self.labels[src_i], dtype=np.float32).copy()
        return torch.from_numpy(x), torch.from_numpy(y)


class MemorySignalDataset(Dataset):
    """Dense in-memory ECG signal dataset."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray) -> None:
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.signals[idx]), torch.from_numpy(self.labels[idx])


class TaggedCachedSignalDataset(Dataset):
    """Cached ECG stream tagged with a stream id and zero teacher logits."""

    def __init__(self, base: CachedSignalDataset, stream_id: int, num_classes: int) -> None:
        self.base = base
        self.stream_id = int(stream_id)
        self.teacher_logits = torch.zeros((int(num_classes),), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        return x, y, torch.tensor(self.stream_id, dtype=torch.long), self.teacher_logits.clone()


class TaggedSignalDataset(Dataset):
    """Tag any two-tensor ECG dataset with a stream id and zero teacher logits."""

    def __init__(self, base: Dataset, stream_id: int, num_classes: int) -> None:
        self.base = base
        self.stream_id = int(stream_id)
        self.teacher_logits = torch.zeros((int(num_classes),), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            torch.tensor(self.stream_id, dtype=torch.long),
            self.teacher_logits.clone(),
        )


class TaggedMemorySignalDataset(Dataset):
    """In-memory ECG stream with optional per-row teacher logits."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        stream_id: int,
        teacher_logits: np.ndarray | None = None,
    ) -> None:
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.stream_id = int(stream_id)
        if teacher_logits is None:
            teacher_logits = np.zeros_like(self.labels, dtype=np.float32)
        teacher_logits = np.asarray(teacher_logits, dtype=np.float32)
        if teacher_logits.shape != self.labels.shape:
            raise ValueError(
                f"teacher_logits shape {teacher_logits.shape} does not match labels {self.labels.shape}"
            )
        self.teacher_logits = teacher_logits

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.signals[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.tensor(self.stream_id, dtype=torch.long),
            torch.from_numpy(self.teacher_logits[idx]),
        )


def matched_signal_stream_weights(
    *,
    source_count: int,
    target_clean_count: int,
    target_adv_count: int,
    source_weight: float,
    target_weight: float,
    target_adv_fraction: float,
) -> dict[str, float | int]:
    """Return copy-invariant row weights for a matched source/target stream."""

    source_count = int(source_count)
    target_clean_count = int(target_clean_count)
    target_adv_count = int(target_adv_count)
    rho = float(target_adv_fraction)
    if min(source_count, target_clean_count, target_adv_count) < 0:
        raise ValueError("matched stream counts must be non-negative")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("target_adv_fraction must be in [0, 1]")
    if float(source_weight) < 0.0 or float(target_weight) <= 0.0:
        raise ValueError("matched stream weights require source_weight >= 0 and target_weight > 0")
    if target_clean_count <= 0:
        raise ValueError("matched stream requires a non-empty target-clean split")
    if rho > 0.0 and target_adv_count <= 0:
        raise ValueError("positive target_adv_fraction requires target adversarial samples")

    target_total_mass = float(target_weight) * target_clean_count
    target_clean_mass = target_total_mass * (1.0 - rho)
    target_adv_mass = target_total_mass * rho
    return {
        "source_count": source_count,
        "target_clean_count": target_clean_count,
        "target_adv_count": target_adv_count,
        "source_row_weight": float(source_weight),
        "target_clean_row_weight": target_clean_mass / target_clean_count,
        "target_adv_row_weight": (
            target_adv_mass / target_adv_count if target_adv_count else 0.0
        ),
        "source_total_mass": float(source_weight) * source_count,
        "target_clean_mass": target_clean_mass,
        "target_adv_mass": target_adv_mass,
        "target_total_mass": target_total_mass,
        "target_adv_mass_fraction": rho,
        "num_samples": source_count + target_clean_count,
    }


def build_weighted_signal_stream_loader_from_datasets(
    *,
    source_dataset: Dataset,
    target_dataset: Dataset,
    source_weight: float,
    target_real_weight: float,
    adv_weight: float,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    adv_signals: np.ndarray | None = None,
    adv_labels: np.ndarray | None = None,
    adv_teacher_logits: np.ndarray | None = None,
    adv_sample_weights: np.ndarray | None = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    target_adv_fraction: float | None = None,
    fixed_num_samples: int | None = None,
    sampler_seed: int | None = None,
) -> DataLoader:
    """Build source/target/adversarial ECG stream loader from prepared datasets."""
    source_ds = TaggedSignalDataset(source_dataset, stream_id=0, num_classes=num_classes)
    target_ds = TaggedSignalDataset(target_dataset, stream_id=1, num_classes=num_classes)

    datasets: list[Dataset] = []
    weights: list[float] = []
    matched_contract: dict[str, float | int] | None = None
    matched_adv_weights: np.ndarray | None = None
    if target_adv_fraction is not None:
        matched_contract = matched_signal_stream_weights(
            source_count=len(source_ds),
            target_clean_count=len(target_ds),
            target_adv_count=0 if adv_signals is None else len(adv_signals),
            source_weight=source_weight,
            target_weight=target_real_weight,
            target_adv_fraction=target_adv_fraction,
        )
        source_weight = float(matched_contract["source_row_weight"])
        target_real_weight = float(matched_contract["target_clean_row_weight"])
        adv_weight = float(matched_contract["target_adv_row_weight"])
        if adv_sample_weights is not None and len(adv_sample_weights) > 0:
            matched_adv_weights = np.asarray(adv_sample_weights, dtype=np.float64)
            if not np.isfinite(matched_adv_weights).all() or np.any(matched_adv_weights < 0):
                raise ValueError("adv_sample_weights must be finite and non-negative")
            total = float(matched_adv_weights.sum())
            if total <= 0.0:
                raise ValueError("matched adv_sample_weights must have positive mass")
            matched_adv_weights = matched_adv_weights * (len(matched_adv_weights) / total)
    if float(source_weight) > 0.0:
        datasets.append(source_ds)
        weights.extend([float(source_weight)] * len(source_ds))
    if float(target_real_weight) > 0.0:
        datasets.append(target_ds)
        weights.extend([float(target_real_weight)] * len(target_ds))
    if (
        adv_signals is not None
        and adv_labels is not None
        and len(adv_signals) > 0
        and float(adv_weight) > 0.0
    ):
        adv_ds = TaggedMemorySignalDataset(
            adv_signals,
            adv_labels,
            stream_id=2,
            teacher_logits=adv_teacher_logits,
        )
        datasets.append(adv_ds)
        if matched_adv_weights is not None:
            weights.extend((float(adv_weight) * matched_adv_weights).tolist())
        elif adv_sample_weights is None or target_adv_fraction is not None:
            weights.extend([float(adv_weight)] * len(adv_ds))
        else:
            adv_sample_weights = np.asarray(adv_sample_weights, dtype=np.float64)
            if adv_sample_weights.shape != (len(adv_ds),):
                raise ValueError("adv_sample_weights length must match adv_signals")
            if not np.isfinite(adv_sample_weights).all() or np.any(adv_sample_weights < 0):
                raise ValueError("adv_sample_weights must be finite and non-negative")
            weights.extend((float(adv_weight) * adv_sample_weights).tolist())
    if not datasets:
        raise RuntimeError("no active training streams; check source/target/adv weights")

    combined = ConcatDataset(datasets)
    num_samples = len(weights) if fixed_num_samples is None else int(fixed_num_samples)
    if matched_contract is not None and fixed_num_samples is None:
        num_samples = int(matched_contract["num_samples"])
    if num_samples <= 0:
        raise ValueError("fixed_num_samples must be positive")
    generator = None
    if sampler_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(sampler_seed))
    sampler = WeightedRandomSampler(
        weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        combined,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
        drop_last=bool(drop_last),
    )
    if matched_contract is not None:
        matched_contract = dict(matched_contract)
        matched_contract["num_samples"] = num_samples
        loader.stream_sampling_contract = matched_contract
    return loader


def build_weighted_signal_stream_loader(
    *,
    source_signals: np.ndarray,
    source_labels: np.ndarray,
    source_indices: np.ndarray,
    target_signals: np.ndarray,
    target_labels: np.ndarray,
    target_indices: np.ndarray,
    source_weight: float,
    target_real_weight: float,
    adv_weight: float,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    adv_signals: np.ndarray | None = None,
    adv_labels: np.ndarray | None = None,
    adv_teacher_logits: np.ndarray | None = None,
    adv_sample_weights: np.ndarray | None = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Build source/target/adversarial ECG stream loader with fixed stream ids."""
    source_ds = CachedSignalDataset(source_signals, source_labels, source_indices)
    target_ds = CachedSignalDataset(target_signals, target_labels, target_indices)
    return build_weighted_signal_stream_loader_from_datasets(
        source_dataset=source_ds,
        target_dataset=target_ds,
        source_weight=source_weight,
        target_real_weight=target_real_weight,
        adv_weight=adv_weight,
        batch_size=batch_size,
        num_workers=num_workers,
        num_classes=num_classes,
        adv_signals=adv_signals,
        adv_labels=adv_labels,
        adv_teacher_logits=adv_teacher_logits,
        adv_sample_weights=adv_sample_weights,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


__all__ = [
    "CachedSignalDataset",
    "MemorySignalDataset",
    "TaggedCachedSignalDataset",
    "TaggedMemorySignalDataset",
    "TaggedSignalDataset",
    "build_weighted_signal_stream_loader",
    "build_weighted_signal_stream_loader_from_datasets",
    "matched_signal_stream_weights",
]
