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
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Build source/target/adversarial ECG stream loader from prepared datasets."""
    source_ds = TaggedSignalDataset(source_dataset, stream_id=0, num_classes=num_classes)
    target_ds = TaggedSignalDataset(target_dataset, stream_id=1, num_classes=num_classes)

    datasets: list[Dataset] = []
    weights: list[float] = []
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
        weights.extend([float(adv_weight)] * len(adv_ds))
    if not datasets:
        raise RuntimeError("no active training streams; check source/target/adv weights")

    combined = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        combined,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0,
        drop_last=bool(drop_last),
    )


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
]
