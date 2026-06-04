"""Checkpoint state helpers shared by resumable training entrypoints."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a torch checkpoint through a sibling temp file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append a JSON object as one line, creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")


def resolve_resume_path(resume: str | None, output_dir: str | Path) -> Path | None:
    """Resolve the legacy resume option into a checkpoint path."""

    if not resume:
        return None
    if str(resume) == "latest":
        return Path(output_dir) / "checkpoints" / "checkpoint_latest.pt"
    return Path(str(resume)).expanduser()


def quality_buffer_state(buffer: Any) -> dict[str, Any]:
    """Serialize the common QualityAwareBuffer tensor/list state."""

    state: dict[str, Any] = {
        "max_size": int(buffer.max_size),
        "size": int(len(buffer)),
        "score_list": list(buffer.score_list),
    }
    if len(buffer) > 0:
        state["ecg_tensor"] = torch.stack(buffer.ecg_list).cpu()
        state["label_tensor"] = torch.stack(buffer.label_list).cpu()
    return state


def restore_quality_buffer_state(buffer: Any, state: dict[str, Any]) -> None:
    """Restore QualityAwareBuffer-like state in-place."""

    state = state or {}
    buffer.max_size = int(state.get("max_size", buffer.max_size))
    ecg_tensor = state.get("ecg_tensor")
    label_tensor = state.get("label_tensor")
    scores = list(state.get("score_list") or [])
    if ecg_tensor is None or label_tensor is None:
        buffer.ecg_list = []
        buffer.label_list = []
        buffer.score_list = []
        return
    buffer.ecg_list = [row.detach().cpu() for row in ecg_tensor]
    buffer.label_list = [row.detach().cpu() for row in label_tensor]
    buffer.score_list = [float(x) for x in scores[: len(buffer.ecg_list)]]


def capture_rng_state(epoch_rng: np.random.Generator, *, include_cuda: bool = True) -> dict[str, Any]:
    """Capture Python, NumPy, epoch-generator, and torch RNG state."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_epoch_generator": epoch_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
    }
    if include_cuda and torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: dict[str, Any],
    epoch_rng: np.random.Generator,
    *,
    restore_cuda: bool = True,
) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""

    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy_global" in state:
        np.random.set_state(state["numpy_global"])
    if "numpy_epoch_generator" in state:
        epoch_rng.bit_generator.state = state["numpy_epoch_generator"]
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if restore_cuda and torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


__all__ = [
    "append_jsonl",
    "atomic_torch_save",
    "capture_rng_state",
    "quality_buffer_state",
    "resolve_resume_path",
    "restore_quality_buffer_state",
    "restore_rng_state",
]
