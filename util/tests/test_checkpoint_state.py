"""CPU-only tests for training checkpoint state helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_adv_gen.training.checkpoint_state import (
    append_jsonl,
    atomic_torch_save,
    capture_rng_state,
    quality_buffer_state,
    resolve_resume_path,
    restore_quality_buffer_state,
    restore_rng_state,
)


class DummyBuffer:
    def __init__(self, max_size: int = 4) -> None:
        self.max_size = max_size
        self.ecg_list: list[torch.Tensor] = []
        self.label_list: list[torch.Tensor] = []
        self.score_list: list[float] = []

    def __len__(self) -> int:
        return len(self.ecg_list)


def test_resolve_resume_path_supports_latest_and_explicit_paths(tmp_path: Path):
    assert resolve_resume_path("", tmp_path) is None
    assert resolve_resume_path("latest", tmp_path) == tmp_path / "checkpoints" / "checkpoint_latest.pt"
    assert resolve_resume_path(str(tmp_path / "manual.pt"), tmp_path) == tmp_path / "manual.pt"


def test_quality_buffer_state_round_trips_tensors_and_scores():
    source = DummyBuffer(max_size=8)
    source.ecg_list = [torch.ones(2, 3), torch.zeros(2, 3)]
    source.label_list = [torch.tensor([1.0, -1.0]), torch.tensor([0.0, 1.0])]
    source.score_list = [0.7, 0.2]

    state = quality_buffer_state(source)
    restored = DummyBuffer(max_size=1)
    restore_quality_buffer_state(restored, state)

    assert state["size"] == 2
    assert restored.max_size == 8
    assert len(restored) == 2
    assert restored.score_list == [0.7, 0.2]
    assert torch.equal(restored.ecg_list[0], torch.ones(2, 3))
    assert torch.equal(restored.label_list[1], torch.tensor([0.0, 1.0]))


def test_restore_quality_buffer_state_clears_empty_or_partial_state():
    buffer = DummyBuffer(max_size=3)
    buffer.ecg_list = [torch.ones(1)]
    buffer.label_list = [torch.ones(1)]
    buffer.score_list = [1.0]

    restore_quality_buffer_state(buffer, {"max_size": 9, "score_list": [0.5]})

    assert buffer.max_size == 9
    assert buffer.ecg_list == []
    assert buffer.label_list == []
    assert buffer.score_list == []


def test_append_jsonl_and_atomic_torch_save_write_expected_files(tmp_path: Path):
    jsonl = tmp_path / "nested" / "events.jsonl"
    append_jsonl(jsonl, {"b": 2})
    append_jsonl(jsonl, {"a": "x"})

    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"b": 2}, {"a": "x"}]

    checkpoint = tmp_path / "ckpt" / "state.pt"
    atomic_torch_save({"x": torch.tensor([1, 2, 3])}, checkpoint)

    assert checkpoint.exists()
    assert not checkpoint.with_name("state.pt.tmp").exists()
    loaded = torch.load(checkpoint, map_location="cpu")
    assert torch.equal(loaded["x"], torch.tensor([1, 2, 3]))


def test_rng_state_restores_python_numpy_torch_and_epoch_generator():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    epoch_rng = np.random.default_rng(123)
    state = capture_rng_state(epoch_rng, include_cuda=False)

    expected = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
        int(epoch_rng.integers(0, 10_000)),
    )

    random.random()
    np.random.rand()
    torch.rand(1)
    epoch_rng.integers(0, 10_000)
    restore_rng_state(state, epoch_rng, restore_cuda=False)

    observed = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
        int(epoch_rng.integers(0, 10_000)),
    )
    assert observed == pytest.approx(expected)


def test_rng_state_restores_torch_cpu_from_serialized_list():
    random.seed(321)
    np.random.seed(321)
    torch.manual_seed(321)
    epoch_rng = np.random.default_rng(321)
    state = capture_rng_state(epoch_rng, include_cuda=False)
    state["torch_cpu"] = state["torch_cpu"].tolist()

    expected = float(torch.rand(1).item())
    torch.rand(1)
    restore_rng_state(state, epoch_rng, restore_cuda=False)

    assert float(torch.rand(1).item()) == pytest.approx(expected)


def test_rng_state_normalizes_cuda_state_before_restore(monkeypatch):
    epoch_rng = np.random.default_rng(7)
    captured = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: captured.extend(states))

    restore_rng_state(
        {"torch_cuda_all": [[1, 2, 3], torch.tensor([4, 5], dtype=torch.int64)]},
        epoch_rng,
        restore_cuda=True,
    )

    assert [state.dtype for state in captured] == [torch.uint8, torch.uint8]
    assert all(state.device.type == "cpu" for state in captured)
