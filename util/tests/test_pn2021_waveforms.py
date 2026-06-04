"""CPU-only tests for PN2021 waveform materialization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecg_adv_gen.data import materialize_pn2021_center_records


@dataclass
class FakeRecord:
    p_signal: np.ndarray | None
    fs: int = 500
    sig_name: list[str] | None = None


def test_materialize_pn2021_center_records_counts_failures_and_preserves_ids():
    records = {
        "group/N1": FakeRecord(np.ones((5000, 12), dtype=np.float64), sig_name=["I", "II"]),
        "group/N2": FakeRecord(None),
        "group/N3": FakeRecord(np.ones((5000, 8), dtype=np.float32)),
        "group/N4": FakeRecord(np.full((5000, 12), 4.0, dtype=np.float32)),
    }

    def read_record(path: str) -> FakeRecord:
        return records[path]

    def preprocess(sig: np.ndarray, *, fs: int, source_leads: list[str] | None, scale: float) -> np.ndarray | None:
        assert fs == 500
        assert sig.dtype == np.float32
        if source_leads == ["I", "II"]:
            return np.full((1000, 12), sig[0, 0] * scale, dtype=np.float32)
        return np.full((1000, 12), sig[0, 0] * scale, dtype=np.float32)

    def label_fn(codes: list[int]) -> np.ndarray:
        return np.asarray([1.0 if 1 in codes else 0.0, 1.0 if 2 in codes else 0.0], dtype=np.float32)

    result = materialize_pn2021_center_records(
        ["group/N1", "group/N2", "group/N3", "group/N4"],
        [[1], [2], [1, 2], [2]],
        read_record=read_record,
        preprocess_signal=preprocess,
        label_fn=label_fn,
        num_classes=2,
        preprocess_kwargs={"scale": 10.0},
    )

    assert result.fail == 2
    assert result.signals.shape == (2, 1000, 12)
    assert result.signals[:, 0, 0].tolist() == [10.0, 40.0]
    assert result.labels.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert result.record_ids.tolist() == ["N1", "N4"]
    assert result.load_time_s >= 0.0


def test_materialize_pn2021_center_records_returns_empty_contract_when_all_fail():
    result = materialize_pn2021_center_records(
        ["N1"],
        [[1]],
        read_record=lambda path: FakeRecord(None),
        preprocess_signal=lambda sig, **kwargs: sig,
        label_fn=lambda codes: np.asarray([1.0, 0.0], dtype=np.float32),
        num_classes=2,
        signal_shape=(1000, 12),
    )

    assert result.fail == 1
    assert result.signals.shape == (0, 1000, 12)
    assert result.labels.shape == (0, 2)
    assert result.record_ids.tolist() == []
