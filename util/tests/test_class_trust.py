from __future__ import annotations

import json

import numpy as np
import pytest

from ecg_adv_gen.data.class_trust import write_real_all_present_trust


def test_write_real_all_present_trust_counts_present_scoped_classes(tmp_path):
    signal_npz = tmp_path / "ningbo_real_k500_seed20260531.signals.npz"
    labels = np.zeros((4, 5), dtype=np.float32)
    labels[0, 0] = 1.0
    labels[1:3, 2] = 1.0
    labels[3, 3] = 1.0
    np.savez(signal_npz, signals=np.zeros((4, 1000, 12), dtype=np.float32), labels=labels)

    out = write_real_all_present_trust(
        signal_npz,
        tmp_path / "config",
        "ningbo",
        classes_in_scope=["CD", "MI"],
        policy="unit-test policy",
    )

    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["tag"] == "ningbo_real_k500_seed20260531_real_all_present"
    assert blob["center"] == "ningbo"
    assert blob["class_counts"] == {"CD": 1, "HYP": 0, "MI": 2, "NORM": 1, "STTC": 0}
    assert blob["class_trust"] == {"CD": 1.0, "HYP": 0.0, "MI": 1.0, "NORM": 0.0, "STTC": 0.0}
    assert blob["policy"] == "unit-test policy"


def test_write_real_all_present_trust_rejects_bad_shape_and_unknown_class(tmp_path):
    signal_npz = tmp_path / "bad.signals.npz"
    np.savez(signal_npz, labels=np.zeros((2, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="labels must have shape"):
        write_real_all_present_trust(signal_npz, tmp_path, "georgia")

    good_npz = tmp_path / "good.signals.npz"
    np.savez(good_npz, labels=np.zeros((2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="Unknown Super5"):
        write_real_all_present_trust(good_npz, tmp_path, "georgia", classes_in_scope=["BAD"])
