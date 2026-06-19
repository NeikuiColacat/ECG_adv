"""CPU tests for locked-protocol raw1000 K-shot materialization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ecg_adv_gen.data.pn2021_raw_kshot import materialize_selected_raw1000_from_ref_meta
from scripts.paper.materialize_pn2021_raw1000_kshot import build_output_base


def test_build_output_base_uses_locked_kshot_layout():
    subset_root = Path("/tmp/subsets")
    base = build_output_base(subset_root, "cpsc_2018", 500, 20260531)

    assert base == subset_root / "cpsc_2018/k500_seed20260531/cpsc_2018_real_k500_seed20260531"


def test_materialize_selected_raw1000_from_ref_meta_preserves_ref_order_and_no_zscore(tmp_path):
    ref_meta = tmp_path / "ningbo_real_k2_seed7.ref_meta.json"
    ref_meta.write_text(
        json.dumps(
            {
                "center": "ningbo",
                "K": 2,
                "selection_seed": 7,
                "ref_record_ids": ["B", "A"],
            }
        ),
        encoding="utf-8",
    )
    records = [
        SimpleNamespace(
            record_id="A",
            record_path=tmp_path / "A",
            snomeds=(1,),
        ),
        SimpleNamespace(
            record_id="B",
            record_path=tmp_path / "B",
            snomeds=(2,),
        ),
    ]
    raw_by_id = {
        "A": np.full((20, 12), 3.0, dtype=np.float32),
        "B": np.full((20, 12), 9.0, dtype=np.float32),
    }

    def fake_scan(_center_dir):
        return list(records)

    def fake_read_record(path):
        rid = str(path).split("/")[-1]
        return SimpleNamespace(p_signal=raw_by_id[rid], fs=500, sig_name=[f"L{i}" for i in range(12)])

    def fake_preprocess(signal, *, fs, source_leads, target_fs, target_len, preprocess_mode, norm_mode):
        assert fs == 500
        assert target_fs == 100
        assert target_len == 1000
        assert preprocess_mode == "minimal_resample"
        assert norm_mode == "none"
        scale = float(signal[0, 0])
        return np.full((1000, 12), scale, dtype=np.float32)

    def fake_label(codes):
        out = np.zeros(5, dtype=np.float32)
        out[0] = float(codes[0])
        return out

    materialized = materialize_selected_raw1000_from_ref_meta(
        center="ningbo",
        ref_meta_json=ref_meta,
        pn2021_root=tmp_path / "physionet2021",
        scan_center_records=fake_scan,
        read_record=fake_read_record,
        preprocess_signal=fake_preprocess,
        label_fn=fake_label,
        class_names=["CD", "HYP", "MI", "NORM", "STTC"],
    )

    assert materialized.signals.shape == (2, 1000, 12)
    assert materialized.record_ids.tolist() == ["B", "A"]
    assert materialized.labels[:, 0].tolist() == [2.0, 1.0]
    assert float(materialized.signals[0].mean()) == 9.0
    assert float(materialized.signals[1].mean()) == 3.0
    assert materialized.metadata["preprocess"]["norm_mode"] == "none"
    assert materialized.metadata["signals_are_pre_zscore"] is True
