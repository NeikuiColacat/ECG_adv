"""Tests for PN2021 K-shot relabeling helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.paper.relabel_pn2021_kshot_super5 import relabel_kshot_artifact_group


def test_relabel_kshot_artifact_group_rewrites_signal_and_latent_labels(tmp_path: Path):
    source_base = tmp_path / "source" / "demo" / "k2_seed7" / "demo_real_k2_seed7"
    output_base = tmp_path / "out" / "demo" / "k2_seed7" / "demo_real_k2_seed7"
    source_base.parent.mkdir(parents=True)

    signals = np.zeros((2, 1000, 12), dtype=np.float32)
    old_labels = np.zeros((2, 5), dtype=np.float32)
    latents = np.zeros((2, 4, 128), dtype=np.float32)
    record_ids = np.asarray(["a", "b"])
    np.savez_compressed(
        source_base.with_suffix(".signals.npz"),
        signals=signals,
        labels=old_labels,
        record_ids=record_ids,
        center_name="demo",
        class_names=np.asarray(["CD", "HYP", "MI", "NORM", "STTC"]),
    )
    np.savez_compressed(
        source_base.with_suffix(".latent.npz"),
        latents=latents,
        labels=old_labels,
        record_ids=record_ids,
        primary_class=np.asarray(["CD", "CD"]),
    )
    source_base.with_suffix(".ref_meta.json").write_text(
        json.dumps({"center": "demo", "ref_record_ids": ["a", "b"]}),
        encoding="utf-8",
    )

    summary = relabel_kshot_artifact_group(
        center="demo",
        source_base=source_base,
        output_base=output_base,
        label_by_record_id={
            "a": np.asarray([1, 0, 0, 0, 0], dtype=np.float32),
            "b": np.asarray([0, 0, 0, 0, 1], dtype=np.float32),
        },
        mapping_metadata={"mapping_version": "test_v", "mapping_hash": "test_h"},
        class_names=["CD", "HYP", "MI", "NORM", "STTC"],
    )

    assert summary["label_counts"] == {"CD": 1, "HYP": 0, "MI": 0, "NORM": 0, "STTC": 1}
    with np.load(output_base.with_suffix(".signals.npz"), allow_pickle=True) as data:
        assert data["labels"].astype(int).tolist() == [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
        assert data["mapping_version"].item() == "test_v"
    with np.load(output_base.with_suffix(".latent.npz"), allow_pickle=True) as data:
        assert data["labels"].astype(int).tolist() == [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
        assert data["primary_class"].astype(str).tolist() == ["CD", "STTC"]
    meta = json.loads(output_base.with_suffix(".ref_meta.json").read_text(encoding="utf-8"))
    assert meta["mapping_version"] == "test_v"
    assert meta["ref_record_ids"] == ["a", "b"]
