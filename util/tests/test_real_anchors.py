"""CPU-only tests for target-center real-anchor pool helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from ecg_adv_gen.data import (
    find_real_anchor_base,
    load_real_anchor_pool,
    load_real_anchor_pool_for_record_ids,
    select_primary_label_proportional_min1,
)
from scripts.paper.run_ecgfounder_vae_only_lhat_head_ft_20260523 import (
    load_anchor_pool,
    real_anchor_base,
)


def _write_latent_npz(base: Path, record_ids: list[str]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    latents = np.arange(len(record_ids) * 4 * 128, dtype=np.float32).reshape(len(record_ids), 4, 128)
    np.savez_compressed(base.with_suffix(".latent.npz"), latents=latents, record_ids=np.asarray(record_ids, dtype=str))


def test_find_real_anchor_base_uses_explicit_managed_candidates(tmp_path: Path):
    root = tmp_path / "anchors"
    base = root / "ningbo" / "k3_seed11" / "ningbo_real_k3_seed11"
    _write_latent_npz(base, ["N1"])

    found = find_real_anchor_base("ningbo", anchor_base_root=root, k=3, seed=11)

    assert found == base
    with pytest.raises(FileNotFoundError, match="missing real-anchor files"):
        find_real_anchor_base("georgia", anchor_base_root=root, k=3, seed=11)


def test_select_primary_label_proportional_min1_matches_runner_behavior():
    labels = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=np.float32,
    )

    assert select_primary_label_proportional_min1(labels, 3, seed=11).tolist() == [2, 3, 4]
    assert select_primary_label_proportional_min1(labels, 5, seed=11).tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="exceeds matched anchors"):
        select_primary_label_proportional_min1(labels, 6, seed=11)


def test_load_real_anchor_pool_matches_pn_cache_by_center_and_record_id(tmp_path: Path):
    root = tmp_path / "anchors"
    base = root / "ningbo" / "k3_seed11" / "ningbo_real_k3_seed11"
    _write_latent_npz(base, ["A", "B", "C", "D", "E", "UNMATCHED"])
    pn_payload = {
        "centers": np.asarray(["ningbo", "ningbo", "ningbo", "ningbo", "ningbo", "georgia"], dtype=str),
        "record_ids": np.asarray(["A", "B", "C", "D", "E", "A"], dtype=str),
        "labels": np.asarray(
            [
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
    }

    pool = load_real_anchor_pool(
        "ningbo",
        k=3,
        seed=11,
        pn_payload=pn_payload,
        anchor_base_root=root,
    )

    assert pool["record_ids"].tolist() == ["C", "D", "E"]
    assert pool["classes_in_scope"] == ["HYP", "MI", "NORM"]
    assert pool["label_counts"] == {"CD": 0, "HYP": 1, "MI": 1, "NORM": 1, "STTC": 0}
    assert pool["source_base"] == str(base)
    assert pool["latents"].shape == (3, 4, 128)


def test_load_real_anchor_pool_for_record_ids_returns_signals_in_sorted_id_order(tmp_path: Path):
    root = tmp_path / "legacy"
    base = root / "georgia" / "georgia_real_k500_seed42"
    _write_latent_npz(base, ["B", "A", "C"])
    signals = np.arange(3 * 12 * 8, dtype=np.float32).reshape(3, 12, 8)
    pn_payload = {
        "centers": np.asarray(["georgia", "georgia", "georgia"], dtype=str),
        "record_ids": np.asarray(["A", "B", "C"], dtype=str),
        "labels": np.eye(5, dtype=np.float32)[:3],
        "signals": signals,
    }

    pool = load_real_anchor_pool_for_record_ids(
        "georgia",
        selected_ids={"C", "A"},
        pn_payload=pn_payload,
        default_roots=[root],
        include_signals=True,
    )

    assert pool["record_ids"].tolist() == ["A", "C"]
    assert pool["signals"].shape == (2, 12, 8)
    np.testing.assert_array_equal(pool["signals"][0], signals[0])
    np.testing.assert_array_equal(pool["signals"][1], signals[2])

    with pytest.raises(RuntimeError, match="selected ids are missing"):
        load_real_anchor_pool_for_record_ids(
            "georgia",
            selected_ids={"MISSING"},
            pn_payload=pn_payload,
            default_roots=[root],
        )


def test_ecgfounder_vae_runner_wrappers_delegate_without_protocol_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "anchors"
    base = root / "ningbo" / "k2_seed5" / "ningbo_real_k2_seed5"
    _write_latent_npz(base, ["A", "B"])
    pn_payload = {
        "centers": np.asarray(["ningbo", "ningbo"], dtype=str),
        "record_ids": np.asarray(["A", "B"], dtype=str),
        "labels": np.asarray([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32),
    }
    args = Namespace(anchor_base_root=str(root), k=2, seed=5)

    assert real_anchor_base("ningbo", args) == base
    pool = load_anchor_pool("ningbo", 2, 5, pn_payload, args)

    assert pool["record_ids"].tolist() == ["A", "B"]
    assert pool["label_counts"]["CD"] == 1
    assert pool["label_counts"]["HYP"] == 1
