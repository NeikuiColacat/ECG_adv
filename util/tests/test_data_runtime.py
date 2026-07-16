from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from data_preprocess.data_runtime import (
    ECGModelTransform,
    RuntimeECGDataset,
    assert_disjoint_selections,
    build_dataloader,
    get_dataloader,
    load_selection,
    per_sample_global_zscore,
    prepare_model_input,
)
from data_preprocess.load_cache import load_cache
from data_preprocess.load_cache import EXPECTED_CLASS_ORDER, EXPECTED_LEADS
from data_preprocess.split_cache import build_cache_splits
from util.tests.test_split_cache import _write_pn2021_cache, _write_ptbxl_cache


def _build_test_splits(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    ptbxl_cache = _write_ptbxl_cache(tmp_path / "ptbxl")
    pn2021_cache = _write_pn2021_cache(tmp_path / "pn2021")
    seed_path = tmp_path / "random_seed.yaml"
    seed_path.write_text(
        "schema_version: 1\nrandom_seed: 20260501\n", encoding="utf-8"
    )
    output_root = tmp_path / "splits"
    config_path = tmp_path / "splits.yaml"
    config = {
        "schema_version": 1,
        "random_seed": {"file": str(seed_path), "namespace": "runtime_test_v1"},
        "output": {"root_dir": str(output_root), "if_exists": "error"},
        "ptbxl": {
            "cache_dir": str(ptbxl_cache),
            "sampling_rate_hz": 100,
            "split_id": "ptbxl_test",
            "output_subdir": "ptbxl",
            "eligible_quality_status": ["clean", "repaired"],
            "all_zero_policy": "exclude",
            "folds": {
                "train": list(range(1, 9)),
                "validation": [9],
                "test": [10],
            },
            "patient_id_column": "patient_id",
            "require_patient_disjoint": True,
        },
        "pn2021": {
            "cache_dir": str(pn2021_cache),
            "sampling_rate_hz": 100,
            "split_id": "pn2021_test",
            "output_subdir": "pn2021",
            "required_mapping_version": "v7_super5_sjr_rgq_review_20260528",
            "required_mapping_hash": "555ec85d5b51",
            "eligible_quality_status": ["clean", "repaired"],
            "k": 2,
            "k500_all_zero_policy": "exclude",
            "sampling": {
                "policy": "uniform_without_replacement",
                "candidate_order": "hash_id_lexicographic",
                "replacement": False,
            },
            "logical_centers": [
                {"name": "ningbo", "source_centers": ["ningbo"]},
                {
                    "name": "chapman_shaoxing",
                    "source_centers": ["chapman_shaoxing"],
                },
                {
                    "name": "cpsc_2018",
                    "source_centers": ["cpsc_2018", "cpsc_2018_extra"],
                },
                {"name": "georgia", "source_centers": ["georgia"]},
            ],
            "ignored_source_centers": ["ptb", "st_petersburg_incart"],
            "hard_excluded_source_centers": ["ptb-xl", "ptbxl"],
            "training": {
                "use_all_k500": True,
                "validation_split": False,
                "checkpoint_policy": "last_checkpoint_only",
            },
            "evaluation": {
                "exclude_k500_hashes": True,
                "emit_all_zero_kept": True,
                "emit_drop_all_zero": True,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    build_cache_splits(config_path)
    return ptbxl_cache, pn2021_cache, output_root / "ptbxl", output_root / "pn2021"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_runtime_corruption_cache(root: Path, source_cache: Path) -> Path:
    root.mkdir()
    source_signals = np.load(source_cache / "signals.npy", allow_pickle=False)
    labels = np.load(source_cache / "labels_super5.npy", allow_pickle=False)
    record_ids = np.load(source_cache / "record_ids.npy", allow_pickle=False)
    hash_ids = np.load(source_cache / "hash_ids.npy", allow_pickle=False)
    records = pd.read_parquet(source_cache / "records.parquet")
    np.save(
        root / "signals_100hz.npy",
        np.stack((source_signals + 1.0, source_signals + 2.0)),
        allow_pickle=False,
    )
    np.save(root / "labels_super5.npy", labels, allow_pickle=False)
    np.save(root / "record_ids.npy", record_ids, allow_pickle=False)
    np.save(root / "hash_ids.npy", hash_ids, allow_pickle=False)
    records.to_parquet(root / "records.parquet", index=False)
    compositions = [
        {
            "view_index": 0,
            "depth": 2,
            "operators": ["powerline_noise", "emg_noise"],
            "composition_id": "d2__powerline_noise__emg_noise",
        },
        {
            "view_index": 1,
            "depth": 3,
            "operators": [
                "baseline_wander",
                "baseline_shift",
                "random_leads_masking",
            ],
            "composition_id": (
                "d3__baseline_wander__baseline_shift__random_leads_masking"
            ),
        },
    ]
    (root / "compositions.json").write_text(
        json.dumps(compositions), encoding="utf-8"
    )
    (root / "build_state.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    count = len(records)
    manifest = {
        "schema_version": 1,
        "dataset": "pn2021c",
        "cache_version": "runtime-test",
        "record_count": count,
        "view_count": 2,
        "source": {
            "manifest_sha256": _sha256(source_cache / "manifest.json"),
            "mapping_version": "v7_super5_sjr_rgq_review_20260528",
            "mapping_hash": "555ec85d5b51",
            "class_order": list(EXPECTED_CLASS_ORDER),
            "lead_order": list(EXPECTED_LEADS),
            "physical_unit": "mV",
            "normalization": "none",
        },
        "waveforms": {
            "100hz": {
                "file": "signals_100hz.npy",
                "shape": [2, count, 1000, 12],
                "dtype": "float32",
                "sampling_rate_hz": 100,
                "duration_seconds": 10,
                "layout": "view_record_time_channel",
                "normalization": "none",
            }
        },
        "files": {
            "labels": "labels_super5.npy",
            "record_ids": "record_ids.npy",
            "hash_ids": "hash_ids.npy",
            "metadata": "records.parquet",
            "compositions": "compositions.json",
            "build_state": "build_state.json",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_load_selection_validates_ptbxl_and_combined_cpsc(tmp_path: Path) -> None:
    ptb_cache_dir, pn_cache_dir, ptb_split_dir, pn_split_dir = _build_test_splits(
        tmp_path
    )
    with load_cache(ptb_cache_dir, mode="mmap") as cache:
        train = load_selection(cache, ptb_split_dir, partition="train")
        assert len(train) == 7
        assert train.dataset == "ptbxl"
        assert train.logical_center is None
        assert tuple(cache.hash_ids[train.indices].astype(str)) == tuple(train.hash_ids)

    with load_cache(pn_cache_dir, mode="mmap") as cache:
        k500 = load_selection(
            cache, pn_split_dir, partition="k500", logical_center="cpsc_2018"
        )
        evaluation = load_selection(
            cache,
            pn_split_dir,
            partition="evaluation_all_zero_kept",
            logical_center="cpsc_2018",
        )
        assert k500.source_centers == ("cpsc_2018", "cpsc_2018_extra")
        assert {cache.records.iloc[index]["center"] for index in k500.indices} == {
            "cpsc_2018",
            "cpsc_2018_extra",
        }
        assert_disjoint_selections(k500, evaluation)


def test_model_transform_sanitizes_after_augmentation_then_global_zscores() -> None:
    signal = torch.linspace(-1.0, 1.0, 120, dtype=torch.float32).reshape(10, 12)
    original = signal.clone()

    def augmentation(value: torch.Tensor) -> torch.Tensor:
        output = value + 2.0
        output[0, 0] = float("nan")
        output[0, 1] = float("inf")
        return output

    transformed = prepare_model_input(
        signal,
        augmentation=augmentation,
        output_layout="channel_time",
    )
    assert transformed.shape == (12, 10)
    assert torch.isfinite(transformed).all()
    assert torch.allclose(transformed.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(
        transformed.std(unbiased=False), torch.tensor(1.0), atol=1e-5
    )
    torch.testing.assert_close(signal, original)

    flat = per_sample_global_zscore(torch.ones(2, 10, 12))
    torch.testing.assert_close(flat, torch.zeros_like(flat))


def test_runtime_dataset_and_dataloader_are_deterministic(tmp_path: Path) -> None:
    ptb_cache_dir, _, ptb_split_dir, _ = _build_test_splits(tmp_path)
    seed_path = tmp_path / "loader_seed.yaml"
    seed_path.write_text(
        "schema_version: 1\nrandom_seed: 20260501\n", encoding="utf-8"
    )
    dataset = RuntimeECGDataset(
        cache_dir=ptb_cache_dir,
        split_dir=ptb_split_dir,
        partition="train",
        cache_mode="mmap",
        transform=ECGModelTransform(output_layout="channel_time"),
    )
    try:
        item = dataset[0]
        assert item["waveform"].shape == (12, 1000)
        assert item["label"].shape == (5,)
        assert item["hash_id"] == dataset.selection.hash_ids[0]
        assert torch.isfinite(item["waveform"]).all()

        loader_a, seed_a = build_dataloader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=0,
            seed_namespace="same_loader",
            seed_config_path=seed_path,
        )
        loader_b, seed_b = build_dataloader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=0,
            seed_namespace="same_loader",
            seed_config_path=seed_path,
        )
        assert seed_a == seed_b
        assert next(iter(loader_a))["hash_id"] == next(iter(loader_b))["hash_id"]

        worker_loader, _ = build_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=1,
            persistent_workers=False,
            seed_namespace="worker_loader",
            seed_config_path=seed_path,
        )
        worker_batch = next(iter(worker_loader))
        assert worker_batch["waveform"].shape == (2, 12, 1000)
    finally:
        dataset.close()


def test_pn2021c_runtime_requires_ref_excluded_partition_and_explicit_view(
    tmp_path: Path,
) -> None:
    _, pn_cache_dir, _, pn_split_dir = _build_test_splits(tmp_path)
    corruption_dir = _write_runtime_corruption_cache(
        tmp_path / "pn2021c", pn_cache_dir
    )
    with pytest.raises(ValueError, match="K500 training anchors"):
        RuntimeECGDataset(
            cache_dir=corruption_dir,
            split_dir=pn_split_dir,
            partition="k500",
            logical_center="cpsc_2018",
            cache_mode="mmap",
            view=0,
        )
    with pytest.raises(ValueError, match="requires an explicit view"):
        RuntimeECGDataset(
            cache_dir=corruption_dir,
            split_dir=pn_split_dir,
            partition="evaluation_drop_all_zero",
            logical_center="cpsc_2018",
            cache_mode="mmap",
        )
    with RuntimeECGDataset(
        cache_dir=corruption_dir,
        split_dir=pn_split_dir,
        partition="evaluation_drop_all_zero",
        logical_center="cpsc_2018",
        cache_mode="mmap",
        view="d2__powerline_noise__emg_noise",
    ) as dataset:
        item = dataset[0]
        assert item["view_index"].item() == 0
        assert item["composition_id"] == "d2__powerline_noise__emg_noise"


def test_get_dataloader_covers_ptbxl_k500_and_refexcluded_views(
    tmp_path: Path,
) -> None:
    _, pn_cache_dir, _, _ = _build_test_splits(tmp_path)
    split_config_path = tmp_path / "splits.yaml"

    ptb_loader = get_dataloader(
        dataset="ptbxl",
        partition="validation",
        split_config_path=split_config_path,
        batch_size=1,
        num_workers=0,
    )
    try:
        ptb_batch = next(iter(ptb_loader))
        assert ptb_batch["waveform"].shape == (1, 12, 1000)
        assert ptb_loader.dataset.selection.partition == "validation"
    finally:
        ptb_loader.close()

    k500_loader = get_dataloader(
        dataset="pn2021",
        partition="k500",
        logical_center="cpsc_2018",
        split_config_path=split_config_path,
        batch_size=2,
        num_workers=0,
    )
    try:
        assert len(k500_loader.dataset) == 2
        assert k500_loader.dataset.selection.source_centers == (
            "cpsc_2018",
            "cpsc_2018_extra",
        )
    finally:
        k500_loader.close()

    eval_loader = get_dataloader(
        dataset="pn2021",
        partition="pn2021_drop_all_zero_refexcluded",
        logical_center="cpsc_2018",
        split_config_path=split_config_path,
        batch_size=2,
        num_workers=0,
    )
    try:
        assert eval_loader.dataset.selection.partition == "evaluation_drop_all_zero"
        assert eval_loader.dataset.selection.is_ref_excluded_evaluation
    finally:
        eval_loader.close()

    corruption_dir = _write_runtime_corruption_cache(
        tmp_path / "runtime-pn2021c", pn_cache_dir
    )
    corruption_config = tmp_path / "augmentation-cache.yaml"
    corruption_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset": "pn2021c",
                "output": {"cache_dir": str(corruption_dir)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    corruption_loader = get_dataloader(
        dataset="pn2021c",
        partition="pn2021c_drop_all_zero_corrupted_refexcluded",
        logical_center="cpsc_2018",
        view=0,
        split_config_path=split_config_path,
        corruption_cache_config_path=corruption_config,
        batch_size=2,
        num_workers=0,
    )
    try:
        batch = next(iter(corruption_loader))
        assert batch["waveform"].shape == (2, 12, 1000)
        assert batch["view_index"].tolist() == [0, 0]
    finally:
        corruption_loader.close()


def test_get_dataloader_uses_yaml_defaults_and_explicit_overrides(
    tmp_path: Path,
) -> None:
    _build_test_splits(tmp_path)
    data_load_path = tmp_path / "data_load.yaml"
    data_load_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataloader": {
                    "batch_size": 3,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "prefetch_factor": 2,
                    "drop_last": False,
                    "cache_mode": "mmap",
                    "validate_values": "sample",
                    "shuffle_partitions": ["train", "k500"],
                },
                "model_input": {
                    "prepare_for_model": True,
                    "sanitize": True,
                    "global_zscore": True,
                    "output_layout": "channel_time",
                    "epsilon": 1.0e-6,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    loader = get_dataloader(
        dataset="ptbxl",
        partition="train",
        split_config_path=tmp_path / "splits.yaml",
        data_load_config_path=data_load_path,
    )
    try:
        assert loader.batch_size == 3
        assert loader.num_workers == 0
        assert loader.pin_memory is False
        assert loader.runtime_config_identity["config_sha256"] == _sha256(
            data_load_path
        )
        assert loader.runtime_config_identity["resolved"]["shuffle"] is True
    finally:
        loader.close()

    overridden = get_dataloader(
        dataset="ptbxl",
        partition="validation",
        batch_size=2,
        pin_memory=True,
        split_config_path=tmp_path / "splits.yaml",
        data_load_config_path=data_load_path,
    )
    try:
        assert overridden.batch_size == 2
        assert overridden.pin_memory is True
        assert overridden.runtime_config_identity["resolved"]["shuffle"] is False
    finally:
        overridden.close()
