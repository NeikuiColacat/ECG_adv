from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import data_preprocess.data_runtime as runtime_module
from data_preprocess.data_runtime import (
    DataLoaderSeedIdentity,
    ECGModelTransform,
    MMapPrefetchConfig,
    RuntimeDataLoader,
    RuntimeECGDataset,
    SelectionResidentECGDataset,
    SequentialEvaluationDataSession,
    assert_disjoint_selections,
    build_dataloader,
    get_dataloader,
    load_data_load_config,
    load_selection,
    per_sample_global_zscore,
    prepare_model_input,
)
from data_preprocess.load_cache import load_cache
from data_preprocess.load_cache import EXPECTED_CLASS_ORDER, EXPECTED_LEADS
from data_preprocess.split_cache import build_cache_splits
from util.tests.test_split_cache import _write_pn2021_cache, _write_ptbxl_cache


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_data_load_config_is_the_single_runtime_profile() -> None:
    config_path = PROJECT_ROOT / "configs/data/data_load.yaml"
    config = load_data_load_config(config_path)

    assert config.path == config_path.resolve()
    assert config.cache_mode == "mmap"
    assert config.mmap_access_order == "cache_index"
    assert config.mmap_batch_read is True
    assert config.mmap_prefetch.enabled is True
    assert config.selection_resident is False
    assert config.selection_resident_pin_memory is False


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
            "tuning_split": {
                "parent_partition": "k500",
                "train_partition": "k500_tune_train",
                "validation_partition": "k500_tune_validation",
                "train_count": 1,
                "validation_count": 1,
                "policy": "deterministic_multilabel_source_stratified",
                "preserve_source_proportions": True,
                "singleton_positive_policy": "prefer_train",
                "seed_namespace": "runtime_test_k500_tuning_split_v1",
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


def test_runtime_dataloader_close_shuts_down_persistent_iterator(tmp_path: Path) -> None:
    class _Iterator:
        def __init__(self) -> None:
            self.shutdown_called = False

        def _shutdown_workers(self) -> None:
            self.shutdown_called = True

    identity = DataLoaderSeedIdentity(
        base_seed=1,
        effective_seed=2,
        namespace="close-test",
        config_path=tmp_path / "seed.yaml",
        config_sha256="0" * 64,
    )
    loader = RuntimeDataLoader(
        [{"waveform": torch.zeros(1)}],
        batch_size=1,
        num_workers=0,
        runtime_seed_identity=identity,
    )
    iterator = _Iterator()
    loader._iterator = iterator

    loader.close()

    assert iterator.shutdown_called is True
    assert loader._iterator is None


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
        tune_train = load_selection(
            cache,
            pn_split_dir,
            partition="k500_tune_train",
            logical_center="cpsc_2018",
        )
        tune_validation = load_selection(
            cache,
            pn_split_dir,
            partition="k500_tune_validation",
            logical_center="cpsc_2018",
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
        assert len(tune_train) == 1
        assert len(tune_validation) == 1
        assert_disjoint_selections(tune_train, tune_validation)
        assert set(tune_train.hash_ids).union(tune_validation.hash_ids) == set(
            k500.hash_ids
        )
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


def test_runtime_dataset_reads_ordered_raw_mv_by_hash_without_transform(
    tmp_path: Path,
) -> None:
    ptb_cache_dir, _, ptb_split_dir, _ = _build_test_splits(tmp_path)
    dataset = RuntimeECGDataset(
        cache_dir=ptb_cache_dir,
        split_dir=ptb_split_dir,
        partition="train",
        cache_mode="mmap",
        transform=ECGModelTransform(output_layout="channel_time"),
    )
    requested = [str(dataset.selection.hash_ids[1]), str(dataset.selection.hash_ids[0])]
    try:
        items = dataset.get_raw_items_by_hashes(requested)
        assert [item["hash_id"] for item in items] == requested
        assert items[0]["waveform_raw"].shape == (1000, 12)
        assert items[0]["waveform_raw"].dtype == np.float32
        assert not items[0]["waveform_raw"].flags.writeable
        assert items[0]["physical_unit"] == "mV"
        assert items[0]["normalization"] == "none"
        assert tuple(items[0]["lead_order"]) == EXPECTED_LEADS
        assert items[0]["sampling_rate_hz"] == 100

        with load_cache(ptb_cache_dir, mode="mmap") as cache:
            expected = cache.get_record(
                int(dataset.selection.indices[1]), layout="time_channel"
            )
        np.testing.assert_array_equal(items[0]["waveform_raw"], expected.signal)
        assert dataset[1]["waveform"].shape == (12, 1000)

        with pytest.raises(KeyError, match="outside this runtime selection"):
            dataset.get_raw_items_by_hashes(["missing-hash"])
    finally:
        dataset.close()


def test_pn2021c_runtime_allows_only_validation_or_refexcluded_with_explicit_view(
    tmp_path: Path,
) -> None:
    _, pn_cache_dir, _, pn_split_dir = _build_test_splits(tmp_path)
    corruption_dir = _write_runtime_corruption_cache(
        tmp_path / "pn2021c", pn_cache_dir
    )
    for forbidden_partition in ("k500", "k500_tune_train"):
        with pytest.raises(ValueError, match="K500 training records"):
            RuntimeECGDataset(
                cache_dir=corruption_dir,
                split_dir=pn_split_dir,
                partition=forbidden_partition,
                logical_center="cpsc_2018",
                cache_mode="mmap",
                view=0,
            )
    with RuntimeECGDataset(
        cache_dir=corruption_dir,
        split_dir=pn_split_dir,
        partition="k500_tune_validation",
        logical_center="cpsc_2018",
        cache_mode="mmap",
        view=0,
    ) as validation_dataset:
        validation_item = validation_dataset[0]
        assert validation_dataset.selection.partition == "k500_tune_validation"
        assert validation_item["view_index"].item() == 0
        assert validation_item["composition_id"] == (
            "d2__powerline_noise__emg_noise"
        )
    with pytest.raises(ValueError, match="requires an explicit view"):
        RuntimeECGDataset(
            cache_dir=corruption_dir,
            split_dir=pn_split_dir,
            partition="k500_tune_validation",
            logical_center="cpsc_2018",
            cache_mode="mmap",
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

    tune_train_loader = get_dataloader(
        dataset="pn2021",
        partition="k500_tune_train",
        logical_center="cpsc_2018",
        split_config_path=split_config_path,
        batch_size=1,
        num_workers=0,
    )
    try:
        assert len(tune_train_loader.dataset) == 1
        assert tune_train_loader.runtime_config_identity["resolved"]["shuffle"] is True
        assert tune_train_loader.dataset.selection.partition == "k500_tune_train"
    finally:
        tune_train_loader.close()

    tune_validation_loader = get_dataloader(
        dataset="pn2021",
        partition="k500_tune_validation",
        logical_center="cpsc_2018",
        split_config_path=split_config_path,
        batch_size=1,
        num_workers=0,
    )
    try:
        assert len(tune_validation_loader.dataset) == 1
        assert tune_validation_loader.runtime_config_identity["resolved"][
            "shuffle"
        ] is False
        assert (
            tune_validation_loader.dataset.selection.partition
            == "k500_tune_validation"
        )
    finally:
        tune_validation_loader.close()

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
                "mmap": {
                    "access_order": "cache_index",
                    "batch_read": True,
                    "prefetch": {
                        "enabled": True,
                        "advice": "sequential_willneed",
                        "window_mib": 1,
                        "ahead_batches": 2,
                    },
                },
                "selection_residency": {
                    "enabled": False,
                    "pin_memory": False,
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
        assert loader.runtime_config_identity["resolved"]["mmap"] == {
            "access_order": "split",
            "batch_read": True,
            "prefetch": {
                "enabled": False,
                "advice": "sequential_willneed",
                "window_mib": 1,
                "ahead_batches": 2,
            },
        }
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
        assert overridden.runtime_config_identity["resolved"]["mmap"][
            "access_order"
        ] == "cache_index"
        assert overridden.runtime_config_identity["resolved"]["mmap"][
            "prefetch"
        ]["enabled"] is True
    finally:
        overridden.close()


def test_runtime_cache_index_order_preserves_selection_and_batch_values(
    tmp_path: Path,
) -> None:
    _, _, _, pn_split_dir = _build_test_splits(tmp_path)
    pn_cache_dir = tmp_path / "pn2021"
    split_order = RuntimeECGDataset(
        cache_dir=pn_cache_dir,
        split_dir=pn_split_dir,
        partition="evaluation_all_zero_kept",
        logical_center="cpsc_2018",
        cache_mode="mmap",
        transform=ECGModelTransform(output_layout="channel_time"),
    )
    cache_order = RuntimeECGDataset(
        cache_dir=pn_cache_dir,
        split_dir=pn_split_dir,
        partition="evaluation_all_zero_kept",
        logical_center="cpsc_2018",
        cache_mode="mmap",
        access_order="cache_index",
        batch_read=True,
        transform=ECGModelTransform(output_layout="channel_time"),
    )
    try:
        assert cache_order.selection.hash_id_set_sha256 == (
            split_order.selection.hash_id_set_sha256
        )
        assert set(cache_order.selection.hash_ids) == set(split_order.selection.hash_ids)
        assert np.all(np.diff(cache_order.selection.indices) > 0)
        positions = list(range(min(3, len(cache_order))))
        batch_items = cache_order.__getitems__(positions)
        scalar_items = [cache_order[position] for position in positions]
        for batched, scalar in zip(batch_items, scalar_items):
            assert batched["hash_id"] == scalar["hash_id"]
            torch.testing.assert_close(batched["waveform"], scalar["waveform"])
            torch.testing.assert_close(batched["label"], scalar["label"])
    finally:
        split_order.close()
        cache_order.close()


def test_pn2021c_mmap_prefetch_advises_only_current_view_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pn_cache_dir, _, pn_split_dir = _build_test_splits(tmp_path)
    corruption_dir = _write_runtime_corruption_cache(
        tmp_path / "runtime-pn2021c-prefetch", pn_cache_dir
    )
    calls: list[tuple[int, int, int, int]] = []

    def fake_posix_fadvise(fd: int, offset: int, length: int, advice: int) -> None:
        calls.append((fd, offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", fake_posix_fadvise)
    dataset = RuntimeECGDataset(
        cache_dir=corruption_dir,
        split_dir=pn_split_dir,
        partition="evaluation_all_zero_kept",
        logical_center="cpsc_2018",
        cache_mode="mmap",
        view=1,
        access_order="cache_index",
        batch_read=True,
        mmap_prefetch=MMapPrefetchConfig(
            enabled=True,
            advice="sequential_willneed",
            window_mib=1,
            ahead_batches=2,
        ),
        transform=ECGModelTransform(output_layout="channel_time"),
    )
    try:
        positions = list(range(min(2, len(dataset))))
        dataset.__getitems__(positions)
        cache = dataset._get_cache()
        assert isinstance(cache.signals, np.memmap)
        record_bytes = 1000 * 12 * np.dtype(np.float32).itemsize
        view_bytes = len(cache) * record_bytes
        view_offset = int(cache.signals.offset) + view_bytes
        assert calls[0][1:] == (
            view_offset,
            view_bytes,
            os.POSIX_FADV_SEQUENTIAL,
        )
        selected = dataset.selection.indices[np.asarray(positions)]
        next_record = int(selected.max()) + 1
        expected_length = min(
            1 * 1024**2,
            len(positions) * 2 * record_bytes,
            (len(cache) - next_record) * record_bytes,
        )
        assert calls[1][1:] == (
            view_offset + next_record * record_bytes,
            expected_length,
            os.POSIX_FADV_WILLNEED,
        )
    finally:
        dataset.close()


def test_selection_residency_preserves_k500_sampler_hashes_and_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, pn_split_dir = _build_test_splits(tmp_path)
    opened_caches = []
    original_load_cache = runtime_module.load_cache

    def tracked_load_cache(*args, **kwargs):
        cache = original_load_cache(*args, **kwargs)
        opened_caches.append(cache)
        return cache

    monkeypatch.setattr(runtime_module, "load_cache", tracked_load_cache)
    common = {
        "dataset": "pn2021",
        "partition": "k500",
        "logical_center": "cpsc_2018",
        "sampling_rate_hz": 100,
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "cache_mode": "mmap",
        "prepare_for_model": False,
        "split_config_path": tmp_path / "splits.yaml",
        "split_dir": pn_split_dir,
        "seed_namespace": "resident_rng_contract",
    }
    mmap_loader = get_dataloader(**common, selection_resident=False)
    resident_loader = get_dataloader(
        **common,
        selection_resident=True,
        selection_resident_pin_memory=False,
    )
    try:
        assert isinstance(resident_loader.dataset, SelectionResidentECGDataset)
        assert resident_loader.dataset.waveforms.device.type == "cpu"
        assert resident_loader.dataset.waveforms.is_contiguous()
        assert resident_loader.dataset.is_pinned is False
        assert opened_caches[-1]._closed is True
        assert resident_loader.dataset.selection.hash_id_set_sha256 == (
            mmap_loader.dataset.selection.hash_id_set_sha256
        )
        assert resident_loader.dataset.selection.source_centers == (
            "cpsc_2018",
            "cpsc_2018_extra",
        )

        mmap_batches = list(mmap_loader)
        resident_batches = list(resident_loader)
        assert [batch["hash_id"] for batch in resident_batches] == [
            batch["hash_id"] for batch in mmap_batches
        ]
        assert [batch["source_center"] for batch in resident_batches] == [
            batch["source_center"] for batch in mmap_batches
        ]
        for mmap_batch, resident_batch in zip(
            mmap_batches, resident_batches, strict=True
        ):
            torch.testing.assert_close(
                resident_batch["waveform"], mmap_batch["waveform"]
            )
            torch.testing.assert_close(resident_batch["label"], mmap_batch["label"])
            assert resident_batch["cache_index"].tolist() == mmap_batch[
                "cache_index"
            ].tolist()
        assert resident_loader.describe()["dataset"]["residency"][
            "source_cache_closed_after_gather"
        ] is True
    finally:
        mmap_loader.close()
        resident_loader.close()


def test_sequential_evaluation_session_reuses_cache_and_selection_across_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pn_cache_dir, _, pn_split_dir = _build_test_splits(tmp_path)
    corruption_cache = _write_runtime_corruption_cache(
        tmp_path / "pn2021c", pn_cache_dir
    )
    load_calls = []
    original_load_cache = runtime_module.load_cache

    def tracked_load_cache(*args, **kwargs):
        load_calls.append((Path(args[0]).resolve(), kwargs["sampling_rate_hz"]))
        return original_load_cache(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "load_cache", tracked_load_cache)
    common = {
        "partition": "evaluation_all_zero_kept",
        "logical_center": "ningbo",
        "sampling_rate_hz": 100,
        "batch_size": 2,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "cache_mode": "mmap",
        "validate_values": "sample",
        "mmap_prefetch_enabled": False,
        "prepare_for_model": False,
        "split_config_path": tmp_path / "splits.yaml",
        "split_dir": pn_split_dir,
    }
    with SequentialEvaluationDataSession() as session:
        clean_loader = session.get_dataloader(
            **common,
            dataset="pn2021",
            view=None,
            cache_dir=pn_cache_dir,
        )
        clean_hashes = next(iter(clean_loader))["hash_id"]
        clean_loader.close()
        assert session.describe()["cache_open_count"] == 1

        view0_loader = session.get_dataloader(
            **common,
            dataset="pn2021c",
            view=0,
            cache_dir=corruption_cache,
        )
        view0 = next(iter(view0_loader))
        assert view0["hash_id"] == clean_hashes
        assert view0_loader.describe()["dataset"]["view"] == 0
        view0_loader.close()
        first_description = session.describe()
        assert first_description["cache_open_count"] == 2
        assert first_description["selection_count"] == 2

        view1_loader = session.get_dataloader(
            **common,
            dataset="pn2021c",
            view=1,
            cache_dir=corruption_cache,
        )
        view1 = next(iter(view1_loader))
        assert view1["hash_id"] == view0["hash_id"]
        assert view1["view_index"].tolist() == [1, 1]
        assert not torch.equal(view1["waveform"], view0["waveform"])
        view1_loader.close()
        assert session.describe()["cache_open_count"] == 2
        assert session.describe()["selection_count"] == 2
        assert load_calls == [
            (pn_cache_dir.resolve(), 100),
            (corruption_cache.resolve(), 100),
        ]

    assert session.describe() == {
        "closed": True,
        "cache_open_count": 0,
        "selection_count": 0,
        "caches": [],
    }
