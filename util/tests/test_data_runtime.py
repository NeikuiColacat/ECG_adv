"""Small CPU contracts for the retained runtime data boundary."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from data_preprocess import data_ledger, data_runtime, load_cache as load_cache_module
from data_preprocess.data_runtime import (
    ECGSelection,
    MMapPrefetchConfig,
    PN2021K500LoaderPlan,
    RuntimeECGDataset,
    SelectionResidentECGDataset,
    SequentialEvaluationDataSession,
    _build_torch_loader,
    _load_runtime_defaults,
)
from data_preprocess.load_cache import (
    EXPECTED_CLASS_ORDER,
    PN2021_MAPPING_HASH,
    PN2021_MAPPING_VERSION,
    ECGCache,
    _close_memmap,
    _open_npy,
    load_cache,
)


REPO = Path(__file__).resolve().parents[2]


def _data_load_fixture(tmp_path: Path, descriptor: dict[str, object]) -> Path:
    payload = yaml.safe_load(
        (REPO / "configs" / "data" / "data_load.yaml").read_text(encoding="utf-8")
    )
    payload["content_ledger"] = descriptor
    path = tmp_path / "configs" / "data" / "data_load.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _split_location_fixture(
    tmp_path: Path, *, cache_dir: Path, sampling_rate_hz: object = 100
) -> Path:
    path = tmp_path / "splits.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "output": {"root_dir": str(tmp_path / "split-artifacts")},
                "pn2021": {
                    "cache_dir": str(cache_dir),
                    "sampling_rate_hz": sampling_rate_hz,
                    "output_subdir": "pn2021",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _selection(partition: str, hash_ids: tuple[str, ...]) -> ECGSelection:
    count = len(hash_ids)
    return ECGSelection(
        dataset="pn2021",
        cache_dataset="pn2021",
        partition=partition,
        logical_center="ningbo",
        source_centers=("ningbo",),
        indices=np.arange(count, dtype=np.int64),
        hash_ids=np.asarray(hash_ids),
        record_ids=np.asarray([f"record-{index}" for index in range(count)]),
        split_id="fixture",
        split_manifest_path=Path("fixture.json"),
        split_manifest_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        hash_id_set_sha256="c" * 64,
        class_order=EXPECTED_CLASS_ORDER,
        mapping_version="v7_super5_sjr_rgq_review_20260528",
        mapping_hash="555ec85d5b51",
    )


def _hash_id_set(values: np.ndarray) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values.astype(str))).encode()
    return hashlib.sha256(payload).hexdigest()


def _tiny_mmap_cache(
    tmp_path: Path,
    name: str,
    *,
    dataset: str = "pn2021",
    record_count: int = 40,
    points: int = 8,
    source_manifest_sha256: str = "1" * 64,
) -> ECGCache:
    cache_dir = (tmp_path / name).resolve()
    cache_dir.mkdir()
    view_count = 2 if dataset == "pn2021c" else 1
    signal_shape = (
        (view_count, record_count, points, 2)
        if dataset == "pn2021c"
        else (record_count, points, 2)
    )
    signals_path = cache_dir / "signals.npy"
    writable = np.lib.format.open_memmap(
        signals_path, mode="w+", dtype=np.float32, shape=signal_shape
    )
    writable[...] = np.arange(np.prod(signal_shape), dtype=np.float32).reshape(
        signal_shape
    )
    writable.flush()
    del writable

    labels = np.zeros((record_count, 5), dtype=np.uint8)
    labels[np.arange(record_count), np.arange(record_count) % 5] = 1
    record_ids = np.asarray([f"record-{index:03d}" for index in range(record_count)])
    hash_ids = np.asarray(
        [hashlib.sha256(f"pn2021:{index}".encode()).hexdigest() for index in range(record_count)]
    )
    is_corruption = dataset == "pn2021c"
    raw_manifest = (
        {"source": {
            "manifest_sha256": source_manifest_sha256,
            "mapping_version": PN2021_MAPPING_VERSION,
            "mapping_hash": PN2021_MAPPING_HASH,
        }}
        if is_corruption
        else {}
    )
    identity = SimpleNamespace(
        cache_dir=cache_dir, manifest_sha256=("2" * 64 if is_corruption else source_manifest_sha256),
        raw_manifest=raw_manifest, is_corruption=is_corruption, dataset=dataset,
        cache_version="tiny-characterization-v1", record_count=record_count,
        view_count=view_count, sampling_rate_hz=100, duration_seconds=10.0,
        signal_shape=signal_shape, lead_order=("I", "II"), physical_unit="mV",
        normalization="none", class_order=EXPECTED_CLASS_ORDER,
        mapping_version=PN2021_MAPPING_VERSION, mapping_hash=PN2021_MAPPING_HASH,
        signals_path=signals_path,
    )
    compositions = tuple(
        {"view_index": index, "depth": 2, "composition_id": name}
        for index, name in enumerate(
            ("d2__powerline_noise__emg_noise", "d2__baseline_wander__baseline_shift")
        )
    ) if is_corruption else ()
    records = pd.DataFrame({
        "cache_index": np.arange(record_count), "record_id": record_ids,
        "hash_id": hash_ids, "center": ["ningbo"] * record_count,
    })
    return ECGCache(
        identity=identity,
        signals=np.load(signals_path, mmap_mode="r", allow_pickle=False),
        labels=labels,
        record_ids=record_ids,
        hash_ids=hash_ids,
        records=records,
        compositions=compositions,
    )


def _tiny_selection(
    cache: ECGCache,
    indices: list[int],
    *,
    partition: str = "k500",
) -> ECGSelection:
    selected = np.asarray(indices, dtype=np.int64)
    hashes = np.asarray(cache.hash_ids[selected]).astype(str)
    source_sha = (
        str(cache.identity.raw_manifest["source"]["manifest_sha256"])
        if cache.is_corruption
        else cache.identity.manifest_sha256
    )
    return replace(
        _selection(partition, tuple(hashes)),
        cache_dataset=cache.dataset,
        indices=selected,
        record_ids=np.asarray(cache.record_ids[selected]).astype(str),
        source_manifest_sha256=source_sha,
        hash_id_set_sha256=_hash_id_set(hashes),
    )


def _runtime_dataset(
    monkeypatch: pytest.MonkeyPatch,
    cache: ECGCache,
    selection: ECGSelection,
) -> RuntimeECGDataset:
    monkeypatch.setattr(data_runtime, "load_cache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(data_runtime, "load_selection", lambda *args, **kwargs: selection)
    return RuntimeECGDataset(
        cache_dir=cache.identity.cache_dir,
        split_dir=cache.identity.cache_dir / "splits",
        partition=selection.partition,
        logical_center=selection.logical_center,
        cache_mode="mmap",
        validate_values="none",
        access_order="split",
    )


def _assert_runtime_items_equal(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> None:
    for actual_item, expected_item in zip(actual, expected, strict=True):
        for key in actual_item:
            if isinstance(actual_item[key], torch.Tensor):
                torch.testing.assert_close(actual_item[key], expected_item[key])
            else:
                assert actual_item[key] == expected_item[key]


def _runtime_surface(kind: str, tmp_path: Path, **override: Any) -> Any:
    constructors: dict[str, Any] = {
        "ptbxl": data_runtime.PTBXLLoaderPlan,
        "k500": data_runtime.PN2021K500LoaderPlan,
        "evaluation": data_runtime.PN2021EvaluationLoaderPlan,
        "request": data_runtime._LoaderRequest,
    }
    values = {
        "train_partition": "train", "validation_partition": "validation",
        "test_partition": "test", "center": "ningbo",
        "clean_partition": "pn2021_all_zero_kept_refexcluded",
        "corrupted_partition": "pn2021c_all_zero_kept_corrupted_refexcluded",
        "train_batch_size": 32, "eval_batch_size": 64, "batch_size": 32,
        "num_workers": 0, "pin_memory": False, "persistent_workers": False,
        "prefetch_factor": 2, "cache_mode": "mmap",
        "validate_values": "sample", "drop_last": False,
        "selection_resident": False, "selection_resident_pin_memory": False,
        "seed_namespace": "strict-types", "dataset": "ptbxl",
        "partition": "train", "logical_center": None, "view": None,
        "shuffle": False, "evaluation_session": None,
        "split_config_path": tmp_path / "splits.yaml",
        "data_load_config_path": tmp_path / "data_load.yaml",
        "corruption_cache_config_path": tmp_path / "corruption.yaml",
        "seed_config_path": tmp_path / "random_seed.yaml",
    }
    constructor = constructors[kind]
    values.update(override)
    return constructor(**{name: values[name] for name in constructor.__dataclass_fields__})


@pytest.mark.parametrize(
    ("category", "invalid"),
    (("batch", True), ("workers", 0.0), ("boolean", "false"), ("enum", 1)),
)
@pytest.mark.parametrize("kind", ("ptbxl", "k500", "evaluation", "request"))
def test_finite_runtime_surfaces_reject_coercible_types(
    kind: str, category: str, invalid: Any, tmp_path: Path
) -> None:
    field = {"batch": "train_batch_size" if kind == "ptbxl" else "batch_size",
             "workers": "num_workers", "boolean": "pin_memory",
             "enum": "cache_mode"}[category]
    with pytest.raises(TypeError):
        _runtime_surface(kind, tmp_path, **{field: invalid})


def test_evaluation_plan_rejects_mismatched_all_zero_views(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same view"):
        _runtime_surface(
            "evaluation",
            tmp_path,
            corrupted_partition="pn2021c_drop_all_zero_corrupted_refexcluded",
        )


@pytest.mark.parametrize("kind", ("ptbxl", "k500", "evaluation"))
def test_finite_plan_describe_exposes_validated_runtime_policy(
    kind: str, tmp_path: Path
) -> None:
    description = _runtime_surface(kind, tmp_path).describe()
    assert {
        "batch_size" if kind != "ptbxl" else "train_batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "cache_mode",
        "validate_values",
        "drop_last",
        "split_config_path",
        "data_load_config_path",
        "seed_config_path",
    } <= description.keys()


def test_canonical_data_load_config_is_the_single_runtime_profile(
    tmp_path: Path,
) -> None:
    config_path = REPO / "configs" / "data" / "data_load.yaml"
    config = _load_runtime_defaults(config_path)

    assert config.path == config_path.resolve()
    assert config.mmap_access_order == "cache_index"
    assert config.describe()["mmap"]["batch_read"] is True
    assert config.mmap_prefetch.enabled is True
    assert config.mmap_prefetch.advice == "sequential_willneed"
    assert config.mmap_prefetch.window_mib == 1024
    assert config.mmap_prefetch.ahead_batches == 2
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["mmap"]["batch_read"] = False
    invalid_path = tmp_path / "data_load.yaml"
    invalid_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mmap.batch_read must be true"):
        _load_runtime_defaults(invalid_path)


def test_public_runtime_surface_is_only_the_five_finite_entrypoints() -> None:
    assert data_runtime.__all__ == [
        "PN2021EvaluationLoaderPlan",
        "PN2021K500LoaderPlan",
        "PTBXLLoaderPlan",
        "RuntimeDataLoader",
        "SequentialEvaluationDataSession",
    ]
    assert all(
        not hasattr(data_runtime.SelectionResidentECGDataset, name)
        for name in ("waveforms", "labels", "is_pinned")
    )
    assert load_cache_module.__all__ == [
        "EXPECTED_CLASS_ORDER", "EXPECTED_LEADS", "PN2021_MAPPING_HASH",
        "PN2021_MAPPING_VERSION", "CacheManifest", "ECGBatch", "ECGCache",
        "load_cache", "load_cache_manifest",
    ]
    assert {"ECGRecord", "SignalLayout"}.isdisjoint(vars(load_cache_module))
    assert {"get_record", "_normalize_index", "_format_signals"}.isdisjoint(
        vars(ECGCache)
    )
    assert tuple(inspect.signature(ECGCache.get_batch).parameters) == (
        "self", "indices", "view",
    )
    assert "batch_read" not in inspect.signature(RuntimeECGDataset).parameters


@pytest.mark.parametrize("mode", ("auto", "ram"))
def test_cache_rejects_non_mmap_mode_before_manifest_access(
    tmp_path: Path, mode: str
) -> None:
    missing = tmp_path / "missing-cache"
    with pytest.raises(ValueError, match="cache mode must be mmap"):
        load_cache(missing, mode=mode)  # type: ignore[arg-type]


def test_npy_cache_arrays_are_always_readonly_memmaps(tmp_path: Path) -> None:
    path = tmp_path / "tiny.npy"
    np.save(path, np.arange(8, dtype=np.float32), allow_pickle=False)
    array = _open_npy(path)
    try:
        assert isinstance(array, np.memmap)
        assert array.flags.writeable is False
    finally:
        _close_memmap(array)


def test_runtime_dataset_rejects_non_mmap_before_shared_cache_bypass(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="runtime cache_mode must be mmap"):
        RuntimeECGDataset(
            cache_dir=tmp_path,
            split_dir=tmp_path,
            partition="k500",
            cache_mode="ram",  # type: ignore[arg-type]
            shared_cache=object(),  # type: ignore[arg-type]
            shared_selection=object(),  # type: ignore[arg-type]
        )


def test_runtime_defaults_treat_ledger_descriptor_as_opaque(tmp_path: Path) -> None:
    path = _data_load_fixture(
        tmp_path,
        {"not": "a runtime-owned field"},
    )

    config = _load_runtime_defaults(path)

    assert config.path == path.resolve()
    assert not (tmp_path / "configs" / "data" / "missing-ledger.jsonl").exists()


def test_mmap_prefetch_config_rejects_unbounded_or_unknown_policy() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        MMapPrefetchConfig(window_mib=0)
    with pytest.raises(ValueError, match="none or sequential_willneed"):
        MMapPrefetchConfig(advice="aggressive")  # type: ignore[arg-type]


def test_runtime_dataset_batched_dense_and_sparse_reads_match_scalar_mmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _tiny_mmap_cache(tmp_path, "batched-cache")
    selection = _tiny_selection(cache, [1, 2, 4, 25, 39])
    batch_selectors: list[object] = []
    original_get_batch = cache.get_batch

    def record_batch(selector: object, **kwargs: Any):
        batch_selectors.append(selector)
        return original_get_batch(selector, **kwargs)

    monkeypatch.setattr(cache, "get_batch", record_batch)
    dataset = _runtime_dataset(monkeypatch, cache, selection)
    try:
        dense_positions = [0, 1, 2]
        sparse_positions = [0, 3, 4]
        dense = dataset.__getitems__(dense_positions)
        sparse = dataset.__getitems__(sparse_positions)
        _assert_runtime_items_equal(
            dense, [dataset[position] for position in dense_positions]
        )
        _assert_runtime_items_equal(
            sparse, [dataset[position] for position in sparse_positions]
        )

        assert isinstance(batch_selectors[0], slice)
        assert batch_selectors[0] == slice(1, 5)
        np.testing.assert_array_equal(
            batch_selectors[1], np.asarray([1, 25, 39], dtype=np.int64)
        )
        assert [int(item["cache_index"]) for item in dense] == [1, 2, 4]
        assert [int(item["cache_index"]) for item in sparse] == [1, 25, 39]
    finally:
        dataset.close()


def _collect_loader_identity(loader: Any) -> dict[str, Any]:
    result: dict[str, list[Any]] = {
        "hash_ids": [],
        "record_ids": [],
        "cache_indices": [],
        "waveforms": [],
        "labels": [],
    }
    for batch in loader:
        result["hash_ids"].extend(str(value) for value in batch["hash_id"])
        result["record_ids"].extend(str(value) for value in batch["record_id"])
        result["cache_indices"].extend(
            int(value) for value in batch["cache_index"].tolist()
        )
        result["waveforms"].append(batch["waveform"].clone())
        result["labels"].append(batch["label"].clone())
    result["waveforms"] = torch.cat(result["waveforms"])
    result["labels"] = torch.cat(result["labels"])
    return result


def test_selection_resident_matches_seeded_mmap_and_closes_source_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_indices = [8, 2, 11, 0, 5, 7, 3, 10]
    source_cache = _tiny_mmap_cache(tmp_path, "resident-gather-source", record_count=12)
    source_selection = _tiny_selection(source_cache, split_indices)
    source = _runtime_dataset(monkeypatch, source_cache, source_selection)
    source_mapping = source_cache.signals._mmap
    resident = SelectionResidentECGDataset(source)
    source.close()
    assert source_cache._closed is True
    assert source_mapping.closed is True

    mmap_cache = _tiny_mmap_cache(tmp_path, "comparison-mmap", record_count=12)
    mmap_selection = _tiny_selection(mmap_cache, split_indices)
    mmap_dataset = _runtime_dataset(monkeypatch, mmap_cache, mmap_selection)
    seed_config = tmp_path / "seed.yaml"
    seed_config.write_text(
        "schema_version: 1\nrandom_seed: 424242\n", encoding="utf-8"
    )
    namespace = "characterization_seeded_hash_order_v1"
    resident_loader, resident_seed = _build_torch_loader(
        resident,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        seed_namespace=namespace,
        seed_config_path=seed_config,
    )
    mmap_loader, mmap_seed = _build_torch_loader(
        mmap_dataset,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        seed_namespace=namespace,
        seed_config_path=seed_config,
    )
    try:
        resident_result = _collect_loader_identity(resident_loader)
        mmap_result = _collect_loader_identity(mmap_loader)
        assert resident_seed == mmap_seed
        assert resident_seed.base_seed == 424242
        assert resident_seed.effective_seed == 211764339
        assert resident_seed.config_sha256 == (
            "9121cc2ee7c1c247a579f40f54740cf535ce09f01389a9d7755df688ce678244"
        )
        assert resident_result["cache_indices"] == [10, 5, 3, 11, 2, 0, 7, 8]
        ordered_hash_digest = hashlib.sha256(
            "".join(f"{value}\n" for value in resident_result["hash_ids"]).encode()
        ).hexdigest()
        assert ordered_hash_digest == (
            "517669fbcaf52745b065b9f04e39b4e4ac630ca6880653610baddb78e687a894"
        )
        for key in resident_result:
            if isinstance(resident_result[key], torch.Tensor):
                torch.testing.assert_close(resident_result[key], mmap_result[key])
            else:
                assert resident_result[key] == mmap_result[key]
    finally:
        resident_loader.close()
        mmap_loader.close()


def test_k500_plan_opens_only_verified_ordered_and_training_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection("k500", tuple(f"hash-{index}" for index in range(500)))
    calls: list[data_runtime._LoaderRequest] = []

    class Dataset:
        sampling_rate_hz = 100

        def __init__(self, order: str) -> None:
            self.selection = selection
            self.order = order

        def describe(self) -> dict[str, Any]:
            return {"mmap": {"access_order": self.order}}

    class Loader:
        drop_last = False

        def __init__(self, order: str) -> None:
            self.dataset = Dataset(order)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def open_loader(request: data_runtime._LoaderRequest) -> Loader:
        calls.append(request)
        return Loader("split" if request.shuffle else "cache_index")

    monkeypatch.setattr(data_runtime, "_build_runtime_loader", open_loader)
    plan = PN2021K500LoaderPlan(
        center="ningbo",
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        cache_mode="mmap",
        validate_values="sample",
        selection_resident=True,
        selection_resident_pin_memory=False,
        drop_last=False,
        seed_namespace="locked-k500",
        split_config_path=tmp_path / "splits.yaml",
        data_load_config_path=tmp_path / "data_load.yaml",
        seed_config_path=tmp_path / "random_seed.yaml",
    )

    plan.open_ordered()
    plan.open_training()

    assert [call.shuffle for call in calls] == [False, True]
    assert all(call.drop_last is False for call in calls)
    assert all(call.dataset == "pn2021" for call in calls)
    assert all(call.partition == "k500" for call in calls)
    assert all(call.view is None for call in calls)


def test_k500_plan_closes_loader_on_early_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection("k500", tuple(f"hash-{index}" for index in range(499)))
    dataset = SimpleNamespace(
        selection=selection,
        sampling_rate_hz=100,
        describe=lambda: {"mmap": {"access_order": "cache_index"}},
    )
    loader = SimpleNamespace(dataset=dataset, drop_last=False, closed=False)

    def close() -> None:
        loader.closed = True

    loader.close = close
    monkeypatch.setattr(
        data_runtime, "_build_runtime_loader", lambda request: loader
    )
    plan = PN2021K500LoaderPlan(
        center="ningbo", batch_size=64, num_workers=0, pin_memory=False,
        persistent_workers=False, prefetch_factor=2, cache_mode="mmap",
        validate_values="sample", selection_resident=True,
        selection_resident_pin_memory=False, drop_last=False,
        seed_namespace="locked-k500",
        split_config_path=tmp_path / "splits.yaml",
        data_load_config_path=tmp_path / "data_load.yaml",
        seed_config_path=tmp_path / "random_seed.yaml",
    )

    with pytest.raises(ValueError, match="records=500"):
        plan.open_ordered()
    assert loader.closed is True


def test_k500_plan_opens_iterates_and_closes_real_runtime_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not hasattr(data_runtime, "load_split_config")
    cache = _tiny_mmap_cache(tmp_path, "real-plan-open", record_count=500)
    selection = _tiny_selection(cache, list(range(500)))
    monkeypatch.setattr(data_runtime, "load_cache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(
        data_runtime, "load_selection", lambda *args, **kwargs: selection
    )
    split_config = _split_location_fixture(
        tmp_path, cache_dir=cache.identity.cache_dir
    )
    seed_config = tmp_path / "random_seed.yaml"
    seed_config.write_text(
        "schema_version: 1\nrandom_seed: 424242\n", encoding="utf-8"
    )
    plan = PN2021K500LoaderPlan(
        center="ningbo",
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        cache_mode="mmap",
        validate_values="none",
        selection_resident=True,
        selection_resident_pin_memory=False,
        drop_last=False,
        seed_namespace="real-plan-open",
        split_config_path=split_config,
        data_load_config_path=REPO / "configs" / "data" / "data_load.yaml",
        seed_config_path=seed_config,
    )

    loader = plan.open_ordered()
    try:
        assert isinstance(loader.dataset, SelectionResidentECGDataset)
        assert cache._closed is True
        batch = next(iter(loader))
        assert tuple(batch["waveform"].shape) == (32, 8, 2)
        assert batch["cache_index"].tolist() == list(range(32))
        assert loader.runtime_seed_identity.namespace == "real-plan-open"
    finally:
        loader.close()
    assert cache._closed is True


@pytest.mark.parametrize("sampling_rate_hz", (500, True, None))
def test_runtime_location_parser_rejects_noncanonical_sampling_rate(
    tmp_path: Path, sampling_rate_hz: object
) -> None:
    split_config = _split_location_fixture(
        tmp_path,
        cache_dir=tmp_path / "cache",
        sampling_rate_hz=sampling_rate_hz,
    )

    with pytest.raises(ValueError, match="sampling_rate_hz must be 100"):
        data_runtime._resolve_runtime_locations(
            dataset="pn2021",
            split_config_path=split_config,
            corruption_cache_config_path=tmp_path / "unused.yaml",
        )


def test_sequential_evaluation_session_reuses_two_mmaps_and_selections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean_cache = _tiny_mmap_cache(tmp_path, "clean-eval", record_count=8)
    corrupted_cache = _tiny_mmap_cache(
        tmp_path,
        "corrupted-eval",
        dataset="pn2021c",
        record_count=8,
        source_manifest_sha256=clean_cache.identity.manifest_sha256,
    )
    clean_selection = _tiny_selection(
        clean_cache, [0, 2, 5], partition="evaluation_all_zero_kept"
    )
    corrupted_selection = _tiny_selection(
        corrupted_cache, [0, 2, 5], partition="evaluation_all_zero_kept"
    )
    caches = {cache.identity.cache_dir: cache for cache in (clean_cache, corrupted_cache)}
    selections = {"pn2021": clean_selection, "pn2021c": corrupted_selection}
    cache_open_calls: list[Path] = []

    def open_cache(cache_dir: str | Path, **kwargs: Any) -> ECGCache:
        del kwargs
        path = Path(cache_dir).resolve()
        cache_open_calls.append(path)
        return caches[path]

    monkeypatch.setattr(data_runtime, "load_cache", open_cache)
    selection_calls: list[str] = []

    def open_selection(cache: ECGCache, *args: Any, **kwargs: Any) -> ECGSelection:
        del args, kwargs
        selection_calls.append(cache.dataset)
        return selections[cache.dataset]

    monkeypatch.setattr(data_runtime, "load_selection", open_selection)
    mappings = [cache.signals._mmap for cache in caches.values()]
    split_config = _split_location_fixture(
        tmp_path, cache_dir=clean_cache.identity.cache_dir
    )
    corruption_config = tmp_path / "corruption.yaml"
    corruption_config.write_text(
        yaml.safe_dump({
            "schema_version": 1, "dataset": "pn2021c",
            "output": {"cache_dir": str(corrupted_cache.identity.cache_dir)},
        }),
        encoding="utf-8",
    )
    seed_config = tmp_path / "seed.yaml"
    seed_config.write_text(
        "schema_version: 1\nrandom_seed: 424242\n", encoding="utf-8"
    )
    plan = data_runtime.PN2021EvaluationLoaderPlan(
        clean_partition="pn2021_all_zero_kept_refexcluded",
        corrupted_partition="pn2021c_all_zero_kept_corrupted_refexcluded",
        batch_size=2, num_workers=0, pin_memory=False,
        persistent_workers=False, prefetch_factor=2, cache_mode="mmap",
        validate_values="none", split_config_path=split_config,
        data_load_config_path=REPO / "configs" / "data" / "data_load.yaml",
        corruption_cache_config_path=corruption_config,
        seed_config_path=seed_config,
    )
    views = [item["composition_id"] for item in corrupted_cache.compositions]
    with plan.open_session() as session:
        opened = [session.open_clean("ningbo")]
        opened.extend(session.open_corrupted("ningbo", view) for view in views)
        try:
            batches = [next(iter(loader)) for loader in opened]
        finally:
            for loader in opened:
                loader.close()
        assert tuple(batches[0]["waveform"].shape) == (2, 8, 2)
        assert batches[0]["cache_index"].tolist() == [0, 2]
        assert not torch.equal(batches[1]["waveform"], batches[2]["waveform"])
        assert cache_open_calls == list(caches)
        assert selection_calls == ["pn2021", "pn2021c"]
        assert (session.describe()["cache_open_count"], session.describe()["selection_count"]) == (2, 2)
        assert all(cache._closed is False for cache in caches.values())
    assert session.describe()["closed"] is True
    assert session.describe()["cache_open_count"] == 0
    assert all(cache._closed for cache in caches.values())
    assert all(mapping.closed for mapping in mappings)


def test_evaluation_session_close_is_best_effort_and_idempotent() -> None:
    calls: list[str] = []

    def fail() -> None:
        calls.append("first")
        raise RuntimeError("first close failed")

    def succeed() -> None:
        calls.append("second")

    session = SequentialEvaluationDataSession()
    session._caches = {
        (Path("first"), 100): SimpleNamespace(close=fail),
        (Path("second"), 100): SimpleNamespace(close=succeed),
    }
    with pytest.raises(RuntimeError, match="first close failed"):
        session.close()
    assert calls == ["first", "second"]
    assert session.describe() == {
        "closed": True, "cache_open_count": 0, "selection_count": 0, "caches": []
    }
    session.close()
    assert calls == ["first", "second"]


def _ledger_roots(base: Path, payload: bytes = b"same-bytes") -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for index, name in enumerate(data_ledger.ROOT_NAMES):
        root = base / name
        (root / "nested").mkdir(parents=True)
        (root / "nested" / f"{index}.bin").write_bytes(payload + bytes([index]))
        roots[name] = root
    return roots


def _canonical_jsonl(payloads: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(item, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for item in payloads
    )


def test_data_ledger_loads_only_runtime_owned_root_fields(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    splits_path = config_root / "data" / "splits.yaml"
    cache_path = config_root / "augmentation" / "cache.yaml"
    splits_path.parent.mkdir(parents=True)
    cache_path.parent.mkdir(parents=True)
    splits_path.write_text(
        yaml.safe_dump(
            {
                "ptbxl": {"cache_dir": "ptb"},
                "pn2021": {"cache_dir": "pn"},
                "output": {"root_dir": "splits"},
            }
        )
    )
    cache_path.write_text(yaml.safe_dump({"output": {"cache_dir": "pn-c"}}))

    roots = data_ledger.load_roots(splits_path)

    assert roots == {
        "ptbxl_cache": REPO / "ptb",
        "pn2021_cache": REPO / "pn",
        "pn2021c_cache": REPO / "pn-c",
        "split_artifacts": REPO / "splits",
    }


def test_data_ledger_is_deterministic_and_relocatable(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    roots_a = _ledger_roots(tmp_path / "a")
    roots_b = _ledger_roots(tmp_path / "b")

    summary = data_ledger.generate_ledger(roots_a, first)
    data_ledger.generate_ledger(dict(reversed(roots_b.items())), second)

    assert first.read_bytes() == second.read_bytes()
    assert summary["ledger_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert summary["member_count"] == 4
    assert summary["roots"]["ptbxl_cache"] == {
        "member_count": 1,
        "total_size_bytes": 11,
    }
    assert len(first.read_text().splitlines()) == 5
    assert str(tmp_path) not in first.read_text()
    assert data_ledger.verify_ledger(first, roots_b, full=True)["verification_mode"] == "full_content"


def test_data_ledger_quick_inventory_rejects_missing_extra_and_size_drift(
    tmp_path: Path,
) -> None:
    roots = _ledger_roots(tmp_path / "roots")
    ledger = tmp_path / "ledger.jsonl"
    data_ledger.generate_ledger(roots, ledger)
    target = roots["ptbxl_cache"] / "nested" / "0.bin"

    target.unlink()
    with pytest.raises(ValueError, match="inventory or file size"):
        data_ledger.verify_ledger(ledger, roots)
    target.write_bytes(b"same-bytes\x00")
    (roots["ptbxl_cache"] / "extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory or file size"):
        data_ledger.verify_ledger(ledger, roots)
    (roots["ptbxl_cache"] / "extra.bin").unlink()
    target.write_bytes(b"different-size")
    with pytest.raises(ValueError, match="inventory or file size"):
        data_ledger.verify_ledger(ledger, roots)


def test_data_ledger_quick_is_explicitly_not_a_content_check(tmp_path: Path) -> None:
    roots = _ledger_roots(tmp_path / "roots")
    ledger = tmp_path / "ledger.jsonl"
    data_ledger.generate_ledger(roots, ledger)
    target = roots["pn2021_cache"] / "nested" / "1.bin"
    original = target.read_bytes()
    target.write_bytes(b"x" * len(original))

    assert data_ledger.verify_ledger(ledger, roots)["verification_mode"] == "inventory_only"
    with pytest.raises(ValueError, match="content SHA-256"):
        data_ledger.verify_ledger(ledger, roots, full=True)


def test_data_ledger_rejects_symlinks_and_output_inside_a_root(tmp_path: Path) -> None:
    roots = _ledger_roots(tmp_path / "roots")
    (roots["ptbxl_cache"] / "link").symlink_to(
        roots["ptbxl_cache"] / "nested" / "0.bin"
    )
    with pytest.raises(ValueError, match="symlink is forbidden"):
        data_ledger.generate_ledger(roots, tmp_path / "ledger.jsonl")
    (roots["ptbxl_cache"] / "link").unlink()
    with pytest.raises(ValueError, match="outside data root"):
        data_ledger.generate_ledger(roots, roots["pn2021_cache"] / "ledger.jsonl")

    existing = tmp_path / "existing.jsonl"
    existing.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        data_ledger.generate_ledger(roots, existing)
    assert existing.read_bytes() == b"preserve"

    real_root = roots["split_artifacts"]
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real_root, target_is_directory=True)
    roots["split_artifacts"] = root_link
    with pytest.raises(ValueError, match="real directory"):
        data_ledger.generate_ledger(roots, tmp_path / "root-link-ledger.jsonl")


def test_data_ledger_parser_fails_closed_on_hash_escape_and_duplicate(tmp_path: Path) -> None:
    roots = _ledger_roots(tmp_path / "roots")
    ledger = tmp_path / "ledger.jsonl"
    data_ledger.generate_ledger(roots, ledger)
    with pytest.raises(ValueError, match="ledger SHA-256 mismatch"):
        data_ledger.verify_ledger(ledger, roots, expected_sha256="0" * 64)

    ledger.write_text("{")
    with pytest.raises(ValueError, match="canonical LF|valid UTF-8 JSONL"):
        data_ledger.verify_ledger(ledger, roots)
    ledger.unlink()
    data_ledger.generate_ledger(roots, ledger)
    lines = [json.loads(line) for line in ledger.read_text().splitlines()]
    lines[1]["path"] = "../escape"
    ledger.write_bytes(_canonical_jsonl(lines))
    with pytest.raises(ValueError, match="escaping"):
        data_ledger.verify_ledger(ledger, roots)

    ledger.unlink()
    data_ledger.generate_ledger(roots, ledger)
    lines = [json.loads(line) for line in ledger.read_text().splitlines()]
    lines.insert(2, dict(lines[1]))
    lines[0]["member_count"] += 1
    lines[0]["total_size_bytes"] += lines[1]["size_bytes"]
    ledger.write_bytes(_canonical_jsonl(lines))
    with pytest.raises(ValueError, match="duplicated or unsorted"):
        data_ledger.verify_ledger(ledger, roots)


def test_data_ledger_discards_generation_on_whole_scan_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _ledger_roots(tmp_path / "roots")
    output = tmp_path / "ledger.jsonl"
    original_hash_file = data_ledger._hash_file
    changed = False

    def mutate_other_member(path: Path, expected: tuple[object, ...]) -> str:
        nonlocal changed
        digest = original_hash_file(path, expected)
        if not changed:
            path.write_bytes(path.read_bytes() + b"drift")
            changed = True
        return digest

    monkeypatch.setattr(data_ledger, "_hash_file", mutate_other_member)
    with pytest.raises(ValueError, match="changed"):
        data_ledger.generate_ledger(roots, output)
    assert not output.exists()
