"""Small CPU contracts for the retained runtime data boundary."""

from __future__ import annotations

import hashlib
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

from data_preprocess import data_ledger, data_runtime
from data_preprocess.data_runtime import (
    ECGSelection,
    MMapPrefetchConfig,
    RuntimeECGDataset,
    SelectionResidentECGDataset,
    SequentialEvaluationDataSession,
    assert_disjoint_selections,
    build_dataloader,
    load_data_load_config,
    per_sample_global_zscore,
    prepare_model_input,
)
from data_preprocess.load_cache import (
    EXPECTED_CLASS_ORDER,
    PN2021_MAPPING_HASH,
    PN2021_MAPPING_VERSION,
    ECGCache,
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
        storage_mode="mmap",
        estimated_resident_bytes=signals_path.stat().st_size,
        available_memory_at_open_bytes=10**9,
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


def test_canonical_data_load_config_is_the_single_runtime_profile() -> None:
    config_path = REPO / "configs" / "data" / "data_load.yaml"
    config = load_data_load_config(config_path)

    assert config.path == config_path.resolve()
    assert config.cache_mode == "mmap"
    assert config.mmap_access_order == "cache_index"
    assert config.mmap_batch_read is True
    assert config.mmap_prefetch.enabled is True
    assert config.selection_resident is False
    assert config.prepare_for_model is True
    assert config.output_layout == "channel_time"


def test_data_load_config_resolves_ledger_without_opening_it(tmp_path: Path) -> None:
    path = _data_load_fixture(
        tmp_path,
        {"path": "data/missing-ledger.jsonl", "sha256": "a" * 64},
    )

    config = load_data_load_config(path)

    assert config.content_ledger.path == (
        tmp_path / "configs" / "data" / "missing-ledger.jsonl"
    ).resolve()
    assert config.content_ledger.sha256 == "a" * 64
    assert not config.content_ledger.path.exists()


@pytest.mark.parametrize(
    ("descriptor", "message"),
    (
        ({"path": "../escape.jsonl", "sha256": "a" * 64}, "escapes config bundle"),
        ({"path": "/tmp/absolute.jsonl", "sha256": "a" * 64}, "must be relative"),
        ({"path": "data/x.jsonl", "sha256": "A" * 64}, "sha256 is invalid"),
        ({"path": "data/x.jsonl", "sha256": "a" * 64, "extra": 1}, "keys mismatch"),
    ),
)
def test_data_load_config_rejects_unsafe_ledger_descriptor(
    tmp_path: Path, descriptor: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_data_load_config(_data_load_fixture(tmp_path, descriptor))


def test_runtime_transform_applies_augmentation_then_sanitize_zscore_and_layout() -> None:
    source = torch.tensor(
        [[1.0, float("nan")], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=torch.float32,
    )
    original = source.clone()
    saw_nonfinite: list[bool] = []

    def augmentation(value: torch.Tensor) -> torch.Tensor:
        saw_nonfinite.append(not bool(torch.isfinite(value).all()))
        return value + 1.0

    transformed = prepare_model_input(source, augmentation=augmentation)

    assert saw_nonfinite == [True]
    assert transformed.shape == (2, 4)
    assert transformed.is_contiguous()
    assert bool(torch.isfinite(transformed).all())
    torch.testing.assert_close(
        transformed.mean(),
        torch.tensor(0.0),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        transformed.std(correction=0),
        torch.tensor(1.0),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(source, original, equal_nan=True)


def test_runtime_transform_rejects_shape_drift_and_unsanitized_nonfinite() -> None:
    source = torch.ones((8, 12), dtype=torch.float32)

    with pytest.raises(ValueError, match="augmentation changed ECG shape"):
        prepare_model_input(source, augmentation=lambda value: value[:-1])

    source[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite values remain"):
        prepare_model_input(source, sanitize=False)


def test_flat_samples_zscore_to_finite_zeros() -> None:
    flat = torch.full((2, 1000, 12), 3.0)
    normalized = per_sample_global_zscore(flat)

    assert bool(torch.isfinite(normalized).all())
    assert torch.count_nonzero(normalized).item() == 0


def test_selection_overlap_is_a_fail_closed_leakage_error() -> None:
    train = _selection("k500_tune_train", ("a", "b"))
    validation = _selection("k500_tune_validation", ("c", "d"))
    assert_disjoint_selections(train, validation)

    leaked = _selection("evaluation_drop_all_zero", ("d", "e"))
    with pytest.raises(ValueError, match="selection leakage"):
        assert_disjoint_selections(validation, leaked)


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
    resident_loader, resident_seed = build_dataloader(
        resident,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        seed_namespace=namespace,
        seed_config_path=seed_config,
    )
    mmap_loader, mmap_seed = build_dataloader(
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
    session = SequentialEvaluationDataSession()
    acquire = {
        "split_dir": tmp_path / "splits",
        "partition": "evaluation_all_zero_kept",
        "logical_center": "ningbo",
        "sampling_rate_hz": 100,
        "validate_values": "none",
    }
    try:
        clean_pair = session._acquire(
            cache_dir=clean_cache.identity.cache_dir,
            **acquire,
        )
        corrupted_pairs = [
            session._acquire(
                cache_dir=corrupted_cache.identity.cache_dir,
                **acquire,
            )
            for _view in corrupted_cache.compositions
        ]
        assert clean_pair == (clean_cache, clean_selection)
        assert corrupted_pairs[0][0] is corrupted_pairs[1][0]
        assert corrupted_pairs[0][1] is corrupted_pairs[1][1]
        views = [item["composition_id"] for item in corrupted_cache.compositions]
        assert not np.array_equal(
            corrupted_cache.get_record(0, view=views[0]).signal,
            corrupted_cache.get_record(0, view=views[1]).signal,
        )
        for cache, selection, view in (
            (clean_cache, clean_selection, None),
            *((corrupted_cache, corrupted_selection, view) for view in views),
        ):
            dataset = RuntimeECGDataset(
                cache_dir=cache.identity.cache_dir, split_dir=tmp_path / "splits",
                partition=selection.partition, logical_center="ningbo", cache_mode="mmap",
                view=view, validate_values="none", shared_cache=cache,
                shared_selection=selection,
            )
            loader, _ = build_dataloader(dataset, batch_size=2, shuffle=False)
            next(iter(loader))
            loader.close()
            assert cache._closed is False
        assert cache_open_calls == list(caches)
        assert selection_calls == ["pn2021", "pn2021c"]
        assert (session.describe()["cache_open_count"], session.describe()["selection_count"]) == (2, 2)
    finally:
        session.close()
    assert session.describe()["closed"] is True
    assert session.describe()["cache_open_count"] == 0
    assert all(cache._closed for cache in caches.values())
    assert all(mapping.closed for mapping in mappings)


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
