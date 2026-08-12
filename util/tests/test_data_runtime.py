"""Small CPU contracts for the retained runtime data boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data_preprocess import data_ledger
from data_preprocess.data_runtime import (
    ECGSelection,
    MMapPrefetchConfig,
    assert_disjoint_selections,
    load_data_load_config,
    per_sample_global_zscore,
    prepare_model_input,
)
from data_preprocess.load_cache import EXPECTED_CLASS_ORDER


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
