"""CPU-only tests for PN2021 metadata and K-shot helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecg_adv_gen.data import (
    KShotArtifactPaths,
    assert_not_forbidden_center,
    center_offset_seed,
    count_center_header_files,
    kshot_ref_meta_path,
    load_include_record_ids_by_center,
    load_include_record_ids_from_meta,
    load_kshot_ref_record_ids,
    load_ref_record_ids_by_center,
    load_selected_record_ids_from_meta,
    parse_header_snomeds,
    proportional_stratified_indices,
    record_id_from_path,
    scan_pn2021_center_records,
    select_kshot_indices_from_ref_ids,
    select_kshot_indices_from_ref_root,
    validate_kshot_artifacts,
)


def test_parse_header_snomeds_matches_legacy_dx_behavior(tmp_path: Path):
    hea = tmp_path / "A0001.hea"
    hea.write_text("12 500 5000\n#Dx: 164889003, 426783006\n", encoding="utf-8")
    bad = tmp_path / "bad.hea"
    bad.write_text("#Dx: 164889003, not-a-code\n", encoding="utf-8")
    missing = tmp_path / "missing.hea"
    missing.write_text("#Age: 60\n", encoding="utf-8")

    assert parse_header_snomeds(hea) == [164889003, 426783006]
    assert parse_header_snomeds(bad) == []
    assert parse_header_snomeds(missing) == []


def test_scan_pn2021_center_records_uses_recursive_headers_and_basename_ids(tmp_path: Path):
    center = tmp_path / "ningbo"
    (center / "g1").mkdir(parents=True)
    (center / "g2").mkdir()
    (center / "g1" / "N0001.hea").write_text("#Dx: 1\n", encoding="utf-8")
    (center / "g1" / "N0001.mat").write_text("ignored\n", encoding="utf-8")
    (center / "g2" / "N0002.hea").write_text("#Dx: 2,3\n", encoding="utf-8")

    records = scan_pn2021_center_records(center)
    records_by_id = {r.record_id: r for r in records}

    assert set(records_by_id) == {"N0001", "N0002"}
    assert str(records_by_id["N0001"].record_path.relative_to(center)) == "g1/N0001"
    assert str(records_by_id["N0002"].record_path.relative_to(center)) == "g2/N0002"
    assert list(records_by_id["N0001"].snomeds) == [1]
    assert list(records_by_id["N0002"].snomeds) == [2, 3]
    assert record_id_from_path(center / "g1" / "N0001.hea") == "N0001"
    assert record_id_from_path(center / "g1" / "N0001") == "N0001"


def test_count_center_header_files_is_bounded_not_full_recursive(tmp_path: Path):
    center = tmp_path / "center"
    group = center / "g1"
    deep = group / "deep"
    deep.mkdir(parents=True)
    (center / "top.hea").write_text("#Dx: 1\n", encoding="utf-8")
    (group / "one_level.hea").write_text("#Dx: 2\n", encoding="utf-8")
    (deep / "too_deep.hea").write_text("#Dx: 3\n", encoding="utf-8")

    assert count_center_header_files(center) == 2
    assert count_center_header_files(tmp_path / "missing") is None


def test_forbidden_center_guard_uses_ptbxl_leak_shards():
    assert_not_forbidden_center("ningbo")
    with pytest.raises(ValueError, match="leak PTB-XL"):
        assert_not_forbidden_center("ptb-xl")
    with pytest.raises(ValueError, match="leak PTB-XL"):
        assert_not_forbidden_center("PTBXL")


def test_ref_and_include_meta_loaders_group_by_center(tmp_path: Path):
    ref_a = tmp_path / "a.ref_meta.json"
    ref_b = tmp_path / "b.ref_meta.json"
    include = tmp_path / "include.ref_meta.json"
    ref_a.write_text(json.dumps({"center": "ningbo", "ref_record_ids": ["N1", "N2"]}), encoding="utf-8")
    ref_b.write_text(json.dumps({"center": "ningbo", "ref_record_ids": ["N2", "N3"]}), encoding="utf-8")
    include.write_text(json.dumps({"center": "georgia", "heldout_record_ids": ["G1"]}), encoding="utf-8")

    assert load_ref_record_ids_by_center([ref_a, ref_b]) == {"ningbo": {"N1", "N2", "N3"}}
    center, ids, key = load_include_record_ids_from_meta(include)
    assert (center, ids, key) == ("georgia", {"G1"}, "heldout_record_ids")
    assert load_include_record_ids_by_center([include]) == {"georgia": {"G1"}}


def test_selected_record_ids_loader_supports_legacy_meta_variants(tmp_path: Path):
    direct = tmp_path / "direct.ref_meta.json"
    direct.write_text(
        json.dumps({"center": "ningbo", "record_ids": ["N1"], "selected_ref_record_ids": ["N2"]}),
        encoding="utf-8",
    )
    assert load_selected_record_ids_from_meta(direct, "ningbo") == {"N1", "N2"}

    items = tmp_path / "items.ref_meta.json"
    items.write_text(
        json.dumps({"items": [{"record_id": "N2"}, {"record": "N3"}, {"bad": "ignored"}]}),
        encoding="utf-8",
    )
    assert load_selected_record_ids_from_meta(items, "ningbo") == {"N2", "N3"}

    rows = tmp_path / "rows.ref_meta.json"
    rows.write_text(
        json.dumps(
            [
                {"center": "ningbo", "record_id": "N4"},
                {"center": "georgia", "record_id": "G1"},
                {"record": "N5"},
            ]
        ),
        encoding="utf-8",
    )
    assert load_selected_record_ids_from_meta(rows, "ningbo") == {"N4", "N5"}

    empty = tmp_path / "empty.ref_meta.json"
    empty.write_text(json.dumps({"items": [{"bad": "ignored"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="could not parse selected ref ids"):
        load_selected_record_ids_from_meta(empty, "ningbo")


def test_kshot_ref_meta_loader_uses_exact_k_or_source_k_fallback(tmp_path: Path):
    source = kshot_ref_meta_path(tmp_path, "ningbo", 500, 20260531)
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps({"center": "ningbo", "ref_record_ids": ["N1", "N2", "N3"]}),
        encoding="utf-8",
    )
    exact = kshot_ref_meta_path(tmp_path, "ningbo", 2, 20260531)
    exact.parent.mkdir(parents=True)
    exact.write_text(
        json.dumps({"center": "ningbo", "ref_record_ids": ["N1", "N3"]}),
        encoding="utf-8",
    )

    assert load_kshot_ref_record_ids(tmp_path, "ningbo", k=2, seed=20260531, source_k=500) == ["N1", "N3"]
    assert load_kshot_ref_record_ids(tmp_path, "ningbo", k=3, seed=20260531, source_k=500) == [
        "N1",
        "N2",
        "N3",
    ]


def test_select_kshot_indices_from_ref_ids_matches_ecgfounder_legacy_policy():
    labels = np.asarray(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=np.float32,
    )
    centers = np.asarray(["ningbo"] * 6 + ["georgia"])
    record_ids = np.asarray(["N0", "N1", "N2", "N3", "N4", "N5", "G0"])

    assert center_offset_seed("ningbo") == 2218
    idx, ids = select_kshot_indices_from_ref_ids(
        labels,
        centers,
        record_ids,
        center="ningbo",
        ref_record_ids=["N0", "N1", "N2", "N3", "N4", "N5"],
        k=3,
        seed=7,
        n_classes=3,
    )
    assert idx.tolist() == [1, 3, 5]
    assert ids == ["N1", "N3", "N5"]

    exact_idx, exact_ids = select_kshot_indices_from_ref_ids(
        labels,
        centers,
        record_ids,
        center="ningbo",
        ref_record_ids=["N1", "N3", "N4"],
        k=3,
        seed=7,
        n_classes=3,
    )
    assert exact_idx.tolist() == [1, 3, 4]
    assert exact_ids == ["N1", "N3", "N4"]


def test_select_kshot_indices_from_ref_root_uses_source_pool_when_exact_missing(tmp_path: Path):
    meta = kshot_ref_meta_path(tmp_path, "ningbo", 6, 11)
    meta.parent.mkdir(parents=True)
    meta.write_text(
        json.dumps({"center": "ningbo", "ref_record_ids": ["N0", "N1", "N2", "N3", "N4", "N5"]}),
        encoding="utf-8",
    )
    labels = np.asarray(
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
        dtype=np.float32,
    )
    centers = np.asarray(["ningbo"] * 6)
    record_ids = np.asarray(["N0", "N1", "N2", "N3", "N4", "N5"])

    idx, ids = select_kshot_indices_from_ref_root(
        labels,
        centers,
        record_ids,
        center="ningbo",
        ref_root=tmp_path,
        k=3,
        source_k=6,
        subset_seed=11,
        seed=7,
        n_classes=3,
    )

    assert idx.tolist() == [1, 3, 5]
    assert ids == ["N1", "N3", "N5"]


def test_proportional_stratified_indices_matches_frozen_legacy_selection():
    labels = np.asarray(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    assert proportional_stratified_indices(labels, 0, seed=7).tolist() == []
    assert proportional_stratified_indices(labels, 3, seed=7).tolist() == [0, 2, 5]
    assert proportional_stratified_indices(labels, 5, seed=7).tolist() == [0, 2, 6, 5, 1]
    assert proportional_stratified_indices(labels, 10, seed=7).tolist() == [4, 0, 5, 2, 3, 6, 1]


def test_validate_kshot_artifacts_checks_required_files_and_counts(tmp_path: Path):
    base = tmp_path / "ningbo_real_k2_seed20260531"
    paths = KShotArtifactPaths.from_base(base)
    labels = np.asarray([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]], dtype=np.float32)
    np.savez_compressed(paths.signals, signals=np.zeros((2, 1000, 12), dtype=np.float32), labels=labels, record_ids=["N1", "N2"])
    np.savez_compressed(paths.latents, latents=np.zeros((2, 4, 128), dtype=np.float32), labels=labels, record_ids=["N1", "N2"])
    paths.meta.write_text(
        json.dumps({"center": "ningbo", "ref_record_ids": ["N1", "N2"]}),
        encoding="utf-8",
    )

    summary = validate_kshot_artifacts(paths, expected_k=2)

    assert summary["center"] == "ningbo"
    assert summary["k"] == 2
    assert summary["n_ref_record_ids"] == 2
    with pytest.raises(ValueError, match="expected K=3"):
        validate_kshot_artifacts(paths, expected_k=3)
