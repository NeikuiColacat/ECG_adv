from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

import util.evaluation.direct_baseline_selection as selection
from models.contracts import CLASS_ORDER


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "train" / "PN2021_direct_tune.yaml"
CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
COMPOSITION_IDS = tuple(f"composition_{index:02d}" for index in range(20))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_lines(values: Sequence[str], *, sort: bool) -> str:
    normalized = [str(value) for value in values]
    if sort:
        normalized.sort()
    return hashlib.sha256(
        "".join(f"{value}\n" for value in normalized).encode("utf-8")
    ).hexdigest()


def _hash_label_set(hash_ids: Sequence[str], targets: np.ndarray) -> str:
    rows = sorted(
        (str(hash_id), "".join(str(int(value)) for value in target))
        for hash_id, target in zip(hash_ids, targets, strict=True)
    )
    return hashlib.sha256(
        "".join(f"{hash_id}|{label}\n" for hash_id, label in rows).encode("utf-8")
    ).hexdigest()


def _targets(*, missing_class: int | None = None) -> np.ndarray:
    targets = np.zeros((100, 5), dtype=np.uint8)
    targets[np.arange(100), np.arange(100) % 5] = 1
    if missing_class is not None:
        targets[:, missing_class] = 0
    return targets


def _logits(targets: np.ndarray, kind: str) -> np.ndarray:
    if kind == "perfect":
        values = np.where(targets == 1, 1001.0, 1000.0)
    elif kind == "reversed":
        values = np.where(targets == 1, 1000.0, 1001.0)
    elif kind == "poor":
        values = np.empty(targets.shape, dtype=np.float32)
        for class_index in range(targets.shape[1]):
            positive = np.flatnonzero(targets[:, class_index] == 1)
            negative = np.flatnonzero(targets[:, class_index] == 0)
            values[positive, class_index] = 1000.0 + np.arange(positive.size)
            values[negative, class_index] = 1100.0 + np.arange(negative.size)
    elif kind == "constant":
        values = np.full(targets.shape, 1000.0)
    else:
        raise ValueError(f"unknown logits kind: {kind}")
    return values.astype(np.float32)


def _comparison_identity(*, epochs: int = 2) -> dict[str, object]:
    return {
        "protocol_id": "pn2021_direct_family_balanced_tuning",
        "method_id": "direct_depth23_fixed20",
        "model_family": "efficientnet1dv2",
        "model_spec": {"name": "efficientnet1dv2"},
        "source_checkpoint_identity": {
            "path": "/external/source/best.pt",
            "sha256": "a" * 64,
        },
        "tuning_config_sha256": _sha256(CONFIG),
        "online_training_config_sha256": "b" * 64,
        "method_profile_sha256": "c" * 64,
        "resolved_training_parameters": {
            "epochs": epochs,
            "learning_rate": 5e-5,
            "weight_decay": 1e-4,
        },
        "pos_weight": None,
        "comparison_group": "pn2021_direct_family_balanced_tuning",
        "replicate_id": 0,
    }


def _write_artifact(
    root: Path,
    *,
    center: str,
    epoch: int,
    clean_kind: str,
    corrupted_kinds: str | Sequence[str],
    comparison_identity: dict[str, object] | None = None,
    missing_class: int | None = None,
) -> Path:
    directory = root / center
    directory.mkdir(parents=True, exist_ok=True)
    hashes = tuple(f"{center}-hash-{index:03d}" for index in range(100))
    targets = _targets(missing_class=missing_class)
    kinds = (
        (corrupted_kinds,) * 20
        if isinstance(corrupted_kinds, str)
        else tuple(corrupted_kinds)
    )
    if len(kinds) != 20:
        raise ValueError("corrupted_kinds must define twenty compositions")
    clean_logits = _logits(targets, clean_kind)
    corrupted_logits = np.stack(
        [_logits(targets, kind) for kind in kinds], axis=0
    ).astype(np.float32, copy=False)
    arrays_path = directory / f"epoch_{epoch:04d}.npz"
    np.savez_compressed(
        arrays_path,
        clean_logits=clean_logits,
        corrupted_logits=corrupted_logits,
        targets=targets,
        hash_ids=np.asarray(hashes, dtype=np.str_),
        composition_ids=np.asarray(COMPOSITION_IDS, dtype=np.str_),
    )
    clean_identity = {
        "dataset": "pn2021",
        "partition": "k500_tune_validation",
        "logical_center": center,
        "record_count": 100,
        "split_id": "pn2021_super5_k500_cpsc_combined_v1",
        "source_manifest_sha256": "d" * 64,
        "mapping_version": "v7_super5_sjr_rgq_review_20260528",
        "mapping_hash": "555ec85d5b51",
    }
    frozen_identity = {
        "corruption_cache_dir": "/external/pn2021c",
        "corruption_cache_version": "pn2021c_depth23_fixed20",
        "corruption_cache_manifest_path": "/external/pn2021c/manifest.json",
        "corruption_cache_manifest_sha256": "e" * 64,
        "corruption_config_identity_hash": "f" * 64,
        "composition_indices": list(range(20)),
        "composition_ids": list(COMPOSITION_IDS),
        "ordered_hash_ids_sha256": _sha256_lines(hashes, sort=False),
        "source_manifest_sha256": "d" * 64,
    }
    sidecar = {
        "schema_version": 1,
        "artifact_type": selection.ARTIFACT_TYPE,
        "artifact_schema": selection.ARTIFACT_SCHEMA,
        "epoch": epoch,
        "center": center,
        "class_order": list(CLASS_ORDER),
        "record_count": 100,
        "composition_count": 20,
        "arrays_file": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "clean_validation_identity": clean_identity,
        "frozen_corruption_identity": frozen_identity,
        "prediction_identity": {
            "ordered_hash_ids_sha256": _sha256_lines(hashes, sort=False),
            "hash_id_set_sha256": _sha256_lines(hashes, sort=True),
            "hash_label_set_sha256": _hash_label_set(hashes, targets),
        },
        "comparison_identity": comparison_identity or _comparison_identity(),
    }
    path = directory / f"epoch_{epoch:04d}.json"
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def _grid(
    root: Path,
    epoch_specs: dict[int, tuple[str, str | Sequence[str]]],
    *,
    missing_class: int | None = None,
) -> list[Path]:
    return [
        _write_artifact(
            root,
            center=center,
            epoch=epoch,
            clean_kind=clean_kind,
            corrupted_kinds=corrupted_kinds,
            missing_class=missing_class,
        )
        for center in CENTERS
        for epoch, (clean_kind, corrupted_kinds) in epoch_specs.items()
    ]


def _patch_clean_reference(monkeypatch, value: float) -> None:
    monkeypatch.setattr(
        selection,
        "_clean_reference",
        lambda config, model_family: {
            "model_family": model_family,
            "macro_auprc": value,
            "artifact_path": "/locked/clean/selection.json",
            "artifact_sha256": "9" * 64,
            "registry_path": str(config.clean_baseline_registry),
            "registry_sha256": "8" * 64,
        },
    )


def test_selector_pools_four_centers_per_composition_and_averages_twenty(
    tmp_path: Path, monkeypatch
):
    _patch_clean_reference(monkeypatch, 0.0)
    epoch_two_corruptions = ("reversed",) + ("perfect",) * 19
    paths = _grid(
        tmp_path / "artifacts",
        {
            1: ("reversed", "reversed"),
            2: ("perfect", epoch_two_corruptions),
        },
    )
    output = tmp_path / "selection.json"

    result = selection.select_direct_baseline_epoch(
        paths,
        config_path=CONFIG,
        output_path=output,
    )

    assert result["selected_epoch"] == 2
    assert result["artifact_type"] == "direct_k500_pooled_epoch_selection"
    assert result["record_count"] == 400
    assert result["composition_count"] == 20
    selected = result["per_epoch"][1]
    composition_metrics = selected["robust"]["composition_metrics"]
    assert selected["clean"]["sample_count"] == 400
    assert len(composition_metrics) == 20
    assert all(item["metrics"]["sample_count"] == 400 for item in composition_metrics)
    expected_robust = np.mean(
        [item["metrics"]["macro_auprc"] for item in composition_metrics]
    )
    assert selected["robust"]["macro_auprc"] == pytest.approx(expected_robust)
    assert selected["score"] == pytest.approx(
        0.5 * selected["clean"]["macro_auprc"] + 0.5 * expected_robust
    )
    assert result["selection_rule"]["input_type"] == "raw_logits"
    assert json.loads(output.read_text(encoding="utf-8"))["selected_epoch"] == 2


def test_same_backbone_clean_floor_excludes_higher_scoring_epoch(
    tmp_path: Path, monkeypatch
):
    _patch_clean_reference(monkeypatch, 1.0)
    paths = _grid(
        tmp_path / "artifacts",
        {
            1: ("perfect", "poor"),
            2: ("constant", "perfect"),
        },
    )

    result = selection.select_direct_baseline_epoch(paths, config_path=CONFIG)

    first, second = result["per_epoch"]
    assert second["score"] > first["score"]
    assert first["eligible"] is True
    assert second["eligible"] is False
    assert result["clean_floor"] == pytest.approx(0.99)
    assert result["selected_epoch"] == 1


def test_clean_floor_failure_produces_no_selection(tmp_path: Path, monkeypatch):
    _patch_clean_reference(monkeypatch, 1.0)
    paths = _grid(
        tmp_path / "artifacts",
        {
            1: ("constant", "perfect"),
            2: ("reversed", "perfect"),
        },
    )

    result = selection.select_direct_baseline_epoch(paths, config_path=CONFIG)

    assert result["status"] == "failed_clean_floor"
    assert result["selected_epoch"] is None
    assert result["selected_score"] is None


def test_exact_score_tie_selects_earliest_epoch(tmp_path: Path, monkeypatch):
    _patch_clean_reference(monkeypatch, 0.0)
    paths = _grid(
        tmp_path / "artifacts",
        {
            1: ("perfect", "perfect"),
            2: ("perfect", "perfect"),
        },
    )

    result = selection.select_direct_baseline_epoch(paths, config_path=CONFIG)

    assert result["per_epoch"][0]["score"] == result["per_epoch"][1]["score"]
    assert result["selected_epoch"] == 1


def test_selector_rejects_comparison_identity_drift(tmp_path: Path, monkeypatch):
    _patch_clean_reference(monkeypatch, 0.0)
    paths = _grid(
        tmp_path / "artifacts",
        {1: ("reversed", "reversed"), 2: ("perfect", "perfect")},
    )
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["comparison_identity"]["resolved_training_parameters"][
        "learning_rate"
    ] = 1e-3
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model/source/hyperparameter identity"):
        selection.select_direct_baseline_epoch(paths, config_path=CONFIG)


def test_selector_keeps_strict_five_class_metrics(tmp_path: Path, monkeypatch):
    _patch_clean_reference(monkeypatch, 0.0)
    paths = _grid(
        tmp_path / "artifacts",
        {1: ("perfect", "perfect"), 2: ("perfect", "perfect")},
        missing_class=2,
    )

    with pytest.raises(ValueError, match="class MI requires both"):
        selection.select_direct_baseline_epoch(paths, config_path=CONFIG)


def test_direct_selection_config_locks_one_overwritten_protocol():
    config = selection.load_direct_selection_config(CONFIG)

    assert config.centers == CENTERS
    assert config.protocol_id == "pn2021_direct_family_balanced_tuning"
    assert config.method_id == "direct_depth23_fixed20"
    assert config.expected_records_per_center == 100
    assert config.expected_pooled_records == 400
    assert config.expected_compositions == 20
    assert config.clean_weight == config.robust_weight == 0.5
    assert config.maximum_clean_drop == 0.01
    assert config.tie_break == "earliest_epoch_on_exact_tie"
