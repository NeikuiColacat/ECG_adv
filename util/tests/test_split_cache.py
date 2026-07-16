from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_preprocess.load_cache import EXPECTED_CLASS_ORDER, EXPECTED_LEADS
from data_preprocess.split_cache import build_cache_splits


def _hash_id(dataset: str, index: int) -> str:
    return hashlib.sha256(f"{dataset}:{index}".encode("utf-8")).hexdigest()


def _write_cache(
    root: Path,
    *,
    dataset: str,
    records: pd.DataFrame,
    labels: np.ndarray,
) -> Path:
    root.mkdir()
    count = len(records)
    signals = np.zeros((count, 1000, 12), dtype=np.float32)
    record_ids = records["record_id"].astype(str).to_numpy(dtype="<U32")
    hash_ids = records["hash_id"].astype(str).to_numpy(dtype="<U64")
    np.save(root / "signals.npy", signals, allow_pickle=False)
    np.save(root / "labels_super5.npy", labels.astype(np.uint8), allow_pickle=False)
    np.save(root / "record_ids.npy", record_ids, allow_pickle=False)
    np.save(root / "hash_ids.npy", hash_ids, allow_pickle=False)
    records.to_parquet(root / "records.parquet", index=False)

    label_manifest = {
        "file": "labels_super5.npy",
        "dtype": "uint8",
        "class_order": list(EXPECTED_CLASS_ORDER),
    }
    if dataset == "pn2021":
        label_manifest.update(
            {
                "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                "mapping_hash": "555ec85d5b51",
                "all_zero_policy": "kept",
            }
        )
    manifest = {
        "schema_version": 3,
        "dataset": dataset,
        "dataset_version": "test-v1",
        "record_count": count,
        "waveform": {
            "file": "signals.npy",
            "shape": [count, 1000, 12],
            "dtype": "float32",
            "sampling_rate_hz": 100,
            "duration_seconds": 10,
            "layout": "time_channel",
            "lead_order": list(EXPECTED_LEADS),
            "physical_unit": "mV",
            "normalization": "none",
        },
        "labels": label_manifest,
        "files": {
            "record_ids": "record_ids.npy",
            "hash_ids": "hash_ids.npy",
            "metadata": "records.parquet",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def _write_ptbxl_cache(root: Path) -> Path:
    count = 10
    records = pd.DataFrame(
        {
            "cache_index": np.arange(count),
            "record_id": [f"ptb-{index}" for index in range(count)],
            "record_key": [f"ptbxl/ptb-{index}" for index in range(count)],
            "hash_id": [_hash_id("ptbxl", index) for index in range(count)],
            "quality_status": ["clean"] * count,
            "patient_id": [f"patient-{index}" for index in range(count)],
            "strat_fold": np.arange(1, 11),
        }
    )
    labels = np.zeros((count, 5), dtype=np.uint8)
    labels[:, 0] = 1
    labels[0] = 0
    return _write_cache(root, dataset="ptbxl", records=records, labels=labels)


def _write_pn2021_cache(root: Path) -> Path:
    centers = (
        ["ningbo"] * 4
        + ["chapman_shaoxing"] * 4
        + ["cpsc_2018"] * 4
        + ["cpsc_2018_extra"] * 4
        + ["georgia"] * 4
        + ["ptb"]
        + ["st_petersburg_incart"]
    )
    count = len(centers)
    records = pd.DataFrame(
        {
            "cache_index": np.arange(count),
            "record_id": [f"pn-{index}" for index in range(count)],
            "record_key": [
                f"{center}/pn-{index}" for index, center in enumerate(centers)
            ],
            "hash_id": [_hash_id("pn2021", index) for index in range(count)],
            "quality_status": ["repaired" if index == 0 else "clean" for index in range(count)],
            "center": centers,
        }
    )
    labels = np.zeros((count, 5), dtype=np.uint8)
    for start in range(0, 20, 4):
        labels[start : start + 3, start // 4] = 1
    labels[20:, 3] = 1
    return _write_cache(root, dataset="pn2021", records=records, labels=labels)


def test_split_builder_uses_official_ptbxl_folds_and_combines_cpsc_sources(
    tmp_path: Path,
) -> None:
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
        "random_seed": {
            "file": str(seed_path),
            "namespace": "test_split_v1",
        },
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

    built_root = build_cache_splits(config_path)
    assert built_root == output_root.resolve()

    ptb_manifest = json.loads(
        (output_root / "ptbxl" / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert ptb_manifest["splits"]["train"]["record_count"] == 7
    assert ptb_manifest["splits"]["validation"]["record_count"] == 1
    assert ptb_manifest["splits"]["test"]["record_count"] == 1
    assert ptb_manifest["patient_overlap_counts"] == {
        "train__validation": 0,
        "train__test": 0,
        "validation__test": 0,
    }

    pn_manifest = json.loads(
        (output_root / "pn2021" / "split_manifest.json").read_text(encoding="utf-8")
    )
    cpsc = pn_manifest["logical_centers"]["cpsc_2018"]
    assert cpsc["source_centers"] == ["cpsc_2018", "cpsc_2018_extra"]
    assert cpsc["candidate_nonzero_count"] == 6
    assert cpsc["k500_count"] == 2
    assert cpsc["evaluation_all_zero_kept_count"] == 6
    assert cpsc["evaluation_drop_all_zero_count"] == 4
    assert set(cpsc["k500_source_center_counts"]) == {
        "cpsc_2018",
        "cpsc_2018_extra",
    }

    center_dir = output_root / "pn2021" / "cpsc_2018"
    k500 = set(np.load(center_dir / "k500_hash_ids.npy").astype(str))
    eval_kept = set(
        np.load(center_dir / "evaluation_all_zero_kept_hash_ids.npy").astype(str)
    )
    eval_drop = set(
        np.load(center_dir / "evaluation_drop_all_zero_hash_ids.npy").astype(str)
    )
    assert len(k500) == 2
    assert k500.isdisjoint(eval_kept)
    assert k500.isdisjoint(eval_drop)

    replay_root = tmp_path / "splits-replay"
    replay_config_path = tmp_path / "splits-replay.yaml"
    config["output"]["root_dir"] = str(replay_root)
    replay_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    build_cache_splits(replay_config_path)
    for center in ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"):
        original = np.load(
            output_root / "pn2021" / center / "k500_hash_ids.npy"
        )
        replay = np.load(replay_root / "pn2021" / center / "k500_hash_ids.npy")
        np.testing.assert_array_equal(original, replay)
