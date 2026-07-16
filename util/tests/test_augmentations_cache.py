from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from data_preprocess import augmentations_cache


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "augmentation" / "cache.yaml"


def test_cache_config_selects_five_centers_and_both_sampling_rates() -> None:
    config = augmentations_cache.load_cache_config(CONFIG)

    assert config["source"]["centers"] == [
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "cpsc_2018_extra",
        "georgia",
    ]
    assert config["corruption"]["domain_sampling_rate_hz"] == 500
    assert config["corruption"]["output_sampling_rates_hz"] == [100, 500]


def test_depth23_expands_to_all_twenty_canonical_combinations() -> None:
    config = augmentations_cache.load_cache_config(CONFIG)
    operator_config = augmentations_cache.load_operators_config(
        config["corruption"]["operators_config"]
    )
    compositions = augmentations_cache.build_compositions(config, operator_config)

    assert len(compositions) == 20
    assert sum(item["depth"] == 2 for item in compositions) == 10
    assert sum(item["depth"] == 3 for item in compositions) == 10
    assert len({item["composition_id"] for item in compositions}) == 20
    assert compositions[0]["operators"] == ["powerline_noise", "emg_noise"]


def test_composition_is_record_deterministic_and_finite() -> None:
    config = augmentations_cache.load_cache_config(CONFIG)
    operator_config = augmentations_cache.load_operators_config(
        config["corruption"]["operators_config"]
    )
    composition = augmentations_cache.build_compositions(
        config, operator_config
    )[0]
    time = np.linspace(0.0, 10.0, 5000, endpoint=False, dtype=np.float32)
    signal = np.stack(
        [np.sin(2.0 * np.pi * (1.0 + lead / 20.0) * time) for lead in range(12)],
        axis=1,
    )

    first = augmentations_cache.apply_composition(
        signal,
        source_hash="record-a",
        composition=composition,
        config=config,
        operator_config=operator_config,
    )
    second = augmentations_cache.apply_composition(
        signal,
        source_hash="record-a",
        composition=composition,
        config=config,
        operator_config=operator_config,
    )
    other_record = augmentations_cache.apply_composition(
        signal,
        source_hash="record-b",
        composition=composition,
        config=config,
        operator_config=operator_config,
    )

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other_record)
    assert first.shape == (5000, 12)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()


def test_500_to_100_linear_adapter_has_expected_contract() -> None:
    signals = np.zeros((2, 5000, 12), dtype=np.float32)
    signals[:, :, 0] = np.linspace(-1.0, 1.0, 5000, dtype=np.float32)

    resized = augmentations_cache.linear_interpolate_time_batch(signals, 1000)

    assert resized.shape == (2, 1000, 12)
    assert resized.dtype == np.float32
    assert np.isfinite(resized).all()
    np.testing.assert_allclose(resized[:, 0], signals[:, 0], atol=1e-6)
    np.testing.assert_allclose(resized[:, -1], signals[:, -1], atol=1e-6)


def test_cache_builder_has_no_legacy_runtime_dependencies() -> None:
    source = (REPO / "data_preprocess" / "augmentations_cache.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "methods.augmix",
        "ecg_adv_gen.data",
        "ecg_adv_gen.training",
        "configs/experiments",
        "configs/local",
    )

    for token in forbidden:
        assert token not in source


def test_small_dualrate_cache_build_smoke(tmp_path: Path, monkeypatch) -> None:
    # The production builder intentionally requires a large free-disk reserve.
    # Keep this tiny functional test independent of the host's /tmp occupancy.
    monkeypatch.setattr(
        augmentations_cache.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1024**4, free=900 * 1024**3),
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    centers = [
        "ningbo",
        "chapman_shaoxing",
        "cpsc_2018",
        "cpsc_2018_extra",
        "georgia",
    ]
    time_500 = np.linspace(0.0, 10.0, 5000, endpoint=False, dtype=np.float32)
    signals_500 = np.stack(
        [
            np.stack(
                [
                    np.sin(2.0 * np.pi * (1.0 + lead / 20.0) * time_500)
                    for lead in range(12)
                ],
                axis=1,
            )
            for _ in centers
        ],
        axis=0,
    ).astype(np.float32)
    signals_100 = augmentations_cache.linear_interpolate_time_batch(
        signals_500, 1000
    )
    labels = np.eye(5, dtype=np.uint8)
    record_ids = np.asarray([f"record-{index}" for index in range(5)])
    hash_ids = np.asarray([f"{'a' * 63}{index}" for index in range(5)])
    np.save(source_dir / "signals.npy", signals_100, allow_pickle=False)
    np.save(source_dir / "signals_500hz.npy", signals_500, allow_pickle=False)
    np.save(source_dir / "labels_super5.npy", labels, allow_pickle=False)
    np.save(source_dir / "record_ids.npy", record_ids, allow_pickle=False)
    np.save(source_dir / "hash_ids.npy", hash_ids, allow_pickle=False)
    pd.DataFrame(
        {
            "cache_index": np.arange(5),
            "record_id": record_ids,
            "record_key": [f"{center}/{record_id}" for center, record_id in zip(centers, record_ids)],
            "hash_id": hash_ids,
            "center": centers,
        }
    ).to_parquet(source_dir / "records.parquet", index=False)
    source_manifest = {
        "schema_version": 3,
        "dataset": "pn2021",
        "record_count": 5,
        "waveform": {
            "file": "signals.npy",
            "shape": [5, 1000, 12],
            "sampling_rate_hz": 100,
            "lead_order": augmentations_cache.EXPECTED_LEADS,
            "physical_unit": "mV",
            "normalization": "none",
        },
        "derived_waveforms": {
            "500hz_linear": {
                "file": "signals_500hz.npy",
                "source_file": "signals.npy",
                "shape": [5, 5000, 12],
                "sampling_rate_hz": 500,
            }
        },
        "labels": {
            "class_order": augmentations_cache.EXPECTED_CLASS_ORDER,
            "mapping_version": "v7_super5_sjr_rgq_review_20260528",
            "mapping_hash": "555ec85d5b51",
        },
    }
    (source_dir / "manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )

    config = augmentations_cache.load_cache_config(CONFIG)
    config["source"]["cache_dir"] = str(source_dir)
    config["output"]["cache_dir"] = str(tmp_path / "output")
    config["corruption"]["operators_config"] = str(
        REPO / "configs" / "augmentation" / "operators.yaml"
    )
    config["corruption"]["random_seed_file"] = str(
        REPO / "configs" / "random_seed.yaml"
    )
    config["execution"]["chunk_size"] = 2
    config["execution"]["checkpoint_every_chunks"] = 1
    config_path = tmp_path / "cache.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    plan = augmentations_cache.describe_cache_plan(config_path)
    assert plan["record_count"] == 5
    assert plan["view_count"] == 20
    assert plan["shape_100hz"] == [20, 5, 1000, 12]
    assert plan["shape_500hz"] == [20, 5, 5000, 12]

    output_dir = augmentations_cache.build_augmentations_cache(config_path)

    output_100 = np.load(output_dir / "signals_100hz.npy", mmap_mode="r")
    output_500 = np.load(output_dir / "signals_500hz.npy", mmap_mode="r")
    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert output_100.shape == (20, 5, 1000, 12)
    assert output_500.shape == (20, 5, 5000, 12)
    assert np.isfinite(output_100).all()
    assert np.isfinite(output_500).all()
    assert manifest["record_count"] == 5
    assert manifest["view_count"] == 20
    assert manifest["corrupted_record_view_count"] == 100
    assert manifest["center_counts"] == {center: 1 for center in sorted(centers)}
    assert not (tmp_path / ".output.building").exists()
