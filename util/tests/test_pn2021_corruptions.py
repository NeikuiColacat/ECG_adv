"""PN2021-C corruption tests against the rebuilt five-operator surface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import data_preprocess.augmentations_cache as cache_builder
from data_preprocess import load_cache as cache_access
from data_preprocess import preprocess_primitives as primitives
from data_preprocess.augmentations_cache import (
    apply_composition,
    build_compositions,
    load_cache_config,
    load_operators_profile,
)
from data_preprocess.load_cache import EXPECTED_CLASS_ORDER, EXPECTED_LEADS
from util.augmentations.operators import baseline_shift, random_leads_masking
from util.augmentations.profile import CANONICAL_OPERATOR_ORDER
from util import config_bundle
from util import pn2021_artifact_contract as artifact_contract
from util.random_seed import derive_seed


REPO = Path(__file__).resolve().parents[2]
CACHE_CONFIG = REPO / "configs" / "augmentation" / "cache.yaml"
OPERATORS_CONFIG = REPO / "configs" / "augmentation" / "operators.yaml"
SEED_CONFIG = REPO / "configs" / "random_seed.yaml"


def _contracts():
    cache = load_cache_config(CACHE_CONFIG)
    profile = load_operators_profile(
        cache["corruption"]["operators_config"],
        owner_config_path=CACHE_CONFIG,
        profile_name=cache["corruption"]["profile"],
        severity=int(cache["corruption"]["severity"]),
        expected_seed_config_path=SEED_CONFIG,
    )
    return cache, profile


def test_locked_profile_and_cache_share_one_pn2021c_contract(monkeypatch) -> None:
    cache, profile = _contracts()

    assert cache_builder.EXPECTED_LEADS is EXPECTED_LEADS
    assert cache_builder.EXPECTED_CLASS_ORDER is EXPECTED_CLASS_ORDER
    assert cache_builder._artifact_contract is artifact_contract
    assert cache_builder._read_yaml_mapping is config_bundle.load_yaml_mapping
    assert cache_builder._load_cache is cache_access.load_cache
    assert cache_builder.primitives is primitives
    assert {
        "_load_source_contract",
        "_select_source_records",
        "_validate_source_arrays",
        "linear_interpolate_time_batch",
        "_resolve_project_path",
        "_sha256_file",
        "sha256_file",
    }.isdisjoint(vars(cache_builder))
    assert profile.canonical_order == CANONICAL_OPERATOR_ORDER
    assert profile.profile_name == "pn2021c_paper_anchored_s5_v1"
    assert profile.severity == 5
    assert cache["storage_policy"] == "canonical_100hz_only"
    assert cache["corruption"]["domain_sampling_rate_hz"] == 500
    assert cache["corruption"]["output_sampling_rates_hz"] == [100]
    assert "waveform_500hz_file" not in cache["source"]
    assert "waveform_500hz_file" not in cache["output"]
    assert cache["corruption"]["normalization"] == "none"
    assert profile.parameters_for("powerline_noise")["max_amplitude"] == 0.5
    assert profile.parameters_for("baseline_shift")["amplitude_mode"] == (
        "signed_shared_uniform"
    )
    assert profile.parameters_for("random_leads_masking")[
        "ensure_at_least_one_lead"
    ] is True
    calls = []
    monkeypatch.setattr(cache_builder, "build_augmentations_cache", calls.append)
    monkeypatch.setattr("sys.argv", ["augmentations_cache.py", "--config", "cache.yaml"])
    cache_builder.main()
    assert calls == ["cache.yaml"]
    assert not hasattr(cache_builder, "describe_cache_plan")
    monkeypatch.setattr("sys.argv", ["augmentations_cache.py", "--plan-only"])
    with pytest.raises(SystemExit, match="2"):
        cache_builder.main()
    assert calls == ["cache.yaml"]


def test_depth2_plus_depth3_expands_to_the_locked_twenty_views() -> None:
    cache, profile = _contracts()
    compositions = build_compositions(cache, profile)

    assert len(compositions) == 20
    assert [item["view_index"] for item in compositions] == list(range(20))
    assert sum(item["depth"] == 2 for item in compositions) == 10
    assert sum(item["depth"] == 3 for item in compositions) == 10
    assert len({item["composition_id"] for item in compositions}) == 20
    for item in compositions:
        operator_indices = [
            CANONICAL_OPERATOR_ORDER.index(name) for name in item["operators"]
        ]
        assert operator_indices == sorted(operator_indices)


def test_cache_staging_persists_only_the_canonical_100hz_waveform(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache, _ = _contracts()
    output_dir = tmp_path / "pn2021c_100hz"
    config = {
        **cache,
        "output": {**cache["output"], "cache_dir": str(output_dir)},
    }
    # This shape/storage test must not depend on the shared host's free space.
    monkeypatch.setattr(cache_builder.shutil, "disk_usage",
                        lambda _: SimpleNamespace(total=100 * 1024**3, free=100 * 1024**3))

    staging, signals_100, state = cache_builder._prepare_staging(
        output_dir=output_dir,
        config_identity={"config_identity_hash": "test"},
        config=config,
        compositions=[{"view_index": 0}],
        record_count=2,
    )

    assert signals_100.shape == (1, 2, 1000, 12)
    assert state["shape_100hz"] == [1, 2, 1000, 12]
    assert "shape_500hz" not in state
    assert not (staging / "signals_500hz.npy").exists()


@pytest.mark.parametrize("total_gib", [100, 1000])
@pytest.mark.parametrize("margin_bytes", [-1, 0])
def test_cache_staging_enforces_exact_disk_reserve_boundary(tmp_path, monkeypatch, total_gib, margin_bytes):
    cache, _ = _contracts()
    output_dir = tmp_path / "disk_reserve_probe"
    config = {**cache, "output": {**cache["output"], "cache_dir": str(output_dir)}}
    total = total_gib * 1024**3
    reserve = max(int(total * 0.10), 50 * 1024**3)
    estimated = 1 * 2 * 1000 * 12 * np.dtype("float32").itemsize
    monkeypatch.setattr(cache_builder.shutil, "disk_usage",
                        lambda _: SimpleNamespace(total=total, free=reserve + estimated + margin_bytes))
    arguments = dict(output_dir=output_dir, config_identity={"config_identity_hash": "reserve"},
                     config=config, compositions=[{"view_index": 0}], record_count=2)
    if margin_bytes < 0:
        with pytest.raises(OSError, match="insufficient target-disk safety margin"):
            cache_builder._prepare_staging(**arguments)
        assert not output_dir.with_name(f".{output_dir.name}.building").exists()
    else:
        staging, signals, state = cache_builder._prepare_staging(**arguments)
        assert staging.is_dir() and signals.shape == (1, 2, 1000, 12)
        assert state["shape_100hz"] == list(signals.shape)


def test_composite_view_is_reproducible_from_record_and_composition_identity() -> None:
    cache, profile = _contracts()
    composition = build_compositions(cache, profile)[0]
    time = np.linspace(0.0, 10.0, 5000, endpoint=False, dtype=np.float32)
    signal = np.stack(
        [np.sin(2.0 * np.pi * (1.0 + lead / 20.0) * time) for lead in range(12)],
        axis=1,
    ).astype(np.float32)

    first = apply_composition(
        signal,
        source_hash="record-hash",
        composition=composition,
        config=cache,
        operator_config=profile,
        seed_config_path=SEED_CONFIG,
    )
    second = apply_composition(
        signal,
        source_hash="record-hash",
        composition=composition,
        config=cache,
        operator_config=profile,
        seed_config_path=SEED_CONFIG,
    )

    assert first.shape == (5000, 12)
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert not np.array_equal(first, signal)


def test_signed_baseline_shift_respects_the_configured_millivolt_bound() -> None:
    shifted = baseline_shift(
        np.zeros((1000, 12), dtype=np.float32),
        max_amplitude=0.5,
        shift_ratio=0.3,
        num_segment=1,
        amplitude_mode="signed_shared_uniform",
        p=1.0,
        rng=np.random.RandomState(7),
    )

    assert 0.0 < float(np.abs(shifted).max()) <= 0.5
    nonzero = shifted[np.nonzero(shifted)]
    assert np.unique(np.sign(nonzero)).size >= 1


def test_random_lead_masking_always_keeps_one_survivor() -> None:
    masked = random_leads_masking(
        np.ones((100, 12), dtype=np.float32),
        mask_leads_prob=1.0,
        ensure_at_least_one_lead=True,
        p=1.0,
        rng=np.random.RandomState(11),
    )
    surviving = (np.abs(masked).sum(axis=0) > 0).sum()

    assert surviving == 1


def test_corruption_seed_is_stable_and_identity_sensitive() -> None:
    first = derive_seed(
        "pn2021c",
        "record-hash",
        "d2__powerline_noise__emg_noise",
        config_path=SEED_CONFIG,
    )

    assert first == derive_seed(
        "pn2021c",
        "record-hash",
        "d2__powerline_noise__emg_noise",
        config_path=SEED_CONFIG,
    )
    assert first != derive_seed(
        "pn2021c",
        "record-hash",
        "d2__powerline_noise__baseline_wander",
        config_path=SEED_CONFIG,
    )
    assert 0 <= first < 2**32
