from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from util.random_seed import (
    DEFAULT_RANDOM_SEED_CONFIG_PATH,
    derive_seed,
    load_random_seed_config,
    make_numpy_rng,
    make_python_rng,
    make_torch_generator,
    seed_process,
)


def test_project_seed_config_has_stable_identity() -> None:
    config = load_random_seed_config()
    assert config.path == DEFAULT_RANDOM_SEED_CONFIG_PATH.resolve()
    assert config.base_seed == 20260501
    assert len(config.sha256) == 64


def test_namespaced_streams_are_repeatable_and_isolated() -> None:
    assert derive_seed("split", "replicate-0") == derive_seed(
        "split", "replicate-0"
    )
    assert derive_seed("split", "replicate-0") != derive_seed(
        "augmentation", "replicate-0"
    )

    np.testing.assert_array_equal(
        make_numpy_rng("augmentation", "sample-7").normal(size=16),
        make_numpy_rng("augmentation", "sample-7").normal(size=16),
    )
    first_python = make_python_rng("augmentation", "sample-7")
    second_python = make_python_rng("augmentation", "sample-7")
    assert [first_python.random() for _ in range(8)] == [
        second_python.random() for _ in range(8)
    ]
    assert torch.equal(
        torch.rand(16, generator=make_torch_generator("cpu", "model", "init")),
        torch.rand(16, generator=make_torch_generator("cpu", "model", "init")),
    )


def test_seed_process_records_identity_and_replays_global_backends() -> None:
    first = seed_process("training", "comparison-a", 0)
    first_values = (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
    )
    second = seed_process("training", "comparison-a", 0)
    second_values = (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
    )
    assert first == second
    assert first_values == second_values
    assert first.namespace == "training"
    assert first.identity == ("comparison-a", "0")
    assert first.config_sha256 == load_random_seed_config().sha256


def test_copied_seed_yaml_changes_only_streams_that_use_it(tmp_path: Path) -> None:
    seed_path = tmp_path / "random_seed.yaml"
    seed_path.write_text(
        yaml.safe_dump({"schema_version": 1, "random_seed": 17}),
        encoding="utf-8",
    )
    assert derive_seed("split", "replicate-0", config_path=seed_path) != derive_seed(
        "split", "replicate-0"
    )
    assert make_numpy_rng(
        "split", "replicate-0", config_path=seed_path
    ).randint(0, 2**31) == make_numpy_rng(
        "split", "replicate-0", config_path=seed_path
    ).randint(0, 2**31)
