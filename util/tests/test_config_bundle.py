from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from core.augmix import load_augmix_config
from core.lhat import load_lhat_config
from data_preprocess.data_runtime import _resolve_runtime_seed_config
from models.factory import build_model
from util.config_bundle import config_bundle_root, resolve_config_reference


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _copy_configs(tmp_path: Path) -> Path:
    destination = tmp_path / "strategy_configs"
    shutil.copytree(PROJECT_ROOT / "configs", destination)
    return destination


def test_config_reference_resolves_inside_copied_bundle(tmp_path: Path) -> None:
    config_root = _copy_configs(tmp_path)
    owner = config_root / "train" / "augmix.yaml"
    assert config_bundle_root(owner) == config_root.resolve()
    assert resolve_config_reference(
        "augmentation/operators.yaml",
        owner_config_path=owner,
        description="operator config",
        must_exist=True,
    ) == (config_root / "augmentation" / "operators.yaml").resolve()


def test_augmix_and_lhat_use_copied_seed_and_operator_yamls(
    tmp_path: Path,
) -> None:
    config_root = _copy_configs(tmp_path)
    copied_seed = config_root / "random_seed.yaml"
    copied_seed.write_text(
        yaml.safe_dump({"schema_version": 1, "random_seed": 314159}),
        encoding="utf-8",
    )

    augmix = load_augmix_config(config_root / "train" / "augmix.yaml")
    lhat = load_lhat_config(config_root / "train" / "lhat.yaml")

    assert augmix.random_seed_config_path == copied_seed.resolve()
    assert augmix.operator_config_path == (
        config_root / "augmentation" / "operators.yaml"
    ).resolve()
    assert lhat.random_seed_config_path == copied_seed.resolve()
    assert _resolve_runtime_seed_config(
        None,
        split_config_path=config_root / "data" / "splits.yaml",
    ) == copied_seed.resolve()

    model = build_model("efficientnet1dv2", config_root=config_root)
    assert model.random_seed_identity.config_path == copied_seed.resolve()
    assert model.random_seed_identity.base_seed == 314159


def test_config_reference_cannot_escape_bundle(tmp_path: Path) -> None:
    config_root = _copy_configs(tmp_path)
    owner = config_root / "data" / "splits.yaml"
    try:
        resolve_config_reference(
            "../outside.yaml",
            owner_config_path=owner,
            description="escaped config",
        )
    except ValueError as exc:
        assert "escapes config bundle" in str(exc)
    else:
        raise AssertionError("escaping a copied config bundle must fail")
