from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from core.augmix import load_augmix_config
from core.lhat import load_lhat_config
from data_preprocess.data_runtime import _resolve_runtime_seed_config
from models.factory import build_model
from util.config_bundle import config_bundle_root, resolve_config_reference


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_PROFILE_NAMES = (
    "a0_clean_v1",
    "a3c_depth23_v1",
    "a5_lhat_threechain_v1",
    "exp_lhat_replay_pool_v1",
    "exp_lhat_as_sixth_branch_v1",
    "exp_paired_augmix_latent_bridge_v1",
    "exp_augmix_guided_latent_simplex_v1",
)


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


def test_nested_method_profile_infers_copied_bundle_root(tmp_path: Path) -> None:
    config_root = _copy_configs(tmp_path)
    owner = config_root / "train" / "methods" / "a5_lhat_threechain_v1.yaml"
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text("schema_version: 1\n", encoding="utf-8")

    assert config_bundle_root(owner) == config_root.resolve()
    assert resolve_config_reference(
        "train/lhat.yaml",
        owner_config_path=owner,
        description="LHAT component config",
        must_exist=True,
    ) == (config_root / "train" / "lhat.yaml").resolve()


@pytest.mark.parametrize("profile_name", METHOD_PROFILE_NAMES)
def test_method_profiles_are_typed_bundle_portable_descriptions(
    tmp_path: Path,
    profile_name: str,
) -> None:
    config_root = _copy_configs(tmp_path)
    path = config_root / "train" / "methods" / f"{profile_name}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert set(payload) == {
        "schema_version",
        "method",
        "nodes",
        "inputs",
        "outputs",
        "objective",
        "resources",
        "contracts",
    }
    assert payload["schema_version"] == 1
    assert payload["method"]["id"] == profile_name
    assert payload["method"]["profile_version"] == 1
    assert payload["method"]["api_version"] == 1
    assert payload["nodes"]
    assert all("type" in node for node in payload["nodes"].values())
    assert all(
        node.get("implementation") in {"builtin", "registered", None}
        for node in payload["nodes"].values()
    )

    forbidden_training_keys = {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "scheduler",
        "device",
        "num_workers",
        "trainable_scope",
        "checkpoint_selection",
    }

    def nested_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key)
                yield from nested_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from nested_keys(child)

    assert forbidden_training_keys.isdisjoint(set(nested_keys(payload)))
    for resource in payload["resources"].values():
        if resource.get("type") == "config_reference":
            assert resolve_config_reference(
                resource["path"],
                owner_config_path=path,
                description=f"{profile_name} resource",
                must_exist=True,
            ).is_file()
        if "seed_config" in resource:
            assert resolve_config_reference(
                resource["seed_config"],
                owner_config_path=path,
                description=f"{profile_name} seed",
                must_exist=True,
            ).is_file()


def test_a5_reuses_one_lhat_view_for_bce_and_uncorrupted_chain3() -> None:
    path = (
        PROJECT_ROOT
        / "configs"
        / "train"
        / "methods"
        / "a5_lhat_threechain_v1.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["nodes"]["threechain_augmix"]["inputs"]["chain3_waveform"] == (
        "lhat_view"
    )
    direct_bce = next(
        term
        for term in payload["objective"]["terms"]
        if term["id"] == "lhat_direct_bce"
    )
    assert direct_bce["view"] == "lhat_view"
    assert payload["contracts"][
        "lhat_direct_bce_and_augmix_chain3_share_exact_view"
    ] is True
    assert payload["contracts"]["augmix_chain3_additional_corruption"] is False


@pytest.mark.parametrize(
    "profile_name",
    tuple(name for name in METHOD_PROFILE_NAMES if name.startswith("exp_")),
)
def test_experimental_profiles_mark_unimplemented_nodes(profile_name: str) -> None:
    path = (
        PROJECT_ROOT / "configs" / "train" / "methods" / f"{profile_name}.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["method"]["status"] == "experimental_unimplemented_nodes"
    assert payload["contracts"]["executable"] is False
    assert any(
        node.get("status") == "experimental_unimplemented"
        and node.get("implementation") is None
        for node in payload["nodes"].values()
    )


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


@pytest.mark.parametrize(
    ("relative", "section", "loader", "message"),
    (
        ("train/augmix.yaml", "method", load_augmix_config, "method keys"),
        ("train/lhat.yaml", "decoder_bridge", load_lhat_config, "decoder_bridge keys"),
    ),
)
def test_mechanism_configs_reject_unknown_keys(
    tmp_path: Path,
    relative: str,
    section: str,
    loader,
    message: str,
) -> None:
    config_root = _copy_configs(tmp_path)
    path = config_root / relative
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload[section]["typo_field"] = True
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        loader(path)
