from __future__ import annotations

import hashlib
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
    "augmix_simclr_lhat",
    "exp_lhat_replay_pool_v1",
    "exp_lhat_as_sixth_branch_v1",
    "exp_paired_augmix_latent_bridge_v1",
    "exp_augmix_guided_latent_simplex_v1",
)


def _copy_configs(tmp_path: Path) -> Path:
    destination = tmp_path / "strategy_configs"
    shutil.copytree(PROJECT_ROOT / "configs", destination)
    return destination


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_k500_handoff_locks_config_and_partition_identities() -> None:
    config_root = PROJECT_ROOT / "configs"
    handoff_path = config_root / "data" / "k500_handoff.yaml"
    handoff = yaml.safe_load(handoff_path.read_text(encoding="utf-8"))

    assert handoff["schema_version"] == 1
    assert handoff["profile_name"] == "pn2021_k500_data_interface_handoff"

    for name, reference in handoff["required_configs"].items():
        relative = Path(reference["path"])
        assert not relative.is_absolute(), name
        resolved = resolve_config_reference(
            reference["path"],
            owner_config_path=handoff_path,
            config_root=config_root,
            description=f"{name} handoff config",
            must_exist=True,
        )
        assert _sha256(resolved) == reference["sha256"]

    split_config = yaml.safe_load(
        (config_root / "data" / "splits.yaml").read_text(encoding="utf-8")
    )
    seed_config = yaml.safe_load(
        (config_root / "random_seed.yaml").read_text(encoding="utf-8")
    )
    mapping_config = yaml.safe_load(
        (config_root / "data" / "PN2021_super5_v7.yaml").read_text(
            encoding="utf-8"
        )
    )
    evaluation_config = yaml.safe_load(
        (config_root / "eval" / "PN2021.yaml").read_text(encoding="utf-8")
    )
    protocol = handoff["protocol"]

    assert protocol["split_id"] == split_config["pn2021"]["split_id"]
    assert protocol["mapping_version"] == mapping_config["mapping_version"]
    assert protocol["mapping_hash"] == mapping_config["mapping_hash"]
    assert protocol["class_order"] == mapping_config["class_order"]
    assert protocol["class_order"] == evaluation_config["protocol"]["class_order"]
    assert protocol["base_seed"] == seed_config["random_seed"]
    assert protocol["seed_namespace"] == split_config["random_seed"]["namespace"]
    assert protocol["k500_count_per_center"] == split_config["pn2021"]["k"]
    assert protocol["heldout"]["exclude_k500_hashes"] is True
    assert protocol["heldout"]["model_adaptation_allowed"] is False
    assert protocol["heldout"]["checkpoint_selection_allowed"] is False

    expected_sources = {
        center["name"]: center["source_centers"]
        for center in split_config["pn2021"]["logical_centers"]
    }
    centers = handoff["split_artifacts"]["logical_centers"]
    assert set(centers) == set(expected_sources)
    for center_name, center in centers.items():
        assert center["source_centers"] == expected_sources[center_name]
        assert center["k500"]["count"] == 500
        assert center["k500_tune_train"]["count"] == 400
        assert center["k500_tune_validation"]["count"] == 100
        for partition_name in (
            "k500",
            "k500_tune_train",
            "k500_tune_validation",
            "evaluation_all_zero_kept",
            "evaluation_drop_all_zero",
        ):
            digest = center[partition_name]["hash_id_set_sha256"]
            assert len(digest) == 64
            assert set(digest) <= set("0123456789abcdef")


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
    owner = config_root / "train" / "methods" / "augmix_simclr_lhat.yaml"
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
    expected_id = (
        "latent_threechain_augmix_residual_depth23_aug075"
        if profile_name == "exp_paired_augmix_latent_bridge_v1"
        else profile_name
    )
    assert payload["method"]["id"] == expected_id
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


def test_mainline_separates_stage1_augmix_from_stage2_contracted_lhat() -> None:
    path = (
        PROJECT_ROOT
        / "configs"
        / "train"
        / "methods"
        / "augmix_simclr_lhat.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["nodes"]["lhat"]["type"] == (
        "vae_lhat_attack_then_contract_view"
    )
    direct_bce = next(
        term
        for term in payload["objective"]["terms"]
        if term["id"] == "lhat_direct_bce"
    )
    assert direct_bce["view"] == "lhat_view"
    assert payload["contracts"]["stage1_vae_lhat_tail_fraction"] == 0.0
    assert payload["contracts"]["auxiliary_gradient_merge"] == "direct_sum"
    assert payload["contracts"]["auxiliary_rng_policy"] == (
        "snapshot_restore_global_rng"
    )
    assert payload["contracts"]["attack_then_contract_version"] == (
        "preflip_maxloss_grid_v1"
    )


@pytest.mark.parametrize(
    "profile_name",
    tuple(
        name
        for name in METHOD_PROFILE_NAMES
        if name.startswith("exp_")
        and name != "exp_paired_augmix_latent_bridge_v1"
    ),
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
