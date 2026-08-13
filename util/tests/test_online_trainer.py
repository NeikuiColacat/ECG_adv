"""Fast characterization contracts for the finite online-training control plane."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import boot_scripts.train_pn2021 as train_boot
import core.online_trainer as trainer
import core.train_PN2021 as train_adapter
import data_preprocess.data_runtime as data_runtime
import models.vae as vae
from core.methods.registry import AuxiliaryVariant, RecipeKind, load_recipe_spec
from core.methods.runtime import build_method_runtime, _derive_seed
from core.train_PN2021 import _validate_locked_source_checkpoint
from models.checkpoints import (
    CheckpointIdentity,
    load_model_checkpoint,
)
from models.contracts import ECGFOUNDER_SPEC, EFFICIENTNET1DV2_SPEC
from util.pn2021_artifact_contract import (
    resolve_artifact_reference,
    validate_training_lineage,
)
from util.random_seed import derive_seed, load_random_seed_config


REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "configs"
RECIPES = CONFIG_ROOT / "train" / "methods"
ONLINE_CONFIG = CONFIG_ROOT / "train" / "PN2021.yaml"
MATCHED_A0_A1_CONFIG = CONFIG_ROOT / "train" / "PN2021_matched_base_a0_a1.yaml"


def _recipe(filename: str):
    return load_recipe_spec(RECIPES / filename)


def test_matched_a0_a1_profile_locks_effnet_e25_and_founder_e30() -> None:
    config = trainer.load_online_train_config(MATCHED_A0_A1_CONFIG)
    effnet, effnet_overrides = trainer.resolve_online_training_parameters(
        config, "efficientnet1dv2"
    )
    founder, founder_overrides = trainer.resolve_online_training_parameters(
        config, "ecgfounder"
    )
    assert effnet_overrides == founder_overrides == {}
    assert (effnet["epochs"], effnet["scheduler_horizon_epochs"], effnet["batch_size"]) == (
        25, 30, 128
    )
    assert (founder["epochs"], founder["scheduler_horizon_epochs"], founder["batch_size"]) == (
        30, 30, 64
    )
    assert effnet["stage1_steps"] == founder["stage1_steps"] == 0
    assert config.payload["protocol"]["validation_split"] is False
    assert config.payload["protocol"]["checkpoint_selection"] == "last"


def _matched():
    payload = yaml.safe_load((RECIPES / "augmix_simclr_lhat.yaml").read_text())
    payload["recipe"].update(id="augmix_simclr_matched_no_vae",
        auxiliary_variant="matched_no_vae", scientific_arm="augmix_simclr_matched_no_vae",
        status="prospective_matched_ablation")
    for name in ("vae", "lhat_config", "lhat_rng"):
        payload["resources"].pop(name)
    return load_recipe_spec(payload)


def _lineage(tmp_path: Path) -> dict:
    recipe = _recipe("a0_clean_v1.yaml")
    config = trainer.load_online_train_config(ONLINE_CONFIG)
    adaptation = {
        "dataset": "pn2021", "partition": "k500", "logical_center": "ningbo",
        "source_centers": ["ningbo"], "record_count": 500, "split_id": "k500",
        "hash_id_set_sha256": "1" * 64, "split_manifest_sha256": "2" * 64,
        "source_manifest_sha256": "3" * 64,
        "mapping_version": "v7_super5_sjr_rgq_review_20260528",
        "mapping_hash": "555ec85d5b51", "class_order": list(EFFICIENTNET1DV2_SPEC.class_order),
    }
    loader = SimpleNamespace(dataset=SimpleNamespace(
        selection=SimpleNamespace(describe=lambda: adaptation)))
    model = {"spec": EFFICIENTNET1DV2_SPEC.describe(),
             "checkpoint_identity": {"path": str(tmp_path / "moved.pt"), "sha256": "4" * 64}}
    seed = {"base_seed": 7, "effective_seed": 9, "namespace": "online",
            "config_sha256": "5" * 64}
    return trainer._training_lineage(model_identity=model, spec=EFFICIENTNET1DV2_SPEC,
        center="ningbo", recipe=recipe, config=config, seed_identity=seed,
        train_dataloader=loader)


def test_non_dry_handoffs_keep_bundle_relative_recipe(monkeypatch, tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"characterization-only checkpoint stub")
    boot_method = Path("train/methods/a0_clean_v1.yaml")
    seen = {}

    monkeypatch.setattr(train_boot, "build_model", lambda *a, **k: torch.nn.Identity())
    def capture_boot(*args, **kwargs):
        seen["boot"] = kwargs["method_config_path"]
        return SimpleNamespace(describe=lambda: {"status": "mocked"})
    monkeypatch.setattr(train_boot, "train_pn2021", capture_boot)
    assert train_boot.main(
        ["--config", str(ONLINE_CONFIG), "--config-root", str(CONFIG_ROOT),
         "--model", "efficientnet1dv2", "--method-config", str(boot_method),
         "--center", "ningbo", "--source-checkpoint", str(checkpoint)]
    ) == 0
    assert seen["boot"] == boot_method and not seen["boot"].is_absolute()
    assert '"status": "mocked"' in capsys.readouterr().out

    adapter_method = Path("train/methods/augmix_simclr_lhat.yaml")
    model = torch.nn.Identity()
    model.model_spec = EFFICIENTNET1DV2_SPEC
    result = object()
    events = []

    class Loader:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(f"{self.name}.close")

    ordered_loader, training_loader = Loader("ordered"), Loader("training")

    class Plan:
        center = "ningbo"

        def open_ordered(self):
            events.append("ordered.open")
            return ordered_loader

        def open_training(self):
            events.append("training.open")
            return training_loader

    loader_plan = Plan()

    def capture(key, value):
        def call(*args, **kwargs):
            seen.setdefault(key, []).append(kwargs["method_config_path"])
            events.append(key)
            return value
        return call
    monkeypatch.setattr(train_adapter, "_validate_locked_source_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(
        train_adapter,
        "build_pn2021_k500_loader_plan",
        lambda **kwargs: loader_plan,
    )
    monkeypatch.setattr(train_adapter.torch.cuda, "is_available", lambda: True)

    def capture_pool(encoder, loader, **kwargs):
        del encoder
        events.append("pool")
        assert loader is ordered_loader
        assert kwargs == {
            "encoder_identity": "a" * 64,
            "device": torch.device("cuda"),
            "num_candidates": 20,
            "standardizer_epsilon": 1e-6,
        }
        return object()

    monkeypatch.setattr(train_adapter, "build_latent_pool", capture_pool)
    monkeypatch.setattr(train_adapter, "train_online_model", capture("trainer", result))

    class Component:
        def __init__(self, name):
            self.name = name

        def to(self, *args, **kwargs):
            events.append(f"{self.name}.to:{args[0]}")
            return self

        def eval(self):
            events.append(f"{self.name}.eval")
            return self

    encoder, decoder = Component("encoder"), Component("decoder")
    encoder.checkpoint_identity = SimpleNamespace(sha256="a" * 64)
    assert train_adapter.train_pn2021(
        model, center="ningbo", method_config_path=adapter_method,
        encoder=encoder, decoder=decoder,
        config_path=ONLINE_CONFIG, config_root=CONFIG_ROOT,
    ) is result
    assert events == [
        "encoder.to:cuda", "encoder.eval", "ordered.open", "pool",
        "ordered.close", "encoder.to:cpu", "training.open", "trainer",
        "training.close", "encoder.to:cpu", "decoder.to:cpu",
    ]

    events.clear()
    assert train_adapter.train_pn2021(
        model, center="ningbo", method_config_path=boot_method,
        config_path=ONLINE_CONFIG, config_root=CONFIG_ROOT,
    ) is result
    assert events == ["training.open", "trainer", "training.close"]
    assert seen == {
        "boot": boot_method,
        "trainer": [adapter_method, boot_method],
    }


def test_source_checkpoint_lock_covers_both_backbones(tmp_path: Path) -> None:
    cases = ((EFFICIENTNET1DV2_SPEC, "checkpoint_identity", "1" * 64, tmp_path / "effnet.pt"),
             (ECGFOUNDER_SPEC, "task_checkpoint_identity", "2" * 64, tmp_path / "founder.pt"))
    registry = tmp_path / "source.yaml"
    registry.write_text("schema_version: 1\nmodels:\n" + "".join(
        f"  {spec.name}:\n    selected_checkpoint: {path}\n"
        f"    selected_checkpoint_sha256: '{sha}'\n" for spec, _, sha, path in cases))
    config = SimpleNamespace(references={"source_baseline_registry": registry})
    for spec, attribute, sha, path in cases:
        _validate_locked_source_checkpoint(
            SimpleNamespace(**{attribute: CheckpointIdentity(path, sha, 1, (), ())}), spec, config)
        for bad, message in ((CheckpointIdentity(path, "f" * 64, 1, (), ()), "SHA256"),
                             (CheckpointIdentity(tmp_path / "wrong.pt", sha, 1, (), ()), "path")):
            with pytest.raises(ValueError, match=message):
                _validate_locked_source_checkpoint(SimpleNamespace(**{attribute: bad}), spec, config)


def test_training_lineage_is_exact_and_binds_managed_selection(tmp_path: Path) -> None:
    lineage = _lineage(tmp_path)
    assert lineage["model"] == {
        "name": "efficientnet1dv2", "spec": EFFICIENTNET1DV2_SPEC.describe()}
    assert lineage["source_checkpoint"] == {"sha256": "4" * 64}
    assert validate_training_lineage(lineage) == lineage
    for field, value in (("unexpected", True), ("center", "georgia"),
                         ("schema_version", True)):
        drifted = {**lineage, field: value}
        with pytest.raises(ValueError):
            validate_training_lineage(drifted)
    for section, field, value in (
        ("seed", "base_seed", None), ("seed", "effective_seed", True),
        ("seed", "namespace", 7), ("comparison", "group", None),
        ("comparison", "replicate_id", "0"), ("method", "schema_version", "2"),
        ("method", "recipe_version", False), ("adaptation_data", "split_id", 7),
        ("adaptation_data", "record_count", 499),
        ("adaptation_data", "source_centers", ["ningbo", "georgia"]),
    ):
        drifted = {**lineage, section: {**lineage[section], field: value}}
        with pytest.raises(ValueError):
            validate_training_lineage(drifted)

    with pytest.raises(ValueError):
        resolve_artifact_reference(
            {"path": 7, "sha256": "0" * 64}, owner=tmp_path / "owner.json", name="fixture"
        )


def test_checkpoint_loader_exposes_schema3_and_finite_schema2(tmp_path: Path) -> None:
    lineage = _lineage(tmp_path); method = lineage["method"]
    model = torch.nn.Linear(2, 1); state = model.state_dict()
    run = {"center": "ningbo", "scientific_arm": method["scientific_arm"],
           "model": {"spec": EFFICIENTNET1DV2_SPEC.describe(),
                     "checkpoint_identity": {"sha256": "4" * 64}},
           "recipe": {"recipe_id": method["recipe_id"],
                      "recipe_spec_sha256": method["recipe_spec_sha256"]},
           "seed": lineage["seed"], "config": {"sha256": lineage["training_config_sha256"]}}
    payload = {"schema_version": 3, "lineage": lineage, "center": "ningbo",
               "method_id": method["recipe_id"], "scientific_arm": method["scientific_arm"],
               "selection": "last", "run_identity": run, "model_state_dict": state}
    path = tmp_path / "schema3.pt"; torch.save(payload, path)
    identity = load_model_checkpoint(torch.nn.Linear(2, 1), path)
    assert (identity.checkpoint_schema_version, identity.lineage,
            identity.legacy_training_identity) == (3, lineage, None)
    for run_drift in (
        {**run, "seed": {**run["seed"], "effective_seed": 10}},
        {**run, "config": {"sha256": "0" * 64}},
        {**run, "model": {**run["model"], "checkpoint_identity": {"sha256": "0" * 64}}},
    ):
        torch.save({**payload, "run_identity": run_drift}, path)
        with pytest.raises(ValueError, match="run_identity"):
            load_model_checkpoint(torch.nn.Linear(2, 1), path)
    payload["center"] = "georgia"; torch.save(payload, path)
    with pytest.raises(ValueError, match="root.*lineage"):
        load_model_checkpoint(torch.nn.Linear(2, 1), path)

    legacy = {"schema_version": 2, "center": "ningbo", "method_id": "a0_clean_v1",
              "scientific_arm": "clean", "selection": "last", "model_state_dict": state,
              "run_identity": {"center": "ningbo", "scientific_arm": "clean",
                  "model": {"spec": EFFICIENTNET1DV2_SPEC.describe()},
                  "method": {"model_name": "efficientnet1dv2",
                             "method": {"profile_name": "a0_clean_v1",
                                        "scientific_arm": "clean"}}}}
    torch.save(legacy, path); identity = load_model_checkpoint(torch.nn.Linear(2, 1), path)
    assert identity.legacy_training_identity == {"model": "efficientnet1dv2",
        "center": "ningbo", "method_id": "a0_clean_v1", "scientific_arm": "clean",
        "selection": "last"}
    raw = tmp_path / "raw.pt"; torch.save(state, raw)
    identity = load_model_checkpoint(torch.nn.Linear(2, 1), raw)
    assert identity.checkpoint_schema_version is None and identity.lineage is None
    assert identity.legacy_training_identity is None


def test_online_config_references_and_overrides_remain_closed() -> None:
    config = trainer.load_online_train_config(ONLINE_CONFIG)
    resolved, supplied = trainer.resolve_online_training_parameters(config, "efficientnet1dv2")
    assert config.path == ONLINE_CONFIG.resolve()
    assert set(config.references) == {"split_config", "data_load_config", "random_seed_config",
                                      "source_baseline_registry"}
    assert all(path.is_file() for path in config.references.values())
    assert "diagnostics" not in config.payload
    assert "diagnostics_file" not in config.payload["output"]
    assert (resolved["epochs"], resolved["scheduler_horizon_epochs"],
            resolved["stage1_steps"], supplied) == (23, 30, 1024, {})
    with pytest.raises(ValueError, match="unknown online training parameters"):
        trainer.resolve_online_training_parameters(config, "efficientnet1dv2", {"typo": 1})
    with pytest.raises(ValueError, match="greater than or equal"):
        trainer.resolve_online_training_parameters(config, "efficientnet1dv2",
            {"epochs": 4, "scheduler_horizon_epochs": 3})


def test_pn2021_loader_plan_is_entirely_yaml_owned() -> None:
    expected = {
        "efficientnet1dv2": 128,
        "ecgfounder": 64,
    }
    for model_name, batch_size in expected.items():
        plan = train_adapter.build_pn2021_k500_loader_plan(
            center="ningbo",
            model_name=model_name,
            config_path=ONLINE_CONFIG,
            config_root=CONFIG_ROOT,
        )
        assert plan.center == "ningbo"
        assert plan.batch_size == batch_size
        assert (
            plan.num_workers,
            plan.pin_memory,
            plan.persistent_workers,
            plan.prefetch_factor,
            plan.cache_mode,
            plan.validate_values,
        ) == (0, True, False, 2, "mmap", "sample")
        assert (
            plan.selection_resident,
            plan.selection_resident_pin_memory,
            plan.drop_last,
        ) == (True, False, False)
        assert plan.seed_namespace == (
            "pn2021_k500_matched:"
            "aligned_targetonly_augmix_lhat_simplified_20260805:"
            f"0:ningbo:{model_name}"
        )
        assert plan.split_config_path == CONFIG_ROOT / "data" / "splits.yaml"
        assert plan.data_load_config_path == CONFIG_ROOT / "data" / "data_load.yaml"
        assert plan.seed_config_path == CONFIG_ROOT / "random_seed.yaml"


def test_pn2021_boot_dry_run_includes_loader_plan_without_side_effects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        train_boot,
        "build_model",
        lambda *args, **kwargs: pytest.fail("model was built"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("GPU availability was queried"),
    )
    monkeypatch.setattr(
        data_runtime,
        "_build_runtime_loader",
        lambda *args, **kwargs: pytest.fail("data was opened"),
    )
    assert train_boot.main(
        [
            "--config", str(ONLINE_CONFIG),
            "--config-root", str(CONFIG_ROOT),
            "--model", "efficientnet1dv2",
            "--method-config", "train/methods/a0_clean_v1.yaml",
            "--center", "ningbo",
            "--source-checkpoint", "/does/not/need/to/exist.pt",
            "--dry-run",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["loader_plan"]["dataset"] == "pn2021"
    assert payload["loader_plan"]["partition"] == "k500"
    assert payload["loader_plan"]["logical_center"] == "ningbo"


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("training", "num_workers"), "0"),
        (("training", "num_workers"), 0.5),
        (("training", "pin_memory"), "false"),
        (("training", "persistent_workers"), 1),
        (("training", "selection_resident"), "true"),
        (("training", "cache_mode"), "ram"),
        (("training", "model_profiles", "efficientnet1dv2", "batch_size"), True),
    ),
)
def test_pn2021_boot_dry_run_rejects_malicious_runtime_yaml_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, path: tuple[str, ...], value: object
) -> None:
    config = trainer.load_online_train_config(ONLINE_CONFIG)
    payload = copy.deepcopy(config.payload)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    poisoned = replace(config, payload=payload)
    monkeypatch.setattr(
        train_boot, "load_online_train_config", lambda *args, **kwargs: poisoned
    )
    monkeypatch.setattr(
        train_adapter, "load_online_train_config", lambda *args, **kwargs: poisoned
    )
    monkeypatch.setattr(
        train_boot,
        "build_model",
        lambda *args, **kwargs: pytest.fail("model was built"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("GPU availability was queried"),
    )
    monkeypatch.setattr(
        data_runtime,
        "_build_runtime_loader",
        lambda *args, **kwargs: pytest.fail("data was opened"),
    )
    with pytest.raises((TypeError, ValueError)):
        train_boot.main(
            [
                "--config", str(ONLINE_CONFIG),
                "--config-root", str(CONFIG_ROOT),
                "--model", "efficientnet1dv2",
                "--method-config", "train/methods/a0_clean_v1.yaml",
                "--center", "ningbo",
                "--source-checkpoint", "/does/not/need/to/exist.pt",
                "--dry-run",
            ]
        )


def test_pn2021_boot_and_adapter_reject_retired_override_surfaces() -> None:
    option_strings = {
        option
        for action in train_boot.build_parser()._actions
        for option in action.option_strings
    }
    assert option_strings == {
        "-h",
        "--help",
        "--config",
        "--config-root",
        "--model",
        "--method-config",
        "--center",
        "--source-checkpoint",
        "--output-dir",
        "--dry-run",
    }
    retired = {
        "--vae-checkpoint",
        "--device",
        "--trainable-scope",
        "--epochs",
        "--scheduler-horizon-epochs",
        "--batch-size",
        "--learning-rate",
        "--weight-decay",
        "--minimum-learning-rate-ratio",
        "--gradient-clip-norm",
        "--amp",
        "--amp-dtype",
        "--num-workers",
        "--pin-memory",
        "--persistent-workers",
        "--prefetch-factor",
        "--cache-mode",
        "--validate-values",
        "--drop-last",
        "--pos-weight",
    }
    assert option_strings.isdisjoint(retired)
    assert set(inspect.signature(train_adapter.train_pn2021).parameters) == {
        "model",
        "center",
        "method_config_path",
        "encoder",
        "decoder",
        "config_path",
        "config_root",
        "output_dir",
    }
    assert not hasattr(train_adapter, "PN2021_DATALOADER_PARAMETER_NAMES")
    assert not hasattr(train_adapter, "build_pn2021_k500_dataloader")
    assert not hasattr(train_adapter, "build_pn2021_latent_pool")
    assert not hasattr(train_adapter, "_encoder_sha256")
    assert not hasattr(trainer, "_iter_exposure_batches")
    assert {
        "__exposure_name", "__exposure_group", "__exposure_index",
        "__composition_index", "__objective_terms", "__loss_scale",
    }.isdisjoint(inspect.getsource(trainer.train_online_model))
    assert vae.__all__ == [
        "build_ecgtwin_vae", "decode_to_ptbxl_waveform", "load_vae_config",
        "prepare_ecgtwin_encoder_input"]
    assert tuple(vae.VAEConfig.__dataclass_fields__) == (
        "checkpoint_path", "expected_encoder_state_keys", "expected_decoder_state_keys")
    assert tuple(inspect.signature(vae.SelfAttention).parameters) == ()
    assert tuple(inspect.signature(vae.VAEAttentionBlock).parameters) == ()
    builder_source = inspect.getsource(vae.build_ecgtwin_vae)
    assert all(f"{name}.vae_config" not in builder_source for name in ("encoder", "decoder"))
    with torch.device("meta"):
        encoder, decoder = vae.VAEEncoder(), vae.VAEDecoder()
    assert (len(encoder.state_dict()), len(decoder.state_dict())) == (104, 136)
    assert tuple(sum(parameter.numel() for parameter in module.parameters())
                 for module in (encoder, decoder)) == (12_218_448, 17_333_664)
    assert tuple(sum(1 for _ in module.named_modules())
                 for module in (encoder, decoder)) == (74, 101)
    assert tuple(inspect.signature(vae.VAEEncoder.forward).parameters) == ("self", "value")
    assert tuple(inspect.signature(vae.decode_to_ptbxl_waveform).parameters) == ("decoder", "latent")


def test_finite_exposure_plans_lock_direct21_a1_rotating5_mainline6() -> None:
    for filename in ("a0_clean_v1.yaml", "a3c_depth23_v1.yaml"):
        steps = trainer._method_exposure_steps(_recipe(filename))
        assert [(step.name, step.loss_scale) for step in steps] == [("base", 1.0)]
    direct = trainer._method_exposure_steps(_recipe("direct_depth23_fixed20.yaml"))
    assert [step.composition_index for step in direct] == [-1, *range(20)]
    assert [step.loss_scale for step in direct] == pytest.approx([.5, *([.025] * 20)])

    a1 = _recipe("a1_corrupt_ft_rot4_v1.yaml")
    a1_schedules = [trainer._method_exposure_steps(a1, epoch=e) for e in range(1, 6)]
    a1_compositions = [
        step.composition_index
        for plan in a1_schedules
        for step in plan
        if step.name.startswith("corruption_")
    ]
    assert sorted(a1_compositions) == list(range(20))
    assert all(len(plan) == 5 for plan in a1_schedules)
    assert [step.loss_scale for step in a1_schedules[0]] == pytest.approx(
        [.5, .125, .125, .125, .125]
    )
    assert a1.scientific_contract["stages"] == ("supervised_adaptation",)
    assert "stage2_teacher" not in a1.scientific_contract
    with pytest.raises(ValueError, match="positive integer"):
        trainer._method_exposure_steps(a1, epoch=0)

    mainline = _recipe("augmix_simclr_lhat.yaml")
    schedules = [trainer._method_exposure_steps(mainline, epoch=e) for e in range(1, 6)]
    compositions = [s.composition_index for plan in schedules for s in plan
                    if s.name.startswith("corruption_")]
    assert sorted(compositions) == list(range(20))
    assert all(len(plan) == 6 and plan[-1].loss_scale == 2.0 for plan in schedules)
    assert [s.loss_scale for s in schedules[0]] == pytest.approx([.5, .125, .125, .125, .125, 2.])
    with pytest.raises(ValueError, match="positive integer"):
        trainer._method_exposure_steps(mainline, epoch=0)

    matched = _matched(); matched_steps = trainer._method_exposure_steps(matched)
    assert (matched.kind, matched.auxiliary_variant) == (
        RecipeKind.TWO_STAGE_AUGMIX_LHAT, AuxiliaryVariant.MATCHED_NO_VAE)
    assert [s.name for s in matched_steps] == [
        "clean", "corruption_00", "corruption_01", "corruption_10", "corruption_11"]
    assert matched.scientific_contract["stages"] == ("augmix_simclr", "supervised_adaptation")
    assert matched.scientific_contract["stage2_teacher"] == "post_stage1_pre_stage2_snapshot"


def test_empty_lhat_auxiliary_is_an_empty_gradient_sum() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD((parameter,), lr=.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    (parameter.square()).backward()
    accumulated = parameter.grad.clone()
    empty = trainer._ObjectiveBatch(
        total=torch.tensor(0.0), raw_terms={}, weighted_terms={},
        valid_counts={"lhat_direct_bce": 0})
    returned = trainer._backward_objective(
        empty, loss_scale=2.0, scaler=scaler, empty_lhat_auxiliary=True)
    assert returned.item() == 0.0 and torch.equal(parameter.grad, accumulated)
    optimizer.step()
    assert parameter.item() == pytest.approx(1.6)

    detached = trainer._ObjectiveBatch(
        total=torch.tensor(1.0), raw_terms={}, weighted_terms={},
        valid_counts={"lhat_direct_bce": 1})
    with pytest.raises(RuntimeError, match="detached"):
        trainer._backward_objective(
            detached, loss_scale=2.0, scaler=scaler,
            empty_lhat_auxiliary=False)


RNG_CASES = [
    ("a3c_depth23_v1.yaml", "base", None, "corruption_rng", "depth23_corruption", "composition_and_operators", 4099549646),
    ("direct_depth23_fixed20.yaml", "corruption_07", 7, "corruption_rng", "depth23_corruption", "composition_and_operators", 1342437248),
    ("augmix_simclr_lhat.yaml", "corruption_07", 7, "corruption_rng", "depth23_corruption", "composition_and_operators", 100675112),
    ("augmix_simclr_lhat.yaml", "auxiliary", None, "lhat_rng", "lhat", "candidate_selection", 1664578656),
]


@pytest.mark.parametrize("filename,exposure,composition,rng_name,node,stream,expected", RNG_CASES)
def test_method_rng_permanent_goldens(filename, exposure, composition, rng_name,
                                      node, stream, expected) -> None:
    config = trainer.load_online_train_config(ONLINE_CONFIG); random = config.payload["random_seed"]
    identity = trainer._method_rng_identity(comparison_group=random["comparison_group"],
        replicate_id=random["replicate_id"], center="ningbo", model_name="efficientnet1dv2",
        epoch=2, batch_hash_sha256=hashlib.sha256(b"second-a\nsecond-b").hexdigest(),
        exposure_name=exposure, composition_index=composition)
    recipe = _recipe(filename)
    base = load_random_seed_config(config.references["random_seed_config"]).base_seed
    assert all("view_execution_step" not in value for value in identity)
    assert _derive_seed(base, recipe.comparison_rng_identity, recipe.rng_namespaces[rng_name],
                        node, stream, identity) == expected


@pytest.mark.parametrize("name,weights", [
    ("clean", None), ("direct", [.5, *([.025] * 20)]),
    ("a1", [.5, .125, .125, .125, .125]),
    ("mainline", [.5, .125, .125, .125, .125, 0]),
    ("matched", [.5, .125, .125, .125, .125])])
def test_batch_norm_plan_preserves_family_weights(name, weights) -> None:
    recipes = {"clean": lambda: _recipe("a0_clean_v1.yaml"),
               "direct": lambda: _recipe("direct_depth23_fixed20.yaml"),
               "a1": lambda: _recipe("a1_corrupt_ft_rot4_v1.yaml"),
               "mainline": lambda: _recipe("augmix_simclr_lhat.yaml"),
               "matched": _matched}
    recipe = recipes[name](); steps = trainer._method_exposure_steps(recipe)
    plan = trainer._build_batch_norm_momentum_plan(
        torch.nn.BatchNorm1d(2, momentum=.1), recipe, exposure_steps=steps)
    if weights is None:
        assert plan is None; return
    assert plan.exposure_weights == pytest.approx(weights)
    contributions = []
    for momentum in plan.schedules[0]:
        contributions = [x * (1 - momentum) for x in contributions] + [momentum]
    assert contributions == pytest.approx([.1 * weight for weight in weights])


def test_stage1_uses_resolved_augmix_constants_and_hash_free_legacy_seed(monkeypatch) -> None:
    recipe = _recipe("augmix_simclr_lhat.yaml")
    runtime = build_method_runtime(recipe, model_name="efficientnet1dv2",
        config_root=CONFIG_ROOT, latent_pool=object(), decoder=torch.nn.Identity())
    augmix = runtime.augmix_config; assert augmix is not None
    captured = []
    def capture(device, namespace, *identity, config_path):
        seed = derive_seed(namespace, *identity, config_path=config_path)
        captured.append((seed, identity)); return torch.Generator(device=device).manual_seed(seed)
    patches = {
        "make_torch_generator": capture,
        "_cache_k500_logits": lambda *a, **k: {},
        "_feature_width": lambda *a, **k: 4,
        "_head_parameters": lambda *a, **k: (),
        "prepare_canonical_model_input": lambda raw, *a, **k: raw.mean((1, 2)).unsqueeze(1),
        "generate_two_chain_augmix_strong_view": lambda raw, **k: SimpleNamespace(mixed_raw=raw * .5),
        "_forward_logits_and_features": lambda model, x, spec: (model(x), model(x)),
        "_teacher_logits_for_hashes": lambda cache, hashes, **k: torch.zeros(len(hashes), 4),
        "_simclr_nt_xent": lambda clean, strong, **k: (clean - strong).square().mean(),
        "_weighted_logit_anchor_loss": lambda logits, teacher, weights: logits.square().mean(),
    }
    for name, replacement in patches.items():
        monkeypatch.setattr(trainer, name, replacement)
    loader = [{"waveform": torch.full((2, 1000, 12), float(step + 1)),
               "hash_id": (f"batch-{step}-a", f"batch-{step}-b")} for step in range(2)]
    summary = trainer._run_augmix_simclr_stage1(torch.nn.Linear(1, 4), loader,
        spec=SimpleNamespace(name="efficientnet1dv2"), device=torch.device("cpu"),
        recipe=recipe, augmix_config=augmix, center="ningbo", base_seed=20260501,
        resolved={"stage1_steps": 2, "stage1_learning_rate": 1e-3,
                  "stage1_weight_decay": 0., "stage1_gradient_clip_norm": 1.},
        normalization_epsilon=1e-6, amp_enabled=False, amp_dtype=torch.bfloat16)
    assert [seed for seed, _ in captured] == [3967304348, 2995841993]
    assert [identity for _, identity in captured] == [
        ("augmix_simclr_lhat", "ningbo", "efficientnet1dv2", "base_seed=20260501", f"stage1_step={i}")
        for i in range(2)]
    assert (summary["simclr_temperature"], summary["augmix_internal_chains"],
            summary["augmix_dirichlet_alpha"], summary["augmix_beta_alpha"],
            summary["logit_anchor_weight"], summary["ptbxl_replay_weight"]) == (
                .5, 2, .5, .5, 5., 0.)
