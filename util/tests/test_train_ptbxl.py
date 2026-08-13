from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import boot_scripts.train_ptbxl_ecgfounder as founder_boot
import boot_scripts.train_ptbxl_effnet as effnet_boot
import core.train_PTBXL as ptbxl_module
import core.supervised_trainer as supervised_module
import data_preprocess.data_runtime as data_runtime
import models
import models.factory as model_factory
from core.supervised_trainer import load_train_config
from core.train_PTBXL import (
    build_ptbxl_loader_plan,
    train_ptbxl,
)
from data_preprocess.data_runtime import PTBXLLoaderPlan
from models.contracts import ECGFOUNDER_SPEC, EFFICIENTNET1DV2_SPEC


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "train" / "PTBXL.yaml"


class _Model(torch.nn.Module):
    def __init__(self, spec) -> None:
        super().__init__()
        self.model_spec = spec
        self.weight = torch.nn.Parameter(torch.zeros(()))


class _Loader:
    def __init__(self, partition: str) -> None:
        self.partition = partition
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def describe(self) -> dict[str, object]:
        return {"partition": self.partition}


class _Plan:
    instances: list[_Plan] = []

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)
        self.opened: list[str] = []
        self.loaders: list[_Loader] = []
        self.__class__.instances.append(self)

    def _open(self, partition: str) -> _Loader:
        self.opened.append(partition)
        loader = _Loader(partition)
        self.loaders.append(loader)
        return loader

    def open_train(self) -> _Loader:
        return self._open(str(self.train_partition))

    def open_validation(self) -> _Loader:
        return self._open(str(self.validation_partition))

    def open_test(self) -> _Loader:
        assert self.test_partition is not None
        return self._open(str(self.test_partition))

    def describe(self) -> dict[str, object]:
        return dict(self.__dict__)


@pytest.fixture(autouse=True)
def _clear_fake_plans() -> None:
    _Plan.instances.clear()


@pytest.mark.parametrize(
    ("spec", "train_batch", "eval_batch"),
    (
        (EFFICIENTNET1DV2_SPEC, 128, 256),
        (ECGFOUNDER_SPEC, 64, 128),
    ),
)
def test_ptbxl_plan_is_fully_derived_from_model_and_yaml(
    monkeypatch: pytest.MonkeyPatch,
    spec,
    train_batch: int,
    eval_batch: int,
) -> None:
    monkeypatch.setattr(ptbxl_module, "PTBXLLoaderPlan", _Plan)
    monkeypatch.setattr(ptbxl_module, "train_model", lambda *args, **kwargs: object())

    train_ptbxl(_Model(spec), config_path=CONFIG)
    plan = _Plan.instances[0]
    assert plan.train_partition == "train"
    assert plan.validation_partition == "validation"
    assert plan.test_partition == "test"
    assert not hasattr(plan, "sampling_rate_hz")
    assert plan.train_batch_size == train_batch
    assert plan.eval_batch_size == eval_batch
    assert plan.num_workers == 4
    assert plan.pin_memory is True
    assert plan.persistent_workers is True
    assert plan.prefetch_factor == 2
    assert plan.cache_mode == "mmap"
    assert plan.validate_values == "sample"
    assert plan.drop_last is False
    assert plan.split_config_path == REPO / "configs" / "data" / "splits.yaml"
    assert plan.data_load_config_path == REPO / "configs" / "data" / "data_load.yaml"
    assert plan.seed_config_path == REPO / "configs" / "random_seed.yaml"
    assert plan.opened == ["train", "validation", "test"]
    assert all(loader.closed for loader in plan.loaders)


@pytest.mark.parametrize("model_spec", (EFFICIENTNET1DV2_SPEC, "efficientnet1dv2"))
def test_ptbxl_loader_plan_is_pure_and_accepts_spec_or_name(
    monkeypatch: pytest.MonkeyPatch, model_spec
) -> None:
    monkeypatch.setattr(ptbxl_module, "PTBXLLoaderPlan", _Plan)
    plan = build_ptbxl_loader_plan(model_spec, config_path=CONFIG)
    assert plan.train_batch_size == 128
    assert plan.opened == []


def test_ptbxl_plan_does_not_open_fold10_when_yaml_disables_final_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_train_config(CONFIG)
    payload = copy.deepcopy(config.payload)
    payload["selection"]["evaluate_test_at_end"] = False
    config = replace(config, payload=payload)
    monkeypatch.setattr(ptbxl_module, "load_train_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(ptbxl_module, "PTBXLLoaderPlan", _Plan)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        ptbxl_module,
        "train_model",
        lambda *args, **kwargs: captured.update(kwargs) or object(),
    )

    train_ptbxl(_Model(EFFICIENTNET1DV2_SPEC))
    plan = _Plan.instances[0]
    assert plan.test_partition is None
    assert plan.opened == ["train", "validation"]
    assert captured["test_dataloader"] is None
    assert all(loader.closed for loader in plan.loaders)


def test_runtime_ptbxl_plan_opens_raw100_btc_with_locked_split_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[data_runtime._LoaderRequest] = []

    def fake_build_runtime_loader(
        request: data_runtime._LoaderRequest,
    ) -> _Loader:
        calls.append(request)
        return _Loader(request.partition)

    monkeypatch.setattr(
        data_runtime, "_build_runtime_loader", fake_build_runtime_loader
    )
    config = load_train_config(CONFIG)
    plan = PTBXLLoaderPlan(
        train_partition="train",
        validation_partition="validation",
        test_partition="test",
        train_batch_size=128,
        eval_batch_size=256,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        cache_mode="mmap",
        validate_values="sample",
        drop_last=False,
        split_config_path=config.config_root / "data" / "splits.yaml",
        data_load_config_path=config.config_root / "data" / "data_load.yaml",
        seed_config_path=config.random_seed_config_path,
    )

    loaders = [plan.open_train(), plan.open_validation(), plan.open_test()]
    try:
        assert [call.partition for call in calls] == [
            "train",
            "validation",
            "test",
        ]
        assert [call.shuffle for call in calls] == [True, False, False]
        assert [call.drop_last for call in calls] == [False, False, False]
        assert [call.batch_size for call in calls] == [128, 256, 256]
        assert all(call.dataset == "ptbxl" for call in calls)
        assert all(call.logical_center is None for call in calls)
        assert all(call.view is None for call in calls)
        fixed_raw_contract = {
            "sampling_rate_hz",
            "prepare_for_model",
            "sanitize",
            "global_zscore",
            "output_layout",
        }
        assert fixed_raw_contract.isdisjoint(
            data_runtime._LoaderRequest.__dataclass_fields__
        )
    finally:
        for loader in loaders:
            loader.close()


@pytest.mark.parametrize(
    ("train_partition", "validation_partition", "test_partition"),
    (
        ("validation", "train", "test"),
        ("train", "test", "validation"),
        ("train", "validation", "train"),
    ),
)
def test_runtime_ptbxl_plan_rejects_swapped_official_partitions(
    train_partition: str,
    validation_partition: str,
    test_partition: str,
) -> None:
    config = load_train_config(CONFIG)
    with pytest.raises(ValueError, match="partition"):
        PTBXLLoaderPlan(
            train_partition=train_partition,
            validation_partition=validation_partition,
            test_partition=test_partition,
            train_batch_size=128,
            eval_batch_size=256,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            cache_mode="mmap",
            validate_values="sample",
            drop_last=False,
            split_config_path=config.config_root / "data" / "splits.yaml",
            data_load_config_path=config.config_root / "data" / "data_load.yaml",
            seed_config_path=config.random_seed_config_path,
        )


def test_runtime_ptbxl_plan_rejects_training_drop_last() -> None:
    config = load_train_config(CONFIG)
    with pytest.raises(ValueError, match="drop_last"):
        PTBXLLoaderPlan(
            train_partition="train",
            validation_partition="validation",
            test_partition="test",
            train_batch_size=128,
            eval_batch_size=256,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            cache_mode="mmap",
            validate_values="sample",
            drop_last=True,
            split_config_path=config.config_root / "data" / "splits.yaml",
            data_load_config_path=config.config_root / "data" / "data_load.yaml",
            seed_config_path=config.random_seed_config_path,
        )


def test_ptbxl_yaml_loader_profile_rejects_runtime_override_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_train_config(CONFIG)
    payload = copy.deepcopy(config.payload)
    payload["boot_models"]["efficientnet1dv2"]["dataloader_parameters"][
        "num_workers"
    ] = 0
    config = replace(config, payload=payload)
    monkeypatch.setattr(ptbxl_module, "load_train_config", lambda *args, **kwargs: config)

    with pytest.raises(ValueError, match="keys mismatch"):
        train_ptbxl(_Model(EFFICIENTNET1DV2_SPEC))
    assert _Plan.instances == []


def test_train_ptbxl_delegates_to_trainer_and_closes_owned_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(ptbxl_module, "PTBXLLoaderPlan", _Plan)

    def fake_train_model(model, train_dataloader, **kwargs):
        captured.update(kwargs)
        captured["model"] = model
        captured["train"] = train_dataloader
        return sentinel

    monkeypatch.setattr(ptbxl_module, "train_model", fake_train_model)
    model = _Model(EFFICIENTNET1DV2_SPEC)
    result = train_ptbxl(
        model,
        config_path=CONFIG,
        output_dir=tmp_path / "run",
        training_parameters={"epochs": 3},
    )

    plan = _Plan.instances[0]
    assert result is sentinel
    assert captured["model"] is model
    assert captured["train"] is plan.loaders[0]
    assert captured["validation_dataloader"] is plan.loaders[1]
    assert captured["test_dataloader"] is plan.loaders[2]
    assert captured["training_parameters"] == {"epochs": 3}
    assert {"device", "pos_weight", "class_names"}.isdisjoint(captured)
    adapter = captured["input_adapter"]
    assert adapter.describe()["source_sampling_rate_hz"] == 100
    assert adapter.describe()["source_layout"] == "time_channel"
    raw = torch.arange(2 * 1000 * 12, dtype=torch.float32).reshape(2, 1000, 12)
    adapted = adapter(raw)
    assert tuple(adapted.shape) == (2, 12, 1000)
    assert all(loader.closed for loader in plan.loaders)

    for failure, opened in (
        ("test", ["train", "validation"]),
        ("trainer", ["train", "validation", "test"]),
    ):
        _Plan.instances.clear()

        class _FailingPlan(_Plan):
            def _open(self, partition: str) -> _Loader:
                if partition == failure:
                    raise RuntimeError(failure)
                return super()._open(partition)

        def fail_trainer(*args, **kwargs):
            if failure == "trainer":
                raise RuntimeError(failure)
            return object()

        monkeypatch.setattr(ptbxl_module, "PTBXLLoaderPlan", _FailingPlan)
        monkeypatch.setattr(ptbxl_module, "train_model", fail_trainer)
        with pytest.raises(RuntimeError, match=failure):
            train_ptbxl(_Model(EFFICIENTNET1DV2_SPEC), config_path=CONFIG)
        failed_plan = _Plan.instances[0]
        assert failed_plan.opened == opened
        assert all(loader.closed for loader in failed_plan.loaders)


def test_ptbxl_founder_adapter_upsamples_raw100_on_the_input_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(ptbxl_module, "PTBXLLoaderPlan", _Plan)

    def fake_train_model(model, train_dataloader, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(ptbxl_module, "train_model", fake_train_model)
    train_ptbxl(_Model(ECGFOUNDER_SPEC), config_path=CONFIG)
    adapter = captured["input_adapter"]
    raw = torch.arange(2 * 1000 * 12, dtype=torch.float32).reshape(2, 1000, 12)
    adapted = adapter(raw)
    assert adapted.device == raw.device
    assert tuple(adapted.shape) == (2, 12, 5000)
    torch.testing.assert_close(
        adapted.std(dim=(1, 2), correction=0),
        torch.ones(2),
        atol=2e-6,
        rtol=0.0,
    )


def _option_strings(parser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }


def test_ptbxl_boot_cli_is_the_launcher_owned_finite_surface() -> None:
    common = {"--config", "--config-root", "--output-dir", "--dry-run"}
    assert _option_strings(effnet_boot.build_parser()) == common
    assert _option_strings(founder_boot.build_parser()) == common | {"--epochs"}
    assert models.__all__ == ["available_models", "build_model", "get_model_spec",
                              "build_ecgtwin_vae", "load_vae_config"]
    assert model_factory.available_models() == ("ecgfounder", "efficientnet1dv2")
    assert supervised_module.__all__ == [
        "DEFAULT_TRAIN_CONFIG", "load_train_config", "train_model"
    ]
    assert ptbxl_module.__all__ == [
        "build_ptbxl_loader_plan", "run_ptbxl_boot", "train_ptbxl"
    ]
    assert not hasattr(ptbxl_module, "PTBXLDataLoaders")
    assert not hasattr(ptbxl_module, "build_ptbxl_dataloaders")
    assert not any(hasattr(model_factory, name) for name in
                   ("MODEL_ALIASES", "MODEL_BUILDERS", "MODEL_SPECS", "normalize_model_name"))
    with pytest.raises(ValueError, match="unknown model"):
        model_factory.get_model_spec("effnet")


def test_ptbxl_boot_dry_runs_preserve_model_profiles(capsys) -> None:
    assert effnet_boot.main(["--dry-run"]) == 0
    effnet = json.loads(capsys.readouterr().out)
    assert effnet["model"]["sampling_rate_hz"] == 100
    assert effnet["training_parameters"]["epochs"] == 30
    assert effnet["loader_plan"]["dataset"] == "ptbxl"
    assert effnet["loader_plan"]["train_batch_size"] == 128
    assert "dataloader_parameters" not in effnet

    assert founder_boot.main(["--dry-run", "--epochs", "10"]) == 0
    founder = json.loads(capsys.readouterr().out)
    assert founder["model"]["sampling_rate_hz"] == 500
    assert founder["trainable_scope"] == "full"
    assert founder["training_parameters"]["epochs"] == 10
    assert founder["loader_plan"]["eval_batch_size"] == 128
    assert "dataloader_parameters" not in founder

    with pytest.raises(SystemExit):
        founder_boot.main(["--dry-run", "--epochs", "9"])


@pytest.mark.parametrize("boot", (effnet_boot, founder_boot))
@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("training", "num_workers"), "4"),
        (("training", "num_workers"), 1.5),
        (("training", "pin_memory"), "true"),
        (("training", "persistent_workers"), 1),
        (("training", "prefetch_factor"), True),
        (("training", "cache_mode"), "ram"),
        (("profile", "train_batch_size"), True),
        (("profile", "eval_batch_size"), 64.0),
    ),
)
def test_ptbxl_boot_dry_run_rejects_malicious_runtime_yaml_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, boot, path: tuple[str, ...], value: object
) -> None:
    config = load_train_config(CONFIG)
    payload = copy.deepcopy(config.payload)
    resolved_path = (
        ("boot_models", boot.MODEL_NAME, "dataloader_parameters", path[-1])
        if path[0] == "profile"
        else path
    )
    target = payload
    for key in resolved_path[:-1]:
        target = target[key]
    target[resolved_path[-1]] = value
    poisoned = replace(config, payload=payload)
    monkeypatch.setattr(
        ptbxl_module, "load_train_config", lambda *args, **kwargs: poisoned
    )
    monkeypatch.setattr(
        ptbxl_module,
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
        boot.main(["--dry-run"])


@pytest.mark.parametrize("value", ("false", 0, None))
def test_ptbxl_boot_dry_run_rejects_nonboolean_test_policy(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    config = load_train_config(CONFIG)
    payload = copy.deepcopy(config.payload)
    payload["selection"]["evaluate_test_at_end"] = value
    poisoned = replace(config, payload=payload)
    monkeypatch.setattr(
        ptbxl_module, "load_train_config", lambda *args, **kwargs: poisoned
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("GPU availability was queried"),
    )
    with pytest.raises(TypeError, match="evaluate_test_at_end"):
        effnet_boot.main(["--dry-run"])


@pytest.mark.parametrize(
    ("boot", "spec", "arguments", "expected_epochs"),
    (
        (effnet_boot, EFFICIENTNET1DV2_SPEC, [], 30),
        (founder_boot, ECGFOUNDER_SPEC, ["--epochs", "10"], 10),
    ),
)
def test_ptbxl_boot_execution_passes_only_yaml_training_profile(
    monkeypatch: pytest.MonkeyPatch,
    boot,
    spec,
    arguments: list[str],
    expected_epochs: int,
) -> None:
    model = _Model(spec)
    captured: dict[str, object] = {}
    monkeypatch.setattr(ptbxl_module, "build_model", lambda *args, **kwargs: model)

    class _Result:
        def describe(self) -> dict[str, object]:
            return {"ok": True}

    def fake_train_ptbxl(actual_model, **kwargs):
        captured["model"] = actual_model
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(ptbxl_module, "train_ptbxl", fake_train_ptbxl)
    assert boot.main(arguments) == 0
    assert captured["model"] is model
    assert captured["training_parameters"]["epochs"] == expected_epochs
    assert "dataloader_parameters" not in captured
    assert "device" not in captured
    assert "pos_weight" not in captured
    assert "class_names" not in captured


@pytest.mark.parametrize(
    ("boot", "arguments"),
    (
        (effnet_boot, ["--num-workers", "0"]),
        (effnet_boot, ["--checkpoint", "/tmp/model.pt"]),
        (founder_boot, ["--device", "cpu"]),
        (founder_boot, ["--trainable-scope", "head"]),
    ),
)
def test_ptbxl_boot_rejects_retired_runtime_flags(boot, arguments) -> None:
    with pytest.raises(SystemExit):
        boot.main(["--dry-run", *arguments])
