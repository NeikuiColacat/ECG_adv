from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import core.pn2021_tuning as tuning
from models.checkpoints import CheckpointIdentity
from models.contracts import ECGFOUNDER_SPEC, EFFICIENTNET1DV2_SPEC


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "train" / "PN2021_direct_tune.yaml"


class _Model(torch.nn.Module):
    model_spec = EFFICIENTNET1DV2_SPEC

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))


class _ECGFounderModel(_Model):
    model_spec = ECGFOUNDER_SPEC


class _Loader:
    def __init__(
        self,
        partition: str,
        *,
        dataset_name: str = "pn2021",
        view: int | None = None,
        hash_ids: tuple[str, ...] | None = None,
    ) -> None:
        self.partition = partition
        self.dataset_name = dataset_name
        self.view = view
        self.closed = False
        hashes = hash_ids or tuple(
            f"{partition}-{index:03d}"
            for index in range(400 if partition == "k500_tune_train" else 100)
        )
        self.dataset = SimpleNamespace(selection=SimpleNamespace(hash_ids=hashes))

    def close(self) -> None:
        self.closed = True

    def describe(self) -> dict[str, object]:
        selection = {
            "partition": self.partition,
            "record_count": len(self.dataset.selection.hash_ids),
            "hash_ids": list(self.dataset.selection.hash_ids),
        }
        dataset: dict[str, object] = {
            "dataset": self.dataset_name,
            "selection": selection,
        }
        if self.view is not None:
            dataset["composition_id"] = f"composition_{self.view:02d}"
        return {"dataset": dataset}


def _fake_loader_from_call(kwargs: dict[str, object]) -> _Loader:
    partition = str(kwargs["partition"])
    dataset_name = str(kwargs["dataset"])
    view = kwargs.get("view")
    hashes = tuple(f"validation-{index:03d}" for index in range(100))
    return _Loader(
        partition,
        dataset_name=dataset_name,
        view=None if view is None else int(view),
        hash_ids=(
            tuple(f"train-{index:03d}" for index in range(400))
            if partition == "k500_tune_train"
            else hashes
        ),
    )


def test_tuning_config_locks_family_balanced_frozen_validation_protocol():
    config = tuning.load_pn2021_tuning_config(CONFIG)

    assert config.profile_name == "pn2021_direct_family_balanced_tuning"
    assert config.payload["protocol_lock"]["method_id"] == "direct_depth23_fixed20"
    assert config.payload["data"]["train_partition"] == "k500_tune_train"
    assert config.payload["data"]["validation_partition"] == "k500_tune_validation"
    assert config.payload["protocol_lock"][
        "validation_records_allowed_in_latent_pool"
    ] is False
    assert config.payload["training"]["family_loss_weights"] == {
        "clean": 0.5,
        "corrupted_total": 0.5,
        "corrupted_per_composition": 0.025,
    }
    assert config.payload["validation_predictions"]["composition_indices"] == list(
        range(20)
    )
    assert config.payload["pooled_selection"]["score"] == {
        "clean_weight": 0.5,
        "robust_weight": 0.5,
    }
    assert config.payload["pooled_selection"]["clean_floor"] == {
        "reference": "same_backbone_locked_clean_selection",
        "maximum_drop_absolute": 0.01,
        "failure_policy": "no_selection",
    }
    assert config.payload["data"]["sampling_rate_hz"] == 100
    assert config.payload["data"]["normalization"] == "none"
    assert config.payload["data"]["model_domain_adapter"]["normalization"] == (
        "per_sample_global_zscore"
    )


def test_tuning_config_rejects_model_spec_rate_cache_contract(tmp_path: Path):
    config_root = tmp_path / "configs"
    shutil.copytree(REPO / "configs", config_root)
    config_path = config_root / "train" / "PN2021_direct_tune.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["data"]["sampling_rate_hz"] = "from_model_spec"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="data.sampling_rate_hz"):
        tuning.load_pn2021_tuning_config(config_path)


def test_tuning_builds_train400_clean100_and_twenty_frozen_views(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_get_dataloader(**kwargs):
        calls.append(kwargs)
        return _fake_loader_from_call(kwargs)

    monkeypatch.setattr(tuning, "get_dataloader", fake_get_dataloader)
    for model, expected_points, expected_rate in (
        (_Model(), 1000, 100),
        (_ECGFounderModel(), 5000, 500),
    ):
        start = len(calls)
        loaders = tuning.build_pn2021_tuning_dataloaders(
            model,
            center="ningbo",
            config_path=CONFIG,
            dataloader_parameters={"num_workers": 0},
        )
        current = calls[start:]
        try:
            assert len(current) == 22
            assert current[0]["dataset"] == "pn2021"
            assert current[0]["partition"] == "k500_tune_train"
            assert current[0]["shuffle"] is True
            assert current[1]["dataset"] == "pn2021"
            assert current[1]["partition"] == "k500_tune_validation"
            assert current[1]["shuffle"] is False
            assert [call["view"] for call in current[2:]] == list(range(20))
            assert all(call["dataset"] == "pn2021c" for call in current[2:])
            assert all(
                call["partition"] == "k500_tune_validation"
                for call in current[2:]
            )
            assert all(call["shuffle"] is False for call in current[1:])
            assert all(call["sampling_rate_hz"] == 100 for call in current)
            assert all(call["prepare_for_model"] is False for call in current)
            assert all(call["sanitize"] is False for call in current)
            assert all(call["global_zscore"] is False for call in current)
            assert all(call["selection_resident"] is True for call in current)
            assert all(call["output_layout"] == "time_channel" for call in current)
            assert len(loaders.corrupted_validation) == 20
            assert all(
                loader.dataset.selection.hash_ids
                == loaders.clean_validation.dataset.selection.hash_ids
                for loader in loaders.corrupted_validation
            )
            adapted = loaders.input_adapter(torch.randn(2, 1000, 12))
            assert tuple(adapted.shape) == (2, 12, expected_points)
            description = loaders.describe()
            assert description["input_adapter"]["model"] == model.model_spec.name
            assert description["input_adapter"]["output_layout"] == "channel_time"
            assert (
                description["input_adapter"]["resampling"]
                == (
                    "identity_100hz"
                    if expected_rate == 100
                    else "linear_100_to_500hz_align_corners_true"
                )
            )
        finally:
            loaders.close()
        assert loaders.train.closed
        assert loaders.clean_validation.closed
        assert all(loader.closed for loader in loaders.corrupted_validation)


def test_tuning_rejects_corruption_hash_order_drift_and_closes_all(monkeypatch):
    opened: list[_Loader] = []

    def fake_get_dataloader(**kwargs):
        loader = _fake_loader_from_call(kwargs)
        if kwargs.get("view") == 7:
            hashes = tuple(reversed(loader.dataset.selection.hash_ids))
            loader.dataset.selection.hash_ids = hashes
        opened.append(loader)
        return loader

    monkeypatch.setattr(tuning, "get_dataloader", fake_get_dataloader)

    with pytest.raises(RuntimeError, match="composition 7 changed validation hash order"):
        tuning.build_pn2021_tuning_dataloaders(
            _Model(),
            center="ningbo",
            config_path=CONFIG,
            dataloader_parameters={"num_workers": 0},
        )

    assert len(opened) == 22
    assert all(loader.closed for loader in opened)


def test_locked_source_checkpoint_identity_matches_registry():
    model = _Model()
    model.checkpoint_identity = CheckpointIdentity(
        path=Path(
            "/home/linbinhao/ECG_adv_data/runs/manual_refactor/"
            "managed_ptbxl_effnet_source_v1/training/checkpoints/best.pt"
        ),
        sha256="fff9bd293677be54a606e28775c66d410d55b0ad7440a4e7fbde55357ec13e68",
        state_key_count=1,
        missing_keys=(),
        unexpected_keys=(),
    )
    config = tuning.load_pn2021_tuning_config(CONFIG)

    identity = tuning.validate_locked_source_checkpoint(model, config)

    assert identity["baseline_id"] == "ptbxl_super5_source_v1"
    assert identity["model_family"] == "efficientnet1dv2"


def test_direct_tuning_delegates_train400_and_epoch_frozen_evaluator(monkeypatch):
    model = _Model()
    train_loader = _Loader("k500_tune_train")
    clean_loader = _Loader("k500_tune_validation")
    corrupted_loaders = tuple(
        _Loader(
            "k500_tune_validation",
            dataset_name="pn2021c",
            view=index,
            hash_ids=clean_loader.dataset.selection.hash_ids,
        )
        for index in range(20)
    )
    loaders = tuning.PN2021TuningDataLoaders(
        train=train_loader,
        clean_validation=clean_loader,
        corrupted_validation=corrupted_loaders,
        center="ningbo",
        model_name="efficientnet1dv2",
        input_adapter=tuning.CanonicalTuningInputAdapter(EFFICIENTNET1DV2_SPEC),
        resolved_parameters={},
        explicit_overrides={},
    )
    observed: dict[str, object] = {}
    sentinel = object()
    evaluator = object()
    monkeypatch.setattr(tuning, "validate_locked_source_checkpoint", lambda *args: {})
    monkeypatch.setattr(
        tuning, "build_pn2021_tuning_dataloaders", lambda *args, **kwargs: loaders
    )

    def fake_evaluator(loaders_arg, config_arg, **kwargs):
        observed["evaluator_loaders"] = loaders_arg
        observed["evaluator_config"] = config_arg
        observed["evaluator_kwargs"] = kwargs
        return evaluator

    def fake_train_online_model(model_arg, train_arg, **kwargs):
        observed["model"] = model_arg
        observed["train"] = train_arg
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(tuning, "FrozenValidationEvaluator", fake_evaluator)
    monkeypatch.setattr(tuning, "train_online_model", fake_train_online_model)

    result = tuning.train_pn2021_direct_tuning(
        model,
        center="ningbo",
        config_path=CONFIG,
    )

    assert result is sentinel
    assert observed["train"] is train_loader
    assert observed["epoch_evaluator"] is evaluator
    assert observed["evaluator_loaders"] is loaders
    assert Path(observed["method_config_path"]).name == "direct_depth23_fixed20.yaml"
    assert train_loader.closed and clean_loader.closed
    assert all(loader.closed for loader in corrupted_loaders)
