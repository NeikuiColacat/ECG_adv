from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from core.supervised_trainer import (
    _classification_metrics,
    load_train_config,
    seed_training_process,
    train_model,
)
from core.pn2021_tuning import CanonicalTuningInputAdapter
from models.contracts import ECGFOUNDER_SPEC


REPO = Path(__file__).resolve().parents[2]


class _TinyECGDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, offset: float = 0.0) -> None:
        generator = torch.Generator().manual_seed(17)
        self.waveforms = torch.randn(8, 12, 20, generator=generator) + offset
        self.labels = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0],
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"waveform": self.waveforms[index], "label": self.labels[index]}


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(12, 5)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(waveform.mean(dim=2))


class _CanonicalRawDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(29)
        self.waveforms = torch.randn(4, 1000, 12, generator=generator)
        self.labels = torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"waveform": self.waveforms[index], "label": self.labels[index]}


class _TinyManagedECGFounder(torch.nn.Module):
    model_spec = ECGFOUNDER_SPEC

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(12, 5)
        self.forward_shapes: list[tuple[int, ...]] = []

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        self.forward_shapes.append(tuple(waveform.shape))
        return self.head(waveform.mean(dim=2))


class _IdentityInputAdapter:
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform

    def describe(self) -> dict[str, object]:
        return {"schema_version": 1, "id": "test_identity_adapter"}


def _loader(offset: float = 0.0) -> DataLoader:
    return DataLoader(
        _TinyECGDataset(offset),
        batch_size=4,
        shuffle=False,
    )


class _ValidationArtifactDataset(_TinyECGDataset):
    def __getitem__(self, index: int) -> dict[str, object]:
        result: dict[str, object] = dict(super().__getitem__(index))
        result["hash_id"] = f"validation_hash_{index:02d}"
        return result


class _ValidationArtifactLoader(DataLoader):
    def __init__(self, *, missing_class_index: int | None = None) -> None:
        dataset = _ValidationArtifactDataset(offset=0.1)
        if missing_class_index is not None:
            dataset.labels[:, int(missing_class_index)] = 0
        super().__init__(dataset, batch_size=4, shuffle=False)
        hashes = [f"validation_hash_{index:02d}" for index in range(len(dataset))]
        payload = "".join(f"{value}\n" for value in sorted(hashes)).encode("utf-8")
        self._hash_id_set_sha256 = hashlib.sha256(payload).hexdigest()

    def describe(self) -> dict[str, object]:
        return {
            "dataset": {
                "selection": {
                    "dataset": "pn2021",
                    "partition": "k500_tune_validation",
                    "logical_center": "ningbo",
                    "record_count": len(self.dataset),
                    "split_id": "tiny_tune_split",
                    "split_manifest_sha256": "a" * 64,
                    "source_manifest_sha256": "b" * 64,
                    "hash_id_set_sha256": self._hash_id_set_sha256,
                    "class_order": ["CD", "HYP", "MI", "NORM", "STTC"],
                    "mapping_version": "v7_super5_sjr_rgq_review_20260528",
                    "mapping_hash": "555ec85d5b51",
                }
            },
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": self.drop_last,
        }


def _small_config_bundle(
    tmp_path: Path,
    *,
    selection_metric: str,
    tensorboard_enabled: bool = False,
) -> Path:
    config_root = tmp_path / "configs"
    shutil.copytree(REPO / "configs", config_root)
    config_path = config_root / "train" / "PTBXL.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["profile_name"] = f"tiny_{selection_metric}_training"
    payload["training"]["epochs"] = 2
    payload["training"]["device"] = "cpu"
    payload["training"]["amp"]["enabled"] = False
    payload["logging"]["tensorboard"]["enabled"] = tensorboard_enabled
    payload["output"]["run_dir"] = str(tmp_path / f"run_{selection_metric}")
    if selection_metric == "last":
        payload["selection"].update(
            {
                "partition": "none",
                "metric": "last",
                "mode": "last",
                "evaluate_test_at_end": False,
            }
        )
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_train_config_resolves_seed_inside_selected_bundle(tmp_path):
    config_path = _small_config_bundle(tmp_path, selection_metric="macro_auprc")
    config = load_train_config(config_path)

    assert config.selection_metric == "macro_auprc"
    assert config.config_root == config_path.parents[1]
    assert config.random_seed_config_path == config_path.parents[1] / "random_seed.yaml"
    assert config.payload["output"]["run_dir"] == str(
        tmp_path / "run_macro_auprc"
    )


def test_matched_method_configs_share_training_seed(tmp_path):
    first_path = _small_config_bundle(tmp_path, selection_metric="last")
    first_payload = yaml.safe_load(first_path.read_text(encoding="utf-8"))
    second_path = first_path.with_name("PN2021_other_arm.yaml")
    second_payload = dict(first_payload)
    second_payload["profile_name"] = "different_method_arm_and_config_sha"
    second_payload["output"] = dict(first_payload["output"])
    second_payload["output"]["run_dir"] = str(tmp_path / "other_arm")
    second_path.write_text(
        yaml.safe_dump(second_payload, sort_keys=False),
        encoding="utf-8",
    )

    first = seed_training_process(load_train_config(first_path))
    second = seed_training_process(load_train_config(second_path))

    assert first.effective_seed == second.effective_seed
    assert first.identity == second.identity == (
        "ptbxl_super5_source_v1",
        "0",
    )
    assert load_train_config(first_path).sha256 != load_train_config(second_path).sha256


def test_train_model_with_validation_and_test_loaders(tmp_path):
    config_path = _small_config_bundle(tmp_path, selection_metric="macro_auprc")
    model = _TinyClassifier()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    result = train_model(
        model,
        _loader(),
        validation_dataloader=_loader(0.1),
        test_dataloader=_loader(0.2),
        config_path=config_path,
        training_parameters={"epochs": 1, "learning_rate": 0.0005},
    )

    assert result.model is model
    assert len(result.history) == 1
    assert result.selected_epoch == 1
    assert result.selected_metric is not None
    assert result.selected_checkpoint_path.is_file()
    assert result.last_checkpoint_path.is_file()
    assert (result.output_dir / "training_history.json").is_file()
    assert (result.output_dir / "train_result.json").is_file()
    assert result.test_metrics is not None
    assert result.test_metrics["sample_count"] == 8
    checkpoint = torch.load(
        result.selected_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["run_identity"]["training_parameters"] == {
            "resolved": {
                "epochs": 1,
                "scheduler_horizon_epochs": 1,
                "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "minimum_learning_rate_ratio": 0.01,
            "gradient_clip_norm": 1.0,
            "amp_enabled": False,
            "amp_dtype": "bfloat16",
        },
        "explicit_overrides": {"epochs": 1, "learning_rate": 0.0005},
    }
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )


def test_train_model_supports_last_epoch_selection_without_validation(tmp_path):
    config_path = _small_config_bundle(tmp_path, selection_metric="last")

    result = train_model(
        _TinyClassifier(),
        _loader(),
        config_path=config_path,
        pos_weight=[1.0, 2.0, 1.0, 1.0, 1.0],
    )

    assert result.selected_epoch == 2
    assert result.selected_metric is None
    assert result.test_metrics is None
    assert all(record["validation"] is None for record in result.history)


def test_trainer_applies_canonical_adapter_after_transfer_before_forward(tmp_path):
    config_path = _small_config_bundle(tmp_path, selection_metric="last")
    model = _TinyManagedECGFounder()
    adapter = CanonicalTuningInputAdapter(ECGFOUNDER_SPEC)

    result = train_model(
        model,
        DataLoader(_CanonicalRawDataset(), batch_size=2, shuffle=False),
        config_path=config_path,
        training_parameters={"epochs": 1},
        input_adapter=adapter,
    )

    assert model.forward_shapes == [(2, 12, 5000), (2, 12, 5000)]
    assert result.input_adapter_identity == adapter.describe()
    assert result.describe()["input_adapter"] == adapter.describe()
    checkpoint = torch.load(
        result.last_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["run_identity"]["input_adapter"] == adapter.describe()


def test_train_model_records_tensorboard_identity_and_closes_writer(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    config_path = _small_config_bundle(
        tmp_path,
        selection_metric="macro_auprc",
        tensorboard_enabled=True,
    )
    result = train_model(
        _TinyClassifier(),
        _loader(),
        validation_dataloader=_loader(0.1),
        config_path=config_path,
        training_parameters={"epochs": 1},
    )

    log_dir = result.output_dir / "tensorboard"
    assert result.logging_identity["enabled"] is True
    assert result.logging_identity["log_dir"] == str(log_dir)
    assert len(list(log_dir.glob("events.out.tfevents.*"))) == 1
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    assert "loss/train_epoch" in accumulator.Tags()["scalars"]
    assert "metrics/validation_macro_auprc" in accumulator.Tags()["scalars"]


def test_trainer_uses_shared_strict_raw_logit_metrics() -> None:
    targets = np.tile(
        np.asarray([0, 1, 0, 1], dtype=np.float32)[:, None],
        (1, 5),
    )
    logits = np.tile(
        np.asarray([17.0, 20.0, 18.0, 21.0], dtype=np.float32)[:, None],
        (1, 5),
    )

    result = _classification_metrics(
        logits,
        targets,
        ("CD", "HYP", "MI", "NORM", "STTC"),
    )

    assert result["macro_auroc"] == 1.0
    assert result["macro_auprc"] == 1.0
    assert result["undefined_class_policy"] == "strict"
    with pytest.raises(ValueError, match="class_order must be"):
        _classification_metrics(
            logits,
            targets,
            ("CD", "CD", "MI", "NORM", "STTC"),
        )


def test_train_model_rejects_noncanonical_class_order_without_validation(
    tmp_path,
) -> None:
    config_path = _small_config_bundle(tmp_path, selection_metric="last")

    with pytest.raises(ValueError, match="canonical Super5 order"):
        train_model(
            _TinyClassifier(),
            _loader(),
            config_path=config_path,
            class_names=("CD", "CD", "MI", "NORM", "STTC"),
        )


def test_validation_prediction_artifacts_are_external_to_history(tmp_path) -> None:
    config_path = _small_config_bundle(tmp_path, selection_metric="macro_auprc")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["validation_predictions"] = {
        "enabled": True,
        "subdir": "validation_predictions",
        "hash_key": "hash_id",
        "artifact_schema": "supervised_validation_predictions_v1",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    adapter = _IdentityInputAdapter()
    result = train_model(
        _TinyClassifier(),
        _loader(),
        validation_dataloader=_ValidationArtifactLoader(),
        config_path=config_path,
        training_parameters={"epochs": 1},
        input_adapter=adapter,
    )

    artifact_dir = result.output_dir / "validation_predictions"
    arrays_path = artifact_dir / "epoch_0001.npz"
    sidecar_path = artifact_dir / "epoch_0001.json"
    assert arrays_path.is_file()
    assert sidecar_path.is_file()
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {"logits", "targets", "hash_ids"}
        assert arrays["logits"].shape == (8, 5)
        assert arrays["logits"].dtype == np.float32
        assert arrays["targets"].dtype == np.uint8
        assert arrays["hash_ids"].dtype.kind == "U"
        assert len(set(arrays["hash_ids"].tolist())) == 8
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["artifact_type"] == "supervised_validation_predictions"
    assert sidecar["arrays_file"] == "epoch_0001.npz"
    assert sidecar["record_count"] == 8
    assert sidecar["validation_identity"]["partition"] == "k500_tune_validation"
    assert sidecar["metrics"]["metric_definition"]["score_input"] == "raw_logits"
    assert sidecar["comparison_identity"]["training_config_sha256"]
    assert sidecar["comparison_identity"]["input_adapter"] == adapter.describe()
    history = json.loads(
        (result.output_dir / "training_history.json").read_text(encoding="utf-8")
    )
    validation_history = history["epochs"][0]["validation"]
    assert "logits" not in validation_history
    assert "targets" not in validation_history
    assert "hash_ids" not in validation_history


def test_external_pooled_validation_exports_when_one_local_class_is_missing(
    tmp_path,
) -> None:
    config_path = _small_config_bundle(tmp_path, selection_metric="last")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["validation_predictions"] = {
        "enabled": True,
        "subdir": "validation_predictions",
        "hash_key": "hash_id",
        "artifact_schema": "supervised_validation_predictions_v1",
        "local_undefined_class_policy": "skip_undefined",
    }
    payload["selection"]["partition"] = "external_pooled_validation"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = train_model(
        _TinyClassifier(),
        _loader(),
        validation_dataloader=_ValidationArtifactLoader(missing_class_index=2),
        config_path=config_path,
        training_parameters={"epochs": 1},
    )

    sidecar = json.loads(
        (result.output_dir / "validation_predictions" / "epoch_0001.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["metrics"]["undefined_class_policy"] == "skip_undefined"
    assert "MI" not in sidecar["metrics"]["classes_used"]
    with np.load(
        result.output_dir / "validation_predictions" / "epoch_0001.npz",
        allow_pickle=False,
    ) as arrays:
        assert int(arrays["targets"][:, 2].sum()) == 0
    assert result.describe()["selection"]["policy"] == (
        "external_pooled_validation_pending"
    )
    assert result.describe()["selection"]["final_model_selected"] is False
