from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from boot_scripts.evaluate_pn2021 import main as evaluate_main
from models.checkpoints import CheckpointIdentity
from models.contracts import (
    CLASS_ORDER,
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
)
from util.evaluation.metrics import (
    CLEAN_VIEW_ALIASES,
    CORRUPTED_VIEW_ALIASES,
)
from util.evaluation.pn2021 import (
    DEFAULT_PN2021_EVAL_CONFIG,
    LOGICAL_CENTERS,
    PN2021_MAPPING_HASH,
    PN2021_MAPPING_VERSION,
    evaluate_pn2021,
    load_pn2021_eval_config,
)


LEAD_ORDER = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_set(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _labels() -> np.ndarray:
    labels = np.zeros((12, 5), dtype=np.float32)
    for index in range(10):
        labels[index, index % 5] = 1.0
        labels[index, (index + 2) % 5] = 1.0
    labels[10] = 1.0
    return labels


def _write_fixture(tmp_path: Path) -> dict[str, object]:
    config_root = tmp_path / "configs"
    for section in ("eval", "data", "augmentation"):
        (config_root / section).mkdir(parents=True, exist_ok=True)
    (config_root / "random_seed.yaml").write_text(
        "schema_version: 1\nrandom_seed: 20260501\n",
        encoding="utf-8",
    )
    (config_root / "data" / "data_load.yaml").write_text(
        "schema_version: 1\n",
        encoding="utf-8",
    )
    (config_root / "data" / "splits.yaml").write_text(
        "schema_version: 1\n",
        encoding="utf-8",
    )

    compositions = []
    operators = [
        "powerline_noise",
        "emg_noise",
        "baseline_wander",
        "baseline_shift",
        "random_leads_masking",
    ]
    for view_index in range(20):
        depth = 2 if view_index < 10 else 3
        chosen = [operators[(view_index + offset) % len(operators)] for offset in range(depth)]
        compositions.append(
            {
                "view_index": view_index,
                "depth": depth,
                "operators": chosen,
                "composition_id": f"view_{view_index:02d}_depth{depth}",
            }
        )
    cache_root = tmp_path / "pn2021c"
    cache_root.mkdir()
    compositions_path = cache_root / "compositions.json"
    compositions_path.write_text(json.dumps(compositions), encoding="utf-8")
    cache_manifest = {
        "schema_version": 1,
        "dataset": "pn2021c",
        "view_count": 20,
        "source": {
            "mapping_version": PN2021_MAPPING_VERSION,
            "mapping_hash": PN2021_MAPPING_HASH,
            "class_order": list(CLASS_ORDER),
        },
        "files": {"compositions": "compositions.json"},
        "corruption": {
            "profile": "tiny_test",
            "severity": 5,
            "domain_sampling_rate_hz": 500,
        },
    }
    (cache_root / "manifest.json").write_text(
        json.dumps(cache_manifest), encoding="utf-8"
    )
    (config_root / "augmentation" / "cache.yaml").write_text(
        yaml.safe_dump({"output": {"cache_dir": str(cache_root)}}),
        encoding="utf-8",
    )

    labels = _labels()
    center_records: dict[str, dict[str, object]] = {}
    split_centers: dict[str, object] = {}
    for center in LOGICAL_CENTERS:
        hashes = [f"{center}_hash_{index:02d}" for index in range(len(labels))]
        records = [f"{center}_record_{index:02d}" for index in range(len(labels))]
        nonzero_hashes = [
            value
            for value, label in zip(hashes, labels, strict=True)
            if bool(label.sum())
        ]
        center_records[center] = {
            "hashes": hashes,
            "records": records,
            "labels": labels.copy(),
        }
        split_centers[center] = {
            "k500_count": 500,
            "k500_hash_id_set_sha256": f"k500_{center}",
            "evaluation_all_zero_kept_count": len(hashes),
            "evaluation_all_zero_kept_hash_id_set_sha256": _hash_set(hashes),
            "evaluation_drop_all_zero_count": len(nonzero_hashes),
            "evaluation_drop_all_zero_hash_id_set_sha256": _hash_set(
                nonzero_hashes
            ),
        }
    split_manifest = {
        "schema_version": 1,
        "dataset": "pn2021",
        "split_id": "tiny_k500_split",
        "source_manifest_sha256": "tiny_source_manifest",
        "mapping_version": PN2021_MAPPING_VERSION,
        "mapping_hash": PN2021_MAPPING_HASH,
        "class_order": list(CLASS_ORDER),
        "logical_center_order": list(LOGICAL_CENTERS),
        "logical_centers": split_centers,
    }
    split_manifest_path = tmp_path / "split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest), encoding="utf-8")

    output_dir = tmp_path / "evaluation_output"
    evaluation_config = {
        "schema_version": 1,
        "profile_name": "tiny_pn2021_eval",
        "references": {
            "split_config": "data/splits.yaml",
            "data_load_config": "data/data_load.yaml",
            "corruption_cache_config": "augmentation/cache.yaml",
            "random_seed_config": "random_seed.yaml",
        },
        "protocol": {
            "mapping_version": PN2021_MAPPING_VERSION,
            "mapping_hash": PN2021_MAPPING_HASH,
            "class_order": list(CLASS_ORDER),
            "logical_centers": list(LOGICAL_CENTERS),
            "reference_count_per_center": 500,
            "partitions": {
                "clean": CLEAN_VIEW_ALIASES[0],
                "corrupted": CORRUPTED_VIEW_ALIASES[0],
            },
            "canonical_views": {
                "clean_kept": CLEAN_VIEW_ALIASES[0],
                "clean_drop": CLEAN_VIEW_ALIASES[1],
                "corrupted_kept": CORRUPTED_VIEW_ALIASES[0],
                "corrupted_drop": CORRUPTED_VIEW_ALIASES[1],
            },
            "corruption": {
                "expected_view_count": 20,
                "depths": [2, 3],
                "combinations_per_depth": {2: 10, 3: 10},
                "aggregation": "equal_views_then_equal_centers",
            },
            "input_pipeline": {
                "canonical_domain": {
                    "sampling_rate_hz": 100,
                    "duration_seconds": 10.0,
                    "points": 1000,
                    "channels": 12,
                    "layout": "time_channel",
                    "physical_unit": "mV",
                    "normalization": "none",
                },
                "sanitization": {
                    "stage": "before_model_domain_adaptation",
                    "nonfinite_replacement": "zero",
                },
                "model_domains": {
                    "efficientnet1dv2": {
                        "sampling_rate_hz": 100,
                        "points": 1000,
                        "adaptation": "identity",
                    },
                    "ecgfounder": {
                        "sampling_rate_hz": 500,
                        "points": 5000,
                        "adaptation": "interpolate",
                        "mode": "linear",
                        "align_corners": True,
                    },
                },
                "normalization": {
                    "stage": "after_model_domain_adaptation",
                    "method": "per_sample_global_zscore",
                    "epsilon": 1e-6,
                    "variance_correction": 0,
                },
                "output": {"layout": "channel_time", "dtype": "float32"},
            },
        },
        "runtime": {
            "device": "cpu",
            "batch_size_by_model": {
                "efficientnet1dv2": 16,
                "ecgfounder": 4,
            },
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "cache_mode": "mmap",
            "validate_values": "sample",
            "amp": {"enabled": False, "dtype": "bfloat16"},
        },
        "output": {
            "run_dir": str(output_dir),
            "if_exists": "error",
            "result_file": "evaluation_result.json",
        },
    }
    config_path = config_root / "eval" / "PN2021.yaml"
    config_path.write_text(
        yaml.safe_dump(evaluation_config, sort_keys=False), encoding="utf-8"
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"tiny checkpoint identity")
    return {
        "config_path": config_path,
        "config_root": config_root,
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
        "split_manifest_path": split_manifest_path,
        "center_records": center_records,
        "composition_ids": {item["composition_id"] for item in compositions},
    }


class _TinyModel(nn.Module):
    model_spec = EFFICIENTNET1DV2_SPEC

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform[:, :5].mean(dim=-1)


class _TinyECGFounderModel(nn.Module):
    model_spec = ECGFOUNDER_SPEC

    def __init__(self) -> None:
        super().__init__()
        self.forward_shapes: list[tuple[int, ...]] = []
        self.first_input: torch.Tensor | None = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        self.forward_shapes.append(tuple(waveform.shape))
        if self.first_input is None:
            self.first_input = waveform.detach().cpu().clone()
        return waveform[:, :5].mean(dim=-1)


class _TinyLoader:
    def __init__(self, batch: dict[str, object], identity: dict[str, object]) -> None:
        self.batch = batch
        self.identity = identity
        self.closed = False

    def __iter__(self):
        yield self.batch

    def describe(self) -> dict[str, object]:
        return self.identity

    def close(self) -> None:
        self.closed = True


class _TinySequentialEvaluationSession:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.requests: list[dict[str, object]] = []
        self.closed = False
        self.close_calls = 0

    def get_dataloader(self, **kwargs):
        if self.closed:
            raise RuntimeError("session is closed")
        self.requests.append(
            {
                "dataset": kwargs["dataset"],
                "logical_center": kwargs["logical_center"],
                "view": kwargs["view"],
            }
        )
        return self.factory(**kwargs)

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


def _raw_waveforms(labels: np.ndarray, *, points: int = 1000) -> np.ndarray:
    trend = np.linspace(-0.25, 0.25, points, dtype=np.float32)
    waveform = np.full((len(labels), points, 12), -4.0, dtype=np.float32)
    waveform += trend[None, :, None]
    for class_index in range(5):
        class_level = np.where(labels[:, class_index] == 1.0, 4.0, -4.0)
        waveform[:, :, class_index] = class_level[:, None] + trend[None, :]
    return waveform


def _checkpoint_identity(fixture: dict[str, object]) -> CheckpointIdentity:
    checkpoint = Path(fixture["checkpoint_path"])
    return CheckpointIdentity(
        path=checkpoint,
        sha256=_sha256(checkpoint),
        state_key_count=1,
        missing_keys=(),
        unexpected_keys=(),
    )


def _loader_factory(
    fixture: dict[str, object],
    *,
    cache_sampling_rate_hz: int = 100,
    cache_points: int = 1000,
):
    split_path = Path(fixture["split_manifest_path"])
    split_sha = _sha256(split_path)
    records_by_center = fixture["center_records"]
    assert isinstance(records_by_center, dict)
    opened_loaders: list[_TinyLoader] = []

    def factory(
        *,
        dataset: str,
        partition: str,
        logical_center: str,
        view,
        **kwargs,
    ):
        assert kwargs["sampling_rate_hz"] == 100
        assert kwargs["prepare_for_model"] is False
        assert kwargs["sanitize"] is False
        assert kwargs["global_zscore"] is False
        assert kwargs["output_layout"] == "time_channel"
        assert kwargs["shuffle"] is False
        assert kwargs["drop_last"] is False
        assert partition == (
            CORRUPTED_VIEW_ALIASES[0]
            if dataset == "pn2021c"
            else CLEAN_VIEW_ALIASES[0]
        )
        record_data = records_by_center[logical_center]
        labels = np.asarray(record_data["labels"], dtype=np.float32)
        hashes = list(record_data["hashes"])
        records = list(record_data["records"])
        waveform = _raw_waveforms(labels, points=cache_points)
        composition_id = "" if view is None else str(view)
        batch = {
            "waveform": torch.from_numpy(waveform),
            "label": torch.from_numpy(labels),
            "cache_index": torch.arange(len(labels), dtype=torch.int64),
            "record_id": records,
            "hash_id": hashes,
            "sampling_rate_hz": torch.full(
                (len(labels),), cache_sampling_rate_hz, dtype=torch.int64
            ),
            "composition_id": [composition_id] * len(labels),
        }
        cache_dataset = "pn2021c" if dataset == "pn2021c" else "pn2021"
        cache_shape = (
            [20, len(labels), cache_points, 12]
            if dataset == "pn2021c"
            else [len(labels), cache_points, 12]
        )
        source_centers = (
            ["cpsc_2018", "cpsc_2018_extra"]
            if logical_center == "cpsc_2018"
            else [logical_center]
        )
        identity = {
            "dataset": {
                "cache": {
                    "dataset": cache_dataset,
                    "view_count": 20 if dataset == "pn2021c" else 1,
                    "sampling_rate_hz": cache_sampling_rate_hz,
                    "duration_seconds": 10.0,
                    "signal_shape": cache_shape,
                    "lead_order": LEAD_ORDER,
                    "class_order": list(CLASS_ORDER),
                    "mapping_version": PN2021_MAPPING_VERSION,
                    "mapping_hash": PN2021_MAPPING_HASH,
                    "physical_unit": "mV",
                    "normalization": "none",
                },
                "selection": {
                    "partition": "evaluation_all_zero_kept",
                    "logical_center": logical_center,
                    "source_centers": source_centers,
                    "record_count": len(labels),
                    "split_id": "tiny_k500_split",
                    "split_manifest_path": str(split_path),
                    "split_manifest_sha256": split_sha,
                    "source_manifest_sha256": "tiny_source_manifest",
                    "hash_id_set_sha256": _hash_set(hashes),
                    "class_order": list(CLASS_ORDER),
                    "mapping_version": PN2021_MAPPING_VERSION,
                    "mapping_hash": PN2021_MAPPING_HASH,
                    "ref_excluded_evaluation": True,
                },
                "view": view,
            }
        }
        loader = _TinyLoader(batch, identity)
        opened_loaders.append(loader)
        return loader

    return factory, opened_loaders


def test_pn2021_evaluation_dry_run_does_not_create_output(
    tmp_path: Path, capsys
) -> None:
    fixture = _write_fixture(tmp_path)
    assert evaluate_main(
        [
            "--model",
            "efficientnet1dv2",
            "--checkpoint",
            str(fixture["checkpoint_path"]),
            "--config",
            str(fixture["config_path"]),
            "--config-root",
            str(fixture["config_root"]),
            "--device",
            "cpu",
            "--dry-run",
        ]
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["protocol"]["corruption_cache"]["view_count"] == 20
    assert len(plan["protocol"]["corruption_views"]) == 20
    assert plan["protocol"]["input_pipeline"]["canonical_domain"] == {
        "sampling_rate_hz": 100,
        "duration_seconds": 10.0,
        "points": 1000,
        "channels": 12,
        "layout": "time_channel",
        "physical_unit": "mV",
        "normalization": "none",
    }
    assert plan["protocol"]["input_pipeline"]["model_domain"]["model"] == (
        "efficientnet1dv2"
    )
    assert not Path(fixture["output_dir"]).exists()


def test_pn2021_evaluation_runs_all_clean_and_corruption_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    config = load_pn2021_eval_config(
        fixture["config_path"], config_root=fixture["config_root"]
    )
    factory, opened_loaders = _loader_factory(fixture)
    sessions: list[_TinySequentialEvaluationSession] = []

    def session_factory() -> _TinySequentialEvaluationSession:
        session = _TinySequentialEvaluationSession(factory)
        sessions.append(session)
        return session

    monkeypatch.setattr(
        "util.evaluation.pn2021.SequentialEvaluationDataSession",
        session_factory,
    )
    checkpoint_identity = _checkpoint_identity(fixture)
    result = evaluate_pn2021(
        _TinyModel(),
        checkpoint_identity,
        config=config,
    )

    assert len(sessions) == 1
    assert sessions[0].closed is True
    assert sessions[0].close_calls == 1
    assert len(sessions[0].requests) == 4 + 20 * 4
    assert [request["view"] for request in sessions[0].requests[:4]] == [None] * 4
    corrupted_requests = sessions[0].requests[4:]
    assert {request["view"] for request in corrupted_requests} == fixture[
        "composition_ids"
    ]
    assert all(request["view"] is not None for request in corrupted_requests)
    assert len(opened_loaders) == 4 + 20 * 4
    assert all(loader.closed for loader in opened_loaders)
    assert result["runtime"]["num_workers"] == 0
    assert result["runtime"]["persistent_workers"] is False
    assert len(result["corrupted"]["per_view"]) == 20
    aggregates = result["corrupted"]["aggregates"]
    assert aggregates["depth2"]["view_count"] == 10
    assert aggregates["depth3"]["view_count"] == 10
    assert aggregates["depth23"]["view_count"] == 20
    for alias in CLEAN_VIEW_ALIASES:
        assert result["clean"]["four_center_mean"][alias]["macro_auroc"] == 1.0
        assert result["clean"]["four_center_mean"][alias]["macro_auprc"] == 1.0
    for alias in CORRUPTED_VIEW_ALIASES:
        payload = aggregates["depth23"]["four_center_mean"][alias]
        assert payload["macro_auroc"] == 1.0
        assert payload["macro_auprc"] == 1.0
        assert payload["evaluation_slice"] == "depth23"
    result_path = Path(result["output"]["result_file"])
    assert result_path.is_file()
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["checkpoint"]["sha256"] == checkpoint_identity.sha256
    assert persisted["protocol"]["input_pipeline"]["canonical_domain"][
        "sampling_rate_hz"
    ] == 100


def test_default_eval_config_uses_canonical_profile() -> None:
    assert DEFAULT_PN2021_EVAL_CONFIG == (
        Path(__file__).resolve().parents[2] / "configs" / "eval" / "PN2021.yaml"
    )


def test_ecgfounder_evaluation_uses_canonical100_adapter_for_all_views(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    config = load_pn2021_eval_config(
        fixture["config_path"], config_root=fixture["config_root"]
    )
    factory, opened_loaders = _loader_factory(fixture)
    model = _TinyECGFounderModel()
    result = evaluate_pn2021(
        model,
        _checkpoint_identity(fixture),
        config=config,
        logical_centers=["ningbo"],
        dataloader_factory=factory,
    )

    assert len(opened_loaders) == 21
    assert all(loader.closed for loader in opened_loaders)
    assert len(model.forward_shapes) == 21
    assert set(model.forward_shapes) == {(12, 12, 5000)}
    assert model.first_input is not None

    records = fixture["center_records"]
    assert isinstance(records, dict)
    labels = np.asarray(records["ningbo"]["labels"], dtype=np.float32)
    raw_100hz_btc = torch.from_numpy(_raw_waveforms(labels))
    model_domain_bct = F.interpolate(
        raw_100hz_btc.transpose(1, 2),
        size=5000,
        mode="linear",
        align_corners=True,
    )
    model_domain_btc = model_domain_bct.transpose(1, 2)
    mean = model_domain_btc.mean(dim=(1, 2), keepdim=True)
    scale = model_domain_btc.std(
        dim=(1, 2), correction=0, keepdim=True
    ).clamp_min(1e-6)
    expected = ((model_domain_btc - mean) / scale).transpose(1, 2).contiguous()
    torch.testing.assert_close(model.first_input, expected)
    assert result["protocol"]["input_pipeline"]["model_domain"] == {
        "model": "ecgfounder",
        "sampling_rate_hz": 500,
        "points": 5000,
        "adaptation": "interpolate",
        "mode": "linear",
        "align_corners": True,
    }


def test_ecgfounder_evaluation_rejects_500hz_cache(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    config = load_pn2021_eval_config(
        fixture["config_path"], config_root=fixture["config_root"]
    )
    factory, opened_loaders = _loader_factory(
        fixture,
        cache_sampling_rate_hz=500,
        cache_points=5000,
    )
    with pytest.raises(ValueError, match="canonical 100 Hz"):
        evaluate_pn2021(
            _TinyECGFounderModel(),
            _checkpoint_identity(fixture),
            config=config,
            logical_centers=["ningbo"],
            dataloader_factory=factory,
        )
    assert len(opened_loaders) == 1
    assert opened_loaders[0].closed
