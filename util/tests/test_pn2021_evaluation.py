from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from boot_scripts.evaluate_pn2021 import _summary, main as evaluate_main
from models.checkpoints import CheckpointIdentity
from models.contracts import (
    CLASS_ORDER,
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
)
from util.evaluation.metrics import (
    CLEAN_VIEW_ALIASES,
    CORRUPTED_VIEW_ALIASES,
    METRIC_DEFINITION,
    aggregate_corruption_views,
)
from util.evaluation.pn2021 import (
    DEFAULT_PN2021_EVAL_CONFIG,
    LOGICAL_CENTERS,
    PN2021_MAPPING_HASH,
    PN2021_MAPPING_VERSION,
    evaluate_pn2021,
    load_corruption_views,
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
CLEAN_MANIFEST_SHA256 = "a" * 64
EXPECTED_PROFILE = "pn2021c_paper_anchored_s5_v1"


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


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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

    operators = [
        "powerline_noise",
        "emg_noise",
        "baseline_wander",
        "baseline_shift",
        "random_leads_masking",
    ]
    compositions: list[dict[str, object]] = []
    for depth in (2, 3):
        for selected in itertools.combinations(operators, depth):
            compositions.append(
                {
                    "view_index": len(compositions),
                    "depth": depth,
                    "operators": list(selected),
                    "composition_id": f"d{depth}__{'__'.join(selected)}",
                }
            )
    cache_root = tmp_path / "pn2021c"
    cache_root.mkdir()
    compositions_path = cache_root / "compositions.json"
    compositions_path.write_text(
        json.dumps(compositions, indent=2) + "\n", encoding="utf-8"
    )
    cache_manifest = {
        "schema_version": 1,
        "dataset": "pn2021c",
        "view_count": 20,
        "source": {
            "manifest_sha256": CLEAN_MANIFEST_SHA256,
            "mapping_version": PN2021_MAPPING_VERSION,
            "mapping_hash": PN2021_MAPPING_HASH,
            "class_order": list(CLASS_ORDER),
        },
        "files": {"compositions": "compositions.json"},
        "corruption": {
            "profile": EXPECTED_PROFILE,
            "severity": 5,
            "domain_sampling_rate_hz": 500,
        },
    }
    cache_manifest_path = cache_root / "manifest.json"
    cache_manifest_path.write_text(
        json.dumps(cache_manifest, indent=2) + "\n", encoding="utf-8"
    )
    cache_config_path = config_root / "augmentation" / "cache.yaml"
    _write_yaml(
        cache_config_path,
        {
            "corruption": {
                "profile": EXPECTED_PROFILE,
                "severity": 5,
                "domain_sampling_rate_hz": 500,
            },
            "output": {"cache_dir": str(cache_root)},
        },
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
        "source_manifest_sha256": CLEAN_MANIFEST_SHA256,
        "mapping_version": PN2021_MAPPING_VERSION,
        "mapping_hash": PN2021_MAPPING_HASH,
        "class_order": list(CLASS_ORDER),
        "logical_center_order": list(LOGICAL_CENTERS),
        "logical_centers": split_centers,
    }
    split_manifest_path = tmp_path / "split_manifest.json"
    split_manifest_path.write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )

    output_dir = tmp_path / "evaluation_output"
    evaluation_config: dict[str, object] = {
        "schema_version": 2,
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
                "expected_profile": EXPECTED_PROFILE,
                "expected_severity": 5,
                "expected_domain_sampling_rate_hz": 500,
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
        "artifact_locks": {
            "clean_cache_manifest_sha256": CLEAN_MANIFEST_SHA256,
            "split_manifest_sha256": _sha256(split_manifest_path),
            "corruption_cache_manifest_sha256": _sha256(cache_manifest_path),
            "compositions_sha256": _sha256(compositions_path),
        },
        "runtime": {
            "device": "cpu",
            "batch_size_by_model": {"efficientnet1dv2": 16, "ecgfounder": 4},
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
    _write_yaml(config_path, evaluation_config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"tiny checkpoint identity")
    return {
        "config_path": config_path,
        "config_root": config_root,
        "cache_config_path": cache_config_path,
        "cache_manifest_path": cache_manifest_path,
        "compositions_path": compositions_path,
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
        "split_manifest_path": split_manifest_path,
        "clean_manifest_sha256": CLEAN_MANIFEST_SHA256,
        "corruption_manifest_sha256": _sha256(cache_manifest_path),
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
    ref_excluded: bool = True,
    source_centers_override: dict[str, list[str]] | None = None,
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
        cache_shape = (
            [20, len(labels), cache_points, 12]
            if dataset == "pn2021c"
            else [len(labels), cache_points, 12]
        )
        default_sources = (
            ["cpsc_2018", "cpsc_2018_extra"]
            if logical_center == "cpsc_2018"
            else [logical_center]
        )
        source_centers = (
            default_sources
            if source_centers_override is None
            else source_centers_override.get(logical_center, default_sources)
        )
        identity = {
            "dataset": {
                "cache": {
                    "manifest_sha256": (
                        fixture["corruption_manifest_sha256"]
                        if dataset == "pn2021c"
                        else fixture["clean_manifest_sha256"]
                    ),
                    "dataset": "pn2021c" if dataset == "pn2021c" else "pn2021",
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
                    "source_manifest_sha256": CLEAN_MANIFEST_SHA256,
                    "hash_id_set_sha256": _hash_set(hashes),
                    "class_order": list(CLASS_ORDER),
                    "mapping_version": PN2021_MAPPING_VERSION,
                    "mapping_hash": PN2021_MAPPING_HASH,
                    "ref_excluded_evaluation": ref_excluded,
                },
                "view": view,
            }
        }
        loader = _TinyLoader(batch, identity)
        opened_loaders.append(loader)
        return loader

    return factory, opened_loaders


def test_pn2021_evaluation_dry_run_validates_locks_without_creating_output(
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
    assert plan["schema_version"] == 2
    assert plan["protocol"]["corruption_cache"]["artifact_locks_verified"] is True
    assert len(plan["protocol"]["corruption_views"]) == 20
    assert not Path(fixture["output_dir"]).exists()


@pytest.mark.parametrize(
    ("artifact", "field", "value", "message"),
    [
        ("config", "profile", "wrong_profile", "cache config corruption semantics"),
        ("config", "severity", 4, "cache config corruption semantics"),
        ("config", "domain_sampling_rate_hz", 100, "cache config corruption semantics"),
        ("manifest", "profile", "wrong_profile", "manifest corruption semantics"),
        ("manifest", "severity", 4, "manifest corruption semantics"),
        ("manifest", "domain_sampling_rate_hz", 100, "manifest corruption semantics"),
    ],
)
def test_corruption_semantics_are_three_way_fail_closed(
    tmp_path: Path,
    artifact: str,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    target = Path(
        fixture["cache_config_path"]
        if artifact == "config"
        else fixture["cache_manifest_path"]
    )
    if artifact == "config":
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
        payload["corruption"][field] = value
        _write_yaml(target, payload)
    else:
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["corruption"][field] = value
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        config_path = Path(fixture["config_path"])
        config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config_payload["artifact_locks"][
            "corruption_cache_manifest_sha256"
        ] = _sha256(target)
        _write_yaml(config_path, config_payload)

    config = load_pn2021_eval_config(
        fixture["config_path"], config_root=fixture["config_root"]
    )
    with pytest.raises(ValueError, match=message):
        load_corruption_views(config)


def test_corruption_compositions_artifact_lock_rejects_byte_drift(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    compositions_path = Path(fixture["compositions_path"])
    compositions_path.write_text(
        compositions_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    config = load_pn2021_eval_config(
        fixture["config_path"], config_root=fixture["config_root"]
    )

    with pytest.raises(ValueError, match="compositions differ from the locked artifact"):
        load_corruption_views(config)


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("clean_cache", "pn2021 cache manifest differs from the locked artifact"),
        ("split", "evaluation split manifest SHA256 mismatch"),
    ],
)
def test_loader_artifact_locks_reject_clean_or_split_drift(
    tmp_path: Path,
    artifact: str,
    message: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    if artifact == "clean_cache":
        fixture["clean_manifest_sha256"] = "b" * 64
    else:
        split_path = Path(fixture["split_manifest_path"])
        split_path.write_text(
            split_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    config = load_pn2021_eval_config(
        fixture["config_path"], config_root=fixture["config_root"]
    )
    factory, opened_loaders = _loader_factory(fixture)

    with pytest.raises(ValueError, match=message):
        evaluate_pn2021(
            _TinyModel(),
            _checkpoint_identity(fixture),
            config=config,
            logical_centers=["ningbo"],
            dataloader_factory=factory,
        )
    assert len(opened_loaders) == 1 and opened_loaders[0].closed


def test_pn2021_evaluation_runs_four_centers_with_truthful_aggregates(
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
        "util.evaluation.pn2021.SequentialEvaluationDataSession", session_factory
    )
    result = evaluate_pn2021(
        _TinyModel(),
        _checkpoint_identity(fixture),
        config=config,
    )

    assert result["schema_version"] == 2
    assert len(sessions) == 1 and sessions[0].closed
    assert sessions[0].close_calls == 1
    assert len(sessions[0].requests) == 4 + 20 * 4
    assert len(opened_loaders) == 4 + 20 * 4
    assert all(loader.closed for loader in opened_loaders)
    clean = result["clean"]
    assert clean["center_count"] == 4
    assert clean["center_order"] == list(LOGICAL_CENTERS)
    assert clean["four_center_mean"] == clean["evaluated_center_mean"]
    for alias in CLEAN_VIEW_ALIASES:
        assert clean["evaluated_center_mean"][alias]["macro_auroc"] == 1.0
        assert clean["evaluated_center_mean"][alias]["macro_auprc"] == 1.0
    aggregates = result["corrupted"]["aggregates"]
    assert aggregates["depth2"]["view_count"] == 10
    assert aggregates["depth3"]["view_count"] == 10
    assert aggregates["depth23"]["view_count"] == 20
    for aggregate in aggregates.values():
        assert aggregate["center_count"] == 4
        assert aggregate["center_order"] == list(LOGICAL_CENTERS)
        assert aggregate["four_center_mean"] == aggregate["evaluated_center_mean"]
    for alias in CORRUPTED_VIEW_ALIASES:
        payload = aggregates["depth23"]["evaluated_center_mean"][alias]
        assert payload["macro_auroc"] == 1.0
        assert payload["macro_auprc"] == 1.0
        assert payload["evaluation_slice"] == "depth23"


def test_single_center_result_never_claims_a_four_center_mean(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    config = load_pn2021_eval_config(
        fixture["config_path"], config_root=fixture["config_root"]
    )
    factory, opened_loaders = _loader_factory(fixture)
    result = evaluate_pn2021(
        _TinyModel(),
        _checkpoint_identity(fixture),
        config=config,
        logical_centers=["ningbo"],
        dataloader_factory=factory,
    )

    assert len(opened_loaders) == 21
    assert all(loader.closed for loader in opened_loaders)
    clean = result["clean"]
    assert clean["center_count"] == 1
    assert clean["center_order"] == ["ningbo"]
    assert clean["evaluated_center_mean"]
    assert clean["four_center_mean"] is None
    for view in result["corrupted"]["per_view"]:
        assert view["center_count"] == 1
        assert view["center_order"] == ["ningbo"]
        assert view["evaluated_center_mean"]
        assert view["four_center_mean"] is None
    for aggregate in result["corrupted"]["aggregates"].values():
        assert aggregate["center_count"] == 1
        assert aggregate["center_order"] == ["ningbo"]
        assert aggregate["evaluated_center_mean"]
        assert aggregate["four_center_mean"] is None


def test_cli_summary_reads_schema1_four_center_field_as_generic_center_mean() -> None:
    legacy_mean = {"legacy_view": {"macro_auroc": 0.5}}
    legacy_result = {
        "status": "complete",
        "evaluation_profile": "legacy",
        "model": {"name": "efficientnet1dv2"},
        "checkpoint": {"sha256": "a" * 64},
        "protocol": {"logical_centers": ["ningbo"]},
        "clean": {"four_center_mean": legacy_mean},
        "corrupted": {
            "aggregates": {
                "depth2": {"four_center_mean": legacy_mean},
                "depth3": {"four_center_mean": legacy_mean},
                "depth23": {"four_center_mean": legacy_mean},
            }
        },
        "output": {},
    }

    summary = _summary(legacy_result)

    assert summary["center_count"] == 1
    assert summary["center_order"] == ["ningbo"]
    assert summary["clean_evaluated_center_mean"] == legacy_mean
    assert summary["pn2021c_depth23_evaluated_center_mean"] == legacy_mean
    assert summary["clean_four_center_mean"] is None
    assert summary["pn2021c_depth2_four_center_mean"] is None
    assert summary["pn2021c_depth3_four_center_mean"] is None
    assert summary["pn2021c_depth23_four_center_mean"] is None


def test_cli_summary_requires_exact_canonical_order_for_four_center_alias() -> None:
    legacy_mean = {"legacy_view": {"macro_auroc": 0.5}}
    wrong_order = list(reversed(LOGICAL_CENTERS))
    legacy_result = {
        "status": "complete",
        "evaluation_profile": "legacy",
        "model": {"name": "efficientnet1dv2"},
        "checkpoint": {"sha256": "a" * 64},
        "protocol": {"logical_centers": wrong_order},
        "clean": {"four_center_mean": legacy_mean},
        "corrupted": {
            "aggregates": {
                "depth2": {"four_center_mean": legacy_mean},
                "depth3": {"four_center_mean": legacy_mean},
                "depth23": {"four_center_mean": legacy_mean},
            }
        },
        "output": {},
    }

    summary = _summary(legacy_result)

    assert summary["center_count"] == 4
    assert summary["center_order"] == wrong_order
    assert summary["clean_evaluated_center_mean"] == legacy_mean
    assert summary["clean_four_center_mean"] is None
    assert summary["pn2021c_depth2_four_center_mean"] is None
    assert summary["pn2021c_depth3_four_center_mean"] is None
    assert summary["pn2021c_depth23_four_center_mean"] is None


def _fake_corrupted_views(value: float, sample_count: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for alias, policy in zip(CORRUPTED_VIEW_ALIASES, ("kept", "drop"), strict=True):
        result[alias] = {
            "canonical_view": alias,
            "all_zero_policy": policy,
            "sample_count": sample_count,
            "all_zero_count_in_source_selection": 0,
            "metric_definition": dict(METRIC_DEFINITION),
            "undefined_class_policy": "strict",
            "class_order": list(CLASS_ORDER),
            "classes_used": list(CLASS_ORDER),
            "n_classes_used": len(CLASS_ORDER),
            "n_classes_total": len(CLASS_ORDER),
            "macro_auroc": value,
            "macro_auprc": value,
            "per_class": {
                class_name: {
                    "positives": 1,
                    "negatives": sample_count - 1,
                    "defined": True,
                    "undefined_reason": None,
                    "auroc": value,
                    "auprc": value,
                }
                for class_name in CLASS_ORDER
            },
        }
    return result


def test_depth23_uses_equal_views_then_equal_centers_not_sample_pooling() -> None:
    center_values = dict(zip(LOGICAL_CENTERS, (0.1, 0.2, 0.3, 0.8), strict=True))
    center_sizes = dict(zip(LOGICAL_CENTERS, (10, 100, 1000, 10000), strict=True))
    per_view = []
    for view_index in range(20):
        depth = 2 if view_index < 10 else 3
        view_offset = view_index / 1000.0
        per_view.append(
            {
                "view_index": view_index,
                "depth": depth,
                "composition_id": f"view-{view_index}",
                "per_center": {
                    center: {
                        "metrics": _fake_corrupted_views(
                            center_values[center] + view_offset,
                            center_sizes[center],
                        )
                    }
                    for center in LOGICAL_CENTERS
                },
            }
        )

    aggregates = aggregate_corruption_views(
        per_view,
        center_order=LOGICAL_CENTERS,
        canonical_four_center_order=LOGICAL_CENTERS,
    )
    expected = np.mean(list(center_values.values())) + np.mean(np.arange(20)) / 1000
    payload = aggregates["depth23"]["evaluated_center_mean"]

    assert aggregates["depth2"]["view_count"] == 10
    assert aggregates["depth3"]["view_count"] == 10
    assert aggregates["depth23"]["view_count"] == 20
    assert aggregates["depth23"]["center_count"] == 4
    assert aggregates["depth23"]["four_center_mean"] == payload
    for alias in CORRUPTED_VIEW_ALIASES:
        assert payload[alias]["macro_auroc"] == pytest.approx(expected)
        assert payload[alias]["macro_auprc"] == pytest.approx(expected)
        assert payload[alias]["evaluation_slice"] == "depth23"


def test_evaluation_rejects_non_ref_excluded_selection(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    config = load_pn2021_eval_config(
        fixture["config_path"], config_root=fixture["config_root"]
    )
    factory, opened_loaders = _loader_factory(fixture, ref_excluded=False)

    with pytest.raises(ValueError, match="not K500-ref-excluded"):
        evaluate_pn2021(
            _TinyModel(),
            _checkpoint_identity(fixture),
            config=config,
            logical_centers=["ningbo"],
            dataloader_factory=factory,
        )
    assert len(opened_loaders) == 1 and opened_loaders[0].closed


def test_evaluation_requires_cpsc_main_and_extra_sources(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    config = load_pn2021_eval_config(
        fixture["config_path"], config_root=fixture["config_root"]
    )
    factory, opened_loaders = _loader_factory(
        fixture,
        source_centers_override={"cpsc_2018": ["cpsc_2018"]},
    )

    with pytest.raises(ValueError, match="physical source centers mismatch"):
        evaluate_pn2021(
            _TinyModel(),
            _checkpoint_identity(fixture),
            config=config,
            logical_centers=["cpsc_2018"],
            dataloader_factory=factory,
        )
    assert len(opened_loaders) == 1 and opened_loaders[0].closed


def test_ecgfounder_uses_canonical100_adapter_for_all_views(tmp_path: Path) -> None:
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
    assert set(model.forward_shapes) == {(12, 12, 5000)}
    assert model.first_input is not None
    records = fixture["center_records"]
    assert isinstance(records, dict)
    labels = np.asarray(records["ningbo"]["labels"], dtype=np.float32)
    raw_100hz_btc = torch.from_numpy(_raw_waveforms(labels))
    model_domain_btc = F.interpolate(
        raw_100hz_btc.transpose(1, 2),
        size=5000,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
    mean = model_domain_btc.mean(dim=(1, 2), keepdim=True)
    scale = model_domain_btc.std(
        dim=(1, 2), correction=0, keepdim=True
    ).clamp_min(1e-6)
    expected = ((model_domain_btc - mean) / scale).transpose(1, 2).contiguous()
    torch.testing.assert_close(model.first_input, expected)
    assert result["clean"]["four_center_mean"] is None


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
    assert len(opened_loaders) == 1 and opened_loaders[0].closed


def test_default_eval_config_uses_canonical_profile_and_artifact_locks() -> None:
    assert DEFAULT_PN2021_EVAL_CONFIG == (
        Path(__file__).resolve().parents[2] / "configs" / "eval" / "PN2021.yaml"
    )
    config = load_pn2021_eval_config(DEFAULT_PN2021_EVAL_CONFIG)
    assert config.payload["schema_version"] == 2
    corruption = config.payload["protocol"]["corruption"]
    assert corruption["expected_profile"] == EXPECTED_PROFILE
    assert corruption["expected_severity"] == 5
    assert corruption["expected_domain_sampling_rate_hz"] == 500
