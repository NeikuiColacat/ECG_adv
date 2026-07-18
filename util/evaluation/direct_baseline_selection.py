"""Four-center robust epoch selection for managed PN2021 K500 methods."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
import yaml

from models.contracts import CLASS_ORDER
from util.config_bundle import config_bundle_root, resolve_config_reference, resolve_entry_config_path
from util.evaluation.metrics import compute_classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECT_TUNE_CONFIG = PROJECT_ROOT / "configs" / "train" / "PN2021_direct_tune.yaml"
ARTIFACT_TYPE = "pn2021_family_balanced_validation_predictions"
ARTIFACT_SCHEMA = "pn2021_family_balanced_validation_predictions"


def locked_pooled_selection_rule() -> dict[str, Any]:
    """Return the single paper-facing pooled K500 checkpoint rule."""

    return {
        "training_partition": "k500_tune_train",
        "validation_partition": "k500_tune_validation",
        "clean_aggregation": "concatenate_four_centers_before_metric",
        "corrupted_aggregation": "concatenate_four_centers_per_composition_before_metric",
        "robust_aggregation": "mean_of_20_composition_macro_auprc",
        "score": "0.5_clean_macro_auprc_plus_0.5_robust_macro_auprc",
        "clean_floor": "same_backbone_locked_clean_macro_auprc_minus_0.01",
        "metric_definition": "sklearn_average_precision",
        "input_type": "raw_logits",
        "strict_all_five_classes": True,
        "tie_break": "earliest_epoch_on_exact_tie",
        "heldout_evaluation_used": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_lines(values: Sequence[str], *, sort: bool) -> str:
    normalized = [str(value) for value in values]
    if sort:
        normalized.sort()
    return hashlib.sha256(
        "".join(f"{value}\n" for value in normalized).encode("utf-8")
    ).hexdigest()


def _hash_label_set_sha256(hash_ids: Sequence[str], targets: np.ndarray) -> str:
    rows = sorted(
        (str(hash_id), "".join(str(int(value)) for value in target))
        for hash_id, target in zip(hash_ids, targets, strict=True)
    )
    return hashlib.sha256(
        "".join(f"{hash_id}|{label}\n" for hash_id, label in rows).encode("utf-8")
    ).hexdigest()


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _positive_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return int(value)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description}: {exc}") from exc
    return _mapping(payload, description)


def _safe_member(owner: Path, raw: Any) -> Path:
    member = Path(str(raw))
    if not str(raw) or member.is_absolute() or ".." in member.parts:
        raise ValueError("artifact arrays_file must be a safe relative path")
    resolved = (owner.parent / member).resolve()
    try:
        resolved.relative_to(owner.parent.resolve())
    except ValueError:
        raise ValueError("artifact arrays_file escapes its directory") from None
    return resolved


@dataclass(frozen=True)
class DirectSelectionConfig:
    path: Path
    sha256: str
    config_root: Path
    centers: tuple[str, ...]
    protocol_id: str
    method_id: str
    prediction_subdir: str
    prediction_glob: str
    output_file: str
    expected_records_per_center: int
    expected_pooled_records: int
    expected_compositions: int
    clean_weight: float
    robust_weight: float
    maximum_clean_drop: float
    tie_break: str
    mapping_version: str
    mapping_hash: str
    clean_baseline_registry: Path

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "config_root": str(self.config_root),
            "centers": list(self.centers),
            "protocol_id": self.protocol_id,
            "method_id": self.method_id,
            "prediction_subdir": self.prediction_subdir,
            "prediction_glob": self.prediction_glob,
            "output_file": self.output_file,
            "expected_records_per_center": self.expected_records_per_center,
            "expected_pooled_records": self.expected_pooled_records,
            "expected_compositions": self.expected_compositions,
            "score_weights": {
                "clean": self.clean_weight,
                "robust": self.robust_weight,
            },
            "maximum_clean_drop": self.maximum_clean_drop,
            "tie_break": self.tie_break,
            "mapping_version": self.mapping_version,
            "mapping_hash": self.mapping_hash,
            "clean_baseline_registry": {
                "path": str(self.clean_baseline_registry),
                "sha256": _sha256_file(self.clean_baseline_registry),
            },
        }


def load_direct_selection_config(
    path: str | Path = DEFAULT_DIRECT_TUNE_CONFIG,
    *,
    config_root: str | Path | None = None,
) -> DirectSelectionConfig:
    config_path = resolve_entry_config_path(path)
    root = config_bundle_root(config_path, config_root=config_root)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root_payload = _mapping(payload, "Direct tuning config")
    if root_payload.get("schema_version") != 1:
        raise ValueError("Direct tuning config must use schema_version=1")
    protocol = _mapping(root_payload.get("protocol_lock"), "protocol_lock")
    references = _mapping(root_payload.get("references"), "references")
    predictions = _mapping(root_payload.get("validation_predictions"), "validation_predictions")
    selection = _mapping(root_payload.get("pooled_selection"), "pooled_selection")
    if protocol.get("status") != "k500_internal_tuning_only":
        raise ValueError("selection requires the K500-internal tuning protocol")
    centers = tuple(str(value) for value in protocol.get("centers", ()))
    if not centers or len(centers) != len(set(centers)):
        raise ValueError("protocol centers must be non-empty and unique")
    if predictions.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError("validation artifact schema mismatch")
    composition_indices = tuple(predictions.get("composition_indices", ()))
    if composition_indices != tuple(range(20)):
        raise ValueError("validation compositions must be canonical 0..19")
    if selection.get("clean_aggregation") != "concatenate_four_centers_before_metric":
        raise ValueError("clean aggregation contract mismatch")
    if selection.get("corrupted_aggregation") != "concatenate_four_centers_per_composition_before_metric":
        raise ValueError("corrupted aggregation must be per composition")
    if selection.get("robust_aggregation") != "arithmetic_mean_of_20_composition_macro_auprc":
        raise ValueError("robust aggregation contract mismatch")
    if selection.get("metric") != "macro_auprc" or selection.get("metric_definition") != "sklearn_average_precision":
        raise ValueError("selection metric must be macro Average Precision")
    if selection.get("input_type") != "raw_logits" or selection.get("strict_all_five_classes") is not True:
        raise ValueError("selection requires raw logits and strict five-class metrics")
    score = _mapping(selection.get("score"), "pooled_selection.score")
    clean_weight = float(score.get("clean_weight", -1.0))
    robust_weight = float(score.get("robust_weight", -1.0))
    if clean_weight != 0.5 or robust_weight != 0.5:
        raise ValueError("selection score must be 0.5 clean + 0.5 robust")
    floor = _mapping(selection.get("clean_floor"), "pooled_selection.clean_floor")
    maximum_drop = float(floor.get("maximum_drop_absolute", -1.0))
    if floor.get("reference") != "same_backbone_locked_clean_selection" or maximum_drop != 0.01:
        raise ValueError("clean floor must use the same-backbone locked clean score minus 0.01")
    if floor.get("failure_policy") != "no_selection":
        raise ValueError("clean-floor failure must produce no selection")
    tie_break = str(selection.get("tie_break", ""))
    if tie_break != "earliest_epoch_on_exact_tie":
        raise ValueError("selection tie break must choose the earliest exact tie")
    subdir = str(predictions.get("subdir", ""))
    output_file = str(selection.get("output_file", ""))
    for value, description in ((subdir, "prediction subdir"), (output_file, "output file")):
        member = Path(value)
        if not value or member.is_absolute() or ".." in member.parts:
            raise ValueError(f"{description} must be a safe relative path")
    clean_registry = resolve_config_reference(
        references.get("clean_baseline_registry"),
        owner_config_path=config_path,
        config_root=root,
        description="references.clean_baseline_registry",
        must_exist=True,
    )
    expected_per_center = _positive_int(selection.get("expected_records_per_center"), "expected_records_per_center")
    expected_pooled = _positive_int(selection.get("expected_pooled_records"), "expected_pooled_records")
    expected_compositions = _positive_int(selection.get("expected_compositions"), "expected_compositions")
    if expected_pooled != expected_per_center * len(centers) or expected_compositions != 20:
        raise ValueError("pooled selection counts are inconsistent")
    return DirectSelectionConfig(
        path=config_path,
        sha256=_sha256_file(config_path),
        config_root=root,
        centers=centers,
        protocol_id=str(protocol.get("protocol_id")),
        method_id=str(protocol.get("method_id")),
        prediction_subdir=subdir,
        prediction_glob="epoch_*.json",
        output_file=output_file,
        expected_records_per_center=expected_per_center,
        expected_pooled_records=expected_pooled,
        expected_compositions=expected_compositions,
        clean_weight=clean_weight,
        robust_weight=robust_weight,
        maximum_clean_drop=maximum_drop,
        tie_break=tie_break,
        mapping_version=str(protocol.get("mapping_version")),
        mapping_hash=str(protocol.get("mapping_hash")),
        clean_baseline_registry=clean_registry,
    )


@dataclass(frozen=True)
class ValidationPredictionArtifact:
    metadata_path: Path
    metadata_sha256: str
    arrays_path: Path
    arrays_sha256: str
    epoch: int
    center: str
    clean_logits: np.ndarray
    corrupted_logits: np.ndarray
    targets: np.ndarray
    hash_ids: tuple[str, ...]
    composition_ids: tuple[str, ...]
    clean_validation_identity: dict[str, Any]
    frozen_corruption_identity: dict[str, Any]
    comparison_identity: dict[str, Any]

    def input_evidence(self) -> dict[str, Any]:
        return {
            "center": self.center,
            "epoch": self.epoch,
            "metadata_path": str(self.metadata_path),
            "metadata_sha256": self.metadata_sha256,
            "arrays_path": str(self.arrays_path),
            "arrays_sha256": self.arrays_sha256,
            "record_count": len(self.hash_ids),
            "composition_count": len(self.composition_ids),
        }


def load_validation_prediction_artifact(metadata_path: str | Path) -> ValidationPredictionArtifact:
    path = Path(metadata_path).expanduser().resolve()
    payload = _read_json(path, "validation prediction metadata")
    if payload.get("schema_version") != 1 or payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("unexpected validation prediction artifact")
    if payload.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError("validation prediction artifact schema mismatch")
    epoch = _positive_int(payload.get("epoch"), "validation epoch")
    center = str(payload.get("center", ""))
    if not center or tuple(payload.get("class_order", ())) != CLASS_ORDER:
        raise ValueError("validation center or class order is invalid")
    if payload.get("record_count") != 100 or payload.get("composition_count") != 20:
        raise ValueError("validation artifact must contain 100 records and 20 compositions")
    arrays_path = _safe_member(path, payload.get("arrays_file"))
    if not arrays_path.is_file() or _sha256_file(arrays_path) != payload.get("arrays_sha256"):
        raise ValueError("validation arrays are missing or fail SHA256 validation")
    with np.load(arrays_path, allow_pickle=False) as archive:
        expected = {"clean_logits", "corrupted_logits", "targets", "hash_ids", "composition_ids"}
        if set(archive.files) != expected:
            raise ValueError("validation NPZ members mismatch")
        clean = np.asarray(archive["clean_logits"])
        corrupted = np.asarray(archive["corrupted_logits"])
        targets = np.asarray(archive["targets"])
        hash_array = np.asarray(archive["hash_ids"])
        composition_array = np.asarray(archive["composition_ids"])
    if clean.dtype != np.float32 or clean.shape != (100, 5) or not np.isfinite(clean).all():
        raise ValueError("clean logits must be finite float32 (100,5)")
    if corrupted.dtype != np.float32 or corrupted.shape != (20, 100, 5) or not np.isfinite(corrupted).all():
        raise ValueError("corrupted logits must be finite float32 (20,100,5)")
    if targets.dtype != np.uint8 or targets.shape != (100, 5) or not np.isin(targets, (0, 1)).all():
        raise ValueError("validation targets must be binary uint8 (100,5)")
    if hash_array.shape != (100,) or hash_array.dtype.kind not in {"U", "S"}:
        raise ValueError("validation hash IDs must be strings with shape (100,)")
    if composition_array.shape != (20,) or composition_array.dtype.kind not in {"U", "S"}:
        raise ValueError("composition IDs must be strings with shape (20,)")
    hashes = tuple(str(value) for value in hash_array.tolist())
    composition_ids = tuple(str(value) for value in composition_array.tolist())
    if len(set(hashes)) != 100 or any(not value for value in hashes):
        raise ValueError("validation hash IDs must be unique and non-empty")
    if len(set(composition_ids)) != 20 or any(not value for value in composition_ids):
        raise ValueError("composition IDs must be unique and non-empty")
    clean_identity = _mapping(payload.get("clean_validation_identity"), "clean_validation_identity")
    frozen_identity = _mapping(payload.get("frozen_corruption_identity"), "frozen_corruption_identity")
    comparison_identity = _mapping(payload.get("comparison_identity"), "comparison_identity")
    prediction_identity = _mapping(payload.get("prediction_identity"), "prediction_identity")
    if prediction_identity.get("ordered_hash_ids_sha256") != _sha256_lines(hashes, sort=False):
        raise ValueError("ordered validation hash identity mismatch")
    if prediction_identity.get("hash_id_set_sha256") != _sha256_lines(hashes, sort=True):
        raise ValueError("validation hash set identity mismatch")
    if prediction_identity.get("hash_label_set_sha256") != _hash_label_set_sha256(hashes, targets):
        raise ValueError("validation hash/label identity mismatch")
    if tuple(frozen_identity.get("composition_ids", ())) != composition_ids:
        raise ValueError("frozen composition identity mismatch")
    return ValidationPredictionArtifact(
        metadata_path=path,
        metadata_sha256=_sha256_file(path),
        arrays_path=arrays_path,
        arrays_sha256=_sha256_file(arrays_path),
        epoch=epoch,
        center=center,
        clean_logits=clean,
        corrupted_logits=corrupted,
        targets=targets,
        hash_ids=hashes,
        composition_ids=composition_ids,
        clean_validation_identity=clean_identity,
        frozen_corruption_identity=frozen_identity,
        comparison_identity=comparison_identity,
    )


def discover_validation_prediction_artifacts(
    center_run_dirs: Mapping[str, str | Path],
    *,
    config_path: str | Path = DEFAULT_DIRECT_TUNE_CONFIG,
    config_root: str | Path | None = None,
) -> tuple[Path, ...]:
    config = load_direct_selection_config(config_path, config_root=config_root)
    if set(center_run_dirs) != set(config.centers):
        raise ValueError("center_run_dirs must cover every configured center exactly")
    paths: list[Path] = []
    for center in config.centers:
        directory = Path(center_run_dirs[center]).expanduser().resolve() / config.prediction_subdir
        members = sorted(directory.glob(config.prediction_glob))
        if not members:
            raise FileNotFoundError(f"no validation artifacts found for {center}: {directory}")
        paths.extend(members)
    return tuple(paths)


def _clean_reference(config: DirectSelectionConfig, model_family: str) -> dict[str, Any]:
    registry = yaml.safe_load(config.clean_baseline_registry.read_text(encoding="utf-8"))
    root = _mapping(registry, "clean baseline registry")
    models = _mapping(root.get("models"), "clean baseline models")
    model = _mapping(models.get(model_family), f"clean baseline model {model_family}")
    selection = _mapping(model.get("selection"), "clean baseline selection")
    value = float(selection.get("pooled_validation_macro_auprc", float("nan")))
    if not np.isfinite(value):
        raise ValueError("clean baseline reference AUPRC is not finite")
    artifact_path = Path(str(selection.get("artifact_path", ""))).expanduser().resolve()
    expected_sha = str(selection.get("artifact_sha256", ""))
    if not artifact_path.is_file() or _sha256_file(artifact_path) != expected_sha:
        raise ValueError("locked clean selection artifact is missing or changed")
    return {
        "model_family": model_family,
        "macro_auprc": value,
        "artifact_path": str(artifact_path),
        "artifact_sha256": expected_sha,
        "registry_path": str(config.clean_baseline_registry),
        "registry_sha256": _sha256_file(config.clean_baseline_registry),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _frozen_cache_identity_sha256(identity: Mapping[str, Any]) -> str:
    """Compare the shared cache/composition contract, not center hash membership."""

    return _canonical_sha256(
        {
            key: identity.get(key)
            for key in (
                "corruption_cache_dir",
                "corruption_cache_version",
                "corruption_cache_manifest_path",
                "corruption_cache_manifest_sha256",
                "corruption_config_identity_hash",
                "composition_indices",
                "composition_ids",
                "source_manifest_sha256",
            )
        }
    )


def select_direct_baseline_epoch(
    metadata_paths: Sequence[str | Path],
    *,
    config_path: str | Path = DEFAULT_DIRECT_TUNE_CONFIG,
    config_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if not metadata_paths:
        raise ValueError("at least one validation artifact is required")
    config = load_direct_selection_config(config_path, config_root=config_root)
    artifacts = [load_validation_prediction_artifact(path) for path in metadata_paths]
    keyed: dict[tuple[str, int], ValidationPredictionArtifact] = {}
    for artifact in artifacts:
        if artifact.center not in config.centers:
            raise ValueError(f"unexpected center: {artifact.center}")
        key = (artifact.center, artifact.epoch)
        if key in keyed:
            raise ValueError(f"duplicate validation artifact for {key}")
        if artifact.clean_validation_identity.get("partition") != "k500_tune_validation":
            raise ValueError("validation artifact uses the wrong partition")
        if artifact.clean_validation_identity.get("mapping_version") != config.mapping_version or artifact.clean_validation_identity.get("mapping_hash") != config.mapping_hash:
            raise ValueError("validation mapping identity mismatch")
        if artifact.comparison_identity.get("protocol_id") != config.protocol_id or artifact.comparison_identity.get("method_id") != config.method_id:
            raise ValueError("validation comparison protocol mismatch")
        if artifact.comparison_identity.get("tuning_config_sha256") != config.sha256:
            raise ValueError("validation tuning config SHA256 mismatch")
        keyed[key] = artifact
    if {center for center, _ in keyed} != set(config.centers):
        raise ValueError("validation artifacts do not cover exactly four centers")
    epoch_grids = {
        center: tuple(sorted(epoch for member_center, epoch in keyed if member_center == center))
        for center in config.centers
    }
    grid = epoch_grids[config.centers[0]]
    if not grid or any(value != grid for value in epoch_grids.values()) or grid != tuple(range(1, grid[-1] + 1)):
        raise ValueError("all centers must provide the same contiguous epoch grid")

    common_comparison = keyed[(config.centers[0], grid[0])].comparison_identity
    common_comparison_sha = _canonical_sha256(common_comparison)
    if any(_canonical_sha256(item.comparison_identity) != common_comparison_sha for item in artifacts):
        raise ValueError("model/source/hyperparameter identity differs across artifacts")
    model_family = str(common_comparison.get("model_family", ""))
    if model_family not in {"efficientnet1dv2", "ecgfounder"}:
        raise ValueError("unsupported comparison model family")
    resolved_training = _mapping(common_comparison.get("resolved_training_parameters"), "resolved training parameters")
    expected_epochs = _positive_int(resolved_training.get("epochs"), "resolved epochs")
    if grid[-1] != expected_epochs:
        raise ValueError("validation epoch grid is incomplete")

    per_center_identity: dict[str, dict[str, Any]] = {}
    center_hash_sets: dict[str, set[str]] = {}
    reference_composition_ids: tuple[str, ...] | None = None
    frozen_identity_sha: str | None = None
    for center in config.centers:
        first = keyed[(center, grid[0])]
        center_hash_sets[center] = set(first.hash_ids)
        if len(center_hash_sets[center]) != config.expected_records_per_center:
            raise ValueError("per-center validation record count mismatch")
        for epoch in grid[1:]:
            current = keyed[(center, epoch)]
            if current.hash_ids != first.hash_ids or not np.array_equal(current.targets, first.targets):
                raise ValueError(f"validation records changed across epochs for {center}")
            if current.composition_ids != first.composition_ids:
                raise ValueError(f"composition order changed across epochs for {center}")
            if _frozen_cache_identity_sha256(current.frozen_corruption_identity) != _frozen_cache_identity_sha256(first.frozen_corruption_identity):
                raise ValueError(f"frozen corruption identity changed for {center}")
        if reference_composition_ids is None:
            reference_composition_ids = first.composition_ids
            frozen_identity_sha = _frozen_cache_identity_sha256(first.frozen_corruption_identity)
        elif first.composition_ids != reference_composition_ids or _frozen_cache_identity_sha256(first.frozen_corruption_identity) != frozen_identity_sha:
            raise ValueError("four centers do not share one frozen corruption identity")
        per_center_identity[center] = {
            "clean_validation_identity": first.clean_validation_identity,
            "ordered_hash_ids_sha256": _sha256_lines(first.hash_ids, sort=False),
            "hash_label_set_sha256": _hash_label_set_sha256(first.hash_ids, first.targets),
        }
    for index, left in enumerate(config.centers):
        for right in config.centers[index + 1 :]:
            if center_hash_sets[left].intersection(center_hash_sets[right]):
                raise ValueError(f"cross-center validation leakage: {left}/{right}")

    clean_reference = _clean_reference(config, model_family)
    clean_floor = float(clean_reference["macro_auprc"]) - config.maximum_clean_drop
    per_epoch: list[dict[str, Any]] = []
    for epoch in grid:
        members = [keyed[(center, epoch)] for center in config.centers]
        targets = np.concatenate([item.targets for item in members], axis=0)
        hashes = tuple(hash_id for item in members for hash_id in item.hash_ids)
        if targets.shape != (config.expected_pooled_records, 5):
            raise ValueError("pooled validation target shape mismatch")
        clean_logits = np.concatenate([item.clean_logits for item in members], axis=0)
        clean_metrics = compute_classification_metrics(clean_logits, targets, class_order=CLASS_ORDER)
        composition_metrics: list[dict[str, Any]] = []
        for composition_index in range(config.expected_compositions):
            logits = np.concatenate(
                [item.corrupted_logits[composition_index] for item in members], axis=0
            )
            metrics = compute_classification_metrics(logits, targets, class_order=CLASS_ORDER)
            composition_metrics.append(
                {
                    "composition_index": composition_index,
                    "composition_id": reference_composition_ids[composition_index],
                    "metrics": metrics,
                }
            )
        robust_auprc = float(np.mean([item["metrics"]["macro_auprc"] for item in composition_metrics]))
        robust_auroc = float(np.mean([item["metrics"]["macro_auroc"] for item in composition_metrics]))
        clean_auprc = float(clean_metrics["macro_auprc"])
        score = config.clean_weight * clean_auprc + config.robust_weight * robust_auprc
        eligible = clean_auprc >= clean_floor
        per_epoch.append(
            {
                "epoch": epoch,
                "clean": clean_metrics,
                "robust": {
                    "macro_auprc": robust_auprc,
                    "macro_auroc": robust_auroc,
                    "composition_metrics": composition_metrics,
                },
                "score": score,
                "clean_floor": clean_floor,
                "eligible": eligible,
                "record_identity": {
                    "record_count": len(hashes),
                    "ordered_hash_ids_sha256": _sha256_lines(hashes, sort=False),
                    "hash_id_set_sha256": _sha256_lines(hashes, sort=True),
                    "hash_label_set_sha256": _hash_label_set_sha256(hashes, targets),
                },
                "member_artifacts": [item.input_evidence() for item in members],
            }
        )
    eligible_epochs = [item for item in per_epoch if item["eligible"]]
    status = "selected" if eligible_epochs else "failed_clean_floor"
    selected = (
        max(eligible_epochs, key=lambda item: (float(item["score"]), -int(item["epoch"])))
        if eligible_epochs
        else None
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": (
            "direct_k500_pooled_epoch_selection"
            if config.method_id == "direct_depth23_fixed20"
            else "pn2021_k500_pooled_epoch_selection"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "selection_rule": locked_pooled_selection_rule(),
        "selected_epoch": None if selected is None else int(selected["epoch"]),
        "selected_score": None if selected is None else float(selected["score"]),
        "selected_clean_macro_auprc": None if selected is None else float(selected["clean"]["macro_auprc"]),
        "selected_clean_macro_auroc": None if selected is None else float(selected["clean"]["macro_auroc"]),
        "selected_robust_macro_auprc": None if selected is None else float(selected["robust"]["macro_auprc"]),
        "selected_robust_macro_auroc": None if selected is None else float(selected["robust"]["macro_auroc"]),
        "clean_reference": clean_reference,
        "clean_floor": clean_floor,
        "epoch_grid": list(grid),
        "class_order": list(CLASS_ORDER),
        "centers": list(config.centers),
        "record_count": config.expected_pooled_records,
        "composition_count": config.expected_compositions,
        "composition_ids": list(reference_composition_ids),
        "config": config.describe(),
        "comparison_identity": common_comparison,
        "comparison_identity_sha256": common_comparison_sha,
        "frozen_corruption_identity_sha256": frozen_identity_sha,
        "per_center_validation_identity": per_center_identity,
        "per_epoch": per_epoch,
        "software": {
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        },
    }
    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() != ".json":
            output = output / config.output_file
        if output.exists():
            raise FileExistsError(f"selection output already exists: {output}")
        _atomic_write_json(output, result)
        result["output_path"] = str(output)
        result["output_sha256"] = _sha256_file(output)
    return result


__all__ = [
    "ARTIFACT_SCHEMA",
    "ARTIFACT_TYPE",
    "DEFAULT_DIRECT_TUNE_CONFIG",
    "DirectSelectionConfig",
    "ValidationPredictionArtifact",
    "discover_validation_prediction_artifacts",
    "locked_pooled_selection_rule",
    "load_direct_selection_config",
    "load_validation_prediction_artifact",
    "select_direct_baseline_epoch",
]
