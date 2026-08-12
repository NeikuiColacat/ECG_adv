"""Evaluate one lineage-bound Super5 checkpoint on PN2021 and PN2021-C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.methods.registry import RecipeSpec, load_recipe_spec
from models.checkpoints import (
    CheckpointIdentity,
    load_model_checkpoint,
    sha256_file,
)
from models.contracts import CLASS_ORDER, ModelSpec
from models.factory import available_models, build_model, get_model_spec
from util.config_bundle import resolve_config_reference
from util.evaluation.metrics import resolve_evaluated_center_mean
from util.evaluation.pn2021 import (
    DEFAULT_PN2021_EVAL_CONFIG,
    LOGICAL_CENTERS,
    PN2021_MAPPING_HASH,
    PN2021_MAPPING_VERSION,
    PN2021EvalConfig,
    build_evaluation_plan,
    evaluate_pn2021,
    load_pn2021_eval_config,
)
from util.pn2021_artifact_contract import validate_pn2021_train_result


LEGACY_METHODS = {
    "a0_clean_v1": "A0",
    "direct_depth23_fixed20": "DirectDepth23Fixed20FamilyBalanced",
}


EvaluationSubject = tuple[Path, tuple[str, ...], dict[str, Any]]


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _artifact_path(value: str | Path | None, description: str) -> Path:
    if value is None:
        raise ValueError(f"{description} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _json_artifact(
    value: str | Path | None, description: str
) -> tuple[Path, dict[str, Any]]:
    path = _artifact_path(value, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {description} JSON at {path}: {exc}") from exc
    return path, _mapping(value, description)


def _validated_checkpoint_reference(
    value: Any, *, description: str
) -> tuple[Path, str]:
    reference = _mapping(value, description)
    if set(reference) != {"path", "sha256"}:
        raise ValueError(f"{description} must contain exactly path and sha256")
    path = _artifact_path(reference["path"], f"{description}.path")
    expected = reference["sha256"]
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"{description} SHA256 mismatch")
    return path, expected


def _recipe_lineage(recipe: RecipeSpec) -> dict[str, Any]:
    return {
        "recipe_id": recipe.recipe_id,
        "scientific_arm": recipe.scientific_arm,
        "recipe_spec_sha256": recipe.recipe_sha256,
        "implementation_identity": recipe.implementation_identity,
        "recipe_version": recipe.recipe_version,
        "kind": recipe.kind.value,
        "auxiliary_variant": recipe.auxiliary_variant.value,
        "schema_version": recipe.schema_version,
    }


def _resolve_prospective_subject(
    args: argparse.Namespace,
    config: PN2021EvalConfig,
    spec: ModelSpec,
) -> EvaluationSubject:
    if args.method_config is None:
        raise ValueError("--train-result requires --method-config")
    if args.center is None or len(args.center) != 1:
        raise ValueError("--train-result requires exactly one --center")

    method_path = resolve_config_reference(
        str(args.method_config),
        owner_config_path=config.path,
        config_root=config.config_root,
        description="method_config",
        must_exist=True,
    )
    recipe = load_recipe_spec(method_path)
    train_result_path, payload = _json_artifact(args.train_result, "train_result")
    train = validate_pn2021_train_result(payload, result_path=train_result_path)
    lineage = train["lineage"]
    center = args.center[0]
    if (
        lineage["model"] != {"name": spec.name, "spec": spec.describe()}
        or _mapping(payload.get("model"), "train_result.model").get("spec")
        != spec.describe()
    ):
        raise ValueError("train_result model lineage differs from --model")
    if payload.get("center") != center or lineage["center"] != center:
        raise ValueError("train_result center lineage differs from --center")
    if lineage["method"] != _recipe_lineage(recipe):
        raise ValueError("train_result method lineage differs from --method-config")
    if (
        payload.get("method_id") != recipe.recipe_id
        or payload.get("scientific_arm") != recipe.scientific_arm
    ):
        raise ValueError("train_result root method identity differs from lineage")
    checkpoint_path = train["checkpoint_path"]
    identity = {
        "mode": "prospective_train_result",
        "train_result": {
            "path": str(train_result_path), "sha256": sha256_file(train_result_path)
        },
        "lineage": lineage,
    }
    return checkpoint_path, (center,), identity


def _legacy_protocol(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config = _mapping(payload.get("config"), "legacy train_result.config")
    training = _mapping(config.get("training"), "legacy train_result.config.training")
    resolved = _mapping(training.get("resolved"), "legacy training config")
    protocol = _mapping(resolved.get("protocol"), "legacy training protocol")
    if (
        protocol.get("partition") != "k500"
        or protocol.get("use_all_k500") is not True
        or protocol.get("validation_split") is not False
        or protocol.get("checkpoint_selection") != "last"
        or protocol.get("heldout_ref_exclusion_required") is not True
        or protocol.get("merge_cpsc_2018_extra_into_cpsc_2018") is not True
        or protocol.get("centers") != list(LOGICAL_CENTERS)
        or protocol.get("class_order") != list(CLASS_ORDER)
        or protocol.get("mapping_version") != PN2021_MAPPING_VERSION
        or protocol.get("mapping_hash") != PN2021_MAPPING_HASH
    ):
        raise ValueError("legacy train_result is not the locked K500 protocol")
    return protocol


def _resolve_legacy_subject(
    args: argparse.Namespace,
    spec: ModelSpec,
) -> EvaluationSubject:
    if args.method_config is not None:
        raise ValueError("legacy --checkpoint rejects --method-config")
    if args.center is None or len(args.center) != 1:
        raise ValueError("legacy --checkpoint requires exactly one --center")
    checkpoint_path = _artifact_path(args.checkpoint, "checkpoint")
    train_result_path = checkpoint_path.parent.parent / "train_result.json"
    _, payload = _json_artifact(train_result_path, "legacy sibling train_result")
    if "schema_version" in payload or "artifact_type" in payload or "lineage" in payload:
        raise ValueError("schema-1 train_result cannot use legacy --checkpoint mode")
    center = args.center[0]
    method_id = payload.get("method_id")
    scientific_arm = payload.get("scientific_arm")
    if LEGACY_METHODS.get(method_id) != scientific_arm:
        raise ValueError("legacy checkpoint method is not A0 or Direct fixed20")
    model = _mapping(payload.get("model"), "legacy train_result.model")
    if model.get("spec") != spec.describe() or payload.get("center") != center:
        raise ValueError("legacy train_result model or center differs from CLI")
    config = _mapping(payload.get("config"), "legacy train_result.config")
    method = _mapping(config.get("method"), "legacy train_result.config.method")
    if (
        method.get("profile_name") != method_id
        or method.get("scientific_arm") != scientific_arm
    ):
        raise ValueError("legacy train_result method identities disagree")
    protocol = _legacy_protocol(payload)
    selection = _mapping(payload.get("selection"), "legacy train_result.selection")
    last = _mapping(payload.get("last_checkpoint"), "legacy last_checkpoint")
    if (
        selection.get("policy") != "last"
        or selection.get("heldout_evaluation_used_for_selection") is not False
        or selection.get("selected_checkpoint") != last
    ):
        raise ValueError("legacy train_result must select its heldout-free last checkpoint")
    bound_path, checkpoint_sha = _validated_checkpoint_reference(
        last, description="legacy train_result.last_checkpoint"
    )
    if bound_path != checkpoint_path:
        raise ValueError("legacy train_result points to a different checkpoint")
    lineage = {
        "scope": "legacy_pn2021_k500_center_adaptation",
        "model": {"name": spec.name, "spec": spec.describe()},
        "center": center,
        "method": {"recipe_id": method_id, "scientific_arm": scientific_arm},
        "adaptation_data": {
            "partition": protocol["partition"],
            "mapping_version": protocol["mapping_version"],
            "mapping_hash": protocol["mapping_hash"],
        },
        "selection": {
            "policy": "last", "heldout_evaluation_used_for_selection": False,
        },
        "checkpoint": {"sha256": checkpoint_sha},
    }
    identity = {
        "mode": "legacy_center_adapted",
        "train_result": {
            "path": str(train_result_path), "sha256": sha256_file(train_result_path)
        },
        "lineage": lineage,
    }
    return checkpoint_path, (center,), identity


def _resolve_source_subject(
    args: argparse.Namespace,
    config: PN2021EvalConfig,
    spec: ModelSpec,
) -> EvaluationSubject:
    if args.method_config is not None:
        raise ValueError("--source-registry rejects --method-config")
    if args.center is not None:
        raise ValueError("--source-registry evaluates the canonical four centers")
    registry_path = resolve_config_reference(
        str(args.source_registry),
        owner_config_path=config.path,
        config_root=config.config_root,
        description="source_registry",
        must_exist=True,
    )
    registry = _mapping(
        yaml.safe_load(registry_path.read_text(encoding="utf-8")), "source registry"
    )
    baseline = _mapping(registry.get("baseline"), "source registry baseline")
    models = _mapping(registry.get("models"), "source registry models")
    policy = _mapping(registry.get("downstream_policy"), "source downstream policy")
    if (
        registry.get("schema_version") != 1
        or baseline.get("class_order") != list(CLASS_ORDER)
        or policy.get("checkpoint_field") != "selected_checkpoint"
        or policy.get("require_sha256_match") is not True
        or policy.get("prohibit_last_checkpoint") is not True
    ):
        raise ValueError("source registry does not enforce the locked best checkpoint")
    model_entry = _mapping(models.get(spec.name), f"source registry model {spec.name}")
    checkpoint_path = _artifact_path(
        model_entry.get("selected_checkpoint"), "source selected_checkpoint"
    )
    checkpoint_sha = model_entry.get("selected_checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError("source selected checkpoint SHA256 mismatch")
    lineage = {
        "scope": "ptbxl_source_global",
        "model": {"name": spec.name, "spec": spec.describe()},
        "source_checkpoint": {"sha256": checkpoint_sha},
        "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
        "selection": {
            "policy": baseline.get("selection_rule"),
            "heldout_evaluation_used_for_selection": False,
        },
    }
    identity = {"mode": "source_registry", "train_result": None, "lineage": lineage}
    return checkpoint_path, LOGICAL_CENTERS, identity


def _resolve_evaluation_subject(
    args: argparse.Namespace,
    config: PN2021EvalConfig,
    spec: ModelSpec,
) -> EvaluationSubject:
    selected = sum(
        value is not None
        for value in (args.train_result, args.checkpoint, args.source_registry)
    )
    if selected != 1:
        raise ValueError(
            "select exactly one evaluation subject: --train-result, --checkpoint, "
            "or --source-registry"
        )
    if args.train_result is not None:
        return _resolve_prospective_subject(args, config, spec)
    if args.checkpoint is not None:
        return _resolve_legacy_subject(args, spec)
    return _resolve_source_subject(args, config, spec)


def _validate_loaded_subject(
    subject: Mapping[str, Any],
    checkpoint: CheckpointIdentity,
    *,
    expected_sha256: str,
) -> None:
    if checkpoint.sha256 != expected_sha256:
        raise ValueError("loaded checkpoint SHA256 differs from the resolved subject")
    schema_version = checkpoint.checkpoint_schema_version
    mode = subject["mode"]
    if mode == "prospective_train_result":
        if schema_version != 3 or checkpoint.lineage != subject["lineage"]:
            raise ValueError("prospective checkpoint schema/lineage differs from train_result")
    elif mode == "legacy_center_adapted":
        lineage = subject["lineage"]
        expected = {
            "model": lineage["model"]["name"],
            "center": lineage["center"],
            "method_id": lineage["method"]["recipe_id"],
            "scientific_arm": lineage["method"]["scientific_arm"],
            "selection": "last",
        }
        if schema_version != 2 or checkpoint.legacy_training_identity != expected:
            raise ValueError("legacy evaluation accepts only schema-2 adapted checkpoints")
    elif mode == "source_registry":
        if schema_version != 1 or checkpoint.lineage is not None:
            raise ValueError("source registry must resolve the locked schema-1 checkpoint")
    else:  # pragma: no cover - EvaluationSubject is code-owned
        raise AssertionError(f"unsupported evaluation subject mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one lineage-bound Super5 artifact on PN2021/PN2021-C."
    )
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-result", type=Path)
    parser.add_argument("--method-config", type=Path)
    parser.add_argument("--source-registry", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_PN2021_EVAL_CONFIG)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--center", action="append", choices=LOGICAL_CENTERS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _summary(result: dict[str, object]) -> dict[str, object]:
    clean = result["clean"]
    corrupted = result["corrupted"]
    assert isinstance(clean, dict) and isinstance(corrupted, dict)
    protocol = result.get("protocol", {})
    assert isinstance(protocol, dict)
    aggregates = corrupted["aggregates"]
    assert isinstance(aggregates, dict)
    depth2, depth3, depth23 = (
        aggregates["depth2"], aggregates["depth3"], aggregates["depth23"]
    )
    assert isinstance(depth2, dict) and isinstance(depth3, dict)
    assert isinstance(depth23, dict)
    center_order = clean.get("center_order", protocol.get("logical_centers", []))
    assert isinstance(center_order, list)
    center_count = clean.get("center_count", len(center_order))

    def verified_four_center_mean(payload: dict[str, object]) -> object:
        if (
            payload.get("center_count", center_count) == len(LOGICAL_CENTERS)
            and payload.get("center_order", center_order) == list(LOGICAL_CENTERS)
        ):
            return payload.get("four_center_mean")
        return None

    return {
        "status": result["status"],
        "evaluation_profile": result["evaluation_profile"],
        "model": result["model"],
        "checkpoint": result["checkpoint"],
        "subject": result.get("subject"),
        "center_count": center_count,
        "center_order": center_order,
        "clean_evaluated_center_mean": resolve_evaluated_center_mean(clean),
        "pn2021c_depth2_evaluated_center_mean": resolve_evaluated_center_mean(depth2),
        "pn2021c_depth3_evaluated_center_mean": resolve_evaluated_center_mean(depth3),
        "pn2021c_depth23_evaluated_center_mean": resolve_evaluated_center_mean(depth23),
        "clean_four_center_mean": verified_four_center_mean(clean),
        "pn2021c_depth2_four_center_mean": verified_four_center_mean(depth2),
        "pn2021c_depth3_four_center_mean": verified_four_center_mean(depth3),
        "pn2021c_depth23_four_center_mean": verified_four_center_mean(depth23),
        "output": result["output"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_pn2021_eval_config(args.config, config_root=args.config_root)
    spec = get_model_spec(args.model)
    checkpoint_path, logical_centers, subject = _resolve_evaluation_subject(
        args, config, spec
    )
    plan = build_evaluation_plan(
        model_spec=spec,
        checkpoint_path=checkpoint_path,
        config=config,
        output_dir=args.output_dir,
        logical_centers=logical_centers,
        subject_identity=subject,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    model = build_model(
        args.model,
        config_root=config.config_root,
        seed_config_path=config.random_seed_config_path,
        seed_namespace="pn2021_evaluation_model_initialization",
        seed_identity=(config.profile_name, plan["checkpoint"]["sha256"]),
    )
    checkpoint_identity = load_model_checkpoint(
        model, checkpoint_path, map_location="cpu", strict=True
    )
    _validate_loaded_subject(
        subject,
        checkpoint_identity,
        expected_sha256=plan["checkpoint"]["sha256"],
    )
    result = evaluate_pn2021(
        model,
        checkpoint_identity,
        config=config,
        output_dir=args.output_dir,
        logical_centers=logical_centers,
        subject_identity=subject,
    )
    print(json.dumps(_summary(result), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
