"""Refit one full-K500 Direct baseline from a pooled selection artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNING_PROTOCOL_ID = "pn2021_direct_family_balanced_tuning"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.online_trainer import ALLOWED_CENTERS, load_online_train_config  # noqa: E402
from core.train_PN2021 import load_pn2021_method_profile, train_pn2021  # noqa: E402
from models import build_model, get_model_spec  # noqa: E402
from models.checkpoints import sha256_file  # noqa: E402
from models.contracts import CLASS_ORDER  # noqa: E402


MODEL_NAMES = ("efficientnet1dv2", "ecgfounder")
FIXED20_METHOD_ID = "direct_depth23_fixed20"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_managed_selection_file(path: Path) -> None:
    try:
        relative = path.relative_to(path.parent.parent)
    except ValueError:
        raise ValueError("selection path has no managed run root") from None
    run_root = path.parent.parent
    index_path = run_root / "run_file_index.json"
    card_path = run_root / "run_card.json"
    if not index_path.is_file() or not card_path.is_file():
        raise ValueError("selection must belong to a finalized managed run")
    index = _mapping(
        json.loads(index_path.read_text(encoding="utf-8")),
        "selection run_file_index",
    )
    files = index.get("files")
    if not isinstance(files, list):
        raise ValueError("selection run_file_index.files must be a list")
    members = [
        item
        for item in files
        if isinstance(item, dict) and item.get("path") == relative.as_posix()
    ]
    if len(members) != 1 or members[0].get("sha256") != _sha256(path):
        raise ValueError("selection SHA256 differs from its managed run index")
    card = _mapping(
        json.loads(card_path.read_text(encoding="utf-8")),
        "selection run_card",
    )
    if card.get("status") != "complete" or int(card.get("exit_code", -1)) != 0:
        raise ValueError("selection managed run is not complete with exit_code=0")


def _load_selection(path: Path, model_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"pooled selection not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid pooled selection JSON: {exc}") from exc
    root = _mapping(payload, "pooled selection")
    _validate_managed_selection_file(path)
    if root.get("schema_version") != 1 or root.get("status") != "selected":
        raise ValueError("pooled selection must be a selected schema_version=1 artifact")
    if root.get("artifact_type") != "direct_k500_pooled_epoch_selection":
        raise ValueError("unexpected pooled selection artifact_type")
    if tuple(root.get("centers", ())) != ALLOWED_CENTERS:
        raise ValueError("pooled selection must cover the locked four centers")
    if tuple(root.get("class_order", ())) != CLASS_ORDER:
        raise ValueError("pooled selection class_order mismatch")
    if int(root.get("record_count", -1)) != 400:
        raise ValueError("pooled selection must contain exactly 400 validation records")
    if int(root.get("composition_count", -1)) != 20:
        raise ValueError("pooled selection must contain twenty corruption compositions")
    rule = _mapping(root.get("selection_rule"), "selection_rule")
    expected_rule = {
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
    if rule != expected_rule:
        raise ValueError("pooled selection rule differs from the locked protocol")
    comparison = _mapping(root.get("comparison_identity"), "comparison_identity")
    if comparison.get("protocol_id") != TUNING_PROTOCOL_ID:
        raise ValueError("selection comparison protocol_id mismatch")
    if comparison.get("method_id") != FIXED20_METHOD_ID:
        raise ValueError("selection comparison method_id mismatch")
    if root.get("comparison_identity_sha256") != _canonical_sha256(comparison):
        raise ValueError("selection comparison_identity SHA256 mismatch")
    if comparison.get("model_family") != model_name:
        raise ValueError("selection model_family does not match --model")
    selected_epoch = root.get("selected_epoch")
    if isinstance(selected_epoch, bool) or not isinstance(selected_epoch, int) or selected_epoch <= 0:
        raise ValueError("selected_epoch must be a positive integer")
    parameters = _mapping(
        comparison.get("resolved_training_parameters"),
        "resolved_training_parameters",
    )
    declared_epochs = parameters.get("epochs")
    if (
        isinstance(declared_epochs, bool)
        or not isinstance(declared_epochs, int)
        or selected_epoch > declared_epochs
    ):
        raise ValueError("selected_epoch lies outside the completed tuning horizon")
    epoch_grid = root.get("epoch_grid")
    if epoch_grid != list(range(1, int(declared_epochs) + 1)):
        raise ValueError("selection epoch_grid is incomplete or non-contiguous")
    per_epoch = root.get("per_epoch")
    if not isinstance(per_epoch, list) or len(per_epoch) != int(declared_epochs):
        raise ValueError("selection per_epoch evidence is incomplete")
    candidates: list[tuple[float, int]] = []
    for expected_epoch, item in enumerate(per_epoch, start=1):
        member = _mapping(item, f"per_epoch[{expected_epoch}]")
        if member.get("epoch") != expected_epoch:
            raise ValueError("selection per_epoch order mismatch")
        value = float(member.get("score", float("nan")))
        if not math.isfinite(value):
            raise ValueError("selection per_epoch score must be finite")
        if member.get("eligible") is True:
            candidates.append((value, expected_epoch))
    if not candidates:
        raise ValueError("selection has no clean-floor-eligible epoch")
    recomputed_epoch = max(candidates, key=lambda item: (item[0], -item[1]))[1]
    if recomputed_epoch != selected_epoch:
        raise ValueError("selected_epoch is inconsistent with the pooled curve")
    selected_metrics = per_epoch[selected_epoch - 1]
    checks = {
        "selected_score": selected_metrics["score"],
        "selected_clean_macro_auprc": selected_metrics["clean"]["macro_auprc"],
        "selected_robust_macro_auprc": selected_metrics["robust"]["macro_auprc"],
    }
    for field, expected in checks.items():
        if not math.isclose(float(root.get(field, float("nan"))), float(expected), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{field} is inconsistent with selected per_epoch evidence")
    source = _mapping(
        comparison.get("source_checkpoint_identity"),
        "source_checkpoint_identity",
    )
    source_path = Path(str(source.get("path", ""))).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"selected source checkpoint not found: {source_path}")
    if sha256_file(source_path) != source.get("sha256"):
        raise ValueError("selected source checkpoint SHA256 mismatch")
    batch_size = parameters.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("selection train batch_size must be positive")
    return root


def _training_parameters(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = selection["comparison_identity"]
    selected = dict(comparison["resolved_training_parameters"])
    batch_size = int(selected["batch_size"])
    epochs = int(selection["selected_epoch"])
    scheduler_horizon = int(selected["scheduler_horizon_epochs"])
    return {
        "epochs": epochs,
        "scheduler_horizon_epochs": scheduler_horizon,
        "batch_size": batch_size,
        "learning_rate": float(selected["learning_rate"]),
        "weight_decay": float(selected["weight_decay"]),
        "minimum_learning_rate_ratio": float(
            selected["minimum_learning_rate_ratio"]
        ),
        "gradient_clip_norm": float(selected["gradient_clip_norm"]),
        "amp_enabled": bool(selected["amp_enabled"]),
        "amp_dtype": str(selected["amp_dtype"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refit the selected family-balanced Direct baseline on full K500."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--center", required=True, choices=ALLOWED_CENTERS)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_online_train_config(args.config, config_root=args.config_root)
    method, _ = load_pn2021_method_profile(
        args.method_config,
        config_path=config.path,
        config_root=config.config_root,
    )
    if method.profile_name != FIXED20_METHOD_ID:
        raise ValueError("Direct refit requires the family-balanced fixed20 profile")
    selection_path = args.selection_json.expanduser().resolve()
    selection = _load_selection(selection_path, args.model)
    training_parameters = _training_parameters(selection)
    comparison = selection["comparison_identity"]
    source_checkpoint = Path(
        comparison["source_checkpoint_identity"]["path"]
    ).expanduser().resolve()
    output_dir = (
        None if args.output_dir is None else args.output_dir.expanduser().resolve()
    )
    dataloader_parameters = (
        {} if args.num_workers is None else {"num_workers": args.num_workers}
    )
    plan = {
        "action": "pn2021_direct_full_k500_refit",
        "model": get_model_spec(args.model).describe(),
        "center": args.center,
        "method": method.describe(),
        "config": config.describe(),
        "selection": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
            "selected_epoch": int(selection["selected_epoch"]),
            "selected_score": float(selection["selected_score"]),
            "selected_clean_macro_auprc": float(
                selection["selected_clean_macro_auprc"]
            ),
            "selected_robust_macro_auprc": float(
                selection["selected_robust_macro_auprc"]
            ),
            "heldout_evaluation_used": False,
        },
        "source_checkpoint": {
            "path": str(source_checkpoint),
            "sha256": comparison["source_checkpoint_identity"]["sha256"],
        },
        "training_parameters": training_parameters,
        "dataloader_parameters": dataloader_parameters,
        "output_dir": None if output_dir is None else str(output_dir),
        "device": args.device or config.payload["training"]["device"],
        "dry_run_side_effects": "none",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if output_dir is None:
        raise ValueError("--output-dir is required outside dry-run mode")

    seed = config.payload["random_seed"]
    model_kwargs: dict[str, Any]
    if args.model == "efficientnet1dv2":
        model_kwargs = {
            "checkpoint_path": source_checkpoint,
            "map_location": "cpu",
        }
    else:
        model_kwargs = {
            "task_checkpoint_path": source_checkpoint,
            "trainable_scope": "full",
            "map_location": "cpu",
        }
    model = build_model(
        args.model,
        config_root=config.config_root,
        seed_namespace="pn2021_direct_refit_model_initialization",
        seed_identity=(
            seed["comparison_group"],
            seed["replicate_id"],
            args.center,
            args.model,
            method.profile_name,
        ),
        **model_kwargs,
    )
    result = train_pn2021(
        model,
        center=args.center,
        method_config_path=method.source_path,
        config_path=config.path,
        config_root=config.config_root,
        output_dir=output_dir,
        device=args.device,
        training_parameters=training_parameters,
        dataloader_parameters=dataloader_parameters,
    )
    result_description = result.describe()
    refit_contract = {
        "schema_version": 1,
        "artifact_type": "pn2021_direct_refit_contract",
        **plan,
        "result": result_description,
    }
    contract_path = output_dir / "refit_contract.json"
    temporary = contract_path.with_name(f".{contract_path.name}.tmp")
    temporary.write_text(
        json.dumps(refit_contract, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(contract_path)
    print(
        json.dumps(
            result_description,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
