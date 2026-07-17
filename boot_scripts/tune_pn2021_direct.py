"""Run one managed PN2021 Direct K500 train400/validation100 tuning job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pn2021_tuning import (  # noqa: E402
    ALLOWED_CENTERS,
    DEFAULT_DIRECT_TUNE_CONFIG,
    load_pn2021_tuning_config,
    train_pn2021_direct_tuning,
)
from core.online_trainer import resolve_online_training_parameters  # noqa: E402
from models import build_model, get_model_spec  # noqa: E402


MODEL_NAMES = ("efficientnet1dv2", "ecgfounder")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune one Direct PN2021 K500 baseline on train400/validation100."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_DIRECT_TUNE_CONFIG)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--center", required=True, choices=ALLOWED_CENTERS)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--trainable-scope", choices=("full", "head"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--scheduler-horizon-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--minimum-learning-rate-ratio", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument(
        "--amp", dest="amp_enabled", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"))
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--cache-mode", choices=("auto", "ram", "mmap"))
    parser.add_argument("--validate-values", choices=("none", "sample", "full"))
    parser.add_argument("--pos-weight", nargs=5, type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _not_none(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _source_profile(config: Any, model_name: str) -> dict[str, Any]:
    registry = yaml.safe_load(
        config.references["source_baseline_registry"].read_text(encoding="utf-8")
    )
    if not isinstance(registry, dict) or not isinstance(registry.get("models"), dict):
        raise ValueError("source baseline registry is invalid")
    profile = registry["models"].get(model_name)
    if not isinstance(profile, dict):
        raise ValueError(f"source registry has no {model_name} profile")
    return dict(profile)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_pn2021_tuning_config(
        args.config,
        config_root=args.config_root,
    )
    source_profile = _source_profile(config, args.model)
    training_parameters, _ = resolve_online_training_parameters(
        config.online, args.model, None
    )
    training_parameters.update(
        _not_none(
            epochs=args.epochs,
            scheduler_horizon_epochs=args.scheduler_horizon_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
            gradient_clip_norm=args.gradient_clip_norm,
            amp_enabled=args.amp_enabled,
            amp_dtype=args.amp_dtype,
            batch_size=args.train_batch_size,
        )
    )
    if args.epochs is not None and args.scheduler_horizon_epochs is None:
        training_parameters["scheduler_horizon_epochs"] = int(args.epochs)
    dataloader_parameters = _not_none(
            train_batch_size=training_parameters["batch_size"],
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            cache_mode=args.cache_mode,
            validate_values=args.validate_values,
    )
    source_raw = args.source_checkpoint or source_profile.get("selected_checkpoint")
    if source_raw is None:
        raise ValueError(f"source registry checkpoint for {args.model} is required")
    source_checkpoint = Path(source_raw).expanduser().resolve()
    trainable_scope = args.trainable_scope or str(
        source_profile.get("trainable_scope", "full")
    )
    if trainable_scope != "full":
        raise ValueError("locked Direct baseline tuning requires full fine-tuning")
    output_dir = (
        Path(config.payload["output"]["run_dir"]).expanduser().resolve()
        / args.model
        / args.center
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    plan = {
        "action": "pn2021_direct_k500_internal_tuning",
        "model": get_model_spec(args.model).describe(),
        "center": args.center,
        "config": config.describe(),
        "source_checkpoint": str(source_checkpoint),
        "expected_source_checkpoint_sha256": source_profile.get(
            "selected_checkpoint_sha256"
        ),
        "method_config": str(config.references["method_config"]),
        "trainable_scope": trainable_scope,
        "output_dir": str(output_dir),
        "device": args.device or config.online.payload["training"]["device"],
        "training_parameters": training_parameters,
        "dataloader_parameters": dataloader_parameters,
        "pos_weight": args.pos_weight,
        "selection": "deferred_to_four_center_pooled_raw_predictions",
        "dry_run_side_effects": "none",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source_checkpoint}")

    seed = config.online.payload["random_seed"]
    model_kwargs: dict[str, Any]
    if args.model == "efficientnet1dv2":
        model_kwargs = {
            "checkpoint_path": source_checkpoint,
            "map_location": "cpu",
        }
    else:
        model_kwargs = {
            "task_checkpoint_path": source_checkpoint,
            "trainable_scope": trainable_scope,
            "map_location": "cpu",
        }
    model = build_model(
        args.model,
        config_root=config.config_root,
        seed_namespace="pn2021_direct_tune_model_initialization",
        seed_identity=(
            seed["comparison_group"],
            seed["replicate_id"],
            args.model,
        ),
        **model_kwargs,
    )
    result = train_pn2021_direct_tuning(
        model,
        center=args.center,
        config_path=config.path,
        config_root=config.config_root,
        output_dir=output_dir,
        device=args.device,
        pos_weight=args.pos_weight,
        training_parameters=training_parameters,
        dataloader_parameters=dataloader_parameters,
    )
    print(json.dumps(result.describe(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
