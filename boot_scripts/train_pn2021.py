"""Boot one matched PN2021 K500 finite-recipe run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.online_trainer import (
    ALLOWED_CENTERS,
    DEFAULT_ONLINE_CONFIG_PATH,
    load_online_train_config,
    resolve_online_training_parameters,
)
from core.train_PN2021 import load_pn2021_recipe_spec, train_pn2021
from models import build_ecgtwin_vae, build_model, get_model_spec, load_vae_config
from util.config_bundle import resolve_config_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one manual PN2021 K500 finite recipe."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_ONLINE_CONFIG_PATH)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument(
        "--model", required=True, choices=("efficientnet1dv2", "ecgfounder")
    )
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--center", required=True, choices=ALLOWED_CENTERS)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--trainable-scope", choices=("full", "head"), default="full")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--scheduler-horizon-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--minimum-learning-rate-ratio", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument(
        "--amp", dest="amp_enabled", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"))
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
    parser.add_argument(
        "--drop-last", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--pos-weight", nargs=5, type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _not_none(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_online_train_config(args.config, config_root=args.config_root)
    recipe, _ = load_pn2021_recipe_spec(
        args.method_config,
        config_path=config.path,
        config_root=config.config_root,
    )
    training_parameters = _not_none(
        epochs=args.epochs,
        scheduler_horizon_epochs=args.scheduler_horizon_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
        amp_enabled=args.amp_enabled,
        amp_dtype=args.amp_dtype,
    )
    resolved_training, supplied_training = resolve_online_training_parameters(
        config, args.model, training_parameters
    )
    dataloader_parameters = _not_none(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        cache_mode=args.cache_mode,
        validate_values=args.validate_values,
        drop_last=args.drop_last,
    )
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    vae_checkpoint = (
        None if args.vae_checkpoint is None else args.vae_checkpoint.expanduser().resolve()
    )
    needs_pool = bool(recipe.requirements.latent_pool)
    needs_runtime_encoder = bool(recipe.requirements.vae_encoder)
    needs_decoder = bool(recipe.requirements.vae_decoder)
    needs_encoder_component = needs_pool or needs_runtime_encoder
    needs_vae = needs_encoder_component or needs_decoder
    if not needs_vae and vae_checkpoint is not None:
        raise ValueError("recipe without latent resources rejects --vae-checkpoint")
    resolved_vae_checkpoint = None
    vae_config_path = None
    if needs_vae:
        vae_resource = recipe.resources.get("vae")
        if not isinstance(vae_resource, Mapping):
            raise ValueError("latent recipe must declare a VAE config resource")
        vae_config_path = resolve_config_reference(
            vae_resource.get("path"),
            owner_config_path=recipe.source_path,
            config_root=config.config_root,
            description="method.resources.vae",
            must_exist=True,
        )
        vae_config = load_vae_config(vae_config_path)
        resolved_vae_checkpoint = vae_checkpoint or vae_config.checkpoint_path
    output_dir = (
        None if args.output_dir is None else args.output_dir.expanduser().resolve()
    )
    plan = {
        "action": "pn2021_k500_online_training",
        "model": get_model_spec(args.model).describe(),
        "method": recipe.describe(),
        "center": args.center,
        "config": config.describe(),
        "source_checkpoint": str(source_checkpoint),
        "vae_checkpoint": (
            None if not needs_vae else str(resolved_vae_checkpoint)
        ),
        "vae_requirements": {
            "latent_pool": needs_pool,
            "runtime_encoder": needs_runtime_encoder,
            "decoder": needs_decoder,
        },
        "output_dir": None if output_dir is None else str(output_dir),
        "device": args.device or config.payload["training"]["device"],
        "training_parameters": {
            "resolved": resolved_training,
            "explicit_overrides": supplied_training,
        },
        "dataloader_parameters": dataloader_parameters,
        "trainable_scope": args.trainable_scope,
        "pos_weight": args.pos_weight,
        "dry_run_side_effects": "none",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source_checkpoint}")

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
            "trainable_scope": args.trainable_scope,
            "map_location": "cpu",
        }
    model = build_model(
        args.model,
        config_root=config.config_root,
        seed_namespace="pn2021_model_initialization",
        seed_identity=(
            seed["comparison_group"],
            seed["replicate_id"],
            args.center,
            args.model,
        ),
        **model_kwargs,
    )

    encoder = decoder = None
    if needs_vae:
        assert vae_config_path is not None
        loaded_encoder, loaded_decoder = build_ecgtwin_vae(
            config_path=vae_config_path,
            checkpoint_path=resolved_vae_checkpoint,
            map_location="cpu",
        )
        encoder = loaded_encoder if needs_encoder_component else None
        decoder = loaded_decoder if needs_decoder else None
    result = train_pn2021(
        model,
        center=args.center,
        method_config_path=recipe.source_path,
        encoder=encoder,
        decoder=decoder,
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
