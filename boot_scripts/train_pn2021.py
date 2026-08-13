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
from core.train_PN2021 import (
    build_pn2021_k500_loader_plan,
    load_pn2021_recipe_spec,
    train_pn2021,
)
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_online_train_config(args.config, config_root=args.config_root)
    recipe, _ = load_pn2021_recipe_spec(
        args.method_config,
        config_path=config.path,
        config_root=config.config_root,
    )
    resolved_training, supplied_training = resolve_online_training_parameters(
        config, args.model
    )
    if supplied_training:
        raise AssertionError("PN2021 training parameters must be YAML-owned")
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    needs_pool = needs_decoder = needs_vae = recipe.requires_vae
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
        resolved_vae_checkpoint = vae_config.checkpoint_path
    output_dir = (
        None if args.output_dir is None else args.output_dir.expanduser().resolve()
    )
    loader_plan = build_pn2021_k500_loader_plan(
        center=args.center,
        model_name=args.model,
        config_path=config.path,
        config_root=config.config_root,
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
            "decoder": needs_decoder,
        },
        "output_dir": None if output_dir is None else str(output_dir),
        "device": config.payload["training"]["device"],
        "training_parameters": {
            "resolved": resolved_training,
            "source": "tracked_yaml",
        },
        "loader_plan": loader_plan.describe(),
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
        encoder = loaded_encoder if needs_pool else None
        decoder = loaded_decoder if needs_decoder else None
    result = train_pn2021(
        model,
        center=args.center,
        method_config_path=args.method_config,
        encoder=encoder,
        decoder=decoder,
        config_path=config.path,
        config_root=config.config_root,
        output_dir=output_dir,
    )
    print(json.dumps(result.describe(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
