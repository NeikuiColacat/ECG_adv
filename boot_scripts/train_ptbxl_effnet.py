"""Boot the manual EfficientNet1DV2 PTB-XL Super5 source training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.supervised_trainer import DEFAULT_TRAIN_CONFIG, load_train_config
from core.train_PTBXL import build_ptbxl_loader_plan, train_ptbxl
from models import build_model, get_model_spec


MODEL_NAME = "efficientnet1dv2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the manual EfficientNet1DV2 on PTB-XL Super5."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _profile(config: Any) -> dict[str, Any]:
    profiles = config.payload.get("boot_models")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(MODEL_NAME), dict):
        raise ValueError(f"training config boot_models.{MODEL_NAME} is required")
    return dict(profiles[MODEL_NAME])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_train_config(args.config, config_root=args.config_root)
    profile = _profile(config)
    training_parameters = dict(profile.get("training_parameters") or {})
    checkpoint = profile.get("checkpoint_path")
    output_dir = args.output_dir or profile.get("output_dir")
    loader_plan = build_ptbxl_loader_plan(
        get_model_spec(MODEL_NAME),
        config_path=config.path,
        config_root=config.config_root,
    )
    plan = {
        "model": get_model_spec(MODEL_NAME).describe(),
        "config": config.describe(),
        "checkpoint_path": None if checkpoint is None else str(Path(checkpoint).resolve()),
        "output_dir": None if output_dir is None else str(Path(output_dir).resolve()),
        "device": config.payload["training"]["device"],
        "training_parameters": training_parameters,
        "loader_plan": loader_plan.describe(),
        "pos_weight": None,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    model = build_model(
        MODEL_NAME,
        config_root=config.config_root,
        seed_namespace="ptbxl_model_initialization",
        seed_identity=(config.profile_name,),
        checkpoint_path=checkpoint,
        map_location="cpu",
    )
    result = train_ptbxl(
        model,
        config_path=config.path,
        config_root=config.config_root,
        output_dir=output_dir,
        training_parameters=training_parameters,
    )
    print(json.dumps(result.describe(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
