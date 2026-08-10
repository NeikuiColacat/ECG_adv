"""Evaluate one rebuilt Super5 checkpoint on clean PN2021 and PN2021-C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.checkpoints import load_model_checkpoint
from models.factory import available_models, build_model, get_model_spec
from util.evaluation.metrics import resolve_evaluated_center_mean
from util.evaluation.pn2021 import (
    DEFAULT_PN2021_EVAL_CONFIG,
    LOGICAL_CENTERS,
    build_evaluation_plan,
    evaluate_pn2021,
    load_pn2021_eval_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a strict task checkpoint on four ref-excluded PN2021 "
            "centers and all 20 PN2021-C depth2/depth3 views."
        )
    )
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_PN2021_EVAL_CONFIG)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--amp",
        dest="amp_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"))
    parser.add_argument(
        "--center",
        action="append",
        choices=LOGICAL_CENTERS,
        help="Evaluate only the named target center; repeat to select a subset.",
    )
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
    depth2 = aggregates["depth2"]
    depth3 = aggregates["depth3"]
    depth23 = aggregates["depth23"]
    assert isinstance(depth2, dict)
    assert isinstance(depth3, dict)
    assert isinstance(depth23, dict)
    center_order = clean.get("center_order", protocol.get("logical_centers", []))
    assert isinstance(center_order, list)
    center_count = clean.get("center_count", len(center_order))

    def verified_four_center_mean(payload: dict[str, object]) -> object:
        payload_center_order = payload.get("center_order", center_order)
        payload_center_count = payload.get("center_count", center_count)
        if (
            isinstance(payload_center_count, int)
            and not isinstance(payload_center_count, bool)
            and payload_center_count == len(LOGICAL_CENTERS)
            and payload_center_order == list(LOGICAL_CENTERS)
        ):
            return payload.get("four_center_mean")
        return None

    return {
        "status": result["status"],
        "evaluation_profile": result["evaluation_profile"],
        "model": result["model"],
        "checkpoint": result["checkpoint"],
        "center_count": center_count,
        "center_order": center_order,
        "clean_evaluated_center_mean": resolve_evaluated_center_mean(clean),
        "pn2021c_depth2_evaluated_center_mean": resolve_evaluated_center_mean(
            depth2
        ),
        "pn2021c_depth3_evaluated_center_mean": resolve_evaluated_center_mean(
            depth3
        ),
        "pn2021c_depth23_evaluated_center_mean": resolve_evaluated_center_mean(
            depth23
        ),
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
    plan = build_evaluation_plan(
        model_spec=spec,
        checkpoint_path=args.checkpoint,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        amp_enabled=args.amp_enabled,
        amp_dtype=args.amp_dtype,
        logical_centers=args.center,
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
        model,
        args.checkpoint,
        map_location="cpu",
        strict=True,
    )
    result = evaluate_pn2021(
        model,
        checkpoint_identity,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        amp_enabled=args.amp_enabled,
        amp_dtype=args.amp_dtype,
        logical_centers=args.center,
    )
    print(json.dumps(_summary(result), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
