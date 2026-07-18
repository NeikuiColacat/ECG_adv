"""Select one global epoch from four managed PN2021 tuning runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.evaluation.direct_baseline_selection import (  # noqa: E402
    DEFAULT_DIRECT_TUNE_CONFIG,
    discover_validation_prediction_artifacts,
    load_direct_selection_config,
    select_direct_baseline_epoch,
)


FAILED_CLEAN_FLOOR_EXIT_CODE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pool four PN2021 validation100 runs and select global E*."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_DIRECT_TUNE_CONFIG)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument(
        "--center-run-dir",
        action="append",
        required=True,
        metavar="CENTER=DIR",
        help="Repeat once for every configured center; DIR owns validation_predictions/.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _center_run_dirs(values: Sequence[str], centers: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        center, separator, raw_path = str(value).partition("=")
        if not separator or center not in centers or not raw_path:
            raise ValueError(
                "--center-run-dir must use a configured CENTER=/path form"
            )
        if center in result:
            raise ValueError(f"duplicate --center-run-dir for {center}")
        result[center] = Path(raw_path).expanduser().resolve()
    if set(result) != set(centers):
        raise ValueError("--center-run-dir must cover every configured center exactly once")
    return result


def _write_selection_tensorboard(result: dict, output_dir: Path) -> Path:
    """Publish the pooled curve without mislabeling a per-center last epoch."""

    tensorboard_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    try:
        for item in result["per_epoch"]:
            epoch = int(item["epoch"])
            writer.add_scalar(
                "selection/clean_macro_auprc",
                float(item["clean"]["macro_auprc"]),
                epoch,
            )
            writer.add_scalar(
                "selection/robust_macro_auprc",
                float(item["robust"]["macro_auprc"]),
                epoch,
            )
            writer.add_scalar("selection/score", float(item["score"]), epoch)
            writer.add_scalar(
                "selection/clean_floor", float(item["clean_floor"]), epoch
            )
            writer.add_scalar(
                "selection/eligible", 1.0 if item["eligible"] else 0.0, epoch
            )
        selected_epoch = result.get("selected_epoch")
        if selected_epoch is not None:
            selected_epoch = int(selected_epoch)
            writer.add_scalar(
                "selection/global_selected_epoch", selected_epoch, selected_epoch
            )
        writer.add_text(
            "selection/protocol",
            "Four validation100 sets are pooled for clean and separately for "
            "each of twenty frozen corruptions. Score is 0.5 clean + 0.5 "
            "mean corruption AUPRC with a clean minus-1pp floor.",
            0 if selected_epoch is None else selected_epoch,
        )
        writer.flush()
    finally:
        writer.close()
    return tensorboard_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_direct_selection_config(
        args.config,
        config_root=args.config_root,
    )
    center_dirs = _center_run_dirs(args.center_run_dir, config.centers)
    output_dir = (
        None if args.output_dir is None else args.output_dir.expanduser().resolve()
    )
    plan = {
        "action": (
            "select_pn2021_direct_global_epoch"
            if config.method_id == "direct_depth23_fixed20"
            else "select_pn2021_global_epoch"
        ),
        "config": config.describe(),
        "center_run_dirs": {
            center: str(center_dirs[center]) for center in config.centers
        },
        "output": (
            None
            if output_dir is None
            else str(output_dir / config.output_file)
        ),
        "aggregation": "pool_four_centers_per_clean_or_composition",
        "metric": "0.5_clean_plus_0.5_mean20_macro_auprc_with_clean_floor",
        "dry_run_side_effects": "none",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if output_dir is None:
        raise ValueError("--output-dir is required outside dry-run mode")
    artifacts = discover_validation_prediction_artifacts(
        center_dirs,
        config_path=config.path,
        config_root=args.config_root,
    )
    result = select_direct_baseline_epoch(
        artifacts,
        config_path=config.path,
        config_root=args.config_root,
        output_path=output_dir,
    )
    result["tensorboard_dir"] = str(
        _write_selection_tensorboard(result, output_dir)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    status = str(result.get("status", ""))
    if status == "selected":
        return 0
    if status == "failed_clean_floor":
        return FAILED_CLEAN_FLOOR_EXIT_CODE
    raise RuntimeError(f"unexpected PN2021 selection status: {status!r}")


if __name__ == "__main__":
    raise SystemExit(main())
