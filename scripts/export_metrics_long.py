#!/usr/bin/env python3
"""Export existing experiment artifacts to metrics_long.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.reporting import MetricsExportError, export_metrics  # noqa: E402
from ecg_adv_gen.config.paths import PathSafetyError, is_under  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True, help="Artifact files or directories")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--expected-mapping-version", default="")
    p.add_argument("--expected-mapping-hash", default="")
    p.add_argument("--allow-mixed-mapping", action="store_true")
    p.add_argument("--no-require-mapping", action="store_true")
    p.add_argument(
        "--run-id",
        default="",
        help="Override run_id for all extracted metric rows, useful for grouping per-center artifacts.",
    )
    p.add_argument(
        "--filter-to-target-center",
        action="store_true",
        help=(
            "Keep only rows whose center matches the artifact target center. "
            "The center is read from JSON center or inferred from path prefixes."
        ),
    )
    p.add_argument(
        "--target-centers",
        nargs="*",
        default=[],
        help="Allowed centers for path-prefix target-center inference.",
    )
    p.add_argument("--write-boundary", default="/home/linbinhao")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    boundary = Path(args.write_boundary).expanduser().resolve()
    if not is_under(output_dir, boundary):
        print(f"[metrics-export-error] output-dir is outside write boundary {boundary}: {output_dir}", file=sys.stderr)
        return 2
    try:
        manifest = export_metrics(
            [Path(p) for p in args.input],
            output_dir,
            expected_mapping_version=args.expected_mapping_version or None,
            expected_mapping_hash=args.expected_mapping_hash or None,
            allow_mixed_mapping=args.allow_mixed_mapping,
            require_mapping=not args.no_require_mapping,
            run_id_override=args.run_id or None,
            filter_to_target_center=args.filter_to_target_center,
            target_centers=args.target_centers,
        )
    except (MetricsExportError, PathSafetyError, OSError) as exc:
        print(f"[metrics-export-error] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: manifest[k] for k in ["metrics_long", "artifact_manifest", "n_artifacts", "n_metric_rows", "views", "canonical_views", "mapping_pairs"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
