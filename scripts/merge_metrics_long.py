#!/usr/bin/env python3
"""Merge exported metrics_long.csv files into one paper-table input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config.paths import is_under  # noqa: E402
from ecg_adv_gen.reporting import MetricsMergeError, merge_metrics_long  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True, help="metrics_long.csv files")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--allow-mixed-mapping", action="store_true")
    p.add_argument("--write-boundary", default="/home/linbinhao")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    boundary = Path(args.write_boundary).expanduser().resolve()
    if not is_under(output_dir, boundary):
        print(f"[metrics-merge-error] output-dir is outside write boundary {boundary}: {output_dir}", file=sys.stderr)
        return 2
    try:
        manifest = merge_metrics_long(
            [Path(p) for p in args.input],
            output_dir,
            allow_mixed_mapping=args.allow_mixed_mapping,
        )
    except (MetricsMergeError, OSError) as exc:
        print(f"[metrics-merge-error] {exc}", file=sys.stderr)
        return 2
    keys = [
        "metrics_long",
        "merge_manifest",
        "n_input_files",
        "n_metric_rows",
        "run_ids",
        "canonical_views",
        "mapping_pairs",
    ]
    print(json.dumps({k: manifest[k] for k in keys}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
