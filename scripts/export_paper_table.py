#!/usr/bin/env python3
"""Export a compact paper table from metrics_long.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config.paths import is_under  # noqa: E402
from ecg_adv_gen.reporting import PaperTableError, export_paper_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-long", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--view",
        required=True,
        help="Canonical view; required to avoid mixing all-zero and drop-all-zero views",
    )
    p.add_argument("--dataset", default="")
    p.add_argument("--centers", nargs="*", default=None)
    p.add_argument("--baseline-run-id", default="")
    p.add_argument("--no-dataset-rows", action="store_true")
    p.add_argument("--no-center-mean", action="store_true")
    p.add_argument("--allow-mixed-mapping", action="store_true")
    p.add_argument("--write-boundary", default="/home/linbinhao")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    boundary = Path(args.write_boundary).expanduser().resolve()
    if not is_under(output_dir, boundary):
        print(f"[paper-table-error] output-dir is outside write boundary {boundary}: {output_dir}", file=sys.stderr)
        return 2
    try:
        manifest = export_paper_table(
            Path(args.metrics_long),
            output_dir,
            view=args.view,
            dataset=args.dataset or None,
            centers=args.centers,
            include_dataset_rows=not args.no_dataset_rows,
            include_center_mean=not args.no_center_mean,
            baseline_run_id=args.baseline_run_id or None,
            allow_mixed_mapping=args.allow_mixed_mapping,
        )
    except (PaperTableError, OSError) as exc:
        print(f"[paper-table-error] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
