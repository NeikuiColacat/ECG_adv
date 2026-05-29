#!/usr/bin/env python3
"""Finalize an ECG_adv_Gen run directory for agent handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config.paths import is_under  # noqa: E402
from ecg_adv_gen.evidence import RunRecordError, finalize_run_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Managed run directory containing run_manifest.json")
    parser.add_argument("--purpose", default="", help="One-sentence experiment intent")
    parser.add_argument("--result-summary", default="", help="One-sentence result summary")
    parser.add_argument("--outcome", default="", help="trusted, provisional, deprecated, failed, or exploratory")
    parser.add_argument("--write-boundary", default="/home/linbinhao")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    boundary = Path(args.write_boundary).expanduser().resolve()
    if not is_under(run_dir, boundary):
        print(f"[run-record-error] run-dir is outside write boundary: {run_dir}", file=sys.stderr)
        return 2
    try:
        card = finalize_run_record(
            run_dir,
            purpose=args.purpose or None,
            result_summary=args.result_summary or None,
            outcome=args.outcome or None,
        )
    except RunRecordError as exc:
        print(f"[run-record-error] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(card, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
