#!/usr/bin/env python3
"""Register a finalized run in the active evidence registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config.paths import is_under  # noqa: E402
from ecg_adv_gen.evidence import RunRecordError, register_run_in_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Finalized run directory containing run_card.json")
    parser.add_argument("--status", default="provisional", choices=["trusted", "provisional", "deprecated", "failed", "exploratory"])
    parser.add_argument("--registry", default="configs/active_evidence_registry.yaml")
    parser.add_argument("--local-config", default="configs/local/linbinhao_server.example.yaml")
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
        registry = register_run_in_registry(
            registry_path=REPO_ROOT / args.registry,
            local_config_path=REPO_ROOT / args.local_config,
            run_dir=run_dir,
            status=args.status,
        )
    except RunRecordError as exc:
        print(f"[run-record-error] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"registered": True, "managed_run_count": len(registry.get("managed_runs") or [])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
