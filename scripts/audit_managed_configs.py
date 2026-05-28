#!/usr/bin/env python3
"""CPU-only audit for active YAML-managed experiment configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config import ConfigError, audit_active_managed_configs, write_audit_report  # noqa: E402
from ecg_adv_gen.config.paths import is_under  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="configs/active_scripts.yaml")
    p.add_argument("--local-config", required=True)
    p.add_argument("--output-dir", default="")
    p.add_argument("--require-existing-inputs", action="store_true")
    p.add_argument("--write-boundary", default="/home/linbinhao")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_active_managed_configs(
            repo_root=REPO_ROOT,
            index_path=(REPO_ROOT / args.index).resolve(),
            local_config_path=Path(args.local_config).expanduser().resolve(),
            require_existing_inputs=args.require_existing_inputs,
        )
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
            boundary = Path(args.write_boundary).expanduser().resolve()
            if not is_under(output_dir, boundary):
                raise ConfigError(f"output-dir is outside write boundary {boundary}: {output_dir}")
            report["output"] = write_audit_report(report, output_dir)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"[managed-config-audit-error] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
