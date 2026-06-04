#!/usr/bin/env python3
"""Validate host-local external model repository handles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config import ConfigError  # noqa: E402
from ecg_adv_gen.config.external_models import audit_external_model_links, external_model_specs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--local-config", default="configs/local/linbinhao_server.example.yaml")
    parser.add_argument("--no-require-existing", action="store_true")
    parser.add_argument(
        "--emit-bootstrap-plan",
        action="store_true",
        help="Print tab-separated name, url, target rows for bootstrap_model_repos.sh.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    local_config = Path(args.local_config).expanduser()
    if not local_config.is_absolute():
        local_config = repo_root / local_config

    try:
        if args.emit_bootstrap_plan:
            for item in external_model_specs(repo_root=repo_root, local_config_path=local_config):
                print(f"{item['name']}\t{item['url']}\t{item['target']}")
            return 0
        report = audit_external_model_links(
            repo_root=repo_root,
            local_config_path=local_config,
            require_existing=not args.no_require_existing,
        )
    except ConfigError as exc:
        print(f"[external-model-error] {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
