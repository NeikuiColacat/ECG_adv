#!/usr/bin/env python3
"""CPU-only audit of the active agent operating layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config.paths import is_under  # noqa: E402
from ecg_adv_gen.evidence import EvidenceAuditError, audit_active_evidence_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default="configs/active_evidence_registry.yaml")
    p.add_argument("--local-config", default="configs/local/linbinhao_server.example.yaml")
    p.add_argument("--output-dir", default="")
    p.add_argument("--skip-existing-artifacts", action="store_true")
    p.add_argument("--skip-git", action="store_true")
    p.add_argument("--write-boundary", default="/home/linbinhao")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None and not is_under(output_dir, Path(args.write_boundary).expanduser().resolve()):
        print(f"[agent-audit-error] output-dir is outside write boundary: {output_dir}", file=sys.stderr)
        return 2
    try:
        report = audit_active_evidence_registry(
            repo_root=REPO_ROOT,
            registry_path=REPO_ROOT / args.registry,
            local_config_path=REPO_ROOT / args.local_config,
            require_existing_artifacts=not args.skip_existing_artifacts,
            check_git=not args.skip_git,
        )
    except EvidenceAuditError as exc:
        print(f"[agent-audit-error] {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, default=str)
    print(text)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "agent_workspace_audit.json").write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
