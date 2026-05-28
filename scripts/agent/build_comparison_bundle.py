#!/usr/bin/env python3
"""Build the registry-managed Direct vs VAE comparison bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config.paths import is_under  # noqa: E402
from ecg_adv_gen.evidence import ComparisonBundleError, build_comparison_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default="configs/active_evidence_registry.yaml")
    p.add_argument("--local-config", default="configs/local/linbinhao_server.example.yaml")
    p.add_argument("--claim-id", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--write-boundary", default="/home/linbinhao")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None and not is_under(output_dir, Path(args.write_boundary).expanduser().resolve()):
        print(f"[comparison-bundle-error] output-dir is outside write boundary: {output_dir}", file=sys.stderr)
        return 2
    try:
        manifest = build_comparison_bundle(
            registry_path=REPO_ROOT / args.registry,
            local_config_path=REPO_ROOT / args.local_config,
            claim_id=args.claim_id or None,
            output_dir=output_dir,
            force=args.force,
        )
    except (ComparisonBundleError, OSError) as exc:
        print(f"[comparison-bundle-error] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
