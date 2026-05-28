#!/usr/bin/env python3
"""Backfill a run-level manifest for the legacy v7 VAE L-HAT evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config.paths import is_under  # noqa: E402
from ecg_adv_gen.evidence import EvidenceAuditError, build_legacy_vae_lhat_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default="configs/active_evidence_registry.yaml")
    p.add_argument("--local-config", default="configs/local/linbinhao_server.example.yaml")
    p.add_argument("--claim-id", default="effnet_v7_vae_lhat_improves_direct_k500")
    p.add_argument("--method-key", default="vae_lhat")
    p.add_argument("--output-path", default="")
    p.add_argument("--skip-checkpoint-hash", action="store_true")
    p.add_argument("--write-boundary", default="/home/linbinhao")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else None
    boundary = Path(args.write_boundary).expanduser().resolve()
    if output_path is not None and not is_under(output_path, boundary):
        print(f"[backfill-error] output-path is outside write boundary: {output_path}", file=sys.stderr)
        return 2
    try:
        manifest = build_legacy_vae_lhat_manifest(
            repo_root=REPO_ROOT,
            registry_path=REPO_ROOT / args.registry,
            local_config_path=REPO_ROOT / args.local_config,
            claim_id=args.claim_id,
            method_key=args.method_key,
            output_path=output_path,
            write_boundary=boundary,
            hash_checkpoints=not args.skip_checkpoint_hash,
        )
    except EvidenceAuditError as exc:
        print(f"[backfill-error] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
