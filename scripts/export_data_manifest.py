#!/usr/bin/env python3
"""Export a lightweight data path manifest for a YAML-managed experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config import ConfigError, load_experiment_config, validate_experiment_config  # noqa: E402
from ecg_adv_gen.config.paths import is_under  # noqa: E402
from ecg_adv_gen.data import DataManifestError, write_data_path_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Tracked experiment YAML")
    p.add_argument("--local-config", required=True, help="Gitignored local machine YAML")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--require-existing", action="store_true")
    p.add_argument(
        "--include-counts",
        action="store_true",
        help="Count immediate PN2021 *.hea files per center without loading waveforms",
    )
    p.add_argument("--write-boundary", default="/home/linbinhao")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    boundary = Path(args.write_boundary).expanduser().resolve()
    if not is_under(output_dir, boundary):
        print(f"[data-manifest-error] output-dir is outside write boundary {boundary}: {output_dir}", file=sys.stderr)
        return 2
    try:
        config = load_experiment_config(
            Path(args.config),
            Path(args.local_config),
            runtime_context={"run_id": "data_manifest"},
        )
        local_paths = validate_experiment_config(config, repo_root=REPO_ROOT)
        manifest = write_data_path_manifest(
            config,
            output_dir / "data_manifest.json",
            local_paths=local_paths,
            require_existing=args.require_existing,
            include_counts=args.include_counts,
        )
    except (ConfigError, DataManifestError, OSError, ValueError) as exc:
        print(f"[data-manifest-error] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "data_manifest": str(output_dir / "data_manifest.json"),
        "experiment_name": manifest["experiment_name"],
        "summary": manifest["summary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
