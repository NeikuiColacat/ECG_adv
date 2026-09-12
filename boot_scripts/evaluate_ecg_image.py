"""Thin, finite image-LLM evaluation delegate for the managed YAML launcher."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import yaml
    payload = yaml.safe_load(args.config.read_text())
    if payload.get("schema_version") == 2:
        from util.evaluation.ecg_image_elastic import run
    else:
        from util.evaluation.ecg_image_llm import run

    run(args.config, args.config_root, args.output_dir)


if __name__ == "__main__":
    main()
