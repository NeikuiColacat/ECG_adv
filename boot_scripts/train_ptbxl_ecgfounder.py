"""Boot the manual ECGFounder PTB-XL Super5 source fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.supervised_trainer import DEFAULT_TRAIN_CONFIG
from core.train_PTBXL import run_ptbxl_boot


MODEL_NAME = "ecgfounder"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the manual ECGFounder on PTB-XL Super5.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, choices=(10,))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_ptbxl_boot(MODEL_NAME, build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
