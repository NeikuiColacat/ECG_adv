"""CPU-only managed coordinator for the frozen eight-job PULSE training pair."""
from pathlib import Path
import argparse
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from util.pulse_training_queue import run
    run(args.config, args.config_root, args.output_dir)


if __name__ == "__main__":
    main()
