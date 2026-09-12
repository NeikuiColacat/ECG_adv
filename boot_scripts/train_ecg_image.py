"""Finite PULSE training delegate; package environment is project-owned."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
PULSE_PYTHON = Path("/home/linbinhao/ECG_adv_data/envs/pulse_llava_infer/bin/python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if Path(sys.prefix) != PULSE_PYTHON.parent.parent:
        os.execv(str(PULSE_PYTHON), [str(PULSE_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    os.environ["HF_HOME"] = "/home/linbinhao/ECG_adv_data/hf_cache/pulse"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(REPO))
    from core.pulse_finetune import run
    run(args.config, args.config_root, args.output_dir)


if __name__ == "__main__":
    main()
