"""Managed, bounded K500 PULSE component timing delegate."""
import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("optimization", "logits", "deployment", "deployment_cache", "deployment_admission"), default="optimization")
    parser.add_argument("--center", choices=("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"))
    args = parser.parse_args()
    python = Path("/home/linbinhao/ECG_adv_data/envs/pulse_llava_infer/bin/python")
    if Path(sys.prefix) != python.parent.parent:
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    os.environ["HF_HOME"] = "/home/linbinhao/ECG_adv_data/hf_cache/pulse"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from util.evaluation.pulse_profile import run
    run(args.config, args.config_root, args.output_dir, suite=args.suite, center=args.center)


if __name__ == "__main__":
    main()
