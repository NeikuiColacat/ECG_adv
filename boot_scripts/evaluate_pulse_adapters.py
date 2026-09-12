"""Finite paired PULSE adapter admission or complete benchmark delegate."""
from pathlib import Path
import argparse
import os
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text())
    pulse_python = Path("/home/linbinhao/ECG_adv_data/envs/pulse_llava_infer/bin/python")
    if config["stage"] == "admission" and Path(sys.prefix) != pulse_python.parent.parent:
        os.execv(str(pulse_python), [str(pulse_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    os.environ["HF_HOME"] = "/home/linbinhao/ECG_adv_data/hf_cache/pulse"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from util.evaluation.pulse_benchmark import run
    run(args.config, args.config_root, args.output_dir)


if __name__ == "__main__":
    main()
