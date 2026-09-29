"""Finite hybrid search/evaluation delegate behind the managed YAML launcher."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
PULSE_PYTHON = Path("/home/linbinhao/ECG_adv_data/envs/pulse_llava_infer/bin/python")


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "config-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    from util.pulse_hybrid_contract import load_config
    config, refs = load_config(args.config, args.config_root)
    if config["mode"] == "evaluate":
        if Path(sys.prefix) != PULSE_PYTHON.parent.parent:
            os.environ.pop("LD_LIBRARY_PATH", None)
            os.execv(str(PULSE_PYTHON), [str(PULSE_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
        os.environ.update(HF_HOME="/home/linbinhao/ECG_adv_data/hf_cache/pulse", HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        from util.evaluation.pulse_hybrid_development import run
        run(config, refs, args.config_root, args.output_dir)
    elif config["mode"] == "fixed":
        from util.pulse_hybrid_workflow import run_fixed
        run_fixed(config, refs, args.config, args.config_root, args.output_dir)
    else:
        # Import SQLite/ICU before Torch can load the system's older libstdc++.
        import optuna
        from util.pulse_hybrid_workflow import run
        run(config, refs, args.config, args.config_root, args.output_dir)


if __name__ == "__main__":
    main()
