#!/usr/bin/env python3
"""Generate the isolated F005 command-fingerprint golden."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ecg_adv_gen.config import build_runner_commands, load_experiment_config  # noqa: E402
from ecg_adv_gen.f005_control import F005_STUDY_SCOPE, F005_VARIANTS  # noqa: E402


CONFIGS = {
    "train": "configs/studies/f005_anchor_geometry_control_train.yaml",
    "smoke": "configs/studies/f005_anchor_geometry_control_smoke.yaml",
    "clean": "configs/studies/f005_anchor_geometry_control_pn2021_clean.yaml",
    "s5": "configs/studies/f005_anchor_geometry_control_pn2021c_s5.yaml",
    "depth23": "configs/studies/f005_anchor_geometry_control_pn2021c_depth23.yaml",
}
LOCAL = REPO / "configs/local/linbinhao_server.example.yaml"
OUTPUT = REPO / "configs/golden/f005_anchor_geometry_control_v1.json"
RUN_ID = "F005_ANCHOR_GEOMETRY_CONTROL_V1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize(argv: list[str], config: dict) -> list[str]:
    replacements = {
        str(config["_project_root"]): "<PROJECT_ROOT>",
        str(config["paths"]["data_root"]): "<DATA_ROOT>",
        str(config["paths"]["output_root"]): "<OUTPUT_ROOT>",
        str(config["python"]["executable"]): "<PYTHON_EXECUTABLE>",
    }
    out = []
    for item in argv:
        value = str(item)
        for source, replacement in replacements.items():
            value = value.replace(source, replacement)
        out.append(value)
    return out


def build() -> dict:
    stages = {}
    config_digests = {}
    counts = {}
    for stage, relative in CONFIGS.items():
        path = REPO / relative
        config = load_experiment_config(path, LOCAL, runtime_context={"run_id": RUN_ID})
        commands = build_runner_commands(config)
        counts[stage] = len(commands)
        config_digests[stage] = {"path": relative, "sha256": _sha(path.read_bytes())}
        rows = []
        for command in commands:
            matrix = command["matrix"]
            case = matrix["case"]
            normalized = _normalize(command["argv"], config)
            rows.append({
                "center": matrix["center"],
                "seed": int(case["seed"]),
                "variant": case["variant"],
                "command_name": command["name"],
                "normalized_argv_sha256": _sha("\0".join(normalized).encode("utf-8")),
                "model_dir": normalized[normalized.index("--model_dir") + 1]
                if "--model_dir" in normalized else "",
                "hull_lambda": normalized[normalized.index("--hull_lambda") + 1]
                if "--hull_lambda" in normalized else "",
            })
        stages[stage] = rows
    return {
        "schema_version": 1,
        "study_scope": F005_STUDY_SCOPE,
        "primary_comparison_eligible": False,
        "run_id": RUN_ID,
        "variants": list(F005_VARIANTS),
        "command_counts": dict(sorted(counts.items())),
        "config_digests": dict(sorted(config_digests.items())),
        "stages": dict(sorted(stages.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"stale F005 golden: {OUTPUT}", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
