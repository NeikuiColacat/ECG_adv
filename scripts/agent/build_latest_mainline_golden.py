#!/usr/bin/env python3
"""Build or check the host-neutral latest-mainline golden contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config import (  # noqa: E402
    build_postprocess_commands,
    build_runner_commands,
    load_active_script_index,
    load_experiment_config,
    make_dry_run_manifest,
    managed_experiment_items,
    validate_experiment_config,
)

RUN_ID = "GOLDEN_LATEST_MAINLINE_V1"
DEFAULT_INDEX = REPO_ROOT / "configs/active_scripts.yaml"
DEFAULT_LOCAL = REPO_ROOT / "configs/local/linbinhao_server.example.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "configs/golden/latest_mainline_contract_v1.json"
VOLATILE_KEYS = frozenset(
    {"created_at_utc", "exists", "git", "inode", "mtime", "mtime_ns", "size_bytes"}
)
ARTIFACT_FIELDS = ("role", "path", "required", "sha256", "ref_hash")
def _ordered_replacements(replacements: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source, placeholder in replacements:
        source = str(source)
        if source and source not in seen:
            seen.add(source)
            unique.append((source, str(placeholder)))
    return sorted(unique, key=lambda item: len(item[0]), reverse=True)

def normalize_value(value: Any, *, replacements: Iterable[tuple[str, str]]) -> Any:
    """Replace configured roots and remove only explicit machine observations."""
    ordered = _ordered_replacements(replacements)

    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items() if key not in VOLATILE_KEYS}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if isinstance(item, Path):
            item = str(item)
        if isinstance(item, str):
            for source, placeholder in ordered:
                item = item.replace(source, placeholder)
        return item

    return normalize(value)

def _replacement_pairs(config: dict[str, Any], paths: dict[str, str]) -> list[tuple[str, str]]:
    fields = (
        ("python_executable", "<PYTHON_EXECUTABLE>"), ("project_root", "<PROJECT_ROOT>"),
        ("model_root", "<MODEL_ROOT>"), ("output_root", "<OUTPUT_ROOT>"),
        ("cache_root", "<CACHE_ROOT>"), ("short_tmp_root", "<SHORT_TMP_ROOT>"),
        ("tmp_root", "<TMP_ROOT>"), ("data_root", "<DATA_ROOT>"),
    )
    pairs = [(paths[key], placeholder) for key, placeholder in fields]
    pairs += [
        (str((config.get("host") or {}).get("home_root") or ""), "<HOME_ROOT>"),
        (paths["write_boundary"], "<WRITE_BOUNDARY>"),
    ]
    return _ordered_replacements(pairs)

def _artifact_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("role") and item.get("path"):
                record = {key: item[key] for key in ARTIFACT_FIELDS if key in item}
                identity = json.dumps(record, sort_keys=True, default=str)
                if identity not in seen:
                    seen.add(identity)
                    records.append(record)
                return
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return records

def _k500_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        metadata = {key: ref[key] for key in ("center", "k", "seed", "ref_hash") if key in ref}
        for artifact in _artifact_records(ref):
            projected = {**metadata, **artifact}
            identity = json.dumps(projected, sort_keys=True, default=str)
            if identity not in seen:
                seen.add(identity)
                output.append(projected)
    return output


def _command(command: dict[str, Any]) -> dict[str, str]:
    return {"name": str(command["name"]), "argv": shlex.join(str(part) for part in command["argv"])}


def build_contract(*, index_path: Path, local_config_path: Path) -> dict[str, Any]:
    index = load_active_script_index(index_path)
    latest = index["latest_mainline"]
    managed = {str(item["config"]): item for item in managed_experiment_items(index)}
    stage_configs = [str(stage["config"]) for stage in latest["stages"]]
    if len(stage_configs) != len(set(stage_configs)) or set(stage_configs) != set(managed):
        raise ValueError("latest_mainline stage configs must be unique and exactly match active managed configs")
    stages = []
    for stage in latest["stages"]:
        config_rel = str(stage["config"])
        if config_rel not in managed:
            raise ValueError(f"latest-mainline config is not active managed: {config_rel}")
        config = load_experiment_config(
            REPO_ROOT / config_rel, local_config_path, runtime_context={"run_id": RUN_ID}
        )
        paths = validate_experiment_config(config, repo_root=REPO_ROOT)
        commands = build_runner_commands(config)
        postprocess = build_postprocess_commands(config)
        manifest = make_dry_run_manifest(
            config, commands=commands, local_paths=paths, run_id=RUN_ID,
            cli_args=argparse.Namespace(dry_run=True, write_plan=False),
            postprocess_commands=postprocess,
        )
        trace = manifest["artifact_trace"]
        raw = {
            "name": stage["name"], "role": stage["role"], "config": config_rel,
            "resolved_experiment": manifest["experiment"]["name"],
            "method_family": managed[config_rel]["method_family"],
            "runner_adapter": config["runner"]["adapter"],
            "runner_entrypoint": config["runner"]["entrypoint"],
            "matrix_cases": [command["matrix"] for command in commands],
            "runner_commands": [_command(command) for command in commands],
            "postprocess_commands": [_command(command) for command in postprocess],
            "evaluation_views": trace["metrics"]["views"],
            "paper_protocol": {"kshot": manifest["paper_protocol"]["kshot"], "selection": trace["selection_policy"]},
            "checkpoint_artifacts": _artifact_records(trace["inputs"]["checkpoints"]),
            "k500_refs": _k500_refs(trace["inputs"]["k500_refs"]),
            "expected_artifact_roles": list(dict.fromkeys(
                record["role"] for record in _artifact_records(trace["expected_outputs"])
            )),
        }
        stages.append(normalize_value(raw, replacements=_replacement_pairs(config, paths)))
    return {
        "schema_version": 1, "run_id": RUN_ID,
        "normalization": {"removed_machine_state_keys": sorted(VOLATILE_KEYS), "replacement_order": "longest_configured_path_first"},
        "latest_mainline": {
            "method": latest["method"], "launcher": latest["launcher"],
            "mapping_version": latest["mapping_version"], "mapping_hash": latest["mapping_hash"],
            "class_order": list(latest["class_order"]), "stage_count": len(latest["stages"]),
        },
        "stages": stages,
    }


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--local-config", default=str(DEFAULT_LOCAL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = _path(args.output)
    rendered = json.dumps(build_contract(index_path=_path(args.index), local_config_path=_path(args.local_config)), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != rendered:
            print(f"[golden-missing-or-stale] {output}", file=sys.stderr)
            return 1
        print(f"[golden-ok] {output} sha256={digest}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"[golden-written] {output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
