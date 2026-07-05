#!/usr/bin/env python3
"""Plan one YAML-managed matrix config with explicit one-GPU-per-command assignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config import (  # noqa: E402
    ConfigError,
    LaunchError,
    build_postprocess_commands,
    build_runner_commands,
    default_run_dir,
    load_experiment_config,
    make_dry_run_manifest,
    prepare_output_dir,
    validate_experiment_config,
)
from ecg_adv_gen.runner.launch_plan import render_launch_command, write_launch_plan_files  # noqa: E402
from ecg_adv_gen.runner.matrix_parallel import assign_gpus_to_commands, parse_gpu_list  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids, one per matrix command")
    parser.add_argument("--matrix-key", default="center")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.resume and args.force:
        raise SystemExit("--resume and --force are mutually exclusive")
    args.write_plan = True
    return args


def main() -> int:
    args = parse_args()
    try:
        gpus = parse_gpu_list(args.gpus)
        config = load_experiment_config(
            Path(args.config),
            Path(args.local_config),
            runtime_context={"run_id": args.run_id},
        )
        local_paths = validate_experiment_config(config, repo_root=REPO_ROOT)
        commands = build_runner_commands(config)
        postprocess_commands = build_postprocess_commands(config)
        assignments = assign_gpus_to_commands(commands, gpus, matrix_key=args.matrix_key)
        manifest = make_dry_run_manifest(
            config,
            commands=commands,
            local_paths=local_paths,
            run_id=args.run_id,
            cli_args=args,
            postprocess_commands=postprocess_commands,
        )
        manifest["matrix_parallel"] = {
            "enabled": True,
            "matrix_key": args.matrix_key,
            "assignments": [
                {
                    "command_index": item.command_index,
                    "name": item.command.get("name"),
                    "matrix_value": item.matrix_value,
                    "cuda_visible_devices": item.gpu,
                }
                for item in assignments
            ],
        }
        manifest["launcher"]["script"] = "scripts/agent/run_matrix_parallel.py"
        manifest["launcher"]["execute"] = False
        manifest["safety"]["managed_child_commands_invoked"] = False
        manifest["status"] = "dry_run"
    except (ConfigError, LaunchError, OSError, ValueError) as exc:
        print(f"[config-error] {exc}", file=sys.stderr)
        return 2

    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str))
    print("\n# Planned matrix child commands; not executed by this tool")
    for assignment in assignments:
        print(f"# command {assignment.command_index} matrix.{args.matrix_key}={assignment.matrix_value} gpu={assignment.gpu}")
        print(render_launch_command(assignment.command))

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = default_run_dir(local_paths, args.run_id, dry_run=True)
    try:
        out_dir = prepare_output_dir(
            out_dir,
            local_paths=local_paths,
            run_id=args.run_id,
            config_hash=manifest["config_hash_sha256"],
            resume=args.resume,
            force=args.force,
        )
        manifest = write_launch_plan_files(out_dir, config, manifest, commands, postprocess_commands)
    except (LaunchError, ValueError) as exc:
        print(f"[launch-error] {exc}", file=sys.stderr)
        return 3
    print(f"\nWrote run plan files: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
