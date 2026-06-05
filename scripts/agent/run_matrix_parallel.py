#!/usr/bin/env python3
"""Run one YAML-managed matrix config with explicit one-GPU-per-command assignment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config import (  # noqa: E402
    ConfigError,
    LaunchError,
    build_postprocess_commands,
    build_runner_commands,
    check_nvidia_smi,
    default_run_dir,
    load_experiment_config,
    make_dry_run_manifest,
    prepare_output_dir,
    run_postprocess_commands,
    validate_experiment_config,
    verify_required_artifacts,
    verify_required_inputs,
)
from ecg_adv_gen.config.launch import (  # noqa: E402
    _append_manifest_record,
    _load_manifest,
    _prepare_expected_artifact_dirs,
    _run_logged_subprocess,
    _utc_now,
    update_manifest_file,
)
from ecg_adv_gen.evidence import finalize_run_record  # noqa: E402
from ecg_adv_gen.runner.launch_plan import experiment_purpose, render_launch_command, write_launch_plan_files  # noqa: E402
from ecg_adv_gen.runner.matrix_parallel import assign_gpus_to_commands, parse_gpu_list  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids, one per matrix command")
    parser.add_argument("--matrix-key", default="center")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        raise SystemExit("exactly one of --dry-run or --execute is required")
    if args.resume and args.force:
        raise SystemExit("--resume and --force are mutually exclusive")
    args.write_plan = True
    return args


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n")


def _run_assigned_command(
    *,
    assignment,
    run_dir: Path,
    manifest_path: Path,
    base_env: dict[str, str],
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    _prepare_expected_artifact_dirs(manifest, assignment.command_index)
    command = dict(assignment.command)
    env = dict(base_env)
    env.update({str(k): str(v) for k, v in (command.get("env") or {}).items()})
    env["CUDA_VISIBLE_DEVICES"] = str(assignment.gpu)
    for key in ("TMPDIR", "XDG_CACHE_HOME"):
        if env.get(key):
            Path(env[key]).expanduser().mkdir(parents=True, exist_ok=True)
    ret, log_paths, started_at, finished_at = _run_logged_subprocess(
        command,
        env=env,
        run_dir=run_dir,
        logs_dir=run_dir / "logs",
        prefix="command",
        idx=assignment.command_index,
    )
    return {
        "command_index": assignment.command_index,
        "name": command.get("name"),
        "cwd": command.get("cwd"),
        "matrix_key": assignment.matrix_key,
        "matrix_value": assignment.matrix_value,
        "cuda_visible_devices": str(assignment.gpu),
        **log_paths,
        "returncode": ret,
        "status": "succeeded" if ret == 0 else "failed",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
    }


def _run_parallel_commands(
    *,
    assignments,
    run_dir: Path,
    manifest_path: Path,
    base_env: dict[str, str],
) -> None:
    update_manifest_file(manifest_path, {"status": "running", "execution_started_at_utc": _utc_now()})
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
        futures = [
            executor.submit(
                _run_assigned_command,
                assignment=assignment,
                run_dir=run_dir,
                manifest_path=manifest_path,
                base_env=base_env,
            )
            for assignment in assignments
        ]
        for future in as_completed(futures):
            records.append(future.result())

    failed = [record for record in records if int(record["returncode"]) != 0]
    for record in sorted(records, key=lambda item: int(item["command_index"])):
        _append_manifest_record(manifest_path, "command_runs", record)
    if failed:
        first = sorted(failed, key=lambda item: int(item["command_index"]))[0]
        update_manifest_file(
            manifest_path,
            {
                "status": "failed",
                "failed_command_index": first["command_index"],
                "failed_command_name": first["name"],
                "failed_returncode": first["returncode"],
                "failed_log": first["log_path"],
                "failed_stdout_log": first["stdout_log_path"],
                "failed_stderr_log": first["stderr_log_path"],
            },
        )
        raise LaunchError(
            f"Parallel command {first['command_index']} failed with return code "
            f"{first['returncode']}; see {first['log_path']}"
        )

    manifest = _load_manifest(manifest_path)
    verification = verify_required_artifacts(manifest, include_postprocess=False)
    if not verification["passed"]:
        update_manifest_file(manifest_path, {"status": "failed", "artifact_verification": verification})
        raise LaunchError("Required artifact verification failed after parallel child commands")
    update_manifest_file(
        manifest_path,
        {
            "status": "succeeded",
            "artifact_verification": verification,
            "child_commands_finished_at_utc": _utc_now(),
        },
    )


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
        if args.execute:
            manifest["gpu_prelaunch"] = {
                "CUDA_VISIBLE_DEVICES": ",".join(gpus),
                "mode": "matrix_parallel_one_gpu_per_command",
                "nvidia_smi": check_nvidia_smi(),
            }
            manifest["status"] = "launch_prepared"
            manifest["launcher"]["execute"] = True
            manifest["safety"]["legacy_child_scripts_invoked"] = False
    except (ConfigError, LaunchError, OSError, ValueError) as exc:
        print(f"[config-error] {exc}", file=sys.stderr)
        return 2

    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str))
    print("\n# Legacy commands")
    for assignment in assignments:
        print(f"# command {assignment.command_index} matrix.{args.matrix_key}={assignment.matrix_value} gpu={assignment.gpu}")
        print(render_launch_command(assignment.command))

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = default_run_dir(local_paths, args.run_id, dry_run=args.dry_run)
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

    if args.dry_run:
        return 0

    manifest_path = out_dir / "run_manifest.json"
    try:
        input_verification = verify_required_inputs(manifest)
        if not input_verification["passed"]:
            manifest["status"] = "failed"
            manifest["input_verification"] = input_verification
            _write_manifest(manifest_path, manifest)
            raise LaunchError("Required input verification failed before legacy commands")
        manifest["safety"]["legacy_child_scripts_invoked"] = True
        manifest["input_verification"] = input_verification
        _write_manifest(manifest_path, manifest)
        _run_parallel_commands(
            assignments=assignments,
            run_dir=out_dir,
            manifest_path=manifest_path,
            base_env=os.environ.copy(),
        )
        run_postprocess_commands(postprocess_commands, run_dir=out_dir, manifest_path=manifest_path)
        finalize_run_record(
            out_dir,
            purpose=experiment_purpose(config),
            result_summary=(
                "Matrix-parallel managed execution completed; artifact verification passed "
                "and replay metrics are recorded in the finalized run record."
            ),
            outcome="succeeded",
        )
    except LaunchError as exc:
        print(f"[launch-error] {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
