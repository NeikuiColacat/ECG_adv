#!/usr/bin/env python3
"""Plan or execute one YAML-managed matrix config over a bounded GPU queue."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config import (  # noqa: E402
    ConfigError,
    LaunchError,
    attach_launch_artifacts,
    attach_replication_preflight,
    build_postprocess_commands,
    build_runner_commands,
    check_nvidia_smi,
    default_run_dir,
    inspect_execution_sources,
    load_experiment_config,
    make_dry_run_manifest,
    prepare_output_dir,
    require_clean_execution_sources,
    validate_experiment_config,
    verify_required_inputs,
)
from ecg_adv_gen.config.paths import is_under  # noqa: E402
from ecg_adv_gen.evidence import RunRecordError, finalize_run_record  # noqa: E402
from ecg_adv_gen.runner.launch_plan import (  # noqa: E402
    experiment_purpose,
    render_launch_command,
    write_launch_plan_files,
)
from ecg_adv_gen.runner.matrix_parallel import (  # noqa: E402
    assign_gpus_to_commands,
    atomic_write_json,
    bind_matrix_resume_contract,
    execution_lock,
    parse_gpu_list,
    read_json_object,
    run_matrix_queue,
    run_postprocess_serial,
    validate_gpu_snapshot,
    validate_matrix_resume_contract,
    validate_parallelism,
)


_SOURCE_POLICY = (
    "execute requires active_scripts.yaml, the entry config, and every experiment "
    "_config_sources file to be clean, tracked, and present; _local_config_sources are excluded"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--matrix-key", default="center")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--min-free-memory-mb",
        type=int,
        default=20_000,
        help="Minimum nvidia-smi free memory required on every selected GPU",
    )
    parser.add_argument(
        "--max-utilization-pct",
        type=int,
        default=10,
        help="Maximum prelaunch nvidia-smi utilization on every selected GPU",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.force and args.execute:
        parser.error("--force is dry-run only for the matrix queue; use --resume after execution")
    if args.min_free_memory_mb < 0:
        parser.error("--min-free-memory-mb must be non-negative")
    if not 0 <= args.max_utilization_pct <= 100:
        parser.error("--max-utilization-pct must be between 0 and 100")
    try:
        gpus = parse_gpu_list(args.gpus)
        parallel = len(gpus) if args.max_parallel is None else args.max_parallel
        args.max_parallel = validate_parallelism(parallel, gpus)
    except ValueError as exc:
        parser.error(str(exc))
    args.write_plan = True
    return args


def _resume_manifest(
    path: Path,
    *,
    commands: list[dict[str, Any]],
    postprocess_commands: list[dict[str, Any]],
    expected_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an existing plan without rewriting its manifest or resolved config."""

    manifest = read_json_object(path)
    if manifest.get("commands") != commands:
        raise LaunchError("resume manifest commands do not match the current resolved config")
    if manifest.get("postprocess_commands") != postprocess_commands:
        raise LaunchError("resume manifest postprocess commands do not match the current resolved config")
    for name in ("run_config.resolved.yaml", "run_config.resolved.json"):
        if not (path.parent / name).is_file():
            raise LaunchError(f"resume must preserve the existing resolved config, but {name} is missing")
    if expected_manifest is not None:
        validate_matrix_resume_contract(manifest, expected_manifest)
    return manifest


def _matrix_contract(
    *,
    assignments: list[Any],
    args: argparse.Namespace,
    gpus: list[str],
) -> dict[str, Any]:
    return {
        "enabled": True,
        "executor": "bounded_queue" if args.execute else "dry_run_preview",
        "matrix_key": args.matrix_key,
        "gpus": gpus,
        "max_parallel": args.max_parallel,
        "prelaunch_thresholds": {
            "min_free_memory_mb": args.min_free_memory_mb,
            "max_utilization_pct": args.max_utilization_pct,
        },
        "assignments": [
            {
                "command_index": item.command_index,
                "name": item.command.get("name"),
                "matrix_value": item.matrix_value,
                "preview_cuda_visible_devices": item.gpu,
            }
            for item in assignments
        ],
    }


def _inspect_source_check(
    config: dict[str, Any],
    *,
    phase: str,
    command_index: int | None = None,
    command_name: Any = None,
    attempt: int | None = None,
    dispatch_kind: str | None = None,
) -> dict[str, Any]:
    report = inspect_execution_sources(
        config,
        repo_root=REPO_ROOT,
        index_path=REPO_ROOT / "configs" / "active_scripts.yaml",
    )
    report["phase"] = phase
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    if command_index is not None:
        report.update(
            {
                "command_index": command_index,
                "command_name": command_name,
                "attempt": attempt,
                "dispatch_kind": dispatch_kind,
            }
        )
    return report


def _with_source_check(
    manifest: dict[str, Any],
    report: dict[str, Any],
    *,
    child_invoked: bool = False,
) -> dict[str, Any]:
    cleanliness = dict(manifest.get("execution_source_cleanliness") or {})
    cleanliness["policy"] = _SOURCE_POLICY
    cleanliness["checks"] = [*(cleanliness.get("checks") or []), report]
    safety = dict(manifest.get("safety") or {})
    safety["managed_child_commands_invoked"] = bool(
        safety.get("managed_child_commands_invoked") or child_invoked
    )
    manifest.update(
        {
            "execution_source_cleanliness": cleanliness,
            "safety": safety,
        }
    )
    return manifest


def _record_source_check(
    manifest_path: Path,
    report: dict[str, Any],
    *,
    child_invoked: bool = False,
) -> dict[str, Any]:
    manifest = _with_source_check(
        read_json_object(manifest_path), report, child_invoked=child_invoked
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


def _update_execution_lifecycle(
    manifest_path: Path,
    *,
    status: str,
    phase: str,
    error: BaseException | None = None,
    finished: bool = False,
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = read_json_object(manifest_path)
    now = datetime.now(timezone.utc).isoformat()
    lifecycle = dict(manifest.get("execution_lifecycle") or {})
    lifecycle.update({"status": status, "phase": phase, "updated_at_utc": now})
    lifecycle.setdefault("started_at_utc", now)
    if status == "running":
        lifecycle.pop("error_summary", None)
        lifecycle.pop("finished_at_utc", None)
        manifest.pop("error_summary", None)
        manifest.pop("finished_at_utc", None)
    if error is not None:
        summary = f"{type(error).__name__}: {str(error)}"[:1000]
        lifecycle["error_summary"] = summary
        manifest["error_summary"] = summary
    if finished:
        lifecycle["finished_at_utc"] = now
        manifest["finished_at_utc"] = now
    manifest.update(patch or {})
    manifest.update(
        {
            "status": status,
            "execution_phase": phase,
            "execution_lifecycle": lifecycle,
            "updated_at_utc": now,
        }
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


def _execute_pipeline(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    commands: list[dict[str, Any]],
    postprocess_commands: list[dict[str, Any]],
    out_dir: Path,
    manifest_path: Path,
    gpus: list[str],
) -> None:
    phase = "pre_execute"
    _update_execution_lifecycle(manifest_path, status="running", phase=phase)
    try:
        source_report = _inspect_source_check(config, phase=phase)
        _record_source_check(manifest_path, source_report)
        require_clean_execution_sources(source_report, phase=phase)

        phase = "gpu_preflight"
        _update_execution_lifecycle(manifest_path, status="running", phase=phase)
        snapshot = check_nvidia_smi()
        selected_rows = validate_gpu_snapshot(
            gpus,
            snapshot,
            min_free_memory_mb=args.min_free_memory_mb,
            max_utilization_pct=args.max_utilization_pct,
        )
        phase = "input_preflight"
        current = _update_execution_lifecycle(
            manifest_path,
            status="running",
            phase=phase,
            patch={
                "gpu_prelaunch": {
                    "nvidia_smi": snapshot,
                    "selected_gpus": selected_rows,
                    "child_cuda_visible_devices_policy": "one physical id per child",
                }
            },
        )
        input_verification = verify_required_inputs(current)
        current = _update_execution_lifecycle(
            manifest_path,
            status="running",
            phase=phase,
            patch={"input_verification": input_verification},
        )
        if not input_verification["passed"]:
            raise LaunchError("Required input verification failed before matrix commands")

        phase = "pre_child"
        _update_execution_lifecycle(manifest_path, status="running", phase=phase)
        source_report = _inspect_source_check(config, phase=phase)
        _record_source_check(manifest_path, source_report)
        require_clean_execution_sources(source_report, phase=phase)

        phase = "queue"
        _update_execution_lifecycle(manifest_path, status="running", phase=phase)

        def before_dispatch(
            index: int,
            command: Any,
            attempt: int,
            *,
            dispatch_kind: str = "matrix",
        ) -> None:
            nonlocal phase
            phase = "pre_child"
            _update_execution_lifecycle(manifest_path, status="running", phase=phase)
            report = _inspect_source_check(
                config,
                phase=phase,
                command_index=index,
                command_name=command.get("name"),
                attempt=attempt,
                dispatch_kind=dispatch_kind,
            )
            _record_source_check(
                manifest_path,
                report,
                child_invoked=report.get("passed") is True,
            )
            require_clean_execution_sources(report, phase=phase)
            phase = "postprocess" if dispatch_kind == "postprocess" else "queue"

        run_matrix_queue(
            commands,
            gpus=gpus,
            max_parallel=args.max_parallel,
            run_dir=out_dir,
            manifest_path=manifest_path,
            resume=args.resume,
            before_dispatch=before_dispatch,
            acquire_lock=False,
        )
        phase = "postprocess"
        _update_execution_lifecycle(manifest_path, status="running", phase=phase)
        run_postprocess_serial(
            postprocess_commands,
            run_dir=out_dir,
            manifest_path=manifest_path,
            before_dispatch=lambda index, command, attempt: before_dispatch(
                index,
                command,
                attempt,
                dispatch_kind="postprocess",
            ),
            acquire_lock=False,
        )
        phase = "finalizer"
        _update_execution_lifecycle(manifest_path, status="running", phase=phase)
        finalize_run_record(
            out_dir,
            purpose=experiment_purpose(config),
            result_summary=(
                "Bounded matrix execution and serial postprocess completed; all declared "
                "artifacts passed full verification."
            ),
            outcome="succeeded",
        )
        _update_execution_lifecycle(
            manifest_path,
            status="succeeded",
            phase="completed",
            finished=True,
        )
    except BaseException as exc:
        terminal = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        _update_execution_lifecycle(
            manifest_path,
            status=terminal,
            phase=phase,
            error=exc,
            finished=True,
        )
        raise


def _prepare_plan(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    commands: list[dict[str, Any]],
    postprocess_commands: list[dict[str, Any]],
    local_paths: dict[str, str],
    manifest: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    out_dir = prepare_output_dir(
        out_dir,
        local_paths=local_paths,
        run_id=args.run_id,
        config_hash=manifest["config_hash_sha256"],
        resume=args.resume,
        force=args.force,
    )
    manifest_path = out_dir / "run_manifest.json"
    expected = bind_matrix_resume_contract(
        attach_launch_artifacts(manifest, run_dir=out_dir)
    )
    if args.resume:
        existing = _resume_manifest(
            manifest_path,
            commands=commands,
            postprocess_commands=postprocess_commands,
            expected_manifest=expected,
        )
        for report in (
            (manifest.get("execution_source_cleanliness") or {}).get("checks") or []
        ):
            existing = _with_source_check(existing, report)
        atomic_write_json(manifest_path, existing)
        return out_dir, existing
    return out_dir, write_launch_plan_files(
        out_dir,
        config,
        expected,
        commands,
        postprocess_commands,
    )


def _record_setup_failure(
    manifest_path: Path,
    *,
    phase: str,
    error: BaseException,
) -> None:
    if not manifest_path.is_file():
        return
    try:
        current = read_json_object(manifest_path)
        lifecycle = current.get("execution_lifecycle") or {}
        if lifecycle.get("status") in {"failed", "interrupted", "succeeded"}:
            return
        _update_execution_lifecycle(
            manifest_path,
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            phase=phase,
            error=error,
            finished=True,
        )
    except Exception:
        # Preserve the original setup error when the manifest itself is unreadable.
        return


def _validate_lock_target(out_dir: Path, local_paths: dict[str, str]) -> Path:
    target = out_dir.expanduser().resolve()
    boundary = Path(local_paths["write_boundary"]).expanduser().resolve()
    if not is_under(target, boundary):
        raise LaunchError(f"output_dir={target} is outside write boundary {boundary}")
    return target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
        manifest = attach_replication_preflight(
            manifest,
            config,
            repo_root=REPO_ROOT,
            index_path=REPO_ROOT / "configs" / "active_scripts.yaml",
        )
        manifest = _with_source_check(
            manifest,
            _inspect_source_check(config, phase="diagnostic"),
        )
        manifest["matrix_parallel"] = _matrix_contract(
            assignments=assignments,
            args=args,
            gpus=gpus,
        )
        manifest["launcher"]["script"] = "scripts/agent/run_matrix_parallel.py"
        manifest["launcher"]["execute"] = args.execute
        manifest["safety"]["managed_child_commands_invoked"] = False
        manifest["status"] = "launch_prepared" if args.execute else "dry_run"
    except (ConfigError, LaunchError, OSError, ValueError) as exc:
        print(f"[config-error] {exc}", file=sys.stderr)
        return 2

    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str))
    print("\n# Matrix child commands" + (" (queued for execution)" if args.execute else " (not executed)"))
    for assignment in assignments:
        print(
            f"# command {assignment.command_index} matrix.{args.matrix_key}="
            f"{assignment.matrix_value} preview_gpu={assignment.gpu}"
        )
        print(render_launch_command(assignment.command))
    if postprocess_commands:
        print("\n# Serial postprocess commands")
        for command in postprocess_commands:
            print(render_launch_command(command))

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_run_dir(local_paths, args.run_id, dry_run=args.dry_run)
    )
    if not args.execute:
        try:
            out_dir, manifest = _prepare_plan(
                args=args,
                config=config,
                commands=commands,
                postprocess_commands=postprocess_commands,
                local_paths=local_paths,
                manifest=manifest,
                out_dir=out_dir,
            )
        except (LaunchError, RunRecordError, OSError, ValueError) as exc:
            print(f"[launch-error] {exc}", file=sys.stderr)
            return 3
        print(f"\nWrote run plan files: {out_dir}")
        return 0

    phase = "prepare_output"
    manifest_path = out_dir / "run_manifest.json"
    lock_acquired = False
    try:
        out_dir = _validate_lock_target(out_dir, local_paths)
        manifest_path = out_dir / "run_manifest.json"
        with execution_lock(out_dir):
            lock_acquired = True
            phase = "resume_contract" if args.resume else "plan"
            out_dir, manifest = _prepare_plan(
                args=args,
                config=config,
                commands=commands,
                postprocess_commands=postprocess_commands,
                local_paths=local_paths,
                manifest=manifest,
                out_dir=out_dir,
            )
            manifest_path = out_dir / "run_manifest.json"
            print(
                f"\nReusing run plan files: {out_dir}"
                if args.resume
                else f"\nWrote run plan files: {out_dir}"
            )
            _execute_pipeline(
                args=args,
                config=config,
                commands=commands,
                postprocess_commands=postprocess_commands,
                out_dir=out_dir,
                manifest_path=manifest_path,
                gpus=gpus,
            )
    except KeyboardInterrupt as exc:
        if lock_acquired:
            _record_setup_failure(manifest_path, phase=phase, error=exc)
        print(
            "[interrupted] matrix execution interrupted; only owned process groups were stopped",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        if lock_acquired:
            _record_setup_failure(manifest_path, phase=phase, error=exc)
        print(f"[launch-error] {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
