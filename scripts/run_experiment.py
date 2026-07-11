#!/usr/bin/env python3
"""Experiment launcher for YAML-managed ECG_adv_Gen runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    inspect_execution_sources,
    require_clean_execution_sources,
    require_cuda_visible_devices,
    run_managed_commands,
    run_postprocess_commands,
    validate_experiment_config,
    verify_required_inputs,
    attach_replication_preflight,
)
from ecg_adv_gen.evidence import RunRecordError, finalize_run_record  # noqa: E402
from ecg_adv_gen.runner.launch_plan import (  # noqa: E402
    experiment_purpose,
    render_launch_command,
    write_launch_plan_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Tracked experiment YAML")
    parser.add_argument("--local-config", required=True, help="Gitignored local machine YAML")
    parser.add_argument("--run-id", required=True, help="Unique run id for manifest naming")
    parser.add_argument("--output-dir", default="", help="Optional run/plan output directory")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Resolve config and print commands only")
    mode.add_argument("--execute", action="store_true", help="Run managed commands after safety gates")
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="Write small resolved config and manifest files without invoking child scripts",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a matching managed run directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reuse a matching dry_run, launch_prepared, or failed managed run directory",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Whitelisted config override. Intended only for resource/training/method "
            "hyperparameters; paper protocol, paths, centers, K-shot, and runner argv are rejected."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.force:
        raise SystemExit("--resume and --force are mutually exclusive")

    try:
        config = load_experiment_config(
            Path(args.config),
            Path(args.local_config),
            overrides=args.set_overrides,
            runtime_context={"run_id": args.run_id},
        )
        local_paths = validate_experiment_config(config, repo_root=REPO_ROOT)
        commands = build_runner_commands(config)
        postprocess_commands = build_postprocess_commands(config)
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
        source_report = inspect_execution_sources(
            config,
            repo_root=REPO_ROOT,
            index_path=REPO_ROOT / "configs" / "active_scripts.yaml",
        )
        source_report["phase"] = "pre_execute" if args.execute else "diagnostic"
        manifest["execution_source_cleanliness"] = {
            "policy": (
                "execute requires active_scripts.yaml, the entry config, and every "
                "experiment _config_sources file to be clean, tracked, and present; "
                "_local_config_sources are excluded"
            ),
            "checks": [source_report],
        }
    except (ConfigError, OSError, ValueError) as exc:
        print(f"[config-error] {exc}", file=sys.stderr)
        return 2

    if args.execute:
        try:
            require_clean_execution_sources(source_report, phase="pre_execute")
            visible = require_cuda_visible_devices()
            gpu_snapshot = check_nvidia_smi()
        except (LaunchError, RunRecordError) as exc:
            label = "run-record-error" if isinstance(exc, RunRecordError) else "launch-error"
            print(f"[{label}] {exc}", file=sys.stderr)
            return 3
        manifest["gpu_prelaunch"] = {
            "CUDA_VISIBLE_DEVICES": visible,
            "nvidia_smi": gpu_snapshot,
        }
        manifest["status"] = "launch_prepared"
        manifest["launcher"]["execute"] = True
        manifest["safety"]["managed_child_commands_invoked"] = False

    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str))
    if args.execute:
        print("\n# Managed commands (will be executed after plan files and input verification)")
    else:
        print("\n# Managed commands (not executed)")
    for command in commands:
        print(render_launch_command(command))
    if postprocess_commands:
        print("\n# Managed postprocess commands")
        for command in postprocess_commands:
            print(render_launch_command(command))

    if args.write_plan or args.execute:
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
        except (LaunchError, ValueError) as exc:
            print(f"[launch-error] {exc}", file=sys.stderr)
            return 3
        try:
            manifest = write_launch_plan_files(out_dir, config, manifest, commands, postprocess_commands)
        except (LaunchError, RunRecordError) as exc:
            label = "run-record-error" if isinstance(exc, RunRecordError) else "launch-error"
            print(f"[{label}] {exc}", file=sys.stderr)
            return 3
        print(f"\nWrote run plan files: {out_dir}")

    if args.execute:
        try:
            manifest_path = out_dir / "run_manifest.json"
            input_verification = verify_required_inputs(manifest)
            if not input_verification["passed"]:
                manifest["status"] = "failed"
                manifest["input_verification"] = input_verification
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
                    encoding="utf-8",
                )
                raise LaunchError("Required input verification failed before managed commands")
            source_recheck = inspect_execution_sources(
                config,
                repo_root=REPO_ROOT,
                index_path=REPO_ROOT / "configs" / "active_scripts.yaml",
            )
            source_recheck["phase"] = "pre_child_invocation"
            manifest["execution_source_cleanliness"]["checks"].append(source_recheck)
            if not source_recheck["passed"]:
                manifest["status"] = "failed"
                manifest["input_verification"] = input_verification
                manifest["safety"]["managed_child_commands_invoked"] = False
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
                    encoding="utf-8",
                )
                require_clean_execution_sources(
                    source_recheck, phase="pre_child_invocation"
                )
            manifest["safety"].update(
                managed_child_commands_invoked=True,
                managed_child_commands_state="possible",
            )
            manifest["input_verification"] = input_verification
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
                encoding="utf-8",
            )
            run_managed_commands(commands, run_dir=out_dir, manifest_path=manifest_path)
            run_postprocess_commands(postprocess_commands, run_dir=out_dir, manifest_path=manifest_path)
            finalize_run_record(
                out_dir,
                purpose=experiment_purpose(config),
                result_summary=(
                    "Managed execution completed; artifact verification passed and replay metrics "
                    "are recorded in the finalized run record."
                ),
                outcome="succeeded",
            )
        except LaunchError as exc:
            print(f"[launch-error] {exc}", file=sys.stderr)
            return 3

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        print(
            "\nNote: real GPU launch will require explicit CUDA_VISIBLE_DEVICES after nvidia-smi check.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
