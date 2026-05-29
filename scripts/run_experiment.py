#!/usr/bin/env python3
"""Experiment launcher for YAML-managed ECG_adv_Gen runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.config import (  # noqa: E402
    ConfigError,
    LaunchError,
    attach_launch_artifacts,
    build_postprocess_commands,
    build_runner_commands,
    check_nvidia_smi,
    default_run_dir,
    load_experiment_config,
    make_dry_run_manifest,
    prepare_output_dir,
    require_cuda_visible_devices,
    run_legacy_commands,
    run_postprocess_commands,
    validate_experiment_config,
    verify_required_inputs,
    write_k500_ref_ids_artifact,
    write_selection_record_artifact,
)
from ecg_adv_gen.data import write_data_path_manifest  # noqa: E402
from ecg_adv_gen.evidence import finalize_run_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Tracked experiment YAML")
    parser.add_argument("--local-config", required=True, help="Gitignored local machine YAML")
    parser.add_argument("--run-id", required=True, help="Unique run id for manifest naming")
    parser.add_argument("--output-dir", default="", help="Optional run/plan output directory")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Resolve config and print commands only")
    mode.add_argument("--execute", action="store_true", help="Run legacy commands after safety gates")
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


def _shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def _render_command_with_env(command: dict) -> str:
    env = command.get("env") or {}
    env_prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted(env.items()))
    rendered = _shell_join(command["argv"])
    return f"{env_prefix} {rendered}".strip()


def _experiment_purpose(config: dict) -> str:
    experiment = config.get("experiment") or {}
    return str(experiment.get("purpose") or experiment.get("description") or "")


def _write_plan_files(
    out_dir: Path,
    config: dict,
    manifest: dict,
    commands: list[dict],
    postprocess_commands: list[dict],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    (out_dir / "run_config.resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    command_text = "#!/usr/bin/env bash\nset -euo pipefail\n\n"
    command_text += "# Legacy child commands\n"
    command_text += "\n\n".join(_render_command_with_env(cmd) for cmd in commands) + "\n"
    if postprocess_commands:
        command_text += "\n# Managed postprocess commands\n"
        command_text += "\n\n".join(_render_command_with_env(cmd) for cmd in postprocess_commands) + "\n"
    (out_dir / "command.sh").write_text(command_text, encoding="utf-8")
    write_data_path_manifest(
        config,
        out_dir / "data_manifest.json",
        local_paths=manifest.get("local_paths") or {},
    )
    write_k500_ref_ids_artifact(manifest, out_dir / "k500_ref_ids.json")
    write_selection_record_artifact(manifest, out_dir / "selection.json")
    manifest = attach_launch_artifacts(manifest, run_dir=out_dir)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    finalize_run_record(
        out_dir,
        purpose=_experiment_purpose(config),
        result_summary="Run plan files were written; execution has not completed yet.",
        outcome=str(manifest.get("status", "dry_run")),
    )
    return json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))


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
    except (ConfigError, OSError, ValueError) as exc:
        print(f"[config-error] {exc}", file=sys.stderr)
        return 2

    if args.execute:
        try:
            visible = require_cuda_visible_devices()
            gpu_snapshot = check_nvidia_smi()
        except LaunchError as exc:
            print(f"[launch-error] {exc}", file=sys.stderr)
            return 3
        manifest["gpu_prelaunch"] = {
            "CUDA_VISIBLE_DEVICES": visible,
            "nvidia_smi": gpu_snapshot,
        }
        manifest["status"] = "launch_prepared"
        manifest["launcher"]["execute"] = True
        manifest["safety"]["legacy_child_scripts_invoked"] = False

    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str))
    if args.execute:
        print("\n# Legacy commands (will be executed after plan files and input verification)")
    else:
        print("\n# Legacy commands (not executed)")
    for command in commands:
        print(_render_command_with_env(command))
    if postprocess_commands:
        print("\n# Managed postprocess commands")
        for command in postprocess_commands:
            print(_render_command_with_env(command))

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
            manifest = _write_plan_files(out_dir, config, manifest, commands, postprocess_commands)
        except LaunchError as exc:
            print(f"[launch-error] {exc}", file=sys.stderr)
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
                raise LaunchError("Required input verification failed before legacy commands")
            manifest["safety"]["legacy_child_scripts_invoked"] = True
            manifest["input_verification"] = input_verification
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n",
                encoding="utf-8",
            )
            run_legacy_commands(commands, run_dir=out_dir, manifest_path=manifest_path)
            run_postprocess_commands(postprocess_commands, run_dir=out_dir, manifest_path=manifest_path)
            finalize_run_record(
                out_dir,
                purpose=_experiment_purpose(config),
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
