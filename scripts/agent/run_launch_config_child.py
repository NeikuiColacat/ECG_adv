#!/usr/bin/env python3
"""Run child commands stored in a managed launch_config.json."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence


RunFn = Callable[..., subprocess.CompletedProcess]


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def run_launch_config(
    launch_config: Path,
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    stages: Sequence[str] = ("train_cmd", "eval_cmd"),
    stage_arg_overrides: Mapping[str, Sequence[str]] | None = None,
    stage_logs: Mapping[str, Path] | None = None,
    run: RunFn = subprocess.run,
) -> None:
    payload = json.loads(launch_config.read_text(encoding="utf-8"))
    child_env = os.environ.copy()
    if env:
        child_env.update(env)

    for stage in stages:
        cmd = payload.get(stage)
        if not cmd:
            continue
        if not isinstance(cmd, list) or not all(isinstance(part, str) for part in cmd):
            raise TypeError(f"{stage} must be a list of strings")
        cmd = list(cmd)
        if stage_arg_overrides and stage in stage_arg_overrides:
            cmd.extend(str(part) for part in stage_arg_overrides[stage])
        print(f"[{_stamp()}] running {stage}: {' '.join(cmd[:4])} ...", flush=True)
        log_path = stage_logs.get(stage) if stage_logs else None
        if log_path is None:
            run(cmd, cwd=cwd, env=child_env, check=True)
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                run(
                    cmd,
                    cwd=cwd,
                    env=child_env,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        print(f"[{_stamp()}] finished {stage}", flush=True)


def _parse_stage_logs(items: Sequence[str]) -> dict[str, Path]:
    logs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--stage-log must use STAGE=PATH, got: {item}")
        stage, raw_path = item.split("=", maxsplit=1)
        if stage not in {"train_cmd", "eval_cmd"}:
            raise ValueError(f"unsupported stage for --stage-log: {stage}")
        logs[stage] = Path(raw_path)
    return logs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_config", type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stage",
        action="append",
        choices=("train_cmd", "eval_cmd"),
        help="Stage to run. Repeat to choose order. Defaults to train_cmd then eval_cmd.",
    )
    parser.add_argument(
        "--train-resume",
        default="",
        help="Append --resume VALUE to train_cmd only, for example latest.",
    )
    parser.add_argument(
        "--stage-log",
        action="append",
        default=[],
        help="Append a stage's stdout/stderr to STAGE=PATH. Repeat for train_cmd/eval_cmd.",
    )
    args = parser.parse_args(argv)

    stages = tuple(args.stage) if args.stage else ("train_cmd", "eval_cmd")
    stage_arg_overrides = {"train_cmd": ["--resume", args.train_resume]} if args.train_resume else None
    stage_logs = _parse_stage_logs(args.stage_log)
    print(f"[{_stamp()}] launch_config child runner start: {args.launch_config}", flush=True)
    run_launch_config(
        args.launch_config,
        cwd=args.cwd,
        stages=stages,
        stage_arg_overrides=stage_arg_overrides,
        stage_logs=stage_logs,
    )
    print(f"[{_stamp()}] launch_config child runner done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
