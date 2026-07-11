from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from ecg_adv_gen.config import LaunchError
from ecg_adv_gen.runner.matrix_parallel import (
    assign_gpus_to_commands,
    execution_lock,
    parse_gpu_list,
    run_matrix_queue,
    run_postprocess_serial,
    validate_gpu_snapshot,
    verify_command_artifacts,
)
from scripts.agent.run_matrix_parallel import _resume_manifest, parse_args


def _write_fake_child(path: Path) -> None:
    path.write_text(
        """
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--env-output")
parser.add_argument("--sleep", type=float, default=0.0)
parser.add_argument("--returncode", type=int, default=0)
parser.add_argument("--tracker-dir")
args = parser.parse_args()

if args.env_output:
    Path(args.env_output).write_text(
        json.dumps({"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}),
        encoding="utf-8",
    )

tracker = Path(args.tracker_dir) if args.tracker_dir else None
if tracker:
    tracker.mkdir(parents=True, exist_ok=True)
    with (tracker / "lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active_path = tracker / "active"
        maximum_path = tracker / "maximum"
        active = int(active_path.read_text() if active_path.exists() else "0") + 1
        maximum = max(active, int(maximum_path.read_text() if maximum_path.exists() else "0"))
        active_path.write_text(str(active), encoding="utf-8")
        maximum_path.write_text(str(maximum), encoding="utf-8")
        fcntl.flock(lock, fcntl.LOCK_UN)

time.sleep(args.sleep)

if tracker:
    with (tracker / "lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active_path = tracker / "active"
        active = int(active_path.read_text()) - 1
        active_path.write_text(str(active), encoding="utf-8")
        fcntl.flock(lock, fcntl.LOCK_UN)

if args.returncode == 0:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("ok\\n", encoding="utf-8")
raise SystemExit(args.returncode)
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _command(
    script: Path,
    output: Path,
    *,
    index: int,
    sleep: float = 0.0,
    returncode: int = 0,
    env_output: Path | None = None,
    tracker_dir: Path | None = None,
) -> dict[str, object]:
    argv = [
        sys.executable,
        str(script),
        "--output",
        str(output),
        "--sleep",
        str(sleep),
        "--returncode",
        str(returncode),
    ]
    if env_output is not None:
        argv.extend(["--env-output", str(env_output)])
    if tracker_dir is not None:
        argv.extend(["--tracker-dir", str(tracker_dir)])
    return {
        "name": f"fake_{index}",
        "argv": argv,
        "cwd": str(script.parent),
        "env": {"CUDA_VISIBLE_DEVICES": "99"},
        "matrix": {"center": f"center_{index}"},
    }


def _write_manifest(run_dir: Path, commands: list[dict[str, object]], outputs: list[Path]) -> Path:
    manifest = {
        "manifest_schema_version": 2,
        "run_id": "queue-test",
        "status": "launch_prepared",
        "commands": commands,
        "artifact_trace": {
            "metrics": {},
            "expected_outputs": {
                "launch_artifacts": [],
                "child_runs": [
                    {
                        "command_index": index,
                        "name": command["name"],
                        "center": command["matrix"]["center"],
                        "child_run_dir": str(output.parent),
                        "expected_artifacts": [
                            {"role": "fake_output", "path": str(output), "required": True}
                        ],
                    }
                    for index, (command, output) in enumerate(zip(commands, outputs, strict=True))
                ],
                "postprocess_runs": [],
            },
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_cli_requires_exactly_one_mode_and_bounds_parallelism() -> None:
    base = [
        "--config",
        "experiment.yaml",
        "--local-config",
        "local.yaml",
        "--run-id",
        "r1",
        "--gpus",
        "0,1",
    ]
    with pytest.raises(SystemExit):
        parse_args(base)
    with pytest.raises(SystemExit):
        parse_args([*base, "--dry-run", "--execute"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--execute", "--max-parallel", "3"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--execute", "--max-parallel", "0"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--execute", "--force"])

    args = parse_args([*base, "--execute", "--max-parallel", "2"])
    assert args.execute is True
    assert args.max_parallel == 2


def test_gpu_list_and_snapshot_validation() -> None:
    assert parse_gpu_list("0, 2") == ["0", "2"]
    with pytest.raises(ValueError, match="numeric"):
        parse_gpu_list("0,gpu1")

    snapshot = {
        "gpus": [
            {
                "index": "0",
                "name": "GPU0",
                "memory_used_mb": "128",
                "memory_total_mb": "24564",
                "utilization_gpu_pct": "0",
            },
            {
                "index": "2",
                "name": "GPU2",
                "memory_used_mb": "24000",
                "memory_total_mb": "24564",
                "utilization_gpu_pct": "99",
            },
        ]
    }
    report = validate_gpu_snapshot(
        ["0"], snapshot, min_free_memory_mb=20_000, max_utilization_pct=10
    )
    assert report[0]["index"] == "0"
    with pytest.raises(LaunchError, match="not present"):
        validate_gpu_snapshot(["1"], snapshot, min_free_memory_mb=0, max_utilization_pct=100)
    with pytest.raises(LaunchError, match="free memory"):
        validate_gpu_snapshot(["2"], snapshot, min_free_memory_mb=20_000, max_utilization_pct=100)


def test_assignment_supports_more_commands_than_gpus() -> None:
    commands = [
        {"name": f"c{i}", "matrix": {"center": f"center_{i}"}}
        for i in range(5)
    ]
    assigned = assign_gpus_to_commands(commands, ["0", "1"])
    assert [item.gpu for item in assigned] == ["0", "1", "0", "1", "0"]


def test_queue_caps_parallelism_and_sets_one_physical_gpu_per_child(tmp_path: Path) -> None:
    script = tmp_path / "fake_child.py"
    _write_fake_child(script)
    tracker = tmp_path / "tracker"
    outputs = [tmp_path / "outputs" / f"{index}.txt" for index in range(6)]
    env_outputs = [tmp_path / "env" / f"{index}.json" for index in range(6)]
    for path in env_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        _command(
            script,
            output,
            index=index,
            sleep=0.08,
            env_output=env_outputs[index],
            tracker_dir=tracker,
        )
        for index, output in enumerate(outputs)
    ]
    run_dir = tmp_path / "run"
    manifest_path = _write_manifest(run_dir, commands, outputs)

    progress = run_matrix_queue(
        commands,
        gpus=["4", "7"],
        max_parallel=2,
        run_dir=run_dir,
        manifest_path=manifest_path,
        base_env={"PATH": os.environ["PATH"], "CUDA_VISIBLE_DEVICES": "88"},
        poll_interval=0.005,
    )

    assert progress["status"] == "succeeded"
    assert int((tracker / "maximum").read_text()) <= 2
    assert {json.loads(path.read_text())["CUDA_VISIBLE_DEVICES"] for path in env_outputs} <= {
        "4",
        "7",
    }
    attempts = [state["attempts"][0] for state in progress["commands"]]
    assert all(attempt["pid"] > 0 for attempt in attempts)
    assert all(attempt["started_at_utc"] and attempt["finished_at_utc"] for attempt in attempts)
    assert all(Path(attempt["stdout_log_path"]).is_file() for attempt in attempts)
    assert all(Path(attempt["stderr_log_path"]).is_file() for attempt in attempts)
    assert not (run_dir / "stdout.log").exists()
    assert not (run_dir / "stderr.log").exists()
    combined_logs = [
        path
        for path in (run_dir / "logs").glob("command_*.log")
        if not path.name.endswith((".stdout.log", ".stderr.log"))
    ]
    assert not combined_logs


def test_execution_lock_rejects_second_owner(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with execution_lock(run_dir):
        with pytest.raises(LaunchError, match="already locked"):
            with execution_lock(run_dir):
                pass


def test_resume_revalidates_success_and_retries_effnet_from_latest(tmp_path: Path) -> None:
    success_script = tmp_path / "fake_child.py"
    _write_fake_child(success_script)
    effnet_script = tmp_path / "effnet_vae_lhat_augmix.py"
    argv_record = tmp_path / "effnet_argv.json"
    child_dir = tmp_path / "effnet_run"
    checkpoint = child_dir / "checkpoints" / "checkpoint_latest.pt"
    effnet_output = child_dir / "result.txt"
    effnet_script.write_text(
        """
import json
import sys
from pathlib import Path

argv_record = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
output = Path(sys.argv[3])
checkpoint.parent.mkdir(parents=True, exist_ok=True)
checkpoint.write_text("checkpoint\\n", encoding="utf-8")
argv_record.write_text(json.dumps(sys.argv[4:]), encoding="utf-8")
if sys.argv[4:] == ["--resume", "latest"]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("ok\\n", encoding="utf-8")
    raise SystemExit(0)
raise SystemExit(7)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    first_output = tmp_path / "first.txt"
    commands = [
        _command(success_script, first_output, index=0),
        {
            "name": "effnet",
            "argv": [
                sys.executable,
                str(effnet_script),
                str(argv_record),
                str(checkpoint),
                str(effnet_output),
            ],
            "cwd": str(tmp_path),
            "env": {},
            "matrix": {"center": "ningbo"},
        },
    ]
    run_dir = tmp_path / "run"
    manifest_path = _write_manifest(run_dir, commands, [first_output, effnet_output])
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_trace"]["expected_outputs"]["child_runs"][1]["child_run_dir"] = str(
        child_dir
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(LaunchError, match="command 1"):
        run_matrix_queue(
            commands,
            gpus=["0"],
            max_parallel=1,
            run_dir=run_dir,
            manifest_path=manifest_path,
            base_env={"PATH": os.environ["PATH"]},
            poll_interval=0.005,
        )

    progress = run_matrix_queue(
        commands,
        gpus=["0"],
        max_parallel=1,
        run_dir=run_dir,
        manifest_path=manifest_path,
        resume=True,
        base_env={"PATH": os.environ["PATH"]},
        poll_interval=0.005,
    )

    assert progress["status"] == "succeeded"
    assert len(progress["commands"][0]["attempts"]) == 1
    assert progress["commands"][0]["resume_action"] == "skipped_revalidated"
    assert len(progress["commands"][1]["attempts"]) == 2
    assert progress["commands"][1]["attempts"][1]["resume"] == "latest"
    assert json.loads(argv_record.read_text()) == ["--resume", "latest"]
    assert json.loads((run_dir / "matrix_progress.json").read_text())["status"] == "succeeded"
    assert json.loads(manifest_path.read_text())["status"] == "succeeded"
    assert not list(run_dir.glob("*.tmp"))


def test_first_failure_stops_dispatch_but_allows_running_child_to_finish(tmp_path: Path) -> None:
    script = tmp_path / "fake_child.py"
    _write_fake_child(script)
    outputs = [tmp_path / f"output_{index}.txt" for index in range(4)]
    commands = [
        _command(script, outputs[0], index=0, returncode=9),
        _command(script, outputs[1], index=1, sleep=0.12),
        _command(script, outputs[2], index=2),
        _command(script, outputs[3], index=3),
    ]
    run_dir = tmp_path / "run"
    manifest_path = _write_manifest(run_dir, commands, outputs)

    with pytest.raises(LaunchError, match="command 0"):
        run_matrix_queue(
            commands,
            gpus=["0", "1"],
            max_parallel=2,
            run_dir=run_dir,
            manifest_path=manifest_path,
            base_env={"PATH": os.environ["PATH"]},
            poll_interval=0.005,
        )

    progress = json.loads((run_dir / "matrix_progress.json").read_text())
    assert [state["status"] for state in progress["commands"]] == [
        "failed",
        "succeeded",
        "pending",
        "pending",
    ]
    assert outputs[1].is_file()
    assert not outputs[2].exists()
    assert not outputs[3].exists()


def test_interrupt_terminates_only_owned_child_process_groups(tmp_path: Path) -> None:
    script = tmp_path / "fake_child.py"
    _write_fake_child(script)
    output = tmp_path / "never.txt"
    command = _command(script, output, index=0, sleep=30.0)
    run_dir = tmp_path / "run"
    manifest_path = _write_manifest(run_dir, [command], [output])
    calls = 0

    def interrupt_after_dispatch() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_matrix_queue(
            [command],
            gpus=["0"],
            max_parallel=1,
            run_dir=run_dir,
            manifest_path=manifest_path,
            base_env={"PATH": os.environ["PATH"]},
            poll_interval=0.005,
            tick_hook=interrupt_after_dispatch,
        )

    progress = json.loads((run_dir / "matrix_progress.json").read_text())
    attempt = progress["commands"][0]["attempts"][0]
    assert progress["status"] == "interrupted"
    assert attempt["status"] == "interrupted"
    with pytest.raises(ProcessLookupError):
        os.kill(attempt["pid"], 0)


def test_artifact_filter_checks_only_requested_command(tmp_path: Path) -> None:
    script = tmp_path / "fake_child.py"
    _write_fake_child(script)
    present = tmp_path / "present.txt"
    missing = tmp_path / "missing.txt"
    present.write_text("ok\n", encoding="utf-8")
    commands = [
        _command(script, present, index=0),
        _command(script, missing, index=1),
    ]
    manifest_path = _write_manifest(tmp_path / "run", commands, [present, missing])
    manifest = json.loads(manifest_path.read_text())

    assert verify_command_artifacts(manifest, 0)["passed"] is True
    report = verify_command_artifacts(manifest, 1)
    assert report["passed"] is False
    assert {item["command_index"] for item in report["missing"]} == {1}


def test_postprocess_runs_serially_then_performs_full_artifact_verification(tmp_path: Path) -> None:
    script = tmp_path / "fake_child.py"
    _write_fake_child(script)
    child_output = tmp_path / "child.txt"
    child_output.write_text("ok\n", encoding="utf-8")
    post_output = tmp_path / "report.json"
    postprocess = _command(script, post_output, index=0)
    run_dir = tmp_path / "run"
    manifest_path = _write_manifest(
        run_dir,
        [_command(script, child_output, index=0)],
        [child_output],
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["postprocess_commands"] = [postprocess]
    manifest["artifact_trace"]["expected_outputs"]["postprocess_runs"] = [
        {
            "command_index": 0,
            "name": "report",
            "expected_artifacts": [
                {"role": "report", "path": str(post_output), "required": True}
            ],
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_postprocess_serial(
        [postprocess],
        run_dir=run_dir,
        manifest_path=manifest_path,
        base_env={"PATH": os.environ["PATH"]},
    )

    assert report["passed"] is True
    final_manifest = json.loads(manifest_path.read_text())
    assert final_manifest["status"] == "succeeded"
    assert final_manifest["postprocess_runs"][0]["status"] == "succeeded"
    assert (run_dir / "logs" / "postprocess_000.stdout.log").is_file()
    assert (run_dir / "logs" / "postprocess_000.stderr.log").is_file()
    assert not (run_dir / "stdout.log").exists()
    assert not (run_dir / "stderr.log").exists()


def test_resume_loads_but_does_not_rewrite_manifest_or_resolved_config(tmp_path: Path) -> None:
    commands = [{"name": "one", "argv": ["python"], "cwd": ".", "matrix": {}}]
    postprocess = [{"name": "report", "argv": ["python"], "cwd": "."}]
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({"commands": commands, "postprocess_commands": postprocess}, indent=1) + "\n",
        encoding="utf-8",
    )
    yaml_path = tmp_path / "run_config.resolved.yaml"
    json_path = tmp_path / "run_config.resolved.json"
    yaml_path.write_text("marker: yaml\n", encoding="utf-8")
    json_path.write_text('{"marker":"json"}\n', encoding="utf-8")
    before = {path: path.read_bytes() for path in (manifest_path, yaml_path, json_path)}

    loaded = _resume_manifest(
        manifest_path,
        commands=commands,
        postprocess_commands=postprocess,
    )

    assert loaded["commands"] == commands
    assert {path: path.read_bytes() for path in before} == before
