from __future__ import annotations

import copy
import errno
import json
import os
import sys
from pathlib import Path

import pytest

from ecg_adv_gen.config import LaunchError
from ecg_adv_gen.runner import matrix_parallel
from ecg_adv_gen.runner.matrix_parallel import (
    bind_matrix_resume_contract,
    run_matrix_queue,
    run_postprocess_serial,
    validate_matrix_resume_contract,
)
from scripts.agent import run_matrix_parallel as matrix_cli
from scripts.agent.run_matrix_parallel import _resume_manifest


def _child(path: Path) -> None:
    path.write_text(
        """
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
env_output = Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("ok\\n", encoding="utf-8")
env_output.write_text(json.dumps({"cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}), encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _command(script: Path, output: Path, env_output: Path, *, index: int = 0) -> dict:
    return {
        "name": f"child_{index}",
        "argv": [sys.executable, str(script), str(output), str(env_output)],
        "cwd": str(script.parent),
        "env": {"CUDA_VISIBLE_DEVICES": "99"},
        "matrix": {"center": f"center_{index}"},
    }


def _manifest(run_dir: Path, commands: list[dict], outputs: list[Path]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_schema_version": 2,
        "run_id": "crashsafe-v2",
        "status": "launch_prepared",
        "commands": commands,
        "postprocess_commands": [],
        "safety": {
            "managed_child_commands_invoked": False,
            "managed_child_commands_state": "none",
            "managed_child_confirmed_pid_count": 0,
        },
        "artifact_trace": {
            "expected_outputs": {
                "launch_artifacts": [],
                "child_runs": [
                    {
                        "command_index": index,
                        "expected_artifacts": [
                            {"path": str(output), "role": "result", "required": True}
                        ],
                    }
                    for index, output in enumerate(outputs)
                ],
                "postprocess_runs": [],
            }
        },
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _dispatching_progress(run_dir: Path, command: dict) -> None:
    now = "2026-07-11T00:00:00+00:00"
    payload = {
        "schema_version": 2,
        "run_id": "crashsafe-v2",
        "status": "running",
        "created_at_utc": now,
        "updated_at_utc": now,
        "commands": [
            {
                "command_index": 0,
                "name": command["name"],
                "matrix": command["matrix"],
                "status": "dispatching",
                "attempts": [
                    {
                        "attempt": 1,
                        "gpu": "0",
                        "pid": None,
                        "pgid": None,
                        "start_new_session": True,
                        "spawn_intent_at_utc": now,
                        "started_at_utc": None,
                        "finished_at_utc": None,
                        "returncode": None,
                        "status": "dispatching",
                        "stdout_log_path": str(run_dir / "logs/a.stdout.log"),
                        "stderr_log_path": str(run_dir / "logs/a.stderr.log"),
                        "resume": None,
                        "reason": None,
                    }
                ],
            }
        ],
    }
    (run_dir / "matrix_progress.json").write_text(json.dumps(payload), encoding="utf-8")


def test_spawn_intent_is_durable_before_spawn_and_spawn_error_is_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "out.txt", tmp_path / "env.json"
    command = _command(script, output, env_output)
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, [command], [output])

    def fail_spawn(*_args, **_kwargs):
        progress = json.loads((run_dir / "matrix_progress.json").read_text())
        attempt = progress["commands"][0]["attempts"][0]
        assert progress["schema_version"] == 2
        assert attempt["status"] == "dispatching"
        assert attempt["pid"] is None and attempt["pgid"] is None
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(matrix_parallel, "_spawn", fail_spawn)
    with pytest.raises(OSError, match="spawn failure"):
        run_matrix_queue(
            [command],
            gpus=["0"],
            max_parallel=1,
            run_dir=run_dir,
            manifest_path=manifest_path,
            base_env={"PATH": os.environ["PATH"]},
        )

    progress = json.loads((run_dir / "matrix_progress.json").read_text())
    attempt = progress["commands"][0]["attempts"][0]
    assert progress["status"] == "failed"
    assert progress["commands"][0]["status"] == "failed"
    assert attempt["status"] == "failed"
    assert attempt["returncode"] != 0
    assert "spawn" in attempt["reason"]
    safety = json.loads(manifest_path.read_text())["safety"]
    assert safety["managed_child_commands_state"] == "possible"
    assert safety["managed_child_commands_invoked"] is True
    assert safety["managed_child_confirmed_pid_count"] == 0


def test_verifying_state_is_durable_before_artifact_check(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "out.txt", tmp_path / "env.json"
    command = _command(script, output, env_output)
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, [command], [output])

    def verify(_manifest: dict, _index: int) -> dict:
        progress = json.loads((run_dir / "matrix_progress.json").read_text())
        attempt = progress["commands"][0]["attempts"][0]
        assert progress["commands"][0]["status"] == "verifying"
        assert attempt["status"] == "verifying" and attempt["returncode"] == 0
        return {"passed": True}

    monkeypatch.setattr(matrix_parallel, "verify_command_artifacts", verify)
    result = run_matrix_queue(
        [command],
        gpus=["0"],
        max_parallel=1,
        run_dir=run_dir,
        manifest_path=manifest_path,
        base_env={"PATH": os.environ["PATH"]},
        poll_interval=0.001,
    )
    assert result["commands"][0]["status"] == "succeeded"


def test_dispatching_resume_is_ambiguous_and_fails_closed(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "out.txt", tmp_path / "env.json"
    command = _command(script, output, env_output)
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, [command], [output])
    _dispatching_progress(run_dir, command)

    with pytest.raises(LaunchError, match="dispatching.*ambiguous"):
        run_matrix_queue(
            [command],
            gpus=["0"],
            max_parallel=1,
            run_dir=run_dir,
            manifest_path=manifest_path,
            resume=True,
            base_env={"PATH": os.environ["PATH"]},
        )
    assert not output.exists()


def test_running_resume_probes_process_group_and_only_esrch_is_stale(
    tmp_path: Path, monkeypatch
) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "out.txt", tmp_path / "env.json"
    command = _command(script, output, env_output)
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, [command], [output])
    _dispatching_progress(run_dir, command)
    progress_path = run_dir / "matrix_progress.json"
    progress = json.loads(progress_path.read_text())
    attempt = progress["commands"][0]["attempts"][0]
    attempt.update(
        {
            "status": "running",
            "pid": 2_147_483_647,
            "pgid": 2_147_483_647,
            "started_at_utc": "2026-07-11T00:00:01+00:00",
        }
    )
    progress["commands"][0]["status"] = "running"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    probes: list[tuple[int, int]] = []

    def missing_group(pgid: int, sig: int) -> None:
        probes.append((pgid, sig))
        raise OSError(errno.ESRCH, "gone")

    monkeypatch.setattr(matrix_parallel.os, "killpg", missing_group)
    result = run_matrix_queue(
        [command],
        gpus=["0"],
        max_parallel=1,
        run_dir=run_dir,
        manifest_path=manifest_path,
        resume=True,
        base_env={"PATH": os.environ["PATH"]},
        poll_interval=0.001,
    )
    assert probes == [(2_147_483_647, 0)]
    assert result["commands"][0]["resume_action"] == "retry_stale"


def test_postprocess_is_cpu_only_and_uses_v2_attempt_logs(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "report.json", tmp_path / "post_env.json"
    command = _command(script, output, env_output)
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, [], [])
    manifest = json.loads(manifest_path.read_text())
    manifest["postprocess_commands"] = [command]
    manifest["artifact_trace"]["expected_outputs"]["postprocess_runs"] = [
        {
            "command_index": 0,
            "expected_artifacts": [
                {"path": str(output), "role": "report", "required": True}
            ],
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_postprocess_serial(
        [command],
        run_dir=run_dir,
        manifest_path=manifest_path,
        base_env={"PATH": os.environ["PATH"], "CUDA_VISIBLE_DEVICES": "7"},
    )
    progress = json.loads((run_dir / "postprocess_progress.json").read_text())
    attempt = progress["commands"][0]["attempts"][0]
    assert result["passed"] is True
    assert progress["schema_version"] == 2
    assert json.loads(env_output.read_text())["cuda"] == ""
    assert "attempt_01" in attempt["stdout_log_path"]
    assert attempt["start_new_session"] is True
    assert attempt["pgid"] == attempt["pid"]


def test_postprocess_resume_rejects_old_stateless_manifest(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "report.json", tmp_path / "post_env.json"
    command = _command(script, output, env_output)
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, [], [])
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "postprocess_commands": [command],
            "postprocess_status": "running",
            "postprocess_runs": [],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(LaunchError, match="stateless.*postprocess"):
        run_postprocess_serial(
            [command],
            run_dir=run_dir,
            manifest_path=manifest_path,
            resume=True,
            base_env={"PATH": os.environ["PATH"]},
        )


def test_postprocess_dispatching_resume_is_ambiguous(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "report.json", tmp_path / "env.json"
    command = _command(script, output, env_output)
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, [], [])
    manifest = json.loads(manifest_path.read_text())
    manifest["postprocess_commands"] = [command]
    manifest["artifact_trace"]["expected_outputs"]["postprocess_runs"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _dispatching_progress(run_dir, command)
    progress = json.loads((run_dir / "matrix_progress.json").read_text())
    progress["commands"][0]["attempts"][0]["gpu"] = "cpu"
    (run_dir / "matrix_progress.json").unlink()
    (run_dir / "postprocess_progress.json").write_text(json.dumps(progress), encoding="utf-8")

    with pytest.raises(LaunchError, match="postprocess command 0 dispatching.*ambiguous"):
        run_postprocess_serial(
            [command],
            run_dir=run_dir,
            manifest_path=manifest_path,
            resume=True,
        )
    assert not output.exists()


def test_resume_contract_binds_plan_bytes_and_config_semantics(tmp_path: Path) -> None:
    config = {"experiment": {"name": "x"}, "value": 1}
    (tmp_path / "run_config.resolved.yaml").write_text(
        "experiment:\n  name: x\nvalue: 1\n", encoding="utf-8"
    )
    (tmp_path / "run_config.resolved.json").write_text(
        json.dumps(config, sort_keys=True) + "\n", encoding="utf-8"
    )
    (tmp_path / "command.sh").write_text("python child.py\n", encoding="utf-8")
    (tmp_path / "data_manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "env.json").write_text("{}\n", encoding="utf-8")
    manifest = {
        "run_id": "r1",
        "config_hash_sha256": "a" * 64,
        "entry_config": "config.yaml",
        "commands": [],
        "postprocess_commands": [],
        "artifact_trace": {"expected_outputs": {}},
    }
    bound = bind_matrix_resume_contract(manifest, run_dir=tmp_path)
    expected = copy.deepcopy(bound)
    contract = bound["matrix_parallel"]["resume_contract"]["contract"]
    assert contract["plan_artifacts"]["resolved_json_semantic_sha256"]
    assert contract["plan_artifacts"]["resolved_yaml_semantic_sha256"] == contract[
        "plan_artifacts"
    ]["resolved_json_semantic_sha256"]

    (tmp_path / "command.sh").write_text("python other.py\n", encoding="utf-8")
    with pytest.raises(LaunchError, match="resume contract"):
        validate_matrix_resume_contract(bound, expected, run_dir=tmp_path)

    (tmp_path / "command.sh").write_text("python child.py\n", encoding="utf-8")
    (tmp_path / "run_config.resolved.yaml").write_text(
        "experiment:\n  name: x\nvalue: 1\n\n", encoding="utf-8"
    )
    with pytest.raises(LaunchError, match="resume contract"):
        validate_matrix_resume_contract(bound, expected, run_dir=tmp_path)

    (tmp_path / "run_config.resolved.yaml").write_text("value: 2\n", encoding="utf-8")
    with pytest.raises(LaunchError, match="semantic"):
        validate_matrix_resume_contract(bound, expected, run_dir=tmp_path)


def test_old_manifest_without_resume_digest_is_not_auto_recovered(tmp_path: Path) -> None:
    config = {"value": 1}
    (tmp_path / "run_config.resolved.yaml").write_text("value: 1\n", encoding="utf-8")
    (tmp_path / "run_config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "command.sh").write_text("true\n", encoding="utf-8")
    (tmp_path / "data_manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "env.json").write_text("{}\n", encoding="utf-8")
    commands: list[dict] = []
    old = {"run_id": "old", "commands": commands, "postprocess_commands": []}
    expected = bind_matrix_resume_contract(old, run_dir=tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(LaunchError, match="resume contract"):
        _resume_manifest(
            manifest_path,
            commands=commands,
            postprocess_commands=[],
            expected_manifest=expected,
        )


def test_second_wave_gpu_recheck_can_block_without_spawning(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    _child(script)
    outputs = [tmp_path / f"out_{index}.txt" for index in range(3)]
    env_outputs = [tmp_path / f"env_{index}.json" for index in range(3)]
    commands = [
        _command(script, output, env_outputs[index], index=index)
        for index, output in enumerate(outputs)
    ]
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, commands, outputs)
    calls: list[int] = []

    def gpu_gate(index: int, _command: dict, attempt: int, gpu: str) -> dict:
        calls.append(index)
        if index == 1:
            raise LaunchError("GPU became busy before second wave")
        return {"assigned_gpu": gpu, "attempt": attempt, "free_memory_mb": 24000}

    with pytest.raises(LaunchError, match="became busy"):
        run_matrix_queue(
            commands,
            gpus=["0"],
            max_parallel=1,
            run_dir=run_dir,
            manifest_path=manifest_path,
            base_env={"PATH": os.environ["PATH"]},
            before_gpu_dispatch=gpu_gate,
            poll_interval=0.001,
        )

    progress = json.loads((run_dir / "matrix_progress.json").read_text())
    assert calls == [0, 1]
    assert outputs[0].is_file() and not outputs[1].exists() and not outputs[2].exists()
    assert progress["commands"][0]["attempts"][0]["gpu_preflight"]["assigned_gpu"] == "0"
    assert progress["commands"][1]["status"] == "failed"
    safety = json.loads(manifest_path.read_text())["safety"]
    assert safety["managed_child_commands_state"] == "confirmed"
    assert safety["managed_child_confirmed_pid_count"] == 1


def test_cli_injects_fresh_gpu_probe_for_each_dispatch(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "out.txt", tmp_path / "env.json"
    command = _command(script, output, env_output)
    manifest_path = _manifest(tmp_path / "run", [command], [output])
    run_dir = manifest_path.parent
    free = {
        "gpus": [
            {
                "index": "0",
                "name": "fake",
                "memory_used_mb": "0",
                "memory_total_mb": "24000",
                "utilization_gpu_pct": "0",
            }
        ]
    }
    busy = copy.deepcopy(free)
    busy["gpus"][0]["memory_used_mb"] = "23000"
    snapshots = iter([free, busy])
    monkeypatch.setattr(matrix_cli, "check_nvidia_smi", lambda: next(snapshots))
    monkeypatch.setattr(matrix_cli, "verify_required_inputs", lambda _manifest: {"passed": True})
    monkeypatch.setattr(
        matrix_cli,
        "inspect_execution_sources",
        lambda *_a, **_k: {
            "passed": True,
            "sources": [],
            "blocking_sources": [],
            "errors": [],
            "excluded_local_config_sources": [],
        },
    )

    def queue_probe(*_args, **kwargs):
        kwargs["before_gpu_dispatch"](0, command, 1, "0")

    monkeypatch.setattr(matrix_cli, "run_matrix_queue", queue_probe)
    monkeypatch.setattr(
        matrix_cli,
        "run_postprocess_serial",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("postprocess must not run")),
    )
    args = type(
        "Args",
        (),
        {
            "min_free_memory_mb": 20_000,
            "max_utilization_pct": 10,
            "max_parallel": 1,
            "resume": False,
        },
    )()
    with pytest.raises(LaunchError, match="free memory"):
        matrix_cli._execute_pipeline(
            args=args,
            config={"experiment": {"purpose": "test"}},
            commands=[command],
            postprocess_commands=[],
            out_dir=run_dir,
            manifest_path=manifest_path,
            gpus=["0"],
        )
    assert json.loads(manifest_path.read_text())["execution_lifecycle"]["phase"] == "queue"


def test_v1_progress_requires_a_new_output_directory(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "out.txt", tmp_path / "env.json"
    command = _command(script, output, env_output)
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, [command], [output])
    (run_dir / "matrix_progress.json").write_text(
        json.dumps({"schema_version": 1, "run_id": "crashsafe-v2"}), encoding="utf-8"
    )
    with pytest.raises(LaunchError, match="unsupported.*new output directory"):
        run_matrix_queue(
            [command],
            gpus=["0"],
            max_parallel=1,
            run_dir=run_dir,
            manifest_path=manifest_path,
            resume=True,
        )


def test_postprocess_resume_skips_only_after_artifact_revalidation(tmp_path: Path) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "report.json", tmp_path / "post_env.json"
    command = _command(script, output, env_output)
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, [], [])
    manifest = json.loads(manifest_path.read_text())
    manifest["postprocess_commands"] = [command]
    manifest["artifact_trace"]["expected_outputs"]["postprocess_runs"] = [
        {
            "command_index": 0,
            "expected_artifacts": [
                {"path": str(output), "role": "report", "required": True}
            ],
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run_postprocess_serial(
        [command], run_dir=run_dir, manifest_path=manifest_path, base_env={"PATH": os.environ["PATH"]}
    )
    before = output.stat().st_mtime_ns
    run_postprocess_serial(
        [command],
        run_dir=run_dir,
        manifest_path=manifest_path,
        resume=True,
        base_env={"PATH": os.environ["PATH"]},
    )
    progress = json.loads((run_dir / "postprocess_progress.json").read_text())
    assert len(progress["commands"][0]["attempts"]) == 1
    assert progress["commands"][0]["resume_action"] == "skipped_revalidated"
    assert output.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "mutation",
    ["success_nonzero", "success_reason", "artifact_false", "root_failed_without_failed_command"],
)
def test_v2_terminal_root_returncode_reason_and_artifacts_are_strict(
    tmp_path: Path, mutation: str
) -> None:
    script = tmp_path / "child.py"
    _child(script)
    output, env_output = tmp_path / "out.txt", tmp_path / "env.json"
    command = _command(script, output, env_output)
    run_dir = tmp_path / "run"
    manifest_path = _manifest(run_dir, [command], [output])
    run_matrix_queue(
        [command],
        gpus=["0"],
        max_parallel=1,
        run_dir=run_dir,
        manifest_path=manifest_path,
        base_env={"PATH": os.environ["PATH"]},
        poll_interval=0.001,
    )
    progress_path = run_dir / "matrix_progress.json"
    progress = json.loads(progress_path.read_text())
    state, attempt = progress["commands"][0], progress["commands"][0]["attempts"][0]
    if mutation == "success_nonzero":
        attempt["returncode"] = 1
    elif mutation == "success_reason":
        attempt["reason"] = "should be absent"
    elif mutation == "artifact_false":
        state["artifact_verification"]["passed"] = False
    else:
        progress["status"] = "failed"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    with pytest.raises(LaunchError, match="progress"):
        run_matrix_queue(
            [command],
            gpus=["0"],
            max_parallel=1,
            run_dir=run_dir,
            manifest_path=manifest_path,
            resume=True,
        )
