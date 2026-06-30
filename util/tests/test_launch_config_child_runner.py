from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "agent" / "run_launch_config_child.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_launch_config_child", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runs_train_and_eval_commands_in_launch_config_order(tmp_path):
    launch_config = tmp_path / "launch_config.json"
    launch_config.write_text(
        json.dumps(
            {
                "train_cmd": ["python", "train.py", "--epochs", "1"],
                "eval_cmd": ["python", "eval.py"],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run(cmd, *, cwd, env, check):
        calls.append((cmd, cwd, env["CUDA_VISIBLE_DEVICES"], check))

    runner = _load_runner()
    runner.run_launch_config(
        launch_config,
        cwd=tmp_path,
        env={"CUDA_VISIBLE_DEVICES": "2"},
        run=fake_run,
    )

    assert calls == [
        (["python", "train.py", "--epochs", "1"], tmp_path, "2", True),
        (["python", "eval.py"], tmp_path, "2", True),
    ]


def test_appends_stage_specific_args(tmp_path):
    launch_config = tmp_path / "launch_config.json"
    launch_config.write_text(
        json.dumps(
            {
                "train_cmd": ["python", "train.py"],
                "eval_cmd": ["python", "eval.py"],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run(cmd, *, cwd, env, check):
        calls.append(cmd)

    runner = _load_runner()
    runner.run_launch_config(
        launch_config,
        cwd=tmp_path,
        stage_arg_overrides={"train_cmd": ["--resume", "latest"]},
        run=fake_run,
    )

    assert calls == [
        ["python", "train.py", "--resume", "latest"],
        ["python", "eval.py"],
    ]


def test_writes_stage_output_to_append_logs(tmp_path):
    launch_config = tmp_path / "launch_config.json"
    launch_config.write_text(
        json.dumps({"train_cmd": ["python", "train.py"]}),
        encoding="utf-8",
    )
    stage_log = tmp_path / "train_stdout.log"
    stage_log.write_text("existing\n", encoding="utf-8")

    def fake_run(cmd, *, cwd, env, check, stdout=None, stderr=None):
        stdout.write("new output\n")

    runner = _load_runner()
    runner.run_launch_config(
        launch_config,
        cwd=tmp_path,
        stage_logs={"train_cmd": stage_log},
        run=fake_run,
    )

    assert stage_log.read_text(encoding="utf-8") == "existing\nnew output\n"
