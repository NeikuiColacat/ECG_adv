"""CPU-only tests for reusable legacy process helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ecg_adv_gen.runner.process import build_process_env, render_command, run_stream


REPO = Path(__file__).resolve().parents[2]


def test_process_helper_import_does_not_load_config_layer():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import ecg_adv_gen.runner.process; "
                "print('config_loaded', 'ecg_adv_gen.config' in sys.modules)"
            ),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == "config_loaded False"


def test_render_command_quotes_env_and_argv():
    rendered = render_command(
        ["python", "script with spaces.py", "--name", "a b"],
        env={"B": "two words", "A": "1"},
    )

    assert rendered == "A=1 B='two words' python 'script with spaces.py' --name 'a b'"


def test_build_process_env_sets_unbuffered_and_creates_runtime_dirs(tmp_path: Path):
    tmp_dir = tmp_path / "tmp"
    cache_dir = tmp_path / "cache"

    env = build_process_env(
        base={"KEEP": "yes"},
        updates={"TMPDIR": tmp_dir, "XDG_CACHE_HOME": cache_dir},
    )

    assert env["KEEP"] == "yes"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["TMPDIR"] == str(tmp_dir)
    assert env["XDG_CACHE_HOME"] == str(cache_dir)
    assert tmp_dir.is_dir()
    assert cache_dir.is_dir()


def test_run_stream_dry_run_writes_command_without_executing(tmp_path: Path):
    log_path = tmp_path / "logs" / "dry.log"

    result = run_stream(
        ["/definitely/missing/binary", "--flag", "two words"],
        log_path=log_path,
        env={"TMPDIR": tmp_path / "tmp"},
        dry_run=True,
    )

    assert result.returncode == 0
    assert result.dry_run is True
    assert log_path.read_text(encoding="utf-8") == (
        "/definitely/missing/binary --flag 'two words'\n"
    )
    assert "TMPDIR=" not in result.rendered


def test_run_stream_dry_run_does_not_log_sensitive_env(tmp_path: Path):
    log_path = tmp_path / "dry.log"

    result = run_stream(
        ["train.py"],
        log_path=log_path,
        env={
            "CUDA_VISIBLE_DEVICES": "3",
            "OPENAI_API_KEY": "secret",
            "TMPDIR": tmp_path / "tmp",
        },
        dry_run=True,
    )

    text = log_path.read_text(encoding="utf-8")
    assert text == "train.py\n"
    assert "CUDA_VISIBLE_DEVICES" not in text
    assert "OPENAI_API_KEY" not in text
    assert "secret" not in result.rendered


def test_run_stream_dry_run_can_append_to_existing_log(tmp_path: Path):
    log_path = tmp_path / "dry.log"
    run_stream(["first"], log_path=log_path, dry_run=True)
    run_stream(["second"], log_path=log_path, dry_run=True, append=True)

    assert log_path.read_text(encoding="utf-8") == "first\nsecond\n"


def test_run_stream_executes_and_logs_output(tmp_path: Path):
    log_path = tmp_path / "run.log"

    result = run_stream(
        [sys.executable, "-c", "import os; print(os.environ['MARKER'])"],
        log_path=log_path,
        env={"MARKER": "ok"},
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.dry_run is False
    assert "ok\n" in log_path.read_text(encoding="utf-8")


def test_run_stream_env_is_final_child_env_not_parent_overlay(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "run.log"
    monkeypatch.setenv("LEAK_PARENT_VAR", "should_not_leak")

    run_stream(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "print(os.environ.get('LEAK_PARENT_VAR', 'missing')); "
                "print(os.environ['MARKER'])"
            ),
        ],
        log_path=log_path,
        env={"MARKER": "ok"},
        cwd=tmp_path,
    )

    text = log_path.read_text(encoding="utf-8")
    assert "missing\n" in text
    assert "ok\n" in text
    assert "should_not_leak" not in text


def test_run_stream_raises_called_process_error_on_failure(tmp_path: Path):
    log_path = tmp_path / "fail.log"

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_stream(
            [sys.executable, "-c", "print('before fail'); raise SystemExit(7)"],
            log_path=log_path,
            cwd=tmp_path,
        )

    assert exc_info.value.returncode == 7
    assert "before fail\n" in log_path.read_text(encoding="utf-8")
