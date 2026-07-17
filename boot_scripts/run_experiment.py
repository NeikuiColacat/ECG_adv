"""Single YAML-managed launcher for the manual-refactor execution surface."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.config_bundle import (  # noqa: E402
    config_bundle_root,
    resolve_config_reference,
    resolve_entry_config_path,
)
from util.run_record import (  # noqa: E402
    RunRecorder,
    ensure_output_outside_worktree,
    sha256_file,
)


ENTRYPOINTS = {
    "train_ptbxl_effnet": PROJECT_ROOT / "boot_scripts" / "train_ptbxl_effnet.py",
    "train_ptbxl_ecgfounder": (
        PROJECT_ROOT / "boot_scripts" / "train_ptbxl_ecgfounder.py"
    ),
    "train_pn2021": PROJECT_ROOT / "boot_scripts" / "train_pn2021.py",
    "tune_pn2021_direct": (
        PROJECT_ROOT / "boot_scripts" / "tune_pn2021_direct.py"
    ),
    "select_pn2021_direct": (
        PROJECT_ROOT / "boot_scripts" / "select_pn2021_direct.py"
    ),
    "refit_pn2021_direct": (
        PROJECT_ROOT / "boot_scripts" / "refit_pn2021_direct.py"
    ),
    "evaluate_pn2021": PROJECT_ROOT / "boot_scripts" / "evaluate_pn2021.py",
}
LAUNCHER_OWNED_FLAGS = frozenset(
    {"--config", "--config-root", "--output-dir", "--dry-run"}
)
REMOVED_ENTRYPOINT_FLAGS = frozenset({"--fixed20-cycles"})


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _yaml_mapping(path: Path, description: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(payload, description)


def _yaml_references(value: Any) -> list[str]:
    references: list[str] = []
    if isinstance(value, dict):
        if value.get("config_closure") == "snapshot_only":
            return references
        for child in value.values():
            references.extend(_yaml_references(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            references.extend(_yaml_references(child))
    elif isinstance(value, str) and Path(value).suffix.lower() in {".yaml", ".yml"}:
        references.append(value)
    return references


def _config_closure(
    *,
    experiment_path: Path,
    entry_config_path: Path,
    config_root: Path,
) -> tuple[tuple[Path, str], ...]:
    ordered: list[tuple[Path, str]] = [
        (experiment_path, "experiment_entry_config"),
        (entry_config_path, "delegate_entry_config"),
    ]
    seen: set[Path] = set()
    index = 0
    while index < len(ordered):
        path, _ = ordered[index]
        index += 1
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        payload = _yaml_mapping(path, f"referenced config {path}")
        for raw_reference in _yaml_references(payload):
            referenced = resolve_config_reference(
                raw_reference,
                owner_config_path=path,
                config_root=config_root,
                description=f"YAML reference in {path.name}",
                must_exist=True,
            )
            if referenced not in seen and all(
                referenced != existing for existing, _ in ordered
            ):
                ordered.append((referenced, "referenced_config"))
    deduplicated: list[tuple[Path, str]] = []
    emitted: set[Path] = set()
    for path, role in ordered:
        resolved = path.resolve()
        if resolved not in emitted:
            emitted.add(resolved)
            deduplicated.append((resolved, role))
    return tuple(deduplicated)


def _validated_arguments(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("entrypoint.arguments must be a list of strings")
    arguments = tuple(value)
    for argument in arguments:
        if "\x00" in argument:
            raise ValueError("entrypoint.arguments may not contain NUL bytes")
        flag = argument.split("=", 1)[0]
        if flag in LAUNCHER_OWNED_FLAGS:
            raise ValueError(f"entrypoint argument is launcher-owned: {flag}")
        if flag in REMOVED_ENTRYPOINT_FLAGS:
            raise ValueError(f"entrypoint argument belongs to a removed protocol: {flag}")
    return arguments


def _safe_delegate_subdir(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("output.delegate_output_subdir must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError("output.delegate_output_subdir must stay inside the run directory")
    return path


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_path: Path
    config_root: Path
    experiment_name: str
    purpose: str
    entrypoint_name: str
    entrypoint_path: Path
    entry_config_path: Path
    entry_arguments: tuple[str, ...]
    config_sources: tuple[tuple[Path, str], ...]
    run_dir: Path
    delegate_output_dir: Path

    def delegate_argv(self) -> list[str]:
        return [
            sys.executable,
            str(self.entrypoint_path),
            "--config",
            str(self.entry_config_path),
            "--config-root",
            str(self.config_root),
            "--output-dir",
            str(self.delegate_output_dir),
            *self.entry_arguments,
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "dry_run",
            "schema_version": 1,
            "experiment_name": self.experiment_name,
            "purpose": self.purpose,
            "experiment_config_path": str(self.experiment_path),
            "experiment_config_sha256": sha256_file(self.experiment_path),
            "config_root": str(self.config_root),
            "entrypoint": {
                "name": self.entrypoint_name,
                "path": str(self.entrypoint_path),
                "config_path": str(self.entry_config_path),
                "arguments": list(self.entry_arguments),
            },
            "config_closure": [
                {
                    "path": str(path),
                    "role": role,
                    "sha256": sha256_file(path),
                }
                for path, role in self.config_sources
            ],
            "run_dir": str(self.run_dir),
            "delegate_output_dir": str(self.delegate_output_dir),
            "delegate_argv": self.delegate_argv(),
            "would_create_directory": True,
            "loads_data": False,
            "loads_model_or_gpu": False,
        }


def load_experiment_plan(
    config_path: str | Path,
    *,
    config_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> ExperimentPlan:
    experiment_path = resolve_entry_config_path(config_path)
    if not experiment_path.is_file():
        raise FileNotFoundError(f"experiment config not found: {experiment_path}")
    payload = _yaml_mapping(experiment_path, "experiment config")
    if set(payload) != {"schema_version", "experiment", "entrypoint", "output"}:
        raise ValueError(
            "experiment config must contain exactly schema_version, experiment, "
            "entrypoint and output"
        )
    if payload.get("schema_version") != 1:
        raise ValueError("experiment schema_version must be 1")
    experiment = _mapping(payload.get("experiment"), "experiment")
    entrypoint = _mapping(payload.get("entrypoint"), "entrypoint")
    output = _mapping(payload.get("output"), "output")
    if set(experiment) != {"name", "purpose"}:
        raise ValueError("experiment must contain exactly name and purpose")
    if set(entrypoint) != {"name", "config", "arguments"}:
        raise ValueError("entrypoint must contain exactly name, config and arguments")
    if set(output) != {"run_dir", "if_exists", "delegate_output_subdir"}:
        raise ValueError(
            "output must contain exactly run_dir, if_exists and delegate_output_subdir"
        )

    experiment_name = experiment.get("name")
    purpose = experiment.get("purpose")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment.name must be a non-empty string")
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("experiment.purpose must be a non-empty string")
    entrypoint_name = entrypoint.get("name")
    if entrypoint_name not in ENTRYPOINTS:
        raise ValueError(
            f"unsupported manual entrypoint {entrypoint_name!r}; "
            f"expected one of {sorted(ENTRYPOINTS)}"
        )
    entrypoint_path = ENTRYPOINTS[str(entrypoint_name)].resolve()
    if not entrypoint_path.is_file():
        raise FileNotFoundError(f"manual entrypoint not found: {entrypoint_path}")

    root = config_bundle_root(experiment_path, config_root=config_root)
    entry_config_path = resolve_config_reference(
        entrypoint.get("config"),
        owner_config_path=experiment_path,
        config_root=root,
        description="entrypoint.config",
        must_exist=True,
    )
    arguments = _validated_arguments(entrypoint.get("arguments"))
    if output.get("if_exists") != "error":
        raise ValueError("output.if_exists must be 'error'")
    selected_run_dir = output.get("run_dir") if run_dir is None else run_dir
    if not isinstance(selected_run_dir, (str, Path)) or not str(selected_run_dir):
        raise ValueError("output.run_dir must be a non-empty path")
    resolved_run_dir = ensure_output_outside_worktree(selected_run_dir)
    if resolved_run_dir.exists():
        raise FileExistsError(f"run directory already exists: {resolved_run_dir}")
    delegate_subdir = _safe_delegate_subdir(output.get("delegate_output_subdir"))
    delegate_output_dir = (resolved_run_dir / delegate_subdir).resolve()
    try:
        delegate_output_dir.relative_to(resolved_run_dir)
    except ValueError:
        raise ValueError("delegate output directory escapes the run directory") from None

    closure = _config_closure(
        experiment_path=experiment_path,
        entry_config_path=entry_config_path,
        config_root=root,
    )
    return ExperimentPlan(
        experiment_path=experiment_path,
        config_root=root,
        experiment_name=experiment_name.strip(),
        purpose=purpose.strip(),
        entrypoint_name=str(entrypoint_name),
        entrypoint_path=entrypoint_path,
        entry_config_path=entry_config_path,
        entry_arguments=arguments,
        config_sources=closure,
        run_dir=resolved_run_dir,
        delegate_output_dir=delegate_output_dir,
    )


def _run_delegate(argv: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return int(process.returncode)


def execute_experiment(
    plan: ExperimentPlan,
    *,
    launcher_argv: Sequence[str],
    delegate_runner: Callable[[list[str], Path], int] = _run_delegate,
) -> int:
    delegate_argv = plan.delegate_argv()
    recorder = RunRecorder.create(
        run_dir=plan.run_dir,
        experiment_name=plan.experiment_name,
        purpose=plan.purpose,
        entrypoint_name=plan.entrypoint_name,
        entrypoint_path=plan.entrypoint_path,
        config_root=plan.config_root,
        config_sources=plan.config_sources,
        exact_launcher_argv=launcher_argv,
        exact_delegate_argv=delegate_argv,
        cwd=PROJECT_ROOT,
    )
    log_path = recorder.run_dir / "logs" / "delegate.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        exit_code = int(delegate_runner(delegate_argv, log_path))
    except BaseException as exc:
        recorder.finalize(
            exit_code=1,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    recorder.finalize(exit_code=exit_code)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one whitelist-only manual-refactor experiment YAML."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_arguments)
    plan = load_experiment_plan(
        args.config,
        config_root=args.config_root,
        run_dir=args.run_dir,
    )
    if args.dry_run:
        print(json.dumps(plan.describe(), indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    launcher_argv = [sys.executable, str(Path(__file__).resolve()), *raw_arguments]
    return execute_experiment(plan, launcher_argv=launcher_argv)


if __name__ == "__main__":
    raise SystemExit(main())
