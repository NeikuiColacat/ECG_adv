"""Single YAML-managed launcher for the manual-refactor execution surface."""

from __future__ import annotations

import argparse
import hashlib
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
    require_mapping as _mapping,
    resolve_config_reference,
    resolve_entry_config_path,
    resolve_yaml_config_closure,
)
from util.run_record import (  # noqa: E402
    RunRecorder,
    VerifiedDataContent,
    ensure_output_outside_worktree,
    sha256_file,
)


@dataclass(frozen=True)
class EntrypointSpec:
    script: Path
    expected_result_name: str
    expected_result_type: str


def _entrypoint(script: str, result: str, result_type: str) -> EntrypointSpec:
    return EntrypointSpec(PROJECT_ROOT / "boot_scripts" / script, result, result_type)


ENTRYPOINTS = {
    "train_ptbxl_effnet": _entrypoint(
        "train_ptbxl_effnet.py", "train_result.json", "supervised_train_result"
    ),
    "train_ptbxl_ecgfounder": _entrypoint(
        "train_ptbxl_ecgfounder.py", "train_result.json", "supervised_train_result"
    ),
    "train_pn2021": _entrypoint(
        "train_pn2021.py", "train_result.json", "pn2021_train_result"
    ),
    "evaluate_pn2021": _entrypoint(
        "evaluate_pn2021.py", "evaluation_result.json", "evaluation_result"
    ),
    "aggregate_pn2021": _entrypoint(
        "aggregate_pn2021.py", "matrix_result.json", "pn2021_matrix_result"
    ),
}
ENTRYPOINT_DATA_ROOTS = {
    "train_ptbxl_effnet": ("ptbxl_cache", "split_artifacts"),
    "train_ptbxl_ecgfounder": ("ptbxl_cache", "split_artifacts"),
    "train_pn2021": ("pn2021_cache", "split_artifacts"),
    "evaluate_pn2021": ("pn2021_cache", "pn2021c_cache", "split_artifacts"),
    "aggregate_pn2021": (),
}
LAUNCHER_OWNED_FLAGS = frozenset(
    {"--config", "--config-root", "--output-dir", "--dry-run"}
)
CONFIG_REFERENCE_FLAGS = frozenset({"--method-config", "--source-registry"})
MODELS = frozenset({"efficientnet1dv2", "ecgfounder"})
CENTERS = frozenset({"ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"})
ARGUMENT_SCHEMAS = {
    "train_ptbxl_effnet": ((),),
    "train_ptbxl_ecgfounder": ((), ("--epochs",)),
    "train_pn2021": (
        ("--model", "--method-config", "--center", "--source-checkpoint"),
    ),
    "evaluate_pn2021": (
        ("--model", "--train-result", "--center", "--method-config"),
        ("--model", "--checkpoint", "--center"),
        ("--model", "--source-registry"),
    ),
    "aggregate_pn2021": (
        ("--model", "--result", "--result", "--result", "--result"),
    ),
}


def _yaml_mapping(path: Path, description: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(payload, description)


def _config_closure(
    *,
    experiment_path: Path,
    entry_config_path: Path,
    entry_arguments: Sequence[str],
    config_root: Path,
) -> tuple[tuple[Path, str], ...]:
    seeds = [experiment_path, entry_config_path]
    for index, argument in enumerate(entry_arguments):
        flag, separator, inline_value = argument.partition("=")
        if flag not in CONFIG_REFERENCE_FLAGS:
            continue
        raw = inline_value if separator else (
            entry_arguments[index + 1] if index + 1 < len(entry_arguments) else None
        )
        seeds.append(
            resolve_config_reference(
                raw,
                owner_config_path=experiment_path,
                config_root=config_root,
                description=f"entrypoint argument {flag}",
                must_exist=True,
            )
        )
    paths = resolve_yaml_config_closure(seeds, config_root=config_root)
    return tuple(
        (
            path,
            "experiment_entry_config"
            if path == experiment_path
            else "delegate_entry_config"
            if path == entry_config_path
            else "referenced_config",
        )
        for path in paths
    )


def _validated_arguments(entrypoint_name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("entrypoint.arguments must be a list of strings")
    arguments = tuple(value)
    for argument in arguments:
        if "\x00" in argument:
            raise ValueError("entrypoint.arguments may not contain NUL bytes")
        flag = argument.split("=", 1)[0]
        if flag in LAUNCHER_OWNED_FLAGS:
            raise ValueError(f"entrypoint argument is launcher-owned: {flag}")
    _validate_argument_schema(entrypoint_name, arguments)
    return arguments


def _validate_argument_schema(
    entrypoint_name: str, arguments: Sequence[str]
) -> None:
    schemas = ARGUMENT_SCHEMAS.get(entrypoint_name)
    if schemas is None:
        raise ValueError(f"unsupported manual entrypoint {entrypoint_name!r}")
    if not isinstance(arguments, (list, tuple)) or not all(
        isinstance(argument, str) for argument in arguments
    ):
        raise ValueError("entrypoint.arguments must contain only strings")
    if any("\x00" in argument for argument in arguments):
        raise ValueError("entrypoint.arguments may not contain NUL bytes")
    if len(arguments) % 2:
        raise ValueError(f"invalid argument schema for {entrypoint_name}")
    flags = tuple(arguments[::2])
    values = tuple(arguments[1::2])
    if flags not in schemas:
        raise ValueError(f"invalid argument schema for {entrypoint_name}")
    if any(not value.strip() or value.lstrip().startswith("-") for value in values):
        raise ValueError("entrypoint argument values must be non-empty and not flags")
    keyed = dict(zip(flags, values, strict=True))
    if "--model" in keyed and keyed["--model"] not in MODELS:
        raise ValueError(f"unsupported model {keyed['--model']!r}")
    if "--center" in keyed and keyed["--center"] not in CENTERS:
        raise ValueError(f"unsupported center {keyed['--center']!r}")
    if entrypoint_name == "train_ptbxl_ecgfounder" and arguments:
        if values != ("10",):
            raise ValueError("train_ptbxl_ecgfounder only permits --epochs 10")
    if entrypoint_name == "aggregate_pn2021":
        results = values[1:]
        if len(set(results)) != 4:
            raise ValueError("aggregate_pn2021 requires four unique results")


def _safe_delegate_subdir(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("output.delegate_output_subdir must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError("output.delegate_output_subdir must stay inside the run directory")
    return path


@dataclass(frozen=True)
class DataLedgerPlan:
    path: Path
    config_relative_path: Path
    expected_sha256: str
    required_roots: tuple[str, ...]
    closure_sha256: tuple[tuple[Path, str], ...]

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "expected_sha256": self.expected_sha256,
            "config_relative_path": self.config_relative_path.as_posix(),
            "required_roots": list(self.required_roots),
            "verification_mode": "deferred_quick_inventory_size",
            "reads_ledger": False, "stats_data": False,
            "config_closure_sha256": {
                str(path): digest for path, digest in self.closure_sha256
            },
        }


def _closure_member(
    sources: Sequence[tuple[Path, str]], config_root: Path, relative: str
) -> Path:
    expected = (config_root / relative).resolve()
    if expected not in {path for path, _ in sources}:
        raise ValueError(f"managed data entrypoint requires {relative} in YAML closure")
    return expected


def _data_ledger_plan(
    entrypoint_name: str,
    sources: Sequence[tuple[Path, str]],
    config_root: Path,
) -> DataLedgerPlan | None:
    required_roots = ENTRYPOINT_DATA_ROOTS[entrypoint_name]
    if not required_roots:
        return None
    data_load = _closure_member(sources, config_root, "data/data_load.yaml")
    splits = _closure_member(sources, config_root, "data/splits.yaml")
    if "pn2021c_cache" in required_roots:
        _closure_member(sources, config_root, "augmentation/cache.yaml")
    descriptor = _mapping(_yaml_mapping(data_load, "data-load config").get(
        "content_ledger"), "data-load config.content_ledger")
    if set(descriptor) != {"path", "sha256"}:
        raise ValueError("content_ledger must contain exactly path and sha256")
    digest = descriptor.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or set(digest).difference("0123456789abcdef")
    ):
        raise ValueError("content_ledger.sha256 is invalid")
    ledger_path = resolve_config_reference(
            descriptor.get("path"), owner_config_path=data_load,
            config_root=config_root, description="content_ledger.path"
        )
    return DataLedgerPlan(
        path=ledger_path,
        config_relative_path=ledger_path.relative_to(config_root),
        expected_sha256=digest,
        required_roots=required_roots,
        closure_sha256=tuple((path, sha256_file(path)) for path, _ in sources),
    )


def _verify_data_config_closure(plan: DataLedgerPlan) -> None:
    if any(sha256_file(path) != digest for path, digest in plan.closure_sha256):
        raise ValueError("data experiment config closure changed after plan resolution")


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
    expected_result_relative_path: Path
    expected_result_type: str
    data_ledger: DataLedgerPlan | None
    run_dir_preexisting: bool

    def delegate_argv(self) -> list[str]:
        snapshot_root = self.run_dir / "configs"
        snapshot_entry_config = snapshot_root / self.entry_config_path.relative_to(
            self.config_root
        )
        return [
            sys.executable,
            str(self.entrypoint_path),
            "--config",
            str(snapshot_entry_config),
            "--config-root",
            str(snapshot_root),
            "--output-dir",
            str(self.delegate_output_dir),
            *self.entry_arguments,
        ]

    def describe(self) -> dict[str, Any]:
        collision = self.run_dir_preexisting
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
            "expected_result": {
                "path": self.expected_result_relative_path.as_posix(),
                "type": self.expected_result_type,
            },
            "data_content": (
                None if self.data_ledger is None else self.data_ledger.describe()
            ),
            "delegate_argv": self.delegate_argv(),
            "run_dir_collision": collision,
            "would_create_directory": not collision,
            "would_fail_execution": collision,
            "execution_blocker": "run_directory_exists" if collision else None,
            "loads_data": False,
            "loads_model_or_gpu": False,
        }


def _verify_entrypoint_binding(plan: ExperimentPlan) -> None:
    payload = _yaml_mapping(plan.experiment_path, "experiment config")
    source = _mapping(payload.get("entrypoint"), "entrypoint")
    if set(source) != {"name", "config", "arguments"}:
        raise ValueError("entrypoint must contain exactly name, config and arguments")
    source_name = source.get("name")
    if source_name not in ENTRYPOINTS:
        raise ValueError(f"unsupported manual entrypoint {source_name!r}")
    source_arguments = _validated_arguments(
        str(source_name), source.get("arguments")
    )
    if (
        source_name != plan.entrypoint_name
        or source_arguments != tuple(plan.entry_arguments)
    ):
        raise ValueError("execution plan arguments differ from the experiment config")


def load_experiment_plan(
    config_path: str | Path,
    *,
    config_root: str | Path | None = None,
    run_dir: str | Path | None = None,
    allow_existing_run_dir: bool = False,
) -> ExperimentPlan:
    """Resolve one managed plan without loading data, models, or accelerators.

    Existing output directories remain an execution error. The explicit
    ``allow_existing_run_dir`` switch exists only so a read-only dry-run can
    describe that collision; :func:`execute_experiment` checks it again.
    """

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
    entrypoint_spec = ENTRYPOINTS[str(entrypoint_name)]
    entrypoint_path = entrypoint_spec.script.resolve()
    if not entrypoint_path.is_file():
        raise FileNotFoundError(f"manual entrypoint not found: {entrypoint_path}")

    arguments = _validated_arguments(
        str(entrypoint_name), entrypoint.get("arguments")
    )
    root = config_bundle_root(experiment_path, config_root=config_root)
    entry_config_path = resolve_config_reference(
        entrypoint.get("config"),
        owner_config_path=experiment_path,
        config_root=root,
        description="entrypoint.config",
        must_exist=True,
    )
    if output.get("if_exists") != "error":
        raise ValueError("output.if_exists must be 'error'")
    selected_run_dir = output.get("run_dir") if run_dir is None else run_dir
    if not isinstance(selected_run_dir, (str, Path)) or not str(selected_run_dir):
        raise ValueError("output.run_dir must be a non-empty path")
    resolved_run_dir = ensure_output_outside_worktree(selected_run_dir)
    run_dir_preexisting = resolved_run_dir.exists()
    if run_dir_preexisting and not allow_existing_run_dir:
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
        entry_arguments=arguments,
        config_root=root,
    )
    data_ledger = _data_ledger_plan(str(entrypoint_name), closure, root)
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
        expected_result_relative_path=delegate_subdir
        / entrypoint_spec.expected_result_name,
        expected_result_type=entrypoint_spec.expected_result_type,
        data_ledger=data_ledger,
        run_dir_preexisting=run_dir_preexisting,
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
    if type(plan.entry_arguments) is not tuple:
        raise ValueError("execution plan arguments must be an immutable tuple")
    _validate_argument_schema(plan.entrypoint_name, plan.entry_arguments)
    _verify_entrypoint_binding(plan)
    if plan.run_dir_preexisting or plan.run_dir.exists():
        raise FileExistsError(f"run directory already exists: {plan.run_dir}")
    verified_data: VerifiedDataContent | None = None
    ledger = plan.data_ledger
    if ledger is not None:
        from data_preprocess.data_ledger import load_roots, verify_ledger

        _verify_data_config_closure(ledger)
        ledger_bytes = ledger.path.read_bytes()
        if hashlib.sha256(ledger_bytes).hexdigest() != ledger.expected_sha256:
            raise ValueError("data ledger SHA256 differs from the execution plan")
        roots = load_roots(
            plan.config_root / "data/splits.yaml",
            plan.config_root / "augmentation/cache.yaml",
            ledger.required_roots,
        )
        verification = verify_ledger(
            ledger.path, roots, ledger.required_roots,
            expected_sha256=ledger.expected_sha256, full=False,
        )
        _verify_data_config_closure(ledger)
        verified_data = VerifiedDataContent(
            ledger_bytes, ledger.expected_sha256, ledger.required_roots,
            ledger.config_relative_path, ledger.closure_sha256, verification,
        )
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
        expected_result_relative_path=plan.expected_result_relative_path,
        expected_result_type=plan.expected_result_type,
        data_content=verified_data,
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
    return recorder.finalize(exit_code=exit_code)


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
        allow_existing_run_dir=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(plan.describe(), indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    launcher_argv = [sys.executable, str(Path(__file__).resolve()), *raw_arguments]
    return execute_experiment(plan, launcher_argv=launcher_argv)


if __name__ == "__main__":
    raise SystemExit(main())
