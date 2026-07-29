"""Managed launcher shim for the isolated Diverse Latent AugMax sandbox.

The whitelist launcher still owns experiment-plan validation, config closure,
run-directory safety and run records.  This shim temporarily replaces only the
selected managed entrypoint with the sibling :mod:`delegate` subprocess and
restores both the entrypoint registry and environment marker in ``finally``.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


SANDBOX_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SANDBOX_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boot_scripts import run_experiment as managed_launcher  # noqa: E402


DELEGATE_ENTRYPOINT_ENV = "ECG_DIVERSE_AUGMAX_SANDBOX_ENTRYPOINT"
DELEGATE_PATH = SANDBOX_ROOT / "delegate.py"
SUPPORTED_ENTRYPOINTS = frozenset(
    {
        "tune_pn2021_direct",
        "select_pn2021_direct",
        "train_pn2021",
        "evaluate_pn2021",
    }
)


@contextmanager
def _temporary_delegate(entrypoint_name: str) -> Iterator[None]:
    if entrypoint_name not in SUPPORTED_ENTRYPOINTS:
        raise ValueError(
            "Diverse AugMax sandbox supports only "
            f"{sorted(SUPPORTED_ENTRYPOINTS)}, got {entrypoint_name!r}"
        )
    if not DELEGATE_PATH.is_file():
        raise FileNotFoundError(f"sandbox delegate not found: {DELEGATE_PATH}")
    original_path = managed_launcher.ENTRYPOINTS[entrypoint_name]
    original_environment = os.environ.get(DELEGATE_ENTRYPOINT_ENV)
    managed_launcher.ENTRYPOINTS[entrypoint_name] = DELEGATE_PATH
    os.environ[DELEGATE_ENTRYPOINT_ENV] = entrypoint_name
    try:
        yield
    finally:
        managed_launcher.ENTRYPOINTS[entrypoint_name] = original_path
        if original_environment is None:
            os.environ.pop(DELEGATE_ENTRYPOINT_ENV, None)
        else:
            os.environ[DELEGATE_ENTRYPOINT_ENV] = original_environment


def main(argv: Sequence[str] | None = None) -> int:
    """Launch one sandbox experiment through the whitelist run recorder."""

    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    launcher_args = managed_launcher.build_parser().parse_args(raw_arguments)
    original_plan = managed_launcher.load_experiment_plan(
        launcher_args.config,
        config_root=launcher_args.config_root,
        run_dir=launcher_args.run_dir,
    )
    with _temporary_delegate(original_plan.entrypoint_name):
        return int(managed_launcher.main(raw_arguments))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DELEGATE_ENTRYPOINT_ENV", "SUPPORTED_ENTRYPOINTS", "main"]
