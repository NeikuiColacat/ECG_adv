"""Aggregate four prospective single-center PN2021 evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.config_bundle import resolve_entry_config_path  # noqa: E402
from util.evaluation.matrix import aggregate_pn2021_matrix  # noqa: E402


def _config(path: Path, config_root: Path | None) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_entry_config_path(
        config_root / path if config_root is not None and not path.is_absolute() else path
    )
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("matrix config must be a schema-1 YAML mapping")
    if payload.get("protocol") != {"logical_centers": ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"], "aggregation": "equal_views_then_equal_centers"}:
        raise ValueError("matrix config protocol differs from the locked four-center contract")
    output = payload.get("output")
    if not isinstance(output, dict) or output.get("result_file") != "matrix_result.json" or output.get("if_exists") != "error":
        raise ValueError("matrix config output contract is invalid")
    return resolved, payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config_path, config = _config(args.config, args.config_root)
    payload = aggregate_pn2021_matrix(args.result, profile_name=str(config["profile_name"]))
    output = (args.output_dir or Path(config["output"]["run_dir"])).expanduser().resolve()
    result_path = output / "matrix_result.json"
    payload["config"] = {
        "path": str(config_path),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    payload["output"] = {"directory": str(output), "result_file": str(result_path)}
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if output.exists():
        raise FileExistsError(f"matrix output already exists: {output}")
    output.mkdir(parents=True)
    _write(result_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
