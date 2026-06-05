"""Helpers for explicit GPU assignment of YAML-managed matrix commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AssignedCommand:
    """A generated matrix command with an explicit single-GPU assignment."""

    command_index: int
    command: Mapping[str, Any]
    gpu: str
    matrix_key: str
    matrix_value: str


def parse_gpu_list(raw: str) -> list[str]:
    """Parse a comma-separated GPU list and reject ambiguous selections."""

    gpus = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU id is required")
    seen: set[str] = set()
    for gpu in gpus:
        if gpu in seen:
            raise ValueError(f"duplicate GPU id in --gpus: {gpu}")
        seen.add(gpu)
    return gpus


def assign_gpus_to_commands(
    commands: Sequence[Mapping[str, Any]],
    gpus: Sequence[str],
    *,
    matrix_key: str = "center",
) -> list[AssignedCommand]:
    """Assign one GPU per generated matrix command without queueing."""

    if len(commands) > len(gpus):
        raise ValueError(f"{len(commands)} commands but only {len(gpus)} GPUs; split into smaller waves")
    assigned: list[AssignedCommand] = []
    for idx, command in enumerate(commands):
        matrix = command.get("matrix") or {}
        value = str(matrix.get(matrix_key) or "")
        if not value:
            raise ValueError(f"command {idx} is missing matrix.{matrix_key}")
        assigned.append(
            AssignedCommand(
                command_index=idx,
                command=command,
                gpu=str(gpus[idx]),
                matrix_key=matrix_key,
                matrix_value=value,
            )
        )
    return assigned
