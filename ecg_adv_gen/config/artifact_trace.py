"""Small artifact-trace record helpers for YAML-managed launch manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ecg_adv_gen.data.kshot_artifacts import KShotArtifactGroup


def path_record(role: str, path: str | Path, *, required: bool = True) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    exists = resolved.exists()
    record: dict[str, Any] = {
        "role": role,
        "path": str(resolved),
        "required": required,
        "exists": exists,
    }
    if exists and resolved.is_file():
        try:
            record["size_bytes"] = resolved.stat().st_size
        except OSError:
            pass
    return record


def append_k500_ref(
    refs: list[dict[str, Any]],
    *,
    center: str,
    k: int,
    seed: int,
    base: Path,
    include_latent: bool,
    signal_path: Path | None = None,
    command_index: int | None = None,
    input_index: int | None = None,
) -> None:
    group = KShotArtifactGroup.from_base(
        base,
        center=center,
        k=k,
        seed=seed,
        include_latent=include_latent,
    )
    signal_path = signal_path or group.signals
    signal_role = "kshot_raw1000_signals" if str(signal_path).endswith(".raw1000.npz") else "kshot_signals"
    record = {
        "center": group.center,
        "k": group.k,
        "seed": group.seed,
        "anchor_base": str(group.base),
        "ref_meta_json": path_record("kshot_ref_meta", group.ref_meta),
        "signals_npz": path_record(signal_role, signal_path),
        "latent_npz": (
            path_record("kshot_latents", group.latents) if include_latent else None
        ),
    }
    if command_index is not None:
        record["command_index"] = int(command_index)
    if input_index is not None:
        record["input_index"] = int(input_index)
    refs.append(record)


def dedupe_path_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for record in records:
        key = (str(record.get("role", "")), str(record.get("path", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out
