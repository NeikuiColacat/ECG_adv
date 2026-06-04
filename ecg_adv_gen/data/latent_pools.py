"""Load and validate frozen ECGTwin latent pools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class LatentPoolError(ValueError):
    """Raised when a latent pool artifact does not match the expected contract."""


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return _to_text(arr.item())
    if arr.size == 1:
        return _to_text(arr.reshape(-1)[0])
    return _to_text(value)


def _text_array(values: Any) -> np.ndarray:
    return np.asarray([_to_text(x) for x in np.asarray(values).reshape(-1)])


def _resolve_latent_path(synth_npz_path: str | Path) -> Path:
    path = Path(synth_npz_path)
    if path.name.endswith(".latent.npz"):
        latent_path = path
    else:
        latent_path = path.with_name(path.stem + ".latent.npz")

    if not latent_path.exists():
        raise LatentPoolError(
            f"latent pool .npz not found: tried {latent_path}. "
            "Re-run generate_center_synth.py with --save_latent."
        )
    return latent_path


def _require_key(npz: np.lib.npyio.NpzFile, key: str, latent_path: Path) -> np.ndarray:
    if key not in npz.files:
        raise LatentPoolError(f"{latent_path} missing required key '{key}'")
    return npz[key]


def load_synth_pool(synth_npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Load a Stage 1 frozen latent pool.

    Accepts either ``{basename}.npz`` and resolves ``{basename}.latent.npz``, or
    accepts ``{basename}.latent.npz`` directly. Returns
    ``(latents, labels, center_name, source_meta)``.
    """
    latent_path = _resolve_latent_path(synth_npz_path)
    with np.load(str(latent_path), allow_pickle=True) as data:
        latents = _require_key(data, "latents", latent_path)
        labels = _require_key(data, "labels", latent_path)
        center = _scalar_text(data["center_name"]) if "center_name" in data.files else "?"

        if latents.ndim != 3 or latents.shape[1:] != (4, 128):
            raise LatentPoolError(f"bad synth latent shape: {latents.shape}")
        if labels.shape[0] != latents.shape[0]:
            raise LatentPoolError("labels length does not match latents")

        has_source_metadata = "source_ids" in data.files and "source_names" in data.files
        if has_source_metadata:
            source_ids = data["source_ids"].astype(np.int64)
            source_names = [_to_text(x) for x in np.asarray(data["source_names"]).reshape(-1)]
            if source_ids.shape[0] != latents.shape[0]:
                raise LatentPoolError("source_ids length does not match latents")
            source_labels = np.asarray(
                [
                    source_names[int(i)]
                    if 0 <= int(i) < len(source_names)
                    else f"source_{int(i)}"
                    for i in source_ids
                ]
            )
        else:
            source_ids = np.zeros((latents.shape[0],), dtype=np.int64)
            source_names = ["unknown"]
            source_labels = np.asarray(["unknown"] * latents.shape[0])

        source_meta: dict[str, Any] = {
            "source_ids": source_ids.copy(),
            "source_names": source_names,
            "source_labels": source_labels.copy(),
            "has_source_metadata": has_source_metadata,
        }
        if "record_ids" in data.files:
            record_ids = _text_array(data["record_ids"])
            if record_ids.shape[0] != latents.shape[0]:
                raise LatentPoolError("record_ids length does not match latents")
            source_meta["record_ids"] = record_ids

        return latents.astype(np.float32), labels.astype(np.float32), center, source_meta
