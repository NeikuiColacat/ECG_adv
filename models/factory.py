"""Single public construction interface for the manual model package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch.nn as nn

from util.random_seed import (
    DEFAULT_RANDOM_SEED_CONFIG_PATH,
    seed_process,
)

from models.contracts import (
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
    ModelSpec,
)
from models.ecgfounder import build_ecgfounder
from models.efficientnet1d import build_efficientnet1dv2


_MODELS = {
    "efficientnet1dv2": (EFFICIENTNET1DV2_SPEC, build_efficientnet1dv2),
    "ecgfounder": (ECGFOUNDER_SPEC, build_ecgfounder),
}


def _model_entry(name: str):
    if not isinstance(name, str) or name not in _MODELS:
        raise ValueError(f"unknown model {name!r}; available={available_models()}")
    return _MODELS[name]


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODELS))


def get_model_spec(name: str) -> ModelSpec:
    return _model_entry(name)[0]


def build_model(
    name: str,
    *,
    config_root: str | Path | None = None,
    seed_config_path: str | Path | None = None,
    seed_namespace: str = "model_initialization",
    seed_identity: Sequence[Any] = (),
    **kwargs: Any,
) -> nn.Module:
    """Build one canonical model after seeding its isolated init stream."""

    spec, builder = _model_entry(name)
    resolved_seed_config = (
        Path(config_root).expanduser().resolve() / "random_seed.yaml"
        if seed_config_path is None and config_root is not None
        else DEFAULT_RANDOM_SEED_CONFIG_PATH
        if seed_config_path is None
        else Path(seed_config_path).expanduser().resolve()
    )
    random_identity = seed_process(
        seed_namespace,
        name,
        *tuple(seed_identity),
        config_path=resolved_seed_config,
    )
    model = builder(**kwargs)
    model.random_seed_identity = random_identity
    if getattr(model, "model_spec", None) != spec:
        raise RuntimeError(f"builder {name} returned a model with wrong spec")
    return model


__all__ = [
    "available_models",
    "build_model",
    "get_model_spec",
]
