"""Single public construction interface for the manual model package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch.nn as nn

from util.random_seed import (
    DEFAULT_RANDOM_SEED_CONFIG_PATH,
    SeedIdentity,
    seed_process,
)

from models.contracts import (
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
    ModelSpec,
)
from models.ecgfounder import build_ecgfounder
from models.efficientnet1d import build_efficientnet1dv2


MODEL_SPECS = {
    EFFICIENTNET1DV2_SPEC.name: EFFICIENTNET1DV2_SPEC,
    ECGFOUNDER_SPEC.name: ECGFOUNDER_SPEC,
}
MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "efficientnet1dv2": build_efficientnet1dv2,
    "ecgfounder": build_ecgfounder,
}
MODEL_ALIASES = {
    "effnet": "efficientnet1dv2",
    "efficientnet": "efficientnet1dv2",
    "efficientnet1d": "efficientnet1dv2",
    "ecg_founder": "ecgfounder",
}


def normalize_model_name(name: str) -> str:
    normalized = str(name).strip().lower()
    return MODEL_ALIASES.get(normalized, normalized)


def available_models() -> tuple[str, ...]:
    return tuple(sorted(MODEL_BUILDERS))


def get_model_spec(name: str) -> ModelSpec:
    normalized = normalize_model_name(name)
    try:
        return MODEL_SPECS[normalized]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; available={available_models()}"
        ) from None


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

    normalized = normalize_model_name(name)
    try:
        builder = MODEL_BUILDERS[normalized]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; available={available_models()}"
        ) from None
    resolved_seed_config = (
        Path(config_root).expanduser().resolve() / "random_seed.yaml"
        if seed_config_path is None and config_root is not None
        else DEFAULT_RANDOM_SEED_CONFIG_PATH
        if seed_config_path is None
        else Path(seed_config_path).expanduser().resolve()
    )
    random_identity: SeedIdentity = seed_process(
        seed_namespace,
        normalized,
        *tuple(seed_identity),
        config_path=resolved_seed_config,
    )
    model = builder(**kwargs)
    model.random_seed_identity = random_identity
    if getattr(model, "model_spec", None) != MODEL_SPECS[normalized]:
        raise RuntimeError(f"builder {normalized} returned a model with wrong spec")
    return model


__all__ = [
    "MODEL_ALIASES",
    "MODEL_BUILDERS",
    "MODEL_SPECS",
    "available_models",
    "build_model",
    "get_model_spec",
    "normalize_model_name",
]
