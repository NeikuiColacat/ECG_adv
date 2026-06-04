"""Registry for typed YAML runner command adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .ecgfounder_vae_lhat import build_ecgfounder_vae_lhat_argv
from .effnet_vae_lhat import build_effnet_vae_lhat_argv


RunnerAdapterBuilder = Callable[[Mapping[str, Any], Mapping[str, Any]], list[Any]]


RUNNER_ADAPTERS: dict[str, RunnerAdapterBuilder] = {
    "ecgfounder_vae_lhat": build_ecgfounder_vae_lhat_argv,
    "effnet_vae_lhat": build_effnet_vae_lhat_argv,
}


def runner_adapter_names() -> list[str]:
    """Return registered typed runner adapter names."""

    return sorted(RUNNER_ADAPTERS)


def build_runner_adapter_argv(
    adapter_name: str,
    *,
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> list[Any]:
    """Build legacy argv for a typed runner adapter."""

    try:
        builder = RUNNER_ADAPTERS[adapter_name]
    except KeyError as exc:
        allowed = ", ".join(runner_adapter_names())
        raise KeyError(f"Unknown runner.adapter {adapter_name!r}; allowed: {allowed}") from exc
    return builder(config, context)
