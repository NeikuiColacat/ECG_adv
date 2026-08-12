"""PN2021 K500 data adapter for the finite-recipe online trainer.

This module is the only training-side owner of PN2021 split/cache selection.
It obtains raw 100 Hz K500 records exclusively through ``data_runtime`` and
passes caller-owned models/components into :mod:`core.online_trainer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import yaml

from core.latent_pool import LatentPool, build_latent_pool
from core.methods import RecipeSpec, load_recipe_spec
from core.online_trainer import (
    ALLOWED_CENTERS,
    DEFAULT_ONLINE_CONFIG_PATH,
    OnlineTrainingResult,
    load_online_train_config,
    resolve_online_training_parameters,
    train_online_model,
)
from data_preprocess.data_runtime import PN2021K500LoaderPlan, RuntimeDataLoader
from models.checkpoints import CheckpointIdentity
from models.contracts import ECGFOUNDER_SPEC, EFFICIENTNET1DV2_SPEC
from util.config_bundle import resolve_config_reference


def load_pn2021_recipe_spec(
    method_config_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_ONLINE_CONFIG_PATH,
    config_root: str | Path | None = None,
) -> tuple[RecipeSpec, Any]:
    """Load one finite recipe from the same portable config bundle."""

    online = load_online_train_config(config_path, config_root=config_root)
    method_path = resolve_config_reference(
        str(method_config_path),
        owner_config_path=online.path,
        config_root=online.config_root,
        description="method_config_path",
        must_exist=True,
    )
    try:
        method_path.relative_to(online.config_root)
    except ValueError:
        raise ValueError(
            "recipe config must belong to the selected config bundle"
        ) from None
    return load_recipe_spec(method_path), online


def _validate_locked_source_checkpoint(
    owner: nn.Module,
    spec: Any,
    config: Any,
) -> None:
    registry_path = config.references.get("source_baseline_registry")
    if not isinstance(registry_path, Path):
        raise ValueError("config must resolve source_baseline_registry")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, Mapping) or registry.get("schema_version") != 1:
        raise ValueError("source baseline registry must be schema_version=1")
    models = registry.get("models")
    expected = models.get(spec.name) if isinstance(models, Mapping) else None
    if not isinstance(expected, Mapping):
        raise ValueError(f"source registry has no model {spec.name!r}")
    attribute = (
        "checkpoint_identity"
        if spec == EFFICIENTNET1DV2_SPEC
        else "task_checkpoint_identity"
    )
    identity = getattr(owner, attribute, None)
    if not isinstance(identity, CheckpointIdentity):
        raise ValueError(f"PN2021 model must carry strict {attribute}")
    if identity.sha256 != expected.get("selected_checkpoint_sha256"):
        raise ValueError("model source checkpoint SHA256 differs from locked registry")
    if identity.path.resolve() != Path(expected["selected_checkpoint"]).resolve():
        raise ValueError("model source checkpoint path differs from locked registry")


def build_pn2021_k500_loader_plan(
    *,
    center: str,
    model_name: str,
    config_path: str | Path = DEFAULT_ONLINE_CONFIG_PATH,
    config_root: str | Path | None = None,
) -> PN2021K500LoaderPlan:
    """Resolve the sole raw-100-Hz K500 loader plan from tracked YAML."""

    if center not in ALLOWED_CENTERS:
        raise ValueError(f"center must be one of {ALLOWED_CENTERS}")
    if model_name not in {EFFICIENTNET1DV2_SPEC.name, ECGFOUNDER_SPEC.name}:
        raise ValueError("model_name must be efficientnet1dv2 or ecgfounder")
    config = load_online_train_config(config_path, config_root=config_root)
    online, supplied = resolve_online_training_parameters(config, model_name)
    if supplied:
        raise AssertionError("PN2021 training parameters must be YAML-owned")
    training = config.payload["training"]
    seed = config.payload["random_seed"]
    seed_namespace = ":".join(
        (
            "pn2021_k500_matched",
            str(seed["comparison_group"]),
            str(seed["replicate_id"]),
            center,
            model_name,
        )
    )
    return PN2021K500LoaderPlan(
        center=center,
        batch_size=online["batch_size"],
        num_workers=training["num_workers"],
        pin_memory=training["pin_memory"],
        persistent_workers=training["persistent_workers"],
        prefetch_factor=training["prefetch_factor"],
        cache_mode=training["cache_mode"],
        validate_values=training["validate_values"],
        selection_resident=training["selection_resident"],
        selection_resident_pin_memory=training["selection_resident_pin_memory"],
        drop_last=training["drop_last"],
        seed_namespace=seed_namespace,
        split_config_path=config.references["split_config"],
        data_load_config_path=config.references["data_load_config"],
        seed_config_path=config.references["random_seed_config"],
    )


def _encoder_sha256(encoder: nn.Module) -> str:
    identity = getattr(encoder, "checkpoint_identity", None)
    value = getattr(identity, "sha256", None)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(
            "VAE encoder must carry its strict checkpoint SHA256 identity"
        )
    return value


def build_pn2021_latent_pool(
    encoder: nn.Module,
    *,
    loader_plan: PN2021K500LoaderPlan,
    method_config_path: str | Path,
    config_path: str | Path = DEFAULT_ONLINE_CONFIG_PATH,
    config_root: str | Path | None = None,
    device: str | torch.device | None = None,
) -> LatentPool:
    """Encode complete clean K500 in deterministic cache-index order."""

    recipe, config = load_pn2021_recipe_spec(
        method_config_path,
        config_path=config_path,
        config_root=config_root,
    )
    if not recipe.requirements.latent_pool:
        raise ValueError("recipe does not require a latent pool")
    resource = recipe.resources.get("lhat_config")
    if not isinstance(resource, Mapping):
        raise ValueError("latent-pool recipe must declare lhat_config")
    lhat = resolve_config_reference(
        resource.get("path"),
        owner_config_path=recipe.source_path,
        config_root=config.config_root,
        description="method.resources.lhat_config",
        must_exist=True,
    )
    from core.lhat import load_lhat_config

    lhat_config = load_lhat_config(lhat, config_root=config.config_root)
    loader = loader_plan.open_ordered()
    try:
        return build_latent_pool(
            encoder,
            loader,
            encoder_identity=_encoder_sha256(encoder),
            device=device,
            num_candidates=lhat_config.num_candidates,
            standardizer_epsilon=lhat_config.standardizer_epsilon,
        )
    finally:
        loader.close()


def train_pn2021(
    model: nn.Module,
    *,
    center: str,
    method_config_path: str | Path,
    encoder: nn.Module | None = None,
    decoder: nn.Module | None = None,
    config_path: str | Path = DEFAULT_ONLINE_CONFIG_PATH,
    config_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> OnlineTrainingResult:
    """Train one finite K500 recipe through the managed data runtime."""

    recipe, config = load_pn2021_recipe_spec(
        method_config_path,
        config_path=config_path,
        config_root=config_root,
    )
    owner = model.module if hasattr(model, "module") else model
    spec = getattr(owner, "model_spec", None)
    if spec not in {EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC}:
        raise ValueError("model must expose a managed EfficientNet/ECGFounder spec")
    _validate_locked_source_checkpoint(owner, spec, config)
    requires_pool = bool(recipe.requirements.latent_pool)
    requires_runtime_encoder = bool(recipe.requirements.vae_encoder)
    requires_decoder = bool(recipe.requirements.vae_decoder)
    requires_encoder_component = requires_pool or requires_runtime_encoder
    if requires_encoder_component != (encoder is not None):
        raise ValueError(
            "VAE encoder presence must match the recipe pool/runtime requirement"
        )
    if requires_decoder != (decoder is not None):
        raise ValueError(
            "VAE decoder presence must exactly match the recipe requirement"
        )

    loader_plan = build_pn2021_k500_loader_plan(
        center=center,
        model_name=spec.name,
        config_path=config.path,
        config_root=config.config_root,
    )

    pool = None
    runtime_encoder = encoder if requires_runtime_encoder else None
    if requires_pool:
        assert encoder is not None
        requested_device = str(config.payload["training"]["device"])
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_device = torch.device(requested_device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        encoder.to(resolved_device).eval()
        try:
            pool = build_pn2021_latent_pool(
                encoder,
                loader_plan=loader_plan,
                method_config_path=method_config_path,
                config_path=config_path,
                config_root=config_root,
                device=resolved_device,
            )
        finally:
            if not requires_runtime_encoder:
                # Pool-only methods release the encoder before the classifier
                # loop. Methods that explicitly require it keep the same
                # managed component for online waveform-to-latent generation.
                encoder.to("cpu")

    loader: RuntimeDataLoader | None = None
    try:
        loader = loader_plan.open_training()
        return train_online_model(
            model,
            loader,
            center=center,
            method_config_path=method_config_path,
            latent_pool=pool,
            encoder=runtime_encoder,
            decoder=decoder,
            config_path=config_path,
            config_root=config_root,
            output_dir=output_dir,
        )
    finally:
        if loader is not None:
            loader.close()
        if encoder is not None:
            encoder.to("cpu")
        if decoder is not None:
            decoder.to("cpu")


__all__ = [
    "build_pn2021_k500_loader_plan",
    "build_pn2021_latent_pool",
    "load_pn2021_recipe_spec",
    "train_pn2021",
]
