"""PN2021 K500 data adapter for the finite-recipe online trainer.

This module is the only training-side owner of PN2021 split/cache selection.
It obtains raw 100 Hz K500 records exclusively through ``data_runtime`` and
passes caller-owned models/components into :mod:`core.online_trainer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn as nn
import yaml

from core.corruption import generate_canonical_corruption
from core.latent_pool import build_latent_pool
from core.methods.registry import RecipeSpec, load_recipe_spec
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
from util.augmentations.profile import load_augmentation_profile
from util.random_seed import make_torch_generator


def _lhat_candidate_source_policy(recipe: RecipeSpec) -> str:
    auxiliary = recipe.scientific_contract.get("lhat_auxiliary")
    if not isinstance(auxiliary, Mapping):
        auxiliary = recipe.scientific_contract.get("auxiliary")
    if not isinstance(auxiliary, Mapping):
        return "clean_only"
    return str(auxiliary.get("candidate_source_policy", "clean_only"))


def _mixed_candidate_pool_batches(
    batches: Iterable[Mapping[str, Any]],
    *,
    recipe: RecipeSpec,
    config: Any,
    center: str,
    model_name: str,
    device: torch.device,
) -> Iterator[Mapping[str, Any]]:
    """Attach one deterministic depth-2/3 corruption per ordered K500 record."""

    resource = recipe.resources.get("operator_profile")
    if not isinstance(resource, Mapping):
        raise ValueError("mixed LHAT candidates require operator_profile")
    operator_path = resolve_config_reference(
        resource.get("path"),
        owner_config_path=recipe.source_path,
        config_root=config.config_root,
        description="method.resources.operator_profile",
        must_exist=True,
    )
    profile = load_augmentation_profile(
        operator_path,
        profile_name=str(resource.get("profile", "")),
        severity=int(resource.get("severity", 0)),
        config_root=config.config_root,
    )
    parameters = {
        name: profile.parameters_for(name) for name in profile.canonical_order
    }
    seed = config.payload["random_seed"]
    generator = make_torch_generator(
        device,
        "lhat_candidate_corruption_bank_v1",
        str(seed["comparison_group"]),
        int(seed["replicate_id"]),
        center,
        model_name,
        config_path=config.references["random_seed_config"],
    )
    for raw_batch in batches:
        if not isinstance(raw_batch, Mapping):
            raise TypeError("ordered latent-pool batch must be a mapping")
        waveform_keys = tuple(
            name for name in ("waveform", "waveform_raw") if name in raw_batch
        )
        if len(waveform_keys) != 1:
            raise ValueError(
                "ordered latent-pool batch must expose one raw waveform"
            )
        waveform = torch.as_tensor(raw_batch[waveform_keys[0]]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        selection = torch.as_tensor(raw_batch["selection_index"]).to(
            device=device, dtype=torch.int64, non_blocking=True
        )
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if selection.ndim == 0:
            selection = selection.unsqueeze(0)
        if waveform.shape != (int(selection.shape[0]), 1000, 12):
            raise ValueError("mixed candidate waveform/selection shapes differ")
        corrupted = generate_canonical_corruption(
            waveform,
            operator_params=parameters,
            generator=generator,
            composition_indices=selection.remainder(20),
        )
        yield {
            **raw_batch,
            "candidate_corrupted_waveform": corrupted.waveform_raw_100hz,
        }


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
    requires_pool = requires_decoder = recipe.requires_vae
    if requires_pool != (encoder is not None):
        raise ValueError(
            "VAE encoder presence must exactly match the latent-pool requirement"
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
            ordered_loader = loader_plan.open_ordered()
            try:
                identity = getattr(encoder, "checkpoint_identity", None)
                encoder_sha256 = getattr(identity, "sha256", None)
                if not isinstance(encoder_sha256, str) or len(encoder_sha256) != 64:
                    raise ValueError(
                        "VAE encoder must carry its strict checkpoint SHA256 identity"
                    )
                candidate_source_policy = _lhat_candidate_source_policy(recipe)
                pool_batches: Iterable[Mapping[str, Any]] = ordered_loader
                if candidate_source_policy != "clean_only":
                    pool_batches = _mixed_candidate_pool_batches(
                        ordered_loader,
                        recipe=recipe,
                        config=config,
                        center=center,
                        model_name=spec.name,
                        device=resolved_device,
                    )
                pool = build_latent_pool(
                    encoder,
                    pool_batches,
                    encoder_identity=encoder_sha256,
                    device=resolved_device,
                    num_candidates=lhat_config.num_candidates,
                    standardizer_epsilon=lhat_config.standardizer_epsilon,
                    candidate_source_policy=candidate_source_policy,
                )
            finally:
                ordered_loader.close()
        finally:
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
    "load_pn2021_recipe_spec",
    "train_pn2021",
]
