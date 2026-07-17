"""PN2021 K500 data adapter for the typed-method online trainer.

This module is the only training-side owner of PN2021 split/cache selection.
It obtains raw 100 Hz K500 records exclusively through ``data_runtime`` and
passes caller-owned models/components into :mod:`core.online_trainer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from core.latent_pool import LatentPool, build_latent_pool
from core.methods import CompiledMethod, compile_method_profile
from core.online_trainer import (
    ALLOWED_CENTERS,
    DEFAULT_ONLINE_CONFIG_PATH,
    OnlineTrainingResult,
    load_online_train_config,
    resolve_online_training_parameters,
    train_online_model,
)
from core.pn2021_tuning import validate_locked_source_checkpoint
from data_preprocess.data_runtime import RuntimeDataLoader, get_dataloader
from models.contracts import ECGFOUNDER_SPEC, EFFICIENTNET1DV2_SPEC
from util.config_bundle import resolve_config_reference


PN2021_DATALOADER_PARAMETER_NAMES = frozenset(
    {
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "cache_mode",
        "validate_values",
        "drop_last",
        "selection_resident",
        "selection_resident_pin_memory",
    }
)

# Selection residency was introduced by the runtime-v2 execution profile.  Keep
# these defaults outside the locked v1 YAML contract so historical configs can
# still be replayed byte-for-byte with their original multi-worker mmap path.
_OPTIONAL_DATALOADER_DEFAULTS = {
    "selection_resident": False,
    "selection_resident_pin_memory": False,
}


def load_pn2021_method_profile(
    method_config_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_ONLINE_CONFIG_PATH,
    config_root: str | Path | None = None,
) -> tuple[CompiledMethod, Any]:
    """Compile one method profile from the same portable config bundle."""

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
            "method profile must belong to the selected config bundle"
        ) from None
    return compile_method_profile(method_path), online


def _resolve_loader_parameters(
    training: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = {} if overrides is None else dict(overrides)
    unknown = sorted(set(supplied) - PN2021_DATALOADER_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown PN2021 dataloader parameters: {unknown}")
    resolved = {}
    for key in PN2021_DATALOADER_PARAMETER_NAMES:
        if key in supplied:
            resolved[key] = supplied[key]
        elif key in training:
            resolved[key] = training[key]
        elif key in _OPTIONAL_DATALOADER_DEFAULTS:
            resolved[key] = _OPTIONAL_DATALOADER_DEFAULTS[key]
        else:
            raise ValueError(f"training.{key} is required")
    workers = resolved["num_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    for key in ("prefetch_factor",):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "pin_memory",
        "persistent_workers",
        "drop_last",
        "selection_resident",
        "selection_resident_pin_memory",
    ):
        if not isinstance(resolved[key], bool):
            raise ValueError(f"{key} must be boolean")
    if resolved["drop_last"]:
        raise ValueError("drop_last=true violates the complete K500 protocol")
    if workers == 0:
        if "persistent_workers" in supplied and resolved["persistent_workers"]:
            raise ValueError("persistent_workers requires num_workers > 0")
        resolved["persistent_workers"] = False
    if resolved["cache_mode"] not in {"auto", "ram", "mmap"}:
        raise ValueError("cache_mode must be auto, ram or mmap")
    if workers > 0 and resolved["cache_mode"] != "mmap":
        raise ValueError("multi-worker PN2021 loading requires mmap")
    if resolved["selection_resident_pin_memory"] and not resolved[
        "selection_resident"
    ]:
        raise ValueError(
            "selection_resident_pin_memory requires selection_resident=true"
        )
    if resolved["selection_resident"]:
        if workers != 0:
            raise ValueError("selection-resident K500 loading requires num_workers=0")
        if resolved["persistent_workers"]:
            raise ValueError(
                "selection-resident K500 loading requires persistent_workers=false"
            )
        if resolved["cache_mode"] != "mmap":
            raise ValueError(
                "selection-resident K500 loading requires a mmap source cache"
            )
    if resolved["validate_values"] not in {"none", "sample", "full"}:
        raise ValueError("validate_values must be none, sample or full")
    return resolved, supplied


def build_pn2021_k500_dataloader(
    *,
    center: str,
    model_name: str,
    shuffle: bool,
    config_path: str | Path = DEFAULT_ONLINE_CONFIG_PATH,
    config_root: str | Path | None = None,
    training_parameters: Mapping[str, Any] | None = None,
    dataloader_parameters: Mapping[str, Any] | None = None,
) -> RuntimeDataLoader:
    """Build a raw-mV ``(B,1000,12)`` K500 loader for one logical center."""

    if center not in ALLOWED_CENTERS:
        raise ValueError(f"center must be one of {ALLOWED_CENTERS}")
    if model_name not in {EFFICIENTNET1DV2_SPEC.name, ECGFOUNDER_SPEC.name}:
        raise ValueError("model_name must be efficientnet1dv2 or ecgfounder")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be boolean")
    config = load_online_train_config(config_path, config_root=config_root)
    online, _ = resolve_online_training_parameters(
        config, model_name, training_parameters
    )
    loader, _ = _resolve_loader_parameters(
        config.payload["training"], dataloader_parameters
    )
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
    return get_dataloader(
        dataset="pn2021",
        partition="k500",
        logical_center=center,
        sampling_rate_hz=100,
        batch_size=int(online["batch_size"]),
        shuffle=shuffle,
        num_workers=int(loader["num_workers"]),
        pin_memory=bool(loader["pin_memory"]),
        persistent_workers=bool(loader["persistent_workers"]),
        prefetch_factor=int(loader["prefetch_factor"]),
        cache_mode=loader["cache_mode"],
        validate_values=loader["validate_values"],
        selection_resident=bool(loader["selection_resident"]),
        selection_resident_pin_memory=bool(
            loader["selection_resident_pin_memory"]
        ),
        drop_last=bool(loader["drop_last"] if shuffle else False),
        prepare_for_model=False,
        sanitize=False,
        global_zscore=False,
        output_layout="time_channel",
        seed_namespace=seed_namespace,
        seed_config_path=config.references["random_seed_config"],
        config_root=config.config_root,
        split_config_path=config.references["split_config"],
        data_load_config_path=config.references["data_load_config"],
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
    center: str,
    model_name: str,
    method_config_path: str | Path,
    config_path: str | Path = DEFAULT_ONLINE_CONFIG_PATH,
    config_root: str | Path | None = None,
    device: str | torch.device | None = None,
    training_parameters: Mapping[str, Any] | None = None,
    dataloader_parameters: Mapping[str, Any] | None = None,
) -> LatentPool:
    """Encode the complete clean K500 selection in deterministic split order."""

    loader = build_pn2021_k500_dataloader(
        center=center,
        model_name=model_name,
        shuffle=False,
        config_path=config_path,
        config_root=config_root,
        training_parameters=training_parameters,
        dataloader_parameters=dataloader_parameters,
    )
    try:
        method, config = load_pn2021_method_profile(
            method_config_path,
            config_path=config_path,
            config_root=config_root,
        )
        if not method.requirements.latent_pool:
            raise ValueError("method does not require a latent pool")
        resource = method.resources.get("lhat_config")
        if not isinstance(resource, Mapping):
            raise ValueError("latent-pool method must declare lhat_config")
        lhat = resolve_config_reference(
            resource.get("path"),
            owner_config_path=method.source_path,
            config_root=config.config_root,
            description="method.resources.lhat_config",
            must_exist=True,
        )
        from core.lhat import load_lhat_config

        lhat_config = load_lhat_config(lhat, config_root=config.config_root)
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
    device: str | torch.device | None = None,
    pos_weight: torch.Tensor | Sequence[float] | None = None,
    training_parameters: Mapping[str, Any] | None = None,
    dataloader_parameters: Mapping[str, Any] | None = None,
) -> OnlineTrainingResult:
    """Train one typed K500 method through the managed data runtime."""

    method, config = load_pn2021_method_profile(
        method_config_path,
        config_path=config_path,
        config_root=config_root,
    )
    if not method.executable:
        raise ValueError(f"method {method.profile_name!r} is not executable")
    owner = model.module if hasattr(model, "module") else model
    spec = getattr(owner, "model_spec", None)
    if spec not in {EFFICIENTNET1DV2_SPEC, ECGFOUNDER_SPEC}:
        raise ValueError("model must expose a managed EfficientNet/ECGFounder spec")
    validate_locked_source_checkpoint(model, config)
    if not method.requirements.latent_pool:
        if encoder is not None or decoder is not None:
            raise ValueError("method without latent resources rejects VAE components")
        pool = None
    else:
        if encoder is None or decoder is None:
            raise ValueError("latent method requires frozen VAE encoder and decoder")
        requested_device = str(device or config.payload["training"]["device"])
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_device = torch.device(requested_device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        encoder.to(resolved_device).eval()
        try:
            pool = build_pn2021_latent_pool(
                encoder,
                center=center,
                model_name=spec.name,
                method_config_path=method.source_path,
                config_path=config_path,
                config_root=config_root,
                device=resolved_device,
                training_parameters=training_parameters,
                dataloader_parameters=dataloader_parameters,
            )
        finally:
            # The encoder is no longer needed after deterministic K500 means
            # are materialized. Release its accelerator allocation before the
            # outer classifier loop; the decoder is moved only afterwards to
            # avoid holding both large VAE components on the GPU at once.
            encoder.to("cpu")
        decoder.to(resolved_device).eval()

    loader: RuntimeDataLoader | None = None
    try:
        loader = build_pn2021_k500_dataloader(
            center=center,
            model_name=spec.name,
            shuffle=True,
            config_path=config_path,
            config_root=config_root,
            training_parameters=training_parameters,
            dataloader_parameters=dataloader_parameters,
        )
        return train_online_model(
            model,
            loader,
            center=center,
            method_config_path=method.source_path,
            latent_pool=pool,
            decoder=decoder,
            config_path=config_path,
            config_root=config_root,
            output_dir=output_dir,
            device=device,
            training_parameters=training_parameters,
            pos_weight=pos_weight,
        )
    finally:
        if loader is not None:
            loader.close()
        if decoder is not None:
            decoder.to("cpu")


__all__ = [
    "PN2021_DATALOADER_PARAMETER_NAMES",
    "build_pn2021_k500_dataloader",
    "build_pn2021_latent_pool",
    "load_pn2021_method_profile",
    "train_pn2021",
]
