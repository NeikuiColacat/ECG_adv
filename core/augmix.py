"""Two-chain AugMix strong-view generation for Stage-1 SimCLR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from core.corruption import (
    CANONICAL_OPERATORS,
    INPUT_POINTS,
    INPUT_SAMPLING_RATE_HZ,
    generate_canonical_corruption,
)
from util.config_bundle import resolve_config_reference, resolve_entry_config_path
from util.augmentations.profile import (
    AugmentationProfile,
    load_augmentation_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUGMIX_CONFIG_PATH = PROJECT_ROOT / "configs" / "train" / "augmix.yaml"


@dataclass(frozen=True)
class AugMixConfig:
    config_path: Path
    random_seed_config_path: Path
    random_namespace: str
    operator_profile_config: AugmentationProfile
    canonical_operators: tuple[str, ...]
    stage1_width: int
    stage1_dirichlet_alpha: float
    stage1_beta_alpha: float
    stage1_simclr_temperature: float


@dataclass(frozen=True)
class TwoChainAugMixBatch:
    """One Stage-1 strong view built from two independent corruption chains."""

    chain1_raw: torch.Tensor
    chain2_raw: torch.Tensor
    mixed_raw: torch.Tensor
    mixture_weights: torch.Tensor
    augmented_strength: torch.Tensor
    chain1_composition_index: torch.Tensor
    chain2_composition_index: torch.Tensor
    chain1_depth: torch.Tensor
    chain2_depth: torch.Tensor
    chain1_operator_mask: torch.Tensor
    chain2_operator_mask: torch.Tensor
    chain1_output_nonfinite_count: torch.Tensor
    chain2_output_nonfinite_count: torch.Tensor
    operator_domain_sampling_rate_hz: int


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def load_augmix_config(
    path: str | Path = DEFAULT_AUGMIX_CONFIG_PATH,
    *,
    config_root: str | Path | None = None,
) -> AugMixConfig:
    config_path = resolve_entry_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"AugMix config not found: {config_path}")
    payload = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "AugMix config"
    )
    expected_root_keys = {
        "schema_version",
        "method",
        "corruption_chains",
        "stage1_twochain",
    }
    if set(payload) != expected_root_keys:
        raise ValueError("AugMix config keys are incomplete or unexpected")
    if payload.get("schema_version") != 1:
        raise ValueError("AugMix config schema_version must be 1")
    method = _mapping(payload.get("method"), "method")
    if set(method) != {
        "name",
        "random_seed_file",
        "random_namespace",
        "corruption_timing",
    }:
        raise ValueError("AugMix method keys are incomplete or unexpected")
    if method.get("name") != "two_chain_augmix_simclr":
        raise ValueError("AugMix method must be two_chain_augmix_simclr")
    if method.get("corruption_timing") != "before_per_sample_global_zscore":
        raise ValueError("AugMix corruption timing must precede global z-score")
    corruptions = _mapping(payload.get("corruption_chains"), "corruption_chains")
    if set(corruptions) != {
        "operator_config",
        "profile",
        "profile_status",
        "severity",
        "depths",
        "sampling",
        "replacement_within_chain",
        "canonical_operator_order",
        "independent_chains",
    }:
        raise ValueError("corruption_chains keys are incomplete or unexpected")
    stage1_twochain = _mapping(
        payload.get("stage1_twochain"), "stage1_twochain"
    )
    if set(stage1_twochain) != {
        "width",
        "view",
        "chain_sampling",
        "dirichlet_alpha",
        "clean_mix",
        "beta_alpha",
        "simclr_temperature",
    }:
        raise ValueError("stage1_twochain keys are incomplete or unexpected")
    seed_config_path = resolve_config_reference(
        method.get("random_seed_file"),
        owner_config_path=config_path,
        config_root=config_root,
        description="method.random_seed_file",
        must_exist=True,
    )
    operator_config_path = resolve_config_reference(
        corruptions.get("operator_config"),
        owner_config_path=config_path,
        config_root=config_root,
        description="corruption_chains.operator_config",
        must_exist=True,
    )
    operator_profile_name = str(corruptions.get("profile", ""))
    operator_severity = int(corruptions.get("severity", 0))
    canonical_operators = tuple(
        str(value)
        for value in corruptions.get("canonical_operator_order", ())
    )
    operator_profile_config = load_augmentation_profile(
        operator_config_path,
        profile_name=operator_profile_name,
        severity=operator_severity,
        expected_canonical_order=canonical_operators,
        expected_seed_config_path=seed_config_path,
        config_root=config_root,
    )
    config = AugMixConfig(
        config_path=config_path,
        random_seed_config_path=seed_config_path,
        random_namespace=str(method.get("random_namespace", "")),
        operator_profile_config=operator_profile_config,
        canonical_operators=canonical_operators,
        stage1_width=int(stage1_twochain.get("width", 0)),
        stage1_dirichlet_alpha=float(
            stage1_twochain.get("dirichlet_alpha", 0.0)
        ),
        stage1_beta_alpha=float(stage1_twochain.get("beta_alpha", 0.0)),
        stage1_simclr_temperature=float(
            stage1_twochain.get("simclr_temperature", 0.0)
        ),
    )
    if tuple(int(value) for value in corruptions.get("depths", ())) != (2, 3):
        raise ValueError("main AugMix corruption depths must be [2, 3]")
    if config.canonical_operators != CANONICAL_OPERATORS:
        raise ValueError("AugMix canonical operator order is invalid")
    if corruptions.get("sampling") != "uniform_over_all_depth2_depth3_compositions":
        raise ValueError("AugMix composition sampler must be uniform over depth2+3")
    if (
        corruptions.get("profile_status")
        != "project_defined_paper_anchored_not_official_preset"
    ):
        raise ValueError("AugMix profile status must not mislabel the custom preset")
    if bool(corruptions.get("replacement_within_chain", True)):
        raise ValueError("operators may not repeat within one corruption chain")
    if not bool(corruptions.get("independent_chains", False)):
        raise ValueError("AugMix corruption chains must be independently sampled")
    if (
        config.stage1_width != 2
        or stage1_twochain.get("view") != "one_strong_view"
        or stage1_twochain.get("chain_sampling")
        != "independent_sequential_locked_rng"
        or stage1_twochain.get("clean_mix") != "beta"
    ):
        raise ValueError("Stage-1 AugMix must use the locked two-chain strong view")
    if (
        config.stage1_dirichlet_alpha != 0.5
        or config.stage1_beta_alpha != 0.5
        or config.stage1_simclr_temperature != 0.5
    ):
        raise ValueError(
            "Stage-1 AugMix locks Dirichlet/Beta alpha and SimCLR temperature to 0.5"
        )
    if not config.random_namespace:
        raise ValueError("AugMix random namespace must be recorded")
    return config


def _load_operator_profile(config: AugMixConfig) -> dict[str, dict[str, Any]]:
    """Return independent kwargs from the already validated shared profile."""

    return {
        operator: config.operator_profile_config.parameters_for(operator)
        for operator in config.canonical_operators
    }


def _validate_waveform(value: torch.Tensor, description: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{description} must be a torch.Tensor")
    if value.ndim != 3 or value.shape[0] < 1 or value.shape[1] < 2 or value.shape[2] != 12:
        raise ValueError(f"{description} must have shape (B,time,12)")
    if not value.is_floating_point():
        raise ValueError(f"{description} must be floating-point raw mV")
    return value


def _validate_generator(generator: torch.Generator, device: torch.device) -> None:
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        raise ValueError("AugMix generator device must match waveform device")
    if device.type == "cuda":
        # ``tensor.device`` may be the current-device alias ``cuda`` while a
        # generator reports the equivalent explicit device ``cuda:0``.  Resolve
        # both aliases before comparing; otherwise a valid single-visible-GPU
        # run is rejected before augmentation starts.
        current_index = torch.cuda.current_device()
        waveform_index = current_index if device.index is None else device.index
        generator_index = (
            current_index
            if generator_device.index is None
            else generator_device.index
        )
        if generator_index != waveform_index:
            raise ValueError(
                "AugMix generator CUDA index must match waveform CUDA index"
            )


def _symmetric_beta(
    batch: int,
    *,
    alpha: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    concentration = torch.full(
        (batch, 2),
        float(alpha),
        device=device,
        dtype=torch.float32,
    )
    gamma = torch._standard_gamma(concentration, generator=generator).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return (gamma[:, 0] / gamma.sum(dim=1)).contiguous()


def _dirichlet(
    batch: int,
    width: int,
    *,
    alpha: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    concentration = torch.full(
        (batch, width),
        float(alpha),
        device=device,
        dtype=torch.float32,
    )
    gamma = torch._standard_gamma(concentration, generator=generator).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return (gamma / gamma.sum(dim=1, keepdim=True)).contiguous()


def generate_two_chain_augmix_strong_view(
    clean_raw: torch.Tensor,
    *,
    sampling_rate_hz: int,
    config: AugMixConfig | None = None,
    generator: torch.Generator,
) -> TwoChainAugMixBatch:
    """Generate the frozen Stage-1 two-chain AugMix strong view.

    The two corruption calls deliberately remain sequential.  This preserves
    the stochastic identity of the development lock; combining them into one
    2B call would change the operator RNG stream even though it is faster.
    """

    resolved = load_augmix_config() if config is None else config
    clean = _validate_waveform(clean_raw, "clean_raw")
    if int(sampling_rate_hz) != INPUT_SAMPLING_RATE_HZ:
        raise ValueError("Stage-1 AugMix accepts only canonical raw 100 Hz input")
    if tuple(clean.shape[1:]) != (INPUT_POINTS, 12):
        raise ValueError("Stage-1 AugMix expects raw (B,1000,12) input")
    if not bool(torch.isfinite(clean).all().item()):
        raise ValueError("Stage-1 AugMix clean input must be finite raw mV")
    _validate_generator(generator, clean.device)

    operator_params = _load_operator_profile(resolved)
    first = generate_canonical_corruption(
        clean,
        operator_params=operator_params,
        generator=generator,
        _input_prevalidated=True,
    )
    second = generate_canonical_corruption(
        clean,
        operator_params=operator_params,
        generator=generator,
        _input_prevalidated=True,
    )
    batch = int(clean.shape[0])
    weights = _dirichlet(
        batch,
        resolved.stage1_width,
        alpha=resolved.stage1_dirichlet_alpha,
        device=clean.device,
        generator=generator,
    )
    augmented_strength = _symmetric_beta(
        batch,
        alpha=resolved.stage1_beta_alpha,
        device=clean.device,
        generator=generator,
    )
    mixture = (
        weights[:, 0].view(-1, 1, 1) * first.waveform_raw_100hz
        + weights[:, 1].view(-1, 1, 1) * second.waveform_raw_100hz
    )
    strength = augmented_strength.view(-1, 1, 1)
    mixed = (
        (1.0 - strength) * clean.to(dtype=torch.float32)
        + strength * mixture
    )
    return TwoChainAugMixBatch(
        chain1_raw=first.waveform_raw_100hz,
        chain2_raw=second.waveform_raw_100hz,
        mixed_raw=mixed.contiguous(),
        mixture_weights=weights,
        augmented_strength=augmented_strength,
        chain1_composition_index=first.diagnostics.composition_index,
        chain2_composition_index=second.diagnostics.composition_index,
        chain1_depth=first.diagnostics.depth,
        chain2_depth=second.diagnostics.depth,
        chain1_operator_mask=first.diagnostics.operator_mask,
        chain2_operator_mask=second.diagnostics.operator_mask,
        chain1_output_nonfinite_count=(
            first.diagnostics.output_nonfinite_count
        ),
        chain2_output_nonfinite_count=(
            second.diagnostics.output_nonfinite_count
        ),
        operator_domain_sampling_rate_hz=(
            first.diagnostics.operator_domain_sampling_rate_hz
        ),
    )


__all__ = [
    "AugMixConfig",
    "TwoChainAugMixBatch",
    "CANONICAL_OPERATORS",
    "DEFAULT_AUGMIX_CONFIG_PATH",
    "generate_two_chain_augmix_strong_view",
    "load_augmix_config",
]
