"""Three-chain ECG AugMix sample generation for online adversarial training.

Chains 1 and 2 are independent configured depth-2/3 corruption compositions.
Chain 3 is the VAE-LHAT raw waveform exactly as supplied: no extra corruption
is permitted on that branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from core.corruption import (
    CANONICAL_OPERATORS,
    CORRUPTION_DOMAIN_POINTS,
    CORRUPTION_DOMAIN_SAMPLING_RATE_HZ,
    INPUT_POINTS,
    INPUT_SAMPLING_RATE_HZ,
    INTERPOLATION_ALIGN_CORNERS,
    INTERPOLATION_MODE,
    OUTPUT_SAMPLING_RATE_HZ,
    OUTPUT_POINTS,
    generate_canonical_corruption,
)
from util.config_bundle import resolve_config_reference, resolve_entry_config_path
from util.random_seed import make_torch_generator
from util.augmentations.profile import (
    AugmentationProfile,
    load_augmentation_profile,
)
from util.augmentations.torch_operators import apply_operator_batch_prevalidated


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUGMIX_CONFIG_PATH = PROJECT_ROOT / "configs" / "train" / "augmix.yaml"


@dataclass(frozen=True)
class AugMixConfig:
    config_path: Path
    random_seed_config_path: Path
    random_namespace: str
    width: int
    chain_roles: tuple[str, str, str]
    input_sampling_rate_hz: int
    input_points: int
    operator_domain_sampling_rate_hz: int
    operator_domain_points: int
    output_sampling_rate_hz: int
    output_points: int
    interpolation_mode: str
    interpolation_align_corners: bool
    operator_config_path: Path
    operator_profile: str
    severity: int
    operator_profile_config: AugmentationProfile
    depths: tuple[int, ...]
    canonical_operators: tuple[str, ...]
    dirichlet_alpha: float
    beta_alpha: float
    vae_chain_weight_cap: float | None
    normalization_epsilon: float
    latent_threechain_width: int
    latent_threechain_depths: tuple[int, ...]
    latent_threechain_dirichlet_alpha: float
    latent_threechain_posterior_sample: bool
    latent_threechain_reconstruction_residual_bypass: bool
    latent_threechain_post_decode_clean_beta_mix: bool


@dataclass(frozen=True)
class AugMixBatch:
    chain1_raw: torch.Tensor
    chain2_raw: torch.Tensor
    chain3_raw: torch.Tensor
    mixed_raw: torch.Tensor
    mixed_normalized: torch.Tensor | None
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


@dataclass(frozen=True)
class LatentThreeChainAugMixBatch:
    """One decoded view mixed from three independently corrupted VAE latents."""

    mixed_raw: torch.Tensor
    mixture_weights: torch.Tensor
    augmented_strength: torch.Tensor
    residual_rms_ratio: torch.Tensor
    chain_depths: torch.Tensor
    chain_operator_mask: torch.Tensor
    chain_output_nonfinite_count: torch.Tensor
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
        "topology",
        "corruption_chains",
        "mixing",
        "output",
        "latent_threechain",
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
    topology = _mapping(payload.get("topology"), "topology")
    expected_topology_keys = {
        "width",
        "chain_roles",
        "chain3_additional_corruption",
        "raw_supervised_side_branch",
        "input_sampling_rate_hz",
        "input_points",
        "operator_domain_sampling_rate_hz",
        "operator_domain_points",
        "output_sampling_rate_hz",
        "output_points",
        "interpolation_mode",
        "interpolation_align_corners",
    }
    if set(topology) != expected_topology_keys:
        raise ValueError("AugMix topology keys are incomplete or unexpected")
    corruptions = _mapping(payload.get("corruption_chains"), "corruption_chains")
    mixing = _mapping(payload.get("mixing"), "mixing")
    output = _mapping(payload.get("output"), "output")
    latent_threechain = _mapping(
        payload.get("latent_threechain"), "latent_threechain"
    )
    expected_latent_threechain_keys = {
        "width",
        "depths",
        "depth_sampling",
        "operator_sampling",
        "operator_application_order",
        "dirichlet_alpha",
        "posterior_sample",
        "reconstruction_residual_bypass",
        "post_decode_clean_beta_mix",
    }
    if set(latent_threechain) != expected_latent_threechain_keys:
        raise ValueError("latent_threechain keys are incomplete or unexpected")
    cap = mixing.get("vae_chain_weight_cap")
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
        width=int(topology.get("width", 0)),
        chain_roles=tuple(str(value) for value in topology.get("chain_roles", ())),
        input_sampling_rate_hz=int(
            topology.get("input_sampling_rate_hz", 0)
        ),
        input_points=int(topology.get("input_points", 0)),
        operator_domain_sampling_rate_hz=int(
            topology.get("operator_domain_sampling_rate_hz", 0)
        ),
        operator_domain_points=int(topology.get("operator_domain_points", 0)),
        output_sampling_rate_hz=int(
            topology.get("output_sampling_rate_hz", 0)
        ),
        output_points=int(topology.get("output_points", 0)),
        interpolation_mode=str(
            topology.get("interpolation_mode", "")
        ),
        interpolation_align_corners=bool(
            topology.get("interpolation_align_corners", False)
        ),
        operator_config_path=operator_config_path,
        operator_profile=operator_profile_name,
        severity=operator_severity,
        operator_profile_config=operator_profile_config,
        depths=tuple(int(value) for value in corruptions.get("depths", ())),
        canonical_operators=canonical_operators,
        dirichlet_alpha=float(mixing.get("dirichlet_alpha", 0.0)),
        beta_alpha=float(mixing.get("beta_alpha", 0.0)),
        vae_chain_weight_cap=None if cap is None else float(cap),
        normalization_epsilon=float(output.get("normalization_epsilon", 0.0)),
        latent_threechain_width=int(latent_threechain.get("width", 0)),
        latent_threechain_depths=tuple(
            int(value) for value in latent_threechain.get("depths", ())
        ),
        latent_threechain_dirichlet_alpha=float(
            latent_threechain.get("dirichlet_alpha", 0.0)
        ),
        latent_threechain_posterior_sample=bool(
            latent_threechain.get("posterior_sample", True)
        ),
        latent_threechain_reconstruction_residual_bypass=bool(
            latent_threechain.get("reconstruction_residual_bypass", False)
        ),
        latent_threechain_post_decode_clean_beta_mix=bool(
            latent_threechain.get("post_decode_clean_beta_mix", True)
        ),
    )
    expected_roles = (
        "paper_anchored_s5_corruption_chain",
        "paper_anchored_s5_corruption_chain",
        "vae_lhat_adversarial_waveform",
    )
    if config.width != 3 or config.chain_roles != expected_roles:
        raise ValueError("main AugMix topology must be the locked three-chain design")
    if bool(topology.get("chain3_additional_corruption", True)):
        raise ValueError("AugMix chain 3 must not receive an additional corruption")
    if bool(topology.get("raw_supervised_side_branch", True)):
        raise ValueError("main AugMix must not add a raw-supervised side branch")
    if (
        config.input_sampling_rate_hz,
        config.operator_domain_sampling_rate_hz,
        config.output_sampling_rate_hz,
    ) != (
        INPUT_SAMPLING_RATE_HZ,
        CORRUPTION_DOMAIN_SAMPLING_RATE_HZ,
        OUTPUT_SAMPLING_RATE_HZ,
    ):
        raise ValueError("AugMix domain contract must be canonical 100->500->100 Hz")
    if (
        config.input_points,
        config.operator_domain_points,
        config.output_points,
    ) != (INPUT_POINTS, CORRUPTION_DOMAIN_POINTS, OUTPUT_POINTS):
        raise ValueError("AugMix point contract must be canonical 1000->5000->1000")
    if (
        config.interpolation_mode != "linear"
        or config.interpolation_align_corners is not True
    ):
        raise ValueError("AugMix resampling must use linear align_corners=true")
    if config.depths != (2, 3):
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
    # The first locked implementation intentionally uses alpha=1, which can be
    # sampled with isolated Torch generators without touching global RNG state.
    if config.dirichlet_alpha != 1.0 or config.beta_alpha != 1.0:
        raise ValueError("manual AugMix v1 currently locks Dirichlet/Beta alpha to 1")
    if config.vae_chain_weight_cap is not None:
        raise ValueError("main AugMix v1 uses uncapped transparent Dirichlet weights")
    if config.normalization_epsilon <= 0.0:
        raise ValueError("AugMix normalization epsilon must be positive")
    if config.latent_threechain_width != 3:
        raise ValueError("latent three-chain AugMix width must be 3")
    allowed_depth_profiles = {
        (1, 2, 3): "uniform_integer_1_to_3",
        (2, 3): "uniform_integer_2_to_3",
    }
    expected_depth_sampling = allowed_depth_profiles.get(
        config.latent_threechain_depths
    )
    if expected_depth_sampling is None:
        raise ValueError(
            "latent three-chain AugMix depths must be [1,2,3] or [2,3]"
        )
    if latent_threechain.get("depth_sampling") != expected_depth_sampling:
        raise ValueError(
            "latent three-chain depth_sampling does not match configured depths"
        )
    if (
        latent_threechain.get("operator_sampling")
        != "random_subset_without_replacement"
    ):
        raise ValueError(
            "latent three-chain operators must be a random subset without replacement"
        )
    if latent_threechain.get("operator_application_order") != "canonical_order":
        raise ValueError(
            "latent three-chain selected operators must use canonical application order"
        )
    if config.latent_threechain_dirichlet_alpha != 1.0:
        raise ValueError("latent three-chain Dirichlet alpha is locked to 1")
    if config.latent_threechain_posterior_sample:
        raise ValueError("latent three-chain encoding must use posterior means")
    if not isinstance(
        latent_threechain.get("reconstruction_residual_bypass"), bool
    ):
        raise ValueError(
            "latent_threechain.reconstruction_residual_bypass must be boolean"
        )
    if not isinstance(
        latent_threechain.get("post_decode_clean_beta_mix"), bool
    ):
        raise ValueError(
            "latent_threechain.post_decode_clean_beta_mix must be boolean"
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


def _dirichlet_one(
    batch: int,
    width: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    uniform = torch.rand(
        batch,
        width,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).clamp_min(torch.finfo(torch.float32).tiny)
    exponentials = -uniform.log()
    return exponentials / exponentials.sum(dim=1, keepdim=True)


def _global_zscore_btc(signal: torch.Tensor, epsilon: float) -> torch.Tensor:
    mean = signal.mean(dim=(1, 2), keepdim=True)
    std = signal.std(dim=(1, 2), correction=0, keepdim=True)
    return ((signal - mean) / std.clamp_min(float(epsilon))).contiguous()


def make_augmix_generator(
    device: str | torch.device,
    *identity: Any,
    config: AugMixConfig | None = None,
) -> torch.Generator:
    """Create one persistent isolated AugMix stream for a run/epoch/worker."""

    resolved = load_augmix_config() if config is None else config
    return make_torch_generator(
        device,
        resolved.random_namespace,
        *identity,
        config_path=resolved.random_seed_config_path,
    )


def generate_three_chain_augmix(
    clean_raw: torch.Tensor,
    vae_lhat_raw: torch.Tensor,
    *,
    sampling_rate_hz: int,
    config: AugMixConfig | None = None,
    generator: torch.Generator | None = None,
    include_normalized: bool = True,
) -> AugMixBatch:
    """Generate the locked three-chain AugMix view.

    Direct callers keep the historical normalized result by default.  The
    composable runtime consumes only ``mixed_raw`` and performs its single
    model-domain normalization later, so it may disable this otherwise unused
    full-waveform allocation without changing the method output.
    """

    resolved = load_augmix_config() if config is None else config
    clean = _validate_waveform(clean_raw, "clean_raw")
    adversarial = _validate_waveform(vae_lhat_raw, "vae_lhat_raw")
    if clean.shape != adversarial.shape:
        raise ValueError("clean and VAE-LHAT waveforms must have identical shape")
    if not bool(
        (torch.isfinite(clean).all() & torch.isfinite(adversarial).all()).item()
    ):
        raise ValueError("clean and VAE-LHAT waveforms must be finite raw mV")
    if int(sampling_rate_hz) != resolved.input_sampling_rate_hz:
        raise ValueError(
            "AugMix accepts only canonical raw 100 Hz input; "
            f"got sampling_rate_hz={sampling_rate_hz}"
        )
    expected_points = resolved.input_points
    if clean.shape[1] != expected_points:
        raise ValueError(
            f"{sampling_rate_hz} Hz AugMix expects {expected_points} points, "
            f"got {clean.shape[1]}"
        )
    if generator is None:
        raise ValueError(
            "generator is required; create one persistent stream with "
            "make_augmix_generator"
        )
    _validate_generator(generator, clean.device)
    if not isinstance(include_normalized, bool):
        raise TypeError("include_normalized must be boolean")

    operator_params = _load_operator_profile(resolved)
    batch = int(clean.shape[0])
    # Stack the two independent corruption branches into one 2B launch.  The
    # shared kernel samples one composition per row and invokes each operator
    # exactly once, rather than grouping up to twenty compositions in Python.
    corruption = generate_canonical_corruption(
        torch.cat((clean, clean), dim=0),
        operator_params=operator_params,
        generator=generator,
        _input_prevalidated=True,
    )
    chain1, chain2 = corruption.waveform_raw_100hz.split(batch, dim=0)
    diagnostics = corruption.diagnostics
    chain1_indices, chain2_indices = diagnostics.composition_index.split(batch)
    chain1_depth, chain2_depth = diagnostics.depth.split(batch)
    chain1_operator_mask, chain2_operator_mask = diagnostics.operator_mask.split(batch)
    chain1_nonfinite, chain2_nonfinite = diagnostics.output_nonfinite_count.split(batch)
    # Chain 3 is deliberately copied directly.  No operator is called on it.
    chain3 = adversarial.to(dtype=torch.float32).clone().contiguous()
    weights = _dirichlet_one(
        batch, resolved.width, device=clean.device, generator=generator
    )
    augmented_strength = torch.rand(
        batch,
        device=clean.device,
        dtype=torch.float32,
        generator=generator,
    )
    mixture = (
        weights[:, 0].view(-1, 1, 1) * chain1
        + weights[:, 1].view(-1, 1, 1) * chain2
        + weights[:, 2].view(-1, 1, 1) * chain3
    )
    strength = augmented_strength.view(-1, 1, 1)
    mixed = (1.0 - strength) * clean.to(dtype=torch.float32) + strength * mixture
    return AugMixBatch(
        chain1_raw=chain1,
        chain2_raw=chain2,
        chain3_raw=chain3,
        mixed_raw=mixed.contiguous(),
        mixed_normalized=(
            _global_zscore_btc(mixed, resolved.normalization_epsilon)
            if include_normalized
            else None
        ),
        mixture_weights=weights.contiguous(),
        augmented_strength=augmented_strength.contiguous(),
        chain1_composition_index=chain1_indices,
        chain2_composition_index=chain2_indices,
        chain1_depth=chain1_depth,
        chain2_depth=chain2_depth,
        chain1_operator_mask=chain1_operator_mask,
        chain2_operator_mask=chain2_operator_mask,
        chain1_output_nonfinite_count=chain1_nonfinite,
        chain2_output_nonfinite_count=chain2_nonfinite,
        operator_domain_sampling_rate_hz=(
            diagnostics.operator_domain_sampling_rate_hz
        ),
    )


def _random_corruption_chains(
    clean_raw: torch.Tensor,
    *,
    allowed_depths: tuple[int, ...],
    operator_params: dict[str, dict[str, Any]],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate three independent configured-depth chains in one batched launch."""

    batch = int(clean_raw.shape[0])
    width = 3
    stacked = torch.cat((clean_raw, clean_raw, clean_raw), dim=0).to(
        dtype=torch.float32
    )
    chain_count = batch * width
    depth_indices = torch.randint(
        0,
        len(allowed_depths),
        (chain_count,),
        device=clean_raw.device,
        dtype=torch.int64,
        generator=generator,
    )
    depth_values = torch.tensor(
        allowed_depths,
        device=clean_raw.device,
        dtype=torch.int64,
    )
    depths = depth_values.index_select(0, depth_indices)
    random_order = torch.rand(
        chain_count,
        len(CANONICAL_OPERATORS),
        device=clean_raw.device,
        dtype=torch.float32,
        generator=generator,
    ).argsort(dim=1)
    ranks = random_order.argsort(dim=1)
    operator_mask = ranks < depths.view(-1, 1)

    waveform = F.interpolate(
        stacked.transpose(1, 2),
        size=CORRUPTION_DOMAIN_POINTS,
        mode=INTERPOLATION_MODE,
        align_corners=INTERPOLATION_ALIGN_CORNERS,
    ).transpose(1, 2).contiguous()
    for operator_index, operator in enumerate(CANONICAL_OPERATORS):
        candidate = apply_operator_batch_prevalidated(
            operator,
            waveform,
            params=operator_params[operator],
            sampling_rate_hz=CORRUPTION_DOMAIN_SAMPLING_RATE_HZ,
            rng=generator,
        )
        selected = operator_mask[:, operator_index].view(-1, 1, 1)
        waveform = torch.where(selected, candidate, waveform)
    output = F.interpolate(
        waveform.transpose(1, 2),
        size=OUTPUT_POINTS,
        mode=INTERPOLATION_MODE,
        align_corners=INTERPOLATION_ALIGN_CORNERS,
    ).transpose(1, 2).contiguous()
    nonfinite = (~torch.isfinite(output)).sum(dim=(1, 2), dtype=torch.int64)
    output = torch.nan_to_num(
        output, nan=0.0, posinf=0.0, neginf=0.0
    ).contiguous()
    return output, depths, operator_mask, nonfinite


def generate_latent_three_chain_augmix(
    clean_raw: torch.Tensor,
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    sampling_rate_hz: int,
    config: AugMixConfig | None = None,
    generator: torch.Generator | None = None,
) -> LatentThreeChainAugMixBatch:
    """Corrupt three chains, mix deterministic VAE latents, then decode once."""

    resolved = load_augmix_config() if config is None else config
    clean = _validate_waveform(clean_raw, "clean_raw")
    if int(sampling_rate_hz) != resolved.input_sampling_rate_hz:
        raise ValueError("latent AugMix accepts only canonical raw 100 Hz input")
    if tuple(clean.shape[1:]) != (resolved.input_points, 12):
        raise ValueError("latent AugMix expects raw (B,1000,12) input")
    if not bool(torch.isfinite(clean).all().item()):
        raise ValueError("latent AugMix clean input must be finite raw mV")
    if not isinstance(encoder, torch.nn.Module) or not isinstance(
        decoder, torch.nn.Module
    ):
        raise TypeError("latent AugMix requires managed VAE encoder and decoder")
    if generator is None:
        raise ValueError("latent AugMix requires an explicit isolated generator")
    _validate_generator(generator, clean.device)

    operator_params = _load_operator_profile(resolved)
    chains, depths, operator_mask, nonfinite = _random_corruption_chains(
        clean,
        allowed_depths=resolved.latent_threechain_depths,
        operator_params=operator_params,
        generator=generator,
    )
    batch = int(clean.shape[0])
    with torch.no_grad():
        from models.vae import (
            decode_to_ptbxl_waveform,
            prepare_ecgtwin_encoder_input,
        )

        encoded = encoder(
            prepare_ecgtwin_encoder_input(chains),
            sample=False,
        )
        if not isinstance(encoded, tuple) or len(encoded) != 3:
            raise TypeError("VAE encoder must return (scaled_latent, mean, log_variance)")
        latent = encoded[0]
        if tuple(latent.shape) != (batch * 3, 4, 128):
            raise ValueError("VAE encoder returned an unexpected latent shape")
        latent_chains = latent.reshape(3, batch, 4, 128).transpose(0, 1)
        weights = _dirichlet_one(
            batch,
            resolved.latent_threechain_width,
            device=clean.device,
            generator=generator,
        )
        mixed_latent = (
            latent_chains
            * weights.to(dtype=latent_chains.dtype).view(batch, 3, 1, 1)
        ).sum(dim=1)
        if resolved.latent_threechain_reconstruction_residual_bypass:
            decoded = decode_to_ptbxl_waveform(
                decoder,
                torch.cat((mixed_latent, latent), dim=0),
                target_points=resolved.output_points,
            )
            decoded_mixture, decoded_chains = decoded.split(
                (batch, batch * 3), dim=0
            )
            decoded_chains = decoded_chains.reshape(
                3, batch, resolved.output_points, 12
            ).transpose(0, 1)
            raw_chains = chains.reshape(
                3, batch, resolved.output_points, 12
            ).transpose(0, 1)
            residual = raw_chains - decoded_chains
            residual_mixture = (
                residual
                * weights.to(dtype=residual.dtype).view(batch, 3, 1, 1)
            ).sum(dim=1)
            decoded_mixture = decoded_mixture + residual_mixture
            residual_rms = residual_mixture.square().mean(dim=(1, 2)).sqrt()
            decoded_rms = decoded_mixture.square().mean(dim=(1, 2)).sqrt()
            residual_rms_ratio = residual_rms / decoded_rms.clamp_min(1.0e-8)
        else:
            decoded_mixture = decode_to_ptbxl_waveform(
                decoder,
                mixed_latent,
                target_points=resolved.output_points,
            )
            residual_rms_ratio = torch.zeros(
                batch,
                device=clean.device,
                dtype=torch.float32,
            )
        if resolved.latent_threechain_post_decode_clean_beta_mix:
            # AugMix samples m ~ Beta(alpha, alpha) and returns
            # (1-m) * clean + m * augmented.  Alpha is locked to one in the
            # shared config, so an isolated uniform draw is exactly Beta(1,1)
            # without touching process-global RNG state.
            augmented_strength = torch.rand(
                batch,
                device=clean.device,
                dtype=torch.float32,
                generator=generator,
            )
            strength = augmented_strength.view(-1, 1, 1)
            mixed = (
                (1.0 - strength) * clean.to(dtype=torch.float32)
                + strength * decoded_mixture.to(dtype=torch.float32)
            )
        else:
            augmented_strength = torch.ones(
                batch,
                device=clean.device,
                dtype=torch.float32,
            )
            mixed = decoded_mixture
    return LatentThreeChainAugMixBatch(
        mixed_raw=mixed.to(dtype=torch.float32).contiguous(),
        mixture_weights=weights.contiguous(),
        augmented_strength=augmented_strength.contiguous(),
        residual_rms_ratio=residual_rms_ratio.to(dtype=torch.float32).contiguous(),
        chain_depths=depths.reshape(3, batch).transpose(0, 1).contiguous(),
        chain_operator_mask=operator_mask.reshape(
            3, batch, len(CANONICAL_OPERATORS)
        ).transpose(0, 1).contiguous(),
        chain_output_nonfinite_count=nonfinite.reshape(3, batch)
        .transpose(0, 1)
        .contiguous(),
        operator_domain_sampling_rate_hz=CORRUPTION_DOMAIN_SAMPLING_RATE_HZ,
    )


def multilabel_jsd(logits_views: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Jensen-Shannon divergence over per-class Bernoulli predictions."""

    if len(logits_views) < 2:
        raise ValueError("multilabel_jsd requires at least two views")
    first_shape = logits_views[0].shape
    if len(first_shape) != 2 or any(value.shape != first_shape for value in logits_views):
        raise ValueError("all JSD logits must share shape (B,classes)")
    # BCE/JSD is numerically sensitive under the default bf16 autocast.  Keep
    # model forwards low precision but evaluate probabilities and logarithms in
    # float32; the casts remain differentiable back to the original logits.
    float_views = tuple(value.float() for value in logits_views)
    epsilon = torch.finfo(torch.float32).eps
    probabilities = torch.stack(
        [torch.sigmoid(value).clamp(epsilon, 1.0 - epsilon) for value in float_views],
        dim=0,
    )
    mean = probabilities.mean(dim=0).clamp(epsilon, 1.0 - epsilon)
    divergence = probabilities * (probabilities / mean).log()
    divergence += (1.0 - probabilities) * (
        (1.0 - probabilities) / (1.0 - mean)
    ).log()
    return divergence.mean()


__all__ = [
    "AugMixBatch",
    "AugMixConfig",
    "LatentThreeChainAugMixBatch",
    "CANONICAL_OPERATORS",
    "DEFAULT_AUGMIX_CONFIG_PATH",
    "generate_three_chain_augmix",
    "generate_latent_three_chain_augmix",
    "load_augmix_config",
    "make_augmix_generator",
    "multilabel_jsd",
]
