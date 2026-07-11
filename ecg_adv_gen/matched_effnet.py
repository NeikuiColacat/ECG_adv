from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple

MATCHED_EFFNET_CONTRACT_VERSION = "matched_effnet_a0_a2_a3_a4_a5_v3"
MATCHED_EFFNET_ARMS = ("a0", "a2", "a3", "a4", "a5")
MATCHED_EFFNET_VAE_HULL_LABEL_MODE = "exact"
MATCHED_EFFNET_VAE_HULL_INCLUDE_ANCHOR = False
MATCHED_EFFNET_NON_VAE_HULL_LABEL_MODE = "compatible"
F004_RHO_SWEEP_PROTOCOL = "f004_full_topology_rho_sweep_v1"
F004_RHO_VALUES = (0.0, 0.25, 0.5)
F004_TOPOLOGY_REFERENCE = "a5_components_only"
F004_ONLY_VARIED_PARAMETER = "target_adv_fraction"
F004_VARIANTS = ("f004_rho0", "f004_rho0p25", "f004_rho0p5")
F004_KSHOT_SEED = 20260601
F004_KSHOT_SUBSET_ROOT_FAMILY = (
    "paper_matched_effnet_k500_v7_fixedk_three_seed_20260711/subsets"
)
F004_FROZEN_TOPOLOGY = MappingProxyType({
    "enable_vae_lhat": True,
    "enable_raw_augmix": True,
    "enable_latent_augmix_consistency": True,
    "latent_augmix_bce_weight": 1.0,
    "latent_augmix_consistency_weight": 2.0,
    "latent_augmix_consistency_loss": "jsd",
    "latent_augmix_third_chain_role": "vae_lhat_adversarial_waveform",
    "latent_augmix_width": 3,
    "latent_augmix_depth": -1,
    "latent_augmix_copies": 2,
    "latent_augmix_chain_base_mode": "clean_clean_third",
    "latent_augmix_adv_base_mix": 1.0,
    "latent_augmix_alpha": 1.0,
    "latent_augmix_severity": 5,
    "latent_augmix_severity_profile": "standard",
    "latent_augmix_ops": (
        "powerline_noise", "emg_noise", "baseline_wander",
        "baseline_shift", "random_leads_masking",
    ),
    "latent_augmix_signal_space": "raw_pre_zscore",
    "hull_M": 20,
    "hull_lambda": 0.6,
    "hull_steps": 5,
    "hull_include_anchor": False,
    "hull_init_logit_gap": 0.0,
    "hull_label_mode": "exact",
    "hull_mix_label_mode": "anchor_soft",
    "hull_lr": 0.25,
    "hull_neighbor_distance_space": "standardized",
    "hull_neighbor_mode": "local_random",
    "hull_neighbor_pool_size": 120,
    "hull_neighbor_pool_multiplier": 4,
    "pgd_eps": 2.0,
})
F004_FROZEN_TOPOLOGY_SHA256 = hashlib.sha256(
    json.dumps(dict(F004_FROZEN_TOPOLOGY), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

class MatchedEffnetArm(NamedTuple):
    role: str; vae_lhat: bool; raw_augmix: bool; augmix_view_bce: bool
    jsd: bool; target_adv_fraction: float; third_chain_route: str

MATCHED_EFFNET_ARM_COMPONENTS = {
    "a0": MatchedEffnetArm("matched_direct_k500_baseline", False, False, False, False, 0.0, "clean_budget_control"),
    "a2": MatchedEffnetArm("raw_augmix_clean_third_control", False, True, True, True, 0.0, "clean_anchor_control"),
    "a3": MatchedEffnetArm("vae_lhat_only", True, False, False, False, 0.5, "no_augmix_route"),
    "a4": MatchedEffnetArm("vae_lhat_threechain_no_jsd", True, True, True, False, 0.5, "vae_lhat_adversarial_waveform"),
    "a5": MatchedEffnetArm("vae_lhat_threechain_full", True, True, True, True, 0.5, "vae_lhat_adversarial_waveform"),
}
MATCHED_EFFNET_THIRD_CHAIN_ROUTES = tuple(dict.fromkeys(row.third_chain_route for row in MATCHED_EFFNET_ARM_COMPONENTS.values()))

def matched_effnet_arm(arm: str) -> MatchedEffnetArm:
    try:
        return MATCHED_EFFNET_ARM_COMPONENTS[str(arm)]
    except KeyError:
        raise ValueError(f"invalid matched EffNet arm: {arm!r}") from None

def is_matched_effnet_arm(arm: object) -> bool:
    return str(arm) in MATCHED_EFFNET_ARM_COMPONENTS


def is_f004_rho_sweep(protocol: object) -> bool:
    return str(protocol or "") == F004_RHO_SWEEP_PROTOCOL


def is_paper_matched_effnet_run(arm: object, protocol: object = "") -> bool:
    return is_matched_effnet_arm(arm) or is_f004_rho_sweep(protocol)


def f004_variant_for_rho(rho: object) -> str:
    if isinstance(rho, bool):
        raise ValueError("F-004 rho must be one of [0.0, 0.25, 0.5]")
    try:
        value = float(rho)
    except (TypeError, ValueError):
        raise ValueError("F-004 rho must be one of [0.0, 0.25, 0.5]") from None
    try:
        return dict(zip(F004_RHO_VALUES, F004_VARIANTS))[value]
    except KeyError:
        raise ValueError("F-004 rho must be one of [0.0, 0.25, 0.5]") from None


def f004_identity(rho: object) -> dict[str, Any]:
    value = float(rho)
    return {
        "comparison_protocol": F004_RHO_SWEEP_PROTOCOL,
        "variant": f004_variant_for_rho(value),
        "rho": value,
        "topology_reference": F004_TOPOLOGY_REFERENCE,
        "topology_version": MATCHED_EFFNET_CONTRACT_VERSION,
        "topology_sha256": F004_FROZEN_TOPOLOGY_SHA256,
        "only_varied_parameter": F004_ONLY_VARIED_PARAMETER,
        "kshot_seed": F004_KSHOT_SEED,
        "kshot_subset_root_family": F004_KSHOT_SUBSET_ROOT_FAMILY,
        "canonical_arm": None,
    }


def validate_f004_kshot_identity(*, kshot_seed: int, kshot_path: str) -> None:
    normalized_kshot_path = str(kshot_path).replace("\\", "/")
    normalized_kshot_path = f"/{normalized_kshot_path.strip('/')}/"
    expected_root_token = f"/{F004_KSHOT_SUBSET_ROOT_FAMILY}/"
    if int(kshot_seed) != F004_KSHOT_SEED or expected_root_token not in normalized_kshot_path:
        raise ValueError(
            "F-004 K500 identity mismatch: "
            f"seed={kshot_seed!r}, path={kshot_path!r}, expected seed "
            f"{F004_KSHOT_SEED} under {F004_KSHOT_SUBSET_ROOT_FAMILY!r}"
        )

def validate_matched_effnet_case(case: Mapping[str, Any]) -> tuple[str, MatchedEffnetArm]:
    arm = str(case.get("arm") or case.get("comparison_arm") or "")
    row = matched_effnet_arm(arm)
    fields = ("role", "vae_lhat", "raw_augmix", "augmix_view_bce", "jsd", "target_adv_fraction", "third_chain_route")
    for key, field in zip((*fields[:5], "rho", fields[-1]), fields):
        value = getattr(row, field)
        if key not in case or case[key] != value:
            raise ValueError(f"canonical matched EffNet arm {arm} requires {key}={value!r}, got {case.get(key)!r}")
    return arm, row

def validate_matched_effnet_runtime(
    arm: str, *, enable_vae_lhat: bool, enable_raw_augmix: bool,
    enable_auxiliary_steps: bool, bce_weight: float, jsd_weight: float,
    third_chain_route: str, hull_label_mode: str, hull_include_anchor: bool,
    target_adv_fraction: float | None = None,
) -> MatchedEffnetArm:
    row = matched_effnet_arm(arm)
    checks = {
        "enable_vae_lhat": (bool(enable_vae_lhat), row.vae_lhat),
        "enable_raw_augmix": (bool(enable_raw_augmix), row.raw_augmix),
        "enable_latent_augmix_consistency": (bool(enable_auxiliary_steps), True),
        "latent_augmix_bce_weight>0": (float(bce_weight) > 0.0, row.augmix_view_bce),
        "latent_augmix_consistency_weight>0": (float(jsd_weight) > 0.0, row.jsd),
        "latent_augmix_third_chain_role": (str(third_chain_route), row.third_chain_route),
    }
    if target_adv_fraction is not None:
        checks["target_adv_fraction"] = (float(target_adv_fraction), row.target_adv_fraction)
    if row.vae_lhat:
        checks.update({
            "hull_label_mode": (
                str(hull_label_mode), MATCHED_EFFNET_VAE_HULL_LABEL_MODE
            ),
            "hull_include_anchor": (
                bool(hull_include_anchor), MATCHED_EFFNET_VAE_HULL_INCLUDE_ANCHOR
            ),
        })
    drift = {key: values for key, values in checks.items() if values[0] != values[1]}
    if drift:
        raise ValueError(f"canonical matched EffNet arm {arm} component drift: {drift}")
    return row


def validate_f004_runtime(
    *, comparison_protocol: str, comparison_arm: str, comparison_variant: str,
    comparison_topology_sha256: str, target_adv_fraction: float,
    kshot_seed: int, kshot_path: str,
    enable_vae_lhat: bool, enable_raw_augmix: bool,
    enable_latent_augmix_consistency: bool,
    latent_augmix_bce_weight: float, latent_augmix_consistency_weight: float,
    latent_augmix_consistency_loss: str, latent_augmix_third_chain_role: str,
    latent_augmix_width: int, latent_augmix_depth: int, latent_augmix_copies: int,
    latent_augmix_chain_base_mode: str, latent_augmix_adv_base_mix: float,
    latent_augmix_alpha: float, latent_augmix_severity: int,
    latent_augmix_severity_profile: str, latent_augmix_ops: list[str] | tuple[str, ...],
    latent_augmix_signal_space: str, hull_M: int, hull_lambda: float,
    hull_steps: int, hull_include_anchor: bool, hull_init_logit_gap: float,
    hull_label_mode: str, hull_mix_label_mode: str, hull_lr: float,
    hull_neighbor_distance_space: str, hull_neighbor_mode: str,
    hull_neighbor_pool_size: int, hull_neighbor_pool_multiplier: int,
    pgd_eps: float,
) -> dict[str, Any]:
    if not is_f004_rho_sweep(comparison_protocol):
        raise ValueError(f"invalid F-004 comparison protocol: {comparison_protocol!r}")
    if comparison_arm != "historical_unmatched":
        raise ValueError("F-004 must not claim a canonical comparison arm")
    if comparison_topology_sha256 != F004_FROZEN_TOPOLOGY_SHA256:
        raise ValueError(
            "F-004 frozen full topology fingerprint mismatch: "
            f"{comparison_topology_sha256!r}"
        )
    validate_f004_kshot_identity(kshot_seed=kshot_seed, kshot_path=kshot_path)
    identity = f004_identity(target_adv_fraction)
    if comparison_variant != identity["variant"]:
        raise ValueError(
            f"F-004 variant/rho mismatch: expected {identity['variant']!r}, "
            f"got {comparison_variant!r}"
        )
    observed = {
        "enable_vae_lhat": bool(enable_vae_lhat),
        "enable_raw_augmix": bool(enable_raw_augmix),
        "enable_latent_augmix_consistency": bool(enable_latent_augmix_consistency),
        "latent_augmix_bce_weight": float(latent_augmix_bce_weight),
        "latent_augmix_consistency_weight": float(latent_augmix_consistency_weight),
        "latent_augmix_consistency_loss": str(latent_augmix_consistency_loss),
        "latent_augmix_third_chain_role": str(latent_augmix_third_chain_role),
        "latent_augmix_width": int(latent_augmix_width),
        "latent_augmix_depth": int(latent_augmix_depth),
        "latent_augmix_copies": int(latent_augmix_copies),
        "latent_augmix_chain_base_mode": str(latent_augmix_chain_base_mode),
        "latent_augmix_adv_base_mix": float(latent_augmix_adv_base_mix),
        "latent_augmix_alpha": float(latent_augmix_alpha),
        "latent_augmix_severity": int(latent_augmix_severity),
        "latent_augmix_severity_profile": str(latent_augmix_severity_profile),
        "latent_augmix_ops": tuple(str(item) for item in latent_augmix_ops),
        "latent_augmix_signal_space": str(latent_augmix_signal_space),
        "hull_M": int(hull_M),
        "hull_lambda": float(hull_lambda),
        "hull_steps": int(hull_steps),
        "hull_include_anchor": bool(hull_include_anchor),
        "hull_init_logit_gap": float(hull_init_logit_gap),
        "hull_label_mode": str(hull_label_mode),
        "hull_mix_label_mode": str(hull_mix_label_mode),
        "hull_lr": float(hull_lr),
        "hull_neighbor_distance_space": str(hull_neighbor_distance_space),
        "hull_neighbor_mode": str(hull_neighbor_mode),
        "hull_neighbor_pool_size": int(hull_neighbor_pool_size),
        "hull_neighbor_pool_multiplier": int(hull_neighbor_pool_multiplier),
        "pgd_eps": float(pgd_eps),
    }
    drift = {
        key: {"observed": observed[key], "expected": expected}
        for key, expected in F004_FROZEN_TOPOLOGY.items()
        if observed[key] != expected
    }
    if drift:
        raise ValueError(f"F-004 frozen full topology drift: {drift}")
    return identity


def validate_f004_protocol_declaration(paper_protocol: Mapping[str, Any]) -> None:
    """Reject managed YAML declarations that disagree with the frozen study."""

    expected = {
        "topology_reference": F004_TOPOLOGY_REFERENCE,
        "topology_version": MATCHED_EFFNET_CONTRACT_VERSION,
        "topology_sha256": F004_FROZEN_TOPOLOGY_SHA256,
        "only_varied_parameter": F004_ONLY_VARIED_PARAMETER,
    }
    for field, value in expected.items():
        observed = paper_protocol.get(field)
        if observed != value:
            raise ValueError(
                f"F-004 paper_protocol.{field}={observed!r}, expected {value!r}"
            )
