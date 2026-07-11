from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from ecg_adv_gen.f004_contract import (
    F004_FROZEN_PROJECTION,
    F004_FROZEN_PROJECTION_SHA256,
    validate_f004_projection,
)

MATCHED_EFFNET_CONTRACT_VERSION = "matched_effnet_a0_a2_a3_a4_a5_v5"
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
# Backward-compatible names.  The package-owned projection is the sole source
# of truth; no second topology mapping is maintained here.
F004_FROZEN_TOPOLOGY = F004_FROZEN_PROJECTION
F004_FROZEN_TOPOLOGY_SHA256 = F004_FROZEN_PROJECTION_SHA256

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
    kshot_seed: int, kshot_path: str, behavior_projection: Mapping[str, Any],
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
    validate_f004_projection(behavior_projection, source="runtime")
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
