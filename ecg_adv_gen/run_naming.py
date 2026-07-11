"""Run-name helpers for managed experiment outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def tag_value(value: float | int | str) -> str:
    """Format a numeric CLI value for a run-name token."""

    return str(value).replace(".", "p").replace("-", "m")


def _tag_float_3g(value: Any) -> str:
    try:
        return f"{float(value):.3g}".replace("-", "m").replace(".", "p")
    except (TypeError, ValueError):
        return str(value).replace("-", "m").replace(".", "p")


def _get(params: Any, name: str, default: Any = None) -> Any:
    if isinstance(params, Mapping):
        return params.get(name, default)
    return getattr(params, name, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "off", "none", "null"}:
        return False
    return True


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return int(default)
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)
    return float(value)


def _as_classes(value: Any) -> Sequence[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value] if value else []
    return value


def build_ecgfounder_fullft_method_tag(params: Any) -> str:
    """Build the ECGFounder full-FT method tag used by the managed runner."""

    vae_enabled = _as_bool(_get(params, "enable_vae_adv_stream", False))
    method_tag = "fullft_vae" if vae_enabled else "fullft"
    classes = _as_classes(_get(params, "vae_classes_in_scope", None))
    if vae_enabled and classes is not None:
        method_tag += "_cls" + "-".join(str(c) for c in classes)
    min_count = _as_int(_get(params, "vae_min_class_count", 1), 1)
    if vae_enabled and min_count > 1:
        method_tag += f"_mincnt{min_count}"
    if vae_enabled:
        method_tag += (
            f"_aw{tag_value(_get(params, 'adv_weight', 20.0))}"
            f"_ka{_as_int(_get(params, 'k_anchor', 100), 100)}"
            f"_M{_as_int(_get(params, 'hull_m', 20), 20)}"
            f"_lam{tag_value(_get(params, 'hull_lambda', 0.15))}"
            f"_hs{_as_int(_get(params, 'hull_steps', 3), 3)}"
            f"_hlr{tag_value(_get(params, 'hull_lr', 0.25))}"
        )
        hull_weight_mode = str(_get(params, "hull_weight_mode", "optimized"))
        if hull_weight_mode != "optimized":
            method_tag += f"_{hull_weight_mode}"
        attack_pos_weight_source = str(_get(params, "hull_attack_pos_weight_source", "none"))
        if attack_pos_weight_source != "none":
            method_tag += (
                f"_apw{attack_pos_weight_source}"
                f"clip{tag_value(_get(params, 'hull_attack_pos_weight_clip', 50.0))}"
            )
        hull_label_mode = str(_get(params, "hull_label_mode", "primary"))
        if hull_label_mode != "primary":
            method_tag += f"_label{hull_label_mode}"
        if _as_bool(_get(params, "hull_include_anchor", False)):
            method_tag += "_includeanchor"
        anchor_sample_mode = str(_get(params, "anchor_sample_mode", "stratified"))
        if anchor_sample_mode != "stratified":
            method_tag += f"_as{anchor_sample_mode}"
        anchor_sample_power = _as_float(_get(params, "anchor_sample_power", 1.0), 1.0)
        if anchor_sample_power != 1.0:
            method_tag += f"_aspow{tag_value(_get(params, 'anchor_sample_power', 1.0))}"
        class_weights = str(_get(params, "anchor_class_sample_weights", "") or "")
        if class_weights:
            compact = class_weights.replace("=", "").replace(",", "-").replace(";", "-")
            method_tag += f"_acw{compact}"
        anchor_class_max_repeat = _as_int(_get(params, "anchor_class_max_repeat", 0), 0)
        if anchor_class_max_repeat > 0:
            method_tag += f"_acmaxrep{anchor_class_max_repeat}"
        adv_warmup_epochs = _as_int(_get(params, "adv_weight_warmup_epochs", 0), 0)
        if adv_warmup_epochs > 0:
            method_tag += f"_awarm{adv_warmup_epochs}"
        adv_bce_weight = _as_float(_get(params, "adv_bce_loss_weight", 1.0), 1.0)
        if adv_bce_weight != 1.0:
            method_tag += f"_abce{tag_value(_get(params, 'adv_bce_loss_weight', 1.0))}"
        clean_anchor_weight = _as_float(_get(params, "adv_clean_logit_anchor_weight", 0.0), 0.0)
        if clean_anchor_weight > 0:
            method_tag += f"_aclean{tag_value(_get(params, 'adv_clean_logit_anchor_weight', 0.0))}"
    return method_tag


def build_ecgfounder_fullft_run_leaf(params: Any) -> str:
    """Build the run directory leaf for ECGFounder full-FT scripts."""

    run_name = str(_get(params, "run_name", "") or "")
    if run_name:
        return run_name
    return (
        f"{_get(params, 'center', 'cpsc_2018')}_K{_as_int(_get(params, 'k', 100), 100)}"
        f"_fullft_ep{_as_int(_get(params, 'epochs', 5), 5)}"
        f"_lr{tag_value(_get(params, 'lr', 1e-4))}"
        f"_sw{tag_value(_get(params, 'source_weight', 1.0))}"
        f"_tw{tag_value(_get(params, 'target_real_weight', 40.0))}"
        f"_{build_ecgfounder_fullft_method_tag(params)}"
        f"_seed{_as_int(_get(params, 'seed', 20260531), 20260531)}"
    )


def build_effnet_direct_run_leaf(params: Any) -> str:
    """Build the run directory leaf for EfficientNet direct K-shot fine-tuning."""

    center = str(_get(params, "center", ""))
    k = _as_int(_get(params, "k", 500), 500)
    epochs = _as_int(_get(params, "epochs", 30), 30)
    seed = _as_int(_get(params, "seed", 20260531), 20260531)
    val_fraction = _as_float(_get(params, "val_fraction", 0.2), 0.2)
    return f"{center}_K{k}_direct_ft_ep{epochs}_seed{seed}_val{val_fraction:g}"


def build_benchmark_direct_run_leaf(params: Any) -> str:
    """Build the run directory leaf for benchmark-backbone direct fine-tuning."""

    center = str(_get(params, "center", ""))
    k = _as_int(_get(params, "k", 500), 500)
    model_name = str(_get(params, "model_name", "benchmark_resnet1d_wang"))
    model_slug = model_name.replace("/", "_").replace(" ", "_")
    epochs = _as_int(_get(params, "epochs", 30), 30)
    seed = _as_int(_get(params, "seed", 20260531), 20260531)
    val_fraction = tag_value(_get(params, "val_fraction", 0.2))
    return f"{center}_K{k}_direct_ft_{model_slug}_ep{epochs}_seed{seed}_val{val_fraction}"


def build_effnet_vae_lhat_run_leaf(params: Any) -> str:
    """Build the run directory leaf for EfficientNet VAE-LHAT/AugMix pilots."""

    from ecg_adv_gen.matched_effnet import F004_RHO_SWEEP_PROTOCOL, f004_variant_for_rho

    if str(_get(params, "comparison_protocol", "")) == F004_RHO_SWEEP_PROTOCOL:
        variant = f004_variant_for_rho(_get(params, "target_adv_fraction", None))
        declared = str(_get(params, "comparison_variant", variant) or variant)
        if declared != variant:
            raise ValueError(f"F-004 variant/rho mismatch: expected {variant!r}, got {declared!r}")
        return variant

    center = str(_get(params, "center", ""))
    hull_m = _as_int(_get(params, "hull_M", _get(params, "hull_m", 20)), 20)
    hull_lambda = _get(params, "hull_lambda", 0.15)
    severity = _as_int(_get(params, "latent_augmix_severity", 2), 2)
    hull_steps = _as_int(_get(params, "hull_steps", 3), 3)
    classes = _as_classes(_get(params, "classes_in_scope", ["CD", "HYP", "MI", "NORM", "STTC"])) or []
    class_tag = "".join(str(cls).lower() for cls in classes)
    label_mode = str(_get(params, "hull_label_mode", "primary"))
    mix_label = str(_get(params, "hull_mix_label_mode", "anchor"))
    distance_space = str(_get(params, "hull_neighbor_distance_space", "raw"))
    neighbor_mode = str(_get(params, "hull_neighbor_mode", "nearest"))
    pool_size = _as_int(_get(params, "hull_neighbor_pool_size", 0), 0)
    pool_multiplier = _as_int(_get(params, "hull_neighbor_pool_multiplier", 4), 4)
    neighbor_tag = f"{distance_space[:3]}_{neighbor_mode}_p{pool_size or pool_multiplier}"

    extra = str(_get(params, "run_tag_extra", "") or "")
    extra_tag = f"_{extra}" if extra else ""
    comparison_arm = str(_get(params, "comparison_arm", "historical_unmatched"))
    epochs = _as_int(_get(params, "epochs", 30), 30)
    seed = _as_int(_get(params, "seed", 20260531), 20260531)
    return (
        f"{center}_realall_targetheavy_M{hull_m}"
        f"_lam{_tag_float_3g(hull_lambda)}"
        f"_augmix_s{severity}"
        f"_hs{hull_steps}_{class_tag}"
        f"_hlabel{label_mode[:3]}_{mix_label}"
        f"_{neighbor_tag}"
        f"_fullft_arm{comparison_arm}"
        f"{extra_tag}_ep{epochs}_seed{seed}"
    )


def build_f005_control_run_leaf(
    center: str, seed: int, variant: str, *, epochs: int
) -> str:
    """Delegate the orthogonal F005 identity without expanding canonical arms."""
    from ecg_adv_gen.f005_control import build_f005_control_run_leaf as _build

    return _build(center, seed, variant, epochs=epochs)


def build_f005_control_producer_dir(
    config: Mapping[str, Any], *, center: str, seed: int, variant: str
) -> str:
    """Derive one exact F005 producer directory for all evaluation consumers."""
    producer = config["f005_producer"]
    leaf = build_f005_control_run_leaf(
        center,
        int(seed),
        variant,
        epochs=int(producer.get("epochs", config["training"]["epochs"])),
    )
    return (
        f"{config['paths']['output_root']}/{producer['experiment_name']}/"
        f"{config.get('runtime', {}).get('run_id', '')}/{leaf}"
    )


def build_matched_effnet_producer_dir(
    config: Mapping[str, Any], *, center: str, arm: str
) -> str:
    """Derive the canonical matched-arm producer directory from resolved config."""

    from ecg_adv_gen.matched_effnet import matched_effnet_arm

    matched_effnet_arm(arm)
    adaptation = config["adaptation"]
    hull = adaptation["hull"]
    latent_augmix = adaptation["latent_augmix"]
    training = config["training"]
    paper = config["paper_protocol"]
    producer = config["matched_effnet_producer"]
    leaf = build_effnet_vae_lhat_run_leaf(
        {
            "center": center,
            "comparison_arm": arm,
            "hull_M": hull["M"],
            "hull_lambda": hull["lambda"],
            "latent_augmix_severity": latent_augmix["severity"],
            "hull_steps": hull["steps"],
            "classes_in_scope": adaptation.get("classes_in_scope", ["CD", "HYP", "MI", "NORM", "STTC"]),
            "hull_label_mode": hull["label_mode"],
            "hull_mix_label_mode": hull["mix_label_mode"],
            "hull_neighbor_distance_space": hull["neighbor_distance_space"],
            "hull_neighbor_mode": hull["neighbor_mode"],
            "hull_neighbor_pool_size": hull["neighbor_pool_size"],
            "hull_neighbor_pool_multiplier": hull["neighbor_pool_multiplier"],
            "run_tag_extra": adaptation.get("run_tag_extra", ""),
            "epochs": training["epochs"],
            "seed": paper["kshot"]["seed"],
        }
    )
    return (
        f"{config['paths']['output_root']}/{producer['experiment_name']}/"
        f"{config.get('runtime', {}).get('run_id', '')}/{leaf}"
    )


def build_f004_effnet_producer_dir(
    config: Mapping[str, Any], *, center: str, rho: float
) -> str:
    """Derive the short F-004 producer directory shared by train and eval."""

    from ecg_adv_gen.matched_effnet import f004_variant_for_rho

    producer = config.get("f004_effnet_producer") or {}
    experiment_name = str(
        producer.get("experiment_name") or "effnet_f004_rho_sweep_k500"
    )
    variant = f004_variant_for_rho(rho)
    return (
        f"{config['paths']['output_root']}/{experiment_name}/"
        f"{config.get('runtime', {}).get('run_id', '')}/runs/{center}/{variant}"
    )
