"""Run-name helpers for legacy-compatible experiment entrypoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def tag_value(value: float | int | str) -> str:
    """Format a numeric CLI value for a legacy run-name token."""

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
    """Build the ECGFounder full-FT method tag used by the legacy runner."""

    vae_enabled = _as_bool(_get(params, "enable_vae_adv_stream", False))
    method_tag = "fullft_vae" if vae_enabled else "fullft"
    if _get(params, "init_head_path", ""):
        method_tag += "_inithead"
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
        hull_partner_pool = str(_get(params, "hull_partner_pool", "target"))
        if hull_partner_pool != "target":
            method_tag += f"_partners{hull_partner_pool}"
        source_partner_limit = _as_int(_get(params, "source_partner_limit_per_class", 0), 0)
        if source_partner_limit > 0:
            method_tag += f"_splim{source_partner_limit}"
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
    run_suffix = str(_get(params, "run_suffix", "") or "")
    if run_suffix:
        method_tag += f"_{run_suffix}"
    return method_tag


def build_ecgfounder_fullft_selection_tag(params: Any) -> str:
    """Build the optional target-validation selection tag."""

    target_val_count = _as_int(_get(params, "target_val_count", 0), 0)
    selection_metric = str(_get(params, "selection_metric", "source_auprc"))
    if target_val_count <= 0 and selection_metric == "source_auprc":
        return ""
    selection_tag = f"_tv{target_val_count}_{selection_metric}"
    split_mode = str(_get(params, "target_val_split_mode", "random"))
    if split_mode != "random":
        selection_tag += f"_{split_mode}"
    target_val_seed = _get(params, "target_val_seed", None)
    if target_val_seed is not None:
        selection_tag += f"_tvseed{target_val_seed}"
    return selection_tag


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
        f"{build_ecgfounder_fullft_selection_tag(params)}"
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

    center = str(_get(params, "center", ""))
    hull_m = _as_int(_get(params, "hull_M", _get(params, "hull_m", 20)), 20)
    hull_lambda = _get(params, "hull_lambda", 0.15)
    severity = _as_int(_get(params, "latent_augmix_severity", 2), 2)
    latent_cap = _get(params, "latent_augmix_latent_weight_cap", 0.3)
    hull_steps = _as_int(_get(params, "hull_steps", 3), 3)
    es_metric = str(_get(params, "es_metric", "target_macro_auprc"))
    classes = _as_classes(_get(params, "classes_in_scope", ["CD", "HYP", "MI", "NORM", "STTC"])) or []
    class_tag = "".join(str(cls).lower() for cls in classes)
    label_mode = str(_get(params, "hull_label_mode", "primary"))
    mix_label = str(_get(params, "hull_mix_label_mode", "anchor"))
    distance_space = str(_get(params, "hull_neighbor_distance_space", "raw"))
    neighbor_mode = str(_get(params, "hull_neighbor_mode", "nearest"))
    pool_size = _as_int(_get(params, "hull_neighbor_pool_size", 0), 0)
    pool_multiplier = _as_int(_get(params, "hull_neighbor_pool_multiplier", 4), 4)
    neighbor_tag = f"{distance_space[:3]}_{neighbor_mode}_p{pool_size or pool_multiplier}"

    if _as_int(_get(params, "unfreeze_last_n_features", 0), 0) > 0:
        adapt_tag = f"last{_as_int(_get(params, 'unfreeze_last_n_features', 0), 0)}"
    elif _as_bool(_get(params, "freeze_backbone_classifier_only", False)):
        adapt_tag = "headfn" if _as_bool(_get(params, "classifier_only_train_final_norm", False)) else "head"
        if str(_get(params, "classifier_adapter_type", "none")) == "lora":
            adapt_tag += f"_lora{_as_int(_get(params, 'classifier_lora_rank', 16), 16)}"
    else:
        adapt_tag = "fullft"

    extra = str(_get(params, "run_tag_extra", "") or "")
    extra_tag = f"_{extra}" if extra else ""
    epochs = _as_int(_get(params, "epochs", 30), 30)
    seed = _as_int(_get(params, "seed", 20260531), 20260531)
    return (
        f"{center}_realall_targetheavy_M{hull_m}"
        f"_lam{_tag_float_3g(hull_lambda)}"
        f"_augmix_s{severity}"
        f"_wlat{_tag_float_3g(latent_cap)}"
        f"_hs{hull_steps}_{es_metric}_{class_tag}"
        f"_hlabel{label_mode[:3]}_{mix_label}"
        f"_{neighbor_tag}"
        f"_{adapt_tag}"
        f"{extra_tag}_ep{epochs}_seed{seed}"
    )
