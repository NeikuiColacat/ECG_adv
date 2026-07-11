"""Normalized F004 child-training behavior contract.

The projection is intentionally expressed in child argparse destination names.
It is the sole input to the F004 topology digest, managed-config validation,
wrapper-to-child validation, and the non-bypassable resume gate.
"""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence


F004_EXEMPT_CHILD_OPTIONS = frozenset({
    "allow_resume_config_drift",
    "center_name",
    "class_trust",
    "comparison_topology_sha256",
    "comparison_topology_version",
    "comparison_variant",
    "device",
    "init_ckpt",
    "num_workers",
    "output_dir",
    "ptbxl_csv",
    "ptbxl_prep",
    "ptbxl_raw",
    "ref_meta_json",
    "resume",
    "synth_npz",
    "target_adv_fraction",
    "target_real_npz",
})

F004_FROZEN_PROJECTION = MappingProxyType({
    "K_anchor": 300,
    "adv_label_mode": "latent_mixed_teacher",
    "adv_teacher_mix": 0.4,
    "adv_weight": 0.3,
    "adv_weight_warmup_epochs": 10,
    "anchor_lambda": 0.05,
    "asr_consec_low_max": 999,
    "asr_high_threshold": 0.7,
    "asr_low_threshold": 0.3,
    "batch_size": 128,
    "classes_in_scope": ("CD", "HYP", "MI", "NORM", "STTC"),
    "comparison_arm": "historical_unmatched",
    "comparison_protocol": "f004_full_topology_rho_sweep_v1",
    "crop_len": 1000,
    "disable_quality_gate": True,
    "enable_latent_augmix_consistency": True,
    "enable_raw_augmix": True,
    "enable_vae_lhat": True,
    "eval_every": 2,
    "ewa_decay": 0.999,
    "final_checkpoint_only": False,
    "hull_M": 20,
    "hull_include_anchor": False,
    "hull_init_logit_gap": 0.0,
    "hull_label_lambda_y": 0.25,
    "hull_label_mode": "exact",
    "hull_label_negative_floor": 0.0,
    "hull_label_new_class_cap": 0.25,
    "hull_label_positive": 0.95,
    "hull_lambda": 0.6,
    "hull_lr": 0.25,
    "hull_mix_label_mode": "anchor_soft",
    "hull_neighbor_distance_space": "standardized",
    "hull_neighbor_mode": "local_random",
    "hull_neighbor_pool_multiplier": 4,
    "hull_neighbor_pool_size": 120,
    "hull_steps": 5,
    "init_checkpoint_sha256": "f7a4b05d85da8352013b67a0d56d8d47378ab2abc196d1f3a6ba8bbf15aa40ea",
    "init_lineage_stage": "ptbxl_source",
    "latent_augmix_adv_base_mix": 1.0,
    "latent_augmix_alpha": 1.0,
    "latent_augmix_bce_weight": 1.0,
    "latent_augmix_chain_base_mode": "clean_clean_third",
    "latent_augmix_chain_weights": "",
    "latent_augmix_consistency_loss": "jsd",
    "latent_augmix_consistency_max_batches": 0,
    "latent_augmix_consistency_weight": 2.0,
    "latent_augmix_copies": 2,
    "latent_augmix_depth": -1,
    "latent_augmix_ops": (
        "powerline_noise",
        "emg_noise",
        "baseline_wander",
        "baseline_shift",
        "random_leads_masking",
    ),
    "latent_augmix_severity": 5,
    "latent_augmix_severity_profile": "standard",
    "latent_augmix_signal_space": "raw_pre_zscore",
    "latent_augmix_third_chain_role": "vae_lhat_adversarial_waveform",
    "latent_augmix_width": 3,
    "lr": 5e-5,
    "model_name": "efficientnet1dv2",
    "n_epochs": 30,
    "pgd_batch": 32,
    "pgd_eps": 2.0,
    "ptbxl_weight": 0.0,
    "qab_size": 2048,
    "rescore_interval": 3,
    "seed": 20260601,
    "selection_metric": "macro_auprc",
    "source_floor_max_drop": 0.02,
    "target_real_norm_mode": "per_sample_global",
    "target_real_val_fraction": 0.2,
    "target_real_val_seed": 20260601,
    "target_real_weight": 80.0,
    "vae_adv_consistency_weight": 0.0,
    "vae_adv_stream_sample_scale": 1.0,
    "weight_decay": 1e-4,
})


def _projection_digest(projection: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        dict(projection), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(rendered).hexdigest()


F004_FROZEN_PROJECTION_SHA256 = _projection_digest(F004_FROZEN_PROJECTION)


def _coerce(field: str, value: Any) -> Any:
    expected = F004_FROZEN_PROJECTION[field]
    if isinstance(expected, bool):
        return bool(value)
    if isinstance(expected, int):
        return int(value)
    if isinstance(expected, float):
        return float(value)
    if isinstance(expected, tuple):
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)
    return str(value)


def _argv_options(argv: Sequence[str]) -> dict[str, tuple[str, ...]]:
    options: dict[str, tuple[str, ...]] = {}
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if not token.startswith("--"):
            index += 1
            continue
        field = token[2:]
        if field in options:
            raise ValueError(f"F-004 child argv repeats --{field}")
        index += 1
        values: list[str] = []
        while index < len(argv) and not str(argv[index]).startswith("--"):
            values.append(str(argv[index]))
            index += 1
        options[field] = tuple(values)
    return options


def project_f004_child_argv(argv: Sequence[str]) -> dict[str, Any]:
    options = _argv_options(argv)
    unknown = sorted(
        set(options) - set(F004_FROZEN_PROJECTION) - F004_EXEMPT_CHILD_OPTIONS
    )
    if unknown:
        raise ValueError(f"F-004 child argv contains unclassified options: {unknown}")
    projection: dict[str, Any] = {}
    for field, expected in F004_FROZEN_PROJECTION.items():
        if field == "latent_augmix_signal_space":
            projection[field] = (
                "raw_pre_zscore" if "enable_raw_augmix" in options else "model_zscore"
            )
            continue
        values = options.get(field)
        if values is None:
            if isinstance(expected, bool):
                projection[field] = False
                continue
            raise ValueError(f"F-004 child argv is missing frozen option --{field}")
        if isinstance(expected, bool):
            if values:
                raise ValueError(f"F-004 flag --{field} must not take a value")
            projection[field] = True
        elif isinstance(expected, tuple):
            projection[field] = _coerce(field, values)
        else:
            if len(values) != 1:
                raise ValueError(
                    f"F-004 option --{field} requires one value, got {values!r}"
                )
            projection[field] = _coerce(field, values[0])
    return projection


def project_f004_runtime_args(args: Mapping[str, Any]) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    missing: list[str] = []
    for field in F004_FROZEN_PROJECTION:
        if field not in args:
            missing.append(field)
            continue
        projection[field] = _coerce(field, args[field])
    if missing:
        raise ValueError(f"F-004 runtime projection missing fields: {missing}")
    return projection


def project_f004_wrapper_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """Project the managed wrapper namespace into child behavior names."""

    aliases = {
        "K_anchor": "k_anchor",
        "batch_size": "train_batch_size",
        "n_epochs": "epochs",
    }
    projection = dict(F004_FROZEN_PROJECTION)
    for field in F004_FROZEN_PROJECTION:
        source = aliases.get(field, field)
        if source in args:
            projection[field] = _coerce(field, args[source])
    projection["latent_augmix_signal_space"] = (
        "raw_pre_zscore" if bool(args.get("enable_raw_augmix")) else "model_zscore"
    )
    return projection


def project_f004_adapter_config(
    config: Mapping[str, Any],
    *,
    comparison_arm: str,
    comparison_protocol: str,
    seed: int,
) -> dict[str, Any]:
    paper = config["paper_protocol"]
    selection = paper["selection"]
    training = config["training"]
    optimizer = training["optimizer"]
    model = config["model"]
    adaptation = config["adaptation"]
    anchors = adaptation["anchors"]
    hull = adaptation["hull"]
    attack = adaptation["attack"]
    loss = adaptation["loss"]
    augmix = adaptation["latent_augmix"]
    consistency = augmix["consistency"]
    data = config["data"]
    observed = dict(F004_FROZEN_PROJECTION)
    observed.update({
        "K_anchor": anchors["k_anchor"],
        "adv_label_mode": loss["label_mode"],
        "adv_teacher_mix": loss["teacher_mix"],
        "adv_weight": loss["adv_weight"],
        "adv_weight_warmup_epochs": loss["adv_weight_warmup_epochs"],
        "asr_high_threshold": attack["target_asr_range"][1],
        "asr_low_threshold": attack["target_asr_range"][0],
        "batch_size": training["batch_size"],
        "classes_in_scope": adaptation.get(
            "classes_in_scope", ("CD", "HYP", "MI", "NORM", "STTC")
        ),
        "comparison_arm": comparison_arm,
        "comparison_protocol": comparison_protocol,
        "final_checkpoint_only": selection["policy"] == "last_checkpoint_only",
        "hull_M": hull["M"],
        "hull_include_anchor": hull["include_anchor"],
        "hull_init_logit_gap": hull["init_logit_gap"],
        "hull_label_lambda_y": hull["label_lambda_y"],
        "hull_label_mode": hull["label_mode"],
        "hull_label_new_class_cap": hull["label_new_class_cap"],
        "hull_lambda": hull["lambda"],
        "hull_lr": hull["lr"],
        "hull_mix_label_mode": hull["mix_label_mode"],
        "hull_neighbor_distance_space": hull["neighbor_distance_space"],
        "hull_neighbor_mode": hull["neighbor_mode"],
        "hull_neighbor_pool_multiplier": hull["neighbor_pool_multiplier"],
        "hull_neighbor_pool_size": hull["neighbor_pool_size"],
        "hull_steps": hull["steps"],
        "init_checkpoint_sha256": (model.get("init_lineage") or {}).get(
            "checkpoint_sha256", ""
        ),
        "init_lineage_stage": (model.get("init_lineage") or {}).get("stage", ""),
        "latent_augmix_adv_base_mix": augmix.get("adv_base_mix", 1.0),
        "latent_augmix_alpha": augmix["alpha"],
        "latent_augmix_bce_weight": consistency["bce_weight"],
        "latent_augmix_chain_base_mode": augmix["chain_base_mode"],
        "latent_augmix_chain_weights": augmix.get("chain_weights") or "",
        "latent_augmix_consistency_loss": consistency["consistency_loss"],
        "latent_augmix_consistency_max_batches": consistency["max_batches"],
        "latent_augmix_consistency_weight": consistency["consistency_weight"],
        "latent_augmix_copies": augmix["copies"],
        "latent_augmix_depth": augmix["depth"],
        "latent_augmix_ops": augmix["ops"],
        "latent_augmix_severity": augmix["severity"],
        "latent_augmix_severity_profile": augmix["severity_profile"],
        "latent_augmix_third_chain_role": augmix.get(
            "third_chain_role", "vae_lhat_adversarial_waveform"
        ),
        "latent_augmix_width": augmix["width"],
        "lr": optimizer["lr"],
        "model_name": model["backbone"],
        "n_epochs": training["epochs"],
        "pgd_batch": attack["pgd_batch"],
        "pgd_eps": attack["pgd_eps"],
        "ptbxl_weight": loss.get("ptbxl_weight", 1.0),
        "seed": seed,
        "selection_metric": selection.get("metric", "macro_auprc"),
        "source_floor_max_drop": (selection.get("source_floor") or {}).get(
            "max_drop", 0.02
        ),
        "target_real_norm_mode": data["target_real_norm_mode"],
        "target_real_val_fraction": selection.get("validation_fraction", 0.2),
        "target_real_val_seed": selection.get("seed", seed),
        "target_real_weight": loss["target_real_weight"],
        "vae_adv_consistency_weight": loss.get("vae_adv_consistency_weight", 0.0),
        "vae_adv_stream_sample_scale": loss.get("vae_adv_stream_sample_scale", 1.0),
        "weight_decay": optimizer["weight_decay"],
    })
    return {
        field: _coerce(field, observed[field])
        for field in F004_FROZEN_PROJECTION
    }


def validate_f004_projection(
    observed: Mapping[str, Any], *, source: str
) -> None:
    missing = sorted(set(F004_FROZEN_PROJECTION) - set(observed))
    extra = sorted(set(observed) - set(F004_FROZEN_PROJECTION))
    normalized = {
        field: _coerce(field, observed[field])
        for field in F004_FROZEN_PROJECTION
        if field in observed
    }
    drift = {
        field: {"observed": normalized[field], "expected": expected}
        for field, expected in F004_FROZEN_PROJECTION.items()
        if field in normalized and normalized[field] != expected
    }
    if missing or extra or drift:
        raise ValueError(
            f"F-004 frozen projection mismatch at {source}: "
            f"missing={missing}, extra={extra}, drift={drift}"
        )


def validate_f004_child_argv(argv: Sequence[str], *, source: str) -> None:
    validate_f004_projection(project_f004_child_argv(argv), source=source)
