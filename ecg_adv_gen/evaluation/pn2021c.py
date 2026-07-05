"""PN2021-C corruption profile and evaluation contract helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from methods.augmix.ecg_ops import (
    BaselineShift,
    BaselineWander,
    EMGNoise,
    PowerlineNoise,
    RandomLeadsMask,
)
from methods.augmix.severity import build_op

PN2021C_OFFICIAL_OPERATORS = (
    "powerline_noise",
    "emg_noise",
    "baseline_wander",
    "baseline_shift",
    "random_leads_masking",
)
PN2021C_DEFAULT_CENTERS = (
    "ningbo",
    "chapman_shaoxing",
    "cpsc_2018",
    "georgia",
)
PN2021C_DEFAULT_CORRUPTIONS = PN2021C_OFFICIAL_OPERATORS
PN2021C_CORRUPTS_PRE_ZSCORE = True
PN2021_C_CACHE_VERSION = "v7_refexcluded_100hz1000"
PUBLIC_TO_INTERNAL_SEVERITY = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
STRESS_PROFILE_CHOICES = ("standard", "stress_v2", "calibrated_10to20pp", "custom")

_STRESS_V2_PARAMS = {
    "powerline_noise": {
        1: {"max_amplitude": 0.15},
        2: {"max_amplitude": 0.30},
        3: {"max_amplitude": 0.60},
        4: {"max_amplitude": 0.90},
        5: {"max_amplitude": 1.20},
    },
    "emg_noise": {
        1: {"max_amplitude": 0.10},
        2: {"max_amplitude": 0.25},
        3: {"max_amplitude": 0.50},
        4: {"max_amplitude": 0.80},
        5: {"max_amplitude": 1.20},
    },
    "baseline_wander": {
        1: {"max_amplitude": 0.20, "k": 3},
        2: {"max_amplitude": 0.40, "k": 3},
        3: {"max_amplitude": 0.80, "k": 4},
        4: {"max_amplitude": 1.20, "k": 4},
        5: {"max_amplitude": 1.60, "k": 5},
    },
    "baseline_shift": {
        1: {"max_amplitude": 0.20, "shift_ratio": 0.10, "num_segment": 1},
        2: {"max_amplitude": 0.50, "shift_ratio": 0.20, "num_segment": 1},
        3: {"max_amplitude": 0.80, "shift_ratio": 0.35, "num_segment": 2},
        4: {"max_amplitude": 1.10, "shift_ratio": 0.55, "num_segment": 2},
        5: {"max_amplitude": 1.40, "shift_ratio": 0.75, "num_segment": 3},
    },
    "random_leads_masking": {
        1: {"mask_leads_prob": 0.15},
        2: {"mask_leads_prob": 0.30},
        3: {"mask_leads_prob": 0.50},
        4: {"mask_leads_prob": 0.70},
        5: {"mask_leads_prob": 0.85},
    },
}

_CALIBRATED_10TO20PP_PARAMS = {
    "powerline_noise": {
        5: {"max_amplitude": 8.0},
    },
    "emg_noise": {
        5: {"max_amplitude": 2.3},
    },
    "baseline_wander": {
        5: {"max_amplitude": 2.5, "k": 6, "max_freq": 0.8},
    },
    "baseline_shift": {
        5: {"max_amplitude": 2.4, "shift_ratio": 0.9, "num_segment": 6},
    },
    "random_leads_masking": {
        5: {"mask_leads_prob": 0.57},
    },
}


def _profile_params_for_op(profile_params, profile_name, corruption, public_severity):
    public_severity = int(public_severity)
    if corruption not in profile_params:
        raise ValueError(f"{profile_name} does not define corruption: {corruption}")
    if public_severity not in profile_params[corruption]:
        raise ValueError(
            f"{profile_name} defines {corruption} only for severities "
            f"{sorted(profile_params[corruption])}; got {public_severity}"
        )
    return dict(profile_params[corruption][public_severity])


def _with_operator_defaults(corruption, params):
    params = dict(params)
    params.setdefault("p", 1.0)
    if corruption == "powerline_noise":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("freq", 100)
        params.setdefault("dependency", False)
        return params
    if corruption == "emg_noise":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("dependency", False)
        return params
    if corruption == "baseline_wander":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("min_freq", 0.03)
        params.setdefault("freq", 100)
        params.setdefault("dependency", False)
        return params
    if corruption == "baseline_shift":
        params.setdefault("min_amplitude", 0.0)
        params.setdefault("freq", 100)
        params.setdefault("dependency", False)
        return params
    if corruption == "random_leads_masking":
        params.setdefault("mask_leads_selection", "random")
        return params
    raise ValueError(f"unknown corruption: {corruption}")


def _build_profile_op(profile_params, profile_name, corruption, public_severity):
    params = _with_operator_defaults(
        corruption,
        _profile_params_for_op(profile_params, profile_name, corruption, public_severity),
    )
    if corruption == "powerline_noise":
        return PowerlineNoise(**params)
    if corruption == "emg_noise":
        return EMGNoise(**params)
    if corruption == "baseline_wander":
        return BaselineWander(**params)
    if corruption == "baseline_shift":
        return BaselineShift(**params)
    if corruption == "random_leads_masking":
        return RandomLeadsMask(**params)
    raise ValueError(f"unknown corruption: {corruption}")


def canonicalize_profile(raw_profile, *, profile_name):
    if not isinstance(raw_profile, dict):
        raise ValueError(f"custom severity profile {profile_name!r} must be a mapping")
    profile = {}
    for corruption, severities in raw_profile.items():
        if not isinstance(severities, dict):
            raise ValueError(f"custom severity profile {profile_name!r} {corruption!r} must map severities")
        profile[str(corruption)] = {}
        for severity, params in severities.items():
            try:
                severity_int = int(severity)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"custom severity profile {profile_name!r} has non-integer severity "
                    f"{severity!r} for {corruption!r}"
                ) from exc
            if not isinstance(params, dict):
                raise ValueError(
                    f"custom severity profile {profile_name!r} {corruption!r} "
                    f"severity {severity_int} must map parameters"
                )
            profile[str(corruption)][severity_int] = dict(params)
    return profile


def load_profile_document(path):
    if not path:
        raise ValueError("--severity_params_file is required when --severity_profile custom")
    if str(path).endswith(".json"):
        with open(path) as f:
            return json.load(f)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read custom severity profile YAML files") from exc
    with open(path) as f:
        return yaml.safe_load(f)


def load_custom_severity_profile(path, profile_name):
    if not profile_name:
        raise ValueError("--severity_params_name is required when --severity_profile custom")
    document = load_profile_document(path)
    if not isinstance(document, dict) or "profiles" not in document:
        raise ValueError("custom severity profile file must contain a top-level 'profiles' mapping")
    profiles = document["profiles"]
    if not isinstance(profiles, dict):
        raise ValueError("custom severity profile file top-level 'profiles' must be a mapping")
    if profile_name not in profiles:
        raise ValueError(
            f"custom severity profile {profile_name!r} not found; available={sorted(profiles)}"
        )
    return canonicalize_profile(profiles[profile_name], profile_name=profile_name)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_severity_profile_args(args):
    params_file = getattr(args, "severity_params_file", None)
    params_name = getattr(args, "severity_params_name", None)
    if args.severity_profile != "custom":
        if params_file or params_name:
            raise ValueError(
                "--severity_params_file/--severity_params_name are only valid with "
                "--severity_profile custom"
            )
        return None
    if getattr(args, "mode", "stream") != "stream":
        raise ValueError("--severity_profile custom requires --mode stream")
    return load_custom_severity_profile(params_file, params_name)


def severity_profile_metadata(args):
    params = getattr(args, "severity_profile_params", None)
    if args.severity_profile != "custom":
        return None
    return {
        "severity_profile": "custom",
        "severity_params_file": str(args.severity_params_file),
        "severity_params_file_sha256": file_sha256(args.severity_params_file),
        "severity_params_name": str(args.severity_params_name),
        "resolved_params": params,
    }


def corruption_sequence(corruption):
    sequence = [part.strip() for part in str(corruption).split("+") if part.strip()]
    if not sequence:
        raise ValueError(f"empty corruption name: {corruption!r}")
    unknown = [part for part in sequence if part not in PN2021C_OFFICIAL_OPERATORS]
    if unknown:
        raise ValueError(
            f"unknown corruption(s) in {corruption!r}: {unknown}; "
            f"allowed={list(PN2021C_OFFICIAL_OPERATORS)}"
        )
    return sequence


def resolve_corruption_profile_params(corruption, public_severity, severity_profile, severity_profile_params=None):
    sequence = corruption_sequence(corruption)
    if len(sequence) > 1:
        return [
            {
                "name": name,
                "params": resolve_corruption_profile_params(
                    name,
                    public_severity,
                    severity_profile,
                    severity_profile_params,
                ),
            }
            for name in sequence
        ]
    if severity_profile == "standard":
        return None
    corruption = sequence[0]
    if severity_profile == "stress_v2":
        params = _profile_params_for_op(_STRESS_V2_PARAMS, "stress_v2", corruption, public_severity)
        return _with_operator_defaults(corruption, params)
    if severity_profile == "calibrated_10to20pp":
        params = _profile_params_for_op(
            _CALIBRATED_10TO20PP_PARAMS,
            "calibrated_10to20pp",
            corruption,
            public_severity,
        )
        return _with_operator_defaults(corruption, params)
    if severity_profile == "custom":
        if severity_profile_params is None:
            raise ValueError("severity_profile_params is required for --severity_profile custom")
        params = _profile_params_for_op(
            severity_profile_params,
            "custom",
            corruption,
            public_severity,
        )
        return _with_operator_defaults(corruption, params)
    raise ValueError(f"unknown severity_profile: {severity_profile}")


def _with_native_sample_rate(op, sample_rate_hz):
    if sample_rate_hz is not None and hasattr(op, "freq"):
        op.freq = float(sample_rate_hz)
    return op


def build_corruption_op(
    corruption,
    public_severity,
    severity_profile,
    *,
    sample_rate_hz=None,
    severity_profile_params=None,
):
    sequence = corruption_sequence(corruption)
    if len(sequence) != 1:
        raise ValueError(
            "build_corruption_op only accepts one operator; "
            "use apply_corruption_sequence for composite corruptions"
        )
    corruption = sequence[0]
    if severity_profile == "standard":
        op = build_op(corruption, PUBLIC_TO_INTERNAL_SEVERITY[int(public_severity)])
        return _with_native_sample_rate(op, sample_rate_hz)
    if severity_profile == "stress_v2":
        op = _build_profile_op(_STRESS_V2_PARAMS, "stress_v2", corruption, public_severity)
        return _with_native_sample_rate(op, sample_rate_hz)
    if severity_profile == "calibrated_10to20pp":
        op = _build_profile_op(_CALIBRATED_10TO20PP_PARAMS, "calibrated_10to20pp", corruption, public_severity)
        return _with_native_sample_rate(op, sample_rate_hz)
    if severity_profile == "custom":
        if severity_profile_params is None:
            raise ValueError("severity_profile_params is required for --severity_profile custom")
        op = _build_profile_op(severity_profile_params, "custom", corruption, public_severity)
        return _with_native_sample_rate(op, sample_rate_hz)
    raise ValueError(f"unknown severity_profile: {severity_profile}")


def stable_seed(base_seed, *parts) -> int:
    payload = "|".join(str(p) for p in (base_seed,) + parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:4], "little")


def seed_corruption_rng(base_seed, *parts):
    import torch

    sample_seed = stable_seed(base_seed, *parts)
    np.random.seed(sample_seed)
    random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    return sample_seed


def apply_corruption_sequence(
    ecg_ct,
    corruption,
    public_severity,
    severity_profile,
    *,
    base_seed,
    seed_parts,
    sample_rate_hz=None,
    severity_profile_params=None,
):
    sequence = corruption_sequence(corruption)
    out = ecg_ct
    for pos, op_name in enumerate(sequence):
        if len(sequence) == 1:
            seed_corruption_rng(base_seed, *seed_parts)
        else:
            seed_corruption_rng(base_seed, *seed_parts, pos, op_name)
        op = build_corruption_op(
            op_name,
            public_severity,
            severity_profile,
            sample_rate_hz=sample_rate_hz,
            severity_profile_params=severity_profile_params,
        )
        out = op(out)
    return out
