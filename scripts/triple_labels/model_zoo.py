"""Small model zoo for Super5 PTB-XL / PN2021 experiments.

The default project model is EfficientNet1DV2.  For architecture robustness
checks we also reuse lightweight 1D models from ``model/ecg_ptbxl_benchmarking``.
All builders accept inputs shaped ``(B, 12, T)`` and return raw logits.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPECG_NOTEBOOKS = Path("/root/autodl-tmp/models/DeepECG/notebooks")
PTBXL_BENCH_CODE = PROJECT_ROOT / "model" / "ecg_ptbxl_benchmarking" / "code"

for _p in (PROJECT_ROOT, DEEPECG_NOTEBOOKS, PTBXL_BENCH_CODE):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _efficientnet1dv2(num_classes: int, input_channels: int = 12) -> nn.Module:
    from EfficientNetv2 import EfficientNet1DV2

    return EfficientNet1DV2(
        variant="s_v2",
        input_channels=input_channels,
        num_classes=num_classes,
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    )


def _benchmark_resnet1d_wang(num_classes: int, input_channels: int = 12) -> nn.Module:
    from models.resnet1d import resnet1d_wang

    return resnet1d_wang(
        num_classes=num_classes,
        input_channels=input_channels,
        kernel_size=5,
        ps_head=0.5,
        lin_ftrs_head=None,
    )


def _benchmark_xresnet1d18(num_classes: int, input_channels: int = 12) -> nn.Module:
    from models.xresnet1d import xresnet1d18

    return xresnet1d18(
        num_classes=num_classes,
        input_channels=input_channels,
        kernel_size=5,
        ps_head=0.5,
        lin_ftrs_head=None,
    )


def _benchmark_xresnet1d101(num_classes: int, input_channels: int = 12) -> nn.Module:
    from models.xresnet1d import xresnet1d101

    return xresnet1d101(
        num_classes=num_classes,
        input_channels=input_channels,
        kernel_size=5,
        ps_head=0.5,
        lin_ftrs_head=None,
    )


def _benchmark_inception1d(num_classes: int, input_channels: int = 12) -> nn.Module:
    from models.inception1d import inception1d

    return inception1d(
        num_classes=num_classes,
        input_channels=input_channels,
        use_residual=True,
        ps_head=0.5,
        lin_ftrs_head=None,
        kernel_size=40,
    )


def _benchmark_fcn_wang(num_classes: int, input_channels: int = 12) -> nn.Module:
    from models.basic_conv1d import fcn_wang

    return fcn_wang(
        num_classes=num_classes,
        input_channels=input_channels,
        ps_head=0.5,
        lin_ftrs_head=None,
    )


def _benchmark_schirrmeister(num_classes: int, input_channels: int = 12) -> nn.Module:
    from models.basic_conv1d import schirrmeister

    return schirrmeister(
        num_classes=num_classes,
        input_channels=input_channels,
        ps_head=0.5,
        lin_ftrs_head=None,
    )


def _benchmark_lstm(num_classes: int, input_channels: int = 12) -> nn.Module:
    from models.rnn1d import RNN1d

    return RNN1d(
        input_channels=input_channels,
        num_classes=num_classes,
        lstm=True,
        bidirectional=False,
        ps_head=0.5,
        lin_ftrs_head=None,
    )


MODEL_BUILDERS: dict[str, Callable[[int, int], nn.Module]] = {
    "efficientnet1dv2": _efficientnet1dv2,
    "benchmark_resnet1d_wang": _benchmark_resnet1d_wang,
    "benchmark_xresnet1d18": _benchmark_xresnet1d18,
    "benchmark_xresnet1d101": _benchmark_xresnet1d101,
    "benchmark_inception1d": _benchmark_inception1d,
    "benchmark_fcn_wang": _benchmark_fcn_wang,
    "benchmark_schirrmeister": _benchmark_schirrmeister,
    "benchmark_lstm": _benchmark_lstm,
}

ALIASES = {
    "resnet1d_wang": "benchmark_resnet1d_wang",
    "xresnet1d18": "benchmark_xresnet1d18",
    "xresnet1d101": "benchmark_xresnet1d101",
    "inception1d": "benchmark_inception1d",
    "fcn_wang": "benchmark_fcn_wang",
    "schirrmeister": "benchmark_schirrmeister",
    "lstm": "benchmark_lstm",
}


def normalize_model_name(model_name: str) -> str:
    name = str(model_name).strip()
    return ALIASES.get(name, name)


def available_model_names() -> list[str]:
    return sorted(MODEL_BUILDERS)


def build_super5_model(
    model_name: str,
    num_classes: int,
    input_channels: int = 12,
) -> nn.Module:
    name = normalize_model_name(model_name)
    if name not in MODEL_BUILDERS:
        valid = ", ".join(available_model_names())
        raise ValueError(f"unknown model_name={model_name!r}; valid: {valid}")
    return MODEL_BUILDERS[name](int(num_classes), int(input_channels))
