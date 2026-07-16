from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import warnings

import pytest
import torch

from models import (
    CLASS_ORDER,
    available_models,
    build_ecgfounder,
    build_efficientnet1dv2,
    build_model,
    get_model_spec,
    validate_model_input,
    validate_model_output,
)


def test_model_specs_lock_super5_input_and_logit_contracts() -> None:
    assert CLASS_ORDER == ("CD", "HYP", "MI", "NORM", "STTC")
    assert available_models() == ("ecgfounder", "efficientnet1dv2")
    efficientnet = get_model_spec("efficientnet1dv2")
    ecgfounder = get_model_spec("ecgfounder")
    assert efficientnet.input_shape == (12, 1000)
    assert efficientnet.sampling_rate_hz == 100
    assert ecgfounder.input_shape == (12, 5000)
    assert ecgfounder.sampling_rate_hz == 500

    x = validate_model_input(torch.zeros(2, 12, 1000), efficientnet)
    assert x.shape == (2, 12, 1000)
    logits = validate_model_output(torch.zeros(2, 5), efficientnet, batch_size=2)
    assert logits.shape == (2, 5)
    with pytest.raises(ValueError, match="input shape"):
        validate_model_input(torch.zeros(2, 1000, 12), efficientnet)
    with pytest.raises(ValueError, match="raw logits"):
        validate_model_output(torch.zeros(2, 4), efficientnet, batch_size=2)


def test_efficientnet_builder_returns_five_raw_logits() -> None:
    torch.manual_seed(7)
    model = build_efficientnet1dv2()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 12, 1000))
    assert logits.shape == (1, 5)
    assert isinstance(model.classifier[-1], torch.nn.Linear)
    assert model.classifier[-1].out_features == 5


def test_ecgfounder_builder_uses_official_architecture_and_trainable_scope() -> None:
    model = build_ecgfounder(trainable_scope="head")
    assert model.dense.in_features == 1024
    assert model.dense.out_features == 5
    assert model.first_conv.conv.in_channels == 12
    assert all(
        parameter.requires_grad == name.startswith("dense.")
        for name, parameter in model.named_parameters()
    )
    del model
    gc.collect()


def test_factory_exposes_only_the_two_manual_models() -> None:
    model = build_model("efficientnet1dv2")
    assert model.model_spec.name == "efficientnet1dv2"
    del model
    gc.collect()
    with pytest.raises(ValueError, match="unknown model"):
        build_model("legacy_model")


def _load_reference_module(name: str, path: Path):
    if not path.is_file():
        pytest.skip(f"external reference model is unavailable: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(module)
    return module


def test_manual_models_match_external_reference_state_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    efficientnet_reference = _load_reference_module(
        "test_efficientnet_reference",
        root / "model" / "DeepECG" / "notebooks" / "EfficientNetv2.py",
    )
    manual_effnet = build_efficientnet1dv2()
    reference_effnet = efficientnet_reference.EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=5,
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    )
    assert list(manual_effnet.state_dict()) == list(reference_effnet.state_dict())
    reference_effnet.load_state_dict(manual_effnet.state_dict(), strict=True)
    manual_effnet.eval()
    reference_effnet.eval()
    waveform = torch.randn(1, 12, 1000)
    with torch.inference_mode():
        torch.testing.assert_close(
            manual_effnet(waveform), reference_effnet(waveform), rtol=0, atol=0
        )
    del manual_effnet, reference_effnet
    gc.collect()

    ecgfounder_reference = _load_reference_module(
        "test_ecgfounder_reference",
        root / "model" / "ecgfounder" / "net1d.py",
    )
    manual_ecgfounder = build_ecgfounder()
    reference_ecgfounder = ecgfounder_reference.Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=5,
        use_bn=False,
        use_do=False,
        return_features=False,
        verbose=False,
    )
    manual_state = manual_ecgfounder.state_dict()
    reference_state = reference_ecgfounder.state_dict()
    assert list(manual_state) == list(reference_state)
    assert all(
        manual_state[key].shape == reference_state[key].shape
        for key in manual_state
    )
