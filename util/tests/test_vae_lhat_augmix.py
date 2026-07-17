from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from core.augmix import (
    generate_three_chain_augmix,
    load_augmix_config,
    multilabel_jsd,
)
from core.lhat import (
    LatentStandardizer,
    generate_lhat_adversarial,
    load_lhat_config,
    select_exact_label_candidates,
)
from models.vae import (
    ECGTWIN_TO_PTBXL_INDICES,
    LATENT_SCALE,
    build_ecgtwin_vae,
    load_vae_config,
    prepare_ecgtwin_encoder_input,
)
from util.random_seed import make_torch_generator


ROOT = Path(__file__).resolve().parents[2]


def test_vae_config_and_real_checkpoint_return_strict_encoder_decoder() -> None:
    config = load_vae_config()
    assert config.input_shape == (1024, 12)
    assert config.latent_shape == (4, 128)
    assert config.latent_scale == pytest.approx(0.18215)
    assert config.freeze is True
    assert ECGTWIN_TO_PTBXL_INDICES == (0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11)
    assert LATENT_SCALE == pytest.approx(0.18215)

    checkpoint = ROOT / "model" / "ECGTwin" / "checkpoints" / "vae_model.pth"
    if not checkpoint.is_file():
        pytest.skip(f"external ECGTwin VAE checkpoint unavailable: {checkpoint}")
    encoder, decoder = build_ecgtwin_vae(checkpoint_path=checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert list(encoder.state_dict()) == list(payload["encoder"])
    assert list(decoder.state_dict()) == list(payload["decoder"])
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    assert encoder.training is False
    assert decoder.training is False

    with torch.inference_mode():
        latent, mean, log_variance = encoder(
            torch.zeros(1, 1024, 12), sample=False
        )
        reconstructed = decoder(latent)
    assert latent.shape == mean.shape == log_variance.shape == (1, 4, 128)
    assert reconstructed.shape == (1, 1024, 12)
    assert torch.isfinite(reconstructed).all()

    lead_constants = torch.arange(12, dtype=torch.float32).view(1, 1, 12)
    bridged = prepare_ecgtwin_encoder_input(lead_constants.expand(1, 1000, 12))
    assert bridged.shape == (1, 1024, 12)
    torch.testing.assert_close(
        bridged[0, 0],
        torch.tensor([0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11], dtype=torch.float32),
    )


def test_latent_standardizer_and_exact_nonself_candidate_contract() -> None:
    latents = torch.arange(8 * 2 * 2, dtype=torch.float32).reshape(8, 2, 2)
    labels = torch.tensor(
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    standardizer = LatentStandardizer.fit(latents)
    standardized = standardizer.transform(latents)
    torch.testing.assert_close(
        standardizer.inverse_transform(standardized), latents
    )
    assert standardizer.identity_sha256

    generator = make_torch_generator("cpu", "test_lhat_candidates")
    indices = select_exact_label_candidates(
        standardized,
        labels,
        anchor_indices=torch.arange(8),
        num_candidates=2,
        mode="nearest",
        local_pool_size=3,
        generator=generator,
    )
    assert indices.shape == (8, 2)
    for anchor, candidates in enumerate(indices.tolist()):
        assert anchor not in candidates
        assert len(set(candidates)) == 2
        for candidate in candidates:
            torch.testing.assert_close(labels[anchor], labels[candidate])

    with pytest.raises(ValueError, match="distinct non-self candidates"):
        select_exact_label_candidates(
            standardized[:3],
            labels[:3],
            anchor_indices=torch.arange(3),
            num_candidates=3,
            mode="nearest",
            local_pool_size=3,
            generator=generator,
        )


class _TinyDecoder(nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        level = latent.flatten(1).mean(dim=1).view(-1, 1, 1)
        time = torch.linspace(-1.0, 1.0, 1024, device=latent.device).view(1, -1, 1)
        leads = torch.linspace(0.5, 1.5, 12, device=latent.device).view(1, 1, -1)
        return level * leads + time * leads


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_points: list[int] = []

    def forward(self, signal_bct: torch.Tensor) -> torch.Tensor:
        self.seen_points.append(int(signal_bct.shape[-1]))
        features = signal_bct[:, :, : signal_bct.shape[-1] // 2].mean(dim=(1, 2))
        return features.unsqueeze(1).repeat(1, 5)


@pytest.mark.parametrize(
    ("model_name", "expected_classifier_points"),
    (("efficientnet1dv2", 1000), ("ecgfounder", 5000)),
)
def test_lhat_attack_returns_raw_chain3_and_non_null_diagnostics(
    model_name: str,
    expected_classifier_points: int,
) -> None:
    config = load_lhat_config()
    assert config.num_candidates == 20
    assert config.include_anchor is False
    assert config.init_logit_gap == 0.0
    assert config.hull_lambda == pytest.approx(0.60)
    assert config.steps == 5
    assert config.random_namespace == "ecg_manual_refactor_lhat_v1"

    anchor = torch.zeros(2, 1, 2)
    candidates = torch.stack(
        [
            torch.linspace(-2.0, 2.0, 20).view(20, 1, 1).expand(20, 1, 2),
            torch.linspace(-1.0, 3.0, 20).view(20, 1, 1).expand(20, 1, 2),
        ]
    )
    fit_pool = torch.cat((anchor, candidates.flatten(0, 1)), dim=0)
    standardizer = LatentStandardizer.fit(fit_pool)
    anchor_std = standardizer.transform(anchor)
    candidate_std = standardizer.transform(candidates)
    labels = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.float32)

    classifier = _TinyClassifier()
    result = generate_lhat_adversarial(
        classifier=classifier,
        decoder=_TinyDecoder(),
        anchor_standardized=anchor_std,
        candidates_standardized=candidate_std,
        targets=labels,
        standardizer=standardizer,
        model_name=model_name,
        config=config,
    )
    assert result.waveform_raw.shape == (2, 1000, 12)
    assert classifier.seen_points
    assert set(classifier.seen_points) == {expected_classifier_points}
    assert result.latent_standardized.shape == anchor.shape
    assert result.weights.shape == (2, 20)
    torch.testing.assert_close(result.weights.sum(dim=1), torch.ones(2))
    assert torch.isfinite(result.waveform_raw).all()
    assert torch.all(result.diagnostics.projection_scale <= 1.0)
    assert torch.all(result.diagnostics.effective_anchor_share >= 0.0)
    assert result.diagnostics.loss_gain.shape == (2,)
    assert result.diagnostics.mean_dict()["decoded_invalid_rate"] == 0.0


def test_three_chain_augmix_is_deterministic_and_keeps_chain3_unmodified() -> None:
    sampling_rate_hz = 100
    points = 1000
    config = load_augmix_config()
    assert config.width == 3
    assert config.depths == (2, 3)
    assert (
        config.input_sampling_rate_hz,
        config.operator_domain_sampling_rate_hz,
        config.output_sampling_rate_hz,
    ) == (100, 500, 100)
    assert (
        config.input_points,
        config.operator_domain_points,
        config.output_points,
    ) == (1000, 5000, 1000)
    assert config.interpolation_mode == "linear"
    assert config.interpolation_align_corners is True
    assert config.chain_roles == (
        "paper_anchored_s5_corruption_chain",
        "paper_anchored_s5_corruption_chain",
        "vae_lhat_adversarial_waveform",
    )
    clean = torch.linspace(-0.5, 0.5, 3 * points * 12).reshape(3, points, 12)
    adversarial = clean + 0.1
    clean_before = clean.clone()
    adversarial_before = adversarial.clone()

    first = generate_three_chain_augmix(
        clean,
        adversarial,
        sampling_rate_hz=sampling_rate_hz,
        config=config,
        generator=make_torch_generator("cpu", "test_augmix_same"),
    )
    second = generate_three_chain_augmix(
        clean,
        adversarial,
        sampling_rate_hz=sampling_rate_hz,
        config=config,
        generator=make_torch_generator("cpu", "test_augmix_same"),
    )
    raw_only = generate_three_chain_augmix(
        clean,
        adversarial,
        sampling_rate_hz=sampling_rate_hz,
        config=config,
        generator=make_torch_generator("cpu", "test_augmix_same"),
        include_normalized=False,
    )
    assert first.mixed_raw.shape == first.mixed_normalized.shape == clean.shape
    assert raw_only.mixed_normalized is None
    torch.testing.assert_close(raw_only.mixed_raw, first.mixed_raw)
    torch.testing.assert_close(first.chain3_raw, adversarial)
    torch.testing.assert_close(first.mixed_raw, second.mixed_raw)
    torch.testing.assert_close(first.mixture_weights, second.mixture_weights)
    torch.testing.assert_close(first.mixture_weights.sum(dim=1), torch.ones(3))
    assert set(first.chain1_depth.tolist()).issubset({2, 3})
    assert set(first.chain2_depth.tolist()).issubset({2, 3})
    assert first.operator_domain_sampling_rate_hz == 500
    assert first.chain1_operator_mask.shape == (3, 5)
    assert first.chain2_operator_mask.shape == (3, 5)
    assert torch.count_nonzero(first.chain1_output_nonfinite_count) == 0
    assert torch.count_nonzero(first.chain2_output_nonfinite_count) == 0
    torch.testing.assert_close(clean, clean_before)
    torch.testing.assert_close(adversarial, adversarial_before)

    expected = (
        (1.0 - first.augmented_strength.view(-1, 1, 1)) * clean
        + first.augmented_strength.view(-1, 1, 1)
        * (
            first.mixture_weights[:, 0].view(-1, 1, 1) * first.chain1_raw
            + first.mixture_weights[:, 1].view(-1, 1, 1) * first.chain2_raw
            + first.mixture_weights[:, 2].view(-1, 1, 1) * adversarial
        )
    )
    torch.testing.assert_close(first.mixed_raw, expected)


def test_three_chain_augmix_rejects_noncanonical_500hz_input() -> None:
    clean = torch.zeros(2, 5000, 12)
    with pytest.raises(ValueError, match="only canonical raw 100 Hz"):
        generate_three_chain_augmix(
            clean,
            clean,
            sampling_rate_hz=500,
            config=load_augmix_config(),
            generator=make_torch_generator("cpu", "test_augmix_500_rejected"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_augmix_accepts_current_cuda_alias_for_explicit_generator_device() -> None:
    config = load_augmix_config()
    device = torch.device("cuda")
    clean = torch.linspace(
        -0.5,
        0.5,
        1000 * 12,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1000, 12)
    result = generate_three_chain_augmix(
        clean,
        clean,
        sampling_rate_hz=100,
        config=config,
        generator=make_torch_generator(device, "test_augmix_cuda_alias"),
    )
    assert result.mixed_raw.device.type == "cuda"
    assert torch.isfinite(result.mixed_raw).all()


def test_multilabel_jsd_is_zero_for_identical_logits() -> None:
    logits = torch.randn(4, 5)
    loss = multilabel_jsd((logits, logits.clone(), logits.clone()))
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)


def test_multilabel_jsd_computes_low_precision_logits_in_float32() -> None:
    first = torch.tensor(
        [[-8.0, -2.0, 0.0, 2.0, 8.0]], dtype=torch.bfloat16, requires_grad=True
    )
    second = (-first.detach()).requires_grad_(True)

    loss = multilabel_jsd((first, second))

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


def test_new_online_modules_only_import_manual_whitelist_surfaces() -> None:
    allowed_roots = {
        "__future__",
        "collections",
        "contextlib",
        "dataclasses",
        "hashlib",
        "functools",
        "itertools",
        "json",
        "math",
        "pathlib",
        "time",
        "typing",
        "torch",
        "yaml",
        "models",
        "util",
        "core",
    }
    for relative in (
        "models/vae.py",
        "core/corruption.py",
        "core/lhat.py",
        "core/augmix.py",
        "core/online_trainer.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        assert roots <= allowed_roots, (relative, sorted(roots - allowed_roots))
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "ecg_adv_gen" not in source
        assert "methods.augmix" not in source
        assert "from adversarial" not in source
        assert "import adversarial" not in source
        assert "model.ECGTwin" not in source
