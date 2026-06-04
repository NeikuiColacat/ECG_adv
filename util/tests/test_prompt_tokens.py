"""Tests for CPU-only ECGTwin prompt-token compilation helpers."""

from __future__ import annotations

import pytest
import torch

from ecg_adv_gen.generation.prompt_tokens import (
    PromptTokenError,
    compile_prompt_text_embed,
    init_token_sequence_from_bank,
    load_text_embed_from_prompt_bank,
    scale_token_sequence,
    token_sequence_from_bank,
)


def test_direct_bank_single_vector_returns_2d_sequence():
    embeddings = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    token_blob = {"token_mode": "direct", "embeddings": embeddings}

    seq = token_sequence_from_bank(token_blob, center_idx=1, class_idx=2)

    assert seq.shape == (1, 4)
    torch.testing.assert_close(seq, embeddings[1, 2].unsqueeze(0))


def test_direct_bank_multi_vector_returns_selected_sequence():
    embeddings = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    token_blob = {"token_mode": "direct", "embeddings": embeddings}

    seq = token_sequence_from_bank(token_blob, center_idx=0, class_idx=1)

    assert seq.shape == (2, 4)
    torch.testing.assert_close(seq, embeddings[0, 1])


def test_factorized_bank_concatenates_center_class_and_residual_vectors():
    center_embeddings = torch.ones(2, 2, 4)
    class_embeddings = torch.full((3, 1, 4), 2.0)
    residual_embeddings = torch.full((2, 3, 2, 4), 3.0)
    token_blob = {
        "token_mode": "factorized",
        "center_embeddings": center_embeddings,
        "class_embeddings": class_embeddings,
        "residual_embeddings": residual_embeddings,
    }

    seq = token_sequence_from_bank(token_blob, center_idx=1, class_idx=2)

    assert seq.shape == (5, 4)
    torch.testing.assert_close(seq[:2], torch.ones(2, 4))
    torch.testing.assert_close(seq[2:3], torch.full((1, 4), 2.0))
    torch.testing.assert_close(seq[3:], torch.full((2, 4), 3.0))


def test_scale_token_sequence_preserves_initialization_delta_semantics():
    learned = torch.full((1, 2, 2, 4), 10.0)
    init = torch.full((1, 2, 2, 4), 2.0)
    token_blob = {
        "token_mode": "direct",
        "embeddings": learned,
        "init_embeddings": init,
    }
    token_seq = token_sequence_from_bank(token_blob, center_idx=0, class_idx=1)

    scaled = scale_token_sequence(token_blob, token_seq, center_idx=0, class_idx=1, scale=0.25)

    torch.testing.assert_close(scaled, torch.full((2, 4), 4.0))
    torch.testing.assert_close(
        init_token_sequence_from_bank(token_blob, center_idx=0, class_idx=1),
        torch.full((2, 4), 2.0),
    )


def test_scale_token_sequence_without_initialization_scales_absolute_token():
    token_blob = {
        "token_mode": "direct",
        "embeddings": torch.full((1, 1, 4), 8.0),
    }
    token_seq = token_sequence_from_bank(token_blob, center_idx=0, class_idx=0)

    scaled = scale_token_sequence(token_blob, token_seq, center_idx=0, class_idx=0, scale=0.5)

    torch.testing.assert_close(scaled, torch.full((1, 4), 4.0))


def test_compile_prompt_text_embed_appends_repeated_scaled_token_and_ones_mask():
    base_text = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
    token_blob = {
        "token_mode": "direct",
        "embeddings": torch.full((1, 1, 2, 4), 10.0),
        "init_embeddings": torch.full((1, 1, 2, 4), 2.0),
    }

    compiled = compile_prompt_text_embed(
        base_text,
        token_blob,
        center_idx=0,
        class_idx=0,
        repeat=2,
        scale=0.5,
    )

    assert compiled.text_embed.shape == (6, 4)
    assert compiled.text_embed_mask.shape == (1, 6)
    torch.testing.assert_close(compiled.text_embed[:2], base_text)
    torch.testing.assert_close(compiled.text_embed[2:], torch.full((4, 4), 6.0))
    torch.testing.assert_close(compiled.text_embed_mask, torch.ones(1, 6))
    assert compiled.token_sequence is not None
    torch.testing.assert_close(compiled.token_sequence, torch.full((4, 4), 6.0))


def test_compile_prompt_text_embed_no_token_returns_base_text_with_nonzero_mask():
    base_text = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)

    compiled = compile_prompt_text_embed(base_text, token_blob=None, no_token=True)

    torch.testing.assert_close(compiled.text_embed, base_text.float())
    torch.testing.assert_close(compiled.text_embed_mask, torch.ones(1, 3))
    assert compiled.token_sequence is None


def test_load_text_embed_from_prompt_bank_prefers_snomed_then_falls_back_to_class():
    snomed_embed = torch.ones(2, 4, dtype=torch.float64, requires_grad=True)
    fallback_embed = torch.full((3, 4), 2.0)
    prompt_bank = {
        "by_snomed": {"123": snomed_embed},
        "by_class": {"MI": fallback_embed},
    }

    from_snomed = load_text_embed_from_prompt_bank(prompt_bank, primary_snomed=123, primary_class="MI")
    from_class = load_text_embed_from_prompt_bank(prompt_bank, primary_snomed=456, primary_class="MI")

    assert from_snomed.dtype == torch.float32
    assert not from_snomed.requires_grad
    torch.testing.assert_close(from_snomed, torch.ones(2, 4))
    torch.testing.assert_close(from_class, fallback_embed.float())


@pytest.mark.parametrize("repeat, scale", [(0, 1.0), (1, -0.1)])
def test_compile_prompt_text_embed_rejects_invalid_repeat_or_scale(repeat: int, scale: float):
    base_text = torch.zeros(1, 4)

    with pytest.raises(PromptTokenError):
        compile_prompt_text_embed(
            base_text,
            {"token_mode": "direct", "embeddings": torch.zeros(1, 1, 4)},
            center_idx=0,
            class_idx=0,
            repeat=repeat,
            scale=scale,
        )
