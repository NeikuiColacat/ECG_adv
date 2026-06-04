"""ECGTwin prompt-token bank decoding and text-embedding compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


class PromptTokenError(ValueError):
    """Raised when a prompt-token bank or compile request violates the contract."""


@dataclass(frozen=True)
class CompiledPromptTextEmbed:
    """Compiled ECGTwin target text embedding and its non-empty attention mask."""

    text_embed: torch.Tensor
    text_embed_mask: torch.Tensor
    token_sequence: torch.Tensor | None


def _as_float_sequence(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise PromptTokenError(f"{name} must be a torch.Tensor")
    seq = value.float()
    if seq.dim() == 1:
        seq = seq.unsqueeze(0)
    if seq.dim() != 2:
        raise PromptTokenError(f"{name} must be a 1D or 2D token sequence, got {tuple(seq.shape)}")
    return seq


def load_text_embed_from_prompt_bank(
    prompt_bank: Mapping[str, object],
    primary_snomed: object,
    primary_class: str,
) -> torch.Tensor:
    """Load diagnosis text embeddings by SNOMED id with class fallback."""

    by_snomed = prompt_bank.get("by_snomed")
    by_class = prompt_bank.get("by_class")
    if not isinstance(by_snomed, Mapping) or not isinstance(by_class, Mapping):
        raise PromptTokenError("prompt_bank must contain mapping fields 'by_snomed' and 'by_class'")

    candidates: list[object] = []
    if primary_snomed is not None:
        candidates.append(primary_snomed)
        try:
            candidates.append(int(primary_snomed))
        except Exception:
            pass
        candidates.append(str(primary_snomed))
    for key in candidates:
        if key in by_snomed:
            value = by_snomed[key]
            if not torch.is_tensor(value):
                raise PromptTokenError(f"prompt_bank by_snomed[{key!r}] must be a torch.Tensor")
            return _as_float_sequence(value.detach(), name=f"prompt_bank by_snomed[{key!r}]")

    if primary_class not in by_class:
        raise PromptTokenError(f"primary_class {primary_class!r} not found in prompt_bank by_class")
    value = by_class[primary_class]
    if not torch.is_tensor(value):
        raise PromptTokenError(f"prompt_bank by_class[{primary_class!r}] must be a torch.Tensor")
    return _as_float_sequence(value.detach(), name=f"prompt_bank by_class[{primary_class!r}]")


def _token_mode(token_blob: Mapping[str, object]) -> str:
    return str(token_blob.get("token_mode", "direct"))


def _has_factorized_fields(token_blob: Mapping[str, object]) -> bool:
    return _token_mode(token_blob) == "factorized" or "center_embeddings" in token_blob


def token_sequence_from_bank(
    token_blob: Mapping[str, object],
    center_idx: int,
    class_idx: int,
) -> torch.Tensor:
    """Return a `(M,D)` prompt-token sequence from direct or factorized banks."""

    if _has_factorized_fields(token_blob):
        try:
            center = _as_float_sequence(token_blob["center_embeddings"][center_idx], name="center token")
            cls = _as_float_sequence(token_blob["class_embeddings"][class_idx], name="class token")
        except KeyError as exc:
            raise PromptTokenError(f"factorized token bank missing field {exc.args[0]!r}") from exc
        pieces = [center, cls]
        residual = token_blob.get("residual_embeddings")
        if torch.is_tensor(residual) and residual.shape[2] > 0:
            pieces.append(
                _as_float_sequence(residual[center_idx, class_idx], name="residual token")
            )
        return torch.cat(pieces, dim=0)

    try:
        embeddings = token_blob["embeddings"]
    except KeyError as exc:
        raise PromptTokenError("direct token bank missing field 'embeddings'") from exc
    if not torch.is_tensor(embeddings):
        raise PromptTokenError("embeddings must be a torch.Tensor")
    emb = embeddings.float()
    if emb.dim() == 3:
        return emb[center_idx, class_idx].unsqueeze(0)
    if emb.dim() == 4:
        return emb[center_idx, class_idx]
    raise PromptTokenError(f"unsupported token embedding shape: {tuple(emb.shape)}")


def init_token_sequence_from_bank(
    token_blob: Mapping[str, object],
    center_idx: int,
    class_idx: int,
) -> torch.Tensor | None:
    """Return a stored initialization sequence, or ``None`` for legacy banks."""

    if _has_factorized_fields(token_blob):
        if "init_center_embeddings" not in token_blob or "init_class_embeddings" not in token_blob:
            return None
        center = _as_float_sequence(
            token_blob["init_center_embeddings"][center_idx],
            name="init center token",
        )
        cls = _as_float_sequence(
            token_blob["init_class_embeddings"][class_idx],
            name="init class token",
        )
        pieces = [center, cls]
        residual = token_blob.get("init_residual_embeddings")
        if torch.is_tensor(residual) and residual.shape[2] > 0:
            pieces.append(
                _as_float_sequence(residual[center_idx, class_idx], name="init residual token")
            )
        return torch.cat(pieces, dim=0)

    init = token_blob.get("init_embeddings")
    if init is None:
        return None
    if not torch.is_tensor(init):
        raise PromptTokenError("init_embeddings must be a torch.Tensor")
    init = init.float()
    if init.dim() == 3:
        return init[center_idx, class_idx].unsqueeze(0)
    if init.dim() == 4:
        return init[center_idx, class_idx]
    return None


def scale_token_sequence(
    token_blob: Mapping[str, object],
    token_seq: torch.Tensor,
    center_idx: int,
    class_idx: int,
    scale: float,
) -> torch.Tensor:
    """Scale learned token strength while preserving initialization semantics."""

    scale = float(scale)
    if scale < 0:
        raise PromptTokenError(f"scale must be >= 0, got {scale}")
    if scale == 1.0:
        return token_seq
    init_seq = init_token_sequence_from_bank(token_blob, center_idx, class_idx)
    if init_seq is not None and init_seq.shape == token_seq.shape:
        return init_seq.to(token_seq.device) + scale * (token_seq - init_seq.to(token_seq.device))
    return token_seq * scale


def compile_prompt_text_embed(
    base_text_embed: torch.Tensor,
    token_blob: Mapping[str, object] | None,
    center_idx: int | None = None,
    class_idx: int | None = None,
    *,
    repeat: int = 1,
    scale: float = 1.0,
    no_token: bool = False,
    device: torch.device | str | None = None,
) -> CompiledPromptTextEmbed:
    """Append a prompt-token sequence to base ECGTwin text embeddings.

    The returned mask is always all ones with shape `(1, L)`, matching ECGTwin's
    requirement that a text path must never receive an all-zero mask.
    """

    if repeat < 1:
        raise PromptTokenError(f"repeat must be >= 1, got {repeat}")
    if scale < 0:
        raise PromptTokenError(f"scale must be >= 0, got {scale}")
    base = _as_float_sequence(base_text_embed, name="base_text_embed")
    if device is not None:
        base = base.to(device)

    token_sequence = None
    if no_token:
        text_embed = base
    else:
        if token_blob is None:
            raise PromptTokenError("token_blob is required unless no_token=True")
        if center_idx is None or class_idx is None:
            raise PromptTokenError("center_idx and class_idx are required unless no_token=True")
        token_sequence = token_sequence_from_bank(token_blob, int(center_idx), int(class_idx))
        token_sequence = scale_token_sequence(
            token_blob,
            token_sequence,
            int(center_idx),
            int(class_idx),
            float(scale),
        )
        token_sequence = token_sequence.repeat(int(repeat), 1)
        if device is not None:
            token_sequence = token_sequence.to(device)
        if token_sequence.shape[-1] != base.shape[-1]:
            raise PromptTokenError(
                "token dimension must match base text embedding dimension: "
                f"{token_sequence.shape[-1]} vs {base.shape[-1]}"
            )
        text_embed = torch.cat([base, token_sequence], dim=0)

    if text_embed.shape[0] < 1:
        raise PromptTokenError("compiled text embedding must contain at least one token")
    mask = torch.ones(1, text_embed.shape[0], dtype=torch.float32, device=text_embed.device)
    return CompiledPromptTextEmbed(
        text_embed=text_embed,
        text_embed_mask=mask,
        token_sequence=token_sequence,
    )
