from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
import torch.nn as nn

from core.latent_pool import LatentPool, build_latent_pool
from core.lhat import select_exact_label_candidates


ROOT = Path(__file__).resolve().parents[2]
ENCODER_IDENTITY = "a" * 64


class _DeterministicMeanEncoder(nn.Module):
    """Tiny frozen encoder with the ECGTwin return contract."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("marker", torch.zeros(()))
        self.sample_arguments: list[bool] = []

    def forward(
        self,
        value: torch.Tensor,
        *,
        sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.sample_arguments.append(bool(sample))
        base = value[:, :4, :].mean(dim=2, keepdim=True)
        latent = base.expand(-1, 4, 128).contiguous()
        return latent, latent / 0.18215, torch.zeros_like(latent)


def _records(count: int = 24) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for selection_index in range(count):
        # The first 21 records share one exact positive set and therefore have
        # exactly M=20 non-self candidates.  The final three are ineligible.
        label = (
            torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
            if selection_index < 21
            else torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32)
        )
        waveform = torch.full(
            (1000, 12), float(selection_index + 1), dtype=torch.float32
        )
        records.append(
            {
                "waveform": waveform,
                "label": label,
                "hash_id": f"{selection_index:064x}",
                "selection_index": torch.tensor(selection_index),
                "cache_index": torch.tensor(1000 + selection_index),
                "sampling_rate_hz": torch.tensor(100),
            }
        )
    return records


def _collate(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "waveform": torch.stack([record["waveform"] for record in records]),
        "label": torch.stack([record["label"] for record in records]),
        "hash_id": [str(record["hash_id"]) for record in records],
        "selection_index": torch.stack(
            [record["selection_index"] for record in records]
        ),
        "cache_index": torch.stack([record["cache_index"] for record in records]),
        "sampling_rate_hz": torch.stack(
            [record["sampling_rate_hz"] for record in records]
        ),
    }


def _build(
    batches: list[Mapping[str, Any]],
    *,
    encoder: _DeterministicMeanEncoder | None = None,
) -> tuple[LatentPool, _DeterministicMeanEncoder]:
    resolved_encoder = _DeterministicMeanEncoder() if encoder is None else encoder
    resolved_encoder.eval()
    pool = build_latent_pool(
        resolved_encoder,
        batches,
        encoder_identity=ENCODER_IDENTITY,
        num_candidates=20,
    )
    return pool, resolved_encoder


def test_build_latent_pool_uses_deterministic_mean_and_sorts_selection() -> None:
    records = _records()
    # Deliberately reverse both batches and records.  Pool identity and lookup
    # order are defined by selection_index, not DataLoader batching details.
    pool, encoder = _build(
        [_collate(list(reversed(records[12:]))), _collate(list(reversed(records[:12])))]
    )

    assert encoder.sample_arguments == [False, False]
    assert pool.hash_ids == tuple(f"{index:064x}" for index in range(24))
    assert pool.selection_indices.tolist() == list(range(24))
    assert pool.cache_indices.tolist() == list(range(1000, 1024))
    assert pool.latents.shape == pool.standardized_latents.shape == (24, 4, 128)
    assert pool.labels.shape == (24, 5)
    assert torch.isfinite(pool.standardized_latents).all()

    # The standardizer is deliberately train-only *and eligibility-only*.
    expected_eligible = pool.latents[:21]
    torch.testing.assert_close(
        pool.standardizer.mean,
        expected_eligible.mean(dim=0, keepdim=True),
    )
    assert pool.standardizer.count == 21


def test_eligibility_is_explicit_exact_label_nonself_and_identity_is_stable() -> None:
    records = _records()
    first, _ = _build([_collate(records[:7]), _collate(records[7:])])
    second, _ = _build([_collate(list(reversed(records)))])

    assert first.eligible_hash_ids == tuple(f"{index:064x}" for index in range(21))
    assert first.ineligible_hash_ids == tuple(f"{index:064x}" for index in range(21, 24))
    assert first.candidate_counts.tolist() == [20] * 21 + [2] * 3
    assert first.identity.identity_sha256 == second.identity.identity_sha256
    assert first.identity.eligibility_sha256 == second.identity.eligibility_sha256
    assert (
        first.identity.exact_neighbor_indices_sha256
        == second.identity.exact_neighbor_indices_sha256
    )
    assert first.identity.standardizer_sha256 == first.standardizer.identity_sha256
    assert first.describe()["eligibility"]["ineligible_count"] == 3


def test_precomputed_neighbor_table_matches_reference_selection_exactly() -> None:
    pool, _ = _build([_collate(_records())])
    anchors = torch.tensor([0, 10], dtype=torch.int64)

    expected_nearest = select_exact_label_candidates(
        pool.standardized_latents,
        pool.labels,
        anchor_indices=anchors,
        num_candidates=20,
        mode="nearest",
        local_pool_size=20,
        generator=torch.Generator(device="cpu").manual_seed(23),
    )
    actual_nearest = pool.get_attack_batch_by_pool_indices(
        anchors,
        mode="nearest",
        local_pool_size=20,
        generator=torch.Generator(device="cpu").manual_seed(23),
    ).candidate_pool_indices
    torch.testing.assert_close(actual_nearest, expected_nearest, rtol=0, atol=0)

    expected_random = select_exact_label_candidates(
        pool.standardized_latents,
        pool.labels,
        anchor_indices=anchors,
        num_candidates=20,
        mode="local_random",
        local_pool_size=20,
        generator=torch.Generator(device="cpu").manual_seed(47),
    )
    actual_random = pool.get_attack_batch_by_pool_indices(
        anchors,
        mode="local_random",
        local_pool_size=20,
        generator=torch.Generator(device="cpu").manual_seed(47),
    ).candidate_pool_indices
    torch.testing.assert_close(actual_random, expected_random, rtol=0, atol=0)


def test_hash_and_selection_batch_access_return_aligned_distinct_candidates() -> None:
    pool, _ = _build([_collate(_records())])
    hashes = (pool.hash_ids[0], pool.hash_ids[10])
    generator = torch.Generator(device="cpu").manual_seed(17)

    by_hash = pool.get_attack_batch_by_hashes(
        hashes,
        mode="nearest",
        local_pool_size=20,
        generator=generator,
    )
    by_selection = pool.get_attack_batch_by_selection_indices(
        torch.tensor([0, 10]),
        mode="nearest",
        local_pool_size=20,
        generator=torch.Generator(device="cpu").manual_seed(17),
    )

    assert by_hash.hash_ids == hashes
    assert by_hash.anchor_standardized.shape == (2, 4, 128)
    assert by_hash.candidates_standardized.shape == (2, 20, 4, 128)
    assert by_hash.candidate_pool_indices.shape == (2, 20)
    torch.testing.assert_close(by_hash.anchor_standardized, by_selection.anchor_standardized)
    torch.testing.assert_close(
        by_hash.candidates_standardized, by_selection.candidates_standardized
    )
    for anchor_hash, candidate_hashes, candidate_indices in zip(
        by_hash.hash_ids,
        by_hash.candidate_hash_ids,
        by_hash.candidate_pool_indices.tolist(),
        strict=True,
    ):
        assert anchor_hash not in candidate_hashes
        assert len(set(candidate_hashes)) == 20
        assert tuple(pool.hash_ids[index] for index in candidate_indices) == candidate_hashes
    torch.testing.assert_close(by_hash.labels, pool.labels[[0, 10]])


def test_ineligible_anchor_fails_explicitly_without_padding_or_silent_drop() -> None:
    pool, _ = _build([_collate(_records())])
    bad_hash = pool.hash_ids[21]
    with pytest.raises(
        ValueError,
        match=rf"ineligible.*{bad_hash}.*2.*requires 20",
    ):
        pool.get_attack_batch_by_hashes(
            [bad_hash],
            mode="nearest",
            local_pool_size=20,
            generator=torch.Generator(device="cpu").manual_seed(1),
        )


def test_builder_rejects_nonraw_shape_duplicates_and_unfrozen_encoder() -> None:
    records = _records()
    malformed = _collate(records)
    malformed["waveform"] = malformed["waveform"].transpose(1, 2)
    encoder = _DeterministicMeanEncoder().eval()
    with pytest.raises(ValueError, match=r"\(B,1000,12\)"):
        build_latent_pool(
            encoder,
            [malformed],
            encoder_identity=ENCODER_IDENTITY,
            num_candidates=20,
        )

    duplicate = _collate(records)
    duplicate["hash_id"][1] = duplicate["hash_id"][0]
    with pytest.raises(ValueError, match="duplicate hash_id"):
        build_latent_pool(
            encoder,
            [duplicate],
            encoder_identity=ENCODER_IDENTITY,
            num_candidates=20,
        )

    trainable = nn.Sequential(nn.Linear(1, 1)).eval()
    with pytest.raises(ValueError, match="frozen"):
        build_latent_pool(
            trainable,
            [_collate(records)],
            encoder_identity=ENCODER_IDENTITY,
            num_candidates=20,
        )


def test_latent_pool_only_imports_manual_refactor_surfaces() -> None:
    path = ROOT / "core" / "latent_pool.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed_roots = {
        "__future__",
        "dataclasses",
        "hashlib",
        "typing",
        "torch",
        "core",
        "models",
    }
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert roots <= allowed_roots
    source = path.read_text(encoding="utf-8")
    assert "ecg_adv_gen" not in source
    assert "data_preprocess" not in source
