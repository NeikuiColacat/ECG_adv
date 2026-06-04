"""Tests for K-shot artifact path and ref-meta contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecg_adv_gen.data.kshot_artifacts import (
    KShotArtifactGroup,
    canonical_kshot_base,
    read_ref_meta_record_ids,
)


def test_canonical_kshot_base_uses_center_k_seed_layout(tmp_path: Path):
    base = canonical_kshot_base(tmp_path / "subsets", "ningbo", k=500, seed=20260531)

    assert base == (
        tmp_path
        / "subsets"
        / "ningbo"
        / "k500_seed20260531"
        / "ningbo_real_k500_seed20260531"
    )


def test_kshot_artifact_group_builds_manifest_paths_from_root(tmp_path: Path):
    group = KShotArtifactGroup.from_root(
        tmp_path / "subsets",
        center="georgia",
        k=500,
        seed=20260531,
        include_latent=True,
        include_trust=True,
    )

    assert group.center == "georgia"
    assert group.k == 500
    assert group.seed == 20260531
    assert group.base.name == "georgia_real_k500_seed20260531"
    assert group.signals.name == "georgia_real_k500_seed20260531.signals.npz"
    assert group.latents.name == "georgia_real_k500_seed20260531.latent.npz"
    assert group.ref_meta.name == "georgia_real_k500_seed20260531.ref_meta.json"
    assert group.class_trust.name == "georgia_real_k500_seed20260531.class_trust.json"
    assert group.manifest_paths() == {
        "anchor_base": str(group.base),
        "signals_npz": group.signals,
        "latent_npz": group.latents,
        "ref_meta_json": group.ref_meta,
        "class_trust_json": group.class_trust,
    }


def test_kshot_artifact_group_can_omit_latent_and_trust(tmp_path: Path):
    group = KShotArtifactGroup.from_root(tmp_path, center="cpsc_2018", k=500, seed=20260531)

    assert group.latents is None
    assert group.class_trust is None
    assert group.manifest_paths()["latent_npz"] is None
    assert group.manifest_paths()["class_trust_json"] is None


def test_kshot_artifact_group_rejects_base_that_does_not_match_protocol(tmp_path: Path):
    bad_base = tmp_path / "ningbo" / "k400_seed42" / "ningbo_real_k400_seed42"

    with pytest.raises(ValueError, match="does not match expected K-shot base name"):
        KShotArtifactGroup.from_base(bad_base, center="ningbo", k=500, seed=20260531)


def test_read_ref_meta_record_ids_validates_center_k_seed_and_duplicates(tmp_path: Path):
    meta_path = tmp_path / "ningbo_real_k2_seed20260531.ref_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "center": "ningbo",
                "K": 2,
                "selection_seed": 20260531,
                "ref_record_ids": ["N2", "N1"],
            }
        ),
        encoding="utf-8",
    )

    result = read_ref_meta_record_ids(
        meta_path,
        expected_center="ningbo",
        expected_k=2,
        expected_seed=20260531,
    )

    assert result.center == "ningbo"
    assert result.record_ids == ["N2", "N1"]
    assert result.sorted_record_ids == ["N1", "N2"]

    meta_path.write_text(
        json.dumps(
            {
                "center": "ningbo",
                "K": 2,
                "selection_seed": 20260531,
                "ref_record_ids": ["N1", "N1"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_ref_meta_record_ids(meta_path, expected_center="ningbo", expected_k=2, expected_seed=20260531)
