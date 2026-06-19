"""CPU-only tests for pure Latent-Hull online AT helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_adv_gen.adaptation import (
    SameLabelLatentIndex,
    StratifiedPoolWalker,
    auroc_to_trust,
    build_anchor_preserving_soft_labels,
    build_k500_internal_val_mask,
    build_latent_augmix_branch_signals,
    build_raw_augmix_views,
    build_raw_corruption_views,
    build_three_chain_vae_lhat_augmix_views,
    derive_class_trust,
    derive_kshot_anchor_class_weights,
    dirichlet_with_first_weight_cap,
    global_zscore_np,
    linear_warmup_value,
    parse_class_source_weight_map,
    parse_class_weight_map,
    parse_source_weight_map,
    split_anchor_sample_mode,
    weighted_anchor_quotas,
)
from ecg_adv_gen.labels.super5 import CLASS_NAMES_SUPER5

SUPER5_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES_SUPER5)}


def test_parse_weight_maps_and_reject_bad_items():
    assert parse_source_weight_map("real=2, ptbxl=0.5") == {"real": 2.0, "ptbxl": 0.5}
    assert parse_class_source_weight_map("NORM:real=3,MI:ptbxl=0") == {
        "NORM": {"real": 3.0},
        "MI": {"ptbxl": 0.0},
    }
    assert parse_class_weight_map("NORM=2,MI=-1") == {"NORM": 2.0, "MI": 0.0}
    with pytest.raises(ValueError, match="source=weight"):
        parse_source_weight_map("bad")
    with pytest.raises(ValueError, match="Unknown Super5"):
        parse_class_weight_map("BAD=1")


def test_anchor_sample_mode_and_linear_warmup_helpers():
    assert split_anchor_sample_mode("base_hard_bce") == ("base", "hard_bce")
    assert split_anchor_sample_mode("target_uncertainty") == ("target_teacher", "uncertainty")
    assert split_anchor_sample_mode("stratified") == ("current", "stratified")

    assert linear_warmup_value(0.0, epoch=1, warmup_epochs=3) == 0.0
    assert linear_warmup_value(2.0, epoch=1, warmup_epochs=0) == 2.0
    assert linear_warmup_value(2.0, epoch=1, warmup_epochs=3, start=0.5) == pytest.approx(0.5)
    assert linear_warmup_value(2.0, epoch=2, warmup_epochs=3, start=0.5) == pytest.approx(1.25)
    assert linear_warmup_value(2.0, epoch=3, warmup_epochs=3, start=0.5) == pytest.approx(2.0)


def test_weighted_anchor_quotas_represent_present_classes_and_redistribute_capacity():
    quotas = weighted_anchor_quotas(
        ["NORM", "MI", "STTC"],
        {"NORM": 5, "MI": 1, "STTC": 10},
        K_anchor=8,
        class_weights={"NORM": 1.0, "MI": 10.0, "STTC": 1.0},
    )

    assert sum(quotas.values()) == 8
    assert quotas["MI"] == 1
    assert quotas["NORM"] >= 1
    assert quotas["STTC"] >= 1
    assert weighted_anchor_quotas(["NORM"], {"NORM": 0}, 5, {}) == {"NORM": 0}


def test_derive_kshot_anchor_class_weights_uses_reference_source_when_present():
    labels = np.zeros((6, 5), dtype=np.float32)
    labels[[0, 1, 2, 3], SUPER5_TO_IDX["NORM"]] = 1.0
    labels[[4], SUPER5_TO_IDX["MI"]] = 1.0
    labels[[5], SUPER5_TO_IDX["STTC"]] = 1.0
    sources = np.asarray(["real", "real", "ptbxl", "ptbxl", "real", "ptbxl"])

    weights, info = derive_kshot_anchor_class_weights(
        labels,
        ["NORM", "MI", "STTC"],
        SUPER5_TO_IDX,
        mode="inv_freq_kshot",
        source_labels=sources,
        reference_source="real",
        gamma=1.0,
        min_weight=0.25,
        max_weight=4.0,
        missing_weight=3.0,
        manual_prior={"MI": 2.0},
    )

    assert info["reference_source_found"] is True
    assert info["reference_counts"] == {"NORM": 2, "MI": 1, "STTC": 0}
    assert weights["NORM"] == 0.75
    assert weights["MI"] == 3.0
    assert weights["STTC"] == 3.0


def test_k500_internal_val_mask_is_deterministic_and_keeps_train_samples():
    labels = np.zeros((10, 5), dtype=np.float32)
    labels[:5, SUPER5_TO_IDX["NORM"]] = 1.0
    labels[5:8, SUPER5_TO_IDX["MI"]] = 1.0
    labels[8:, SUPER5_TO_IDX["STTC"]] = 1.0

    mask1 = build_k500_internal_val_mask(labels, val_fraction=0.3, seed=123)
    mask2 = build_k500_internal_val_mask(labels, val_fraction=0.3, seed=123)

    assert mask1.dtype == bool
    assert mask1.tolist() == mask2.tolist()
    assert 0 < int(mask1.sum()) < len(mask1)
    with pytest.raises(ValueError, match="at least 5"):
        build_k500_internal_val_mask(labels[:4], val_fraction=0.2, seed=1)


def test_anchor_preserving_soft_labels_preserve_anchor_and_suppress_abnormal_norm():
    anchor = np.zeros((2, 5), dtype=np.float32)
    anchor[0, SUPER5_TO_IDX["MI"]] = 1.0
    anchor[1, SUPER5_TO_IDX["NORM"]] = 1.0
    candidates = np.zeros((2, 2, 5), dtype=np.float32)
    candidates[0, 0, SUPER5_TO_IDX["MI"]] = 1.0
    candidates[0, 1, SUPER5_TO_IDX["NORM"]] = 1.0
    candidates[1, 0, SUPER5_TO_IDX["NORM"]] = 1.0
    candidates[1, 1, SUPER5_TO_IDX["STTC"]] = 1.0
    weights = np.asarray([[0.7, 0.3], [0.6, 0.4]], dtype=np.float32)

    soft = build_anchor_preserving_soft_labels(
        anchor,
        candidates,
        weights,
        lambda_y=0.5,
        positive_value=0.95,
        negative_floor=0.02,
        new_class_cap=0.2,
    )

    assert soft[0, SUPER5_TO_IDX["MI"]] == pytest.approx(0.95)
    assert soft[0, SUPER5_TO_IDX["NORM"]] == 0.0
    assert soft[1, SUPER5_TO_IDX["NORM"]] == pytest.approx(0.95)
    assert soft[1, SUPER5_TO_IDX["STTC"]] == pytest.approx(0.2)


def test_trust_zscore_and_dirichlet_cap_helpers():
    assert auroc_to_trust(None) == 0.0
    assert auroc_to_trust(0.56) == 0.5
    assert auroc_to_trust(0.71) == 1.0
    trust = derive_class_trust({"NORM": 0.8, "MI": 0.6, "STTC": 0.5})
    assert trust["NORM"] == 1.0
    assert trust["MI"] == 0.5
    assert trust["STTC"] == 0.0
    assert trust["CD"] == 0.0
    assert trust["HYP"] == 0.0

    sig = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    z = global_zscore_np(sig)
    assert float(z.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(z.std()) == pytest.approx(1.0, abs=1e-6)

    rng = np.random.default_rng(0)
    weights = dirichlet_with_first_weight_cap(4, alpha=0.1, first_weight_cap=0.25, rng=rng)
    assert weights[0] <= 0.25 + 1e-6
    assert float(weights.sum()) == pytest.approx(1.0)


def test_raw_augmix_views_use_chains_and_forward_profile():
    signals = np.zeros((2, 12, 8), dtype=np.float32)
    calls: list[tuple[str, int, str]] = []

    def apply_op(sig: np.ndarray, op_name: str, severity: int, severity_profile: str) -> np.ndarray:
        calls.append((op_name, severity, severity_profile))
        return sig + (1.0 if op_name == "op_a" else 2.0)

    views, stats = build_raw_augmix_views(
        signals,
        copies=2,
        severity=5,
        severity_profile="calibrated_10to20pp",
        width=3,
        depth=2,
        alpha=1.0,
        ops=["op_a", "op_b"],
        rng=np.random.default_rng(123),
        op_apply_fn=apply_op,
        available_ops=["op_a", "op_b"],
        renorm=False,
        clip_abs=0.0,
    )

    assert views.shape == (4, 12, 8)
    assert stats["enabled"] is True
    assert stats["n_generated"] == 4
    assert stats["copies"] == 2
    assert stats["severity_profile"] == "calibrated_10to20pp"
    assert stats["width"] == 3
    assert stats["chain_depth_mean"] == pytest.approx(2.0)
    assert len(calls) == 2 * 2 * 3 * 2
    assert {call[2] for call in calls} == {"calibrated_10to20pp"}
    assert {call[1] for call in calls} == {5}
    assert float(np.abs(views).sum()) > 0.0


def test_raw_augmix_views_can_use_fixed_corruption_mixture():
    signals = np.zeros((1, 12, 4), dtype=np.float32)

    def apply_op(sig: np.ndarray, op_name: str, severity: int, severity_profile: str) -> np.ndarray:
        return sig + 4.0

    views, stats = build_raw_augmix_views(
        signals,
        copies=1,
        severity=5,
        severity_profile="calibrated_10to20pp",
        width=1,
        depth=1,
        alpha=1.0,
        mixture_mode="fixed",
        mixture_prob=0.75,
        ops=["op_a"],
        rng=np.random.default_rng(123),
        op_apply_fn=apply_op,
        available_ops=["op_a"],
        renorm=False,
        clip_abs=0.0,
    )

    assert views.shape == (1, 12, 4)
    assert np.allclose(views, 3.0)
    assert stats["mixture_mode"] == "fixed"
    assert stats["mixture_prob"] == pytest.approx(0.75)
    assert stats["beta_m_mean"] == pytest.approx(0.75)
    assert stats["view_ops"] == ["op_a"]


def test_raw_augmix_views_record_mixed_view_ops_when_chain_has_multiple_ops():
    signals = np.zeros((1, 12, 4), dtype=np.float32)

    def apply_op(sig: np.ndarray, op_name: str, severity: int, severity_profile: str) -> np.ndarray:
        return sig + (1.0 if op_name == "op_a" else 2.0)

    _views, stats = build_raw_augmix_views(
        signals,
        copies=1,
        severity=5,
        severity_profile="calibrated_10to20pp",
        width=2,
        depth=1,
        alpha=1.0,
        mixture_mode="fixed",
        mixture_prob=1.0,
        ops=["op_a", "op_b"],
        rng=np.random.default_rng(123),
        op_apply_fn=apply_op,
        available_ops=["op_a", "op_b"],
        renorm=False,
        clip_abs=0.0,
    )

    assert stats["view_ops"] == ["__mixed__"]


def test_three_chain_vae_lhat_augmix_uses_two_corruption_chains_and_uncorrupted_adv_chain():
    anchor = np.zeros((2, 12, 8), dtype=np.float32)
    adv = np.full((2, 12, 8), 7.0, dtype=np.float32)
    op_inputs: list[float] = []
    calls: list[tuple[str, int, str]] = []

    def apply_op(sig: np.ndarray, op_name: str, severity: int, severity_profile: str) -> np.ndarray:
        op_inputs.append(float(sig.mean()))
        calls.append((op_name, severity, severity_profile))
        return sig + (1.0 if op_name == "op_a" else 2.0)

    views, stats = build_three_chain_vae_lhat_augmix_views(
        anchor,
        adv,
        copies=1,
        severity=5,
        severity_profile="pn2021c_official_s5",
        width=3,
        depth=2,
        alpha=1.0,
        ops=["op_a", "op_b"],
        rng=np.random.default_rng(123),
        op_apply_fn=apply_op,
        available_ops=["op_a", "op_b"],
        renorm=False,
        clip_abs=0.0,
    )

    assert views.shape == (2, 12, 8)
    assert stats["topology"] == "locked_three_chain_vae_lhat_augmix"
    assert stats["width"] == 3
    assert stats["corruption_chain_count"] == 2
    assert stats["adversarial_chain_count"] == 1
    assert stats["adversarial_chain_corrupted"] is False
    assert stats["severity_profile"] == "pn2021c_official_s5"
    assert stats["renorm"] is False
    assert len(calls) == 2 * 2 * 2
    assert {call[1] for call in calls} == {5}
    assert {call[2] for call in calls} == {"pn2021c_official_s5"}
    assert all(mean < 7.0 for mean in op_inputs)
    assert float(np.max(views)) > 0.0


def test_three_chain_vae_lhat_augmix_rejects_degenerate_width():
    anchor = np.zeros((1, 12, 8), dtype=np.float32)
    adv = np.ones((1, 12, 8), dtype=np.float32)

    def apply_op(sig: np.ndarray, op_name: str, severity: int, severity_profile: str) -> np.ndarray:
        return sig + 1.0

    with pytest.raises(ValueError, match="exactly three chains"):
        build_three_chain_vae_lhat_augmix_views(
            anchor,
            adv,
            copies=1,
            severity=5,
            width=1,
            depth=1,
            alpha=1.0,
            ops=["op_a"],
            rng=np.random.default_rng(123),
            op_apply_fn=apply_op,
            available_ops=["op_a"],
        )


def test_raw_corruption_views_apply_optional_postprocess_fn():
    signals = np.zeros((2, 12, 4), dtype=np.float32)

    def apply_op(sig: np.ndarray, op_name: str, severity: int) -> np.ndarray:
        return sig + 1.0

    views, stats = build_raw_corruption_views(
        signals,
        copies=1,
        severity=5,
        ops=["op_a"],
        prob=1.0,
        rng=np.random.default_rng(123),
        op_apply_fn=apply_op,
        available_ops=["op_a"],
        renorm=False,
        clip_abs=0.0,
        postprocess_fn=lambda sig: sig + 10.0,
        postprocess_name="stabilizer35",
    )

    assert np.allclose(views, 11.0)
    assert stats["postprocess"] == "stabilizer35"


def test_raw_augmix_views_apply_optional_postprocess_fn():
    signals = np.zeros((1, 12, 4), dtype=np.float32)

    def apply_op(sig: np.ndarray, op_name: str, severity: int, severity_profile: str) -> np.ndarray:
        return sig + 2.0

    views, stats = build_raw_augmix_views(
        signals,
        copies=1,
        severity=5,
        severity_profile="calibrated_10to20pp",
        width=1,
        depth=1,
        alpha=1.0,
        mixture_mode="fixed",
        mixture_prob=1.0,
        ops=["op_a"],
        rng=np.random.default_rng(123),
        op_apply_fn=apply_op,
        available_ops=["op_a"],
        renorm=False,
        clip_abs=0.0,
        postprocess_fn=lambda sig: sig * 3.0,
        postprocess_name="stabilizer35",
    )

    assert np.allclose(views, 6.0)
    assert stats["postprocess"] == "stabilizer35"


def test_stratified_pool_walker_draws_without_revisit_until_pool_exhaustion():
    labels = np.zeros((8, 5), dtype=np.float32)
    labels[:4, SUPER5_TO_IDX["NORM"]] = 1.0
    labels[4:6, SUPER5_TO_IDX["MI"]] = 1.0
    labels[6:, SUPER5_TO_IDX["STTC"]] = 1.0

    walker = StratifiedPoolWalker(
        labels,
        ["NORM", "MI", "STTC"],
        SUPER5_TO_IDX,
        seed=7,
    )
    first = walker.sample({"NORM": 2, "MI": 1})
    second = walker.sample({"NORM": 2, "MI": 1})

    assert walker.class_sizes() == {"NORM": 4, "MI": 2, "STTC": 2}
    assert set(first["NORM"]).isdisjoint(set(second["NORM"]))
    assert set(first["MI"]).isdisjoint(set(second["MI"]))
    assert second["STTC"].size == 0


def test_stratified_pool_walker_source_weighted_sampling_tracks_sources():
    labels = np.zeros((6, 5), dtype=np.float32)
    labels[:, SUPER5_TO_IDX["NORM"]] = 1.0
    sources = np.asarray(["real", "real", "real", "synth", "synth", "synth"])
    walker = StratifiedPoolWalker(
        labels,
        ["NORM"],
        SUPER5_TO_IDX,
        seed=3,
        source_labels=sources,
        source_sampling_strategy="source_weighted",
        source_weights={"real": 1.0, "synth": 0.0},
    )

    picked = walker.sample({"NORM": 3})["NORM"]

    assert picked.size == 3
    assert set(sources[picked]) == {"real"}
    assert walker.source_class_sizes()["NORM"] == {"real": 3, "synth": 3}
    assert walker.last_source_counts == {"real": 3, "synth": 0}


def test_same_label_latent_index_primary_include_self_and_compatible_modes():
    latents = np.arange(5 * 4 * 128, dtype=np.float32).reshape(5, 4, 128)
    labels = np.zeros((5, 5), dtype=np.float32)
    labels[[0, 1], SUPER5_TO_IDX["NORM"]] = 1.0
    labels[[2], SUPER5_TO_IDX["MI"]] = 1.0
    labels[[3], SUPER5_TO_IDX["STTC"]] = 1.0
    labels[[4], SUPER5_TO_IDX["CD"]] = 1.0

    primary = SameLabelLatentIndex(latents, labels, label_mode="primary", seed=1)
    cand = primary.candidates_for(np.asarray([0]), M=2)
    assert cand.shape == (1, 2, 4, 128)
    assert primary.last_candidate_indices.tolist() == [[1, 1]]

    self_index = SameLabelLatentIndex(latents, labels, label_mode="primary", include_self=True)
    self_index.candidates_for(np.asarray([2]), M=2)
    assert self_index.last_candidate_indices.tolist() == [[2, 2]]

    compatible = SameLabelLatentIndex(latents, labels, label_mode="compatible", seed=1)
    compatible.candidates_for(np.asarray([2]), M=2)
    assert compatible.last_candidate_indices[0, 0] in {3, 4}
    assert set(compatible.class_sizes()) == {"NORM_ONLY", "ABNORMAL"}


def test_latent_augmix_core_uses_injected_ops_and_caps_latent_branch():
    anchor = np.zeros((2, 12, 10), dtype=np.float32)
    adv = np.ones((2, 12, 10), dtype=np.float32)
    calls: list[tuple[str, int]] = []

    def apply_op(
        sig: np.ndarray,
        op_name: str,
        severity: int,
        severity_profile: str = "standard",
    ) -> np.ndarray:
        assert severity_profile == "standard"
        calls.append((op_name, severity))
        if op_name == "offset":
            return sig + severity * 0.01
        if op_name == "scale":
            return sig * (1.0 + severity * 0.01)
        raise AssertionError(op_name)

    mixed, stats = build_latent_augmix_branch_signals(
        anchor,
        adv,
        copies=2,
        severity=4,
        width=3,
        depth=1,
        alpha=0.5,
        latent_weight_cap=0.35,
        ops=["offset", "scale"],
        rng=np.random.default_rng(12),
        op_apply_fn=apply_op,
        available_ops=["offset", "scale"],
        renorm=False,
        clip_abs=6.0,
    )

    assert mixed.shape == (4, 12, 10)
    assert np.isfinite(mixed).all()
    assert calls
    assert all(severity == 4 for _, severity in calls)
    assert stats["enabled"] is True
    assert stats["n_generated"] == 4
    assert stats["latent_weight_max"] <= 0.35 + 1e-6
    assert stats["ops"] == ["offset", "scale"]

    with pytest.raises(ValueError, match="width"):
        build_latent_augmix_branch_signals(
            anchor,
            adv,
            copies=1,
            severity=4,
            width=1,
            depth=1,
            alpha=0.5,
            latent_weight_cap=0.35,
            ops=["offset"],
            rng=np.random.default_rng(1),
            op_apply_fn=apply_op,
            available_ops=["offset"],
        )


def test_latent_augmix_core_forwards_severity_profile_to_chain_ops():
    anchor = np.zeros((1, 12, 10), dtype=np.float32)
    adv = np.ones((1, 12, 10), dtype=np.float32)
    calls: list[tuple[str, int, str]] = []

    def apply_op(sig: np.ndarray, op_name: str, severity: int, severity_profile: str) -> np.ndarray:
        calls.append((op_name, severity, severity_profile))
        return sig + 0.01

    mixed, stats = build_latent_augmix_branch_signals(
        anchor,
        adv,
        copies=1,
        severity=5,
        severity_profile="calibrated_10to20pp",
        width=3,
        depth=1,
        alpha=1.0,
        latent_weight_cap=0.25,
        ops=["baseline_wander", "random_leads_masking"],
        rng=np.random.default_rng(7),
        op_apply_fn=apply_op,
        available_ops=["baseline_wander", "random_leads_masking"],
        renorm=False,
        clip_abs=6.0,
    )

    assert mixed.shape == (1, 12, 10)
    assert calls
    assert all(severity == 5 for _, severity, _ in calls)
    assert all(profile == "calibrated_10to20pp" for _, _, profile in calls)
    assert stats["severity_profile"] == "calibrated_10to20pp"


def test_raw_corruption_views_use_single_ops_and_prob_gate():
    signals = np.ones((3, 12, 8), dtype=np.float32)
    calls: list[tuple[str, int]] = []

    def apply_op(sig: np.ndarray, op_name: str, severity: int) -> np.ndarray:
        calls.append((op_name, severity))
        if op_name == "offset":
            return sig + 0.5
        if op_name == "scale":
            return sig * 2.0
        raise AssertionError(op_name)

    corrupted, stats = build_raw_corruption_views(
        signals,
        copies=2,
        severity=3,
        ops=["offset", "scale"],
        prob=1.0,
        rng=np.random.default_rng(7),
        op_apply_fn=apply_op,
        available_ops=["offset", "scale"],
        renorm=False,
        clip_abs=6.0,
    )

    assert corrupted.shape == (6, 12, 8)
    assert calls
    assert len(calls) == 6
    assert all(severity == 3 for _, severity in calls)
    assert stats["enabled"] is True
    assert stats["n_generated"] == 6
    assert stats["n_corrupted"] == 6
    assert stats["ops"] == ["offset", "scale"]
    assert np.isfinite(corrupted).all()

    passthrough, passthrough_stats = build_raw_corruption_views(
        signals,
        copies=1,
        severity=3,
        ops=["offset"],
        prob=0.0,
        rng=np.random.default_rng(3),
        op_apply_fn=apply_op,
        available_ops=["offset"],
        renorm=False,
    )
    assert np.allclose(passthrough, signals)
    assert passthrough_stats["n_corrupted"] == 0
