import numpy as np

from ecg_adv_gen.adaptation.lhat import build_three_chain_vae_lhat_augmix_views


def _identity_op(sig_ct, op_name, severity, severity_profile):
    return sig_ct


def _plus_one_op(sig_ct, op_name, severity, severity_profile):
    return sig_ct + 1.0


def test_three_chain_augmix_locked_corrupts_two_clean_chains_and_keeps_adv_raw():
    anchor = np.ones((1, 12, 16), dtype=np.float32)
    adv = np.full((1, 12, 16), 9.0, dtype=np.float32)
    corruption_inputs = []

    def record_plus_one(sig_ct, op_name, severity, severity_profile):
        corruption_inputs.append(sig_ct.copy())
        return sig_ct + 1.0

    views, stats = build_three_chain_vae_lhat_augmix_views(
        anchor,
        adv,
        copies=1,
        severity=5,
        depth=1,
        alpha=1.0,
        ops=["baseline_shift"],
        rng=np.random.default_rng(7),
        op_apply_fn=record_plus_one,
        available_ops=["baseline_shift"],
        mixture_mode="fixed",
        mixture_prob=1.0,
        chain_weights=[1 / 3, 1 / 3, 1 / 3],
        chain_base_mode="clean_clean_third",
        clip_abs=0.0,
    )

    assert len(corruption_inputs) == 2
    assert all(np.allclose(sig, anchor[0]) for sig in corruption_inputs)
    assert np.allclose(views, (2.0 + 2.0 + 9.0) / 3.0)
    assert stats["chain_roles"] == [
        "clean_anchor_corruption",
        "clean_anchor_corruption",
        "vae_lhat_adversarial_waveform",
    ]
    assert stats["corruption_chain_count"] == 2
    assert stats["adversarial_chain_count"] == 1
    assert stats["adversarial_chain_corrupted"] is False


def test_three_chain_augmix_allows_clean_anchor_third_chain_role():
    anchor = np.ones((2, 12, 16), dtype=np.float32)
    adv = np.full((2, 12, 16), 9.0, dtype=np.float32)
    views, stats = build_three_chain_vae_lhat_augmix_views(
        anchor,
        adv,
        copies=2,
        severity=5,
        depth=1,
        alpha=1.0,
        ops=["baseline_shift"],
        rng=np.random.default_rng(7),
        op_apply_fn=_identity_op,
        available_ops=["baseline_shift"],
        mixture_mode="fixed",
        mixture_prob=1.0,
        chain_weights=[0.45, 0.45, 0.10],
        third_chain_role="clean_anchor_control",
    )

    assert views.shape == (4, 12, 16)
    assert np.allclose(views, 1.0)
    assert stats["chain_roles"] == [
        "clean_anchor_corruption",
        "clean_anchor_corruption",
        "clean_anchor_control",
    ]
    assert stats["adversarial_chain_count"] == 0
    assert stats["clean_anchor_control_chain_count"] == 1


def test_three_chain_augmix_all_adv_corrupts_every_chain_from_adv_base():
    anchor = np.ones((1, 12, 16), dtype=np.float32)
    adv = np.full((1, 12, 16), 9.0, dtype=np.float32)
    views, stats = build_three_chain_vae_lhat_augmix_views(
        anchor,
        adv,
        copies=1,
        severity=5,
        depth=1,
        alpha=1.0,
        ops=["baseline_shift"],
        rng=np.random.default_rng(7),
        op_apply_fn=_plus_one_op,
        available_ops=["baseline_shift"],
        mixture_mode="fixed",
        mixture_prob=1.0,
        chain_weights=[1 / 3, 1 / 3, 1 / 3],
        chain_base_mode="all_adv",
        clip_abs=0.0,
    )

    assert views.shape == (1, 12, 16)
    assert np.allclose(views, 10.0)
    assert stats["chain_base_mode"] == "all_adv"
    assert stats["corruption_chain_count"] == 3
    assert stats["adversarial_chain_count"] == 3
    assert stats["adversarial_chain_corrupted"] is True


def test_three_chain_augmix_all_adv_can_use_soft_adv_base():
    anchor = np.ones((1, 12, 16), dtype=np.float32)
    adv = np.full((1, 12, 16), 9.0, dtype=np.float32)
    views, stats = build_three_chain_vae_lhat_augmix_views(
        anchor,
        adv,
        copies=1,
        severity=5,
        depth=1,
        alpha=1.0,
        ops=["baseline_shift"],
        rng=np.random.default_rng(7),
        op_apply_fn=_plus_one_op,
        available_ops=["baseline_shift"],
        mixture_mode="fixed",
        mixture_prob=1.0,
        chain_weights=[1 / 3, 1 / 3, 1 / 3],
        chain_base_mode="all_adv",
        adv_base_mix=0.25,
        clip_abs=0.0,
    )

    assert views.shape == (1, 12, 16)
    assert np.allclose(views, 4.0)
    assert stats["chain_base_mode"] == "all_adv"
    assert stats["adv_base_mix"] == 0.25


def test_three_chain_augmix_one_adv_corrupts_only_third_chain_from_adv_base():
    anchor = np.ones((1, 12, 16), dtype=np.float32)
    adv = np.full((1, 12, 16), 9.0, dtype=np.float32)
    views, stats = build_three_chain_vae_lhat_augmix_views(
        anchor,
        adv,
        copies=1,
        severity=5,
        depth=1,
        alpha=1.0,
        ops=["baseline_shift"],
        rng=np.random.default_rng(7),
        op_apply_fn=_plus_one_op,
        available_ops=["baseline_shift"],
        mixture_mode="fixed",
        mixture_prob=1.0,
        chain_weights=[1 / 3, 1 / 3, 1 / 3],
        chain_base_mode="one_adv",
        clip_abs=0.0,
    )

    assert views.shape == (1, 12, 16)
    assert np.allclose(views, (2.0 + 2.0 + 10.0) / 3.0)
    assert stats["chain_base_mode"] == "one_adv"
    assert stats["chain_roles"] == [
        "clean_anchor_corruption",
        "clean_anchor_corruption",
        "vae_lhat_adversarial_corruption",
    ]
    assert stats["corruption_chain_count"] == 3
    assert stats["adversarial_chain_count"] == 1
    assert stats["clean_anchor_control_chain_count"] == 2
    assert stats["adversarial_chain_corrupted"] is True


def test_three_chain_augmix_one_adv_can_use_soft_adv_base():
    anchor = np.ones((1, 12, 16), dtype=np.float32)
    adv = np.full((1, 12, 16), 9.0, dtype=np.float32)
    views, stats = build_three_chain_vae_lhat_augmix_views(
        anchor,
        adv,
        copies=1,
        severity=5,
        depth=1,
        alpha=1.0,
        ops=["baseline_shift"],
        rng=np.random.default_rng(7),
        op_apply_fn=_plus_one_op,
        available_ops=["baseline_shift"],
        mixture_mode="fixed",
        mixture_prob=1.0,
        chain_weights=[1 / 3, 1 / 3, 1 / 3],
        chain_base_mode="one_adv",
        adv_base_mix=0.25,
        clip_abs=0.0,
    )

    assert views.shape == (1, 12, 16)
    assert np.allclose(views, (2.0 + 2.0 + 4.0) / 3.0)
    assert stats["chain_base_mode"] == "one_adv"
    assert stats["adv_base_mix"] == 0.25


def test_three_chain_augmix_all_clean_plus_vae_adv_keeps_augmix_clean_based():
    anchor = np.ones((1, 12, 16), dtype=np.float32)
    adv = np.full((1, 12, 16), 9.0, dtype=np.float32)
    views, stats = build_three_chain_vae_lhat_augmix_views(
        anchor,
        adv,
        copies=1,
        severity=5,
        depth=1,
        alpha=1.0,
        ops=["baseline_shift"],
        rng=np.random.default_rng(7),
        op_apply_fn=_plus_one_op,
        available_ops=["baseline_shift"],
        mixture_mode="fixed",
        mixture_prob=1.0,
        chain_weights=[1 / 3, 1 / 3, 1 / 3],
        chain_base_mode="all_clean_plus_vae_adv",
        clip_abs=0.0,
    )

    assert views.shape == (1, 12, 16)
    assert np.allclose(views, 2.0)
    assert stats["chain_base_mode"] == "all_clean_plus_vae_adv"
    assert stats["chain_roles"] == ["clean_anchor_corruption"] * 3
    assert stats["corruption_chain_count"] == 3
    assert stats["adversarial_chain_count"] == 0
    assert stats["clean_anchor_control_chain_count"] == 3
    assert stats["adv_weight_mean"] == 0.0
