from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import numpy as np
import torch
import torch.nn as nn
import yaml

from ecg_adv_gen.config import (
    ConfigError,
    build_runner_commands,
    load_experiment_config,
    make_dry_run_manifest,
    validate_experiment_config,
)
from ecg_adv_gen.evaluation.selection import build_matched_training_record
from ecg_adv_gen.training.resume_contract import (
    LOCKED_LATENT_AUGMIX_SIGNAL_SPACE,
    RESUME_CONTRACT_KEYS,
    resume_contract_mismatches,
)


REPO = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = REPO / "configs/local/linbinhao_server.example.yaml"


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _arm_case(row, arm: str) -> dict[str, object]:
    return {
        "arm": arm,
        "role": row.role,
        "vae_lhat": row.vae_lhat,
        "raw_augmix": row.raw_augmix,
        "augmix_view_bce": row.augmix_view_bce,
        "jsd": row.jsd,
        "rho": row.target_adv_fraction,
        "third_chain_route": row.third_chain_route,
    }


def _five_arm_commands():
    matched = importlib.import_module("ecg_adv_gen.matched_effnet")
    config = load_experiment_config(
        REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml",
        LOCAL_CONFIG,
        runtime_context={"run_id": "pytest_task6a_five"},
    )
    config["runner"]["matrix"] = {
        "center": ["ningbo"],
        "case": [
            _arm_case(matched.MATCHED_EFFNET_ARM_COMPONENTS[arm], arm)
            for arm in matched.MATCHED_EFFNET_ARMS
        ],
    }
    return matched, config, build_runner_commands(config)


def test_canonical_effnet_arm_table_has_exact_five_operational_rows():
    module_name = "ecg_adv_gen.matched_effnet"
    assert importlib.util.find_spec(module_name) is not None, "canonical arm module is missing"
    matched = importlib.import_module(module_name)

    assert matched.MATCHED_EFFNET_ARMS == ("a0", "a2", "a3", "a4", "a5")
    assert "a1" not in matched.MATCHED_EFFNET_ARM_COMPONENTS
    assert matched.MATCHED_EFFNET_CONTRACT_VERSION == "matched_effnet_a0_a2_a3_a4_a5_v5"
    expected = {
        "a0": ("matched_direct_k500_baseline", False, False, False, False, 0.0, "clean_budget_control"),
        "a2": ("raw_augmix_clean_third_control", False, True, True, True, 0.0, "clean_anchor_control"),
        "a3": ("vae_lhat_only", True, False, False, False, 0.5, "no_augmix_route"),
        "a4": ("vae_lhat_threechain_no_jsd", True, True, True, False, 0.5, "vae_lhat_adversarial_waveform"),
        "a5": ("vae_lhat_threechain_full", True, True, True, True, 0.5, "vae_lhat_adversarial_waveform"),
    }
    observed = {
        arm: (
            row.role,
            row.vae_lhat,
            row.raw_augmix,
            row.augmix_view_bce,
            row.jsd,
            row.target_adv_fraction,
            row.third_chain_route,
        )
        for arm, row in matched.MATCHED_EFFNET_ARM_COMPONENTS.items()
    }
    assert observed == expected


def test_adapter_resolves_canonical_case_rows_into_distinct_operational_argv():
    matched = importlib.import_module("ecg_adv_gen.matched_effnet")
    config = load_experiment_config(
        REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml",
        LOCAL_CONFIG,
        runtime_context={"run_id": "pytest_task6a_arms"},
    )
    config["runner"]["matrix"] = {
        "center": ["ningbo"],
        "case": [
            _arm_case(matched.MATCHED_EFFNET_ARM_COMPONENTS[arm], arm)
            for arm in matched.MATCHED_EFFNET_ARMS
        ],
    }

    commands = build_runner_commands(config)

    assert [item["matrix"]["case"]["arm"] for item in commands] == list(
        matched.MATCHED_EFFNET_ARMS
    )
    by_arm = {
        item["matrix"]["case"]["arm"]: item["argv"]
        for item in commands
    }
    for arm, row in matched.MATCHED_EFFNET_ARM_COMPONENTS.items():
        argv = by_arm[arm]
        assert _option(argv, "--comparison_arm") == arm
        assert _option(argv, "--target_adv_fraction") == str(row.target_adv_fraction)
        assert _option(argv, "--latent_augmix_third_chain_role") == row.third_chain_route
        assert ("--enable_vae_lhat" in argv) is row.vae_lhat
        assert ("--disable_vae_lhat" in argv) is (not row.vae_lhat)
        assert ("--enable_raw_augmix" in argv) is row.raw_augmix
        assert ("--disable_raw_augmix" in argv) is (not row.raw_augmix)
        assert (float(_option(argv, "--latent_augmix_bce_weight")) > 0.0) is row.augmix_view_bce
        assert (float(_option(argv, "--latent_augmix_consistency_weight")) > 0.0) is row.jsd
        assert "--enable_latent_augmix_consistency" in argv

    assert by_arm["a3"] != by_arm["a4"]
    for arm in ("a0", "a2"):
        argv = by_arm[arm]
        assert _option(argv, "--hull_label_mode") == "compatible"
        assert "--hull_include_anchor" not in argv
    for arm in ("a3", "a4", "a5"):
        argv = by_arm[arm]
        assert _option(argv, "--hull_M") == "20"
        assert _option(argv, "--hull_lambda") == "0.6"
        assert _option(argv, "--hull_steps") == "5"
        assert _option(argv, "--hull_init_logit_gap") == "0.0"
        assert _option(argv, "--hull_label_mode") == "exact"
        assert _option(argv, "--pgd_eps") == "2.0"
        assert "--hull_include_anchor" not in argv


def test_matched_default_promotes_exact_without_changing_generic_vae_defaults():
    generic = yaml.safe_load(
        (REPO / "configs/defaults/vae_lhat_defaults.yaml").read_text(encoding="utf-8")
    )
    matched = yaml.safe_load(
        (REPO / "configs/defaults/effnet_matched_f005_locked.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert generic["adaptation"]["hull"]["label_mode"] == "compatible"
    assert generic["adaptation"]["hull"]["include_anchor"] is True
    assert matched["adaptation"]["hull"]["label_mode"] == "exact"
    assert matched["adaptation"]["hull"]["include_anchor"] is False


def test_adapter_rejects_case_component_drift_from_canonical_table():
    matched = importlib.import_module("ecg_adv_gen.matched_effnet")
    config = load_experiment_config(
        REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml",
        LOCAL_CONFIG,
        runtime_context={"run_id": "pytest_task6a_drift"},
    )
    case = _arm_case(matched.MATCHED_EFFNET_ARM_COMPONENTS["a4"], "a4")
    case["jsd"] = True
    config["runner"]["matrix"] = {"center": ["ningbo"], "case": [case]}

    with pytest.raises(ValueError, match="canonical matched EffNet arm a4"):
        build_runner_commands(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [("label_mode", "compatible"), ("include_anchor", True)],
)
def test_adapter_rejects_canonical_vae_hull_geometry_drift(field: str, value):
    matched = importlib.import_module("ecg_adv_gen.matched_effnet")
    config = load_experiment_config(
        REPO / "configs/experiments/effnet_vae_lhat_augmix_threechain_locked_k500.yaml",
        LOCAL_CONFIG,
        runtime_context={"run_id": "pytest_task8_geometry_drift"},
    )
    config["runner"]["matrix"] = {
        "center": ["ningbo"],
        "case": [_arm_case(matched.MATCHED_EFFNET_ARM_COMPONENTS["a3"], "a3")],
    }
    config["adaptation"]["hull"][field] = value

    with pytest.raises(ConfigError, match="canonical matched EffNet arm a3"):
        build_runner_commands(config)


def test_wrapper_and_child_parser_preserve_canonical_components(monkeypatch):
    from ecg_adv_gen.runner import effnet_vae_lhat_augmix as wrapper
    from ecg_adv_gen.runner import synth_online_at_super5 as child
    from ecg_adv_gen.runner.effnet_vae_lhat import (
        build_effnet_vae_lhat_train_cmd,
        resolve_effnet_vae_lhat_paths,
    )

    matched, _, commands = _five_arm_commands()
    for command in commands:
        arm = command["matrix"]["case"]["arm"]
        row = matched.MATCHED_EFFNET_ARM_COMPONENTS[arm]
        args = wrapper.parse_args(command["argv"][2:])
        assert args.enable_vae_lhat is row.vae_lhat
        assert args.enable_raw_augmix is row.raw_augmix
        paths = resolve_effnet_vae_lhat_paths(
            args, data_root=Path(args.data_root), out_root=Path(args.out_root)
        )
        child_argv = build_effnet_vae_lhat_train_cmd(
            args,
            python=sys.executable,
            data_root=Path(args.data_root),
            paths=paths,
            class_trust=Path("/tmp/class_trust.json"),
        )
        parsed = child.parse_args(child_argv[3:])
        assert parsed.comparison_arm == arm
        assert parsed.enable_vae_lhat is row.vae_lhat
        assert parsed.enable_raw_augmix is row.raw_augmix
        assert parsed.latent_augmix_third_chain_role == row.third_chain_route
        assert parsed.target_adv_fraction == row.target_adv_fraction
        assert parsed.latent_augmix_signal_space == (
            LOCKED_LATENT_AUGMIX_SIGNAL_SPACE
            if arm in {"a2", "a4", "a5"}
            else "model_zscore"
        )
        if row.vae_lhat:
            assert parsed.hull_label_mode == "exact"
            assert parsed.hull_include_anchor is False


def test_canonical_vae_arms_prepare_exact_eligibility_but_no_vae_arms_do_not(
    tmp_path: Path,
):
    from ecg_adv_gen.runner import effnet_vae_lhat_augmix as wrapper
    from ecg_adv_gen.runner import synth_online_at_super5 as child
    from ecg_adv_gen.runner.effnet_vae_lhat import (
        build_effnet_vae_lhat_train_cmd,
        resolve_effnet_vae_lhat_paths,
    )

    labels = np.asarray([[1, 0, 0, 0, 0]] * 3, dtype=np.float32)
    source_meta = {"record_ids": np.asarray(["r0", "r1", "r2"])}
    matched, _, commands = _five_arm_commands()
    for command in commands:
        arm = command["matrix"]["case"]["arm"]
        row = matched.MATCHED_EFFNET_ARM_COMPONENTS[arm]
        args = wrapper.parse_args(command["argv"][2:])
        paths = resolve_effnet_vae_lhat_paths(
            args, data_root=Path(args.data_root), out_root=Path(args.out_root)
        )
        child_argv = build_effnet_vae_lhat_train_cmd(
            args,
            python=sys.executable,
            data_root=Path(args.data_root),
            paths=paths,
            class_trust=Path("/dev/shm/class_trust.json"),
        )
        parsed = child.parse_args(child_argv[3:])
        parsed.output_dir = str(tmp_path / arm)
        (tmp_path / arm).mkdir()

        manifest, eligible = child.prepare_exact_nonself_eligibility(
            parsed, labels, source_meta
        )
        manifest_path = tmp_path / arm / "eligible_anchor_manifest.json"
        if row.vae_lhat:
            assert manifest is not None
            assert eligible == [0, 1, 2]
            assert manifest_path.exists()
        else:
            assert manifest is None and eligible is None
            assert not manifest_path.exists()


def test_selection_and_artifact_contract_use_best_checkpoint_for_all_five_arms():
    matched, config, commands = _five_arm_commands()
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=validate_experiment_config(config, repo_root=REPO),
        run_id="pytest_task6a_five",
        cli_args=Namespace(dry_run=True, write_plan=True),
    )
    child_runs = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    assert len(child_runs) == 5
    for child_run in child_runs:
        roles = {item["role"] for item in child_run["expected_artifacts"]}
        assert "best_model" in roles

    split = {
        "train_record_ids": ["train"],
        "val_record_ids": ["val"],
        "train_record_ids_sha256": "a" * 64,
        "val_record_ids_sha256": "b" * 64,
        "val_fraction": 0.2,
        "seed": 20260601,
    }
    for arm, row in matched.MATCHED_EFFNET_ARM_COMPONENTS.items():
        record = build_matched_training_record(
            comparison_arm=arm,
            source_checkpoint_path="/tmp/source.pt",
            source_checkpoint_sha256="c" * 64,
            split=split,
            selection_metric="macro_auprc",
            source_floor_max_drop=0.02,
            epochs=30,
            optimizer_steps_per_epoch=4,
            realized_optimizer_steps=120,
            scheduler_steps=30,
            source_floor_result={"source_floor_passed": True},
        )
        assert record["contract"] == matched.MATCHED_EFFNET_CONTRACT_VERSION
        assert record["role"] == row.role
        assert record["selection"]["checkpoint"] == "best_model.pt"
        assert record["method_components"] == {
            "vae_lhat": row.vae_lhat,
            "raw_augmix": row.raw_augmix,
            "augmix_view_bce": row.augmix_view_bce,
            "jsd": row.jsd,
        }


def test_no_vae_arms_do_not_require_latent_artifacts():
    matched, config, _ = _five_arm_commands()
    config["runner"]["matrix"]["case"] = [
        _arm_case(matched.MATCHED_EFFNET_ARM_COMPONENTS[arm], arm)
        for arm in ("a0", "a2")
    ]
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=validate_experiment_config(config, repo_root=REPO),
        run_id="pytest_task6a_no_vae",
        cli_args=Namespace(dry_run=True, write_plan=True),
    )
    refs = manifest["artifact_trace"]["inputs"]["k500_refs"]
    assert refs
    assert all(item["latent_npz"] is None for item in refs)


@pytest.mark.parametrize(
    "key",
    [
        "comparison_arm",
        "enable_vae_lhat",
        "enable_raw_augmix",
        "latent_augmix_third_chain_role",
        "latent_augmix_chain_base_mode",
        "latent_augmix_ops",
        "latent_augmix_consistency_weight",
        "latent_augmix_consistency_loss",
        "latent_augmix_bce_weight",
        "vae_adv_consistency_weight",
    ],
)
def test_resume_contract_rejects_matched_arm_component_drift(key: str):
    assert key in RESUME_CONTRACT_KEYS
    saved = {key: "saved"}
    current = {key: "current"}
    assert resume_contract_mismatches(saved, current) == [
        {"key": key, "saved": "saved", "current": "current"}
    ]


def test_no_vae_asset_route_never_loads_ecgtwin_or_latent_pool():
    from ecg_adv_gen.runner import synth_online_at_super5 as child

    def forbidden(*args, **kwargs):
        raise AssertionError("no-VAE arm touched a VAE-only loader")

    assets = child.load_optional_vae_assets(
        Namespace(enable_vae_lhat=False, device="cpu", synth_npz="/missing.latent.npz"),
        ecgtwin_factory=forbidden,
        latent_pool_loader=forbidden,
    )
    assert assets == (None, None)
    assert child.load_optional_vae_component(False, forbidden) is None


def test_clean_budget_control_builds_identity_views_without_raw_ops():
    from ecg_adv_gen.runner import synth_online_at_super5 as child

    class TinyDataset:
        signals = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
        labels = np.eye(2, dtype=np.float32)[np.arange(4) % 2]

        def __len__(self):
            return len(self.signals)

        def __getitem__(self, index):
            return torch.from_numpy(self.signals[index]), torch.from_numpy(self.labels[index])

    args = Namespace(
        K_anchor=4,
        seed=7,
        enable_raw_augmix=False,
        latent_augmix_copies=2,
    )
    bundle = child.build_clean_anchor_augmix_epoch(TinyDataset(), args, epoch=1)
    assert np.array_equal(bundle["views"], np.tile(bundle["clean"], (2, 1, 1)))
    assert bundle["stats"]["control"] == "clean_budget_control"
    assert bundle["stats"]["raw_augmix"] is False


def test_a2_raw_first_augmix_uses_paired_raw_crop_then_one_model_zscore(monkeypatch):
    from ecg_adv_gen.adaptation import lhat
    from ecg_adv_gen.runner import synth_online_at_super5 as child

    time = np.linspace(-2.0, 2.0, 1000, dtype=np.float32)[:, None]
    leads = np.arange(12, dtype=np.float32)[None, :] * 0.75
    signals = (50.0 + 8.0 * time + leads)[None, ...].astype(np.float32)
    labels = np.asarray([[1.0, 0.0]], dtype=np.float32)
    dataset = child.TargetRealRawFirstCorruptionDataset(
        signals,
        labels,
        crop_len=250,
        mode="eval",
        clean_norm_mode="per_sample_global",
    )

    captured_ops: list[tuple[np.ndarray, float]] = []

    def capture_raw_op(signal, *args, sample_rate_hz, **kwargs):
        captured_ops.append((signal.detach().cpu().numpy().copy(), float(sample_rate_hz)))
        return signal + 0.25

    real_global_zscore = lhat.global_zscore_np
    mixed_zscore_inputs: list[np.ndarray] = []

    def count_mixed_zscore(signal):
        mixed_zscore_inputs.append(np.asarray(signal).copy())
        return real_global_zscore(signal)

    monkeypatch.setattr(child, "apply_corruption_sequence", capture_raw_op)
    monkeypatch.setattr(lhat, "global_zscore_np", count_mixed_zscore)
    args = Namespace(
        K_anchor=1,
        seed=11,
        enable_raw_augmix=True,
        latent_augmix_copies=1,
        latent_augmix_severity=5,
        latent_augmix_severity_profile="standard",
        latent_augmix_width=3,
        latent_augmix_depth=1,
        latent_augmix_alpha=1.0,
        latent_augmix_ops=["powerline_noise"],
        latent_augmix_third_chain_role="clean_anchor_control",
        latent_augmix_chain_base_mode="clean_clean_third",
        latent_augmix_chain_weights="1,0,0",
        latent_augmix_signal_space=LOCKED_LATENT_AUGMIX_SIGNAL_SPACE,
    )

    bundle = child.build_clean_anchor_augmix_epoch(dataset, args, epoch=1)

    expected_raw_ct = signals[0, 375:625, :].T
    expected_clean_ct = child._per_sample_global_zscore_np(signals[0, 375:625, :]).T
    np.testing.assert_allclose(bundle["raw"], expected_raw_ct[None, ...])
    np.testing.assert_allclose(bundle["clean"], expected_clean_ct[None, ...], atol=1e-6)
    assert captured_ops
    for op_input, sample_rate_hz in captured_ops:
        assert op_input.shape == (12, 250)
        assert sample_rate_hz == 100.0
        np.testing.assert_allclose(op_input, expected_raw_ct)
        assert abs(float(op_input.mean())) > 10.0
        assert float(op_input.std()) > 1.0
    assert len(mixed_zscore_inputs) == 1
    assert bundle["views"].shape == (1, 12, 250)
    assert float(bundle["clean"].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(bundle["clean"].std()) == pytest.approx(1.0, abs=1e-6)
    assert float(bundle["views"].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(bundle["views"].std()) == pytest.approx(1.0, abs=1e-6)
    assert bundle["stats"]["signal_space"] == LOCKED_LATENT_AUGMIX_SIGNAL_SPACE
    assert bundle["stats"]["corruption_source"] == "target_real_npz.signals_raw1000"


def test_auxiliary_bce_supervises_augmix_views_without_clean_bce_double_count():
    from ecg_adv_gen.runner.synth_online_at_super5 import train_latent_augmix_consistency_epoch

    clean = np.zeros((2, 1, 1), dtype=np.float32)
    views = np.full((4, 1, 1), 2.0, dtype=np.float32)
    labels = np.zeros((2, 1), dtype=np.float32)
    model = nn.Sequential(nn.Flatten(), nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[1].weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    stats = train_latent_augmix_consistency_epoch(
        model=model,
        clean_signals_ct=clean,
        augmix_signals_ct=views,
        labels_np=labels,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        copies=2,
        consistency_weight=0.0,
        bce_weight=1.0,
        consistency_loss="jsd",
        batch_size=2,
        crop_len=1,
        grad_clip=0.0,
        trainable_params=list(model.parameters()),
    )
    expected_view_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.full((4, 1), 2.0), torch.zeros((4, 1))
    ).item()
    assert stats["bce_loss"] == pytest.approx(expected_view_bce)


def test_all_five_arms_spend_the_same_auxiliary_optimizer_step_budget():
    from ecg_adv_gen.runner import synth_online_at_super5 as child
    from ecg_adv_gen.matched_effnet import MATCHED_EFFNET_ARM_COMPONENTS

    train_latent_augmix_consistency_epoch = child.train_latent_augmix_consistency_epoch

    clean = np.zeros((6, 1, 1), dtype=np.float32)
    labels = np.zeros((6, 1), dtype=np.float32)
    steps = {}
    stats_by_arm = {}
    for arm, row in MATCHED_EFFNET_ARM_COMPONENTS.items():
        views = np.tile(clean + (1.0 if row.raw_augmix else 0.0), (2, 1, 1))
        model = nn.Sequential(nn.Flatten(), nn.Linear(1, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        inactive = train_latent_augmix_consistency_epoch(
            model=model,
            clean_signals_ct=clean,
            augmix_signals_ct=views,
            labels_np=labels,
            optimizer=optimizer,
            criterion=nn.BCEWithLogitsLoss(reduction="none"),
            device="cpu",
            copies=2,
            consistency_weight=2.0 if row.jsd else 0.0,
            bce_weight=1.0 if row.augmix_view_bce else 0.0,
            consistency_loss="jsd",
            batch_size=2,
            crop_len=1,
            grad_clip=0.0,
            trainable_params=list(model.parameters()),
        )
        if row.augmix_view_bce or row.jsd:
            stats = inactive
        else:
            assert inactive["n_batches"] == 0
            stats = child.run_zero_effect_auxiliary_optimizer_control(
                optimizer,
                n_samples=len(clean),
                batch_size=2,
            )
        steps[arm] = stats["n_batches"]
        stats_by_arm[arm] = stats
    assert set(steps.values()) == {3}
    assert stats_by_arm["a0"]["control"] == "zero_effect_optimizer_control"
    assert stats_by_arm["a3"]["control"] == "zero_effect_optimizer_control"
    assert stats_by_arm["a4"]["bce_weight"] > 0.0
    assert stats_by_arm["a4"]["consistency_weight"] == 0.0
    assert stats_by_arm["a5"]["consistency_weight"] > 0.0


@pytest.mark.parametrize("arm", ["a0", "a3"])
def test_zero_weight_aux_control_has_no_forward_or_training_state_effect(arm: str):
    from ecg_adv_gen.runner import synth_online_at_super5 as child

    class ForwardCountingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(1)
            self.head = nn.Linear(2, 1)
            self.forward_calls = 0

        def forward(self, inputs):
            self.forward_calls += 1
            return self.head(self.bn(inputs).flatten(1))

    class CountingAdamW(torch.optim.AdamW):
        def __init__(self, params):
            super().__init__(params, lr=0.01, weight_decay=0.1)
            self.realized_steps = 0

        def step(self, closure=None):
            self.realized_steps += 1
            return super().step(closure)

    torch.manual_seed(19)
    model = ForwardCountingModel()
    optimizer = CountingAdamW(model.parameters())
    warm = torch.randn(4, 1, 2)
    model(warm).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer.realized_steps = 0
    ewa_params = [parameter.detach().clone() for parameter in model.parameters()]
    params_before = [parameter.detach().clone() for parameter in model.parameters()]
    bn_mean_before = model.bn.running_mean.detach().clone()
    bn_var_before = model.bn.running_var.detach().clone()
    optimizer_state_before = {
        parameter: {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in state.items()
        }
        for parameter, state in optimizer.state.items()
    }
    forward_calls_before = model.forward_calls
    clean = np.zeros((5, 1, 2), dtype=np.float32)
    views = np.tile(clean, (2, 1, 1))
    labels = np.zeros((5, 1), dtype=np.float32)

    inactive = child.train_latent_augmix_consistency_epoch(
        model=model,
        clean_signals_ct=clean,
        augmix_signals_ct=views,
        labels_np=labels,
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(reduction="none"),
        device="cpu",
        copies=2,
        consistency_weight=0.0,
        bce_weight=0.0,
        consistency_loss="jsd",
        batch_size=2,
        crop_len=2,
        grad_clip=0.0,
        trainable_params=list(model.parameters()),
    )
    assert inactive["n_batches"] == 0
    control = child.run_zero_effect_auxiliary_optimizer_control(
        optimizer,
        n_samples=len(clean),
        batch_size=2,
    )

    assert control["n_batches"] == 3
    assert optimizer.realized_steps == 3
    assert model.forward_calls == forward_calls_before
    torch.testing.assert_close(model.bn.running_mean, bn_mean_before)
    torch.testing.assert_close(model.bn.running_var, bn_var_before)
    for actual, expected in zip(model.parameters(), params_before):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(ewa_params, params_before):
        torch.testing.assert_close(actual, expected)
    assert optimizer.state.keys() == optimizer_state_before.keys()
    for parameter, state_before in optimizer_state_before.items():
        for key, expected in state_before.items():
            actual = optimizer.state[parameter][key]
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual, expected)
            else:
                assert actual == expected


def test_zero_effect_auxiliary_control_is_matched_only():
    from ecg_adv_gen.runner import synth_online_at_super5 as child

    assert child.should_run_zero_effect_auxiliary_control(
        matched_comparison=True,
        consistency_weight=0.0,
        bce_weight=0.0,
    )
    assert not child.should_run_zero_effect_auxiliary_control(
        matched_comparison=False,
        consistency_weight=0.0,
        bce_weight=0.0,
    )


def test_a3_raw_augmix_disabled_skips_identity_builder_and_model_scoring(monkeypatch):
    from ecg_adv_gen.runner import synth_online_at_super5 as child

    def forbidden_builder(*args, **kwargs):
        raise AssertionError("A3 must not build target-real identity AugMix views")

    class ForbiddenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, inputs):
            self.calls += 1
            raise AssertionError("A3 disabled raw-AugMix views must not be scored")

    monkeypatch.setattr(child, "build_clean_anchor_augmix_epoch", forbidden_builder)
    template_signals = np.zeros((3, 12, 250), dtype=np.float32)
    template_labels = np.zeros((3, 5), dtype=np.float32)
    bundle = child.build_no_raw_augmix_epoch(
        object(),
        Namespace(latent_augmix_signal_space="model_zscore"),
        epoch=1,
        matched_comparison=True,
    )
    views = np.empty((0,) + template_signals.shape[1:], dtype=np.float32)
    stats = child.raw_augmix_disabled_stats("model_zscore")
    victim, teacher = ForbiddenModel(), ForbiddenModel()
    if views.size:
        victim(torch.from_numpy(views))
        teacher(torch.from_numpy(views))

    assert bundle is None
    assert views.shape == (0, 12, 250)
    assert template_labels.shape == (3, 5)
    assert stats == {
        "enabled": False,
        "reason": "raw_augmix_disabled",
        "raw_augmix": False,
        "n_generated": 0,
        "signal_space": "model_zscore",
    }
    assert victim.calls == teacher.calls == 0


def test_manifest_resolves_required_metrics_per_arm_without_dropping_common_metrics():
    matched, config, _ = _five_arm_commands()
    config["logging"]["required_epoch_metrics"] = ["epoch", "train_loss"]
    config["logging"]["required_epoch_metrics_by_arm"] = {
        "a3": ["atk_init", "latent_hull_diagnostics"],
        "a4": ["atk_init", "latent_hull_diagnostics"],
        "a5": ["atk_init", "latent_hull_diagnostics"],
    }
    commands = build_runner_commands(config)
    manifest = make_dry_run_manifest(
        config,
        commands=commands,
        local_paths=validate_experiment_config(config, repo_root=REPO),
        run_id="pytest_task6a_metrics",
        cli_args=Namespace(dry_run=True, write_plan=True),
    )
    children = manifest["artifact_trace"]["expected_outputs"]["child_runs"]
    by_arm = {child["matrix"]["case"]["arm"]: child for child in children}
    for arm in ("a0", "a2"):
        assert by_arm[arm]["required_epoch_metrics"] == ["epoch", "train_loss"]
    for arm in ("a3", "a4", "a5"):
        assert by_arm[arm]["required_epoch_metrics"] == [
            "epoch", "train_loss", "atk_init", "latent_hull_diagnostics"
        ]


def test_per_arm_metric_validation_rejects_missing_common_and_null_fake_values(tmp_path: Path):
    from ecg_adv_gen.config.launch import _validate_training_log_metrics

    path = tmp_path / "training_log.json"
    path.write_text(json.dumps({"epochs": [{"atk_init": None}]}), encoding="utf-8")
    record = {
        "role": "training_log",
        "path": str(path),
        "required_epoch_metrics": ["epoch", "atk_init"],
    }
    errors = _validate_training_log_metrics(
        path,
        manifest={
            "artifact_trace": {
                "expected_outputs": {"required_epoch_metrics": ["epoch"]}
            }
        },
        record=record,
    )
    assert errors[0]["missing_metrics"] == ["epoch", "atk_init"]
