from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import boot_scripts.train_pn2021 as boot_module
import core.train_PN2021 as pn_module
from core.train_PN2021 import build_pn2021_k500_dataloader, train_pn2021
from models.checkpoints import sha256_file
from models.contracts import CLASS_ORDER, EFFICIENTNET1DV2_SPEC
from util.config_bundle import resolve_yaml_config_closure
from util.evaluation.direct_baseline_selection import locked_pooled_selection_rule
from util.run_record import build_run_file_index, snapshot_yaml_files


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "train" / "PN2021.yaml"
DIRECT_CONFIG = REPO / "configs" / "train" / "PN2021_fixed20.yaml"


def _selection_fixture(
    tmp_path: Path,
) -> tuple[Path, SimpleNamespace, Path, Path, SimpleNamespace]:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    vae = tmp_path / "vae.pt"
    vae.write_bytes(b"vae")
    config_root = tmp_path / "configs"
    method_path = config_root / "train" / "methods" / "method.yaml"
    method_path.parent.mkdir(parents=True)
    method_path.write_text("schema_version: 1\n", encoding="utf-8")
    method = SimpleNamespace(
        profile_name="latent_threechain_augmix_residual_depth23_aug075",
        source_path=method_path,
    )
    seed_path = config_root / "random_seed.yaml"
    seed_path.write_text("schema_version: 1\nseed: 20260715\n", encoding="utf-8")
    registry_path = config_root / "baselines" / "source_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "baseline": {
                    "random_seed_config_sha256": sha256_file(seed_path),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    online_path = config_root / "train" / "PN2021.yaml"
    online_path.parent.mkdir(parents=True, exist_ok=True)
    online_payload = {
        "schema_version": 1,
        "profile_name": "fixture_online_training",
        "references": {
            "random_seed_config": "random_seed.yaml",
            "source_baseline_registry": "baselines/source_registry.yaml",
        },
        "protocol": {"status": "fixture", "partition": "k500"},
        "random_seed": {
            "stream_namespace": "pn2021_fixture_stream",
            "comparison_group": "pn2021_matched_test",
            "replicate_id": 0,
            "deterministic_algorithms": False,
        },
        "fairness": {"budget_policy": "matched_base"},
        "data": {
            "model_normalization": "per_sample_global_zscore",
            "normalization_epsilon": 1.0e-6,
        },
        "training": {"drop_last": False, "cache_mode": "mmap"},
        "diagnostics": {"diagnostics_every_steps": 1},
        "logging": {"tensorboard": {"enabled": False}},
        "output": {"root_dir": str(tmp_path / "output")},
    }
    online_path.write_text(
        yaml.safe_dump(online_payload, sort_keys=False), encoding="utf-8"
    )
    tuning_path = config_root / "train" / "PN2021_tuning.yaml"
    tuning_path.write_text(
        "schema_version: 1\nprofile_name: fixture_tuning\n", encoding="utf-8"
    )
    config = SimpleNamespace(
        payload=online_payload,
        references={
            "random_seed_config": seed_path,
            "source_baseline_registry": registry_path,
        },
        config_root=config_root,
        path=online_path,
        sha256=sha256_file(online_path),
    )
    parameters = {
        "epochs": 2,
        "scheduler_horizon_epochs": 2,
        "batch_size": 4,
        "learning_rate": 5.0e-5,
        "weight_decay": 1.0e-4,
        "minimum_learning_rate_ratio": 0.01,
        "gradient_clip_norm": 1.0,
        "amp_enabled": True,
        "amp_dtype": "bfloat16",
    }
    comparison = {
        "model_family": "efficientnet1dv2",
        "method_id": method.profile_name,
        "method_profile_sha256": sha256_file(method_path),
        "online_training_config_sha256": sha256_file(online_path),
        "tuning_config_sha256": sha256_file(tuning_path),
        "comparison_group": "pn2021_matched_test",
        "replicate_id": 0,
        "pos_weight": None,
        "source_checkpoint_identity": {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
        },
        "vae_encoder_checkpoint": {"sha256": sha256_file(vae)},
        "vae_decoder_checkpoint": {"sha256": sha256_file(vae)},
        "resolved_training_parameters": parameters,
    }
    per_epoch = [
        {
            "epoch": 1,
            "eligible": True,
            "score": 0.8,
            "clean": {"macro_auprc": 0.9},
            "robust": {"macro_auprc": 0.7},
        },
        {
            "epoch": 2,
            "eligible": True,
            "score": 0.7,
            "clean": {"macro_auprc": 0.8},
            "robust": {"macro_auprc": 0.6},
        },
    ]
    selection = {
        "schema_version": 1,
        "status": "selected",
        "artifact_type": "pn2021_k500_pooled_epoch_selection",
        "centers": list(boot_module.ALLOWED_CENTERS),
        "class_order": list(CLASS_ORDER),
        "record_count": 400,
        "composition_count": 20,
        "selection_rule": locked_pooled_selection_rule(),
        "comparison_identity": comparison,
        "comparison_identity_sha256": boot_module._canonical_sha256(comparison),
        "selected_epoch": 1,
        "selected_score": 0.8,
        "selected_clean_macro_auprc": 0.9,
        "selected_robust_macro_auprc": 0.7,
        "epoch_grid": [1, 2],
        "per_epoch": per_epoch,
    }
    run_root = tmp_path / "managed-selection"
    path = run_root / "training" / "selection.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(selection), encoding="utf-8")
    closure = resolve_yaml_config_closure(
        (method_path, seed_path, registry_path, online_path, tuning_path),
        config_root=config_root,
    )
    snapshots = snapshot_yaml_files(
        run_root,
        [(item, "referenced_config") for item in closure],
        config_root=config_root,
    )
    (run_root / "run_card.json").write_text(
        json.dumps({"status": "complete", "exit_code": 0}), encoding="utf-8"
    )
    index_path = run_root / "run_file_index.json"
    index_path.write_text(
        json.dumps(build_run_file_index(run_root)), encoding="utf-8"
    )
    (run_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_file_index_sha256": sha256_file(index_path),
                "config_snapshots": snapshots,
            }
        ),
        encoding="utf-8",
    )
    return path, method, source, vae, config


def test_selected_refit_contract_validates_epoch_and_resource_identity(tmp_path):
    path, method, source, vae, config = _selection_fixture(tmp_path)
    resolved = {
        "epochs": 2,
        "scheduler_horizon_epochs": 2,
        "batch_size": 4,
        "learning_rate": 5.0e-5,
        "weight_decay": 1.0e-4,
        "minimum_learning_rate_ratio": 0.01,
        "gradient_clip_norm": 1.0,
        "amp_enabled": True,
        "amp_dtype": "bfloat16",
    }

    result = boot_module._validate_refit_selection(
        path,
        model_name="efficientnet1dv2",
        method=method,
        source_checkpoint=source.resolve(),
        vae_checkpoint=vae.resolve(),
        resolved_training=resolved,
        config=config,
        trainable_scope="full",
        pos_weight=None,
    )

    assert result["selected_epoch"] == 1
    assert result["training_parameters"]["epochs"] == 1
    assert result["heldout_evaluation_used"] is False
    assert result["sha256"] == sha256_file(path)
    with pytest.raises(ValueError, match="explicit refit epochs"):
        boot_module._validate_refit_selection(
            path,
            model_name="efficientnet1dv2",
            method=method,
            source_checkpoint=source.resolve(),
            vae_checkpoint=vae.resolve(),
            resolved_training=resolved,
            config=config,
            trainable_scope="full",
            pos_weight=None,
            supplied_training={"epochs": 2},
        )


def test_selected_refit_rejects_unmanaged_selection_copy(tmp_path):
    path, method, source, vae, config = _selection_fixture(tmp_path)
    unmanaged = tmp_path / "unmanaged-selection.json"
    unmanaged.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="finalized managed run"):
        boot_module._validate_refit_selection(
            unmanaged,
            model_name="efficientnet1dv2",
            method=method,
            source_checkpoint=source.resolve(),
            vae_checkpoint=vae.resolve(),
            resolved_training={
                "epochs": 1,
                "scheduler_horizon_epochs": 2,
                "batch_size": 4,
                "learning_rate": 5.0e-5,
                "weight_decay": 1.0e-4,
                "minimum_learning_rate_ratio": 0.01,
                "gradient_clip_norm": 1.0,
                "amp_enabled": True,
                "amp_dtype": "bfloat16",
            },
            config=config,
            trainable_scope="full",
            pos_weight=None,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"trainable_scope": "head"}, "trainable_scope=full"),
        ({"pos_weight": [1.0] * 5}, "pos_weight differs"),
    ],
)
def test_selected_refit_rejects_optimization_context_drift(
    tmp_path, override, message
):
    path, method, source, vae, config = _selection_fixture(tmp_path)
    resolved = {
        "epochs": 1,
        "scheduler_horizon_epochs": 2,
        "batch_size": 4,
        "learning_rate": 5.0e-5,
        "weight_decay": 1.0e-4,
        "minimum_learning_rate_ratio": 0.01,
        "gradient_clip_norm": 1.0,
        "amp_enabled": True,
        "amp_dtype": "bfloat16",
    }
    arguments = {
        "model_name": "efficientnet1dv2",
        "method": method,
        "source_checkpoint": source.resolve(),
        "vae_checkpoint": vae.resolve(),
        "resolved_training": resolved,
        "config": config,
        "trainable_scope": "full",
        "pos_weight": None,
        **override,
    }
    with pytest.raises(ValueError, match=message):
        boot_module._validate_refit_selection(path, **arguments)


def test_selected_refit_rejects_seed_group_or_replicate_drift(tmp_path):
    path, method, source, vae, config = _selection_fixture(tmp_path)
    config.payload["random_seed"]["replicate_id"] = 1
    with pytest.raises(ValueError, match="replicate_id"):
        boot_module._validate_refit_selection(
            path,
            model_name="efficientnet1dv2",
            method=method,
            source_checkpoint=source.resolve(),
            vae_checkpoint=vae.resolve(),
            resolved_training={
                "epochs": 1,
                "scheduler_horizon_epochs": 2,
                "batch_size": 4,
                "learning_rate": 5.0e-5,
                "weight_decay": 1.0e-4,
                "minimum_learning_rate_ratio": 0.01,
                "gradient_clip_norm": 1.0,
                "amp_enabled": True,
                "amp_dtype": "bfloat16",
            },
            config=config,
            trainable_scope="full",
            pos_weight=None,
        )


def test_selected_refit_rejects_scientific_online_config_drift(tmp_path):
    path, method, source, vae, config = _selection_fixture(tmp_path)
    config.payload["data"]["normalization_epsilon"] = 1.0e-5
    with pytest.raises(ValueError, match="scientific online-training config"):
        boot_module._validate_refit_selection(
            path,
            model_name="efficientnet1dv2",
            method=method,
            source_checkpoint=source.resolve(),
            vae_checkpoint=vae.resolve(),
            resolved_training={
                "epochs": 1,
                "scheduler_horizon_epochs": 2,
                "batch_size": 4,
                "learning_rate": 5.0e-5,
                "weight_decay": 1.0e-4,
                "minimum_learning_rate_ratio": 0.01,
                "gradient_clip_norm": 1.0,
                "amp_enabled": True,
                "amp_dtype": "bfloat16",
            },
            config=config,
            trainable_scope="full",
            pos_weight=None,
        )


def test_selected_refit_rejects_seed_namespace_or_determinism_drift(tmp_path):
    path, method, source, vae, config = _selection_fixture(tmp_path)
    config.payload["random_seed"]["stream_namespace"] = "drifted_stream"
    config.payload["random_seed"]["deterministic_algorithms"] = True
    with pytest.raises(ValueError, match="scientific online-training config"):
        boot_module._validate_refit_selection(
            path,
            model_name="efficientnet1dv2",
            method=method,
            source_checkpoint=source.resolve(),
            vae_checkpoint=vae.resolve(),
            resolved_training={
                "epochs": 1,
                "scheduler_horizon_epochs": 2,
                "batch_size": 4,
                "learning_rate": 5.0e-5,
                "weight_decay": 1.0e-4,
                "minimum_learning_rate_ratio": 0.01,
                "gradient_clip_norm": 1.0,
                "amp_enabled": True,
                "amp_dtype": "bfloat16",
            },
            config=config,
            trainable_scope="full",
            pos_weight=None,
        )


def test_selected_refit_rejects_incomplete_selection_rule(tmp_path):
    path, method, source, vae, config = _selection_fixture(tmp_path)
    selection = json.loads(path.read_text(encoding="utf-8"))
    selection["selection_rule"] = {"heldout_evaluation_used": False}
    path.write_text(json.dumps(selection), encoding="utf-8")
    run_root = path.parents[1]
    index_path = run_root / "run_file_index.json"
    index_path.write_text(
        json.dumps(build_run_file_index(run_root)), encoding="utf-8"
    )
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_file_index_sha256"] = sha256_file(index_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="selection rule differs"):
        boot_module._validate_refit_selection(
            path,
            model_name="efficientnet1dv2",
            method=method,
            source_checkpoint=source.resolve(),
            vae_checkpoint=vae.resolve(),
            resolved_training={
                "epochs": 1,
                "scheduler_horizon_epochs": 2,
                "batch_size": 4,
                "learning_rate": 5.0e-5,
                "weight_decay": 1.0e-4,
                "minimum_learning_rate_ratio": 0.01,
                "gradient_clip_norm": 1.0,
                "amp_enabled": True,
                "amp_dtype": "bfloat16",
            },
            config=config,
            trainable_scope="full",
            pos_weight=None,
        )


def test_selected_refit_rejects_seed_config_content_drift(tmp_path):
    path, method, source, vae, config = _selection_fixture(tmp_path)
    config.references["random_seed_config"].write_text(
        "schema_version: 1\nseed: 999\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="pooled tuning snapshot"):
        boot_module._validate_refit_selection(
            path,
            model_name="efficientnet1dv2",
            method=method,
            source_checkpoint=source.resolve(),
            vae_checkpoint=vae.resolve(),
            resolved_training={
                "epochs": 1,
                "scheduler_horizon_epochs": 2,
                "batch_size": 4,
                "learning_rate": 5.0e-5,
                "weight_decay": 1.0e-4,
                "minimum_learning_rate_ratio": 0.01,
                "gradient_clip_norm": 1.0,
                "amp_enabled": True,
                "amp_dtype": "bfloat16",
            },
            config=config,
            trainable_scope="full",
            pos_weight=None,
        )


def test_k500_adapter_requests_raw_100hz_runtime_data(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_get_dataloader(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(pn_module, "get_dataloader", fake_get_dataloader)
    result = build_pn2021_k500_dataloader(
        center="cpsc_2018",
        model_name="ecgfounder",
        shuffle=True,
        config_path=CONFIG,
        training_parameters={"batch_size": 7},
        dataloader_parameters={"num_workers": 0},
    )

    assert result is sentinel
    assert captured["dataset"] == "pn2021"
    assert captured["partition"] == "k500"
    assert captured["logical_center"] == "cpsc_2018"
    assert captured["sampling_rate_hz"] == 100
    assert captured["batch_size"] == 7
    assert captured["prepare_for_model"] is False
    assert captured["sanitize"] is False
    assert captured["global_zscore"] is False
    assert captured["output_layout"] == "time_channel"
    assert captured["shuffle"] is True
    assert captured["persistent_workers"] is False
    assert captured["selection_resident"] is True
    assert captured["selection_resident_pin_memory"] is False


def test_default_online_config_uses_canonical_profile():
    assert pn_module.DEFAULT_ONLINE_CONFIG_PATH == CONFIG


def test_k500_matched_seed_namespace_does_not_append_method_profile(monkeypatch):
    captured = {}

    def fake_get_dataloader(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(pn_module, "get_dataloader", fake_get_dataloader)
    build_pn2021_k500_dataloader(
        center="ningbo",
        model_name="efficientnet1dv2",
        shuffle=True,
        config_path=DIRECT_CONFIG,
        dataloader_parameters={"num_workers": 0},
    )

    namespace = captured["seed_namespace"]
    # The frozen comparison-group label is shared by Direct and latent arms;
    # the actual method profile must not be appended to the sampling stream.
    assert namespace.startswith(
        "pn2021_k500_matched:pn2021_direct_depth23_fixed20_family_balanced:0:"
    )
    assert "latent_threechain" not in namespace
    assert "lhat" not in namespace
    assert "augmix" not in namespace


def test_canonical_config_uses_single_process_selection_residency(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_get_dataloader(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(pn_module, "get_dataloader", fake_get_dataloader)
    result = build_pn2021_k500_dataloader(
        center="ningbo",
        model_name="efficientnet1dv2",
        shuffle=False,
        config_path=CONFIG,
    )

    assert result is sentinel
    assert captured["num_workers"] == 0
    assert captured["persistent_workers"] is False
    assert captured["cache_mode"] == "mmap"
    assert captured["selection_resident"] is True
    assert captured["selection_resident_pin_memory"] is False


def test_k500_adapter_rejects_drop_last(monkeypatch):
    monkeypatch.setattr(
        pn_module,
        "get_dataloader",
        lambda **kwargs: pytest.fail("runtime loader must not be built"),
    )
    with pytest.raises(ValueError, match="complete K500"):
        build_pn2021_k500_dataloader(
            center="ningbo",
            model_name="efficientnet1dv2",
            shuffle=True,
            config_path=CONFIG,
            dataloader_parameters={"num_workers": 0, "drop_last": True},
        )


def test_k500_selection_residency_rejects_worker_processes(monkeypatch):
    monkeypatch.setattr(
        pn_module,
        "get_dataloader",
        lambda **kwargs: pytest.fail("runtime loader must not be built"),
    )
    with pytest.raises(ValueError, match="selection-resident.*num_workers=0"):
        build_pn2021_k500_dataloader(
            center="ningbo",
            model_name="efficientnet1dv2",
            shuffle=True,
            config_path=CONFIG,
            dataloader_parameters={"num_workers": 1},
        )


def test_train_pn2021_routes_runtime_encoder_without_building_latent_pool(
    monkeypatch, tmp_path
):
    method = SimpleNamespace(
        executable=True,
        profile_name="latent_threechain_augmix",
        source_path=tmp_path / "method.yaml",
        requirements=SimpleNamespace(
            latent_pool=False,
            vae_encoder=True,
            vae_decoder=True,
        ),
    )
    config = SimpleNamespace(payload={"training": {"device": "cpu"}})
    monkeypatch.setattr(
        pn_module,
        "load_pn2021_method_profile",
        lambda *args, **kwargs: (method, config),
    )
    monkeypatch.setattr(
        pn_module,
        "validate_locked_source_checkpoint",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pn_module,
        "build_pn2021_latent_pool",
        lambda *args, **kwargs: pytest.fail("latent pool must not be built"),
    )

    class _Loader:
        closed = False

        def close(self):
            self.closed = True

    loader = _Loader()
    monkeypatch.setattr(
        pn_module,
        "build_pn2021_k500_dataloader",
        lambda **kwargs: loader,
    )
    captured = {}
    sentinel = object()

    def fake_train_online_model(model, train_loader, **kwargs):
        captured.update(kwargs)
        assert train_loader is loader
        return sentinel

    monkeypatch.setattr(pn_module, "train_online_model", fake_train_online_model)

    class _Model(torch.nn.Module):
        model_spec = EFFICIENTNET1DV2_SPEC

    encoder = torch.nn.Identity()
    decoder = torch.nn.Identity()
    result = train_pn2021(
        _Model(),
        center="ningbo",
        method_config_path=method.source_path,
        encoder=encoder,
        decoder=decoder,
        config_path=CONFIG,
        output_dir=tmp_path / "run",
        device="cpu",
    )

    assert result is sentinel
    assert captured["latent_pool"] is None
    assert captured["encoder"] is encoder
    assert captured["decoder"] is decoder
    assert loader.closed is True
