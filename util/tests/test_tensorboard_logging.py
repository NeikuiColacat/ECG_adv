from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from util.tensorboard_logging import build_tensorboard_monitor
from util.visualize_ecg import CANONICAL_LEAD_ORDER, create_ecg_figure


def _logging_config() -> dict[str, object]:
    return {
        "tensorboard": {
            "enabled": True,
            "subdir": "tensorboard",
            "max_queue": 2,
            "flush_secs": 1,
            "scalars": {
                "epoch_metrics": True,
                "batch_loss_interval_steps": 1,
            },
            "ecg_figures": {
                "enabled": True,
                "interval_epochs": 1,
                "max_samples": 1,
                "fixed_hash_ids": ["hash-a"],
                "views": [
                    "clean",
                    "lhat",
                    "lhat_difference",
                    "augmix",
                    "augmix_difference",
                ],
                "signal_domain": "raw_mv",
                "save_png": True,
                "save_raw_arrays": True,
                "raw_arrays_subdir": "samples",
            },
        }
    }


def test_create_ecg_figure_uses_ecgtwin_author_plot_contract(monkeypatch) -> None:
    import ecg_plot

    captured: dict[str, object] = {}

    def fake_plot(ecg, sample_rate, **kwargs):
        captured["ecg"] = np.asarray(ecg).copy()
        captured["sample_rate"] = sample_rate
        captured.update(kwargs)
        plt.subplots()

    monkeypatch.setattr(ecg_plot, "plot", fake_plot)
    waveform = np.arange(1000 * 12, dtype=np.float32).reshape(1000, 12)
    original = waveform.copy()

    figure = create_ecg_figure(
        waveform,
        sampling_rate_hz=100,
        layout="time_channel",
        title="probe",
    )

    assert np.asarray(captured["ecg"]).shape == (12, 1000)
    np.testing.assert_array_equal(np.asarray(captured["ecg"]), waveform.T)
    assert captured["sample_rate"] == 100.0
    assert tuple(captured["lead_index"]) == CANONICAL_LEAD_ORDER
    assert captured["columns"] == 1
    assert captured["row_height"] == 4
    np.testing.assert_array_equal(waveform, original)
    plt.close(figure)


def test_tensorboard_monitor_writes_scalars_ecg_png_arrays_and_manifest(
    tmp_path: Path,
) -> None:
    monitor = build_tensorboard_monitor(_logging_config(), tmp_path)
    clean = np.linspace(-1.0, 1.0, 1000 * 12, dtype=np.float32).reshape(1000, 12)
    lhat = clean + np.float32(0.1)
    augmix = clean * np.float32(0.9)

    monitor.log_train_step(
        loss=0.7,
        learning_rate=1.0e-3,
        global_step=1,
        loss_components={
            "clean": 0.5,
            "lhat": 0.6,
            "augmix": 0.65,
            "jsd": 0.02,
        },
        performance={
            "data_wait_ms": 1.5,
            "h2d_ms": 0.4,
            "augmentation_ms": 2.0,
            "forward_backward_ms": 4.5,
            "step_ms": 8.4,
            "samples_per_sec": 128.0,
        },
    )
    monitor.log_epoch(
        {
            "epoch": 1,
            "learning_rate": 1.0e-3,
            "train": {
                "loss": 0.6,
                "clean_bce": 0.5,
                "lhat_hard_bce": 0.55,
                "augmix_bce": 0.58,
                "augmix_jsd": 0.02,
            },
            "diagnostics": {"loss_gain": 0.1, "attack_success": 0.4},
            "performance": {"augmentation_ms": 2.0, "samples_per_sec": 128.0},
            "validation": {
                "loss": 0.5,
                "macro_auroc": 0.8,
                "macro_auprc": 0.7,
                "per_class_auroc": {"CD": 0.75},
                "per_class_auprc": {"CD": 0.65},
            },
        },
        is_selected=True,
    )
    logged = monitor.log_ecg_views(
        hash_id="hash-a",
        epoch=1,
        sampling_rate_hz=100,
        raw_views={"clean": clean, "lhat": lhat, "augmix": augmix},
        metadata={"record_id": "record-a", "seed_namespace": "probe-v1"},
        force=True,
    )
    monitor.log_test(
        {"loss": 0.4, "macro_auroc": 0.85, "macro_auprc": 0.72},
        global_step=1,
    )
    monitor.close()

    assert logged
    log_dir = tmp_path / "tensorboard"
    events = list(log_dir.glob("events.out.tfevents.*"))
    assert len(events) == 1
    sample_dir = log_dir / "samples" / "hash-a"
    assert (sample_dir / "clean_raw.npy").is_file()
    assert (sample_dir / "epoch_0001_lhat_raw.npy").is_file()
    assert (sample_dir / "epoch_0001_augmix_raw.npy").is_file()
    assert (sample_dir / "epoch_0001_clean.png").is_file()
    assert (sample_dir / "epoch_0001_lhat.png").is_file()
    assert (sample_dir / "epoch_0001_augmix.png").is_file()
    manifest = json.loads(
        (log_dir / "visualization_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["records"][0]["hash_id"] == "hash-a"
    assert manifest["records"][0]["sampling_rate_hz"] == 100
    assert manifest["records"][0]["physical_unit"] == "mV"

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    assert {
        "loss/train_step",
        "loss/train_step_components/total",
        "loss/train_step_components/clean",
        "loss/train_step_components/lhat",
        "loss/train_step_components/augmix",
        "loss/train_step_components/jsd",
        "perf/train_step/data_wait_ms",
        "perf/train_step/h2d_ms",
        "perf/train_step/augmentation_ms",
        "perf/train_step/forward_backward_ms",
        "perf/train_step/step_ms",
        "perf/train_step/samples_per_sec",
        "loss/train_epoch",
        "loss/train_epoch_components/clean_bce",
        "loss/train_epoch_components/lhat_hard_bce",
        "loss/train_epoch_components/augmix_bce",
        "loss/train_epoch_components/augmix_jsd",
        "diagnostics/train_epoch/loss_gain",
        "diagnostics/train_epoch/attack_success",
        "perf/train_epoch/augmentation_ms",
        "perf/train_epoch/samples_per_sec",
        "loss/validation",
        "metrics/validation_macro_auroc",
        "metrics/validation_macro_auprc",
        "optimizer/learning_rate",
        "selection/is_selected",
        "loss/test",
        "metrics/test_macro_auroc",
        "metrics/test_macro_auprc",
    }.issubset(scalar_tags)
    assert accumulator.Scalars("loss/train_step_components/total")[0].value == pytest.approx(
        0.7
    )
    assert (
        accumulator.Scalars("perf/train_step/samples_per_sec")[0].value
        == 128.0
    )
    assert "ecg/hash-a/clean" in set(accumulator.Tags()["images"])


def test_disabled_tensorboard_monitor_has_no_side_effects(tmp_path: Path) -> None:
    config = _logging_config()
    config["tensorboard"]["enabled"] = False

    monitor = build_tensorboard_monitor(config, tmp_path)
    monitor.log_train_step(loss=1.0, learning_rate=1.0e-3, global_step=1)
    monitor.close()

    assert monitor.describe()["enabled"] is False
    assert not (tmp_path / "tensorboard").exists()


def test_tensorboard_wildcard_logs_profile_owned_view_names(tmp_path: Path) -> None:
    config = _logging_config()
    config["tensorboard"]["ecg_figures"]["views"] = ["*"]
    monitor = build_tensorboard_monitor(config, tmp_path)
    clean = np.zeros((1000, 12), dtype=np.float32)
    corrupted = np.ones((1000, 12), dtype=np.float32)
    try:
        assert monitor.log_ecg_views(
            hash_id="hash-a",
            epoch=1,
            sampling_rate_hz=100,
            raw_views={"clean": clean, "corrupted": corrupted},
            force=True,
        )
    finally:
        monitor.close()

    manifest = json.loads(
        (tmp_path / "tensorboard" / "visualization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["records"][0]["views"] == [
        "clean",
        "corrupted",
        "corrupted_difference",
    ]


def test_tensorboard_probe_preflight_deduplicates_named_composition(tmp_path: Path) -> None:
    config = _logging_config()
    config["tensorboard"]["ecg_figures"]["views"] = ["*"]
    monitor = build_tensorboard_monitor(config, tmp_path)
    clean = np.zeros((1000, 12), dtype=np.float32)
    corrupted = np.ones((1000, 12), dtype=np.float32)
    try:
        assert monitor.should_log_ecg_views(
            hash_id="hash-a", epoch=1, probe_id="composition_00"
        )
        assert monitor.log_ecg_views(
            hash_id="hash-a",
            epoch=1,
            sampling_rate_hz=100,
            raw_views={"clean": clean, "corrupted": corrupted},
            probe_id="composition_00",
        )
        assert not monitor.should_log_ecg_views(
            hash_id="hash-a", epoch=1, probe_id="composition_00"
        )
        assert not monitor.log_ecg_views(
            hash_id="hash-a",
            epoch=1,
            sampling_rate_hz=100,
            raw_views={"clean": clean, "corrupted": corrupted},
            probe_id="composition_00",
        )
    finally:
        monitor.close()

    sample_dir = tmp_path / "tensorboard" / "samples" / "hash-a"
    assert (sample_dir / "epoch_0001_composition_00_corrupted.png").is_file()
    manifest = json.loads(
        (tmp_path / "tensorboard" / "visualization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["records"]) == 1
    assert manifest["records"][0]["probe_id"] == "composition_00"


def test_final_only_ecg_figures_batch_manifest_and_skip_event_images(
    tmp_path: Path,
) -> None:
    config = _logging_config()
    figures = config["tensorboard"]["ecg_figures"]
    figures["epoch_policy"] = "final_only"
    figures["tensorboard_image"] = False
    figures["views"] = ["clean"]
    monitor = build_tensorboard_monitor(config, tmp_path)
    clean = np.zeros((1000, 12), dtype=np.float32)
    manifest_path = tmp_path / "tensorboard" / "visualization_manifest.json"
    try:
        assert not monitor.should_log_ecg_views(
            hash_id="hash-a",
            epoch=1,
            is_final_epoch=False,
        )
        assert monitor.log_ecg_views(
            hash_id="hash-a",
            epoch=2,
            sampling_rate_hz=100,
            raw_views={"clean": clean},
            is_final_epoch=True,
        )
        assert not manifest_path.exists()
        monitor.flush()
        assert manifest_path.is_file()
    finally:
        monitor.close()

    sample_dir = tmp_path / "tensorboard" / "samples" / "hash-a"
    assert (sample_dir / "epoch_0002_clean.png").is_file()
    assert (sample_dir / "clean_raw.npy").is_file()
    accumulator = EventAccumulator(str(tmp_path / "tensorboard"))
    accumulator.Reload()
    assert accumulator.Tags()["images"] == []


def test_train_step_legacy_call_remains_compatible(tmp_path: Path) -> None:
    monitor = build_tensorboard_monitor(_logging_config(), tmp_path)
    monitor.log_train_step(loss=0.7, learning_rate=1.0e-3, global_step=1)
    monitor.close()

    accumulator = EventAccumulator(str(tmp_path / "tensorboard"))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    assert "loss/train_step" in scalar_tags
    assert "loss/train_step_components/total" in scalar_tags


@pytest.mark.parametrize(
    ("keyword", "payload", "error_type"),
    [
        ("loss_components", {"": 0.1}, ValueError),
        ("loss_components", {1: 0.1}, TypeError),
        ("loss_components", {"clean": float("nan")}, ValueError),
        ("performance", {"": 1.0}, ValueError),
        ("performance", {1: 1.0}, TypeError),
        ("performance", {"step_ms": float("inf")}, ValueError),
    ],
)
def test_train_step_scalar_mappings_require_string_keys_and_finite_values(
    tmp_path: Path,
    keyword: str,
    payload: dict[object, float],
    error_type: type[Exception],
) -> None:
    monitor = build_tensorboard_monitor(_logging_config(), tmp_path)
    try:
        with pytest.raises(error_type):
            monitor.log_train_step(
                loss=0.7,
                learning_rate=1.0e-3,
                global_step=1,
                **{keyword: payload},
            )
    finally:
        monitor.close()
