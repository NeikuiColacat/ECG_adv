"""TensorBoard observer for supervised and online ECG training.

Training JSON, checkpoints, and metric artifacts remain the experiment source
of truth.  This module is intentionally a thin, optional observer with a lazy
TensorBoard import and no ownership of model, optimizer, data, AugMix, or LHAT
logic.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from util.visualize_ecg import CANONICAL_LEAD_ORDER, create_ecg_figure


# ``*`` means "all views supplied by the method executor".  Concrete view
# names are method-profile owned; the observer must not encode a fixed
# clean/LHAT/AugMix topology.
SUPPORTED_ECG_VIEWS = ("*",)


def _require_mapping(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    expected: set[str],
    description: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{description} keys mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _positive_int(value: Any, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return int(value)


def _boolean(value: Any, *, description: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{description} must be boolean")
    return value


def _relative_member(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{description} must stay inside the run directory")
    return value


@dataclass(frozen=True)
class ECGFigureLoggingConfig:
    enabled: bool
    interval_epochs: int
    epoch_policy: str
    max_samples: int
    fixed_hash_ids: tuple[str, ...]
    views: tuple[str, ...]
    signal_domain: str
    save_png: bool
    tensorboard_image: bool
    save_raw_arrays: bool
    raw_arrays_subdir: str

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_epochs": self.interval_epochs,
            "epoch_policy": self.epoch_policy,
            "max_samples": self.max_samples,
            "fixed_hash_ids": list(self.fixed_hash_ids),
            "views": list(self.views),
            "signal_domain": self.signal_domain,
            "save_png": self.save_png,
            "tensorboard_image": self.tensorboard_image,
            "save_raw_arrays": self.save_raw_arrays,
            "raw_arrays_subdir": self.raw_arrays_subdir,
        }


@dataclass(frozen=True)
class TensorBoardLoggingConfig:
    enabled: bool
    subdir: str
    max_queue: int
    flush_secs: int
    epoch_metrics: bool
    batch_loss_interval_steps: int
    ecg_figures: ECGFigureLoggingConfig

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "subdir": self.subdir,
            "max_queue": self.max_queue,
            "flush_secs": self.flush_secs,
            "scalars": {
                "epoch_metrics": self.epoch_metrics,
                "batch_loss_interval_steps": self.batch_loss_interval_steps,
            },
            "ecg_figures": self.ecg_figures.describe(),
        }


def parse_tensorboard_logging_config(
    logging_section: Mapping[str, Any],
) -> TensorBoardLoggingConfig:
    """Strictly validate the top-level training ``logging`` section."""

    logging = _require_mapping(logging_section, description="logging")
    _require_exact_keys(logging, expected={"tensorboard"}, description="logging")
    tensorboard = _require_mapping(
        logging["tensorboard"], description="logging.tensorboard"
    )
    _require_exact_keys(
        tensorboard,
        expected={
            "enabled",
            "subdir",
            "max_queue",
            "flush_secs",
            "scalars",
            "ecg_figures",
        },
        description="logging.tensorboard",
    )
    scalars = _require_mapping(
        tensorboard["scalars"], description="logging.tensorboard.scalars"
    )
    _require_exact_keys(
        scalars,
        expected={"epoch_metrics", "batch_loss_interval_steps"},
        description="logging.tensorboard.scalars",
    )
    figures = _require_mapping(
        tensorboard["ecg_figures"],
        description="logging.tensorboard.ecg_figures",
    )
    required_figure_keys = {
        "enabled",
        "interval_epochs",
        "max_samples",
        "fixed_hash_ids",
        "views",
        "signal_domain",
        "save_png",
        "save_raw_arrays",
        "raw_arrays_subdir",
    }
    optional_figure_keys = {"epoch_policy", "tensorboard_image"}
    missing_figure_keys = required_figure_keys - set(figures)
    unexpected_figure_keys = set(figures) - required_figure_keys - optional_figure_keys
    if missing_figure_keys or unexpected_figure_keys:
        raise ValueError(
            "logging.tensorboard.ecg_figures keys mismatch: "
            f"missing={sorted(missing_figure_keys)}, "
            f"unexpected={sorted(unexpected_figure_keys)}"
        )
    raw_hashes = figures["fixed_hash_ids"]
    if not isinstance(raw_hashes, list) or any(
        not isinstance(value, str) or not value for value in raw_hashes
    ):
        raise ValueError("ecg_figures.fixed_hash_ids must be a string list")
    fixed_hash_ids = tuple(raw_hashes)
    if len(set(fixed_hash_ids)) != len(fixed_hash_ids):
        raise ValueError("ecg_figures.fixed_hash_ids contains duplicates")
    max_samples = _positive_int(
        figures["max_samples"], description="ecg_figures.max_samples"
    )
    if len(fixed_hash_ids) > max_samples:
        raise ValueError("fixed_hash_ids cannot exceed ecg_figures.max_samples")
    raw_views = figures["views"]
    if not isinstance(raw_views, list) or any(
        not isinstance(value, str) for value in raw_views
    ):
        raise ValueError("ecg_figures.views must be a string list")
    views = tuple(raw_views)
    if not views or len(set(views)) != len(views):
        raise ValueError("ecg_figures.views must be non-empty and unique")
    if "*" in views and views != ("*",):
        raise ValueError("ecg_figures.views '*' must be used alone")
    for view in views:
        if view != "*":
            _safe_tag_component(view)
    if figures["signal_domain"] != "raw_mv":
        raise ValueError("ecg_figures.signal_domain must be raw_mv")
    epoch_policy = str(figures.get("epoch_policy", "interval"))
    if epoch_policy not in {"interval", "final_only"}:
        raise ValueError("ecg_figures.epoch_policy must be interval or final_only")
    return TensorBoardLoggingConfig(
        enabled=_boolean(
            tensorboard["enabled"], description="logging.tensorboard.enabled"
        ),
        subdir=_relative_member(
            tensorboard["subdir"], description="logging.tensorboard.subdir"
        ),
        max_queue=_positive_int(
            tensorboard["max_queue"], description="logging.tensorboard.max_queue"
        ),
        flush_secs=_positive_int(
            tensorboard["flush_secs"], description="logging.tensorboard.flush_secs"
        ),
        epoch_metrics=_boolean(
            scalars["epoch_metrics"], description="scalars.epoch_metrics"
        ),
        batch_loss_interval_steps=_positive_int(
            scalars["batch_loss_interval_steps"],
            description="scalars.batch_loss_interval_steps",
        ),
        ecg_figures=ECGFigureLoggingConfig(
            enabled=_boolean(
                figures["enabled"], description="ecg_figures.enabled"
            ),
            interval_epochs=_positive_int(
                figures["interval_epochs"],
                description="ecg_figures.interval_epochs",
            ),
            epoch_policy=epoch_policy,
            max_samples=max_samples,
            fixed_hash_ids=fixed_hash_ids,
            views=views,
            signal_domain="raw_mv",
            save_png=_boolean(
                figures["save_png"], description="ecg_figures.save_png"
            ),
            tensorboard_image=_boolean(
                figures.get("tensorboard_image", True),
                description="ecg_figures.tensorboard_image",
            ),
            save_raw_arrays=_boolean(
                figures["save_raw_arrays"],
                description="ecg_figures.save_raw_arrays",
            ),
            raw_arrays_subdir=_relative_member(
                figures["raw_arrays_subdir"],
                description="ecg_figures.raw_arrays_subdir",
            ),
        ),
    )


def _run_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("TensorBoard path escapes the run directory") from None
    return path


def _finite_scalar(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _validated_scalar_mapping(
    value: Mapping[str, Any] | None,
    *,
    description: str,
) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    result: dict[str, float] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{description} keys must be strings")
        if not key.strip():
            raise ValueError(f"{description} keys must be non-empty strings")
        safe_key = _safe_tag_component(key)
        if safe_key in result:
            raise ValueError(
                f"{description} keys collide after TensorBoard tag sanitization: "
                f"{safe_key}"
            )
        result[safe_key] = _finite_scalar(
            raw_value,
            description=f"{description}.{key}",
        )
    return result


def _safe_tag_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    if not sanitized:
        raise ValueError("TensorBoard tag component is empty after sanitization")
    return sanitized


def _raw_time_channel(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != len(CANONICAL_LEAD_ORDER):
        raise ValueError("raw ECG view must have shape (time,12)")
    result = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("raw ECG view contains NaN or Inf")
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class TensorBoardMonitor:
    """Optional TensorBoard event sink with ECG probe artifact support."""

    def __init__(self, config: TensorBoardLoggingConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir.expanduser().resolve()
        self.log_dir = _run_member(self.run_dir, config.subdir)
        self._writer: Any | None = None
        self._closed = False
        self._logged_hash_ids: set[str] = set()
        self._logged_probe_keys: set[tuple[int, str, str]] = set()
        self._manifest_records: list[dict[str, Any]] = []
        self._manifest_dirty = False
        if config.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "TensorBoard logging is enabled but tensorboard is unavailable"
                ) from exc
            self.log_dir.mkdir(parents=True, exist_ok=False)
            self._writer = SummaryWriter(
                log_dir=str(self.log_dir),
                max_queue=config.max_queue,
                flush_secs=config.flush_secs,
            )

    @property
    def enabled(self) -> bool:
        return self._writer is not None and not self._closed

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "torch.utils.tensorboard.SummaryWriter",
            "enabled": self.config.enabled,
            "log_dir": str(self.log_dir) if self.config.enabled else None,
            "config": self.config.describe(),
            "torch_version": str(torch.__version__),
        }

    def log_train_step(
        self,
        *,
        loss: float,
        learning_rate: float,
        global_step: int,
        loss_components: Mapping[str, Any] | None = None,
        performance: Mapping[str, Any] | None = None,
    ) -> None:
        """Write one sampled train-step record.

        ``loss`` remains the canonical total loss and keeps the legacy
        ``loss/train_step`` tag. Optional mappings let supervised, LHAT, and
        AugMix callers expose method-specific components and measured timing or
        throughput without coupling this observer to one training algorithm.
        """

        if not self.enabled:
            return
        if isinstance(global_step, bool) or int(global_step) <= 0:
            raise ValueError("global_step must be a positive integer")
        step = int(global_step)
        if step % self.config.batch_loss_interval_steps != 0:
            return
        loss_value = _finite_scalar(loss, description="training step loss")
        learning_rate_value = _finite_scalar(
            learning_rate,
            description="learning rate",
        )
        components = _validated_scalar_mapping(
            loss_components,
            description="training step loss_components",
        )
        if "total" in components and not math.isclose(
            components["total"],
            loss_value,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise ValueError("loss_components.total must match the canonical loss")
        performance_values = _validated_scalar_mapping(
            performance,
            description="training step performance",
        )
        self._writer.add_scalar(
            "loss/train_step",
            loss_value,
            step,
        )
        self._writer.add_scalar(
            "loss/train_step_components/total",
            loss_value,
            step,
        )
        for name, value in components.items():
            if name == "total":
                continue
            self._writer.add_scalar(
                f"loss/train_step_components/{name}",
                value,
                step,
            )
        for name, value in performance_values.items():
            self._writer.add_scalar(
                f"perf/train_step/{name}",
                value,
                step,
            )
        self._writer.add_scalar(
            "optimizer/learning_rate_step",
            learning_rate_value,
            step,
        )

    def _log_metric_group(
        self,
        *,
        prefix: str,
        metrics: Mapping[str, Any],
        step: int,
    ) -> None:
        for key in ("loss", "macro_auroc", "macro_auprc"):
            value = metrics.get(key)
            if value is None:
                continue
            tag = f"loss/{prefix}" if key == "loss" else f"metrics/{prefix}_{key}"
            self._writer.add_scalar(
                tag,
                _finite_scalar(value, description=f"{prefix} {key}"),
                step,
            )
        for metric_key, tag_suffix in (
            ("per_class_auroc", "auroc"),
            ("per_class_auprc", "auprc"),
        ):
            values = metrics.get(metric_key)
            if values is None:
                continue
            if not isinstance(values, Mapping):
                raise TypeError(f"{prefix} {metric_key} must be a mapping")
            for class_name, value in values.items():
                if value is None:
                    continue
                self._writer.add_scalar(
                    f"metrics/{prefix}_{tag_suffix}/{_safe_tag_component(str(class_name))}",
                    _finite_scalar(value, description=f"{prefix} {metric_key}"),
                    step,
                )

    def log_epoch(self, record: Mapping[str, Any], *, is_selected: bool) -> None:
        if not self.enabled or not self.config.epoch_metrics:
            return
        epoch = int(record["epoch"])
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        train = _require_mapping(record.get("train"), description="epoch train metrics")
        self._writer.add_scalar(
            "loss/train_epoch",
            _finite_scalar(train["loss"], description="training epoch loss"),
            epoch,
        )
        for name in (
            "clean_bce",
            "lhat_hard_bce",
            "augmix_bce",
            "augmix_jsd",
            "weighted_clean_contribution",
            "weighted_lhat_contribution",
            "weighted_augmix_contribution",
            "weighted_jsd_contribution",
        ):
            if name in train:
                self._writer.add_scalar(
                    f"loss/train_epoch_components/{name}",
                    _finite_scalar(train[name], description=f"training epoch {name}"),
                    epoch,
                )
        for group_name in ("objective_terms", "weighted_objective_terms"):
            raw_group = train.get(group_name)
            if raw_group is None:
                continue
            group = _validated_scalar_mapping(
                _require_mapping(
                    raw_group,
                    description=f"training epoch {group_name}",
                ),
                description=f"training epoch {group_name}",
            )
            for name, value in group.items():
                self._writer.add_scalar(
                    f"loss/train_epoch_{group_name}/{name}",
                    value,
                    epoch,
                )
        for group_name, tag_root in (
            ("diagnostics", "diagnostics/train_epoch"),
            ("performance", "perf/train_epoch"),
        ):
            raw_group = record.get(group_name)
            if raw_group is None:
                continue
            group = _validated_scalar_mapping(
                _require_mapping(raw_group, description=f"epoch {group_name}"),
                description=f"epoch {group_name}",
            )
            for name, value in group.items():
                self._writer.add_scalar(f"{tag_root}/{name}", value, epoch)
        self._writer.add_scalar(
            "optimizer/learning_rate",
            _finite_scalar(record["learning_rate"], description="learning rate"),
            epoch,
        )
        validation = record.get("validation")
        if validation is not None:
            self._log_metric_group(
                prefix="validation",
                metrics=_require_mapping(
                    validation, description="epoch validation metrics"
                ),
                step=epoch,
            )
        self._writer.add_scalar(
            "selection/is_selected", 1.0 if is_selected else 0.0, epoch
        )
        self.flush()

    def log_test(self, metrics: Mapping[str, Any], *, global_step: int) -> None:
        if not self.enabled:
            return
        self._log_metric_group(prefix="test", metrics=metrics, step=int(global_step))
        self.flush()

    def _should_log_hash(self, hash_id: str) -> bool:
        configured = self.config.ecg_figures.fixed_hash_ids
        if configured:
            return hash_id in configured
        if hash_id in self._logged_hash_ids:
            return True
        return len(self._logged_hash_ids) < self.config.ecg_figures.max_samples

    def should_log_ecg_views(
        self,
        *,
        hash_id: str,
        epoch: int,
        probe_id: str = "default",
        is_final_epoch: bool = False,
        force: bool = False,
    ) -> bool:
        """Cheap preflight used before a caller synchronizes GPU probe tensors."""

        figures = self.config.ecg_figures
        if not self.enabled or not figures.enabled:
            return False
        if not isinstance(hash_id, str) or not hash_id:
            raise ValueError("hash_id must be a non-empty string")
        epoch = _positive_int(epoch, description="ECG figure epoch")
        safe_probe = _safe_tag_component(probe_id)
        if safe_probe != probe_id:
            raise ValueError(f"probe_id is not TensorBoard-safe: {probe_id!r}")
        if not isinstance(is_final_epoch, bool):
            raise TypeError("is_final_epoch must be boolean")
        if not force:
            if figures.epoch_policy == "final_only":
                if not is_final_epoch:
                    return False
            elif epoch != 1 and epoch % figures.interval_epochs != 0:
                return False
        if (epoch, hash_id, probe_id) in self._logged_probe_keys:
            return False
        return self._should_log_hash(hash_id)

    def log_ecg_views(
        self,
        *,
        hash_id: str,
        epoch: int,
        sampling_rate_hz: int,
        raw_views: Mapping[str, np.ndarray | torch.Tensor],
        metadata: Mapping[str, Any] | None = None,
        probe_id: str = "default",
        is_final_epoch: bool = False,
        force: bool = False,
    ) -> bool:
        """Log fixed-hash raw-mV views supplied by a method executor.

        This method never generates a corruption and therefore never consumes
        the training RNG.  The online training caller owns probe hash and RNG
        selection and passes already-generated waveforms here.
        """

        figures = self.config.ecg_figures
        if not self.should_log_ecg_views(
            hash_id=hash_id,
            epoch=epoch,
            probe_id=probe_id,
            is_final_epoch=is_final_epoch,
            force=force,
        ):
            return False
        epoch = _positive_int(epoch, description="ECG figure epoch")
        sampling_rate_hz = _positive_int(
            sampling_rate_hz, description="ECG sampling_rate_hz"
        )
        safe_probe = _safe_tag_component(probe_id)

        if not isinstance(raw_views, Mapping) or "clean" not in raw_views:
            raise ValueError("raw_views must be a mapping containing 'clean'")
        converted: dict[str, np.ndarray] = {}
        for raw_name, raw_view in raw_views.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError("raw view names must be non-empty strings")
            name = _safe_tag_component(raw_name)
            if name != raw_name:
                raise ValueError(
                    f"raw view name is not TensorBoard-safe: {raw_name!r}"
                )
            if name.endswith("_difference"):
                raise ValueError(
                    "raw view names may not use the reserved _difference suffix"
                )
            if name in converted:
                raise ValueError(f"duplicate raw view name: {name}")
            converted[name] = _raw_time_channel(raw_view)
        clean = converted["clean"]
        for name, view in converted.items():
            if view.shape != clean.shape:
                raise ValueError(
                    f"{name} and clean ECG shapes differ: {view.shape} != {clean.shape}"
                )
        if clean.shape[0] != sampling_rate_hz * 10:
            raise ValueError(
                "ECG probe must be exactly ten seconds: "
                f"samples={clean.shape[0]}, sampling_rate_hz={sampling_rate_hz}"
            )

        available: dict[str, np.ndarray] = {"clean": clean}
        for name, view in converted.items():
            if name == "clean":
                continue
            available[name] = view
            available[f"{name}_difference"] = view - clean
        requested = tuple(available) if figures.views == ("*",) else figures.views
        requested_missing = [view for view in requested if view not in available]
        if requested_missing:
            raise ValueError(
                f"configured ECG views are unavailable: {requested_missing}"
            )

        safe_hash = _safe_tag_component(hash_id)
        sample_dir = _run_member(
            self.log_dir,
            f"{figures.raw_arrays_subdir}/{safe_hash}",
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        emitted: list[str] = []
        for view_name in requested:
            waveform = available[view_name]
            title = (
                f"{view_name} | hash={hash_id} | {sampling_rate_hz} Hz | raw mV"
            )
            file_prefix = (
                f"epoch_{epoch:04d}"
                if probe_id == "default"
                else f"epoch_{epoch:04d}_{safe_probe}"
            )
            if figures.save_png or figures.tensorboard_image:
                figure = create_ecg_figure(
                    waveform,
                    sampling_rate_hz=sampling_rate_hz,
                    layout="time_channel",
                    lead_order=CANONICAL_LEAD_ORDER,
                    title=title,
                )
                if figures.save_png:
                    figure.savefig(
                        sample_dir / f"{file_prefix}_{view_name}.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                if figures.tensorboard_image:
                    self._writer.add_figure(
                        (
                            f"ecg/{safe_hash}/{view_name}"
                            if probe_id == "default"
                            else f"ecg/{safe_hash}/{safe_probe}/{view_name}"
                        ),
                        figure,
                        global_step=epoch,
                        close=True,
                    )
                else:
                    from matplotlib import pyplot as plt

                    plt.close(figure)
            if figures.save_raw_arrays:
                array_path = (
                    sample_dir / "clean_raw.npy"
                    if view_name == "clean" and probe_id == "default"
                    else sample_dir / f"{file_prefix}_{view_name}_raw.npy"
                )
                if view_name != "clean" or not array_path.exists():
                    np.save(array_path, waveform, allow_pickle=False)
            emitted.append(view_name)

        self._logged_hash_ids.add(hash_id)
        self._logged_probe_keys.add((epoch, hash_id, probe_id))
        record = {
            "hash_id": hash_id,
            "epoch": epoch,
            "sampling_rate_hz": sampling_rate_hz,
            "duration_seconds": 10,
            "lead_order": list(CANONICAL_LEAD_ORDER),
            "layout": "time_channel",
            "physical_unit": "mV",
            "normalization": "none",
            "probe_id": probe_id,
            "views": emitted,
            "metadata": {} if metadata is None else dict(metadata),
        }
        self._manifest_records.append(record)
        self._manifest_dirty = True
        return True

    def flush(self) -> None:
        if self.enabled:
            if self._manifest_dirty:
                _write_json(
                    self.log_dir / "visualization_manifest.json",
                    {"schema_version": 1, "records": self._manifest_records},
                )
                self._manifest_dirty = False
            self._writer.flush()

    def close(self) -> None:
        if self._closed:
            return
        if self._writer is not None:
            self.flush()
            self._writer.close()
        self._closed = True

    def __enter__(self) -> TensorBoardMonitor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def build_tensorboard_monitor(
    logging_section: Mapping[str, Any],
    run_dir: str | Path,
) -> TensorBoardMonitor:
    return TensorBoardMonitor(
        parse_tensorboard_logging_config(logging_section),
        Path(run_dir),
    )


__all__ = [
    "ECGFigureLoggingConfig",
    "SUPPORTED_ECG_VIEWS",
    "TensorBoardLoggingConfig",
    "TensorBoardMonitor",
    "build_tensorboard_monitor",
    "parse_tensorboard_logging_config",
]
