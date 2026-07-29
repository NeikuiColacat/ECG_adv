"""Process-local PTB-XL replay bridge for matched target adaptation arms.

The bridge is activated only by an allowlisted online-training profile.  It
adds one deterministic PTB-XL BCE term to the clean exposure of every target
base batch.  Direct and VAE/AugMix arms therefore see the exact same source
stream and source-loss weight.  Source forwards restore BatchNorm buffers and
the process RNG state, so only parameter gradients carry source information.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.methods import CompiledMethod
from data_preprocess.data_runtime import get_dataloader
from models.contracts import validate_model_output
from models.input_adapter import prepare_canonical_model_input


PROFILE_TOTAL_WEIGHTS = {
    "pn2021_fixed20_source_replay_w010_sandbox": 0.10,
}
ALLOWED_METHOD_IDS = frozenset(
    {
        "direct_depth23_fixed20",
        "fixed20_pcgrad_d13_aux_a050_v1",
        "fixed20_search_d19_v1",
    }
)
CLEAN_FAMILY_WEIGHT = 0.5
SOURCE_PARTITION = "train"


@dataclass
class _SourceReplayState:
    active: bool = False
    loader: Any | None = None
    iterator: Iterator[Any] | None = None
    profile_name: str | None = None
    total_weight: float | None = None
    method_id: str | None = None
    center: str | None = None
    model_name: str | None = None
    source_batches: int = 0
    source_samples: int = 0
    source_loss_sum: float = 0.0
    source_hash_digests: list[str] = field(default_factory=list)
    loader_identity: Mapping[str, Any] | None = None

    def reset(self) -> None:
        self.active = False
        self.loader = None
        self.iterator = None
        self.profile_name = None
        self.total_weight = None
        self.method_id = None
        self.center = None
        self.model_name = None
        self.source_batches = 0
        self.source_samples = 0
        self.source_loss_sum = 0.0
        self.source_hash_digests.clear()
        self.loader_identity = None


_STATE = _SourceReplayState()


def _disable_bn_running_stats(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Use batch statistics without mutating buffers needed by later backward.

    Restoring running_mean/running_var tensors immediately after a forward
    changes their autograd version counters.  Temporarily disabling tracking
    prevents those writes in the first place and is safe to undo before
    backward because only the boolean module attribute changes.
    """

    captured: list[tuple[Any, ...]] = []
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        captured.append((module, bool(module.track_running_stats)))
        module.track_running_stats = False
    return tuple(captured)


def _restore_bn_tracking(captured: Sequence[tuple[Any, ...]]) -> None:
    for module, track_running_stats in captured:
        module.track_running_stats = bool(track_running_stats)


def _capture_rng_state(model: nn.Module):
    cpu = torch.random.get_rng_state()
    device = next(model.parameters()).device
    cuda = None
    if device.type == "cuda":
        cuda = torch.cuda.get_rng_state(device)
    return cpu, device, cuda


def _restore_rng_state(state: tuple[torch.Tensor, torch.device, torch.Tensor | None]):
    cpu, device, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, device)


def _method_id(method: Any) -> str | None:
    return method.profile_name if isinstance(method, CompiledMethod) else None


def _is_clean_objective(names: Any) -> bool:
    if names is None:
        return False
    selected = tuple(str(value) for value in names)
    return bool(selected) and selected[0] == "clean_bce" and (
        "corrupted_bce" not in selected
    )


def _next_source_batch() -> Mapping[str, Any]:
    if _STATE.loader is None:
        raise RuntimeError("source replay loader is not initialized")
    if _STATE.iterator is None:
        _STATE.iterator = iter(_STATE.loader)
    try:
        batch = next(_STATE.iterator)
    except StopIteration:
        _STATE.iterator = iter(_STATE.loader)
        batch = next(_STATE.iterator)
    if not isinstance(batch, Mapping):
        raise TypeError("source replay batch must be a mapping")
    return batch


def _batch_hash_digest(batch: Mapping[str, Any], batch_size: int) -> str:
    raw = batch.get("hash_id")
    if raw is None:
        values = tuple(f"index={index}" for index in range(batch_size))
    elif isinstance(raw, (list, tuple)):
        values = tuple(str(value) for value in raw)
    else:
        try:
            values = tuple(str(value) for value in raw)
        except TypeError:
            values = (str(raw),)
    if len(values) != batch_size:
        raise ValueError("source replay hash IDs do not align to the batch")
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _source_loss(kwargs: Mapping[str, Any]) -> torch.Tensor:
    model = kwargs.get("model")
    spec = kwargs.get("spec")
    epsilon = kwargs.get("normalization_epsilon")
    pos_weight = kwargs.get("pos_weight")
    if not isinstance(model, nn.Module):
        raise TypeError("source replay objective requires a torch model")
    if spec is None:
        raise TypeError("source replay objective requires a model spec")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("source replay normalization epsilon must be numeric")
    batch = _next_source_batch()
    raw = batch.get("waveform")
    targets = batch.get("label")
    if not isinstance(raw, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("source replay waveform and label must be tensors")
    if tuple(raw.shape[1:]) != (1000, 12):
        raise ValueError("source replay waveform must have shape (B,1000,12)")
    batch_size = int(raw.shape[0])
    if targets.shape != (batch_size, 5):
        raise ValueError("source replay labels must have shape (B,5)")
    device = next(model.parameters()).device
    raw = raw.to(device=device, dtype=torch.float32, non_blocking=True)
    targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
    if not bool(torch.isfinite(raw).all().item()):
        raise FloatingPointError("source replay waveform contains NaN or Inf")
    if not bool(torch.isfinite(targets).all().item()):
        raise FloatingPointError("source replay labels contain NaN or Inf")

    rng_state = _capture_rng_state(model)
    bn_state = _disable_bn_running_stats(model)
    try:
        model_input = prepare_canonical_model_input(
            raw,
            spec,
            epsilon=float(epsilon),
        )
        logits = validate_model_output(
            model(model_input),
            spec,
            batch_size=batch_size,
            check_finite=False,
        )
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )
    finally:
        _restore_bn_tracking(bn_state)
        _restore_rng_state(rng_state)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("source replay BCE became NaN or Inf")
    _STATE.source_batches += 1
    _STATE.source_samples += batch_size
    _STATE.source_loss_sum += float(loss.detach().float().cpu())
    _STATE.source_hash_digests.append(_batch_hash_digest(batch, batch_size))
    return loss


def diagnostics_payload() -> dict[str, Any] | None:
    if _STATE.profile_name is None:
        return None
    mean_loss = (
        _STATE.source_loss_sum / _STATE.source_batches
        if _STATE.source_batches
        else 0.0
    )
    return {
        "schema_version": 1,
        "profile_name": _STATE.profile_name,
        "method_id": _STATE.method_id,
        "center": _STATE.center,
        "model_name": _STATE.model_name,
        "source_partition": SOURCE_PARTITION,
        "source_total_loss_weight": _STATE.total_weight,
        "clean_family_weight": CLEAN_FAMILY_WEIGHT,
        "source_batches": _STATE.source_batches,
        "source_samples": _STATE.source_samples,
        "source_bce_mean": mean_loss,
        "ordered_source_batch_hashes_sha256": hashlib.sha256(
            "\n".join(_STATE.source_hash_digests).encode("utf-8")
        ).hexdigest(),
        "batch_norm_policy": "disable_running_stat_tracking_for_source_forward",
        "rng_policy": "snapshot_restore_source_forward",
        "loader": dict(_STATE.loader_identity or {}),
    }


def write_diagnostics(output_dir: str | Path) -> Path | None:
    payload = diagnostics_payload()
    if payload is None:
        return None
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "source_replay_diagnostics.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


@contextmanager
def patch_online_trainer_runtime() -> Iterator[None]:
    """Install the allowlisted source replay adapter for one delegate run."""

    import core.online_trainer as online_trainer
    import core.pn2021_tuning as pn2021_tuning
    import core.train_PN2021 as train_pn2021

    original_train = online_trainer.train_online_model
    original_tuning_alias = pn2021_tuning.train_online_model
    original_train_alias = train_pn2021.train_online_model
    original_objective = online_trainer._compute_objective
    _STATE.reset()

    def patched_objective(**kwargs: Any):
        result = original_objective(**kwargs)
        if not _STATE.active:
            return result
        method_id = _method_id(kwargs.get("method"))
        if method_id != _STATE.method_id or not _is_clean_objective(
            kwargs.get("objective_term_names")
        ):
            return result
        if _STATE.total_weight is None or not math.isfinite(_STATE.total_weight):
            raise RuntimeError("source replay total weight is unavailable")
        source_bce = _source_loss(kwargs)
        # The whitelist trainer multiplies the clean exposure by 0.5.  Divide
        # here so the final contribution equals the declared total weight.
        source_scale = _STATE.total_weight / CLEAN_FAMILY_WEIGHT
        result_type = type(result)
        return result_type(
            total=result.total + source_bce * source_scale,
            raw_terms=dict(result.raw_terms),
            weighted_terms=dict(result.weighted_terms),
            valid_counts=dict(result.valid_counts),
        )

    def patched_train(*args: Any, **kwargs: Any):
        config_path = kwargs.get("config_path", online_trainer.DEFAULT_ONLINE_CONFIG_PATH)
        config_root = kwargs.get("config_root")
        config = online_trainer.load_online_train_config(
            config_path,
            config_root=config_root,
        )
        profile_name = str(config.payload["profile_name"])
        if profile_name not in PROFILE_TOTAL_WEIGHTS:
            return original_train(*args, **kwargs)
        if _STATE.active:
            raise RuntimeError("nested source replay runs are not supported")
        method_path = Path(str(kwargs.get("method_config_path"))).expanduser()
        if not method_path.is_absolute():
            method_path = config.config_root / method_path
        method = online_trainer.compile_method_profile(method_path.resolve())
        method_id = method.profile_name
        if method_id not in ALLOWED_METHOD_IDS:
            raise ValueError(
                f"source replay method is not allowlisted: {method_id!r}"
            )
        train_loader = args[1] if len(args) > 1 else kwargs.get("train_dataloader")
        batch_size = getattr(train_loader, "batch_size", None)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("source replay requires a positive target batch size")
        model = args[0] if args else kwargs.get("model")
        if not isinstance(model, nn.Module):
            raise TypeError("source replay requires a torch model")
        center = str(kwargs.get("center", ""))
        if not center:
            raise ValueError("source replay requires a center")
        model_name = online_trainer._model_spec(model).name
        source_loader = get_dataloader(
            dataset="ptbxl",
            partition=SOURCE_PARTITION,
            batch_size=batch_size,
            sampling_rate_hz=100,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
            persistent_workers=False,
            cache_mode="ram",
            validate_values="sample",
            selection_resident=False,
            prepare_for_model=False,
            sanitize=False,
            global_zscore=False,
            output_layout="time_channel",
            seed_namespace=f"matched_source_replay_v1:{center}:{model_name}",
            seed_config_path=config.references["random_seed_config"],
            config_root=config.config_root,
        )
        _STATE.active = True
        _STATE.loader = source_loader
        _STATE.profile_name = profile_name
        _STATE.total_weight = PROFILE_TOTAL_WEIGHTS[profile_name]
        _STATE.method_id = method_id
        _STATE.center = center
        _STATE.model_name = model_name
        describe = getattr(source_loader, "describe", None)
        _STATE.loader_identity = describe() if callable(describe) else {
            "type": source_loader.__class__.__name__,
            "batch_size": batch_size,
        }
        try:
            return original_train(*args, **kwargs)
        finally:
            _STATE.active = False
            _STATE.iterator = None
            close = getattr(source_loader, "close", None)
            if callable(close):
                close()
            _STATE.loader = None

    online_trainer._compute_objective = patched_objective
    online_trainer.train_online_model = patched_train
    pn2021_tuning.train_online_model = patched_train
    train_pn2021.train_online_model = patched_train
    try:
        yield
    finally:
        train_pn2021.train_online_model = original_train_alias
        pn2021_tuning.train_online_model = original_tuning_alias
        online_trainer.train_online_model = original_train
        online_trainer._compute_objective = original_objective
        if _STATE.active:
            raise RuntimeError("source replay adapter exited during an active run")


__all__ = [
    "ALLOWED_METHOD_IDS",
    "PROFILE_TOTAL_WEIGHTS",
    "diagnostics_payload",
    "patch_online_trainer_runtime",
    "write_diagnostics",
]
