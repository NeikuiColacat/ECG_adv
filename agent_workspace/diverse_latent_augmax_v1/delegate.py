"""Fixed dispatch target for managed Diverse Latent AugMax sandbox runs."""

from __future__ import annotations

import os
import hashlib
import json
import shutil
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


# Limit inherited BLAS/OpenMP fan-out before importing torch on the shared host.
CPU_THREAD_LIMIT = int(os.environ.get("ECG_DIVERSE_AUGMAX_CPU_THREADS", "4"))
if CPU_THREAD_LIMIT <= 0:
    raise ValueError("ECG_DIVERSE_AUGMAX_CPU_THREADS must be positive")
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, str(CPU_THREAD_LIMIT))

# PyTorch's deterministic-algorithm contract requires an explicit CuBLAS
# workspace configuration on CUDA >= 10.2.  Set it before importing torch so
# managed runs behave identically whether or not the parent shell exported it.
CUBLAS_WORKSPACE_CONFIG = os.environ.get(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)
if CUBLAS_WORKSPACE_CONFIG not in {":4096:8", ":16:8"}:
    raise ValueError(
        "CUBLAS_WORKSPACE_CONFIG must be ':4096:8' or ':16:8' for "
        "deterministic CUDA execution"
    )
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)


SANDBOX_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SANDBOX_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SANDBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SANDBOX_ROOT))

import torch  # noqa: E402
from boot_scripts import (  # noqa: E402
    evaluate_pn2021,
    select_pn2021_direct,
    train_pn2021,
    tune_pn2021_direct,
)
import runtime_adapter  # noqa: E402
import objective_scale_adapter  # noqa: E402
import paired_path_runtime  # noqa: E402
import compatible_runtime  # noqa: E402
import pcgrad_runtime  # noqa: E402
import asl_runtime  # noqa: E402
import targeted_lhat_runtime  # noqa: E402
import soft_teacher_runtime  # noqa: E402
import source_replay_runtime  # noqa: E402
import unlabeled_teacher_runtime  # noqa: E402
import coverage_anchor_runtime  # noqa: E402
import rotating4_runtime  # noqa: E402
from canonical_pool import patch_canonical_pool_runtime  # noqa: E402


DELEGATE_ENTRYPOINT_ENV = "ECG_DIVERSE_AUGMAX_SANDBOX_ENTRYPOINT"
Entrypoint = Callable[[Sequence[str] | None], int]
ENTRYPOINTS: dict[str, Entrypoint] = {
    "tune_pn2021_direct": tune_pn2021_direct.main,
    "select_pn2021_direct": select_pn2021_direct.main,
    "train_pn2021": train_pn2021.main,
    "evaluate_pn2021": evaluate_pn2021.main,
}
RUNTIME_PATCHED_ENTRYPOINTS = frozenset(
    {"tune_pn2021_direct", "train_pn2021"}
)
RUNTIME_SOURCE_SNAPSHOT_DIRNAME = "runtime_source_snapshot"
RUNTIME_SOURCE_IDENTITY_FILENAME = "runtime_source_identity.json"
RUNTIME_SOURCE_FILES: tuple[tuple[str, Path, Path], ...] = (
    (
        "delegate",
        Path(__file__),
        Path("agent_workspace/diverse_latent_augmax_v1/delegate.py"),
    ),
    (
        "objective_scale_adapter",
        Path(objective_scale_adapter.__file__),
        Path(
            "agent_workspace/diverse_latent_augmax_v1/"
            "objective_scale_adapter.py"
        ),
    ),
    (
        "soft_teacher_runtime",
        Path(soft_teacher_runtime.__file__),
        Path(
            "agent_workspace/diverse_latent_augmax_v1/"
            "soft_teacher_runtime.py"
        ),
    ),
    (
        "runtime_adapter",
        Path(runtime_adapter.__file__),
        Path("agent_workspace/diverse_latent_augmax_v1/runtime_adapter.py"),
    ),
    (
        "paired_path_runtime",
        Path(paired_path_runtime.__file__),
        Path(
            "agent_workspace/diverse_latent_augmax_v1/"
            "paired_path_runtime.py"
        ),
    ),
    (
        "compatible_runtime",
        Path(compatible_runtime.__file__),
        Path(
            "agent_workspace/diverse_latent_augmax_v1/"
            "compatible_runtime.py"
        ),
    ),
    (
        "pcgrad_runtime",
        Path(pcgrad_runtime.__file__),
        Path("agent_workspace/diverse_latent_augmax_v1/pcgrad_runtime.py"),
    ),
    (
        "asl_runtime",
        Path(asl_runtime.__file__),
        Path("agent_workspace/diverse_latent_augmax_v1/asl_runtime.py"),
    ),
    (
        "targeted_lhat_runtime",
        Path(targeted_lhat_runtime.__file__),
        Path(
            "agent_workspace/diverse_latent_augmax_v1/"
            "targeted_lhat_runtime.py"
        ),
    ),
    (
        "source_replay_runtime",
        Path(source_replay_runtime.__file__),
        Path(
            "agent_workspace/diverse_latent_augmax_v1/"
            "source_replay_runtime.py"
        ),
    ),
    (
        "unlabeled_teacher_runtime",
        Path(unlabeled_teacher_runtime.__file__),
        Path(
            "agent_workspace/diverse_latent_augmax_v1/"
            "unlabeled_teacher_runtime.py"
        ),
    ),
    (
        "coverage_anchor_runtime",
        Path(coverage_anchor_runtime.__file__),
        Path(
            "agent_workspace/diverse_latent_augmax_v1/"
            "coverage_anchor_runtime.py"
        ),
    ),
    (
        "rotating4_runtime",
        Path(rotating4_runtime.__file__),
        Path(
            "agent_workspace/diverse_latent_augmax_v1/"
            "rotating4_runtime.py"
        ),
    ),
    (
        "canonical_pool",
        SANDBOX_ROOT / "canonical_pool.py",
        Path("agent_workspace/diverse_latent_augmax_v1/canonical_pool.py"),
    ),
    (
        "a7_global_search",
        SANDBOX_ROOT / "a7_global_search.py",
        Path("agent_workspace/diverse_latent_augmax_v1/a7_global_search.py"),
    ),
    (
        "launch",
        SANDBOX_ROOT / "launch.py",
        Path("agent_workspace/diverse_latent_augmax_v1/launch.py"),
    ),
    (
        "core.online_trainer",
        PROJECT_ROOT / "core" / "online_trainer.py",
        Path("core/online_trainer.py"),
    ),
    (
        "core.methods.runtime",
        PROJECT_ROOT / "core" / "methods" / "runtime.py",
        Path("core/methods/runtime.py"),
    ),
)


def _runtime_source_capture() -> tuple[list[dict[str, Any]], dict[Path, bytes]]:
    entries: list[dict[str, Any]] = []
    payloads: dict[Path, bytes] = {}
    labels: set[str] = set()
    source_paths: set[Path] = set()
    snapshot_paths: set[Path] = set()
    for label, raw_source, raw_snapshot_path in RUNTIME_SOURCE_FILES:
        if not isinstance(label, str) or not label:
            raise ValueError("runtime source labels must be non-empty strings")
        source = raw_source.expanduser().resolve(strict=True)
        snapshot_path = Path(raw_snapshot_path)
        if (
            snapshot_path.is_absolute()
            or not snapshot_path.parts
            or ".." in snapshot_path.parts
        ):
            raise ValueError(
                f"unsafe runtime source snapshot path: {snapshot_path}"
            )
        if not source.is_file():
            raise FileNotFoundError(f"runtime source is not a file: {source}")
        if label in labels or source in source_paths or snapshot_path in snapshot_paths:
            raise ValueError("runtime source snapshot entries must be unique")
        labels.add(label)
        source_paths.add(source)
        snapshot_paths.add(snapshot_path)
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        entries.append(
            {
                "label": label,
                "original_path": str(source),
                "snapshot_path": snapshot_path.as_posix(),
                "sha256": digest,
                "size_bytes": len(payload),
            }
        )
        payloads[snapshot_path] = payload
    return entries, payloads


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _persist_runtime_source_snapshot(
    output_dir: Path,
    *,
    entrypoint_name: str,
    entries: Sequence[dict[str, Any]],
    payloads: dict[Path, bytes],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / RUNTIME_SOURCE_SNAPSHOT_DIRNAME
    if destination.exists():
        raise FileExistsError(
            f"runtime source snapshot already exists: {destination}"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{RUNTIME_SOURCE_SNAPSHOT_DIRNAME}.",
            suffix=".tmp",
            dir=output_dir,
        )
    )
    try:
        for entry in entries:
            relative = Path(str(entry["snapshot_path"]))
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = payloads[relative]
            target.write_bytes(payload)
            if (
                target.stat().st_size != entry["size_bytes"]
                or hashlib.sha256(target.read_bytes()).hexdigest()
                != entry["sha256"]
            ):
                raise RuntimeError(
                    f"runtime source snapshot verification failed: {relative}"
                )
        manifest = {
            "schema_version": 1,
            "artifact_type": "sandbox_runtime_source_snapshot",
            "capture_phase": "before_delegate_entrypoint",
            "entrypoint": entrypoint_name,
            "files": list(entries),
        }
        _write_json_atomic(staging / "manifest.json", manifest)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "manifest.json"


def _persist_runtime_source_identity(
    output_dir: Path,
    *,
    entrypoint_name: str,
    lifecycle_status: str,
    initial_entries: Sequence[dict[str, Any]],
    payloads: dict[Path, bytes],
) -> bool:
    post_entries: list[dict[str, Any]] | None
    recheck_error: str | None
    try:
        post_entries, _ = _runtime_source_capture()
    except Exception as exc:
        post_entries = None
        recheck_error = f"{type(exc).__name__}: {exc}"
    else:
        recheck_error = None
    initial_by_label = {str(entry["label"]): entry for entry in initial_entries}
    post_by_label = (
        {}
        if post_entries is None
        else {str(entry["label"]): entry for entry in post_entries}
    )
    drifted_labels = sorted(
        label
        for label in set(initial_by_label) | set(post_by_label)
        if initial_by_label.get(label) != post_by_label.get(label)
    )
    source_stable = (
        None if post_entries is None else not bool(drifted_labels)
    )
    resolved_lifecycle_status = lifecycle_status
    if lifecycle_status == "delegate_returned" and source_stable is not True:
        resolved_lifecycle_status = (
            "source_recheck_failed"
            if post_entries is None
            else "source_drift_detected"
        )
    manifest_path = _persist_runtime_source_snapshot(
        output_dir,
        entrypoint_name=entrypoint_name,
        entries=initial_entries,
        payloads=payloads,
    )
    identity = {
        "schema_version": 1,
        "artifact_type": "sandbox_runtime_source_identity",
        "entrypoint": entrypoint_name,
        "lifecycle_status": resolved_lifecycle_status,
        "source_stable_during_delegate": source_stable,
        "drifted_labels": drifted_labels,
        "post_capture_error": recheck_error,
        "snapshot_manifest": {
            "path": (
                Path(RUNTIME_SOURCE_SNAPSHOT_DIRNAME) / manifest_path.name
            ).as_posix(),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "size_bytes": manifest_path.stat().st_size,
        },
    }
    if drifted_labels:
        identity["post_delegate_files"] = post_entries
    _write_json_atomic(
        output_dir / RUNTIME_SOURCE_IDENTITY_FILENAME,
        identity,
    )
    return source_stable is True


@contextmanager
def _runtime_source_snapshot_scope(
    output_dir: str | Path,
    *,
    entrypoint_name: str,
) -> Iterator[None]:
    output = Path(output_dir).expanduser().resolve()
    initial_entries, payloads = _runtime_source_capture()
    try:
        yield
    except BaseException:
        try:
            _persist_runtime_source_identity(
                output,
                entrypoint_name=entrypoint_name,
                lifecycle_status="delegate_raised",
                initial_entries=initial_entries,
                payloads=payloads,
            )
        except Exception:
            # Preserve the original delegate failure.  No successful source
            # identity is emitted when evidence persistence itself fails.
            pass
        raise
    else:
        stable = _persist_runtime_source_identity(
            output,
            entrypoint_name=entrypoint_name,
            lifecycle_status="delegate_returned",
            initial_entries=initial_entries,
            payloads=payloads,
        )
        if not stable:
            raise RuntimeError(
                "runtime source changed or became unreadable during delegate "
                "execution; refusing a successful reproducibility record"
            )


def _selected_entrypoint() -> str:
    name = os.environ.get(DELEGATE_ENTRYPOINT_ENV)
    if not name:
        raise RuntimeError(
            f"{DELEGATE_ENTRYPOINT_ENV} is required; invoke through launch.py"
        )
    if name not in ENTRYPOINTS:
        raise ValueError(
            f"unsupported sandbox delegate entrypoint {name!r}; "
            f"expected one of {sorted(ENTRYPOINTS)}"
        )
    return name


def _option_value(
    arguments: Sequence[str],
    flag: str,
    *,
    required: bool,
    default: str | None = None,
) -> str | None:
    values: list[str] = []
    prefix = f"{flag}="
    index = 0
    while index < len(arguments):
        token = str(arguments[index])
        if token == flag:
            if index + 1 >= len(arguments):
                raise ValueError(f"{flag} requires a value")
            values.append(str(arguments[index + 1]))
            index += 2
            continue
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
        index += 1
    if len(values) > 1:
        raise ValueError(f"{flag} may be supplied at most once")
    if not values:
        if required:
            raise ValueError(f"verified full-K500 promotion requires {flag}")
        return default
    if not values[0]:
        raise ValueError(f"{flag} value must be non-empty")
    return values[0]


def _without_option_pair(
    arguments: Sequence[str],
    flag: str,
) -> tuple[str, ...]:
    """Remove one already-verified private promotion argument."""

    result: list[str] = []
    prefix = f"{flag}="
    index = 0
    removed = 0
    while index < len(arguments):
        token = str(arguments[index])
        if token == flag:
            if index + 1 >= len(arguments):
                raise ValueError(f"{flag} requires a value")
            removed += 1
            index += 2
            continue
        if token.startswith(prefix):
            removed += 1
            index += 1
            continue
        result.append(token)
        index += 1
    if removed != 1:
        raise ValueError(f"{flag} must occur exactly once")
    return tuple(result)


def _promotion_invocation(arguments: Sequence[str]) -> dict[str, object]:
    config_root_raw = _option_value(arguments, "--config-root", required=True)
    method_raw = _option_value(arguments, "--method-config", required=True)
    selection_raw = _option_value(arguments, "--selection-json", required=True)
    source_raw = _option_value(arguments, "--source-checkpoint", required=True)
    model = _option_value(arguments, "--model", required=True)
    epochs_raw = _option_value(arguments, "--epochs", required=True)
    horizon_raw = _option_value(
        arguments, "--scheduler-horizon-epochs", required=True
    )
    scope = _option_value(
        arguments,
        "--trainable-scope",
        required=False,
        default="full",
    )
    assert all(
        value is not None
        for value in (
            config_root_raw,
            method_raw,
            selection_raw,
            source_raw,
            model,
            epochs_raw,
            horizon_raw,
            scope,
        )
    )
    config_root = Path(str(config_root_raw)).expanduser().resolve()
    method_path = Path(str(method_raw)).expanduser()
    if not method_path.is_absolute():
        method_path = config_root / method_path
    return {
        "entrypoint": "train_pn2021",
        "model_family": str(model),
        "method_config_path": method_path.resolve(),
        "selection_json_path": Path(str(selection_raw)).expanduser().resolve(),
        "source_checkpoint_path": Path(str(source_raw)).expanduser().resolve(),
        "epochs": int(str(epochs_raw)),
        "scheduler_horizon_epochs": int(str(horizon_raw)),
        "trainable_scope": str(scope),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the whitelist entrypoint with a scoped runtime patch when needed."""

    torch.set_num_threads(CPU_THREAD_LIMIT)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    name = _selected_entrypoint()
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    entrypoint_arguments = arguments
    # The sidecar path is delivered only through this sandbox subprocess's
    # environment.  Consume it immediately so whitelist parsers and any later
    # child process cannot observe a hidden CLI/config surface.
    promotion_sidecar = os.environ.pop(
        compatible_runtime.PROMOTION_SIDECAR_ENV,
        None,
    )
    output_raw = _option_value(
        entrypoint_arguments,
        "--output-dir",
        required=False,
    )
    if name in RUNTIME_PATCHED_ENTRYPOINTS and output_raw is None:
        raise ValueError(
            "runtime-patched training requires --output-dir so source "
            "snapshots can enter the managed run file index"
        )
    with ExitStack() as stack:
        if name in RUNTIME_PATCHED_ENTRYPOINTS:
            assert output_raw is not None
            # Capture bytes before any training branch executes, but defer
            # filesystem creation until the delegate has created output_dir.
            # This scope exits last, after every runtime patch is restored and
            # every adapter diagnostic has been persisted.
            stack.enter_context(
                _runtime_source_snapshot_scope(
                    output_raw,
                    entrypoint_name=name,
                )
            )
        if promotion_sidecar is not None:
            if name != "train_pn2021":
                raise ValueError(
                    "full-K500 promotion sidecar is valid only for train_pn2021"
                )
            promotion = stack.enter_context(
                compatible_runtime.selected_full_k500_refit_scope(
                    promotion_sidecar,
                    invocation=_promotion_invocation(arguments),
                )
            )
            # The whitelist refit parser accepts only an identical tuning
            # method profile.  A7 deliberately derives a full-K500 outer
            # profile whose only changes are the verified population
            # contracts.  Its dedicated sidecar validates that derivation and
            # the complete pooled curve, so consume --selection-json here
            # instead of asking the generic parser to reject the intentional
            # profile-hash change.  Legacy D13 promotion keeps its original
            # forwarding behavior.
            if (
                promotion.policy
                == compatible_runtime.A7_ECGFOUNDER_PROMOTION_POLICY
            ):
                entrypoint_arguments = _without_option_pair(
                    arguments,
                    "--selection-json",
                )
        if name == "evaluate_pn2021":
            checkpoint_raw = _option_value(
                arguments,
                "--checkpoint",
                required=True,
            )
            assert checkpoint_raw is not None
            stack.enter_context(
                unlabeled_teacher_runtime.patch_pn2021_evaluation_runtime(
                    checkpoint_raw
                )
            )
        if name in RUNTIME_PATCHED_ENTRYPOINTS:
            # Both patches are scoped to this delegate process and restore all
            # imported whitelist symbols in ``finally``.  The pool bridge is
            # intentionally separate from the method runtime patch so the
            # train400 geometry/K500 outer-membership contract stays auditable.
            stack.enter_context(patch_canonical_pool_runtime())
            stack.enter_context(runtime_adapter.patch_online_trainer_runtime())
            stack.enter_context(
                objective_scale_adapter.patch_online_trainer_runtime()
            )
            stack.enter_context(paired_path_runtime.patch_online_trainer_runtime())
            # Keep increasingly narrow experimental adapters last; each one
            # intercepts only its explicit method allowlist and delegates every
            # other profile through the already-scoped patch chain.
            stack.enter_context(compatible_runtime.patch_online_trainer_runtime())
            stack.enter_context(pcgrad_runtime.patch_online_trainer_runtime())
            stack.enter_context(targeted_lhat_runtime.patch_online_trainer_runtime())
            # ASL stays outermost so it sees both Direct and PCGrad-routed BCE
            # terms while leaving LHAT attack generation and JSD untouched.
            stack.enter_context(asl_runtime.patch_online_trainer_runtime())
            stack.enter_context(soft_teacher_runtime.patch_online_trainer_runtime())
            # Source replay stays outermost so it augments the final clean
            # objective while PCGrad continues to see the combined base
            # (target fixed20 + source) gradient at the clipping boundary.
            stack.enter_context(source_replay_runtime.patch_online_trainer_runtime())
            # This exploratory adapter is opt-in through an explicit method
            # contract. It consumes ref-excluded target waveforms without
            # labels and therefore records a transductive-evidence warning.
            stack.enter_context(
                unlabeled_teacher_runtime.patch_online_trainer_runtime()
            )
            # Coverage-anchor remains the outermost runtime factory so its
            # fixed20-only profile cannot be intercepted by broad exploration
            # adapters, while every other method delegates unchanged.
            stack.enter_context(coverage_anchor_runtime.patch_online_trainer_runtime())
            # Rotating4 is the final, narrowest adapter. It changes only
            # profiles with an explicit rotating4_schedule contract and wraps
            # the fully composed Direct/PCGrad runtime produced above.
            stack.enter_context(rotating4_runtime.patch_online_trainer_runtime())
        result = int(ENTRYPOINTS[name](entrypoint_arguments))
        if output_raw is not None:
            pcgrad_runtime.write_diagnostics(output_raw)
            asl_runtime.write_diagnostics(output_raw)
            targeted_lhat_runtime.write_diagnostics(output_raw)
            soft_teacher_runtime.write_diagnostics(output_raw)
            source_replay_runtime.write_diagnostics(output_raw)
            unlabeled_teacher_runtime.write_diagnostics(output_raw)
            coverage_anchor_runtime.write_diagnostics(output_raw)
        return result


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENTRYPOINTS",
    "RUNTIME_PATCHED_ENTRYPOINTS",
    "main",
]
