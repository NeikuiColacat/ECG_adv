"""Boot one matched PN2021 K500 typed-method run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.online_trainer import (
    ALLOWED_CENTERS,
    DEFAULT_ONLINE_CONFIG_PATH,
    load_online_train_config,
    resolve_online_training_parameters,
)
from core.pn2021_tuning import validate_refit_reproducibility_context
from core.train_PN2021 import load_pn2021_method_profile, train_pn2021
from models import build_ecgtwin_vae, build_model, get_model_spec, load_vae_config
from models.checkpoints import sha256_file
from models.contracts import CLASS_ORDER
from util.config_bundle import resolve_config_reference, resolve_yaml_config_closure
from util.evaluation.direct_baseline_selection import locked_pooled_selection_rule
from util.run_record import (
    resolve_run_config_snapshot_by_sha256,
    validate_managed_run_member,
    validate_run_config_snapshots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one manual PN2021 K500 typed method."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_ONLINE_CONFIG_PATH)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument(
        "--model", required=True, choices=("efficientnet1dv2", "ecgfounder")
    )
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--center", required=True, choices=ALLOWED_CENTERS)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--trainable-scope", choices=("full", "head"), default="full")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--scheduler-horizon-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--minimum-learning-rate-ratio", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument(
        "--amp", dest="amp_enabled", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--cache-mode", choices=("auto", "ram", "mmap"))
    parser.add_argument("--validate-values", choices=("none", "sample", "full"))
    parser.add_argument(
        "--drop-last", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--pos-weight", nargs=5, type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _not_none(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    except (TypeError, ValueError):
        return False


def _validate_refit_selection(
    path: Path,
    *,
    model_name: str,
    method: Any,
    source_checkpoint: Path,
    vae_checkpoint: Path | None,
    resolved_training: Mapping[str, Any],
    config: Any,
    trainable_scope: str,
    pos_weight: Sequence[float] | None,
    supplied_training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selection_path = path.expanduser().resolve()
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"pooled selection not found: {selection_path}"
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid pooled selection JSON: {exc}") from exc
    selection = _mapping(payload, "pooled selection")
    managed = validate_managed_run_member(selection_path)
    if selection.get("schema_version") != 1 or selection.get("status") != "selected":
        raise ValueError("pooled selection must be a selected schema_version=1 artifact")
    if selection.get("artifact_type") != "pn2021_k500_pooled_epoch_selection":
        raise ValueError("pooled selection is not a managed non-Direct K500 artifact")
    if tuple(selection.get("centers", ())) != ALLOWED_CENTERS:
        raise ValueError("pooled selection must cover the locked four centers")
    if tuple(selection.get("class_order", ())) != CLASS_ORDER:
        raise ValueError("pooled selection class_order mismatch")
    if int(selection.get("record_count", -1)) != 400:
        raise ValueError("pooled selection must contain exactly 400 validation records")
    if int(selection.get("composition_count", -1)) != 20:
        raise ValueError("pooled selection must contain twenty corruption compositions")
    rule = _mapping(selection.get("selection_rule"), "selection_rule")
    if rule != locked_pooled_selection_rule():
        raise ValueError("pooled selection rule differs from the locked protocol")

    comparison = _mapping(
        selection.get("comparison_identity"), "comparison_identity"
    )
    if selection.get("comparison_identity_sha256") != _canonical_sha256(comparison):
        raise ValueError("selection comparison_identity SHA256 mismatch")
    online_config_snapshot = resolve_run_config_snapshot_by_sha256(
        managed["run_dir"],
        comparison.get("online_training_config_sha256"),
    )
    tuning_config_snapshot = resolve_run_config_snapshot_by_sha256(
        managed["run_dir"],
        comparison.get("tuning_config_sha256"),
    )
    if comparison.get("model_family") != model_name:
        raise ValueError("selection model_family does not match --model")
    if comparison.get("method_id") != method.profile_name:
        raise ValueError("selection method_id does not match --method-config")
    if comparison.get("method_profile_sha256") != sha256_file(method.source_path):
        raise ValueError("selection method profile SHA256 mismatch")
    config_closure = resolve_yaml_config_closure(
        (method.source_path, *config.references.values()),
        config_root=config.config_root,
    )
    config_snapshots = validate_run_config_snapshots(
        managed["run_dir"],
        config_closure,
        config_root=config.config_root,
    )
    source = _mapping(
        comparison.get("source_checkpoint_identity"),
        "source_checkpoint_identity",
    )
    expected_source_path = Path(str(source.get("path", ""))).expanduser().resolve()
    if expected_source_path != source_checkpoint:
        raise ValueError("selection source checkpoint path mismatch")
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source_checkpoint}")
    if source.get("sha256") != sha256_file(source_checkpoint):
        raise ValueError("selection source checkpoint SHA256 mismatch")

    selected_epoch = selection.get("selected_epoch")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch <= 0
    ):
        raise ValueError("selected_epoch must be a positive integer")
    parameters = _mapping(
        comparison.get("resolved_training_parameters"),
        "resolved_training_parameters",
    )
    tuning_epochs = parameters.get("epochs")
    if (
        isinstance(tuning_epochs, bool)
        or not isinstance(tuning_epochs, int)
        or selected_epoch > tuning_epochs
    ):
        raise ValueError("selected_epoch lies outside the tuning horizon")
    refit_training = dict(parameters)
    refit_training["epochs"] = selected_epoch
    for key in (
        "scheduler_horizon_epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "minimum_learning_rate_ratio",
        "gradient_clip_norm",
    ):
        if not _same_number(resolved_training.get(key), parameters.get(key)):
            raise ValueError(f"refit {key} differs from the pooled tuning contract")
    for key in ("amp_enabled", "amp_dtype"):
        if resolved_training.get(key) != parameters.get(key):
            raise ValueError(f"refit {key} differs from the pooled tuning contract")
    for key, value in (supplied_training or {}).items():
        expected = refit_training.get(key)
        matches = (
            value == expected
            if key in {"amp_enabled", "amp_dtype"}
            else _same_number(value, expected)
        )
        if not matches:
            raise ValueError(
                f"explicit refit {key} differs from the pooled selection contract"
            )

    effective_pos_weight = (
        comparison.get("pos_weight") if pos_weight is None else pos_weight
    )
    reproducibility = validate_refit_reproducibility_context(
        comparison,
        config,
        selection_online_config_path=online_config_snapshot["path"],
        strict_online_config_sha256=False,
        trainable_scope=trainable_scope,
        pos_weight=effective_pos_weight,
    )

    epoch_grid = selection.get("epoch_grid")
    if epoch_grid != list(range(1, tuning_epochs + 1)):
        raise ValueError("selection epoch_grid is incomplete or non-contiguous")
    per_epoch = selection.get("per_epoch")
    if not isinstance(per_epoch, list) or len(per_epoch) != tuning_epochs:
        raise ValueError("selection per_epoch evidence is incomplete")
    eligible: list[tuple[float, int]] = []
    for expected_epoch, item in enumerate(per_epoch, start=1):
        member = _mapping(item, f"per_epoch[{expected_epoch}]")
        if member.get("epoch") != expected_epoch:
            raise ValueError("selection per_epoch order mismatch")
        score = float(member.get("score", float("nan")))
        if not math.isfinite(score):
            raise ValueError("selection per_epoch score must be finite")
        if member.get("eligible") is True:
            eligible.append((score, expected_epoch))
    if not eligible:
        raise ValueError("selection has no clean-floor-eligible epoch")
    recomputed_epoch = max(eligible, key=lambda item: (item[0], -item[1]))[1]
    if recomputed_epoch != selected_epoch:
        raise ValueError("selected_epoch is inconsistent with the pooled curve")
    selected_metrics = per_epoch[selected_epoch - 1]
    checks = {
        "selected_score": selected_metrics["score"],
        "selected_clean_macro_auprc": selected_metrics["clean"]["macro_auprc"],
        "selected_robust_macro_auprc": selected_metrics["robust"]["macro_auprc"],
    }
    for field, expected in checks.items():
        if not _same_number(selection.get(field), expected):
            raise ValueError(f"{field} is inconsistent with selected per_epoch evidence")

    if vae_checkpoint is not None:
        if not vae_checkpoint.is_file():
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")
        vae_sha256 = sha256_file(vae_checkpoint)
        for field in ("vae_encoder_checkpoint", "vae_decoder_checkpoint"):
            identity = _mapping(comparison.get(field), field)
            if identity.get("sha256") != vae_sha256:
                raise ValueError(f"selection {field} SHA256 mismatch")

    return {
        "path": str(selection_path),
        "sha256": sha256_file(selection_path),
        "comparison_identity_sha256": selection["comparison_identity_sha256"],
        "selected_epoch": selected_epoch,
        "selected_score": float(selection["selected_score"]),
        "selected_clean_macro_auprc": float(
            selection["selected_clean_macro_auprc"]
        ),
        "selected_robust_macro_auprc": float(
            selection["selected_robust_macro_auprc"]
        ),
        "heldout_evaluation_used": False,
        "reproducibility": reproducibility,
        "managed_run": {
            "run_dir": str(managed["run_dir"]),
            "relative_path": managed["relative_path"],
        },
        "config_snapshots": config_snapshots,
        "selection_config_snapshots": {
            "online_training": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in online_config_snapshot.items()
            },
            "tuning": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in tuning_config_snapshot.items()
            },
        },
        "training_parameters": refit_training,
        "pos_weight": reproducibility["pos_weight"],
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_online_train_config(args.config, config_root=args.config_root)
    if (
        config.payload["protocol"]["status"] == "locked_matched_k500_comparison"
        and args.selection_json is None
    ):
        raise ValueError("locked matched K500 refit requires --selection-json")
    method, _ = load_pn2021_method_profile(
        args.method_config,
        config_path=config.path,
        config_root=config.config_root,
    )
    training_parameters = _not_none(
        epochs=args.epochs,
        scheduler_horizon_epochs=args.scheduler_horizon_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
        amp_enabled=args.amp_enabled,
        amp_dtype=args.amp_dtype,
    )
    resolved_training, supplied_training = resolve_online_training_parameters(
        config, args.model, training_parameters
    )
    dataloader_parameters = _not_none(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        cache_mode=args.cache_mode,
        validate_values=args.validate_values,
        drop_last=args.drop_last,
    )
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    vae_checkpoint = (
        None if args.vae_checkpoint is None else args.vae_checkpoint.expanduser().resolve()
    )
    needs_pool = bool(method.requirements.latent_pool)
    needs_runtime_encoder = bool(method.requirements.vae_encoder)
    needs_decoder = bool(method.requirements.vae_decoder)
    needs_encoder_component = needs_pool or needs_runtime_encoder
    needs_vae = needs_encoder_component or needs_decoder
    if not needs_vae and vae_checkpoint is not None:
        raise ValueError("method without latent resources rejects --vae-checkpoint")
    resolved_vae_checkpoint = None
    vae_config_path = None
    if needs_vae:
        vae_resource = method.resources.get("vae")
        if not isinstance(vae_resource, dict):
            raise ValueError("latent method must declare a VAE config resource")
        vae_config_path = resolve_config_reference(
            vae_resource.get("path"),
            owner_config_path=method.source_path,
            config_root=config.config_root,
            description="method.resources.vae",
            must_exist=True,
        )
        vae_config = load_vae_config(vae_config_path)
        resolved_vae_checkpoint = vae_checkpoint or vae_config.checkpoint_path
    selection = (
        None
        if args.selection_json is None
        else _validate_refit_selection(
            args.selection_json,
            model_name=args.model,
            method=method,
            source_checkpoint=source_checkpoint,
            vae_checkpoint=resolved_vae_checkpoint,
            resolved_training=resolved_training,
            supplied_training=supplied_training,
            config=config,
            trainable_scope=args.trainable_scope,
            pos_weight=args.pos_weight,
        )
    )
    effective_training_parameters = (
        training_parameters
        if selection is None
        else dict(selection["training_parameters"])
    )
    effective_pos_weight = (
        args.pos_weight if selection is None else selection["pos_weight"]
    )
    if selection is not None:
        resolved_training = dict(selection["training_parameters"])
    output_dir = (
        None if args.output_dir is None else args.output_dir.expanduser().resolve()
    )
    plan = {
        "action": (
            "pn2021_k500_online_training"
            if selection is None
            else "pn2021_selected_full_k500_refit"
        ),
        "model": get_model_spec(args.model).describe(),
        "method": method.describe(),
        "center": args.center,
        "config": config.describe(),
        "source_checkpoint": str(source_checkpoint),
        "selection": selection,
        "vae_checkpoint": (
            None if not needs_vae else str(resolved_vae_checkpoint)
        ),
        "vae_requirements": {
            "latent_pool": needs_pool,
            "runtime_encoder": needs_runtime_encoder,
            "decoder": needs_decoder,
        },
        "output_dir": None if output_dir is None else str(output_dir),
        "device": args.device or config.payload["training"]["device"],
        "training_parameters": {
            "resolved": resolved_training,
            "explicit_overrides": supplied_training,
            "selection_derived": (
                None if selection is None else selection["training_parameters"]
            ),
        },
        "dataloader_parameters": dataloader_parameters,
        "trainable_scope": args.trainable_scope,
        "pos_weight": effective_pos_weight,
        "dry_run_side_effects": "none",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source_checkpoint}")

    seed = config.payload["random_seed"]
    model_kwargs: dict[str, Any]
    if args.model == "efficientnet1dv2":
        model_kwargs = {
            "checkpoint_path": source_checkpoint,
            "map_location": "cpu",
        }
    else:
        model_kwargs = {
            "task_checkpoint_path": source_checkpoint,
            "trainable_scope": args.trainable_scope,
            "map_location": "cpu",
        }
    model = build_model(
        args.model,
        config_root=config.config_root,
        seed_namespace="pn2021_model_initialization",
        seed_identity=(
            seed["comparison_group"],
            seed["replicate_id"],
            args.center,
            args.model,
        ),
        **model_kwargs,
    )

    encoder = decoder = None
    if needs_vae:
        assert vae_config_path is not None
        loaded_encoder, loaded_decoder = build_ecgtwin_vae(
            config_path=vae_config_path,
            checkpoint_path=resolved_vae_checkpoint,
            map_location="cpu",
        )
        encoder = loaded_encoder if needs_encoder_component else None
        decoder = loaded_decoder if needs_decoder else None
    result = train_pn2021(
        model,
        center=args.center,
        method_config_path=method.source_path,
        encoder=encoder,
        decoder=decoder,
        config_path=config.path,
        config_root=config.config_root,
        output_dir=output_dir,
        device=args.device,
        pos_weight=effective_pos_weight,
        training_parameters=effective_training_parameters,
        dataloader_parameters=dataloader_parameters,
    )
    result_description = result.describe()
    if selection is not None:
        contract_path = result.output_dir / "refit_contract.json"
        _write_json_atomic(
            contract_path,
            {
                "schema_version": 1,
                "artifact_type": "pn2021_selected_k500_refit_contract",
                "plan": plan,
                "result": result_description,
            },
        )
        result_description = {
            **result_description,
            "refit_contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
        }
    print(json.dumps(result_description, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
