"""Matched ECGFounder A0/A3/A5 comparison contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ecg_adv_gen.matched_effnet import MatchedEffnetArm, matched_effnet_arm


MATCHED_ECGFOUNDER_CONTRACT_VERSION = "ecgfounder_matched_a035_v1"
MATCHED_ECGFOUNDER_ARMS = ("a0", "a3", "a5")
MATCHED_ECGFOUNDER_ARM_COMPONENTS: dict[str, MatchedEffnetArm] = {
    arm: matched_effnet_arm(arm) for arm in MATCHED_ECGFOUNDER_ARMS
}


def matched_ecgfounder_arm(arm: str) -> MatchedEffnetArm:
    """Return one of the only three allowed matched ECGFounder arms."""

    try:
        return MATCHED_ECGFOUNDER_ARM_COMPONENTS[str(arm)]
    except KeyError:
        raise ValueError(f"matched ECGFounder contract allows only A0/A3/A5, got {arm!r}") from None


def external_repository_provenance(path: str | Path) -> dict[str, Any]:
    """Record a read-only external-repository identity for run provenance."""

    repo = Path(path).expanduser().resolve(strict=False)

    def git(*args: str) -> tuple[int, str]:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode, proc.stdout.strip()

    inside_rc, inside = git("rev-parse", "--is-inside-work-tree")
    if inside_rc != 0 or inside != "true":
        return {"path": str(repo), "is_git_checkout": False, "exists": repo.exists()}
    head_rc, head = git("rev-parse", "HEAD")
    status_rc, status = git("status", "--porcelain=v1", "--untracked-files=no")
    return {
        "path": str(repo),
        "exists": repo.exists(),
        "is_git_checkout": True,
        "head": head if head_rc == 0 else None,
        "dirty": bool(status) if status_rc == 0 else None,
        "status_tracked_only": status.splitlines() if status_rc == 0 else [],
    }


def should_evaluate_heldout_target(matched_contract: str) -> bool:
    """Held-out PN2021 is a separate stage for the matched contract."""

    return str(matched_contract) != MATCHED_ECGFOUNDER_CONTRACT_VERSION


def source_fold9_selection_result(
    *, best_metric: float, candidate_fold9_metric: float, epoch: int
) -> dict[str, Any]:
    """Return the source selection decision without accepting fold-10 input."""

    return {
        "epoch": int(epoch),
        "candidate_metric": float(candidate_fold9_metric),
        "selection_data": "ptbxl_fold9",
        "report_only_data": ["ptbxl_fold10"],
        "source_floor_passed": True,
        "selected": float(candidate_fold9_metric) > float(best_metric),
    }


def validate_matched_runtime_args(args: Any) -> MatchedEffnetArm | None:
    """Fail closed when a managed matched ECGFounder command drifts."""

    if str(getattr(args, "matched_contract", "")) != MATCHED_ECGFOUNDER_CONTRACT_VERSION:
        return None
    if args.stage == "ptbxl_source":
        if args.init_model_path or args.enable_vae_adv_stream or args.enable_latent_augmix_branch:
            raise ValueError("matched ptbxl_source must be a fresh source-only fullFT run")
        if args.selection_metric != "macro_auprc":
            raise ValueError("matched source selection must use fold9 macro_auprc")
        return None

    row = matched_ecgfounder_arm(args.comparison_arm)
    checks = {
        "VAE component": (bool(args.enable_vae_adv_stream), row.vae_lhat),
        "raw AugMix component": (bool(args.enable_latent_augmix_branch), row.raw_augmix),
        "target_adv_fraction": (float(args.target_adv_fraction), row.target_adv_fraction),
    }
    drift = {name: values for name, values in checks.items() if values[0] != values[1]}
    if drift:
        raise ValueError(f"matched ECGFounder {args.comparison_arm} component drift: {drift}")
    fixed_checks = {
        "source best": str(args.init_model_path).endswith("/ptbxl_source_fullft/best_model.pt"),
        "10 epochs": args.epochs == 10,
        "internal val fraction": args.target_real_val_fraction == 0.2,
        "batch size": args.batch_size == 64,
        "K anchor": args.k_anchor == 160,
        "learning rate": args.lr == 0.00002,
        "full auxiliary budget": args.latent_augmix_consistency_max_batches == 0,
        "no extra VAE consistency": args.vae_adv_consistency_weight == 0.0,
        "split seed": args.target_real_val_seed == args.seed,
        "selection metric": args.selection_metric == "macro_auprc",
        "source floor": args.source_floor_max_drop == 0.02,
    }
    failed = [name for name, passed in fixed_checks.items() if not passed]
    if failed:
        raise ValueError(f"matched ECGFounder fixed runtime contract drift: {failed}")
    if row.vae_lhat:
        geometry = {
            "hull_label_mode": (args.hull_label_mode, "exact"),
            "hull_include_anchor": (args.hull_include_anchor, False),
            "hull_m": (args.hull_m, 20),
            "hull_lambda": (args.hull_lambda, 0.6),
            "hull_steps": (args.hull_steps, 5),
            "hull_init_logit_gap": (args.hull_init_logit_gap, 0.0),
            "hull_neighbor_distance_space": (args.hull_neighbor_distance_space, "standardized"),
            "hull_neighbor_mode": (args.hull_neighbor_mode, "local_random"),
        }
        geometry_drift = {name: values for name, values in geometry.items() if values[0] != values[1]}
        if geometry_drift:
            raise ValueError(f"matched ECGFounder latent geometry drift: {geometry_drift}")
    if row.raw_augmix and (
        args.latent_augmix_copies != 2
        or args.latent_augmix_bce_weight != 1.0
        or args.latent_augmix_consistency_weight != 2.0
        or args.latent_augmix_consistency_loss != "jsd"
    ):
        raise ValueError("matched ECGFounder A5 requires copies=2, view BCE=1, and JSD=2")
    return row


def validate_matched_ecgfounder_case(case: Mapping[str, Any]) -> tuple[str, MatchedEffnetArm]:
    """Validate the intentionally tiny managed matrix case surface."""

    if set(case) != {"arm"}:
        raise ValueError("matched ECGFounder matrix cases may declare only arm")
    arm = str(case.get("arm") or "")
    return arm, matched_ecgfounder_arm(arm)


def build_matched_k500_contract(
    record_ids: np.ndarray,
    labels: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Build the deterministic K500 split and train-only exact eligibility."""

    ids = np.asarray(record_ids, dtype=str)
    labels_arr = np.asarray(labels, dtype=np.float32)
    from ecg_adv_gen.adaptation.lhat import build_exact_eligibility_manifest
    from ecg_adv_gen.data.kshot import matched_k500_split

    split = matched_k500_split(ids, labels_arr, val_fraction=val_fraction, seed=seed)
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    eligibility = build_exact_eligibility_manifest(
        labels_arr[train_indices],
        ids[train_indices],
        min_nonself=2,
    )
    return {"split": split, "exact_eligibility": eligibility}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_view_anchor_identity(
    record_ids: list[str] | np.ndarray,
    *,
    train_record_ids: set[str],
    val_record_ids: set[str],
) -> dict[str, Any]:
    """Validate and hash ordered VAE/view anchors against the internal K500 split."""

    ordered = [str(value) for value in np.asarray(record_ids).astype(str).tolist()]
    train = {str(value) for value in train_record_ids}
    val = {str(value) for value in val_record_ids}
    outside_train = sorted(set(ordered).difference(train))
    val_overlap = sorted(set(ordered).intersection(val))
    if outside_train or val_overlap:
        raise ValueError(
            "matched VAE/view anchors leave the train split: "
            f"outside_train={outside_train}, val_overlap={val_overlap}"
        )
    return {
        "record_ids_sha256": hashlib.sha256(
            ("\n".join(ordered) + "\n").encode("utf-8")
        ).hexdigest(),
        "count": len(ordered),
        "unique_count": len(set(ordered)),
        "val_overlap_count": 0,
        "scope": "k500_train_only",
    }


def load_source_checkpoint_identity(path: str | Path) -> dict[str, Any]:
    """Validate and identify the common selected PTB-XL source checkpoint."""

    checkpoint = Path(path)
    if checkpoint.name != "best_model.pt" or not checkpoint.is_file():
        raise ValueError("matched ECGFounder source must be the selected best_model.pt")
    result_path = checkpoint.parent / "eval_result.json"
    if not result_path.is_file():
        raise ValueError(f"source checkpoint has no eval_result.json: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selection = result.get("selection") or {}
    if result.get("stage") != "ptbxl_source":
        raise ValueError("matched ECGFounder initialization must come from stage ptbxl_source")
    if result.get("selected_checkpoint") != checkpoint.name:
        raise ValueError("source eval_result selected checkpoint does not match best_model.pt")
    if selection.get("selection_data") != "ptbxl_fold9":
        raise ValueError("source checkpoint selection must use ptbxl_fold9")
    if selection.get("report_only_data") != ["ptbxl_fold10"]:
        raise ValueError("source fold10 must be report-only")
    return {
        "path": str(checkpoint),
        "sha256": _file_sha256(checkpoint),
        "stage": "ptbxl_source",
        "selected_checkpoint": checkpoint.name,
        "selection_metric": str(selection.get("metric") or ""),
        "selection_data": "ptbxl_fold9",
        "report_only_data": ["ptbxl_fold10"],
        "best_epoch": selection.get("best_epoch"),
    }


def build_matched_ecgfounder_training_record(
    *,
    comparison_arm: str,
    source_checkpoint: Mapping[str, Any],
    split: Mapping[str, Any],
    exact_eligibility: Mapping[str, Any],
    selection_metric: str,
    source_floor_max_drop: float,
    source_floor_result: Mapping[str, Any],
    epochs: int,
    optimizer_steps_per_epoch: int,
    realized_optimizer_steps: int,
    actual_param_update_steps: int,
    scheduler_steps: int,
    realized_stream_counts: Mapping[str, int],
    realized_target_adv_fraction: float,
    auxiliary_clean_exposure: list[Mapping[str, Any]],
    view_anchor_exposure: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Serialize the matched training facts consumed by audits and reports."""

    components = matched_ecgfounder_arm(comparison_arm)
    if source_checkpoint.get("stage") != "ptbxl_source":
        raise ValueError("matched ECGFounder source checkpoint must record ptbxl_source")
    if source_checkpoint.get("selected_checkpoint") != "best_model.pt":
        raise ValueError("matched ECGFounder source checkpoint must be best_model.pt")
    return {
        "contract": MATCHED_ECGFOUNDER_CONTRACT_VERSION,
        "comparison_arm": str(comparison_arm),
        "role": components.role,
        "method_components": {
            "vae_lhat": components.vae_lhat,
            "raw_augmix": components.raw_augmix,
            "augmix_view_bce": components.augmix_view_bce,
            "jsd": components.jsd,
        },
        "target_adv_fraction": components.target_adv_fraction,
        "source_checkpoint": dict(source_checkpoint),
        "k500_split": {
            "train_count": len(split["train_record_ids"]),
            "val_count": len(split["val_record_ids"]),
            "train_record_ids_sha256": str(split["train_record_ids_sha256"]),
            "val_record_ids_sha256": str(split["val_record_ids_sha256"]),
            "validation_fraction": float(split["val_fraction"]),
            "seed": int(split["seed"]),
        },
        "exact_eligibility": {
            "eligible_count": int(exact_eligibility["eligible_count"]),
            "manifest_sha256": str(exact_eligibility["manifest_sha256"]),
        },
        "selection": {
            "metric": str(selection_metric),
            "source_floor_max_drop": float(source_floor_max_drop),
            "checkpoint": "best_model.pt",
            "source_floor_result": dict(source_floor_result),
        },
        "budget": {
            "epochs": int(epochs),
            "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
            "realized_optimizer_steps": int(realized_optimizer_steps),
            "actual_param_update_steps": int(actual_param_update_steps),
            "scheduler_steps": int(scheduler_steps),
        },
        "auxiliary_clean_exposure": [dict(item) for item in auxiliary_clean_exposure],
        "view_anchor_exposure": [dict(item) for item in view_anchor_exposure],
        "stream": {
            "realized_counts": {key: int(value) for key, value in realized_stream_counts.items()},
            "realized_target_adv_fraction": float(realized_target_adv_fraction),
        },
    }


__all__ = [
    "MATCHED_ECGFOUNDER_ARM_COMPONENTS",
    "MATCHED_ECGFOUNDER_ARMS",
    "MATCHED_ECGFOUNDER_CONTRACT_VERSION",
    "build_matched_ecgfounder_training_record",
    "build_matched_k500_contract",
    "build_view_anchor_identity",
    "external_repository_provenance",
    "load_source_checkpoint_identity",
    "matched_ecgfounder_arm",
    "should_evaluate_heldout_target",
    "source_fold9_selection_result",
    "validate_matched_runtime_args",
    "validate_matched_ecgfounder_case",
]
