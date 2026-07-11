"""Pure contracts for the isolated F005 anchor-geometry mechanism study."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, NamedTuple

from ecg_adv_gen.matched_effnet import matched_effnet_arm, validate_matched_effnet_case


F005_STUDY_SCOPE = "f005_anchor_geometry_control"
F005_BASE_TOPOLOGY = "a5"
F005_CONTROL_VERSION = "f005_anchor_geometry_exact_pair_v1"
F005_DEFAULT_SEEDS = (20260531, 20260601, 20260611)
F005_CONSUMERS = ("clean", "s5", "depth23")


class F005Variant(NamedTuple):
    hull_lambda: float
    label_mode: str
    include_anchor: bool
    base_topology: str


F005_VARIANTS = {
    "balanced_l060": F005Variant(0.60, "exact", False, F005_BASE_TOPOLOGY),
    "neighbor_only_l100": F005Variant(1.00, "exact", False, F005_BASE_TOPOLOGY),
}


def f005_variant(name: str) -> F005Variant:
    try:
        return F005_VARIANTS[str(name)]
    except KeyError:
        raise ValueError(f"unknown F005 variant: {name!r}") from None


def validate_f005_case(case: Mapping[str, Any]) -> tuple[str, int, F005Variant]:
    """Validate one atomic seed x variant A5 study row."""
    if str(case.get("study_scope") or "") != F005_STUDY_SCOPE:
        raise ValueError(f"F005 study_scope must be {F005_STUDY_SCOPE!r}")
    if str(case.get("base_topology") or "") != F005_BASE_TOPOLOGY:
        raise ValueError("F005 base topology must be canonical A5")
    if str(case.get("arm") or "") != F005_BASE_TOPOLOGY:
        raise ValueError("F005 base topology requires arm='a5'")
    if case.get("primary_comparison_eligible") is not False:
        raise ValueError("F005 primary_comparison_eligible must be false")
    if "hull_lambda" in case:
        raise ValueError("F005 forbids arbitrary case-level hull_lambda")
    validate_matched_effnet_case(case)
    variant_name = str(case.get("variant") or "")
    variant = f005_variant(variant_name)
    try:
        seed = int(case["seed"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("F005 case seed must be an integer") from None
    return variant_name, seed, variant


def validate_f005_matrix_cases(
    cases: Iterable[Mapping[str, Any]],
    *,
    expected_seeds: Iterable[int],
    expected_variants: Iterable[str] = F005_VARIANTS,
) -> None:
    expected = {
        (int(seed), str(variant))
        for seed in expected_seeds
        for variant in expected_variants
    }
    seen: set[tuple[int, str]] = set()
    for case in cases:
        variant, seed, _ = validate_f005_case(case)
        key = (seed, variant)
        if key in seen:
            raise ValueError(f"F005 matrix contains duplicate seed x variant cell {key!r}")
        seen.add(key)
    if seen != expected:
        missing, extra = sorted(expected - seen), sorted(seen - expected)
        raise ValueError(
            "F005 matrix must contain the complete seed x variant surface; "
            f"missing={missing}, extra={extra}"
        )


def validate_f005_runtime(
    *,
    study_scope: str,
    mechanism_variant: str,
    comparison_arm: str,
    hull_lambda: float,
    hull_label_mode: str,
    hull_include_anchor: bool,
) -> F005Variant | None:
    if not study_scope and not mechanism_variant:
        return None
    if study_scope != F005_STUDY_SCOPE:
        raise ValueError(f"unknown mechanism study scope: {study_scope!r}")
    variant = f005_variant(mechanism_variant)
    matched_effnet_arm(comparison_arm)
    drift = {
        "comparison_arm": (str(comparison_arm), F005_BASE_TOPOLOGY),
        "hull_lambda": (float(hull_lambda), variant.hull_lambda),
        "hull_label_mode": (str(hull_label_mode), variant.label_mode),
        "hull_include_anchor": (bool(hull_include_anchor), variant.include_anchor),
    }
    drift = {key: values for key, values in drift.items() if values[0] != values[1]}
    if drift:
        raise ValueError(f"F005 runtime contract drift: {drift}")
    return variant


def build_f005_control_run_leaf(center: str, seed: int, variant: str, *, epochs: int) -> str:
    spec = f005_variant(variant)
    lambda_tag = f"{spec.hull_lambda:g}".replace(".", "p")
    return (
        f"{F005_STUDY_SCOPE}_{center}_a5_{variant}_lam{lambda_tag}"
        f"_exact_noanchor_ep{int(epochs)}_seed{int(seed)}"
    )


def _require_f005_record(record: Mapping[str, Any]) -> tuple[str, int, str]:
    center = str(record.get("center") or "")
    seed = int(record.get("seed", -1))
    variant_name = str(record.get("variant") or "")
    spec = f005_variant(variant_name)
    if record.get("study_scope") != F005_STUDY_SCOPE:
        raise ValueError("F005 bundle contains a foreign study scope")
    if record.get("base_topology") != F005_BASE_TOPOLOGY:
        raise ValueError("F005 bundle base topology must be A5")
    if record.get("primary_comparison_eligible") is not False:
        raise ValueError("F005 controls cannot be primary-comparison eligible")
    if record.get("hull_label_mode") != "exact" or bool(record.get("hull_include_anchor")):
        raise ValueError("F005 paired bundle requires exact/no-anchor controls")
    if float(record.get("hull_lambda", -1)) != spec.hull_lambda:
        raise ValueError(f"F005 variant {variant_name} has the wrong hull lambda")
    consumers = record.get("consumers") or {}
    if set(consumers) != set(F005_CONSUMERS) or any(
        (consumers.get(name) or {}).get("status") != "complete" for name in F005_CONSUMERS
    ):
        raise ValueError("F005 record consumers must contain complete clean/S5/depth23 results")
    attack = record.get("attack_audit") or {}
    if int(attack.get("training_steps", 0)) != 5 or int(attack.get("audit_steps", 0)) != 20:
        raise ValueError("F005 record requires the fixed 5-step/20-step attack audit")
    audit_gain = float(attack.get("audit_loss_gain_median", 0.0))
    train_gain = float(attack.get("training_loss_gain_median", 0.0))
    if audit_gain <= 0 or train_gain / audit_gain < 0.8 - 1e-12:
        raise ValueError("F005 five-step loss gain must retain at least 80% of the 20-step audit")
    return center, seed, variant_name


def validate_f005_paired_bundle(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_centers: Iterable[str] = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"),
    expected_seeds: Iterable[int] = F005_DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Fail closed unless records form the complete paired exact/exact study."""
    rows = list(records)
    expected = {
        (str(center), int(seed), variant)
        for center in expected_centers
        for seed in expected_seeds
        for variant in F005_VARIANTS
    }
    by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    retentions: list[float] = []
    for row in rows:
        key = _require_f005_record(row)
        if key in by_key:
            raise ValueError(f"duplicate F005 bundle record: {key}")
        by_key[key] = row
        attack = row["attack_audit"]
        retentions.append(
            float(attack["training_loss_gain_median"])
            / float(attack["audit_loss_gain_median"])
        )
    if set(by_key) != expected:
        raise ValueError("F005 bundle requires the complete paired center x seed x variant surface")
    git_shas = {str(row.get("git_sha") or "") for row in rows}
    if len(git_shas) != 1 or "" in git_shas:
        raise ValueError(f"F005 bundle contains mixed git commits: {sorted(git_shas)}")

    paired: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for (center, seed, _), row in by_key.items():
        paired[(center, seed)].append(row)
    shared_fields = (
        "source_checkpoint_sha256", "k500_train_ids_sha256",
        "eligibility_manifest_sha256", "selection_metric",
        "source_floor_max_drop", "realized_optimizer_steps", "scheduler_steps",
    )
    for pair_key, pair_rows in paired.items():
        for field in shared_fields:
            values = {json_safe(row.get(field)) for row in pair_rows}
            if len(values) != 1 or None in values or "" in values:
                raise ValueError(f"F005 pair {pair_key} has unmatched {field}: {values}")
    return {
        "schema_version": 1,
        "contract_version": F005_CONTROL_VERSION,
        "study_scope": F005_STUDY_SCOPE,
        "variants": list(F005_VARIANTS),
        "n_records": len(rows),
        "n_pairs": len(paired),
        "git_sha": next(iter(git_shas)),
        "five_step_retention_min": min(retentions),
        "primary_comparison_eligible": False,
    }


def json_safe(value: Any) -> Any:
    """Return a hashable comparison value for small manifest fields."""
    if isinstance(value, list):
        return tuple(json_safe(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), json_safe(item)) for key, item in value.items()))
    return value


__all__ = [
    "F005_BASE_TOPOLOGY", "F005_CONTROL_VERSION", "F005_DEFAULT_SEEDS",
    "F005_STUDY_SCOPE", "F005_VARIANTS", "build_f005_control_run_leaf",
    "f005_variant", "validate_f005_case", "validate_f005_matrix_cases",
    "validate_f005_paired_bundle", "validate_f005_runtime",
]
