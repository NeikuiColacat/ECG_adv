"""Paper-safe PN2021 evaluation protocol checks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


PAPER_EVAL_PROTOCOL = "paper_refexcluded"
DIAGNOSTIC_EVAL_PROTOCOL = "diagnostic_unrefexcluded"
ALLOWED_EVAL_PROTOCOLS = (PAPER_EVAL_PROTOCOL, DIAGNOSTIC_EVAL_PROTOCOL)


class PN2021ProtocolError(ValueError):
    """Raised when a PN2021 eval result is not paper-safe."""


def _missing_centers(per_center: Mapping[str, Any], eval_centers: Sequence[str]) -> list[str]:
    return [str(center) for center in eval_centers if str(center) not in per_center]


def normalize_paper_preprocess_mode(
    *,
    preprocess_mode: str,
    norm_mode: str,
    crop_len: int,
) -> str:
    """Return the protocol-facing preprocessing mode name for eval safety checks."""

    if (
        str(preprocess_mode) == "minimal_resample"
        and str(norm_mode) == "per_sample_global"
        and int(crop_len) == 1000
    ):
        return "paper"
    return str(preprocess_mode)


def validate_pn2021_eval_protocol(
    *,
    eval_protocol: str,
    per_center: Mapping[str, Mapping[str, Any]],
    target_centers: Sequence[str],
    eval_centers: Sequence[str],
    preprocess_mode: str,
    crop_len: int,
    diagnostic_legacy_eval: bool = False,
    allow_missing_centers: bool = False,
    min_target_ref_excluded: int = 500,
) -> dict[str, Any]:
    """Validate that PN2021 metrics satisfy the requested protocol."""

    if eval_protocol not in ALLOWED_EVAL_PROTOCOLS:
        allowed = ", ".join(ALLOWED_EVAL_PROTOCOLS)
        raise PN2021ProtocolError(f"eval_protocol must be one of {allowed}, got {eval_protocol!r}")

    missing = _missing_centers(per_center, eval_centers)
    if missing and eval_protocol == PAPER_EVAL_PROTOCOL:
        raise PN2021ProtocolError(f"missing eval centers: {missing}")
    if missing and not allow_missing_centers:
        raise PN2021ProtocolError(f"missing eval centers: {missing}")

    if eval_protocol == DIAGNOSTIC_EVAL_PROTOCOL:
        if preprocess_mode != "paper" and not diagnostic_legacy_eval:
            raise PN2021ProtocolError(
                "legacy preprocessing requires --diagnostic_legacy_eval in diagnostic_unrefexcluded mode"
            )
        return {
            "status": "diagnostic",
            "eval_protocol": eval_protocol,
            "missing_eval_centers": missing,
            "target_ref_exclusion_required": False,
            "preprocess_mode": str(preprocess_mode),
            "crop_len": int(crop_len),
        }

    if preprocess_mode != "paper":
        raise PN2021ProtocolError(
            f"paper_refexcluded requires preprocess_mode='paper', got {preprocess_mode!r}"
        )
    if int(crop_len) != 1000:
        raise PN2021ProtocolError(f"paper_refexcluded requires crop_len=1000, got {crop_len}")

    ref_failures = []
    for center in target_centers:
        row = per_center.get(str(center))
        if row is None:
            ref_failures.append(f"{center}: no metrics row")
            continue
        if int(row.get("n_excluded_ref", 0)) < int(min_target_ref_excluded):
            ref_failures.append(f"{center}: n_excluded_ref={row.get('n_excluded_ref', 0)}")
    if ref_failures:
        raise PN2021ProtocolError("paper_refexcluded requires target ref exclusion: " + "; ".join(ref_failures))

    return {
        "status": "paper_safe",
        "eval_protocol": eval_protocol,
        "missing_eval_centers": missing,
        "target_ref_exclusion_required": True,
        "min_target_ref_excluded": int(min_target_ref_excluded),
        "preprocess_mode": str(preprocess_mode),
        "crop_len": int(crop_len),
    }
