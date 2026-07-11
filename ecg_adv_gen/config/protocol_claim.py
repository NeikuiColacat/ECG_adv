"""Pure operator-set resolution for the narrowed PN2021-C claim."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ecg_adv_gen.evaluation.pn2021c import PN2021C_OFFICIAL_OPERATORS
from ecg_adv_gen.matched_effnet import is_matched_effnet_arm, matched_effnet_arm


KNOWN_FAMILY_CLAIM_SCOPE = "known_family_corruption_robustness"


class ProtocolClaimError(ValueError):
    """Raised when a corruption claim is incomplete or over-broad."""


def resolve_operator_set(
    values: Sequence[Any], *, allowed: Sequence[str], allow_composites: bool, field: str
) -> tuple[str, ...]:
    """Validate declarations and return unique atomic operators in stable first-seen order."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ProtocolClaimError(f"{field} must be non-empty")
    if any(not isinstance(value, str) for value in values):
        raise ProtocolClaimError(f"{field} entries must be strings")
    tokens = [value.strip() for value in values]
    if any(not token for token in tokens):
        raise ProtocolClaimError(f"{field} entries must be non-empty strings")
    duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
    if duplicates:
        raise ProtocolClaimError(f"{field} contains duplicate entries: {duplicates}")
    allowed_order = tuple(str(item) for item in allowed)
    allowed_set = set(allowed_order)
    resolved: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not allow_composites and "+" in token:
            raise ProtocolClaimError(f"{field} must contain atomic operators, got {token!r}")
        atoms = token.split("+") if allow_composites else [token]
        if len(atoms) != len(set(atoms)):
            raise ProtocolClaimError(f"{field} contains duplicate atomic operators in {token!r}")
        unknown = [atom for atom in atoms if atom not in allowed_set]
        if unknown:
            raise ProtocolClaimError(f"{field} contains unknown operators: {unknown}")
        for atom in atoms:
            if atom not in seen:
                resolved.append(atom)
                seen.add(atom)
    return tuple(resolved)


def resolve_protocol_claim(
    paper_protocol: Mapping[str, Any], *, arm: str | None = None,
    evaluation_operators: Sequence[Any] | None = None,
    training_operators: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Resolve one global or per-arm claim and reject unsafe train/eval overlap."""

    scope = str(paper_protocol.get("claim_scope") or "")
    operator_sets = paper_protocol.get("operator_sets") or {}
    allowed = PN2021C_OFFICIAL_OPERATORS
    train_global = resolve_operator_set(
        operator_sets.get("train_atomic") or (), allowed=allowed,
        allow_composites=False, field="paper_protocol.operator_sets.train_atomic",
    )
    eval_declared = resolve_operator_set(
        operator_sets.get("eval_atomic") or (), allowed=allowed,
        allow_composites=False, field="paper_protocol.operator_sets.eval_atomic",
    )
    if train_global != allowed or eval_declared != allowed:
        raise ProtocolClaimError(
            "paper_protocol.operator_sets train_atomic/eval_atomic must exactly match "
            "the five official operators"
        )
    if training_operators is not None:
        executable_train = resolve_operator_set(
            training_operators, allowed=allowed, allow_composites=False,
            field="command.--latent_augmix_ops",
        )
        if executable_train != train_global:
            raise ProtocolClaimError("training command operators do not match train_atomic")
    eval_resolved = eval_declared
    if evaluation_operators is not None:
        executable_eval = resolve_operator_set(
            evaluation_operators, allowed=allowed, allow_composites=True,
            field="evaluation.corruptions",
        )
        has_composites = any(
            isinstance(operator, str) and "+" in operator
            for operator in evaluation_operators
        )
        if has_composites:
            if set(executable_eval) != set(eval_declared):
                raise ProtocolClaimError(
                    "evaluation operators must resolve to paper_protocol.operator_sets.eval_atomic"
                )
            eval_resolved = tuple(
                operator for operator in eval_declared if operator in set(executable_eval)
            )
        else:
            eval_resolved = executable_eval
            if eval_resolved != eval_declared:
                raise ProtocolClaimError(
                    "evaluation operators must resolve to paper_protocol.operator_sets.eval_atomic"
                )
    train_resolved = train_global
    if arm is not None:
        train_resolved = (
            train_global
            if is_matched_effnet_arm(arm) and matched_effnet_arm(arm).raw_augmix
            else ()
        )
    overlap = tuple(operator for operator in train_resolved if operator in set(eval_resolved))
    allowed_by_scope = not overlap or scope == KNOWN_FAMILY_CLAIM_SCOPE
    if not allowed_by_scope:
        raise ProtocolClaimError(
            f"train/eval operator overlap {list(overlap)!r} requires "
            f"claim_scope={KNOWN_FAMILY_CLAIM_SCOPE!r}"
        )
    resolved = {
        "claim_scope": scope,
        "train_operator_set_resolved": list(train_resolved),
        "eval_operator_set_resolved": list(eval_resolved),
        "overlap_operator_set_resolved": list(overlap),
        "overlap_allowed_by_claim_scope": allowed_by_scope,
    }
    if arm is not None:
        resolved = {"arm": str(arm), **resolved}
    return resolved
