"""Stable PTB-XL Super5 metadata used by configs and paper exports.

This module intentionally contains only metadata needed by the production
wrapper/reporting layer. Full label-conversion logic remains in the legacy
label scheme module until the data pipeline is extracted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CLASS_NAMES_SUPER5 = ("CD", "HYP", "MI", "NORM", "STTC")
NUM_SUPER5 = len(CLASS_NAMES_SUPER5)

SUPER5_PN2021_MAPPING_VERSION = "v7_super5_sjr_rgq_review_20260528"
SUPER5_PN2021_MAPPING_HASH = "555ec85d5b51"


class Super5MetadataError(ValueError):
    """Raised when config or metric artifacts disagree with Super5 metadata."""


@dataclass(frozen=True)
class Super5Metadata:
    mapping_version: str
    mapping_hash: str
    class_order: tuple[str, ...]
    num_classes: int


def get_super5_metadata() -> Super5Metadata:
    return Super5Metadata(
        mapping_version=SUPER5_PN2021_MAPPING_VERSION,
        mapping_hash=SUPER5_PN2021_MAPPING_HASH,
        class_order=CLASS_NAMES_SUPER5,
        num_classes=NUM_SUPER5,
    )


def default_class_order() -> list[str]:
    return list(CLASS_NAMES_SUPER5)


def pn2021_super5_label_mapping_payload() -> dict[str, dict[str, str]]:
    """Return the canonical label-mapping block stored in metric artifacts."""
    metadata = get_super5_metadata()
    return {
        "pn2021_super5": {
            "mapping_version": metadata.mapping_version,
            "mapping_hash": metadata.mapping_hash,
        }
    }


def validate_super5_metadata(
    *,
    mapping_version: str,
    mapping_hash: str,
    class_order: Iterable[str],
    num_classes: int,
) -> None:
    expected = get_super5_metadata()
    if mapping_version != expected.mapping_version:
        raise Super5MetadataError(
            f"Mapping version mismatch: config={mapping_version} code={expected.mapping_version}"
        )
    if mapping_hash != expected.mapping_hash:
        raise Super5MetadataError(
            f"Mapping hash mismatch: config={mapping_hash} code={expected.mapping_hash}"
        )
    got_order = tuple(class_order)
    if got_order != expected.class_order:
        raise Super5MetadataError(
            f"Class order mismatch: config={list(got_order)} code={list(expected.class_order)}"
        )
    if int(num_classes) != expected.num_classes:
        raise Super5MetadataError(
            f"num_classes mismatch: config={num_classes} code={expected.num_classes}"
        )
