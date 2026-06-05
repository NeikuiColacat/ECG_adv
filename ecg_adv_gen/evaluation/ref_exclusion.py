"""Helpers for paper-safe PN2021 K-shot reference exclusion."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ecg_adv_gen.data import PN2021_TARGET_CENTERS_4, kshot_ref_meta_path

_KSHOT_DIR_RE = re.compile(r"^k(?P<k>\d+)_seed(?P<seed>\d+)$")


def target_ref_meta_paths(
    ref_root: str | Path,
    *,
    k: int,
    seed: int,
    centers: Iterable[str] = PN2021_TARGET_CENTERS_4,
) -> list[Path]:
    """Return canonical ref-meta paths for every target center in paper eval."""

    return [kshot_ref_meta_path(ref_root, center, k, seed) for center in centers]


def append_target_ref_exclusion_args(
    argv: list[str],
    ref_root: str | Path,
    *,
    k: int,
    seed: int,
    centers: Iterable[str] = PN2021_TARGET_CENTERS_4,
) -> None:
    """Append ``--exclude_ref_ids`` with all paper target-center K-shot refs."""

    argv.append("--exclude_ref_ids")
    argv.extend(str(path) for path in target_ref_meta_paths(ref_root, k=k, seed=seed, centers=centers))


def target_ref_meta_paths_from_anchor_base(
    anchor_base: str | Path,
    *,
    centers: Iterable[str] = PN2021_TARGET_CENTERS_4,
) -> list[Path]:
    """Return all target ref-meta paths from a canonical K-shot anchor base."""

    base = Path(anchor_base)
    match = _KSHOT_DIR_RE.match(base.parent.name)
    if not match:
        raise ValueError(f"anchor_base parent must look like k<K>_seed<SEED>: {base}")
    ref_root = base.parents[2]
    return target_ref_meta_paths(
        ref_root,
        k=int(match.group("k")),
        seed=int(match.group("seed")),
        centers=centers,
    )


def append_target_ref_exclusion_args_from_anchor_base(
    argv: list[str],
    anchor_base: str | Path,
    *,
    centers: Iterable[str] = PN2021_TARGET_CENTERS_4,
) -> None:
    """Append target-center ref exclusions derived from a K-shot anchor base."""

    argv.append("--exclude_ref_ids")
    argv.extend(
        str(path)
        for path in target_ref_meta_paths_from_anchor_base(anchor_base, centers=centers)
    )
