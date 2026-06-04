#!/usr/bin/env python3
"""Export PN2021 percent-shot ECGTwin anchors under the v7 Super5 mapping.

The current K500 mainline uses real target-center ECGTwin VAE anchors plus
matching 100 Hz classifier signals. This helper prepares the same artifact
contract for per-center percentage protocols such as 10% and 20%.

Selection is deterministic random sampling from each center's v7 Super5
non-zero pool, restricted to records that also exist in the ECGTwin full-latent
cache. K is computed from the raw v7 non-zero count, while export falls back to
the VAE-compatible eligible pool when a small number of records failed ECGTwin
preprocessing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.data.kshot import center_offset_seed  # noqa: E402
from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5,
    get_super5_pn2021_mapping_metadata,
)


DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
DEFAULT_CACHE_ROOT = DATA_ROOT / "ecgtwin_prompt_token_super5/cache_v1"
DEFAULT_MMAP_ROOT = DATA_ROOT / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]


def _percent_tag(percent: float) -> str:
    value = percent * 100.0
    if abs(value - round(value)) < 1e-9:
        return f"p{int(round(value))}"
    return ("p" + f"{value:g}").replace(".", "p")


def _requested_k(nonzero_count: int, percent: float) -> int:
    return max(1, int(math.floor(float(nonzero_count) * float(percent) + 0.5)))


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_center_cache(cache_root: Path, center: str) -> dict[str, Any]:
    path = cache_root / "center_full_latents" / f"{center}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_mmap_center(mmap_root: Path, center: str) -> dict[str, Any]:
    mapping = get_super5_pn2021_mapping_metadata()
    version = str(mapping["mapping_version"]).replace("_20260528", "")
    mmap_dir = mmap_root / f"super5_{center}_100hz1000_{version}"
    if not mmap_dir.exists():
        raise FileNotFoundError(mmap_dir)
    return {
        "dir": mmap_dir,
        "signals": np.load(mmap_dir / "signals.npy", mmap_mode="r"),
        "labels": np.load(mmap_dir / "labels.npy", mmap_mode="r"),
        "record_ids": np.load(mmap_dir / "record_ids.npy", allow_pickle=True).astype(str),
        "metadata_path": mmap_dir / "metadata.json",
    }


def export_center_percent(
    *,
    center: str,
    percent: float | None,
    fixed_k: int | None = None,
    seed: int,
    cache_root: Path,
    mmap_root: Path,
    output_root: Path,
    force: bool,
    canonical_alias: bool = True,
) -> dict[str, Any]:
    class_names = list(CLASS_NAMES_SUPER5)
    mapping = get_super5_pn2021_mapping_metadata()
    cache = _load_center_cache(cache_root, center)
    mmap = _load_mmap_center(mmap_root, center)

    cache_record_ids = [str(x) for x in cache["record_ids"]]
    cache_id_to_idx = {rid: i for i, rid in enumerate(cache_record_ids)}
    mmap_record_ids = mmap["record_ids"]
    labels_mmap = mmap["labels"]
    nonzero_mask = np.asarray(labels_mmap).sum(axis=1) > 0
    raw_nonzero_count = int(nonzero_mask.sum())
    if fixed_k is not None:
        k = int(fixed_k)
        protocol = "fixed_k"
        protocol_tag = f"k{k}"
        if k <= 0:
            raise ValueError(f"fixed K must be positive, got {k}")
    else:
        if percent is None:
            raise ValueError("percent is required when fixed_k is not set")
        k = _requested_k(raw_nonzero_count, percent)
        protocol = "percent"
        protocol_tag = _percent_tag(percent)

    eligible_rows: list[int] = []
    eligible_cache_indices: list[int] = []
    missing_latent_ids: list[str] = []
    for row_idx, rid in enumerate(mmap_record_ids):
        if not bool(nonzero_mask[row_idx]):
            continue
        cache_idx = cache_id_to_idx.get(str(rid))
        if cache_idx is None:
            missing_latent_ids.append(str(rid))
            continue
        eligible_rows.append(row_idx)
        eligible_cache_indices.append(cache_idx)
    if len(eligible_rows) < k:
        raise RuntimeError(
            f"{center}: requested K={k} from {raw_nonzero_count} v7 non-zero records, "
            f"but only {len(eligible_rows)} records have ECGTwin latents"
        )

    protocol_seed = int(fixed_k) * 13 if fixed_k is not None else int(round(float(percent) * 10000))
    rng = np.random.default_rng(int(seed) + center_offset_seed(center) + protocol_seed)
    chosen_local = np.sort(rng.choice(np.arange(len(eligible_rows)), size=k, replace=False))
    chosen_rows = np.asarray([eligible_rows[int(i)] for i in chosen_local], dtype=np.int64)
    chosen_cache_idx = np.asarray([eligible_cache_indices[int(i)] for i in chosen_local], dtype=np.int64)

    signals = np.asarray(mmap["signals"][chosen_rows], dtype=np.float32)
    labels = np.asarray(labels_mmap[chosen_rows], dtype=np.float32)
    record_ids = mmap_record_ids[chosen_rows].astype(str)
    latents = _to_numpy(cache["latents"])[chosen_cache_idx].astype(np.float32, copy=False)
    if signals.shape != (k, 1000, 12):
        raise ValueError(f"{center}: expected signals {(k, 1000, 12)}, got {signals.shape}")
    if latents.shape != (k, 4, 128):
        raise ValueError(f"{center}: expected latents {(k, 4, 128)}, got {latents.shape}")
    if labels.shape != (k, len(class_names)):
        raise ValueError(f"{center}: expected labels {(k, len(class_names))}, got {labels.shape}")
    if int((labels.sum(axis=1) > 0).sum()) != k:
        raise ValueError(f"{center}: selected rows include all-zero labels")

    if fixed_k is not None:
        subset_dir = output_root / center / f"k{k}_seed{seed}"
    else:
        subset_dir = output_root / center / f"k{k}_seed{seed}_{protocol_tag}"
    base = subset_dir / f"{center}_real_k{k}_seed{seed}"
    paths = {
        "signals": base.with_suffix(".signals.npz"),
        "latents": base.with_suffix(".latent.npz"),
        "meta": base.with_suffix(".ref_meta.json"),
        "trust": base.with_suffix(".class_trust.json"),
    }
    protocol_label = f"k{int(fixed_k)}" if fixed_k is not None else _percent_tag(float(percent))
    if all(path.exists() for path in paths.values()) and not force:
        print(f"[skip] {center} {protocol_label} K={k}: {base}", flush=True)
    else:
        subset_dir.mkdir(parents=True, exist_ok=True)
        primary_class = np.asarray([class_names[int(np.argmax(row))] for row in labels])
        np.savez_compressed(
            paths["signals"],
            signals=signals,
            labels=labels,
            record_ids=record_ids,
            center_name=np.asarray(center),
            class_names=np.asarray(class_names),
            mapping_version=np.asarray(str(mapping["mapping_version"])),
            mapping_hash=np.asarray(str(mapping["mapping_hash"])),
        )
        np.savez_compressed(
            paths["latents"],
            latents=latents,
            labels=labels,
            center_name=np.asarray(center),
            class_names=np.asarray(class_names),
            source_indices=chosen_cache_idx,
            record_ids=record_ids,
            primary_class=primary_class,
            source_ids=np.zeros((k,), dtype=np.int16),
            source_names=np.asarray(["real_anchor"]),
            source_local_indices=np.arange(k, dtype=np.int32),
            mapping_version=np.asarray(str(mapping["mapping_version"])),
            mapping_hash=np.asarray(str(mapping["mapping_hash"])),
        )
        label_counts = labels.sum(axis=0).astype(int).tolist()
        meta = {
            "center": center,
            "K": int(k),
            "protocol": protocol,
            "percent": None if percent is None else float(percent),
            "fixed_k": None if fixed_k is None else int(fixed_k),
            "protocol_tag": protocol_tag,
            "selection_seed": int(seed),
            "raw_v7_nonzero_count": raw_nonzero_count,
            "eligible_v7_nonzero_with_latents": int(len(eligible_rows)),
            "missing_latent_count": int(len(missing_latent_ids)),
            "missing_latent_record_ids_head": missing_latent_ids[:10],
            "sampling_policy": "deterministic_random_from_v7_nonzero_eligible_pool",
            "denominator_policy": (
                "K=round(percent * raw center v7 Super5 non-zero count)"
                if fixed_k is None
                else "fixed user-requested K"
            ),
            "ref_record_ids": record_ids.tolist(),
            "mmap_row_indices": chosen_rows.astype(int).tolist(),
            "source_indices": chosen_cache_idx.astype(int).tolist(),
            "source_full_latent_cache": str(cache_root / "center_full_latents" / f"{center}.pt"),
            "source_mmap_dir": str(mmap["dir"]),
            "mapping_version": str(mapping["mapping_version"]),
            "mapping_hash": str(mapping["mapping_hash"]),
            "class_names": class_names,
            "label_counts": {cls: int(v) for cls, v in zip(class_names, label_counts)},
            "note": (
                "Percent-shot protocol for Direct/VAE matched runs. Final PN2021 "
                "evaluation should exclude every ref_record_id listed here."
            ),
        }
        paths["meta"].write_text(
            json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        class_trust = {
            cls: (1.0 if int(count) > 0 else 0.0)
            for cls, count in zip(class_names, label_counts)
        }
        paths["trust"].write_text(
            json.dumps(
            {
                "center": center,
                "tag": f"{center}_real_k{k}_{protocol_tag}",
                "mapping_version": str(mapping["mapping_version"]),
                    "mapping_hash": str(mapping["mapping_hash"]),
                    "policy": "real_all_present under v7 percent-shot subset",
                    "class_trust": class_trust,
                    "label_counts": {cls: int(v) for cls, v in zip(class_names, label_counts)},
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[write] {center} {protocol_tag} K={k} "
            f"counts={dict(zip(class_names, label_counts))} -> {base}",
            flush=True,
        )

    alias_dir = output_root / center / f"k{k}_seed{seed}"
    if canonical_alias and alias_dir != subset_dir:
        if alias_dir.is_symlink():
            current = alias_dir.readlink()
            if current != Path(subset_dir.name):
                if force:
                    alias_dir.unlink()
                    alias_dir.symlink_to(Path(subset_dir.name), target_is_directory=True)
                else:
                    raise RuntimeError(f"{alias_dir} already points to {current}, not {subset_dir.name}")
        elif alias_dir.exists():
            if not force:
                raise RuntimeError(f"{alias_dir} exists and is not a symlink; use --force to replace only if safe")
            raise RuntimeError(f"Refusing to replace non-symlink canonical alias: {alias_dir}")
        else:
            alias_dir.symlink_to(Path(subset_dir.name), target_is_directory=True)

    return {
        "center": center,
        "protocol": protocol,
        "percent": None if percent is None else float(percent),
        "fixed_k": None if fixed_k is None else int(fixed_k),
        "protocol_tag": protocol_tag,
        "K": int(k),
        "raw_v7_nonzero_count": raw_nonzero_count,
        "eligible_v7_nonzero_with_latents": int(len(eligible_rows)),
        "missing_latent_count": int(len(missing_latent_ids)),
        "base": str(base),
        "paths": {key: str(value) for key, value in paths.items()},
        "label_counts": {
            cls: int(v)
            for cls, v in zip(class_names, np.asarray(labels).sum(axis=0).astype(int).tolist())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    parser.add_argument("--percents", nargs="*", type=float, default=[0.10, 0.20])
    parser.add_argument("--fixed-ks", nargs="+", type=int, default=[])
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--mmap-root", default=str(DEFAULT_MMAP_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-canonical-alias", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    mmap_root = Path(args.mmap_root)
    output_root = Path(args.output_root)
    summaries: list[dict[str, Any]] = []
    for percent in args.percents:
        if not (0.0 < float(percent) <= 1.0):
            raise ValueError(f"percent must be in (0,1], got {percent}")
        for center in args.centers:
            summaries.append(
                export_center_percent(
                    center=center,
                    percent=float(percent),
                    fixed_k=None,
                    seed=int(args.seed),
                    cache_root=cache_root,
                    mmap_root=mmap_root,
                    output_root=output_root,
                    force=bool(args.force),
                    canonical_alias=not bool(args.no_canonical_alias),
                )
            )
    for fixed_k in args.fixed_ks:
        for center in args.centers:
            summaries.append(
                export_center_percent(
                    center=center,
                    percent=None,
                    fixed_k=int(fixed_k),
                    seed=int(args.seed),
                    cache_root=cache_root,
                    mmap_root=mmap_root,
                    output_root=output_root,
                    force=bool(args.force),
                    canonical_alias=not bool(args.no_canonical_alias),
                )
            )
    summary_path = Path(args.summary_path) if args.summary_path else (
        output_root / f"percent_shot_v7_seed{args.seed}_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "mapping": get_super5_pn2021_mapping_metadata(),
                "seed": int(args.seed),
                "selection_policy": "deterministic_random_from_v7_nonzero_eligible_pool",
                "denominator_policy": "percent runs use rounded nonzero count; fixed_k runs use requested K",
                "cache_root": str(cache_root),
                "mmap_root": str(mmap_root),
                "output_root": str(output_root),
                "items": summaries,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
