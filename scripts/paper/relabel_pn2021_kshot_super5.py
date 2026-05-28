#!/usr/bin/env python3
"""Rebuild PN2021 K-shot anchor artifacts under the active Super5 mapping."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.data import scan_pn2021_center_records  # noqa: E402
from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5,
    get_scheme,
    get_super5_pn2021_mapping_metadata,
)


_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        str(_MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")),
    )
)
DEFAULT_SOURCE_ROOT = DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516/subsets"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"


def _class_names_array(class_names: Sequence[str]) -> np.ndarray:
    return np.asarray([str(c) for c in class_names])


def _primary_class(labels: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32)
    names = list(class_names)
    out: list[str] = []
    for row in labels:
        if float(np.max(row)) <= 0.0:
            out.append("ALL_ZERO")
        else:
            out.append(names[int(np.argmax(row))])
    return np.asarray(out)


def _label_counts(labels: np.ndarray, class_names: Sequence[str]) -> dict[str, int]:
    counts = np.asarray(labels).sum(axis=0).astype(int).tolist()
    return {str(cls): int(count) for cls, count in zip(class_names, counts)}


def build_label_lookup_from_headers(center_dir: Path) -> dict[str, np.ndarray]:
    scheme = get_scheme("super5")
    label_fn = scheme["pn2021_fn"]
    out: dict[str, np.ndarray] = {}
    for record in scan_pn2021_center_records(center_dir):
        out[str(record.record_id)] = np.asarray(label_fn(record.snomeds), dtype=np.float32)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _copy_npz_with_labels(
    source_path: Path,
    output_path: Path,
    *,
    labels: np.ndarray,
    mapping_metadata: Mapping[str, Any],
    class_names: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["labels"] = labels.astype(np.float32, copy=False)
    payload["class_names"] = _class_names_array(class_names)
    payload["mapping_version"] = np.asarray(str(mapping_metadata["mapping_version"]))
    payload["mapping_hash"] = np.asarray(str(mapping_metadata["mapping_hash"]))
    if "latents" in payload:
        payload["primary_class"] = _primary_class(labels, class_names)
    np.savez_compressed(output_path, **payload)


def relabel_kshot_artifact_group(
    *,
    center: str,
    source_base: Path,
    output_base: Path,
    label_by_record_id: Mapping[str, np.ndarray],
    mapping_metadata: Mapping[str, Any],
    class_names: Sequence[str] = CLASS_NAMES_SUPER5,
) -> dict[str, Any]:
    signal_path = source_base.with_suffix(".signals.npz")
    latent_path = source_base.with_suffix(".latent.npz")
    meta_path = source_base.with_suffix(".ref_meta.json")
    if not signal_path.exists() or not latent_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing source K-shot artifact group for {source_base}")

    with np.load(signal_path, allow_pickle=True) as data:
        signals = data["signals"]
        record_ids = data["record_ids"].astype(str)
    labels = []
    missing = []
    for rid in record_ids:
        label = label_by_record_id.get(str(rid))
        if label is None:
            missing.append(str(rid))
        else:
            labels.append(np.asarray(label, dtype=np.float32))
    if missing:
        raise RuntimeError(f"{center}: {len(missing)} record ids missing PN2021 labels; first={missing[:5]}")
    labels_arr = np.stack(labels).astype(np.float32)
    if labels_arr.shape != (len(record_ids), len(class_names)):
        raise ValueError(f"{center}: labels shape {labels_arr.shape} does not match records/classes")
    if len(signals) != len(record_ids):
        raise ValueError(f"{center}: signals and record_ids length mismatch")

    _copy_npz_with_labels(
        signal_path,
        output_base.with_suffix(".signals.npz"),
        labels=labels_arr,
        mapping_metadata=mapping_metadata,
        class_names=class_names,
    )
    _copy_npz_with_labels(
        latent_path,
        output_base.with_suffix(".latent.npz"),
        labels=labels_arr,
        mapping_metadata=mapping_metadata,
        class_names=class_names,
    )

    source_meta = _read_json(meta_path)
    label_counts = _label_counts(labels_arr, class_names)
    output_meta = dict(source_meta)
    output_meta.update(
        {
            "center": center,
            "ref_record_ids": record_ids.astype(str).tolist(),
            "parent": str(meta_path),
            "policy": "same K-shot record ids relabeled under active PN2021 Super5 mapping",
            "mapping_version": str(mapping_metadata["mapping_version"]),
            "mapping_hash": str(mapping_metadata["mapping_hash"]),
            "class_names": [str(c) for c in class_names],
            "label_counts": label_counts,
        }
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_base.with_suffix(".ref_meta.json").write_text(
        json.dumps(output_meta, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    class_trust = {
        str(cls): (1.0 if int(label_counts[str(cls)]) > 0 else 0.0)
        for cls in class_names
    }
    output_base.with_suffix(".class_trust.json").write_text(
        json.dumps(
            {
                "center": center,
                "mapping_version": str(mapping_metadata["mapping_version"]),
                "mapping_hash": str(mapping_metadata["mapping_hash"]),
                "policy": "real_all_present under v7 relabeled K-shot subset",
                "class_trust": class_trust,
                "label_counts": label_counts,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "center": center,
        "n": int(len(record_ids)),
        "source_base": str(source_base),
        "output_base": str(output_base),
        "mapping_version": str(mapping_metadata["mapping_version"]),
        "mapping_hash": str(mapping_metadata["mapping_hash"]),
        "label_counts": label_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centers", nargs="+", default=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"])
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pn2021-root", default=str(DATA_ROOT / "physionet2021"))
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    pn2021_root = Path(args.pn2021_root)
    mapping_metadata = get_super5_pn2021_mapping_metadata()
    class_names = list(CLASS_NAMES_SUPER5)
    summaries = []

    for center in args.centers:
        source_base = source_root / center / f"k{args.k}_seed{args.seed}" / f"{center}_real_k{args.k}_seed{args.seed}"
        output_base = output_root / center / f"k{args.k}_seed{args.seed}" / f"{center}_real_k{args.k}_seed{args.seed}"
        required_outputs = [
            output_base.with_suffix(".signals.npz"),
            output_base.with_suffix(".latent.npz"),
            output_base.with_suffix(".ref_meta.json"),
            output_base.with_suffix(".class_trust.json"),
        ]
        if all(path.exists() for path in required_outputs) and not args.force:
            print(f"[skip] {center}: existing v7 K-shot artifacts at {output_base}", flush=True)
            continue
        label_lookup = build_label_lookup_from_headers(pn2021_root / "training" / center)
        summary = relabel_kshot_artifact_group(
            center=center,
            source_base=source_base,
            output_base=output_base,
            label_by_record_id=label_lookup,
            mapping_metadata=mapping_metadata,
            class_names=class_names,
        )
        summaries.append(summary)
        print(f"[wrote] {center}: {summary['label_counts']} -> {output_base}", flush=True)

    summary_path = Path(args.summary_path) if args.summary_path else output_root / "relabel_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "mapping": mapping_metadata,
                "source_root": str(source_root),
                "output_root": str(output_root),
                "centers": summaries,
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
