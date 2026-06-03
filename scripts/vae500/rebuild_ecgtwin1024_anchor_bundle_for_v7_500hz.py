#!/usr/bin/env python
"""Rebuild ECGTwin-1024 latent anchor bundles with current labels and 500 Hz signals.

This is a compatibility utility for a cheap diagnostic:

* keep the original ECGTwin VAE latent vectors, usually shaped ``(N, 4, 128)``;
* relabel the same PN2021 record ids under the active Super5 mapping;
* rebuild the companion real ECG signal bundle at the classifier protocol
  requested by the caller, e.g. 500 Hz and 5000 samples.

The resulting files can be used by ``synth_online_at_super5.py`` with
``--vae_backend ecgtwin1024 --input_len 5000 --crop_len 5000``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import wfdb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.data import scan_pn2021_center_records  # noqa: E402
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000  # noqa: E402
from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5,
    get_super5_pn2021_mapping_metadata,
    snomed_list_to_super5,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_preprocess(
    record_path: Path,
    *,
    target_fs: int,
    target_len: int,
    preprocess_mode: str,
    norm_mode: str,
) -> np.ndarray | None:
    try:
        rec = wfdb.rdrecord(str(record_path))
    except Exception:
        return None
    if rec.p_signal is None or rec.p_signal.shape[1] < 12:
        return None
    source_leads = [s.strip() for s in rec.sig_name] if getattr(rec, "sig_name", None) else None
    return unified_preprocess_to_1000(
        np.asarray(rec.p_signal, dtype=np.float32),
        fs=rec.fs,
        source_leads=source_leads,
        target_fs=target_fs,
        target_len=target_len,
        preprocess_mode=preprocess_mode,
        norm_mode=norm_mode,
    )


def _primary_class(labels: np.ndarray) -> np.ndarray:
    class_names = np.asarray(CLASS_NAMES_SUPER5)
    primary = []
    for row in labels:
        if float(np.max(row)) <= 0.0:
            primary.append("ALL_ZERO")
        else:
            primary.append(str(class_names[int(np.argmax(row))]))
    return np.asarray(primary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", required=True)
    parser.add_argument("--source-latent-npz", required=True)
    parser.add_argument("--source-ref-meta-json", default="")
    parser.add_argument("--pn2021-root", default="/root/autodl-tmp/physionet2021/training")
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--sampling-rate", type=int, default=500)
    parser.add_argument("--input-len", type=int, default=5000)
    parser.add_argument("--preprocess-mode", default="minimal_resample")
    parser.add_argument("--norm-mode", default="per_sample_global")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_latent_path = Path(args.source_latent_npz)
    output_base = Path(args.output_base)
    signal_path = output_base.with_suffix(".signals.npz")
    latent_path = output_base.with_suffix(".latent.npz")
    meta_path = output_base.with_suffix(".ref_meta.json")
    trust_path = output_base.with_suffix(".class_trust.json")
    outputs = [signal_path, latent_path, meta_path, trust_path]
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError(f"output exists; use --force to overwrite: {output_base}")

    with np.load(source_latent_path, allow_pickle=True) as data:
        if "latents" not in data.files or "record_ids" not in data.files:
            raise KeyError(f"{source_latent_path} must contain latents and record_ids")
        latents = data["latents"].astype(np.float32, copy=False)
        record_ids = data["record_ids"].astype(str)
        copied_payload = {key: data[key] for key in data.files if key not in {"labels", "class_names", "primary_class"}}

    record_by_id = {
        str(record.record_id): record
        for record in scan_pn2021_center_records(Path(args.pn2021_root) / args.center)
    }
    signals: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    kept_ids: list[str] = []
    kept_latents: list[np.ndarray] = []
    failed: list[str] = []
    missing: list[str] = []

    for idx, rid in enumerate(record_ids.tolist()):
        record = record_by_id.get(str(rid))
        if record is None:
            missing.append(str(rid))
            continue
        proc = _read_preprocess(
            record.record_path,
            target_fs=int(args.sampling_rate),
            target_len=int(args.input_len),
            preprocess_mode=str(args.preprocess_mode),
            norm_mode=str(args.norm_mode),
        )
        if proc is None:
            failed.append(str(rid))
            continue
        signals.append(proc.astype(np.float32, copy=False))
        labels.append(np.asarray(snomed_list_to_super5(record.snomeds), dtype=np.float32))
        kept_ids.append(str(rid))
        kept_latents.append(latents[int(idx)])

    if missing:
        raise RuntimeError(f"{args.center}: {len(missing)} record ids missing; first={missing[:5]}")
    if not signals:
        raise RuntimeError(f"{args.center}: no usable signals rebuilt")

    signals_arr = np.stack(signals).astype(np.float32, copy=False)
    labels_arr = np.stack(labels).astype(np.float32, copy=False)
    latents_arr = np.stack(kept_latents).astype(np.float32, copy=False)
    kept_record_ids = np.asarray(kept_ids)
    mapping_meta = get_super5_pn2021_mapping_metadata()
    class_names = np.asarray(CLASS_NAMES_SUPER5)
    class_counts = labels_arr.sum(axis=0).astype(int)
    primary = _primary_class(labels_arr)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        signal_path,
        signals=signals_arr,
        labels=labels_arr,
        record_ids=kept_record_ids,
        center_name=np.asarray(args.center),
        class_names=class_names,
        mapping_version=np.asarray(str(mapping_meta["mapping_version"])),
        mapping_hash=np.asarray(str(mapping_meta["mapping_hash"])),
    )

    copied_payload.update(
        {
            "latents": latents_arr,
            "labels": labels_arr,
            "record_ids": kept_record_ids,
            "center_name": np.asarray(args.center),
            "class_names": class_names,
            "primary_class": primary,
            "source_ids": np.zeros((len(kept_record_ids),), dtype=np.int16),
            "source_names": np.asarray(["real_ecgtwin1024_anchor"]),
            "source_local_indices": np.arange(len(kept_record_ids), dtype=np.int32),
            "mapping_version": np.asarray(str(mapping_meta["mapping_version"])),
            "mapping_hash": np.asarray(str(mapping_meta["mapping_hash"])),
        }
    )
    np.savez_compressed(latent_path, **copied_payload)

    source_meta: dict[str, Any] = {}
    if args.source_ref_meta_json:
        try:
            source_meta = json.loads(Path(args.source_ref_meta_json).read_text(encoding="utf-8"))
        except FileNotFoundError:
            source_meta = {}
    label_counts = dict(zip(CLASS_NAMES_SUPER5, class_counts.tolist(), strict=True))
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "center": args.center,
        "n": int(len(kept_record_ids)),
        "source_latent_npz": str(source_latent_path),
        "source_ref_meta_json": str(args.source_ref_meta_json),
        "source_meta_parent": source_meta,
        "policy": "preserve ECGTwin-1024 latents; rebuild v7 labels and 500Hz classifier signals for same record ids",
        "mapping": mapping_meta,
        "sampling_rate": int(args.sampling_rate),
        "input_len": int(args.input_len),
        "preprocess_mode": args.preprocess_mode,
        "norm_mode": args.norm_mode,
        "latent_shape": list(latents_arr.shape[1:]),
        "signal_shape": list(signals_arr.shape[1:]),
        "label_counts": label_counts,
        "ref_record_ids": kept_record_ids.tolist(),
        "failed_record_ids": failed,
        "outputs": {
            "signals": str(signal_path),
            "latents": str(latent_path),
            "class_trust": str(trust_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    trust_path.write_text(
        json.dumps(
            {
                "center": args.center,
                "mapping_version": str(mapping_meta["mapping_version"]),
                "mapping_hash": str(mapping_meta["mapping_hash"]),
                "policy": "real_all_present under rebuilt ECGTwin-1024/v7/500Hz anchor bundle",
                "label_counts": label_counts,
                "class_trust": {
                    cls: (1.0 if int(class_counts[i]) > 0 else 0.0)
                    for i, cls in enumerate(CLASS_NAMES_SUPER5)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
