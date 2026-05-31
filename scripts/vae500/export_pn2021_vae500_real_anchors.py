#!/usr/bin/env python
"""Export PN2021 target-center real anchors encoded by the 500 Hz VAE."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wfdb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.data import scan_pn2021_center_records
from ecg_adv_gen.vae import VAE500RuntimeWrapper
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, snomed_list_to_super5


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


def _stratified_indices(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nonzero = np.where(labels.sum(axis=1) > 0)[0]
    if nonzero.size == 0:
        raise ValueError("no nonzero Super5 labels available for anchor selection")
    y = labels[nonzero]
    primary = y.argmax(axis=1)
    pools = []
    for c in range(y.shape[1]):
        idx = nonzero[np.where(primary == c)[0]]
        rng.shuffle(idx)
        pools.append(idx)
    counts = np.asarray([len(p) for p in pools], dtype=np.float64)
    raw = counts / max(counts.sum(), 1.0) * int(k)
    take = np.floor(raw).astype(int)
    for c in np.argsort(-(raw - take)):
        if take.sum() >= k:
            break
        if take[c] < len(pools[c]):
            take[c] += 1
    for c in range(len(take)):
        take[c] = min(take[c], len(pools[c]))
    selected = np.concatenate([pools[c][: take[c]] for c in range(len(pools)) if take[c] > 0])
    if selected.size < k:
        used = set(int(i) for i in selected)
        rest = np.asarray([int(i) for i in nonzero if int(i) not in used], dtype=np.int64)
        rng.shuffle(rest)
        selected = np.concatenate([selected, rest[: k - selected.size]])
    rng.shuffle(selected)
    return selected[:k].astype(np.int64)


def _read_preprocess(record_path: Path, *, target_fs: int, target_len: int, preprocess_mode: str, norm_mode: str) -> np.ndarray | None:
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


@torch.no_grad()
def _encode_latents(model: VAE500RuntimeWrapper, signals: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    chunks = []
    for i in range(0, signals.shape[0], batch_size):
        x = torch.from_numpy(np.ascontiguousarray(signals[i:i + batch_size])).float().to(device)
        chunks.append(model.encode(x, deterministic=True).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", required=True)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pn2021_root", default="/root/autodl-tmp/physionet2021/training")
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--vae_variant", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sampling_rate", type=int, default=500)
    parser.add_argument("--input_len", type=int, default=5000)
    parser.add_argument("--preprocess_mode", default="minimal_resample")
    parser.add_argument("--norm_mode", default="per_sample_global")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit_records", type=int, default=None)
    args = parser.parse_args()

    center_dir = Path(args.pn2021_root) / args.center
    records = scan_pn2021_center_records(center_dir)
    if args.limit_records:
        records = records[: args.limit_records]
    labels = np.stack([snomed_list_to_super5(r.snomeds) for r in records]).astype(np.float32)
    selected = _stratified_indices(labels, args.k, args.seed)

    selected_records = [records[int(i)] for i in selected]
    selected_labels = labels[selected]
    signals = []
    kept_labels = []
    kept_records = []
    failed = []
    for rec, label in zip(selected_records, selected_labels, strict=True):
        proc = _read_preprocess(
            rec.record_path,
            target_fs=args.sampling_rate,
            target_len=args.input_len,
            preprocess_mode=args.preprocess_mode,
            norm_mode=args.norm_mode,
        )
        if proc is None:
            failed.append(rec.record_id)
            continue
        signals.append(proc.astype(np.float32, copy=False))
        kept_labels.append(label)
        kept_records.append(rec)
    if len(signals) == 0:
        raise RuntimeError("all selected records failed preprocessing")
    if len(signals) < args.k:
        print(f"[warn] kept {len(signals)}/{args.k} after preprocessing failures", flush=True)
    signals_arr = np.stack(signals).astype(np.float32, copy=False)
    labels_arr = np.stack(kept_labels).astype(np.float32, copy=False)

    vae = VAE500RuntimeWrapper(args.vae_ckpt, variant=args.vae_variant, device=args.device)
    latents = _encode_latents(vae, signals_arr, batch_size=args.batch_size, device=args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.center}_real_k{len(signals_arr)}_seed{args.seed}_vae500"
    signal_path = output_dir / f"{stem}.signals.npz"
    latent_path = output_dir / f"{stem}.latent.npz"
    meta_path = output_dir / f"{stem}.ref_meta.json"
    trust_path = output_dir / f"{stem}.class_trust.json"

    record_ids = np.asarray([r.record_id for r in kept_records])
    primary = np.asarray(CLASS_NAMES_SUPER5)[labels_arr.argmax(axis=1)]
    np.savez_compressed(
        signal_path,
        signals=signals_arr,
        labels=labels_arr,
        record_ids=record_ids,
        center_name=np.asarray(args.center),
        class_names=np.asarray(CLASS_NAMES_SUPER5),
    )
    np.savez_compressed(
        latent_path,
        latents=latents,
        labels=labels_arr,
        record_ids=record_ids,
        center_name=np.asarray(args.center),
        class_names=np.asarray(CLASS_NAMES_SUPER5),
        primary_class=primary,
        source_ids=np.zeros((len(record_ids),), dtype=np.int16),
        source_names=np.asarray(["real_anchor"]),
        source_local_indices=np.arange(len(record_ids), dtype=np.int32),
    )
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "center": args.center,
        "K_requested": int(args.k),
        "K_kept": int(len(record_ids)),
        "selection_seed": int(args.seed),
        "mapping": "v7_super5_sjr_rgq_review_20260528",
        "sampling_rate": int(args.sampling_rate),
        "input_len": int(args.input_len),
        "preprocess_mode": args.preprocess_mode,
        "norm_mode": args.norm_mode,
        "vae_ckpt": str(Path(args.vae_ckpt).expanduser().resolve()),
        "latent_shape": list(latents.shape[1:]),
        "ref_record_ids": record_ids.tolist(),
        "failed_record_ids": failed,
        "outputs": {
            "signals": str(signal_path),
            "latents": str(latent_path),
            "class_trust": str(trust_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
    class_counts = labels_arr.sum(axis=0).astype(int)
    trust = {
        cls: (1.0 if int(class_counts[i]) > 0 else 0.0)
        for i, cls in enumerate(CLASS_NAMES_SUPER5)
    }
    trust_path.write_text(
        json.dumps(
            {
                "tag": stem,
                "class_trust": trust,
                "label_counts": dict(zip(CLASS_NAMES_SUPER5, class_counts.tolist(), strict=True)),
                "policy": "VAE500 real target anchors: trust every Super5 class present in the selected K set.",
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
