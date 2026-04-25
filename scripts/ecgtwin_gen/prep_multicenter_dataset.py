"""Build a multi-center PN2021 dataset for Style Translator training.

Scans N centers (default 3: cpsc_2018_extra + chapman_shaoxing + cpsc_2018),
preprocesses, VAE-encodes, attaches canonical Tier-M text embeddings, and
packs everything into a single .pt file with per-record `center_id` label.

Output format: list of dicts
  [{'data': latent (4,128), 'label': {center_id, center_name, diagnostic_class,
                                       text_embed (Lt,768), hr, age, sex,
                                       strat_fold, record_id}}, ...]

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \\
    scripts/ecgtwin_gen/prep_multicenter_dataset.py \\
    --centers cpsc_2018_extra chapman_shaoxing cpsc_2018 \\
    --out /root/autodl-tmp/center_aware_ibe/datasets/3center_styled.pt
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.crosscenter_v2.label_alignment_v2 import (  # noqa: E402
    TIER_M, TIER_M_IDX, snomed_to_26,
)
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000  # noqa: E402
from scripts.crosscenter_tierM.eval_crosscenter_tierM import (  # noqa: E402
    scan_center_records,
)
from scripts.ecgtwin_gen.prep_center_dataset import (  # noqa: E402
    TIER_M_CANONICAL_PROMPT, _encode_batch, _fold_from_hash,
    _parse_header_meta, _pick_primary_class,
)
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402

DEFAULT_PN2021_DIR = "/root/autodl-tmp/physionet2021/training"
DEFAULT_CENTERS = ["cpsc_2018_extra", "chapman_shaoxing", "cpsc_2018"]


def scan_and_filter(center_dir: str, max_records: int = 0):
    """Return (record_paths, labels_6) kept records with ≥1 Tier-M positive."""
    record_paths, snomed_lists = scan_center_records(center_dir)
    if not record_paths:
        return [], np.zeros((0, 6), dtype=np.float32)
    labels_26 = np.stack([snomed_to_26(s) for s in snomed_lists])
    labels_6 = labels_26[:, TIER_M_IDX].astype(np.float32)
    tier_m_pos_mask = labels_6.sum(axis=1) > 0
    keep_idx = np.where(tier_m_pos_mask)[0]
    if max_records > 0:
        keep_idx = keep_idx[:max_records]
    kept_paths = [record_paths[i] for i in keep_idx]
    kept_labels = labels_6[keep_idx]
    return kept_paths, kept_labels


def preprocess_records(record_paths: List[str]):
    """Return (signals (N, 12, 1000), kept_paths, kept_metas) after filtering failures."""
    import wfdb
    signals = []
    metas = []
    kept_paths = []
    t0 = time.time()
    for j, rp in enumerate(record_paths):
        try:
            rec = wfdb.rdrecord(rp)
        except Exception:
            continue
        sig = rec.p_signal
        if sig is None or sig.shape[1] < 12:
            continue
        sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, "sig_name", None) else None
        proc = unified_preprocess_to_1000(
            sig.astype(np.float32),
            fs=rec.fs,
            source_leads=sig_names,
            target_fs=100,
            target_len=1000,
            apply_filter=True,
            apply_zscore=True,
        )
        if proc is None:
            continue
        signals.append(proc.T.astype(np.float32))  # (12, 1000)
        meta = _parse_header_meta(rp + ".hea")
        metas.append(meta)
        kept_paths.append(rp)
        if (j + 1) % 200 == 0:
            print(f"    [{j + 1}/{len(record_paths)}] elapsed {time.time() - t0:.0f}s", flush=True)
    if not signals:
        return np.empty((0, 12, 1000), dtype=np.float32), [], []
    return np.stack(signals, axis=0), kept_paths, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS,
                    help="PN2021 center names (order → center_id)")
    ap.add_argument("--data_dir", default=DEFAULT_PN2021_DIR)
    ap.add_argument("--out", required=True, help=".pt output path")
    ap.add_argument("--encode_batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_records_per_center", type=int, default=0,
                    help="0 = no limit; used for ningbo subsample in Stage 0-multi")
    ap.add_argument("--subsample_centers", nargs="*", default=[],
                    help="Center names to apply max_records_per_center to (others unlimited)")
    ap.add_argument("--subsample_target", type=int, default=3000,
                    help="Cap for centers in --subsample_centers")
    args = ap.parse_args()

    print(f"[prep] centers (in center_id order): {args.centers}")
    center_to_id = {name: i for i, name in enumerate(args.centers)}

    # Load ECGTwin once (encoder + text model shared across centers)
    print(f"[prep] loading ECGTwin on {args.device} (encoder + text model)...", flush=True)
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=True)

    # Compute 6 canonical Tier-M class text embeddings once
    print("[prep] computing 6 canonical Tier-M class text embeddings...")
    class_text_embeds: dict = {}
    for cls in TIER_M:
        emb = wrapper.get_text_embedding(TIER_M_CANONICAL_PROMPT[cls]).detach().cpu()
        class_text_embeds[cls] = emb
        print(f"  {cls}: '{TIER_M_CANONICAL_PROMPT[cls]}' -> {tuple(emb.shape)}")

    all_samples = []
    counts_per_center: dict = {}

    for center_name in args.centers:
        cid = center_to_id[center_name]
        center_dir = os.path.join(args.data_dir, center_name)
        if not os.path.isdir(center_dir):
            print(f"[prep] ⚠ center dir not found, skipping: {center_dir}")
            continue

        # Per-center cap
        cap = args.subsample_target if center_name in args.subsample_centers else args.max_records_per_center
        print(f"\n[prep] === center {cid}: {center_name} (cap={cap or 'none'}) ===")

        t0 = time.time()
        record_paths, labels_6 = scan_and_filter(center_dir, max_records=cap)
        print(f"[prep] {len(record_paths)} Tier-M positive records ({time.time() - t0:.0f}s)")
        if len(record_paths) == 0:
            continue

        t0 = time.time()
        signals, kept_paths, metas = preprocess_records(record_paths)
        # Re-match labels to kept_paths via path equality
        path_to_lbl = {record_paths[i]: labels_6[i] for i in range(len(record_paths))}
        labels_kept = np.stack([path_to_lbl[p] for p in kept_paths]) if kept_paths else np.empty((0, 6), dtype=np.float32)
        print(f"[prep] preprocessed {signals.shape[0]} records ({time.time() - t0:.0f}s)")
        if signals.shape[0] == 0:
            continue

        t0 = time.time()
        latents = _encode_batch(wrapper, signals, batch=args.encode_batch)
        print(f"[prep] VAE latents {tuple(latents.shape)} ({time.time() - t0:.0f}s)")
        assert latents.shape[1:] == (4, 128)
        assert torch.isfinite(latents).all().item()

        center_samples = 0
        class_counts_per_center = {c: 0 for c in TIER_M}
        for k, rp in enumerate(kept_paths):
            rec_id = os.path.basename(rp)
            meta = metas[k]
            cls_i = _pick_primary_class(labels_kept[k])
            if cls_i is None:
                continue
            cls_name = TIER_M[cls_i]
            class_counts_per_center[cls_name] += 1

            label = {
                "center_id": cid,
                "center_name": center_name,
                "diagnostic_class": cls_name,
                "class_idx": int(cls_i),
                "disease_6hot": labels_kept[k].astype(np.float32),
                "hr": float(meta["hr"]) if meta["hr"] is not None else 75.0,
                "age": float(meta["age"]) if meta["age"] is not None else 60.0,
                "sex": meta["sex"] if meta["sex"] is not None else "U",
                "text_embed": class_text_embeds[cls_name].clone(),
                "strat_fold": _fold_from_hash(rec_id),
                "record_id": rec_id,
            }
            all_samples.append({"data": latents[k].clone(), "label": label})
            center_samples += 1

        counts_per_center[center_name] = {
            "n": center_samples,
            "class_counts": class_counts_per_center,
        }
        print(f"[prep] center '{center_name}' -> {center_samples} samples; class counts: {class_counts_per_center}")

    if not all_samples:
        raise SystemExit("no samples produced across all centers")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_samples, out)
    size_mb = out.stat().st_size / 1e6
    print(f"\n[prep] ====== DONE ======")
    print(f"[prep] wrote {len(all_samples)} samples -> {out}  ({size_mb:.1f} MB)")
    print(f"[prep] center → sample count:")
    for c, info in counts_per_center.items():
        print(f"  {c:25s} n={info['n']:5d}   classes={info['class_counts']}")


if __name__ == "__main__":
    main()
