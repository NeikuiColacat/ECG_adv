"""Build a per-PN2021-center dataset for CenterToken training.

For a target center (e.g. cpsc_2018_extra):
  1. Scan .hea files, parse SNOMED dx codes, filter Tier-M positives.
  2. Unified-preprocess each ECG to (1000, 12) float32 @ 100 Hz, zscored.
  3. Batch VAE-encode to (N, 4, 128) latent (ECGTwin lead order + 1024 interp).
  4. Pre-compute 6 canonical Tier-M class nomic text embeddings (reuse per class).
  5. Parse age/hr/sex from .hea when available, else fallback.
  6. Assign strat_fold via deterministic hash → train/val/test split.
  7. Save list of {'data': latent (4,128), 'label': {...}} dicts to .pt
     matching the ListDataset format used by CenterTokenTrainer.

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/prep_center_dataset.py \
    --center cpsc_2018_extra \
    --out /root/autodl-tmp/center_token/cpsc_2018_extra.pt
"""
import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.crosscenter_v2.label_alignment_v2 import (  # noqa: E402
    TIER_M, TIER_M_IDX, NUM_CLASSES_TIER_M, snomed_to_26, has_any_scored_class,
)
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000  # noqa: E402
from scripts.crosscenter_tierM.eval_crosscenter_tierM import (  # noqa: E402
    parse_header_snomed, scan_center_records,
)
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402

DEFAULT_PN2021_DIR = "/root/autodl-tmp/physionet2021/training"

# Canonical per-class diagnostic text for nomic embedding. Each class gets ONE
# fixed prompt; all records with that class reuse the same text_embed tensor.
TIER_M_CANONICAL_PROMPT = {
    "NSR":   "sinus rhythm|normal ecg",
    "STach": "sinus tachycardia|abnormal ecg",
    "AF":    "atrial fibrillation|abnormal ecg",
    "IAVB":  "first degree atrioventricular block|abnormal ecg",
    "LBBB":  "complete left bundle branch block|abnormal ecg",
    "RBBB":  "complete right bundle branch block|abnormal ecg",
}


def _parse_header_meta(header_path: str) -> dict:
    """Parse #Age, #Sex, rough HR estimate if present."""
    meta = {"age": None, "sex": None, "hr": None}
    try:
        with open(header_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#Age:'):
                    val = line.split(':', 1)[1].strip()
                    try:
                        meta["age"] = float(val)
                    except ValueError:
                        pass
                elif line.startswith('#Sex:'):
                    val = line.split(':', 1)[1].strip().upper()
                    meta["sex"] = "M" if val.startswith("M") else ("F" if val.startswith("F") else "U")
    except Exception:
        pass
    return meta


def _fold_from_hash(record_id: str) -> int:
    """Deterministic fold 1..10 from record id; use 9=val, 10=test, others=train."""
    h = int(hashlib.sha1(record_id.encode()).hexdigest()[:8], 16)
    return (h % 10) + 1


def _pick_primary_class(label_6: np.ndarray) -> Optional[int]:
    """Return index of first positive Tier-M class, or None."""
    pos = np.where(label_6 == 1.0)[0]
    if len(pos) == 0:
        return None
    return int(pos[0])


def _encode_batch(wrapper: ECGTwinWrapper, signals_bct_1000: np.ndarray, batch: int) -> torch.Tensor:
    """(N, 12, 1000) PTBXL → (N, 4, 128) latent via ECGTwin lead order + interp to 1024."""
    dev = wrapper.device
    out = []
    for i in range(0, signals_bct_1000.shape[0], batch):
        chunk = torch.from_numpy(signals_bct_1000[i:i + batch]).float().to(dev)
        chunk_et = chunk[:, ECGTWIN_TO_PTBXL_INDICES, :]
        chunk_1024 = F.interpolate(chunk_et, size=1024, mode="linear", align_corners=True)
        with torch.no_grad():
            z = wrapper.encode_ecg(chunk_1024)  # (b, 4, 128)
        out.append(z.detach().cpu())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True, help="PN2021 center name, e.g. cpsc_2018_extra")
    ap.add_argument("--data_dir", default=DEFAULT_PN2021_DIR)
    ap.add_argument("--out", required=True, help=".pt output path")
    ap.add_argument("--encode_batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_records", type=int, default=0, help="0=no limit")
    args = ap.parse_args()

    center_dir = os.path.join(args.data_dir, args.center)
    if not os.path.isdir(center_dir):
        raise SystemExit(f"Center dir not found: {center_dir}")

    print(f"[prep] scanning {center_dir} ...", flush=True)
    t0 = time.time()
    record_paths, snomed_lists = scan_center_records(center_dir)
    print(f"[prep] {len(record_paths)} scored records ({time.time() - t0:.0f}s)")
    if not record_paths:
        raise SystemExit("no scored records")

    labels_26 = np.stack([snomed_to_26(s) for s in snomed_lists])
    labels_6 = labels_26[:, TIER_M_IDX].astype(np.float32)
    tier_m_pos_mask = labels_6.sum(axis=1) > 0
    keep_idx = np.where(tier_m_pos_mask)[0]
    if args.max_records > 0:
        keep_idx = keep_idx[:args.max_records]
    print(f"[prep] {len(keep_idx)} records with at least 1 Tier-M class")

    print(f"[prep] loading ECGTwin (encoder + text_model) on {args.device} ...", flush=True)
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=True)

    print("[prep] computing 6 canonical class text embeddings (nomic)...")
    class_text_embeds: dict = {}
    for cls in TIER_M:
        emb = wrapper.get_text_embedding(TIER_M_CANONICAL_PROMPT[cls]).detach().cpu()
        class_text_embeds[cls] = emb
        print(f"  {cls}: '{TIER_M_CANONICAL_PROMPT[cls]}' → shape {tuple(emb.shape)}")

    # Preprocess signals
    import wfdb
    print(f"[prep] preprocessing {len(keep_idx)} records ...", flush=True)
    signals_1000_bct = []
    kept = []
    t0 = time.time()
    for j, i in enumerate(keep_idx):
        rp = record_paths[i]
        try:
            rec = wfdb.rdrecord(rp)
        except Exception:
            continue
        sig = rec.p_signal
        if sig is None or sig.shape[1] < 12:
            continue
        sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, 'sig_name', None) else None
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
        signals_1000_bct.append(proc.T.astype(np.float32))  # (12, 1000)
        kept.append(i)
        if (j + 1) % 200 == 0:
            print(f"    [{j + 1}/{len(keep_idx)}] elapsed {time.time() - t0:.0f}s", flush=True)
    print(f"[prep] preprocessed {len(kept)} records ({time.time() - t0:.0f}s)")
    if not kept:
        raise SystemExit("no records survived preprocessing")

    signals_arr = np.stack(signals_1000_bct, axis=0)  # (N, 12, 1000)
    labels_kept = labels_6[np.array(kept)]  # (N, 6)

    print(f"[prep] VAE-encoding {signals_arr.shape[0]} records (batch={args.encode_batch}) ...", flush=True)
    t0 = time.time()
    latents = _encode_batch(wrapper, signals_arr, batch=args.encode_batch)
    print(f"[prep] latents {tuple(latents.shape)} in {time.time() - t0:.0f}s")

    assert latents.shape[1:] == (4, 128), f"bad latent shape: {latents.shape}"
    assert torch.isfinite(latents).all().item(), "non-finite latents produced!"

    # Build final sample list
    samples = []
    fold_counts = {f: 0 for f in range(1, 11)}
    class_counts = {c: 0 for c in TIER_M}
    for k, i in enumerate(kept):
        rec_id = os.path.basename(record_paths[i])
        meta = _parse_header_meta(record_paths[i] + ".hea")
        cls_i = _pick_primary_class(labels_kept[k])
        if cls_i is None:
            continue  # shouldn't happen given filter
        cls_name = TIER_M[cls_i]
        fold = _fold_from_hash(rec_id)
        fold_counts[fold] += 1
        class_counts[cls_name] += 1

        label = {
            "hr": meta["hr"] if meta["hr"] is not None else 75.0,
            "age": meta["age"] if meta["age"] is not None else 60.0,
            "sex": meta["sex"] if meta["sex"] is not None else "U",
            "text_embed": class_text_embeds[cls_name].clone(),   # (L_i, 768) on CPU
            "diagnostic_class": cls_name,
            "strat_fold": fold,
            "record_id": rec_id,
        }
        samples.append({"data": latents[k].clone(), "label": label})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, out_path)
    print(f"[prep] wrote {len(samples)} samples → {out_path}")
    print(f"[prep] strat_fold distribution: {fold_counts}")
    print(f"[prep] class distribution: {class_counts}")


if __name__ == "__main__":
    main()
