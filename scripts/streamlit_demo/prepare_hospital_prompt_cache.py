#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from apps.streamlit_ecg_demo.services.preprocessing import CLASS_NAMES, crop_or_pad_ct  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import PTBXL_TO_ECGTWIN_INDICES  # noqa: E402


VERSION = "streamlit_hospital_prompt_cache_v1"
PRIORITY = ["MI", "HYP", "CD", "STTC", "NORM"]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def primary_class(row: np.ndarray) -> str | None:
    for cls in PRIORITY:
        if float(row[CLASS_NAMES.index(cls)]) > 0.5:
            return cls
    return None


def fold_from_record_id(record_id: str) -> int:
    h = int(hashlib.sha1(record_id.encode()).hexdigest()[:8], 16)
    return (h % 10) + 1


def hybrid_select(primary_classes: list[str], k: int, floor_per_class: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_class = {cls: [] for cls in CLASS_NAMES}
    for idx, cls in enumerate(primary_classes):
        if cls in by_class:
            by_class[cls].append(idx)
    selected: list[int] = []
    for cls in CLASS_NAMES:
        pool = by_class[cls]
        take = min(int(floor_per_class), len(pool))
        if take:
            selected.extend(rng.sample(pool, take))
    selected_set = set(selected)
    remaining = [idx for idx in range(len(primary_classes)) if idx not in selected_set]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, int(k) - len(selected))])
    return selected[: int(k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_npz", required=True)
    ap.add_argument("--center", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--floor_per_class", type=int, default=10)
    ap.add_argument("--encode_batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)
    out_root = Path(args.out_root)
    full_dir = out_root / "center_full_latents"
    selection_dir = out_root / "ref_selection"
    full_dir.mkdir(parents=True, exist_ok=True)
    selection_dir.mkdir(parents=True, exist_ok=True)
    full_path = full_dir / f"{args.center}.pt"
    selection_path = selection_dir / f"{args.center}_k{args.K}_seed{args.seed}.json"

    with np.load(args.dataset_npz, allow_pickle=True) as data:
        signals = data["signals"].astype(np.float32)
        labels = data["labels"].astype(np.float32)
    if signals.shape[0] != labels.shape[0]:
        raise ValueError(f"signals N={signals.shape[0]} but labels N={labels.shape[0]}")

    kept_signals = []
    kept_labels = []
    primary_classes = []
    primary_indices = []
    record_ids = []
    for i in range(signals.shape[0]):
        cls = primary_class(labels[i])
        if cls is None:
            continue
        sig_ct = crop_or_pad_ct(signals[i], target_len=1000)
        if not np.isfinite(sig_ct).all():
            continue
        kept_signals.append(sig_ct)
        kept_labels.append(labels[i])
        primary_classes.append(cls)
        primary_indices.append(CLASS_NAMES.index(cls))
        record_ids.append(f"{args.center}_{i:06d}")
    if not kept_signals:
        raise RuntimeError("no labeled finite ECG survived prompt-cache preparation")

    wrapper = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=False)
    latents = []
    for start in range(0, len(kept_signals), args.encode_batch):
        chunk = torch.from_numpy(np.stack(kept_signals[start:start + args.encode_batch])).float().to(wrapper.device)
        chunk = chunk[:, PTBXL_TO_ECGTWIN_INDICES, :]
        chunk = F.interpolate(chunk, size=1024, mode="linear", align_corners=True)
        chunk = chunk.transpose(1, 2).contiguous()
        with torch.no_grad():
            z = wrapper.encode_ecg(chunk)
        latents.append(z.detach().cpu())
    latents_t = torch.cat(latents, dim=0).float()
    labels_t = torch.from_numpy(np.stack(kept_labels)).float()

    cache = {
        "version": VERSION,
        "center": args.center,
        "source_npz": str(args.dataset_npz),
        "preprocessing": {
            "input_assumption": "uploaded ECG already aligned to PTB-XL lead order and canonicalized to (12, 1000)",
            "target_len": 1024,
            "lead_order": "ECGTwin/MIMIC",
            "scale_warning": "uploaded hospital ECG scale is accepted as provided; mV calibration should be verified for production use",
        },
        "record_ids": record_ids,
        "wfdb_paths": [""] * len(record_ids),
        "latents": latents_t,
        "age": torch.full((len(record_ids),), 60.0, dtype=torch.float32),
        "hr": torch.full((len(record_ids),), 75.0, dtype=torch.float32),
        "sex": ["U"] * len(record_ids),
        "snomed_codes": [[] for _ in record_ids],
        "primary_snomed": [None] * len(record_ids),
        "primary_class": primary_classes,
        "primary_class_idx": torch.tensor(primary_indices, dtype=torch.long),
        "super5_multi_hot": labels_t,
        "strat_fold": torch.tensor([fold_from_record_id(rid) for rid in record_ids], dtype=torch.long),
    }
    torch.save(cache, full_path)

    selected_idx = hybrid_select(primary_classes, args.K, args.floor_per_class, args.seed)
    selected_ids = [record_ids[i] for i in selected_idx]
    counts = Counter(primary_classes[i] for i in selected_idx)
    selection = {
        "version": VERSION,
        "center": args.center,
        "K_request": int(args.K),
        "K": len(selected_idx),
        "seed": int(args.seed),
        "sampling_policy": "hybrid_floor_plus_natural",
        "floor_per_class": int(args.floor_per_class),
        "selected_indices_in_full_cache": selected_idx,
        "selected_record_ids": selected_ids,
        "ref_record_ids": selected_ids,
        "primary_class_counts": dict(counts),
        "full_cache": str(full_path),
        "note": "Built from user-uploaded Streamlit hospital ECG. Verify signal scale and lead order before clinical use.",
    }
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"[write] full cache: {full_path}")
    print(f"[write] selection: {selection_path}")
    print(f"[summary] kept={len(record_ids)} selected={len(selected_idx)} counts={dict(counts)}")


if __name__ == "__main__":
    main()

