"""Build ECGTwin-compatible full latent caches and K=500 selections.

This is the data layer for the textual-inversion-style center prompt-token
pipeline. It intentionally does not reuse old center_token_ablation K500 files:
those were encoded after classifier-style filter + global z-score.

For ECGTwin VAE semantics, this script keeps raw mV scale, reorders leads,
resamples/pads to 1024, converts PTBXL order to ECGTwin order, then encodes.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wfdb

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.crosscenter_v2.preprocess_utils import (  # noqa: E402
    pad_or_truncate_tc,
    reorder_leads_tc,
    resample_tc,
)
from scripts.triple_labels.eval_crosscenter import parse_header_snomed  # noqa: E402
from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5,
    SUPER5_TO_IDX,
    get_super5_pn2021_mapping_metadata,
    snomed_list_to_super5,
    SNOMED_TO_SUPER5_POSITIVE,
)
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import PTBXL_TO_ECGTWIN_INDICES  # noqa: E402


DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DEFAULT_OUT_ROOT = "/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1"
DEFAULT_DATA_ROOT = "/root/autodl-tmp/physionet2021/training"
SUPER5_PRIORITY = ["MI", "HYP", "CD", "STTC", "NORM"]
VERSION = "ecgtwin_prompt_token_cache_v1"


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_header_meta(header_path):
    meta = {"age": None, "sex": None, "hr": None}
    try:
        with open(header_path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("#"):
                    continue
                body = line[1:].strip()
                if body.startswith("Age:"):
                    try:
                        val = float(body.split(":", 1)[1].strip())
                        if np.isfinite(val):
                            meta["age"] = val
                    except Exception:
                        pass
                elif body.startswith("Sex:"):
                    val = body.split(":", 1)[1].strip().upper()
                    if val.startswith("M"):
                        meta["sex"] = "M"
                    elif val.startswith("F"):
                        meta["sex"] = "F"
                    else:
                        meta["sex"] = "U"
    except Exception:
        pass
    return meta


def _fold_from_hash(record_id):
    h = int(hashlib.sha1(record_id.encode()).hexdigest()[:8], 16)
    return (h % 10) + 1


def _primary_class(label):
    for cls in SUPER5_PRIORITY:
        if label[SUPER5_TO_IDX[cls]] == 1.0:
            return cls
    return None


def _primary_snomed(codes, primary):
    for code in codes:
        if SNOMED_TO_SUPER5_POSITIVE.get(code) == primary:
            return code
    return None


def _scan_records(center_dir):
    records = []
    for root, _, files in os.walk(center_dir):
        for name in files:
            if not name.endswith(".hea"):
                continue
            hea = os.path.join(root, name)
            codes = parse_header_snomed(hea)
            if not codes:
                continue
            label = snomed_list_to_super5(codes)
            primary = _primary_class(label)
            if primary is None:
                continue
            meta = _parse_header_meta(hea)
            path = hea[:-4]
            records.append({
                "record_id": os.path.basename(path),
                "path": path,
                "snomed_codes": codes,
                "label": label.astype(np.float32),
                "primary_class": primary,
                "primary_class_idx": SUPER5_TO_IDX[primary],
                "primary_snomed": _primary_snomed(codes, primary),
                "age": meta["age"],
                "sex": meta["sex"] or "U",
                "hr": meta["hr"],
                "strat_fold": _fold_from_hash(os.path.basename(path)),
            })
    return records


def _raw_to_ecgtwin_1024(record_path):
    rec = wfdb.rdrecord(record_path)
    sig = rec.p_signal
    if sig is None or sig.shape[1] < 12:
        return None
    source_leads = [s.strip() for s in rec.sig_name] if getattr(rec, "sig_name", None) else None
    sig_tc = np.asarray(sig, dtype=np.float32)
    if np.isnan(sig_tc).any():
        sig_tc = np.nan_to_num(sig_tc, nan=0.0)
    if source_leads is not None:
        sig_tc = reorder_leads_tc(sig_tc, source_leads)
        if sig_tc is None:
            return None
    if sig_tc.shape[1] != 12:
        return None
    sig_tc = resample_tc(sig_tc, rec.fs, 102.4)
    sig_tc = pad_or_truncate_tc(sig_tc, 1024)
    if not np.isfinite(sig_tc).all():
        return None
    sig_ct = torch.from_numpy(np.ascontiguousarray(sig_tc.T)).float()
    sig_ct = sig_ct[PTBXL_TO_ECGTWIN_INDICES, :]
    return sig_ct.T.contiguous()  # (1024, 12)


def hybrid_select(records, k, floor_per_class, seed):
    rng = random.Random(seed)
    by_class = {cls: [] for cls in CLASS_NAMES_SUPER5}
    for idx, rec in enumerate(records):
        by_class[rec["primary_class"]].append(idx)

    selected = []
    for cls in CLASS_NAMES_SUPER5:
        pool = by_class[cls]
        n_take = min(floor_per_class, len(pool))
        if n_take:
            selected.extend(rng.sample(pool, n_take))

    selected_set = set(selected)
    remaining = [i for i in range(len(records)) if i not in selected_set]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, k - len(selected))])
    return selected[:k]


def build_center(args, center):
    center_dir = os.path.join(args.data_root, center)
    if not os.path.isdir(center_dir):
        raise FileNotFoundError(center_dir)

    out_root = Path(args.out_root)
    full_dir = out_root / "center_full_latents"
    selection_dir = out_root / "ref_selection"
    full_dir.mkdir(parents=True, exist_ok=True)
    selection_dir.mkdir(parents=True, exist_ok=True)
    full_path = full_dir / f"{center}.pt"
    selection_path = selection_dir / f"{center}_k{args.K}_seed{args.seed}.json"

    if full_path.exists() and not args.overwrite:
        print(f"[cache] full latent cache exists: {full_path}")
        cache = torch.load(full_path, map_location="cpu", weights_only=False)
    else:
        print(f"[scan] {center}")
        records = _scan_records(center_dir)
        print(f"[scan] {center}: {len(records)} super5-positive records")
        if not records:
            raise RuntimeError(f"{center}: no super5-positive records")

        wrapper = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=False)
        latents = []
        kept = []
        batch = []
        batch_records = []
        t0 = time.time()
        for i, rec in enumerate(records):
            try:
                sig = _raw_to_ecgtwin_1024(rec["path"])
            except Exception as e:
                print(f"[warn] {center} {rec['record_id']} failed: {e}")
                sig = None
            if sig is None:
                continue
            batch.append(sig)
            batch_records.append(rec)
            if len(batch) == args.encode_batch:
                x = torch.stack(batch, dim=0).to(wrapper.device)
                with torch.no_grad():
                    z = wrapper.encode_ecg(x)
                latents.append(z.detach().cpu())
                kept.extend(batch_records)
                batch.clear()
                batch_records.clear()
            if (i + 1) % args.log_every == 0:
                print(f"  {center}: [{i+1}/{len(records)}] kept={len(kept)} elapsed={time.time()-t0:.0f}s", flush=True)
        if batch:
            x = torch.stack(batch, dim=0).to(wrapper.device)
            with torch.no_grad():
                z = wrapper.encode_ecg(x)
            latents.append(z.detach().cpu())
            kept.extend(batch_records)

        if not kept:
            raise RuntimeError(f"{center}: no records survived VAE preprocessing")
        latents = torch.cat(latents, dim=0).float()
        labels = torch.from_numpy(np.stack([r["label"] for r in kept])).float()
        primary_class_idx = torch.tensor([r["primary_class_idx"] for r in kept], dtype=torch.long)
        age = torch.tensor([
            60.0 if r["age"] is None or not np.isfinite(float(r["age"])) else float(r["age"])
            for r in kept
        ], dtype=torch.float32)
        hr = torch.tensor([
            75.0 if r["hr"] is None or not isinstance(r["hr"], (int, float)) else float(r["hr"])
            for r in kept
        ], dtype=torch.float32)
        cache = {
            "version": VERSION,
            "center": center,
            "source_root": args.data_root,
            "preprocessing": {
                "scale": "raw_mV",
                "apply_filter": False,
                "apply_zscore": False,
                "source_leads": "wfdb.sig_name reordered to PTBXL then ECGTwin",
                "target_len": 1024,
                "target_fs": 102.4,
                "lead_order": "ECGTwin/MIMIC",
            },
            "pn2021_mapping": get_super5_pn2021_mapping_metadata(),
            "record_ids": [r["record_id"] for r in kept],
            "wfdb_paths": [r["path"] for r in kept],
            "latents": latents,
            "age": age,
            "hr": hr,
            "sex": [r["sex"] for r in kept],
            "snomed_codes": [r["snomed_codes"] for r in kept],
            "primary_snomed": [r["primary_snomed"] for r in kept],
            "primary_class": [r["primary_class"] for r in kept],
            "primary_class_idx": primary_class_idx,
            "super5_multi_hot": labels,
            "strat_fold": torch.tensor([r["strat_fold"] for r in kept], dtype=torch.long),
        }
        torch.save(cache, full_path)
        print(f"[write] {center}: {len(kept)} latents -> {full_path}")

    records_for_selection = [
        {"record_id": rid, "primary_class": cls}
        for rid, cls in zip(cache["record_ids"], cache["primary_class"])
    ]
    selected_idx = hybrid_select(records_for_selection, args.K, args.floor_per_class, args.seed)
    selected_ids = [cache["record_ids"][i] for i in selected_idx]
    counts = Counter(cache["primary_class"][i] for i in selected_idx)
    selection = {
        "version": VERSION,
        "center": center,
        "K_request": args.K,
        "K": len(selected_idx),
        "seed": args.seed,
        "sampling_policy": "hybrid_floor_plus_natural",
        "floor_per_class": args.floor_per_class,
        "selected_indices_in_full_cache": selected_idx,
        "selected_record_ids": selected_ids,
        "ref_record_ids": selected_ids,
        "primary_class_counts": dict(counts),
        "full_cache": str(full_path),
        "pn2021_mapping": get_super5_pn2021_mapping_metadata(),
        "ref_exclusion_split_note": "Pass this JSON to eval/fine-tune exclusion logic; these refs are not validation/eval samples.",
    }
    with open(selection_path, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"[write] {center}: selection {dict(counts)} -> {selection_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--floor_per_class", type=int, default=30)
    ap.add_argument("--encode_batch", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    set_all_seeds(args.seed)

    for center in args.centers:
        build_center(args, center)


if __name__ == "__main__":
    main()
