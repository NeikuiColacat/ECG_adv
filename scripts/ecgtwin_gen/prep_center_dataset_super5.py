"""Build a per-PN2021-center Super5 ref pool for CenterToken training.

Fork of prep_center_dataset.py for the Super5 (5-class) scheme:
  CD / HYP / MI / NORM / STTC

Differences from the Tier-M version (each is a Plan Rev 7/8 issue):
  • Issue #19  primary class via severity-priority (MI > HYP > CD > STTC > NORM)
                rather than first-positive-in-list (which would hijack to NORM)
  • Issue #20  per-record SNOMED-keyed text_embed (CD/STTC sub-codes get distinct
                prompts) rather than one shared canonical prompt per class
  • Issue #18  K<50 → all records go to train fold (val_loss is meaningless on
                near-empty val set); K≥50 keeps hash-based fold split
  • Issue #33  hybrid sampling: per-class floor of N_FLOOR (default 10) with
                natural-distribution fill — guards against centers like
                cpsc_2018_extra (~80% MI) collapsing class diversity at K=500
  • Issue #39  meta dict records `ref_record_ids` so eval can exclude these
                records from the same-center test split
  • Issue #42  full-pipeline determinism: --seed 42 + cudnn deterministic
  • Issue #43  multiprocessing(12) for SNOMED + .hea metadata parsing (~30×
                speedup on cpsc_2018_extra n=3453 + ningbo n=34905)
  • Issue #13  preserves BOTH `diagnostic_class` (primary class string for the
                trainer's L_inv group key) AND `super5_multi_hot` (5-vec for
                downstream multi-pos-class ref retrieval)

Output schema (matches the ListDataset format used by CenterTokenTrainer):
  - {out}.pt        torch.save(samples, ...) where samples is a list of
                    {"data": tensor(4,128), "label": {...}} dicts.
  - {out}.meta.json companion JSON with run-level metadata, e.g.
                    {center, K, K_request, scheme, seed, ref_record_ids,
                     primary_class_counts, fold_strategy, floor_per_class}.

  Each sample's label dict:
      hr, age, sex,                  # from .hea
      text_embed: (num_reports, 768) # per-record SNOMED-keyed embed
      diagnostic_class: str          # one of {NORM, MI, HYP, CD, STTC}
      strat_fold: int 1..10          # for trainer val split (K<50 → all=1)
      record_id: str                 # PN2021 record basename, e.g. Q0001
      super5_multi_hot: tensor(5,)   # multi-pos-class for retrieval

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/prep_center_dataset_super5.py \
    --center ningbo --K 500 \
    --text_embeds /root/autodl-tmp/center_token_super5/super5_text_embeds.pt \
    --out /root/autodl-tmp/center_token_super5/ningbo_k500.pt
"""
import argparse
import functools
import hashlib
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000  # noqa: E402
from scripts.triple_labels.eval_crosscenter import parse_header_snomed  # noqa: E402
from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5, NUM_SUPER5, SUPER5_TO_IDX,
    SNOMED_TO_SUPER5, snomed_list_to_super5,
)
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402

DEFAULT_PN2021_DIR = "/root/autodl-tmp/physionet2021/training"
DEFAULT_TEXT_EMBEDS = "/root/autodl-tmp/center_token_super5/super5_text_embeds.pt"

# Severity-priority ordering for primary-class selection (Issue #19).
# MI ranks first because the毕设 medical narrative emphasises infarction;
# NORM is last so any abnormal positive overrides "sinus rhythm" co-occurrence.
#
# Plan Rev 2026-05-01: keep all 5 super5 classes for center prompt-token
# training/validation. HYP/CD are no longer filtered at prep time; downstream
# generation/augmentation still requires class-specific digital and teacher gates.
SUPER5_PRIORITY = ["MI", "HYP", "CD", "STTC", "NORM"]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_header_meta(header_path: str) -> dict:
    """Parse # Age, # Sex from a PN2021 .hea file.

    Note: PN2021 .hea uses '# Age:' (with a space after #), so this version
    strips the leading '#' before checking the field name — matches eval_crosscenter
    parse_header_snomed() conventions.

    NaN guard: ningbo .hea sometimes contains 'Age: nan' (literal string) which
    float() parses without raising; we treat that as missing.
    """
    import math
    meta = {"age": None, "sex": None, "hr": None}
    try:
        with open(header_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line.startswith('#'):
                    continue
                body = line[1:].strip()
                if body.startswith('Age:'):
                    val = body.split(':', 1)[1].strip()
                    try:
                        a = float(val)
                        if math.isfinite(a):
                            meta["age"] = a
                    except ValueError:
                        pass
                elif body.startswith('Sex:'):
                    val = body.split(':', 1)[1].strip().upper()
                    meta["sex"] = "M" if val.startswith("M") else (
                        "F" if val.startswith("F") else "U")
    except Exception:
        pass
    return meta


def _fold_from_hash(record_id: str) -> int:
    h = int(hashlib.sha1(record_id.encode()).hexdigest()[:8], 16)
    return (h % 10) + 1


def _pick_primary_super5(multi_hot: np.ndarray) -> Optional[str]:
    """Severity-priority primary class string, or None if all-zero."""
    for cls in SUPER5_PRIORITY:
        if multi_hot[SUPER5_TO_IDX[cls]] == 1.0:
            return cls
    return None


def _pick_primary_snomed(snomed_codes: List[int], primary: str) -> Optional[int]:
    """Among the record's SNOMEDs, pick the first that maps to `primary`.

    Used by Issue #20 dynamic per-record text_embed selection.
    """
    for code in snomed_codes:
        if SNOMED_TO_SUPER5.get(code) == primary:
            return code
    return None


def parse_record_worker(hea_path: str) -> Optional[dict]:
    """Multiprocessing worker (Issue #43): one .hea → record dict or None.

    Pure-Python only — no torch / no GPU access. Returns None on parse failure
    or if the record has no super5-positive class.
    """
    snomed_codes = parse_header_snomed(hea_path)
    if not snomed_codes:
        return None
    multi_hot = snomed_list_to_super5(snomed_codes)
    primary = _pick_primary_super5(multi_hot)
    if primary is None:
        return None
    primary_code = _pick_primary_snomed(snomed_codes, primary)
    meta = _parse_header_meta(hea_path)
    record_path = hea_path[:-4]  # strip .hea → wfdb basename
    record_id = os.path.basename(record_path)
    return {
        "hea_path":      hea_path,
        "record_path":   record_path,
        "record_id":     record_id,
        "snomed_codes":  snomed_codes,
        "multi_hot":     multi_hot,         # np.ndarray (5,) float32
        "primary":       primary,           # str
        "primary_code":  primary_code,      # int or None
        "age":           meta["age"],
        "sex":           meta["sex"],
        "hr":            meta["hr"],
    }


def _scan_hea(center_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(center_dir):
        for f in files:
            if f.endswith('.hea'):
                out.append(os.path.join(root, f))
    return out


def hybrid_sample(records: List[dict], K: int, floor: int, seed: int) -> List[dict]:
    """Per-class floor=floor + natural-distribution fill (Issue #33).

    Step 1: For each super5 class, take min(floor, available) records uniformly
    at random.
    Step 2: Fill remaining K-budget with the leftover natural-distribution pool.
    """
    rng = random.Random(seed)
    by_class = {c: [] for c in CLASS_NAMES_SUPER5}
    for r in records:
        by_class[r["primary"]].append(r)

    selected = []
    for cls in CLASS_NAMES_SUPER5:
        pool = by_class[cls]
        n_take = min(floor, len(pool))
        if n_take > 0:
            selected.extend(rng.sample(pool, n_take))

    selected_ids = {r["record_id"] for r in selected}
    remaining = [r for r in records if r["record_id"] not in selected_ids]
    n_more = K - len(selected)
    if n_more > 0 and remaining:
        rng.shuffle(remaining)
        selected.extend(remaining[:n_more])
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", required=True,
                    help="PN2021 center, e.g. cpsc_2018_extra | ningbo")
    ap.add_argument("--data_dir", default=DEFAULT_PN2021_DIR)
    ap.add_argument("--out", required=True, help=".pt output path")
    ap.add_argument("--text_embeds", default=DEFAULT_TEXT_EMBEDS,
                    help="Bundle from super5_text_embeds.py")
    ap.add_argument("--K", type=int, default=500,
                    help="ref pool size (default 500)")
    ap.add_argument("--floor_per_class", type=int, default=10,
                    help="hybrid sampling per-class floor (Issue #33)")
    ap.add_argument("--encode_batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=12,
                    help="multiprocessing workers for SNOMED parsing (Issue #43)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)

    center_dir = os.path.join(args.data_dir, args.center)
    if not os.path.isdir(center_dir):
        raise SystemExit(f"Center dir not found: {center_dir}")

    # ─── Step 1: scan + parse SNOMED via multiprocessing pool (Issue #43) ───
    print(f"[prep5] scanning {center_dir} ...", flush=True)
    t0 = time.time()
    hea_paths = _scan_hea(center_dir)
    print(f"[prep5] {len(hea_paths)} .hea files ({time.time() - t0:.0f}s)")
    if not hea_paths:
        raise SystemExit("no .hea files found")

    print(f"[prep5] parsing SNOMED with {args.num_workers} workers ...", flush=True)
    t0 = time.time()
    with Pool(processes=args.num_workers) as pool:
        parsed = pool.map(parse_record_worker, hea_paths)
    records = [r for r in parsed if r is not None]
    print(f"[prep5] kept {len(records)} super5-positive records "
          f"({time.time() - t0:.0f}s)")
    if not records:
        raise SystemExit("no records have a super5 primary class")

    # Sanity-print the natural class distribution
    from collections import Counter
    natural_counts = Counter(r["primary"] for r in records)
    print(f"[prep5] natural primary-class distribution: {dict(natural_counts)}")

    # ─── Step 2: hybrid sample K records (Issue #33) ────────────────────────
    selected = hybrid_sample(records, K=args.K, floor=args.floor_per_class,
                             seed=args.seed)
    selected_counts = Counter(r["primary"] for r in selected)
    print(f"[prep5] sampled {len(selected)} records (K_request={args.K}, "
          f"floor={args.floor_per_class})")
    print(f"[prep5] sampled primary-class distribution: {dict(selected_counts)}")

    # ─── Step 3: load text_embeds bundle ────────────────────────────────────
    print(f"[prep5] loading text_embeds: {args.text_embeds}")
    text_bundle = torch.load(args.text_embeds, map_location="cpu", weights_only=False)
    by_snomed = text_bundle["by_snomed"]
    by_class = text_bundle["by_class"]

    # ─── Step 4: load ECGTwin VAE encoder ───────────────────────────────────
    print(f"[prep5] loading ECGTwin (encoder only) on {args.device} ...", flush=True)
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=True,
                             load_text_model=False)

    # ─── Step 5: preprocess records (single process, wfdb IO + DSP) ─────────
    print(f"[prep5] preprocessing {len(selected)} records ...", flush=True)
    import wfdb
    signals_1000 = []
    kept = []
    t0 = time.time()
    for j, rec in enumerate(selected):
        try:
            r = wfdb.rdrecord(rec["record_path"])
        except Exception as e:
            print(f"[prep5] wfdb fail {rec['record_id']}: {e}")
            continue
        sig = r.p_signal
        if sig is None or sig.shape[1] < 12:
            continue
        sig_names = [s.strip() for s in r.sig_name] if getattr(r, 'sig_name', None) else None
        proc = unified_preprocess_to_1000(
            sig.astype(np.float32),
            fs=r.fs,
            source_leads=sig_names,
            target_fs=100,
            target_len=1000,
            apply_filter=True,
            apply_zscore=True,
        )
        if proc is None:
            continue
        signals_1000.append(proc.T.astype(np.float32))   # (12, 1000)
        kept.append(rec)
        if (j + 1) % 50 == 0:
            print(f"  [{j + 1}/{len(selected)}] elapsed {time.time() - t0:.0f}s",
                  flush=True)
    print(f"[prep5] preprocessed {len(kept)} records ({time.time() - t0:.0f}s)")
    if not kept:
        raise SystemExit("no records survived preprocessing")

    signals_arr = np.stack(signals_1000, axis=0)         # (N, 12, 1000)

    # ─── Step 6: VAE encode (batched on GPU) ────────────────────────────────
    print(f"[prep5] VAE-encoding {signals_arr.shape[0]} records "
          f"(batch={args.encode_batch}) ...", flush=True)
    t0 = time.time()
    latents = []
    dev = wrapper.device
    for i in range(0, signals_arr.shape[0], args.encode_batch):
        chunk = torch.from_numpy(signals_arr[i:i + args.encode_batch]).float().to(dev)
        chunk_et = chunk[:, ECGTWIN_TO_PTBXL_INDICES, :]
        chunk_1024 = F.interpolate(chunk_et, size=1024, mode="linear", align_corners=True)
        with torch.no_grad():
            z = wrapper.encode_ecg(chunk_1024)            # (b, 4, 128)
        latents.append(z.detach().cpu())
    latents = torch.cat(latents, dim=0)
    print(f"[prep5] latents {tuple(latents.shape)} in {time.time() - t0:.0f}s")
    assert latents.shape[1:] == (4, 128), f"bad latent shape: {latents.shape}"
    assert torch.isfinite(latents).all().item(), "non-finite latents produced!"

    # ─── Step 7: build sample list with per-record text_embed (Issue #20) ───
    use_train_only_fold = (len(kept) < 50)   # Issue #18

    samples = []
    fold_counts = {f: 0 for f in range(1, 11)}
    text_embed_shape_counter = Counter()
    for k, rec in enumerate(kept):
        # Per-record dynamic text_embed: by_snomed[primary_code] if available,
        # else fall back to by_class[primary_class] (Issue #20).
        if rec["primary_code"] is not None and rec["primary_code"] in by_snomed:
            text_embed = by_snomed[rec["primary_code"]].clone()
        else:
            text_embed = by_class[rec["primary"]].clone()
        text_embed_shape_counter[tuple(text_embed.shape)] += 1

        fold = 1 if use_train_only_fold else _fold_from_hash(rec["record_id"])
        fold_counts[fold] += 1

        import math
        def _finite_or(default: float, x):
            if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                return default
            return float(x)
        label = {
            "hr":               _finite_or(75.0, rec["hr"]),
            "age":              _finite_or(60.0, rec["age"]),
            "sex":              rec["sex"] if rec["sex"] is not None else "U",
            "text_embed":       text_embed,                          # (L, 768)
            "diagnostic_class": rec["primary"],                      # str (Issue #13)
            "strat_fold":       fold,
            "record_id":        rec["record_id"],
            "super5_multi_hot": torch.from_numpy(rec["multi_hot"]),  # (5,)
        }
        samples.append({"data": latents[k].clone(), "label": label})

    final_class_counts = Counter(s["label"]["diagnostic_class"] for s in samples)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, out_path)
    meta_path = out_path.with_suffix(".meta.json")
    import json
    meta = {
        "center":               args.center,
        "K":                    len(samples),
        "K_request":            args.K,
        "scheme":               "super5",
        "seed":                 args.seed,
        "ref_record_ids":       [s["label"]["record_id"] for s in samples],
        "primary_class_counts": dict(final_class_counts),
        "fold_strategy":        "train_only" if use_train_only_fold else "hash",
        "floor_per_class":      args.floor_per_class,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prep5] wrote {len(samples)} samples → {out_path}")
    print(f"[prep5] wrote meta → {meta_path}")
    print(f"[prep5] strat_fold counts: {fold_counts}")
    print(f"[prep5] final primary-class counts: {dict(final_class_counts)}")
    print(f"[prep5] text_embed shape distribution: "
          f"{dict(text_embed_shape_counter)}")


if __name__ == "__main__":
    main()
