"""Per-center few-shot ECG reference sampler for Tier-M adversarial generation.

Samples K reference ECGs from each PN2021 main center with Tier-M 6-class-balanced
allocation (~K/6 per class; when a class has fewer records in a center, we take
all available and backfill the deficit from that center's overall pool).

Output is a plain dict keyed by center name. Signals are preprocessed via the
same `unified_preprocess_to_1000` pipeline as the Tier-M training data, so
they are (1000, 12) float32 @100Hz, per-sample z-scored.

Results are cached to NPZ so repeated runs are cheap. Disk-size at K=32 across
4 centers: ~4 × 32 × 1000 × 12 × 4 B = 6 MB. Fits anywhere.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ecg_adv_gen.labels.tier26 import (  # noqa: E402
    TIER_M, TIER_M_IDX, NUM_CLASSES_TIER_M, snomed_to_26, has_any_scored_class,
)
from ecg_adv_gen.preprocessing import unified_preprocess_to_1000  # noqa: E402

# Reuse header parser + scan from eval script
from ecg_adv_gen.evaluation.tierm_crosscenter import (  # noqa: E402
    parse_header_snomed, scan_center_records,
)


DEFAULT_PN2021_DIR = "/root/autodl-tmp/physionet2021/training"
DEFAULT_REF_CACHE_DIR = "/root/autodl-tmp/crosscenter_tierM_online"
MAIN_CENTERS_4 = ['chapman_shaoxing', 'cpsc_2018', 'georgia', 'ningbo']


def _preprocess_one(record_path: str) -> Optional[np.ndarray]:
    """record_path -> (1000, 12) float32 or None on failure."""
    try:
        import wfdb
        rec = wfdb.rdrecord(record_path)
    except Exception:
        return None
    sig = rec.p_signal
    if sig is None or sig.shape[1] < 12:
        return None
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
    return proc


def sample_refs_per_center(
    k_per_center: int = 32,
    centers: List[str] = None,
    data_dir: str = DEFAULT_PN2021_DIR,
    cache_path: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Return ``{center: {"signals_tc": (K, 1000, 12), "labels_6": (K, 6),
    "record_ids": List[str]}}``.

    Class-balanced sampling: distribute K records across 6 Tier-M classes as
    evenly as possible (excess remainder goes to the first few classes). For a
    class with fewer positives than its quota, take all and backfill from
    "any-class" residual pool. This guarantees each center's ref pool covers
    every non-empty class.
    """
    if centers is None:
        centers = list(MAIN_CENTERS_4)
    rng = np.random.default_rng(seed)

    # Load cache if present
    if cache_path is not None and os.path.exists(cache_path):
        if verbose:
            print(f"[refs] cache hit: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        out = {}
        for c in centers:
            if f"{c}__signals" not in data.files:
                if verbose:
                    print(f"[refs] cache missing center {c}, will refresh")
                out = None
                break
            out[c] = {
                "signals_tc": data[f"{c}__signals"],
                "labels_6":   data[f"{c}__labels6"],
                "record_ids": list(data[f"{c}__record_ids"]),
            }
        if out is not None:
            return out

    out: Dict[str, Dict] = {}
    per_class_quota_base = k_per_center // NUM_CLASSES_TIER_M     # e.g. 32 // 6 = 5
    extra = k_per_center - per_class_quota_base * NUM_CLASSES_TIER_M  # 32 - 5*6 = 2 extra → first 2 classes
    quotas = [per_class_quota_base + (1 if i < extra else 0) for i in range(NUM_CLASSES_TIER_M)]
    if verbose:
        print(f"[refs] per-class quota (out of K={k_per_center}): "
              f"{ {TIER_M[i]: quotas[i] for i in range(NUM_CLASSES_TIER_M)} }")

    for center in centers:
        t0 = time.time()
        center_dir = os.path.join(data_dir, center)
        if not os.path.isdir(center_dir):
            if verbose:
                print(f"[refs] WARN: center dir not found: {center_dir}  (skip)")
            continue

        if verbose:
            print(f"[refs] {center}: scanning...", end=' ', flush=True)
        record_paths, snomed_lists = scan_center_records(center_dir)
        if not record_paths:
            if verbose:
                print("0 scored records")
            continue
        labels_26 = np.stack([snomed_to_26(s) for s in snomed_lists])
        labels_6 = labels_26[:, TIER_M_IDX].astype(np.float32)
        if verbose:
            print(f"{len(record_paths)} scored records ({time.time()-t0:.0f}s). "
                  f"Class positives: " +
                  ", ".join(f"{TIER_M[i]}={int((labels_6[:, i]==1.0).sum())}"
                            for i in range(NUM_CLASSES_TIER_M)))

        selected_idx: List[int] = []
        taken = set()
        for cls_i, quota in enumerate(quotas):
            pos_mask = labels_6[:, cls_i] == 1.0
            pos_idx = np.where(pos_mask)[0]
            pos_idx = np.array([i for i in pos_idx if int(i) not in taken])
            if len(pos_idx) == 0:
                if verbose:
                    print(f"    [{center}] class {TIER_M[cls_i]}: 0 positives — will backfill")
                continue
            n_take = min(quota, len(pos_idx))
            chosen = rng.choice(pos_idx, size=n_take, replace=False)
            selected_idx.extend(int(i) for i in chosen)
            taken.update(int(i) for i in chosen)

        # Backfill any remaining deficit from any-record pool (random, not class-targeted)
        deficit = k_per_center - len(selected_idx)
        if deficit > 0:
            all_idx = np.arange(len(record_paths))
            remaining = np.array([int(i) for i in all_idx if int(i) not in taken])
            if len(remaining) == 0:
                if verbose:
                    print(f"    [{center}] WARN: only {len(selected_idx)} records (< K={k_per_center})")
            else:
                n_take = min(deficit, len(remaining))
                chosen = rng.choice(remaining, size=n_take, replace=False)
                selected_idx.extend(int(i) for i in chosen)
                if verbose:
                    print(f"    [{center}] backfilled {n_take} additional records")

        # Preprocess the selected records
        signals_tc: List[np.ndarray] = []
        kept_labels: List[np.ndarray] = []
        kept_ids: List[str] = []
        for i in selected_idx:
            proc = _preprocess_one(record_paths[i])
            if proc is None:
                continue
            signals_tc.append(proc)
            kept_labels.append(labels_6[i])
            kept_ids.append(os.path.basename(record_paths[i]))

        if not signals_tc:
            if verbose:
                print(f"    [{center}] WARN: 0 successfully preprocessed (skip)")
            continue

        # Truncate to K (may be under-K if preprocess failures)
        signals_tc = np.stack(signals_tc[:k_per_center], axis=0).astype(np.float32)
        kept_labels_arr = np.stack(kept_labels[:k_per_center], axis=0).astype(np.float32)
        kept_ids = kept_ids[:k_per_center]

        out[center] = {
            "signals_tc": signals_tc,
            "labels_6":   kept_labels_arr,
            "record_ids": kept_ids,
        }
        if verbose:
            print(f"    [{center}] done: K={signals_tc.shape[0]} refs  "
                  f"(per-class pos in refs: " +
                  ", ".join(f"{TIER_M[j]}={int((kept_labels_arr[:, j]==1.0).sum())}"
                            for j in range(NUM_CLASSES_TIER_M)) +
                  f"), elapsed {time.time()-t0:.0f}s")

    # Cache
    if cache_path is not None and out:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        dump = {}
        for c, d in out.items():
            dump[f"{c}__signals"] = d["signals_tc"]
            dump[f"{c}__labels6"] = d["labels_6"]
            dump[f"{c}__record_ids"] = np.array(d["record_ids"], dtype=object)
        np.savez_compressed(cache_path, **dump)
        if verbose:
            print(f"[refs] cached -> {cache_path}")

    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--centers', nargs='+', default=MAIN_CENTERS_4)
    p.add_argument('--data_dir', default=DEFAULT_PN2021_DIR)
    p.add_argument('--cache', default=None,
                   help=f"cache NPZ path; default: {DEFAULT_REF_CACHE_DIR}/refs_cache_k{{k}}.npz")
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    cache = args.cache or os.path.join(DEFAULT_REF_CACHE_DIR, f"refs_cache_k{args.k}.npz")
    refs = sample_refs_per_center(
        k_per_center=args.k,
        centers=args.centers,
        data_dir=args.data_dir,
        cache_path=cache,
        seed=args.seed,
    )
    print("\n[refs] summary:")
    for c, d in refs.items():
        print(f"  {c}: K={d['signals_tc'].shape[0]} "
              f"signals_tc={d['signals_tc'].shape} labels_6={d['labels_6'].shape}")
