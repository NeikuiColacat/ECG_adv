"""Build quality/diversity reference selections for center prompt-token training.

This script reuses the existing `center_full_latents/{center}.pt` cache and
writes a selection JSON compatible with `train_center_prompt_tokens.py` and
`generate_center_prompt_token_synth.py`.

The first version is intentionally cheap: it uses metadata and VAE latent
typicality/diversity, not full WFDB digital features. That keeps it suitable for
quick K500 selection experiments without recomputing latents or materializing
waveforms.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()
DEFAULT_CACHE_ROOT = str(DATA_ROOT / "ecgtwin_prompt_token_super5/cache_v1")


def _as_float_array(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _base_quality(cache: Dict, idx: int, class_distance: float, p95_distance: float) -> float:
    multi_hot = _as_float_array(cache["super5_multi_hot"][idx])
    age = float(cache["age"][idx])
    hr = float(cache["hr"][idx])
    sex = str(cache["sex"][idx]).upper()
    primary_snomed = cache["primary_snomed"][idx]

    score = 1.0
    if np.isfinite(age) and 10.0 <= age <= 100.0:
        score += 0.08
    else:
        score -= 0.10
    if np.isfinite(hr) and 35.0 <= hr <= 160.0:
        score += 0.04
    if sex.startswith(("M", "F")):
        score += 0.04
    if primary_snomed is not None:
        score += 0.08

    n_labels = int((multi_hot > 0.5).sum())
    if n_labels == 1:
        score += 0.20
    elif n_labels > 1:
        score -= 0.10 * (n_labels - 1)

    if p95_distance > 0:
        rel = class_distance / p95_distance
        if rel > 1.25:
            score -= 0.30
        elif rel > 1.0:
            score -= 0.10
        elif 0.25 <= rel <= 0.90:
            score += 0.08
    return float(score)


def _greedy_quality_diverse(
    indices: List[int],
    latents_flat: np.ndarray,
    scores: Dict[int, float],
    n_take: int,
    seed: int,
    shortlist_multiplier: int = 5,
    diversity_weight: float = 0.35,
) -> List[int]:
    if n_take <= 0 or not indices:
        return []
    rng = random.Random(seed)
    ranked = sorted(indices, key=lambda i: (scores[i], rng.random()), reverse=True)
    shortlist_n = min(len(ranked), max(n_take, n_take * shortlist_multiplier))
    pool = ranked[:shortlist_n]
    selected = [pool[0]]
    remaining = pool[1:]
    score_values = np.asarray([scores[i] for i in pool], dtype=np.float32)
    score_span = float(score_values.max() - score_values.min()) or 1.0
    score_min = float(score_values.min())

    while remaining and len(selected) < n_take:
        sel_flat = latents_flat[np.asarray(selected)]
        rem_flat = latents_flat[np.asarray(remaining)]
        diff = rem_flat[:, None, :] - sel_flat[None, :, :]
        dist = np.sqrt(np.einsum("nmd,nmd->nm", diff, diff)).min(axis=1)
        dist = dist / (float(dist.max()) or 1.0)
        q = np.asarray([(scores[i] - score_min) / score_span for i in remaining], dtype=np.float32)
        objective = diversity_weight * dist + (1.0 - diversity_weight) * q
        best_i = int(np.argmax(objective))
        selected.append(remaining.pop(best_i))
    return selected


def build_selection(args: argparse.Namespace, center: str) -> None:
    cache_root = Path(args.cache_root)
    full_path = cache_root / "center_full_latents" / f"{center}.pt"
    if not full_path.exists():
        raise FileNotFoundError(full_path)
    selection_dir = cache_root / "ref_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    out_path = selection_dir / f"{center}_k{args.K}_seed{args.out_seed}.json"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] exists: {out_path}")
        return

    cache = torch.load(full_path, map_location="cpu", weights_only=False)
    latents = _as_float_array(cache["latents"]).reshape(len(cache["record_ids"]), -1)
    primary_classes = list(cache["primary_class"])
    by_class: Dict[str, List[int]] = {cls: [] for cls in CLASS_NAMES_SUPER5}
    for i, cls in enumerate(primary_classes):
        if cls in by_class:
            by_class[cls].append(i)

    distances = np.zeros((latents.shape[0],), dtype=np.float32)
    p95_by_class: Dict[str, float] = {}
    for cls, idxs in by_class.items():
        if not idxs:
            p95_by_class[cls] = 0.0
            continue
        arr = latents[np.asarray(idxs)]
        center_vec = arr.mean(axis=0, keepdims=True)
        dist = np.sqrt(((arr - center_vec) ** 2).sum(axis=1))
        distances[np.asarray(idxs)] = dist
        p95_by_class[cls] = float(np.percentile(dist, 95)) if len(dist) else 0.0

    scores = {
        i: _base_quality(cache, i, float(distances[i]), p95_by_class.get(primary_classes[i], 0.0))
        for i in range(latents.shape[0])
    }

    selected: List[int] = []
    selected_set = set()
    per_class_target = {cls: int(args.floor_per_class) for cls in CLASS_NAMES_SUPER5}
    for item in args.class_quota or []:
        if "=" not in item:
            raise ValueError(f"bad class quota {item!r}; expected CLASS=N")
        cls, value = item.split("=", 1)
        cls = cls.strip().upper()
        if cls not in per_class_target:
            raise ValueError(f"unknown class {cls!r}")
        per_class_target[cls] = int(value)

    for cls in CLASS_NAMES_SUPER5:
        pool = by_class.get(cls, [])
        n_take = min(per_class_target[cls], len(pool), max(0, args.K - len(selected)))
        picks = _greedy_quality_diverse(
            pool, latents, scores, n_take,
            seed=args.out_seed + CLASS_NAMES_SUPER5.index(cls) * 1009,
            shortlist_multiplier=args.shortlist_multiplier,
            diversity_weight=args.diversity_weight,
        )
        selected.extend(picks)
        selected_set.update(picks)

    remaining = [i for i in range(latents.shape[0]) if i not in selected_set]
    fill = _greedy_quality_diverse(
        remaining, latents, scores, max(0, args.K - len(selected)),
        seed=args.out_seed + 99991,
        shortlist_multiplier=max(1, args.shortlist_multiplier),
        diversity_weight=args.global_diversity_weight,
    )
    selected.extend(fill)
    selected = selected[:args.K]

    selected_ids = [cache["record_ids"][i] for i in selected]
    counts = Counter(primary_classes[i] for i in selected)
    score_values = [scores[i] for i in selected]
    selection = {
        "version": "ecgtwin_prompt_token_quality_ref_selection_v1",
        "center": center,
        "K_request": int(args.K),
        "K": len(selected),
        "seed": int(args.out_seed),
        "sampling_policy": "quality_diverse_floor_plus_quality_fill",
        "floor_per_class": int(args.floor_per_class),
        "class_quota": per_class_target,
        "shortlist_multiplier": int(args.shortlist_multiplier),
        "diversity_weight": float(args.diversity_weight),
        "global_diversity_weight": float(args.global_diversity_weight),
        "selected_indices_in_full_cache": selected,
        "selected_record_ids": selected_ids,
        "ref_record_ids": selected_ids,
        "primary_class_counts": dict(counts),
        "score_summary": {
            "mean": float(np.mean(score_values)) if score_values else None,
            "min": float(np.min(score_values)) if score_values else None,
            "max": float(np.max(score_values)) if score_values else None,
        },
        "selected_scores": [
            {
                "index": int(i),
                "record_id": str(cache["record_ids"][i]),
                "primary_class": str(primary_classes[i]),
                "score": float(scores[i]),
                "latent_class_distance": float(distances[i]),
                "multi_hot_n": int((_as_float_array(cache["super5_multi_hot"][i]) > 0.5).sum()),
                "primary_snomed": cache["primary_snomed"][i],
            }
            for i in selected
        ],
        "full_cache": str(full_path),
        "ref_exclusion_split_note": "Pass this JSON to eval/fine-tune exclusion logic; these refs are not validation/eval samples.",
    }
    with open(out_path, "w") as f:
        json.dump(selection, f, indent=2)
    print(f"[write] {center}: {dict(counts)} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--out_seed", type=int, default=1042)
    ap.add_argument("--floor_per_class", type=int, default=80)
    ap.add_argument("--class_quota", nargs="*", default=None,
                    help="Optional overrides like MI=120 STTC=120")
    ap.add_argument("--shortlist_multiplier", type=int, default=5)
    ap.add_argument("--diversity_weight", type=float, default=0.35)
    ap.add_argument("--global_diversity_weight", type=float, default=0.20)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for center in args.centers:
        build_selection(args, center)


if __name__ == "__main__":
    main()
