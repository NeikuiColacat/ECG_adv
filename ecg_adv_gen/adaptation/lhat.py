"""Pure Latent-Hull online AT helper logic.

This module deliberately excludes model state, ECGTwin decoding, PGD attacks,
DataLoaders, GPU operations, and filesystem side effects. It is a stable home
for sampling, soft-label, validation-split, and trust-map logic shared by the
legacy VAE-LHAT scripts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ecg_adv_gen.evaluation.pn2021c_protocol import (
    OFFICIAL_S5_COMPOSITE_OPS,
    official_s5_depth23_composites,
)
from ecg_adv_gen.labels.super5 import CLASS_NAMES_SUPER5

SUPER5_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES_SUPER5)}
DEFAULT_TRUST_HARDCODE = {"HYP": 0.0, "CD": 0.0}
OFFICIAL_S5_DEPTH23_COMPOSITE_CYCLE = "official_s5_depth23_composite_cycle"


def _ordered_ids_sha256(record_ids: Sequence[str]) -> str:
    payload = "\n".join(str(item) for item in record_ids) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_exact_eligibility_manifest(
    labels_one_hot: np.ndarray,
    record_ids: Sequence[str],
    *,
    min_nonself: int = 2,
) -> dict[str, Any]:
    """Describe anchors with enough distinct exact-label non-self partners."""
    labels = np.asarray(labels_one_hot, dtype=np.float32)
    ids = np.asarray(record_ids).astype(str)
    if labels.ndim != 2 or len(ids) != len(labels):
        raise ValueError("record_ids and 2D labels must align for exact eligibility")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("exact eligibility record_ids must be unique")
    min_nonself = int(min_nonself)
    if min_nonself < 1:
        raise ValueError("min_nonself must be positive")
    binary = (labels > 0.5).astype(np.int8)
    keys = ["".join(str(int(value)) for value in row) for row in binary]
    pools: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        pools.setdefault(key, []).append(index)
    eligible = [
        index for index, key in enumerate(keys)
        if len(pools[key]) - 1 >= min_nonself
    ]
    eligible_set = set(eligible)
    label_sets = {
        key: {
            "total_count": len(indices),
            "eligible_count": sum(index in eligible_set for index in indices),
            "distinct_nonself_count": max(0, len(indices) - 1),
        }
        for key, indices in sorted(pools.items())
    }
    class_counts = {}
    for class_index in range(labels.shape[1]):
        class_name = (
            CLASS_NAMES_SUPER5[class_index]
            if class_index < len(CLASS_NAMES_SUPER5)
            else f"class_{class_index}"
        )
        positive = np.flatnonzero(binary[:, class_index] > 0).tolist()
        class_counts[class_name] = {
            "total_positive": len(positive),
            "eligible_positive": sum(index in eligible_set for index in positive),
        }
    eligible_ids = ids[np.asarray(eligible, dtype=np.int64)].tolist()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "partner_policy": "exact_nonself",
        "min_nonself": min_nonself,
        "total_count": int(len(labels)),
        "eligible_count": len(eligible),
        "ineligible_count": int(len(labels) - len(eligible)),
        "eligible_pool_indices": eligible,
        "eligible_record_ids": eligible_ids,
        "ordered_record_ids_sha256": _ordered_ids_sha256(eligible_ids),
        "exact_label_sets": label_sets,
        "class_counts": class_counts,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def build_k500_internal_val_mask(
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
    min_per_present_class: int = 1,
) -> np.ndarray:
    """Build a deterministic validation mask using only K-shot labels."""
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim != 2:
        raise ValueError(f"expected labels 2D, got {labels.shape}")
    n = int(labels.shape[0])
    if n < 5:
        raise ValueError(f"need at least 5 target-real samples for internal val, got {n}")
    frac = float(val_fraction)
    if not 0.0 < frac < 0.5:
        raise ValueError(f"target_real_val_fraction must be in (0,0.5), got {frac}")
    rng = np.random.default_rng(int(seed))
    target_n = max(1, int(round(n * frac)))
    picked: set[int] = set()
    for cls_i in range(labels.shape[1]):
        pos = np.where(labels[:, cls_i] > 0.5)[0]
        if len(pos) == 0:
            continue
        take = min(len(pos), max(int(min_per_present_class), int(round(len(pos) * frac))))
        if take > 0:
            picked.update(int(i) for i in rng.choice(pos, size=take, replace=False))
    if len(picked) < target_n:
        rest = np.asarray([i for i in range(n) if i not in picked], dtype=np.int64)
        if len(rest) > 0:
            take = min(target_n - len(picked), len(rest))
            picked.update(int(i) for i in rng.choice(rest, size=take, replace=False))
    if len(picked) >= n:
        picked = set(sorted(picked)[:-1])
    mask = np.zeros((n,), dtype=bool)
    if picked:
        mask[np.asarray(sorted(picked), dtype=np.int64)] = True
    return mask


def parse_source_weight_map(raw: str | None) -> dict[str, float]:
    """Parse ``source=weight,source2=weight2`` into a dict."""
    if not raw:
        return {}
    out: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad source weight item {item!r}; expected source=weight")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def parse_class_source_weight_map(raw: str | None) -> dict[str, dict[str, float]]:
    """Parse ``CLASS:source=weight,CLASS2:source2=weight2`` overrides."""
    if not raw:
        return {}
    out: dict[str, dict[str, float]] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item or "=" not in item:
            raise ValueError(f"Bad class-source weight item {item!r}; expected CLASS:source=weight")
        cls, rest = item.split(":", 1)
        source, value = rest.split("=", 1)
        out.setdefault(cls.strip(), {})[source.strip()] = float(value)
    return out


def parse_class_weight_map(raw: str | None) -> dict[str, float]:
    """Parse ``CLASS=weight,CLASS2=weight2`` into a Super5 class weight map."""
    if not raw:
        return {}
    out: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad class weight item {item!r}; expected CLASS=weight")
        cls, value = item.split("=", 1)
        cls = cls.strip()
        if cls not in SUPER5_TO_IDX:
            raise ValueError(f"Unknown Super5 class in --anchor_class_weights: {cls!r}")
        out[cls] = max(0.0, float(value))
    return out


def split_anchor_sample_mode(mode: str) -> tuple[str, str]:
    """Split prefixed anchor sampling mode into scoring head and difficulty mode."""
    mode = str(mode)
    if mode.startswith("base_"):
        return "base", mode[len("base_"):]
    if mode.startswith("target_"):
        return "target_teacher", mode[len("target_"):]
    return "current", mode


def linear_warmup_value(
    target: float,
    epoch: int,
    warmup_epochs: int,
    *,
    start: float | None = None,
) -> float:
    """Linear epoch-level warmup used by adversarial stream weights."""
    target = float(target)
    if target <= 0.0:
        return 0.0
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return target
    start_value = 0.0 if start is None else float(start)
    if warmup_epochs == 1 or int(epoch) >= warmup_epochs:
        return target
    frac = max(0.0, float(int(epoch) - 1) / float(max(1, warmup_epochs - 1)))
    return start_value + frac * (target - start_value)


def weighted_anchor_quotas(
    classes_in_scope: list[str],
    class_sizes: dict[str, int],
    K_anchor: int,
    class_weights: dict[str, float],
) -> dict[str, int]:
    """Allocate per-class anchor quotas using only the known K-shot pool."""
    present = [c for c in classes_in_scope if int(class_sizes.get(c, 0)) > 0]
    if not present or K_anchor <= 0:
        return {c: 0 for c in classes_in_scope}
    weights = np.asarray([max(float(class_weights.get(c, 1.0)), 0.0) for c in present], dtype=np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones((len(present),), dtype=np.float64)
    raw = weights / weights.sum() * int(K_anchor)
    base = np.floor(raw).astype(int)
    if int(K_anchor) >= len(present):
        base = np.maximum(base, 1)
    rem = int(K_anchor) - int(base.sum())
    frac_order = np.argsort(-(raw - np.floor(raw)))
    i = 0
    while rem > 0:
        base[int(frac_order[i % len(frac_order)])] += 1
        rem -= 1
        i += 1
    while rem < 0:
        for idx in np.argsort(raw - np.floor(raw)):
            floor = 1 if int(K_anchor) >= len(present) else 0
            if base[int(idx)] > floor:
                base[int(idx)] -= 1
                rem += 1
                break
        else:
            break
    capacity = np.asarray([int(class_sizes.get(c, 0)) for c in present], dtype=int)
    base = np.minimum(base, capacity)

    remaining = min(int(K_anchor), int(capacity.sum())) - int(base.sum())
    while remaining > 0:
        available = np.where(base < capacity)[0]
        if available.size == 0:
            break
        safe_weights = np.where(weights > 0.0, weights, 1.0)
        ratios = base[available] / safe_weights[available]
        pick = int(available[int(np.argmin(ratios))])
        base[pick] += 1
        remaining -= 1

    quotas = {c: 0 for c in classes_in_scope}
    quotas.update({c: int(k) for c, k in zip(present, base)})
    return quotas


def derive_kshot_anchor_class_weights(
    labels_one_hot: np.ndarray,
    classes_in_scope: list[str],
    class_to_idx: dict[str, int],
    *,
    mode: str,
    source_labels: np.ndarray | None,
    reference_source: str,
    gamma: float,
    min_weight: float,
    max_weight: float,
    missing_weight: float,
    manual_prior: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Derive anchor quota weights from the known K-shot training subset."""
    if mode == "manual":
        weights = dict(manual_prior or {})
        return weights, {
            "mode": mode,
            "manual_prior": weights,
            "reference_source": reference_source,
            "reference_source_found": None,
            "reference_counts": {},
            "derived_weights": weights,
        }
    if mode != "inv_freq_kshot":
        raise ValueError(f"unsupported anchor_class_weight_mode={mode!r}")

    labels_one_hot = np.asarray(labels_one_hot, dtype=np.float32)
    ref_mask = np.ones((labels_one_hot.shape[0],), dtype=bool)
    reference_source_found: bool | None = None
    if source_labels is not None and reference_source and reference_source != "all":
        source_labels_arr = np.asarray(source_labels).astype(str)
        candidate_mask = source_labels_arr == str(reference_source)
        reference_source_found = bool(candidate_mask.any())
        if reference_source_found:
            ref_mask = candidate_mask
    ref_labels = labels_one_hot[ref_mask]
    counts: dict[str, int] = {}
    for cls in classes_in_scope:
        idx = class_to_idx[cls]
        counts[cls] = int((ref_labels[:, idx] > 0.5).sum())

    nonzero = np.asarray([v for v in counts.values() if v > 0], dtype=np.float64)
    reference_count = float(np.median(nonzero)) if nonzero.size else 1.0
    gamma = max(float(gamma), 0.0)
    min_weight = max(float(min_weight), 0.0)
    max_weight = max(float(max_weight), min_weight)
    missing_weight = max(float(missing_weight), 0.0)
    weights: dict[str, float] = {}
    for cls in classes_in_scope:
        count = int(counts[cls])
        if count <= 0:
            weight = missing_weight
        else:
            weight = (reference_count / max(float(count), 1.0)) ** gamma
            weight = min(max(weight, min_weight), max_weight)
        if manual_prior and cls in manual_prior:
            weight *= max(float(manual_prior[cls]), 0.0)
        weights[cls] = float(weight)
    return weights, {
        "mode": mode,
        "manual_prior": dict(manual_prior or {}),
        "reference_source": reference_source,
        "reference_source_found": reference_source_found,
        "reference_count": reference_count,
        "reference_counts": counts,
        "gamma": gamma,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "missing_weight": missing_weight,
        "derived_weights": weights,
    }


class StratifiedPoolWalker:
    """No-revisit-per-epoch stratified walker over a frozen latent/signal pool."""

    def __init__(
        self,
        labels_one_hot: np.ndarray,
        classes_in_scope: list[str],
        class_to_idx: dict[str, int],
        seed: int = 42,
        source_labels: np.ndarray | None = None,
        source_sampling_strategy: str = "class_balanced",
        source_weights: dict[str, float] | None = None,
        source_class_weights: dict[str, dict[str, float]] | None = None,
        source_floor_per_class: int = 0,
        eligible_indices: Sequence[int] | None = None,
    ):
        self.classes = list(classes_in_scope)
        self.class_to_idx = class_to_idx
        self.cls_pools: dict[str, np.ndarray] = {}
        self.cursors: dict[str, int] = {}
        self.epochs_completed: dict[str, int] = {c: 0 for c in self.classes}
        self.rng = np.random.default_rng(seed)
        self.source_sampling_strategy = source_sampling_strategy
        self.source_weights = dict(source_weights or {})
        self.source_class_weights = dict(source_class_weights or {})
        self.source_floor_per_class = max(0, int(source_floor_per_class))
        self.source_labels = None
        self.source_names: list[str] = []
        self.source_cls_pools: dict[tuple[str, str], np.ndarray] = {}
        self.source_cursors: dict[tuple[str, str], int] = {}
        self.source_epochs_completed: dict[tuple[str, str], int] = {}
        self.last_source_counts: dict[str, int] = {}
        self.last_class_source_counts: dict[str, dict[str, int]] = {}
        eligible_mask = np.ones(labels_one_hot.shape[0], dtype=bool)
        if eligible_indices is not None:
            eligible_mask[:] = False
            indices = np.asarray(list(eligible_indices), dtype=np.int64)
            if indices.size and (indices.min() < 0 or indices.max() >= len(eligible_mask)):
                raise ValueError("eligible_indices contains an out-of-range pool index")
            eligible_mask[indices] = True
        if source_labels is not None:
            if len(source_labels) != labels_one_hot.shape[0]:
                raise ValueError("source_labels length must match labels_one_hot")
            self.source_labels = np.asarray(source_labels).astype(str)
            self.source_names = sorted(str(s) for s in np.unique(self.source_labels))
        for c in self.classes:
            j = class_to_idx[c]
            mask = (labels_one_hot[:, j] > 0.5) & eligible_mask
            idx = np.where(mask)[0].copy()
            if len(idx) > 0:
                self.rng.shuffle(idx)
            self.cls_pools[c] = idx
            self.cursors[c] = 0
            if self.source_labels is not None:
                for source in self.source_names:
                    source_mask = mask & (self.source_labels == source)
                    source_idx = np.where(source_mask)[0].copy()
                    if len(source_idx) > 0:
                        self.rng.shuffle(source_idx)
                    key = (c, source)
                    self.source_cls_pools[key] = source_idx
                    self.source_cursors[key] = 0
                    self.source_epochs_completed[key] = 0

    def class_sizes(self) -> dict[str, int]:
        return {c: int(len(self.cls_pools[c])) for c in self.classes}

    def source_class_sizes(self) -> dict[str, dict[str, int]]:
        if self.source_labels is None:
            return {}
        return {
            c: {s: int(len(self.source_cls_pools.get((c, s), []))) for s in self.source_names}
            for c in self.classes
        }

    def _source_weight(self, cls: str, source: str) -> float:
        if cls in self.source_class_weights and source in self.source_class_weights[cls]:
            return float(self.source_class_weights[cls][source])
        return float(self.source_weights.get(source, 1.0))

    def _draw_from_pool(
        self,
        key: Any,
        k: int,
        pools: dict[Any, np.ndarray],
        cursors: dict[Any, int],
        epochs_completed: dict[Any, int],
    ) -> np.ndarray:
        pool = pools[key]
        if k <= 0 or len(pool) == 0:
            return np.empty(0, dtype=np.int64)
        cur = cursors[key]
        if cur + k > len(pool):
            self.rng.shuffle(pool)
            cur = 0
            epochs_completed[key] = int(epochs_completed.get(key, 0)) + 1
        out = pool[cur:cur + k].copy()
        cursors[key] = cur + len(out)
        return out

    def _source_quotas_for_class(self, cls: str, k: int) -> dict[str, int]:
        active = []
        for source in self.source_names:
            capacity = int(len(self.source_cls_pools.get((cls, source), [])))
            weight = self._source_weight(cls, source)
            if capacity > 0 and weight > 0.0:
                active.append((source, capacity, weight))
        if not active or k <= 0:
            return {}
        quotas = {source: 0 for source, _, _ in active}
        remaining = int(k)

        if self.source_floor_per_class > 0:
            for source, capacity, _ in active:
                take = min(self.source_floor_per_class, capacity, remaining)
                quotas[source] += take
                remaining -= take
                if remaining <= 0:
                    break

        while remaining > 0:
            candidates = [
                (source, weight)
                for source, capacity, weight in active
                if quotas[source] < capacity
            ]
            if not candidates:
                break
            probs = np.asarray([w for _, w in candidates], dtype=np.float64)
            probs = probs / probs.sum()
            chosen_i = int(self.rng.choice(np.arange(len(candidates)), p=probs))
            quotas[candidates[chosen_i][0]] += 1
            remaining -= 1
        return quotas

    def sample(self, k_per_class: dict[str, int]) -> dict[str, np.ndarray]:
        """Draw ``k_per_class[c]`` indices for each class without revisit per epoch."""
        out: dict[str, np.ndarray] = {}
        self.last_source_counts = {}
        self.last_class_source_counts = {}
        for c in self.classes:
            k = int(k_per_class.get(c, 0))
            if (
                self.source_sampling_strategy == "source_weighted"
                and self.source_labels is not None
                and self.source_names
            ):
                quotas = self._source_quotas_for_class(c, k)
                pieces = []
                self.last_class_source_counts[c] = {}
                for source in self.source_names:
                    q = int(quotas.get(source, 0))
                    key = (c, source)
                    drawn = self._draw_from_pool(
                        key,
                        q,
                        self.source_cls_pools,
                        self.source_cursors,
                        self.source_epochs_completed,
                    )
                    if drawn.size > 0:
                        pieces.append(drawn)
                    self.last_class_source_counts[c][source] = int(drawn.size)
                    self.last_source_counts[source] = self.last_source_counts.get(source, 0) + int(drawn.size)
                out[c] = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
                if out[c].size > 1:
                    self.rng.shuffle(out[c])
                continue
            out[c] = self._draw_from_pool(c, k, self.cls_pools, self.cursors, self.epochs_completed)
        return out


class SameLabelLatentIndex:
    """Nearest-neighbor candidate index for constrained latent-hull attacks."""

    def __init__(
        self,
        latents: np.ndarray,
        labels_one_hot: np.ndarray,
        label_mode: str = "primary",
        seed: int = 42,
        include_self: bool = False,
        distance_space: str = "raw",
        neighbor_mode: str = "nearest",
        neighbor_pool_size: int = 0,
        neighbor_pool_multiplier: int = 4,
    ):
        if label_mode not in {"primary", "exact", "compatible"}:
            raise ValueError(f"label_mode must be primary|exact|compatible, got {label_mode!r}")
        if distance_space not in {"raw", "standardized"}:
            raise ValueError(f"distance_space must be raw|standardized, got {distance_space!r}")
        if neighbor_mode not in {"nearest", "local_random", "random"}:
            raise ValueError(f"neighbor_mode must be nearest|local_random|random, got {neighbor_mode!r}")
        self.latents = latents.astype(np.float32, copy=False)
        self.labels = labels_one_hot.astype(np.float32, copy=False)
        self.label_mode = label_mode
        self.include_self = bool(include_self)
        self.distance_space = distance_space
        self.neighbor_mode = neighbor_mode
        self.neighbor_pool_size = int(neighbor_pool_size)
        self.neighbor_pool_multiplier = max(1, int(neighbor_pool_multiplier))
        self.rng = np.random.default_rng(seed)
        self.last_candidate_indices: np.ndarray | None = None
        flat_latents = self.latents.reshape(self.latents.shape[0], -1)
        if distance_space == "standardized":
            mu = flat_latents.mean(axis=0, keepdims=True)
            sigma = flat_latents.std(axis=0, keepdims=True)
            sigma = np.where(sigma < 1e-6, 1.0, sigma)
            self.flat_distance = ((flat_latents - mu) / sigma).astype(np.float32)
        else:
            self.flat_distance = flat_latents.astype(np.float32, copy=False)
        if label_mode == "primary":
            keys = [int(i) for i in self.labels.argmax(axis=1)]
        elif label_mode == "exact":
            keys = [tuple(int(v) for v in row) for row in (self.labels > 0.5).astype(np.int8)]
        else:
            norm_idx = SUPER5_TO_IDX["NORM"]
            abnormal = np.delete(np.arange(self.labels.shape[1]), norm_idx)
            is_norm_only = (self.labels[:, norm_idx] > 0.5) & (self.labels[:, abnormal].sum(axis=1) == 0)
            has_abnormal = self.labels[:, abnormal].sum(axis=1) > 0
            keys = ["NORM_ONLY" if n else "ABNORMAL" if a else "OTHER" for n, a in zip(is_norm_only, has_abnormal)]
        self.keys = keys
        self.pools: dict[Any, np.ndarray] = {}
        for i, key in enumerate(keys):
            self.pools.setdefault(key, []).append(i)
        self.pools = {k: np.asarray(v, dtype=np.int64) for k, v in self.pools.items()}

    def class_sizes(self) -> dict[str, int]:
        return {str(k): int(len(v)) for k, v in self.pools.items()}

    def eligible_anchor_indices(self, *, min_nonself: int = 2) -> np.ndarray:
        min_nonself = int(min_nonself)
        if min_nonself < 1:
            raise ValueError("min_nonself must be positive")
        return np.asarray(
            [
                index for index, key in enumerate(self.keys)
                if len(self.pools.get(key, ())) - 1 >= min_nonself
            ],
            dtype=np.int64,
        )

    def _choose_neighbors(self, pool: np.ndarray, order: np.ndarray, M: int) -> np.ndarray:
        if self.neighbor_mode == "nearest":
            return pool[order[:M]]
        if self.neighbor_mode == "random":
            if len(pool) <= M:
                return pool.copy()
            return self.rng.choice(pool, size=M, replace=False).astype(np.int64)
        top_k = self.neighbor_pool_size
        if top_k <= 0:
            top_k = max(M, M * self.neighbor_pool_multiplier)
        local_pool = pool[order[: min(int(top_k), len(order))]]
        if len(local_pool) <= M:
            return local_pool.copy()
        return self.rng.choice(local_pool, size=M, replace=False).astype(np.int64)

    def candidates_for(self, anchor_indices: np.ndarray, M: int) -> np.ndarray:
        """Return ``(B, M, 4, 128)`` same-label candidate latents."""
        out = np.empty((len(anchor_indices), M, 4, 128), dtype=np.float32)
        out_indices = np.empty((len(anchor_indices), M), dtype=np.int64)
        for row_i, anchor_idx in enumerate(anchor_indices):
            anchor_idx = int(anchor_idx)
            key = self.keys[anchor_idx]
            pool = self.pools.get(key, np.empty((0,), dtype=np.int64))
            pool = pool[pool != anchor_idx]
            if len(pool) == 0 and not self.include_self:
                raise ValueError(
                    f"anchor {anchor_idx} has no non-self candidate for "
                    f"label_mode={self.label_mode!r}; anchor fallback is forbidden"
                )
            diff = self.flat_distance[pool] - self.flat_distance[anchor_idx]
            dist2 = np.einsum("ij,ij->i", diff, diff)
            order = np.argsort(dist2)
            if self.include_self:
                neighbor_budget = max(M - 1, 0)
                chosen = np.concatenate(
                    [
                        np.asarray([anchor_idx], dtype=np.int64),
                        self._choose_neighbors(pool, order, neighbor_budget),
                    ]
                )
            else:
                chosen = self._choose_neighbors(pool, order, M)
            if len(chosen) < M:
                pad_value = int(chosen[-1]) if len(chosen) else anchor_idx
                pad = np.full((M - len(chosen),), pad_value, dtype=np.int64)
                chosen = np.concatenate([chosen, pad])
            chosen = chosen[:M]
            out_indices[row_i] = chosen
            out[row_i] = self.latents[chosen]
        self.last_candidate_indices = out_indices
        return out


def build_anchor_preserving_soft_labels(
    anchor_labels: np.ndarray,
    candidate_labels: np.ndarray,
    weights: np.ndarray,
    *,
    lambda_y: float,
    positive_value: float,
    negative_floor: float,
    new_class_cap: float,
    class_to_idx: dict[str, int] | None = None,
) -> np.ndarray:
    """Build label-smoothing targets from latent-hull candidate weights."""
    anchor = anchor_labels.astype(np.float32, copy=False)
    cand = candidate_labels.astype(np.float32, copy=False)
    w = weights.astype(np.float32, copy=False)
    q = (w[:, :, None] * cand).sum(axis=1)

    out = np.full(anchor.shape, float(negative_floor), dtype=np.float32)
    pos_mask = anchor > 0.5
    out[pos_mask] = float(positive_value)
    new_soft = np.clip(float(lambda_y) * q, float(negative_floor), float(new_class_cap))
    out[~pos_mask] = new_soft[~pos_mask]

    class_to_idx = class_to_idx or SUPER5_TO_IDX
    norm_idx = class_to_idx["NORM"]
    abnormal_idx = [i for i in range(anchor.shape[1]) if i != norm_idx]
    abnormal_anchor = anchor[:, abnormal_idx].sum(axis=1) > 0.5
    out[abnormal_anchor, norm_idx] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def auroc_to_trust(auroc: float | None) -> float:
    """Plan Rev 13: AUROC > 0.7 -> 1.0; > 0.55 -> 0.5; else 0.0."""
    if auroc is None:
        return 0.0
    if auroc > 0.7:
        return 1.0
    if auroc > 0.55:
        return 0.5
    return 0.0


def derive_class_trust(
    per_class_auroc: dict[str, float | None],
    *,
    class_names: tuple[str, ...] = CLASS_NAMES_SUPER5,
    hardcoded: dict[str, float] | None = None,
) -> dict[str, float]:
    out = {c: auroc_to_trust(per_class_auroc.get(c)) for c in class_names}
    out.update(hardcoded if hardcoded is not None else DEFAULT_TRUST_HARDCODE)
    return out


def global_zscore_np(sig_ct: np.ndarray) -> np.ndarray:
    """Per-record global z-score for canonical ``(12, L)`` ECG tensors."""
    mean = float(sig_ct.mean())
    std = float(sig_ct.std())
    if std < 1e-8:
        return (sig_ct - mean).astype(np.float32)
    return ((sig_ct - mean) / std).astype(np.float32)


def build_three_chain_vae_lhat_augmix_views(
    anchor_signals_ct: np.ndarray,
    adv_signals_ct: np.ndarray,
    *,
    copies: int,
    severity: int,
    depth: int,
    alpha: float,
    ops: Sequence[str],
    rng: np.random.Generator,
    op_apply_fn: Callable[[np.ndarray, str, int, str], np.ndarray],
    available_ops: Sequence[str],
    severity_profile: str = "standard",
    width: int = 3,
    mixture_mode: str = "beta",
    mixture_prob: float = 0.5,
    mixture_beta_a: float | None = None,
    mixture_beta_b: float | None = None,
    op_schedule: str = "random",
    chain_weights: Sequence[float] | None = None,
    renorm: bool = False,
    clip_abs: float = 6.0,
    third_chain_role: str = "vae_lhat_adversarial_waveform",
    chain_base_mode: str = "clean_clean_third",
    adv_base_mix: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Locked PN2021-C topology with an explicit chain base ablation knob."""
    if copies <= 0:
        return (
            np.empty((0,) + tuple(anchor_signals_ct.shape[1:]), dtype=np.float32),
            {
                "enabled": False,
                "topology": "locked_three_chain_vae_lhat_augmix",
                "n_generated": 0,
            },
        )
    if anchor_signals_ct.ndim != 3:
        raise ValueError(f"anchor_signals_ct must be 3D (N,12,L), got {anchor_signals_ct.shape}")
    if anchor_signals_ct.shape != adv_signals_ct.shape:
        raise ValueError(f"anchor/adv shape mismatch: {anchor_signals_ct.shape} vs {adv_signals_ct.shape}")
    if int(width) != 3:
        raise ValueError("locked VAE-LHAT AugMix requires exactly three chains: two corruption chains plus one adversarial chain")
    if not (1 <= int(severity) <= 10):
        raise ValueError(f"severity must be in [1,10], got {severity}")
    if float(alpha) <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    mixture_mode = str(mixture_mode)
    if mixture_mode not in {"beta", "fixed"}:
        raise ValueError(f"mixture_mode must be 'beta' or 'fixed', got {mixture_mode!r}")
    if mixture_mode == "fixed" and not (0.0 <= float(mixture_prob) <= 1.0):
        raise ValueError("fixed mixture_prob must be in [0, 1]")
    op_schedule = str(op_schedule)
    if op_schedule not in {"random", "cycle", "per_op", OFFICIAL_S5_DEPTH23_COMPOSITE_CYCLE}:
        raise ValueError(
            "op_schedule must be random|cycle|per_op|"
            f"{OFFICIAL_S5_DEPTH23_COMPOSITE_CYCLE}, got {op_schedule!r}"
        )
    beta_a = float(alpha) if mixture_beta_a is None else float(mixture_beta_a)
    beta_b = float(alpha) if mixture_beta_b is None else float(mixture_beta_b)
    if beta_a <= 0 or beta_b <= 0:
        raise ValueError("mixture beta parameters must be > 0")
    if not ops:
        raise ValueError("ops must contain at least one op")
    third_chain_role = str(third_chain_role)
    if third_chain_role not in {"vae_lhat_adversarial_waveform", "clean_anchor_control"}:
        raise ValueError(
            "third_chain_role must be vae_lhat_adversarial_waveform or clean_anchor_control"
        )
    chain_base_mode = str(chain_base_mode)
    if chain_base_mode not in {
        "clean_clean_third",
        "all_clean",
        "one_adv",
        "all_adv",
        "all_clean_plus_vae_adv",
    }:
        raise ValueError(
            "chain_base_mode must be clean_clean_third, all_clean, one_adv, "
            "all_adv, or all_clean_plus_vae_adv"
        )
    adv_base_mix = float(adv_base_mix)
    if not (0.0 <= adv_base_mix <= 1.0):
        raise ValueError("adv_base_mix must be in [0, 1]")
    available = set(str(op) for op in available_ops)
    for op_name in ops:
        if op_name not in available:
            raise ValueError(f"unknown locked VAE-LHAT AugMix op: {op_name}")
    official_combos: list[tuple[str, ...]] = []
    if op_schedule == OFFICIAL_S5_DEPTH23_COMPOSITE_CYCLE:
        official_ops = set(OFFICIAL_S5_COMPOSITE_OPS)
        if set(str(op) for op in ops) != official_ops:
            raise ValueError(
                f"{OFFICIAL_S5_DEPTH23_COMPOSITE_CYCLE} requires exactly the official S5 ops"
            )
        if not official_ops.issubset(available):
            raise ValueError(f"available_ops must contain all official S5 ops for {op_schedule}")
        official_combos = [tuple(name.split("+")) for name in official_s5_depth23_composites()]
    fixed_chain_weights: np.ndarray | None = None
    if chain_weights is not None:
        fixed_chain_weights = np.asarray(list(chain_weights), dtype=np.float32)
        if fixed_chain_weights.shape != (3,):
            raise ValueError("chain_weights must contain exactly three values for locked_three_chain")
        if not np.isfinite(fixed_chain_weights).all() or np.any(fixed_chain_weights < 0):
            raise ValueError("chain_weights must be finite non-negative values")
        weight_sum = float(fixed_chain_weights.sum())
        if weight_sum <= 0.0:
            raise ValueError("chain_weights must sum to a positive value")
        fixed_chain_weights = (fixed_chain_weights / weight_sum).astype(np.float32)

    mixed: list[np.ndarray] = []
    adv_weights: list[float] = []
    beta_ms: list[float] = []
    chain_depths: list[int] = []
    used_ops: list[str] = []
    view_ops: list[str] = []
    view_combos: list[str] = []

    for copy_i in range(int(copies)):
        for i in range(anchor_signals_ct.shape[0]):
            x0 = anchor_signals_ct[i].astype(np.float32, copy=False)
            x_adv = adv_signals_ct[i].astype(np.float32, copy=False)
            if fixed_chain_weights is None:
                weights = rng.dirichlet([float(alpha)] * 3).astype(np.float32)
            else:
                weights = fixed_chain_weights
            if mixture_mode == "fixed":
                m = float(mixture_prob)
            else:
                m = float(rng.beta(beta_a, beta_b))
            beta_ms.append(m)

            branch_mix = np.zeros_like(x0, dtype=np.float32)
            ops_for_view: list[str] = []
            full_corruption_modes = {"all_clean", "one_adv", "all_adv", "all_clean_plus_vae_adv"}
            corruption_chain_count = 3 if chain_base_mode in full_corruption_modes else 2
            soft_adv_base = (1.0 - adv_base_mix) * x0 + adv_base_mix * x_adv
            for chain_i in range(corruption_chain_count):
                corruption_base = (
                    soft_adv_base
                    if chain_base_mode == "all_adv"
                    or (chain_base_mode == "one_adv" and chain_i == 2)
                    else x0
                )
                if official_combos:
                    combo_i = (
                        copy_i * anchor_signals_ct.shape[0] * corruption_chain_count
                        + i * corruption_chain_count
                        + chain_i
                    ) % len(official_combos)
                    combo_ops = official_combos[combo_i]
                    combo_name = "+".join(combo_ops)
                    chain_depths.append(len(combo_ops))
                    sig = corruption_base.copy()
                    for op_name in combo_ops:
                        sig = op_apply_fn(sig, op_name, int(severity), str(severity_profile)).astype(
                            np.float32,
                            copy=False,
                        )
                        used_ops.append(op_name)
                    ops_for_view.append(combo_name)
                    branch_mix = branch_mix + float(weights[chain_i]) * sig
                    continue
                d = int(depth) if int(depth) > 0 else int(rng.integers(1, 4))
                chain_depths.append(d)
                sig = corruption_base.copy()
                for step_i in range(d):
                    if op_schedule == "per_op":
                        op_name = str(ops[int(copy_i) % len(ops)])
                    elif op_schedule == "cycle":
                        op_i = int(copy_i + i + chain_i + step_i) % len(ops)
                        op_name = str(ops[op_i])
                    else:
                        op_name = str(rng.choice(ops))
                    sig = op_apply_fn(sig, op_name, int(severity), str(severity_profile)).astype(
                        np.float32,
                        copy=False,
                    )
                    used_ops.append(op_name)
                    ops_for_view.append(op_name)
                branch_mix = branch_mix + float(weights[chain_i]) * sig

            if chain_base_mode == "clean_clean_third":
                third_chain_weight = float(weights[2])
                adv_weights.append(
                    third_chain_weight
                    if third_chain_role == "vae_lhat_adversarial_waveform"
                    else 0.0
                )
                third_chain_signal = (
                    x_adv
                    if third_chain_role == "vae_lhat_adversarial_waveform"
                    else x0
                )
                branch_mix = branch_mix + third_chain_weight * third_chain_signal
            else:
                adv_weights.append(
                    float(weights[2])
                    if chain_base_mode == "one_adv"
                    else 1.0 if chain_base_mode == "all_adv" else 0.0
                )

            mix_base = soft_adv_base if chain_base_mode == "all_adv" else x0
            out = (1.0 - m) * mix_base + m * branch_mix
            if renorm:
                out = global_zscore_np(out)
            else:
                out = out.astype(np.float32, copy=False)
            if clip_abs > 0:
                out = np.clip(out, -float(clip_abs), float(clip_abs)).astype(np.float32)
            mixed.append(out)
            view_ops.append(
                ops_for_view[0]
                if ops_for_view and len(set(ops_for_view)) == 1
                else "__mixed__"
            )
            view_combos.append("+".join(ops_for_view) if ops_for_view else "")

    arr = np.stack(mixed, axis=0).astype(np.float32) if mixed else np.empty(
        (0,) + tuple(anchor_signals_ct.shape[1:]), dtype=np.float32
    )
    op_counts = {op: int(used_ops.count(op)) for op in sorted(set(used_ops))}
    if chain_base_mode == "all_adv":
        chain_roles = ["vae_lhat_adversarial_corruption"] * 3
    elif chain_base_mode == "one_adv":
        chain_roles = [
            "clean_anchor_corruption",
            "clean_anchor_corruption",
            "vae_lhat_adversarial_corruption",
        ]
    elif chain_base_mode in {"all_clean", "all_clean_plus_vae_adv"}:
        chain_roles = ["clean_anchor_corruption"] * 3
    else:
        chain_roles = ["clean_anchor_corruption", "clean_anchor_corruption", third_chain_role]
    stats = {
        "enabled": True,
        "topology": "locked_three_chain_vae_lhat_augmix",
        "view_mode": "vae_lhat_augmix",
        "n_generated": int(arr.shape[0]),
        "n_corrupted": int(arr.shape[0]),
        "corrupt_fraction": 1.0,
        "copies": int(copies),
        "severity": int(severity),
        "severity_profile": str(severity_profile),
        "width": 3,
        "depth": int(depth),
        "alpha": float(alpha),
        "mixture_mode": mixture_mode,
        "mixture_prob": float(mixture_prob),
        "mixture_beta_a": beta_a,
        "mixture_beta_b": beta_b,
        "op_schedule": op_schedule,
        "chain_base_mode": chain_base_mode,
        "adv_base_mix": float(adv_base_mix),
        "composite_schedule": bool(official_combos),
        "composite_count": int(len(official_combos)),
        "chain_weight_mode": "fixed" if fixed_chain_weights is not None else "dirichlet",
        "chain_weights": (
            [round(float(v), 6) for v in fixed_chain_weights.tolist()]
            if fixed_chain_weights is not None
            else []
        ),
        "corruption_chain_count": (
            3
            if chain_base_mode in {"all_clean", "one_adv", "all_adv", "all_clean_plus_vae_adv"}
            else 2
        ),
        "adversarial_chain_count": (
            3 if chain_base_mode == "all_adv"
            else 1 if chain_base_mode == "one_adv"
            else 0 if chain_base_mode in {"all_clean", "all_clean_plus_vae_adv"}
            else 1 if third_chain_role == "vae_lhat_adversarial_waveform"
            else 0
        ),
        "clean_anchor_control_chain_count": 3 if chain_base_mode in {"all_clean", "all_clean_plus_vae_adv"} else (
            2 if chain_base_mode == "one_adv" else
            1 if third_chain_role == "clean_anchor_control" else 0
        ),
        "adversarial_chain_index": 2 if chain_base_mode in {"clean_clean_third", "one_adv"} else None,
        "adversarial_chain_corrupted": chain_base_mode in {"one_adv", "all_adv"},
        "chain_roles": chain_roles,
        "adv_weight_mean": float(np.mean(adv_weights)) if adv_weights else float("nan"),
        "adv_weight_max": float(np.max(adv_weights)) if adv_weights else float("nan"),
        "beta_m_mean": float(np.mean(beta_ms)) if beta_ms else float("nan"),
        "chain_depth_mean": float(np.mean(chain_depths)) if chain_depths else float("nan"),
        "ops": list(ops),
        "op_counts": op_counts,
        "view_ops": view_ops,
        "view_combos": view_combos,
        "renorm": bool(renorm),
        "clip_abs": float(clip_abs),
    }
    return arr, stats
