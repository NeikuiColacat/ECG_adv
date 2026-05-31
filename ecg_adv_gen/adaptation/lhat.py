"""Pure Latent-Hull online AT helper logic.

This module deliberately excludes model state, ECGTwin decoding, PGD attacks,
DataLoaders, GPU operations, and filesystem side effects. It is a stable home
for sampling, soft-label, validation-split, and trust-map logic shared by the
legacy VAE-LHAT scripts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ecg_adv_gen.labels.super5 import CLASS_NAMES_SUPER5

SUPER5_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES_SUPER5)}
DEFAULT_TRUST_HARDCODE = {"HYP": 0.0, "CD": 0.0}


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
        if source_labels is not None:
            if len(source_labels) != labels_one_hot.shape[0]:
                raise ValueError("source_labels length must match labels_one_hot")
            self.source_labels = np.asarray(source_labels).astype(str)
            self.source_names = sorted(str(s) for s in np.unique(self.source_labels))
        for c in self.classes:
            j = class_to_idx[c]
            mask = labels_one_hot[:, j] > 0.5
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
        """Return ``(B, M, *latent_shape)`` same-label candidate latents."""
        out = np.empty((len(anchor_indices), M, *self.latents.shape[1:]), dtype=np.float32)
        out_indices = np.empty((len(anchor_indices), M), dtype=np.int64)
        for row_i, anchor_idx in enumerate(anchor_indices):
            anchor_idx = int(anchor_idx)
            key = self.keys[anchor_idx]
            pool = self.pools.get(key, np.asarray([anchor_idx], dtype=np.int64))
            pool = pool[pool != anchor_idx]
            if len(pool) == 0:
                pool = np.asarray([anchor_idx], dtype=np.int64)
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


def dirichlet_with_first_weight_cap(
    width: int,
    alpha: float,
    first_weight_cap: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample AugMix branch weights while capping branch 0."""
    weights = rng.dirichlet([float(alpha)] * int(width)).astype(np.float32)
    cap = float(np.clip(first_weight_cap, 0.0, 1.0))
    if width <= 1 or weights[0] <= cap:
        return weights
    rest = weights[1:]
    rest_sum = float(rest.sum())
    weights[0] = cap
    if rest_sum <= 1e-8:
        weights[1:] = (1.0 - cap) / float(width - 1)
    else:
        weights[1:] = (1.0 - cap) * rest / rest_sum
    return weights.astype(np.float32)


def build_latent_augmix_branch_signals(
    anchor_signals_ct: np.ndarray,
    adv_signals_ct: np.ndarray,
    *,
    copies: int,
    severity: int,
    width: int,
    depth: int,
    alpha: float,
    latent_weight_cap: float,
    ops: Sequence[str],
    rng: np.random.Generator,
    op_apply_fn: Callable[[np.ndarray, str, int], np.ndarray],
    available_ops: Sequence[str],
    renorm: bool = True,
    clip_abs: float = 6.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the latent-hull adversarial decode as one AugMix branch."""
    if copies <= 0:
        return (
            np.empty((0,) + tuple(anchor_signals_ct.shape[1:]), dtype=np.float32),
            {"enabled": False, "n_generated": 0},
        )
    if width < 2:
        raise ValueError("--latent_augmix_width must be >= 2 because branch 0 is x_adv")
    if not (1 <= severity <= 10):
        raise ValueError(f"--latent_augmix_severity must be in [1, 10], got {severity}")
    if not ops:
        raise ValueError("--latent_augmix_ops must contain at least one op")
    available = set(str(op) for op in available_ops)
    for op_name in ops:
        if op_name not in available:
            raise ValueError(f"unknown latent AugMix op: {op_name}")
    if anchor_signals_ct.shape != adv_signals_ct.shape:
        raise ValueError(f"anchor/adv shape mismatch: {anchor_signals_ct.shape} vs {adv_signals_ct.shape}")

    mixed: list[np.ndarray] = []
    latent_weights: list[float] = []
    beta_ms: list[float] = []
    chain_depths: list[int] = []
    for _copy_i in range(int(copies)):
        for i in range(anchor_signals_ct.shape[0]):
            x0 = anchor_signals_ct[i].astype(np.float32, copy=False)
            x_adv = adv_signals_ct[i].astype(np.float32, copy=False)
            weights = dirichlet_with_first_weight_cap(
                width=width,
                alpha=alpha,
                first_weight_cap=latent_weight_cap,
                rng=rng,
            )
            m = float(rng.beta(float(alpha), float(alpha)))

            branch_mix = weights[0] * x_adv
            latent_weights.append(float(weights[0]))
            beta_ms.append(m)

            for branch_i in range(1, width):
                d = int(depth) if depth > 0 else int(rng.integers(1, 4))
                chain_depths.append(d)
                sig = x0.copy()
                for _ in range(d):
                    op_name = str(rng.choice(ops))
                    sig = op_apply_fn(sig, op_name, int(severity)).astype(np.float32, copy=False)
                branch_mix = branch_mix + float(weights[branch_i]) * sig

            out = (1.0 - m) * x0 + m * branch_mix
            if renorm:
                out = global_zscore_np(out)
            else:
                out = out.astype(np.float32, copy=False)
            if clip_abs > 0:
                out = np.clip(out, -float(clip_abs), float(clip_abs)).astype(np.float32)
            mixed.append(out)

    mixed_arr = np.stack(mixed, axis=0).astype(np.float32) if mixed else np.empty(
        (0,) + tuple(anchor_signals_ct.shape[1:]), dtype=np.float32
    )
    stats = {
        "enabled": True,
        "n_generated": int(mixed_arr.shape[0]),
        "copies": int(copies),
        "severity": int(severity),
        "width": int(width),
        "depth": int(depth),
        "alpha": float(alpha),
        "latent_weight_cap": float(latent_weight_cap),
        "latent_weight_mean": float(np.mean(latent_weights)) if latent_weights else float("nan"),
        "latent_weight_max": float(np.max(latent_weights)) if latent_weights else float("nan"),
        "beta_m_mean": float(np.mean(beta_ms)) if beta_ms else float("nan"),
        "chain_depth_mean": float(np.mean(chain_depths)) if chain_depths else float("nan"),
        "renorm": bool(renorm),
        "clip_abs": float(clip_abs),
        "ops": list(ops),
    }
    return mixed_arr, stats
