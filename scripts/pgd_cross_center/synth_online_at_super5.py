"""Online AT on Super5 victim with synth-anchored PGD (Plan Rev 8).

Each epoch:
  A) Sample K_anchor latents (stratified by class) from a frozen Stage-1 synth
     pool (.latent.npz).
  B) Run K_pgd-step PGD on the current victim → (K, 12, 1000) adv signals +
     one-hot target labels.
  C) Run gates per epoch (Plan Rev 6 Issue #38):
       - ASR gate     : compute_asr → require asr_overall ≥ 0.30 over a sliding
                         3-epoch window (else raise — PGD is broken)
       - Semantic gate: compute_semantic_gate → if Einthoven p95 / HR / QRS
                         ratio fail, do not push the (otherwise high-confidence)
                         garbage into the buffer.
  D) On gate pass, push (signal_ct_250, target_label_5_with_-1_sentinel, score)
     into a QualityAwareBuffer (FIFO + informativeness eviction).
  E) Build a mixed DataLoader with three streams:
       PTBXL real  weight 1.0
       roundtrip   weight 0.5    (Plan Issue #29 — keep against AugMix-off)
       adv buffer  weight 2.0    (cold-start guard: weight=0 in epoch 0/empty)
  F) train_one_epoch with masked BCE on the -1 sentinel + EWA anchor regularizer
     (Plan Issue #21 Q4 — ADR ICLR 2024 EMA self-distill).
  G) every eval_every epoch: PTB-XL fold9 val + PN2021 quick subset macro AUROC,
     update best ckpt.

NOT done in this fork (per Plan Rev 8 explicit non-goals):
  - AugMix latent injection (Issue #25 (b) — off in pilot).
  - K-sensitivity grid (Issue #25 (a) — K=200 only).
  - BoundaryAdvDiff (already known to fail -0.91pp; replaced with PGDAdvDiff).
  - CenterToken hook injection at training time (Stage 1 produced the synth
    latents already; the trainer only sees decoded signals).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import (
    DataLoader, ConcatDataset, TensorDataset, WeightedRandomSampler,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path("/root/autodl-tmp/models/DeepECG/notebooks")))

from adversarial.adv_validation import compute_asr, compute_semantic_gate  # noqa: E402
from adversarial.efficientnet_victim_tierM import (  # noqa: E402
    EfficientNetVictimTierM, TIERM_INPUT_LENGTH,
)
from adversarial.latent_hull_pgd import LatentHullPGDGenerator  # noqa: E402
from adversarial.pgd_advdiff import PGDAdvDiffGenerator  # noqa: E402

from scripts.crosscenter_tierM.online_adv_train_tierM import (  # noqa: E402
    _center_crop_ct,
    QualityAwareBufferTierM as QualityAwareBuffer,
    build_roundtrip_anchor_dataset,
    train_one_epoch_tierM as train_one_epoch_masked_bce,
)
from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5, NUM_SUPER5, snomed_list_to_super5,
)
from scripts.triple_labels.model_zoo import available_model_names  # noqa: E402
from scripts.triple_labels.train_ptbxl import (  # noqa: E402
    PTBXLDatasetScheme, compute_pos_weight, compute_macro_auroc_auprc,
    evaluate, get_ptbxl_labels_for_scheme, preprocess_ptbxl_all,
)
from scripts.triple_labels.eval_crosscenter import parse_header_snomed  # noqa: E402
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402

DEFAULT_PN2021_DIR = "/root/autodl-tmp/physionet2021/training"
DEFAULT_PTBXL_RAW = "/root/autodl-tmp/ptbxl/raw100.npy"
DEFAULT_PTBXL_CSV = "/root/autodl-tmp/ptbxl/ptbxl_database.csv"
DEFAULT_PTBXL_PREP = "/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy"
DEFAULT_SUPER5_CKPT = "/root/autodl-tmp/triple_labels/super5/best_model.pt"

PN2021_FORBIDDEN = {"ptb-xl", "ptbxl"}     # never eval against this shard

# Plan Rev 11/13: synth scope narrowed to 3 classes — HYP/CD synth disabled
# because their digital-GT validation fails 0/3 best-cell.
SUPER5_GEN_SUBSET = {"NORM", "MI", "STTC"}

# Plan Rev 11/13: hardcoded distrust regardless of Stage 0.4 sanity output
# (HYP synth fails Sokolow voltage; CD synth fails QRS broadening).
DEFAULT_TRUST_HARDCODE = {"HYP": 0.0, "CD": 0.0}

from scripts.triple_labels.label_schemes import SUPER5_TO_IDX  # noqa: E402

# Indices of in-scope generation classes (NORM/MI/STTC) in the 5-class scheme.
SUPER5_GEN_SUBSET_IDX = sorted(SUPER5_TO_IDX[c] for c in SUPER5_GEN_SUBSET)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_source_logit_anchor_epoch(
    model: nn.Module,
    teacher_model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: str,
    weight: float,
    max_batches: int = 0,
    grad_clip: float = 0.0,
) -> float:
    """One lightweight source-consistency pass against the frozen PTB-XL teacher.

    This is used after the mixed target/adv epoch to reduce PTB-XL source
    forgetting. It does not change labels; it only constrains source logits.
    """
    if weight <= 0:
        return float("nan")
    model.train()
    teacher_model.eval()
    losses: List[float] = []
    for batch_i, batch in enumerate(loader, start=1):
        signals = batch[0].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        with torch.no_grad():
            teacher_logits = teacher_model(signals)
        loss = F.mse_loss(logits, teacher_logits) * float(weight)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
        if max_batches > 0 and batch_i >= max_batches:
            break
    return float(np.mean(losses)) if losses else float("nan")


# ────────────────────────────────────────────────────────────────────────────
# Quick eval (Super5 PN2021 multi-center stratified subset)
# ────────────────────────────────────────────────────────────────────────────

def build_quick_eval_subset_super5(
    centers: List[str],
    data_dir: str,
    n_per_center: int,
    cache_path: str,
    seed: int = 0,
    exclude_record_ids: Optional[set] = None,    # Issue #39
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Stratified-by-class-presence subsample of PN2021 records, super5 labels.

    `exclude_record_ids` lets us drop the ref pool's records from the same
    center's eval split (Issue #39 patient-level isolation).
    """
    cache_key = f"{cache_path}.super5.npz"
    if os.path.exists(cache_key):
        data = np.load(cache_key, allow_pickle=True)
        out = {}
        for c in centers:
            if f"{c}__signals" in data.files:
                out[c] = {
                    "signals_tc": data[f"{c}__signals"],
                    "labels_5":   data[f"{c}__labels5"],
                }
        if len(out) == len(centers):
            if verbose:
                print(f"[quick_eval] cache hit: {cache_key}")
            return out

    import wfdb
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}
    for center in centers:
        if center.lower() in PN2021_FORBIDDEN:
            print(f"[quick_eval] SKIP forbidden shard: {center}")
            continue
        center_dir = os.path.join(data_dir, center)
        if not os.path.isdir(center_dir):
            continue
        t0 = time.time()
        hea_paths = []
        for root, _, files in os.walk(center_dir):
            for f in files:
                if f.endswith('.hea'):
                    hea_paths.append(os.path.join(root, f))
        if not hea_paths:
            continue

        # Parse SNOMED → super5 multi-hot for all
        labels_5 = []
        usable_idx = []
        for i, hea in enumerate(hea_paths):
            rec_id = os.path.basename(hea)[:-4]
            if exclude_record_ids and rec_id in exclude_record_ids:
                continue
            codes = parse_header_snomed(hea)
            labels_5.append(snomed_list_to_super5(codes))
            usable_idx.append(i)
        if not usable_idx:
            continue
        labels_5 = np.stack(labels_5).astype(np.float32)

        # Stratified pick
        if len(usable_idx) <= n_per_center:
            chosen_local = list(range(len(usable_idx)))
        else:
            per_class_quota = max(10, n_per_center // 10)
            picked = set()
            for cls_i in range(NUM_SUPER5):
                pos = np.where(labels_5[:, cls_i] == 1.0)[0]
                pos = np.array([p for p in pos if int(p) not in picked])
                if len(pos) == 0:
                    continue
                n_take = min(per_class_quota, len(pos))
                chosen = rng.choice(pos, size=n_take, replace=False)
                picked.update(int(c) for c in chosen)
            deficit = n_per_center - len(picked)
            if deficit > 0:
                rest = np.array([i for i in range(len(usable_idx)) if int(i) not in picked])
                if len(rest) > 0:
                    chosen = rng.choice(rest, size=min(deficit, len(rest)), replace=False)
                    picked.update(int(c) for c in chosen)
            chosen_local = sorted(picked)

        signals = []
        kept_labels = []
        for ci in chosen_local:
            i = usable_idx[ci]
            try:
                rec = wfdb.rdrecord(hea_paths[i][:-4])
            except Exception:
                continue
            sig = rec.p_signal
            if sig is None or sig.shape[1] < 12:
                continue
            sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, 'sig_name', None) else None
            proc = unified_preprocess_to_1000(
                sig.astype(np.float32), fs=rec.fs, source_leads=sig_names,
                target_fs=100, target_len=1000,
                apply_filter=True, apply_zscore=True,
            )
            if proc is None:
                continue
            signals.append(proc)
            kept_labels.append(labels_5[ci])
        if not signals:
            continue
        out[center] = {
            "signals_tc": np.stack(signals).astype(np.float32),
            "labels_5":   np.stack(kept_labels).astype(np.float32),
        }
        if verbose:
            print(f"  [quick_eval] {center}: {out[center]['signals_tc'].shape[0]} records "
                  f"({time.time() - t0:.0f}s)"
                  + (f"; excluded {len(hea_paths) - len(usable_idx)} ref ids" if exclude_record_ids else ""))

    os.makedirs(os.path.dirname(cache_key) or ".", exist_ok=True)
    dump = {}
    for c, d in out.items():
        dump[f"{c}__signals"] = d["signals_tc"]
        dump[f"{c}__labels5"] = d["labels_5"]
    np.savez_compressed(cache_key, **dump)
    if verbose:
        print(f"[quick_eval] cached → {cache_key}")
    return out


@torch.no_grad()
def quick_eval_super5(
    model: nn.Module, quick_subset: Dict[str, Dict], device: str,
    crop_len: int = TIERM_INPUT_LENGTH, min_pos: int = 10,
) -> Dict[str, Any]:
    model.eval()
    result: Dict[str, Any] = {"per_center": {}}
    macro_aurocs, macro_auprcs = [], []
    for center, data in quick_subset.items():
        signals_tc = data["signals_tc"]   # (N, 1000, 12)
        labels_5 = data["labels_5"]       # (N, 5)
        N = signals_tc.shape[0]

        x_ct = np.transpose(signals_tc, (0, 2, 1)).astype(np.float32)  # (N, 12, 1000)
        start = (x_ct.shape[-1] - crop_len) // 2
        x_ct = x_ct[..., start:start + crop_len]
        x_t = torch.from_numpy(x_ct).to(device)

        all_logits = []
        for i in range(0, N, 128):
            lg = model(x_t[i:i + 128])
            all_logits.append(lg.cpu().numpy())
        logits = np.concatenate(all_logits)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

        m = compute_macro_auroc_auprc(labels_5, probs, CLASS_NAMES_SUPER5,
                                      min_pos=min_pos)
        result["per_center"][center] = {
            "n":           N,
            "macro_auroc": round(m["macro_auroc"], 4) if not math.isnan(m["macro_auroc"]) else None,
            "macro_auprc": round(m["macro_auprc"], 4) if not math.isnan(m["macro_auprc"]) else None,
            "n_classes_used": m["n_classes_used"],
            "per_class":   m["per_class"],
        }
        if not math.isnan(m["macro_auroc"]):
            macro_aurocs.append(m["macro_auroc"])
            macro_auprcs.append(m["macro_auprc"])

    result["avg_macro_auroc"] = round(float(np.mean(macro_aurocs)), 4) if macro_aurocs else float('nan')
    result["avg_macro_auprc"] = round(float(np.mean(macro_auprcs)), 4) if macro_auprcs else float('nan')
    return result


# ────────────────────────────────────────────────────────────────────────────
# PGD-on-frozen-synth-pool epoch step
# ────────────────────────────────────────────────────────────────────────────

def stratified_sample_synth(
    labels_npz: np.ndarray, K_anchor: int, num_classes: int, rng: np.random.Generator,
) -> np.ndarray:
    """LEGACY (Plan Rev 8): with-replacement stratified sample over all classes.

    Kept for backward compat / smoke. Plan Rev 13.2 onward uses StratifiedPoolWalker
    instead — that gives no-revisit-per-epoch + restricts to SUPER5_GEN_SUBSET.
    """
    cls_ids = labels_npz.argmax(axis=1)
    per_cls = max(1, K_anchor // num_classes)
    picked = []
    for c in range(num_classes):
        pool = np.where(cls_ids == c)[0]
        if len(pool) == 0:
            continue
        n_take = min(per_cls, len(pool))
        picked.extend(rng.choice(pool, size=n_take, replace=False).tolist())
    picked = list(set(picked))
    deficit = K_anchor - len(picked)
    if deficit > 0:
        rest = np.array([i for i in range(len(labels_npz)) if i not in set(picked)])
        if len(rest) > 0:
            picked.extend(rng.choice(rest, size=min(deficit, len(rest)), replace=False).tolist())
    return np.array(sorted(set(picked))[:K_anchor])


def parse_source_weight_map(raw: Optional[str]) -> Dict[str, float]:
    """Parse `source=weight,source2=weight2` into a dict."""
    if not raw:
        return {}
    out: Dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad source weight item {item!r}; expected source=weight")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def parse_class_source_weight_map(raw: Optional[str]) -> Dict[str, Dict[str, float]]:
    """Parse `CLASS:source=weight,CLASS2:source2=weight2` overrides."""
    if not raw:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item or "=" not in item:
            raise ValueError(
                f"Bad class-source weight item {item!r}; expected CLASS:source=weight"
            )
        cls, rest = item.split(":", 1)
        source, value = rest.split("=", 1)
        out.setdefault(cls.strip(), {})[source.strip()] = float(value)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Plan Rev 13.2: StratifiedPoolWalker — no-revisit-per-epoch over synth pool
# ─────────────────────────────────────────────────────────────────────────


class StratifiedPoolWalker:
    """Walks a frozen synth pool stratified by class with cursor state across
    epochs. Plan Rev 9 / Rev 13: each epoch we draw K_anchor anchors stratified
    by NORM/MI/STTC. Cursors persist across epochs; when a class's pool is
    exhausted we reshuffle that class and reset the cursor. This breaks the
    pool-starvation pattern of pilot iter1-3 (with-replacement on a 100-pool
    meant the victim could memorize all anchors after ~5 epochs).
    """

    def __init__(
        self,
        labels_one_hot: np.ndarray,
        classes_in_scope: List[str],
        class_to_idx: Dict[str, int],
        seed: int = 42,
        source_labels: Optional[np.ndarray] = None,
        source_sampling_strategy: str = "class_balanced",
        source_weights: Optional[Dict[str, float]] = None,
        source_class_weights: Optional[Dict[str, Dict[str, float]]] = None,
        source_floor_per_class: int = 0,
    ):
        self.classes = list(classes_in_scope)
        self.class_to_idx = class_to_idx
        self.cls_pools: Dict[str, np.ndarray] = {}
        self.cursors: Dict[str, int] = {}
        self.epochs_completed: Dict[str, int] = {c: 0 for c in self.classes}
        self.rng = np.random.default_rng(seed)
        self.source_sampling_strategy = source_sampling_strategy
        self.source_weights = dict(source_weights or {})
        self.source_class_weights = dict(source_class_weights or {})
        self.source_floor_per_class = max(0, int(source_floor_per_class))
        self.source_labels = None
        self.source_names: List[str] = []
        self.source_cls_pools: Dict[Tuple[str, str], np.ndarray] = {}
        self.source_cursors: Dict[Tuple[str, str], int] = {}
        self.source_epochs_completed: Dict[Tuple[str, str], int] = {}
        self.last_source_counts: Dict[str, int] = {}
        self.last_class_source_counts: Dict[str, Dict[str, int]] = {}
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

    def class_sizes(self) -> Dict[str, int]:
        return {c: int(len(self.cls_pools[c])) for c in self.classes}

    def source_class_sizes(self) -> Dict[str, Dict[str, int]]:
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
        pools: Dict[Any, np.ndarray],
        cursors: Dict[Any, int],
        epochs_completed: Dict[Any, int],
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

    def _source_quotas_for_class(self, cls: str, k: int) -> Dict[str, int]:
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

    def sample(self, k_per_class: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Draw k_per_class[c] indices for each class without revisit per epoch."""
        out: Dict[str, np.ndarray] = {}
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
                        key, q, self.source_cls_pools,
                        self.source_cursors, self.source_epochs_completed,
                    )
                    if drawn.size > 0:
                        pieces.append(drawn)
                    self.last_class_source_counts[c][source] = int(drawn.size)
                    self.last_source_counts[source] = (
                        self.last_source_counts.get(source, 0) + int(drawn.size)
                    )
                out[c] = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
                if out[c].size > 1:
                    self.rng.shuffle(out[c])
                continue
            out[c] = self._draw_from_pool(
                c, k, self.cls_pools, self.cursors, self.epochs_completed,
            )
        return out


class SameLabelLatentIndex:
    """Nearest-neighbor candidate index for constrained latent-hull attacks.

    Modes:
      primary: same argmax class as the anchor.
      exact: exact same multi-hot super5 vector as the anchor.
      compatible: NORM-only anchors mix only with NORM-only anchors; abnormal
        anchors mix with any non-NORM abnormal anchor. This expands the search
        space while avoiding the main super5 contradiction, NORM vs abnormal.
    """

    def __init__(
        self,
        latents: np.ndarray,
        labels_one_hot: np.ndarray,
        label_mode: str = "primary",
        seed: int = 42,
        include_self: bool = False,
    ):
        if label_mode not in {"primary", "exact", "compatible"}:
            raise ValueError(
                f"label_mode must be primary|exact|compatible, got {label_mode!r}"
            )
        self.latents = latents.astype(np.float32, copy=False)
        self.labels = labels_one_hot.astype(np.float32, copy=False)
        self.label_mode = label_mode
        self.include_self = bool(include_self)
        self.rng = np.random.default_rng(seed)
        self.last_candidate_indices: Optional[np.ndarray] = None
        if label_mode == "primary":
            keys = [int(i) for i in self.labels.argmax(axis=1)]
        elif label_mode == "exact":
            keys = [tuple(int(v) for v in row) for row in (self.labels > 0.5).astype(np.int8)]
        else:
            norm_idx = SUPER5_TO_IDX["NORM"]
            abnormal = np.delete(np.arange(self.labels.shape[1]), norm_idx)
            is_norm_only = (self.labels[:, norm_idx] > 0.5) & (
                self.labels[:, abnormal].sum(axis=1) == 0
            )
            has_abnormal = self.labels[:, abnormal].sum(axis=1) > 0
            keys = ["NORM_ONLY" if n else "ABNORMAL" if a else "OTHER"
                    for n, a in zip(is_norm_only, has_abnormal)]
        self.keys = keys
        self.pools: Dict[Any, np.ndarray] = {}
        for i, key in enumerate(keys):
            self.pools.setdefault(key, []).append(i)
        self.pools = {k: np.asarray(v, dtype=np.int64) for k, v in self.pools.items()}

    def class_sizes(self) -> Dict[str, int]:
        return {str(k): int(len(v)) for k, v in self.pools.items()}

    def candidates_for(self, anchor_indices: np.ndarray, M: int) -> np.ndarray:
        """Return (B,M,4,128) nearest same-label candidates.

        By default this excludes the anchor itself when possible, matching the
        historical Latent-Hull setup. For no-lambda convex-hull ablations,
        include_self=True prepends the anchor as candidate 0 so
        z_adv=sum_i softmax(a_i) z_i can still stay near z0 if that is optimal.
        """
        out = np.empty((len(anchor_indices), M, 4, 128), dtype=np.float32)
        out_indices = np.empty((len(anchor_indices), M), dtype=np.int64)
        flat_latents = self.latents.reshape(self.latents.shape[0], -1)
        for row_i, anchor_idx in enumerate(anchor_indices):
            anchor_idx = int(anchor_idx)
            key = self.keys[anchor_idx]
            pool = self.pools.get(key, np.asarray([anchor_idx], dtype=np.int64))
            pool = pool[pool != anchor_idx]
            if len(pool) == 0:
                pool = np.asarray([anchor_idx], dtype=np.int64)
            diff = flat_latents[pool] - flat_latents[anchor_idx]
            dist2 = np.einsum("ij,ij->i", diff, diff)
            order = np.argsort(dist2)
            if self.include_self:
                neighbor_budget = max(M - 1, 0)
                chosen = np.concatenate(
                    [
                        np.asarray([anchor_idx], dtype=np.int64),
                        pool[order[:neighbor_budget]],
                    ]
                )
            else:
                chosen = pool[order[:M]]
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
) -> np.ndarray:
    """Build label-smoothing targets from latent-hull candidate weights.

    Anchor positives stay near hard positives. Classes not present in the anchor
    can receive weak fractional targets from the weighted compatible candidates,
    capped to keep secondary evidence uncertain rather than hard-positive.
    """
    anchor = anchor_labels.astype(np.float32, copy=False)
    cand = candidate_labels.astype(np.float32, copy=False)
    w = weights.astype(np.float32, copy=False)
    q = (w[:, :, None] * cand).sum(axis=1)

    out = np.full(anchor.shape, float(negative_floor), dtype=np.float32)
    pos_mask = anchor > 0.5
    out[pos_mask] = float(positive_value)
    new_soft = np.clip(float(lambda_y) * q, float(negative_floor), float(new_class_cap))
    out[~pos_mask] = new_soft[~pos_mask]

    # Preserve the super5 NORM policy: abnormal anchors should not acquire a
    # fractional NORM target from any accidental compatible fallback.
    norm_idx = SUPER5_TO_IDX["NORM"]
    abnormal_idx = [i for i in range(anchor.shape[1]) if i != norm_idx]
    abnormal_anchor = anchor[:, abnormal_idx].sum(axis=1) > 0.5
    out[abnormal_anchor, norm_idx] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Plan Rev 13 Stage 0.4: synth pool sanity → class_trust map
# ─────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_synth_sanity_auroc(
    synth_pool_signals: np.ndarray,        # (N, 12, 1000) z-scored
    labels_one_hot: np.ndarray,            # (N, 5)
    victim: nn.Module,
    device: str,
    crop_len: int = TIERM_INPUT_LENGTH,
    batch_size: int = 128,
) -> Dict[str, Optional[float]]:
    """Forward synth pool through Super5 victim, compute per-class AUROC.

    A class's synth is "trustworthy" iff victim AUROC > 0.55 (separable from
    the other 4 classes' decision regions).
    """
    N = synth_pool_signals.shape[0]
    crops = []
    for i in range(N):
        sig_tc = synth_pool_signals[i].T  # (1000, 12)
        start = (sig_tc.shape[0] - crop_len) // 2
        crop = sig_tc[start:start + crop_len, :]
        crops.append(crop.T.astype(np.float32))
    crops_arr = np.stack(crops, axis=0)
    victim.eval() if hasattr(victim, 'eval') else None
    all_logits = []
    for i in range(0, N, batch_size):
        x = torch.from_numpy(crops_arr[i:i + batch_size]).float().to(device)
        if hasattr(victim, 'compute_logits_from_ecg'):
            lg = victim.compute_logits_from_ecg(x)
        else:
            lg = victim(x)
        all_logits.append(lg.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    per_class: Dict[str, Optional[float]] = {}
    for j, c in enumerate(CLASS_NAMES_SUPER5):
        col = labels_one_hot[:, j]
        n_pos = int((col == 1.0).sum())
        if n_pos == 0 or n_pos == N:
            per_class[c] = None
            continue
        try:
            per_class[c] = float(roc_auc_score(col, probs[:, j]))
        except Exception:
            per_class[c] = None
    return per_class


def auroc_to_trust(auroc: Optional[float]) -> float:
    """Plan Rev 13: AUROC > 0.7 → 1.0; > 0.55 → 0.5; else 0.0."""
    if auroc is None:
        return 0.0
    if auroc > 0.7:
        return 1.0
    if auroc > 0.55:
        return 0.5
    return 0.0


def derive_class_trust(per_class_auroc: Dict[str, Optional[float]]) -> Dict[str, float]:
    out = {c: auroc_to_trust(per_class_auroc.get(c)) for c in CLASS_NAMES_SUPER5}
    out.update(DEFAULT_TRUST_HARDCODE)
    return out


def run_pgd_on_synth_pool(
    pgd_gen: PGDAdvDiffGenerator,
    synth_latents: np.ndarray,    # (N, 4, 128)
    synth_labels: np.ndarray,     # (N, C) one-hot
    K_anchor: int,
    pgd_batch: int,
    rng: np.random.Generator,
    device: str,
    picked_indices: Optional[np.ndarray] = None,
    attack_mode: str = "pgd",
    latent_hull_index: Optional[SameLabelLatentIndex] = None,
    hull_M: int = 10,
    hull_mix_label_mode: str = "anchor",
    hull_label_lambda_y: float = 0.5,
    hull_label_positive: float = 0.95,
    hull_label_negative_floor: float = 0.0,
    hull_label_new_class_cap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Sample K anchors stratified by class, run PGD in batches of pgd_batch.

    If `picked_indices` is given (e.g. from StratifiedPoolWalker), use it and
    skip the with-replacement stratified sampler. This is the Plan Rev 13.2
    path; the legacy path (no picked_indices) is kept for backward compat
    with smoke / Plan Rev 8.

    Returns:
      adv_signals_ct:   (K, 12, 1000)  float32
      anchor_signals_ct:(K, 12, 1000)  float32  (clean reference for sem-gate)
      labels_one_hot:   (K, C)         float32
      delta_stats:      mean / max L2 norm plus optional latent-hull weight stats
    """
    if picked_indices is None:
        num_classes = synth_labels.shape[1]
        pick = stratified_sample_synth(synth_labels, K_anchor, num_classes, rng)
    else:
        pick = picked_indices

    if len(pick) == 0:
        return (np.empty((0, 12, 1000), dtype=np.float32),
                np.empty((0, 12, 1000), dtype=np.float32),
                np.empty((0, synth_labels.shape[1]), dtype=np.float32),
                {"mean_delta_norm": float('nan'), "max_delta_norm": float('nan')})

    y_anchor_np = synth_labels[pick].astype(np.float32, copy=False)
    z_anchors = torch.from_numpy(synth_latents[pick]).float()        # (K, 4, 128)
    y_anchors = torch.from_numpy(y_anchor_np).float()                # (K, C)

    adv_chunks, anc_chunks, label_chunks, delta_norms = [], [], [], []
    hull_entropies, hull_top1 = [], []
    for i in range(0, z_anchors.shape[0], pgd_batch):
        z_b = z_anchors[i:i + pgd_batch].to(device)
        y_b = y_anchors[i:i + pgd_batch].to(device)
        if attack_mode == "latent_hull":
            if latent_hull_index is None:
                raise ValueError("latent_hull_index is required when attack_mode=latent_hull")
            batch_pick = pick[i:i + pgd_batch]
            cand_np = latent_hull_index.candidates_for(batch_pick, hull_M)
            cand_b = torch.from_numpy(cand_np).float().to(device)
            x_adv, delta = pgd_gen.attack_from_latent(
                z_b, y_b, candidate_latents=cand_b
            )
            if hull_mix_label_mode == "anchor":
                label_chunks.append(y_anchor_np[i:i + pgd_batch])
            elif hull_mix_label_mode == "anchor_soft":
                cand_idx = getattr(latent_hull_index, "last_candidate_indices", None)
                weights_t = getattr(pgd_gen, "last_weights", None)
                if cand_idx is None or weights_t is None:
                    raise RuntimeError(
                        "latent-hull soft labels require candidate indices and weights"
                    )
                weights_np = weights_t.numpy().astype(np.float32, copy=False)
                cand_labels = synth_labels[cand_idx]
                label_chunks.append(build_anchor_preserving_soft_labels(
                    y_anchor_np[i:i + pgd_batch],
                    cand_labels,
                    weights_np,
                    lambda_y=hull_label_lambda_y,
                    positive_value=hull_label_positive,
                    negative_floor=hull_label_negative_floor,
                    new_class_cap=hull_label_new_class_cap,
                ))
            else:
                raise ValueError(
                    "hull_mix_label_mode must be anchor|anchor_soft, "
                    f"got {hull_mix_label_mode!r}"
                )
            hull_info = getattr(pgd_gen, "last_info", {})
            if "hull_weight_entropy_mean" in hull_info:
                hull_entropies.append(float(hull_info["hull_weight_entropy_mean"]))
            if "hull_weight_top1_mean" in hull_info:
                hull_top1.append(float(hull_info["hull_weight_top1_mean"]))
        else:
            # Random init delta — Plan Issue #41 clean-anchor restart each epoch
            delta_init = torch.randn_like(z_b) * pgd_gen.delta_init_scale
            x_adv, delta = pgd_gen.attack_from_latent(z_b, y_b, delta_init=delta_init)
            label_chunks.append(y_anchor_np[i:i + pgd_batch])
        adv_chunks.append(x_adv.detach().cpu().numpy().astype(np.float32))
        # Clean anchor reference (z_b alone, no delta)
        with torch.no_grad():
            anc_x = pgd_gen._decode_to_ptbxl_1000(z_b)
        anc_chunks.append(anc_x.detach().cpu().numpy().astype(np.float32))
        delta_norms.extend(delta.detach().flatten(1).norm(dim=1).cpu().tolist())

    adv_signals = np.concatenate(adv_chunks, axis=0)                  # (K, 12, 1000)
    anc_signals = np.concatenate(anc_chunks, axis=0)
    stats = {
        "mean_delta_norm": float(np.mean(delta_norms)),
        "max_delta_norm":  float(np.max(delta_norms)),
    }
    if hull_entropies:
        stats.update({
            "hull_weight_entropy_mean": float(np.mean(hull_entropies)),
            "hull_weight_top1_mean": float(np.mean(hull_top1)) if hull_top1 else float('nan'),
        })
    out_labels = np.concatenate(label_chunks, axis=0).astype(np.float32)
    return adv_signals, anc_signals, out_labels, stats


# ────────────────────────────────────────────────────────────────────────────
# Adv buffer push with masked-BCE label semantics
# ────────────────────────────────────────────────────────────────────────────

def push_adv_to_buffer(
    buffer: QualityAwareBuffer,
    adv_signals_ct: np.ndarray,    # (K, 12, 1000)
    target_one_hot: np.ndarray,    # (K, C)
    victim_logits: np.ndarray,     # (K, C)
    crop_len: int,
    class_trust: Optional[Dict[str, float]] = None,
    boundary_prob_min: float = 0.0,
    boundary_prob_max: float = 1.0,
    teacher_probs: Optional[np.ndarray] = None,
    label_mode: str = "hard",
    teacher_mix: float = 0.7,
    soft_target_floor: float = 0.0,
) -> Dict[str, int]:
    """Push gates-passed adv signals into the buffer with -1 sentinel labels.

    Label scheme: target_only (Plan Issue #28 — CheXpert U-Ignore + SPML).
    Non-target dims set to -1.0 → masked out by masked_bce_with_logits.

    Plan Rev 13.2: per-sample buffer push score is multiplied by class_trust
    (HYP/CD trust=0 → effectively dropped even if Stage 1 produced any). With
    the 3-class generation scope (NORM/MI/STTC) HYP/CD synth is already absent
    from the synth pool; this is a defense-in-depth check.
    """
    n_pushed = 0
    n_dropped_by_trust = 0
    n_dropped_by_boundary = 0
    for i in range(adv_signals_ct.shape[0]):
        sig_250 = _center_crop_ct(adv_signals_ct[i], crop_len)         # (12, 250)
        target_idx = int(target_one_hot[i].argmax())
        target_class = CLASS_NAMES_SUPER5[target_idx] if target_idx < len(CLASS_NAMES_SUPER5) else None
        trust = float(class_trust.get(target_class, 1.0)) if class_trust else 1.0
        if trust <= 0.0:
            n_dropped_by_trust += 1
            continue
        prob_t = float(1.0 / (1.0 + math.exp(-min(50.0, max(-50.0, victim_logits[i, target_idx])))))
        if prob_t < boundary_prob_min or prob_t > boundary_prob_max:
            n_dropped_by_boundary += 1
            continue
        if label_mode == "hard":
            lbl = torch.full((target_one_hot.shape[1],), -1.0)
            lbl[target_idx] = 1.0
        elif label_mode == "multi_hot_hard":
            lbl = torch.from_numpy((target_one_hot[i] > 0.5).astype(np.float32))
        elif label_mode == "latent_soft":
            lbl = torch.from_numpy(np.clip(target_one_hot[i].astype(np.float32), 0.0, 1.0))
        else:
            if teacher_probs is None:
                raise ValueError(f"teacher_probs required for label_mode={label_mode}")
            soft = torch.from_numpy(
                np.clip(teacher_probs[i].astype(np.float32), 0.0, 1.0)
            )
            if label_mode == "teacher_soft":
                lbl = soft
            elif label_mode == "mixed_soft":
                hard_full = torch.zeros((target_one_hot.shape[1],), dtype=torch.float32)
                hard_full[target_idx] = 1.0
                mix = max(0.0, min(1.0, float(teacher_mix)))
                lbl = mix * soft + (1.0 - mix) * hard_full
            elif label_mode == "latent_mixed_teacher":
                latent_soft = torch.from_numpy(
                    np.clip(target_one_hot[i].astype(np.float32), 0.0, 1.0)
                )
                mix = max(0.0, min(1.0, float(teacher_mix)))
                lbl = mix * soft + (1.0 - mix) * latent_soft
            else:
                raise ValueError(f"unsupported adv label_mode={label_mode}")
            if soft_target_floor > 0.0:
                lbl[target_idx] = torch.clamp(lbl[target_idx], min=float(soft_target_floor))
        score = (1.0 - 2.0 * abs(prob_t - 0.5)) * trust                # ∈ [0, trust]
        buffer.add_one(
            torch.from_numpy(np.ascontiguousarray(sig_250)).float(),
            lbl,
            score,
        )
        n_pushed += 1
    return {
        "n_pushed": n_pushed,
        "n_dropped_by_trust": n_dropped_by_trust,
        "n_dropped_by_boundary": n_dropped_by_boundary,
        "label_mode": label_mode,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--center_name", required=True, help="cpsc_2018_extra | ningbo")
    p.add_argument("--ref_meta_json",
                   help="path to {tag}_k200.meta.json (for record_id exclusion in eval)")
    p.add_argument("--synth_npz", required=True,
                   help="Stage 1 latent npz: {latents (N,4,128), labels (N,5)}")
    p.add_argument("--init_ckpt", default=DEFAULT_SUPER5_CKPT)
    p.add_argument("--model_name", default="efficientnet1dv2",
                   choices=available_model_names())
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_real_npz", default="",
                   help="Optional selected target-center real ECG npz with signals (N,1000,12) and labels.")

    # PN2021 quick eval
    p.add_argument("--data_dir", default=DEFAULT_PN2021_DIR)
    p.add_argument("--quick_eval_centers", nargs='+',
                   default=["chapman_shaoxing", "cpsc_2018_extra", "georgia",
                            "ningbo"])
    p.add_argument("--quick_eval_n_per_center", type=int, default=1000)

    # PTBXL paths
    p.add_argument("--ptbxl_raw", default=DEFAULT_PTBXL_RAW)
    p.add_argument("--ptbxl_csv", default=DEFAULT_PTBXL_CSV)
    p.add_argument("--ptbxl_prep", default=DEFAULT_PTBXL_PREP)

    # PGD (Plan Rev 13: K_pgd=10 + ε=2.0 + K_anchor=300)
    p.add_argument("--pgd_eps", type=float, default=2.0)
    p.add_argument("--pgd_K", type=int, default=10)
    p.add_argument("--pgd_batch", type=int, default=32)        # Issue #45
    p.add_argument("--K_anchor", type=int, default=300)
    p.add_argument("--pgd_alpha", type=float, default=None)
    p.add_argument("--delta_init_scale", type=float, default=0.1)
    p.add_argument("--attack_mode", choices=["pgd", "latent_hull"], default="pgd",
                   help="pgd = free z0+delta PGD; latent_hull = same-label convex hull")
    p.add_argument("--hull_M", type=int, default=10,
                   help="Number of same-label candidate latents for latent_hull")
    p.add_argument("--hull_lambda", type=float, default=0.25)
    p.add_argument("--hull_steps", type=int, default=5)
    p.add_argument("--hull_lr", type=float, default=0.3)
    p.add_argument(
        "--hull_weight_mode",
        choices=["optimized", "one_hot", "uniform", "dirichlet"],
        default="optimized",
        help="Latent-Hull coefficient policy: optimized=C3 main, or fixed C0/C1/C2 ablations.",
    )
    p.add_argument("--hull_dirichlet_alpha", type=float, default=1.0)
    p.add_argument(
        "--hull_label_mode",
        choices=["primary", "exact", "compatible"],
        default="primary",
    )
    p.add_argument(
        "--hull_mix_label_mode",
        choices=["anchor", "anchor_soft"],
        default="anchor",
        help="anchor keeps the anchor multi-hot label; anchor_soft builds "
             "anchor-preserving fractional labels from latent-hull weights.",
    )
    p.add_argument("--hull_label_lambda_y", type=float, default=0.5)
    p.add_argument("--hull_label_positive", type=float, default=0.95)
    p.add_argument("--hull_label_negative_floor", type=float, default=0.0)
    p.add_argument("--hull_label_new_class_cap", type=float, default=0.5)
    p.add_argument(
        "--hull_include_anchor",
        action="store_true",
        help="Include the anchor latent itself as candidate 0 in same-label hull. "
             "Useful for no-lambda convex-hull ablations with --hull_lambda 1.0.",
    )
    p.add_argument("--source_sampling_strategy",
                   choices=["class_balanced", "source_weighted"],
                   default="class_balanced",
                   help="class_balanced ignores source metadata; source_weighted "
                        "allocates each class quota by source weights when the "
                        "latent pool has source_ids/source_names.")
    p.add_argument("--source_weights", default=None,
                   help="Comma map for source_weighted, e.g. real_anchor=1.0,prompt_token=0.35")
    p.add_argument("--source_class_weights", default=None,
                   help="Comma overrides, e.g. MI:prompt_token=1.0,STTC:prompt_token=0.5")
    p.add_argument("--source_floor_per_class", type=int, default=0,
                   help="Minimum anchors per positive-weight source within each class quota.")
    p.add_argument("--classes_in_scope", nargs="+", default=sorted(SUPER5_GEN_SUBSET),
                   help="Super5 classes sampled as adversarial anchors. Default keeps historical NORM/MI/STTC.")
    p.add_argument("--allow_hyp_cd_trust", action="store_true",
                   help="Do not hard-force HYP/CD class_trust to zero.")
    p.add_argument("--boundary_prob_min", type=float, default=0.0,
                   help="Only push adv samples whose target sigmoid probability is >= this value.")
    p.add_argument("--boundary_prob_max", type=float, default=1.0,
                   help="Only push adv samples whose target sigmoid probability is <= this value.")
    p.add_argument("--adv_label_mode",
                   choices=[
                       "hard", "multi_hot_hard", "latent_soft",
                       "mixed_soft", "teacher_soft", "latent_mixed_teacher",
                   ],
                   default="hard",
                   help="Label policy for generated adversarial ECGs: hard keeps historical target-only labels; "
                        "multi_hot_hard keeps the full anchor multi-hot label; "
                        "latent_soft uses latent-hull fractional labels; "
                        "mixed_soft blends frozen-teacher probabilities with a hard target; "
                        "latent_mixed_teacher blends frozen-teacher probabilities with latent_soft labels; "
                        "teacher_soft uses the frozen initial teacher probabilities directly.")
    p.add_argument("--adv_teacher_mix", type=float, default=0.7,
                   help="For --adv_label_mode mixed_soft, weight on frozen-teacher probabilities.")
    p.add_argument("--adv_soft_target_floor", type=float, default=0.0,
                   help="For soft adv labels, clamp the intended target class label to at least this value.")

    # Mix loader (Plan Rev 13.2: real-dominated mix, adv_w=0.5 vs Wang 2023 0.7 reverse)
    p.add_argument("--ptbxl_weight", type=float, default=1.0)
    p.add_argument("--target_real_weight", type=float, default=0.0,
                   help="Sampling weight for --target_real_npz supervised stream.")
    p.add_argument("--roundtrip_weight", type=float, default=0.5)
    p.add_argument("--adv_weight", type=float, default=0.5)
    p.add_argument("--roundtrip_anchor_n", type=int, default=1500)
    p.add_argument("--qab_size", type=int, default=2048)
    p.add_argument(
        "--source_logit_anchor_weight",
        type=float,
        default=0.0,
        help=(
            "Optional source-consistency distillation weight. After each mixed "
            "training epoch, run a PTB-XL source pass and penalize MSE between "
            "current logits and the frozen initial source model logits."
        ),
    )
    p.add_argument(
        "--source_logit_anchor_batches",
        type=int,
        default=0,
        help="Max PTB-XL source batches per source-logit anchor pass; 0 uses the full source loader.",
    )

    # Class trust (Plan Rev 13 H4 gate)
    p.add_argument("--class_trust", default=None,
                   help="Path to class_trust.json (Stage 0.4 sanity output). Required unless --build_class_trust.")
    p.add_argument("--build_class_trust", action="store_true",
                   help="Run sanity AUROC on synth pool, write class_trust.json next to synth pool, then EXIT.")

    # Optim (Plan Rev 13.1: 100 ep + early-stop patience=20 on val_macro_auroc)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20,
                   help="Early stop after this many quick_eval rounds without improvement")
    p.add_argument("--es_metric",
                   choices=[
                       "val_macro_auroc",
                       "val_macro_auprc",
                       "target_macro_auroc",
                       "target_macro_auprc",
                   ],
                   default="val_macro_auroc")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=12)      # Issue #46
    p.add_argument("--anchor_lambda", type=float, default=0.05)
    p.add_argument("--ewa_decay", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=3)
    p.add_argument("--rescore_interval", type=int, default=3)

    # Gates (Plan Issue #38)
    p.add_argument("--asr_consec_low_max", type=int, default=3,
                   help="Halt with RuntimeError after this many consecutive low-ASR epochs")
    p.add_argument("--asr_low_threshold", type=float, default=0.30)
    p.add_argument("--einthoven_p95_max", type=float, default=0.5)
    p.add_argument(
        "--disable_quality_gate",
        action="store_true",
        help="Do not skip buffer push when semantic/quality gate fails. "
             "Still compute and log gate metrics for ablation.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--crop_len", type=int, default=TIERM_INPUT_LENGTH)
    p.add_argument("--smoke", action="store_true",
                   help="Run only --n_epochs but with tiny subsets for sanity")
    return p.parse_args()


def load_synth_pool(synth_npz_path: str) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """Load Stage 1 frozen latent pool. Accepts either:

      - {basename}.npz       (signals + labels): auto-finds {basename}.latent.npz
      - {basename}.latent.npz (latents + labels): used directly

    Returns (latents (N,4,128), labels (N,C), center_name, source_meta).
    """
    p = Path(synth_npz_path)
    if p.name.endswith(".latent.npz"):
        latent_path = p
    else:
        # Convert e.g. extra_pool300.npz → extra_pool300.latent.npz
        latent_path = p.with_name(p.stem + ".latent.npz")
        if not latent_path.exists():
            raise SystemExit(
                f"latent pool .npz not found: tried {latent_path}. "
                f"Re-run generate_center_synth.py with --save_latent."
            )
    d = np.load(str(latent_path), allow_pickle=True)
    latents = d["latents"]
    labels = d["labels"]
    center = str(d["center_name"]) if "center_name" in d.files else "?"
    assert latents.ndim == 3 and latents.shape[1:] == (4, 128), \
        f"bad synth latent shape: {latents.shape}"
    assert labels.shape[0] == latents.shape[0]

    source_ids = None
    source_names = None
    source_labels = None
    if "source_ids" in d.files and "source_names" in d.files:
        source_ids = d["source_ids"].astype(np.int64)
        source_names = [str(x) for x in d["source_names"].tolist()]
        if source_ids.shape[0] != latents.shape[0]:
            raise ValueError("source_ids length does not match latents")
        source_labels = np.asarray([
            source_names[int(i)] if 0 <= int(i) < len(source_names) else f"source_{int(i)}"
            for i in source_ids
        ])
    else:
        source_ids = np.zeros((latents.shape[0],), dtype=np.int64)
        source_names = ["unknown"]
        source_labels = np.asarray(["unknown"] * latents.shape[0])

    source_meta = {
        "source_ids": source_ids,
        "source_names": source_names,
        "source_labels": source_labels,
        "has_source_metadata": "source_ids" in d.files and "source_names" in d.files,
    }
    return latents.astype(np.float32), labels.astype(np.float32), center, source_meta


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("Synth-anchored online AT (Super5) — Plan Rev 13.2")
    print("=" * 72)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 72)

    # ── Build Super5 victim early (needed for both sanity + training) ──────
    print("[setup] Loading ECGTwin (encoder + decoder, no text model)...")
    ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=False)
    print(f"[setup] Loading Super5 victim from {args.init_ckpt}")
    victim = EfficientNetVictimTierM(
        weight_path=args.init_ckpt,
        device=args.device,
        ecgtwin_wrapper=ecgtwin,
        num_classes=NUM_SUPER5,
        crop_len=args.crop_len,
        model_name=args.model_name,
    )

    # ── Plan Rev 13 Stage 0.4 sanity-build mode (early exit) ────────────────
    if args.build_class_trust:
        # Need synth_pool's signals (decoded) — load from .npz (signals key) or
        # re-derive from latents via VAE decoder.
        signal_npz = args.synth_npz
        if signal_npz.endswith(".latent.npz"):
            signal_npz = signal_npz.replace(".latent.npz", ".npz")
        if not os.path.exists(signal_npz):
            raise SystemExit(f"signal pool .npz not found at {signal_npz}; "
                              "regen synth with --save_latent (this stores both signals + latents).")
        sig_data = np.load(signal_npz, allow_pickle=True)
        if "signals" not in sig_data.files:
            raise SystemExit(f"{signal_npz} missing 'signals' key — re-run "
                              "generate_center_synth.py with the current head.")
        signals = sig_data["signals"].astype(np.float32)
        labels_oh = sig_data["labels"].astype(np.float32)
        print(f"[sanity] forwarding {signals.shape} synth signals through "
              f"victim for per-class AUROC ...")
        per_class = compute_synth_sanity_auroc(
            signals, labels_oh, victim, args.device,
            crop_len=args.crop_len, batch_size=128,
        )
        trust = derive_class_trust(per_class)
        ct_path = args.class_trust or str(
            Path(args.synth_npz).with_suffix("").with_suffix(".class_trust.json"))
        os.makedirs(os.path.dirname(ct_path) or ".", exist_ok=True)
        with open(ct_path, "w") as f:
            json.dump({
                "tag": args.center_name,
                "synth_pool": args.synth_npz,
                "per_class_auroc": per_class,
                "class_trust": trust,
                "policy": "AUROC>0.7→1.0; >0.55→0.5; else 0.0; HYP/CD hardcoded 0.0",
            }, f, indent=2)
        print(f"[sanity] per-class AUROC: {per_class}")
        print(f"[sanity] class_trust: {trust}")
        print(f"[sanity] wrote → {ct_path}")
        print("[sanity] DONE — exiting (build_class_trust mode)")
        return

    # ── Load class_trust (required for training) ───────────────────────────
    if not args.class_trust or not os.path.exists(args.class_trust):
        raise SystemExit(f"--class_trust required for training (got {args.class_trust!r}). "
                          f"Run with --build_class_trust first to derive it.")
    with open(args.class_trust) as f:
        trust_blob = json.load(f)
    class_trust: Dict[str, float] = dict(trust_blob["class_trust"])
    if not args.allow_hyp_cd_trust:
        class_trust.update(DEFAULT_TRUST_HARDCODE)   # Plan Rev 11 hard-enforce
    print(f"[setup] class_trust loaded: {class_trust}")
    classes_in_scope = []
    for cls in args.classes_in_scope:
        cls = cls.upper()
        if cls not in SUPER5_TO_IDX:
            raise SystemExit(f"unknown --classes_in_scope class {cls!r}; valid={CLASS_NAMES_SUPER5}")
        classes_in_scope.append(cls)
    if not classes_in_scope:
        raise SystemExit("--classes_in_scope must contain at least one class")
    print(f"[setup] classes_in_scope={classes_in_scope}")
    print(f"[setup] boundary target probability window=[{args.boundary_prob_min}, {args.boundary_prob_max}]")
    print(f"[setup] adv label mode={args.adv_label_mode} "
          f"teacher_mix={args.adv_teacher_mix} target_floor={args.adv_soft_target_floor}")
    print(f"[setup] hull mix label mode={args.hull_mix_label_mode} "
          f"lambda_y={args.hull_label_lambda_y} pos={args.hull_label_positive} "
          f"neg_floor={args.hull_label_negative_floor} new_cap={args.hull_label_new_class_cap}")

    # ── Load synth pool (Stage 1 frozen) for training ──────────────────────
    synth_latents, synth_labels, synth_center, source_meta = load_synth_pool(args.synth_npz)
    print(f"[setup] synth pool: {synth_latents.shape} labels={synth_labels.shape} "
          f"center={synth_center}")
    cls_dist = synth_labels.argmax(1)
    from collections import Counter
    pool_class_counts = Counter(int(c) for c in cls_dist)
    print(f"[setup] synth class counts (idx): {dict(pool_class_counts)}")
    source_counts = Counter(str(s) for s in source_meta["source_labels"])
    print(f"[setup] synth source counts: {dict(source_counts)} "
          f"has_metadata={source_meta['has_source_metadata']}")
    if args.source_sampling_strategy == "source_weighted" and not source_meta["has_source_metadata"]:
        print("[setup] WARNING: source_weighted requested but pool lacks source metadata; "
              "all samples use source='unknown'.")

    # ── PTBXL super5 train / val ────────────────────────────────────────────
    from scripts.triple_labels.label_schemes import get_scheme
    scheme = get_scheme("super5")

    label_cache = os.path.join(args.output_dir, "ptbxl_labels")
    train_idx, train_labels, _ = get_ptbxl_labels_for_scheme(
        args.ptbxl_csv, scheme, label_cache, folds=list(range(1, 9))
    )
    val_idx, val_labels, _ = get_ptbxl_labels_for_scheme(
        args.ptbxl_csv, scheme, label_cache, folds=[9]
    )

    cache_path = args.ptbxl_prep if os.path.exists(args.ptbxl_prep) else \
        os.path.join(args.output_dir, "ptbxl_preprocessed.npy")
    print(f"[setup] PTBXL preprocessed cache → {cache_path}")
    all_sig = preprocess_ptbxl_all(args.ptbxl_raw, cache_path)
    train_signals = np.asarray(all_sig[train_idx])
    val_signals = np.asarray(all_sig[val_idx])

    if args.smoke:
        train_signals = train_signals[:512]
        train_labels = train_labels[:512]
        val_signals = val_signals[:128]
        val_labels = val_labels[:128]
        args.roundtrip_anchor_n = 64
        print("[smoke] truncated PTBXL train/val + roundtrip_anchor_n=64")

    train_ds = PTBXLDatasetScheme(train_signals, train_labels,
                                  crop_len=args.crop_len, mode='train')
    val_ds = PTBXLDatasetScheme(val_signals, val_labels,
                                crop_len=args.crop_len, mode='eval')
    source_logit_anchor_loader = None
    if args.source_logit_anchor_weight > 0:
        source_logit_anchor_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
        print(
            f"[setup] source-logit anchor enabled: weight={args.source_logit_anchor_weight} "
            f"max_batches={args.source_logit_anchor_batches or 'full'}",
            flush=True,
        )
    target_real_ds = None
    if args.target_real_npz:
        with np.load(args.target_real_npz, allow_pickle=True) as real_data:
            real_signals = np.asarray(real_data["signals"], dtype=np.float32)
            real_labels = np.asarray(real_data["labels"], dtype=np.float32)
        if real_signals.ndim != 3:
            raise ValueError(f"target_real_npz signals must be 3D, got {real_signals.shape}")
        if real_signals.shape[1:] == (12, 1000):
            real_signals = real_signals.transpose(0, 2, 1)
        if real_signals.shape[1:] != (1000, 12):
            raise ValueError(f"target_real_npz signals must be (N,1000,12) or (N,12,1000), got {real_signals.shape}")
        if real_labels.shape[0] != real_signals.shape[0] or real_labels.shape[1] != NUM_SUPER5:
            raise ValueError(f"target_real_npz labels mismatch: signals={real_signals.shape} labels={real_labels.shape}")
        target_real_ds = PTBXLDatasetScheme(
            real_signals,
            real_labels,
            crop_len=args.crop_len,
            mode='train',
        )
        print(
            f"[setup] target-real supervised stream: n={len(target_real_ds)} "
            f"weight={args.target_real_weight} path={args.target_real_npz}",
            flush=True,
        )

    pos_weight = torch.tensor(
        compute_pos_weight(train_labels, NUM_SUPER5),
        dtype=torch.float32, device=args.device)
    print(f"[loss] pos_weight: {pos_weight.cpu().tolist()}")

    # reduction='none' for mask × bce (-1 sentinel handling)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ── Roundtrip anchor (cached) ────────────────────────────────────────────
    rt_cache = os.path.join(args.output_dir, f"roundtrip_anchor_n{args.roundtrip_anchor_n}.npz")
    roundtrip_ds: Optional[TensorDataset] = build_roundtrip_anchor_dataset(
        train_ds=train_ds, ecgtwin=ecgtwin, n_samples=args.roundtrip_anchor_n,
        device=args.device, crop_len=args.crop_len,
        seed=args.seed, cache_path=rt_cache,
    )
    if roundtrip_ds is not None:
        print(f"[setup] roundtrip-anchor: n={len(roundtrip_ds)}, weight={args.roundtrip_weight}")

    # ── Quick eval subset (Issue #39 ref-record exclusion) ──────────────────
    excluded = None
    if args.ref_meta_json and os.path.exists(args.ref_meta_json):
        with open(args.ref_meta_json) as f:
            meta = json.load(f)
        excluded = set(meta.get("ref_record_ids", []))
        print(f"[setup] excluding {len(excluded)} ref_record_ids from "
              f"quick_eval (center={args.center_name})")

    qe_cache = os.path.join(args.output_dir,
                            f"quick_eval_subset_n{args.quick_eval_n_per_center}.cache")
    quick_subset = build_quick_eval_subset_super5(
        centers=args.quick_eval_centers, data_dir=args.data_dir,
        n_per_center=args.quick_eval_n_per_center, cache_path=qe_cache,
        seed=args.seed, exclude_record_ids=excluded,
    )

    print("[baseline] Computing baseline quick-eval ...")
    baseline_qe = quick_eval_super5(victim.model, quick_subset, args.device,
                                    crop_len=args.crop_len)
    print(f"[baseline] avg macro AUROC={baseline_qe['avg_macro_auroc']}, "
          f"AUPRC={baseline_qe['avg_macro_auprc']}")
    for c, info in baseline_qe["per_center"].items():
        print(f"    {c}: AUROC={info['macro_auroc']}  AUPRC={info['macro_auprc']}  "
              f"(n={info['n']}  classes_used={info['n_classes_used']})")

    # ── PGD / Latent-Hull generator + buffer ────────────────────────────────
    # Note: generator __init__ calls victim.parameters().requires_grad_(False)
    # which would prevent us from training the victim afterwards. We re-enable
    # requires_grad on all params right after, then snapshot EWA + build optimizer.
    latent_hull_index = None
    if args.attack_mode == "latent_hull":
        pgd_gen = LatentHullPGDGenerator(
            ecgtwin_wrapper=ecgtwin, victim=victim,
            epsilon=args.pgd_eps,
            hull_lambda=args.hull_lambda,
            hull_steps=args.hull_steps,
            hull_lr=args.hull_lr,
            weight_mode=args.hull_weight_mode,
            dirichlet_alpha=args.hull_dirichlet_alpha,
            device=args.device,
        )
        latent_hull_index = SameLabelLatentIndex(
            synth_latents, synth_labels,
            label_mode=args.hull_label_mode,
            seed=args.seed,
            include_self=args.hull_include_anchor,
        )
        print(f"[setup] latent-hull index mode={args.hull_label_mode} "
              f"include_anchor={args.hull_include_anchor} "
              f"sizes={latent_hull_index.class_sizes()}")
    else:
        pgd_gen = PGDAdvDiffGenerator(
            ecgtwin_wrapper=ecgtwin, victim=victim,
            epsilon=args.pgd_eps, K_pgd=args.pgd_K,
            alpha=args.pgd_alpha, delta_init_scale=args.delta_init_scale,
            device=args.device,
        )
    for p in victim.model.parameters():
        p.requires_grad_(True)
    source_logit_teacher_model = None
    if args.source_logit_anchor_weight > 0:
        source_logit_teacher_model = copy.deepcopy(victim.model).to(args.device)
        source_logit_teacher_model.eval()
        for p in source_logit_teacher_model.parameters():
            p.requires_grad_(False)
        print("[setup] frozen source-logit teacher enabled")
    teacher_model = None
    teacher_label_modes = {"mixed_soft", "teacher_soft", "latent_mixed_teacher"}
    if args.adv_label_mode in teacher_label_modes:
        teacher_model = copy.deepcopy(victim.model).to(args.device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print("[setup] frozen initial teacher enabled for soft adv labels")
    buffer = QualityAwareBuffer(max_size=args.qab_size)

    # EWA anchor snapshot — only trainable params (after PGD-freeze override)
    ewa_params = [p.data.clone().detach()
                  for p in victim.model.parameters() if p.requires_grad]
    print(f"[setup] EWA anchor: {len(ewa_params)} param tensors snapshotted "
          f"({sum(p.numel() for p in ewa_params):,} elements)")

    # ── Optimizer / scheduler ───────────────────────────────────────────────
    trainable = [p for p in victim.model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs,
                                  eta_min=args.lr * 0.01)

    log: Dict[str, Any] = {
        "args": vars(args),
        "baseline_quick_eval": baseline_qe,
        "class_trust": class_trust,
        "source_meta": {
            "source_names": source_meta["source_names"],
            "source_counts": dict(source_counts),
            "has_source_metadata": source_meta["has_source_metadata"],
        },
        "adv_label_policy": {
            "mode": args.adv_label_mode,
            "teacher_mix": args.adv_teacher_mix,
            "soft_target_floor": args.adv_soft_target_floor,
            "hull_mix_label_mode": args.hull_mix_label_mode,
            "hull_label_lambda_y": args.hull_label_lambda_y,
            "hull_label_positive": args.hull_label_positive,
            "hull_label_negative_floor": args.hull_label_negative_floor,
            "hull_label_new_class_cap": args.hull_label_new_class_cap,
        },
        "epochs": [],
    }
    def selected_es_metric(qe: Dict[str, Any]) -> float:
        if args.es_metric == "val_macro_auroc":
            return float(qe.get("avg_macro_auroc", float("nan")))
        if args.es_metric == "val_macro_auprc":
            return float(qe.get("avg_macro_auprc", float("nan")))
        center_info = qe.get("per_center", {}).get(args.center_name, {})
        if args.es_metric == "target_macro_auroc":
            return float(center_info.get("macro_auroc", float("nan")))
        if args.es_metric == "target_macro_auprc":
            return float(center_info.get("macro_auprc", float("nan")))
        raise ValueError(f"unsupported es_metric={args.es_metric}")

    best_metric = selected_es_metric(baseline_qe)
    if best_metric != best_metric:
        best_metric = -1.0  # NaN-safe
    best_epoch = 0
    epochs_since_best = 0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    torch.save(victim.model.state_dict(), best_ckpt_path)
    log_path = os.path.join(args.output_dir, "training_log.json")
    es_path = os.path.join(args.output_dir, "early_stop_info.json")

    # Plan Rev 13.2: stratified pool walker over NORM/MI/STTC scope only
    source_weight_map = parse_source_weight_map(args.source_weights)
    source_class_weight_map = parse_class_source_weight_map(args.source_class_weights)
    walker = StratifiedPoolWalker(
        labels_one_hot=synth_labels,
        classes_in_scope=classes_in_scope,
        class_to_idx=SUPER5_TO_IDX, seed=args.seed,
        source_labels=source_meta["source_labels"],
        source_sampling_strategy=args.source_sampling_strategy,
        source_weights=source_weight_map,
        source_class_weights=source_class_weight_map,
        source_floor_per_class=args.source_floor_per_class,
    )
    print(f"[setup] walker class sizes: {walker.class_sizes()}")
    if args.source_sampling_strategy == "source_weighted":
        print(f"[setup] walker source-class sizes: {walker.source_class_sizes()}")
        print(f"[setup] source weights: global={source_weight_map or {'<default>': 1.0}} "
              f"class_overrides={source_class_weight_map or {}} "
              f"floor_per_class={args.source_floor_per_class}")

    rng = np.random.default_rng(args.seed)
    consecutive_low_asr = 0

    # ── Main loop ───────────────────────────────────────────────────────────
    for epoch in range(1, args.n_epochs + 1):
        epoch_t0 = time.time()

        # Phase A: PGD on synth pool with the *current* victim
        # Plan Rev 13.2: StratifiedPoolWalker draws no-revisit-per-epoch,
        # restricted to NORM/MI/STTC scope.
        victim.model.eval()
        per_cls = max(1, args.K_anchor // len(classes_in_scope))
        k_per_cls = {c: per_cls for c in classes_in_scope}
        rem = args.K_anchor - per_cls * len(classes_in_scope)
        for i_extra in range(rem):
            k_per_cls[classes_in_scope[i_extra % len(classes_in_scope)]] += 1
        drawn = walker.sample(k_per_cls)
        if args.source_sampling_strategy == "source_weighted":
            print(f"[ep{epoch:02d}] anchor source counts: {walker.last_source_counts} "
                  f"class_source={walker.last_class_source_counts}", flush=True)
        all_picks = np.concatenate(
            [drawn[c] for c in classes_in_scope if drawn[c].size > 0]
        ) if any(drawn[c].size > 0 for c in classes_in_scope) else np.empty(0, dtype=np.int64)
        adv_signals, anc_signals, target_oh, delta_stats = run_pgd_on_synth_pool(
            pgd_gen=pgd_gen,
            synth_latents=synth_latents,
            synth_labels=synth_labels,
            K_anchor=args.K_anchor,
            pgd_batch=args.pgd_batch,
            rng=rng, device=args.device,
            picked_indices=all_picks,
            attack_mode=args.attack_mode,
            latent_hull_index=latent_hull_index,
            hull_M=args.hull_M,
            hull_mix_label_mode=args.hull_mix_label_mode,
            hull_label_lambda_y=args.hull_label_lambda_y,
            hull_label_positive=args.hull_label_positive,
            hull_label_negative_floor=args.hull_label_negative_floor,
            hull_label_new_class_cap=args.hull_label_new_class_cap,
        )
        if adv_signals.shape[0] == 0:
            print(f"[ep{epoch:02d}] empty walker pick — skip epoch", flush=True)
            continue

        # Phase B: gates
        # ASR (signals → victim)
        asr_info = compute_asr(victim, adv_signals, target_oh,
                               device=args.device, batch_size=128)
        # Semantic (Einthoven, HR, QRS)
        sem_info = compute_semantic_gate(
            adv_signals, anc_signals,
            einthoven_p95_max=args.einthoven_p95_max,
        )

        gate_skipped = False
        if (not args.disable_quality_gate) and (not sem_info.get("PASS", False)):
            gate_skipped = True
            print(f"[ep{epoch:02d}] medical gate FAIL: {sem_info.get('fail_reasons')} "
                  f"— skip buffer push this epoch", flush=True)
        else:
            if args.disable_quality_gate and not sem_info.get("PASS", False):
                print(f"[ep{epoch:02d}] medical gate FAIL ignored: "
                      f"{sem_info.get('fail_reasons')}", flush=True)
            # Center-crop adv (B, 12, 1000) → (B, 12, crop_len) on the time axis
            start = (adv_signals.shape[-1] - args.crop_len) // 2
            adv_ct_crop = adv_signals[..., start:start + args.crop_len]
            with torch.no_grad():
                lg_chunks = []
                teacher_prob_chunks = []
                for i in range(0, adv_ct_crop.shape[0], 128):
                    x_t = torch.from_numpy(adv_ct_crop[i:i + 128]).float().to(args.device)
                    lg_chunks.append(victim.model(x_t).cpu().numpy())
                    if teacher_model is not None:
                        teacher_prob_chunks.append(torch.sigmoid(teacher_model(x_t)).cpu().numpy())
                logits_arr = np.concatenate(lg_chunks)
                teacher_probs_arr = (
                    np.concatenate(teacher_prob_chunks)
                    if teacher_prob_chunks else None
                )
            push_stats = push_adv_to_buffer(
                buffer=buffer, adv_signals_ct=adv_signals,
                target_one_hot=target_oh, victim_logits=logits_arr,
                crop_len=args.crop_len, class_trust=class_trust,
                boundary_prob_min=args.boundary_prob_min,
                boundary_prob_max=args.boundary_prob_max,
                teacher_probs=teacher_probs_arr,
                label_mode=args.adv_label_mode,
                teacher_mix=args.adv_teacher_mix,
                soft_target_floor=args.adv_soft_target_floor,
            )
        # Track consecutive low ASR
        if asr_info["asr_overall"] < args.asr_low_threshold:
            consecutive_low_asr += 1
        else:
            consecutive_low_asr = 0
        if consecutive_low_asr >= args.asr_consec_low_max:
            raise RuntimeError(
                f"PGD broken: ASR < {args.asr_low_threshold} for "
                f"{args.asr_consec_low_max} consecutive epochs — abort training.")

        if epoch > 1 and (epoch - 1) % args.rescore_interval == 0 and len(buffer) > 0:
            buffer.rescore(victim.model, args.device)

        # Phase C: build mixed loader (cold-start guard for empty buffer)
        buf_ds = buffer.to_dataset()
        streams = []
        if args.ptbxl_weight > 0:
            streams.append((train_ds, args.ptbxl_weight, None))
        if target_real_ds is not None and args.target_real_weight > 0:
            streams.append((target_real_ds, args.target_real_weight, None))
        if roundtrip_ds is not None and args.roundtrip_weight > 0:
            streams.append((roundtrip_ds, args.roundtrip_weight, None))
        if buf_ds is not None and len(buf_ds) > 0:
            streams.append((buf_ds, args.adv_weight, buffer.get_sampling_weights()))

        if len(streams) == 0:
            raise RuntimeError(
                "No training streams are active. Check ptbxl_weight, "
                "target_real_weight, roundtrip_weight, and adv buffer gates."
            )
        if len(streams) == 1:
            only_ds = streams[0][0]
            train_loader = DataLoader(only_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True,
                                      drop_last=True,
                                      persistent_workers=args.num_workers > 0)
        else:
            weights = []
            total_n = 0
            for ds_i, base_w, per_sample_w in streams:
                n = len(ds_i)
                if per_sample_w is None:
                    weights.extend([base_w] * n)
                else:
                    weights.extend([base_w * max(w, 0.05) for w in per_sample_w])
                total_n += n
            sampler = WeightedRandomSampler(weights, num_samples=total_n, replacement=True)
            combined = ConcatDataset([s[0] for s in streams])
            train_loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler,
                                      num_workers=args.num_workers, pin_memory=True,
                                      drop_last=True,
                                      persistent_workers=args.num_workers > 0)

        # Phase D: train
        train_loss = train_one_epoch_masked_bce(
            victim.model, train_loader, optimizer, criterion, args.device,
            grad_clip=args.grad_clip, ewa_params=ewa_params,
            anchor_lambda=args.anchor_lambda, ewa_decay=args.ewa_decay,
        )
        source_logit_anchor_loss = float("nan")
        if (
            args.source_logit_anchor_weight > 0
            and source_logit_teacher_model is not None
            and source_logit_anchor_loader is not None
        ):
            source_logit_anchor_loss = train_source_logit_anchor_epoch(
                model=victim.model,
                teacher_model=source_logit_teacher_model,
                loader=source_logit_anchor_loader,
                optimizer=optimizer,
                device=args.device,
                weight=args.source_logit_anchor_weight,
                max_batches=args.source_logit_anchor_batches,
                grad_clip=args.grad_clip,
            )
        scheduler.step()

        # Phase E: PTBXL val loss
        victim.model.eval()
        val_losses = []
        with torch.no_grad():
            for sigs, labels in val_loader:
                sigs = sigs.to(args.device)
                labels = labels.to(args.device)
                logits = victim.model(sigs)
                mask = (labels >= 0).float()
                labels_safe = torch.where(mask.bool(), labels, torch.zeros_like(labels))
                per_elem = criterion(logits, labels_safe)
                denom = mask.sum().clamp(min=1.0)
                vl = (per_elem * mask).sum() / denom
                val_losses.append(vl.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

        elapsed = time.time() - epoch_t0
        entry = {
            "epoch": epoch,
            "attack_mode": args.attack_mode,
            "train_loss": round(train_loss, 4),
            "source_logit_anchor_loss": round(source_logit_anchor_loss, 6)
            if source_logit_anchor_loss == source_logit_anchor_loss else None,
            "val_loss":   round(val_loss, 4),
            "asr_overall": round(float(asr_info["asr_overall"]), 4),
            "asr_per_class": {k: round(float(v), 4) for k, v in asr_info["per_class_asr"].items()},
            "multilabel_positive_label_asr": round(
                float(asr_info.get("multilabel_positive_label_asr", float("nan"))), 4
            ),
            "sample_any_positive_below_0p5_asr": round(
                float(asr_info.get("sample_any_positive_below_0p5_asr", float("nan"))), 4
            ),
            "sample_all_positive_below_0p5_asr": round(
                float(asr_info.get("sample_all_positive_below_0p5_asr", float("nan"))), 4
            ),
            "sample_all_positive_recognized_rate": round(
                float(asr_info.get("sample_all_positive_recognized_rate", float("nan"))), 4
            ),
            "per_class_positive_label_asr": {
                k: round(float(v), 4)
                for k, v in asr_info.get("per_class_positive_label_asr", {}).items()
            },
            "einthoven_p95":  round(float(sem_info.get("einthoven_mean_p95", float('nan'))), 4),
            "hr_mean_delta": round(float(sem_info.get("hr_mean_delta", float('nan'))), 4),
            "qrs_amp_ratio": round(float(sem_info.get("qrs_amp_ratio", float('nan'))), 4)
                if sem_info.get("qrs_amp_ratio", None) == sem_info.get("qrs_amp_ratio", None)
                else None,
            "buffer_skipped": gate_skipped,
            "buffer_size":   len(buffer),
            "push_stats": push_stats if not gate_skipped else {},
            "delta_mean":    round(delta_stats["mean_delta_norm"], 4),
            "delta_max":     round(delta_stats["max_delta_norm"], 4),
            "lr":            round(optimizer.param_groups[0]["lr"], 6),
            "time_s":        round(elapsed, 1),
        }
        if args.source_sampling_strategy == "source_weighted":
            entry.update({
                "anchor_source_counts": dict(walker.last_source_counts),
                "anchor_class_source_counts": walker.last_class_source_counts,
            })
        if args.attack_mode == "latent_hull":
            entry.update({
                "hull_M": args.hull_M,
                "hull_lambda": args.hull_lambda,
                "hull_steps": args.hull_steps,
                "hull_label_mode": args.hull_label_mode,
                "hull_mix_label_mode": args.hull_mix_label_mode,
                "hull_label_lambda_y": args.hull_label_lambda_y,
                "hull_label_positive": args.hull_label_positive,
                "hull_label_negative_floor": args.hull_label_negative_floor,
                "hull_label_new_class_cap": args.hull_label_new_class_cap,
                "hull_include_anchor": args.hull_include_anchor,
                "hull_weight_entropy_mean": round(
                    float(delta_stats.get("hull_weight_entropy_mean", float('nan'))), 4
                ),
                "hull_weight_top1_mean": round(
                    float(delta_stats.get("hull_weight_top1_mean", float('nan'))), 4
                ),
            })
        print(f"Ep {epoch:2d}/{args.n_epochs} | train={train_loss:.4f} val={val_loss:.4f} | "
              f"asr={asr_info['asr_overall']:.2f} "
              f"ml_any={asr_info.get('sample_any_positive_below_0p5_asr', float('nan')):.2f} "
              f"ml_pos={asr_info.get('multilabel_positive_label_asr', float('nan')):.2f} "
              f"eint_p95={sem_info.get('einthoven_mean_p95', float('nan')):.3f} "
              f"buf={len(buffer)} skip={gate_skipped} attack={args.attack_mode} | {elapsed:.0f}s")

        # Phase F: quick eval (every eval_every; also last epoch)
        if (epoch % args.eval_every == 0) or (epoch == args.n_epochs):
            qe = quick_eval_super5(victim.model, quick_subset, args.device,
                                   crop_len=args.crop_len)
            entry["quick_eval"] = qe
            print(f"   quick eval: avg AUROC={qe['avg_macro_auroc']}  "
                  f"AUPRC={qe['avg_macro_auprc']}")
            for c, info in qe["per_center"].items():
                print(f"      {c}: AUROC={info['macro_auroc']}  AUPRC={info['macro_auprc']}")
            cur = selected_es_metric(qe)
            improved = (cur == cur) and (cur > best_metric + 1e-6)   # NaN-safe
            if improved:
                best_metric = cur
                best_epoch = epoch
                epochs_since_best = 0
                torch.save(victim.model.state_dict(), best_ckpt_path)
                print(f"   ** saved best @ ep{epoch}: {args.es_metric} {best_metric}")
                entry["best_update"] = True
            else:
                epochs_since_best += args.eval_every
                entry["best_update"] = False

        log["epochs"].append(entry)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2, default=str)

        # Plan Rev 13.1: early-stop on val_macro_auroc plateau
        if (epoch % args.eval_every == 0) and epochs_since_best >= args.patience:
            print(f"\n[early-stop] patience {args.patience} hit at ep{epoch}; "
                  f"best @ ep{best_epoch} ({args.es_metric}={best_metric})")
            with open(es_path, "w") as f:
                json.dump({
                    "stopped_epoch": epoch, "best_epoch": best_epoch,
                    "best_metric": best_metric, "es_metric": args.es_metric, "patience": args.patience,
                    "n_epochs_run": epoch, "early_stopped": True,
                }, f, indent=2)
            break
    else:
        # Loop completed without early-stop
        with open(es_path, "w") as f:
            json.dump({
                "stopped_epoch": args.n_epochs, "best_epoch": best_epoch,
                "best_metric": best_metric, "es_metric": args.es_metric, "patience": args.patience,
                "n_epochs_run": args.n_epochs, "early_stopped": False,
            }, f, indent=2)

    # Final result
    final = {
        "args":               vars(args),
        "baseline_quick_eval": baseline_qe,
        "best_metric":         best_metric,
        "es_metric":           args.es_metric,
        "n_epochs_run":       len(log["epochs"]),
        "last_quick_eval":    log["epochs"][-1].get("quick_eval") if log["epochs"] else None,
    }
    with open(os.path.join(args.output_dir, "train_result.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"Training done. best {args.es_metric}={best_metric} → {best_ckpt_path}")


if __name__ == "__main__":
    main()
