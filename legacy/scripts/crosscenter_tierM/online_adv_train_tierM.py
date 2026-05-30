"""
Tier-M 6-class 在线对抗训练 (Iterative adv-gen ↔ fine-tune loop)

Pipeline (per epoch):
  1. Weak-spot 预算 → 对 (center, class) 分配 N_adv 配额
  2. ECGTwin + AdvDiff 边界引导生成 adv latent（prob ∈ [accept_lo, accept_hi]）
     - ref = 来自目标域中心的真实 ECG（已做 unified_preprocess），先 VAE encode 得 z_ref
     - prepare_conditions(ref_latent=z_ref, target_text=TIER_M_TEXT_PROMPT[class])
     - BoundaryAdvDiffGenerator.generate_batch → 接收条件 logits(target_class)→0 的 latent
  3. Augmix-in-latent：每个 adv 样本用其 ref 做时域增强 chain 基底，adv latent 作为 1 条 chain
  4. QualityAwareBuffer 收纳（按 informativeness = 1 - 2|prob(target)-0.5| 评分淘汰）
  5. 构造混合 DataLoader（PTBXL real + buffer adv，WeightedRandomSampler 权重 2:1）
  6. 训练 1 epoch（BCE + EWA anchor 正则，防止 PTBXL 源性能遗忘）
  7. 每 eval_every 个 epoch：quick eval on 4 主中心 stratified subset
     - 若主 4 中心 avg macro AUROC 创新高，保存 best_model.pt
     - 刷新 budget 权重

输出: /root/autodl-tmp/crosscenter_tierM_online/
  best_model.pt, training_log.json, quick_eval_epoch_N.json,
  sample_adv_epoch_N.png (可视化), train_result.json
"""

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path("/root/autodl-tmp/models/DeepECG/notebooks")))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from adversarial.adv_generate import (  # noqa: E402
    BoundaryAdvDiffGenerator, prepare_conditions_for_prompt,
)
from adversarial.efficientnet_victim_tierM import (  # noqa: E402
    EfficientNetVictimTierM, DEFAULT_TIERM_CKPT, TIERM_INPUT_LENGTH,
)
from adversarial.tierM_labels import (  # noqa: E402
    TIER_M_CLASSES, TIER_M_CLASS_TO_IDX, TIER_M_TEXT_PROMPT,
    get_target_indices_tierM,
)
from methods.augmix.latent_viz.latent_augmix import latent_augmix_on_signal  # noqa: E402
from scripts.crosscenter_v2.label_alignment_v2 import (  # noqa: E402
    TIER_M_IDX, NUM_CLASSES_TIER_M, snomed_to_26, has_any_scored_class,
)
from scripts.crosscenter_v2.preprocess_utils import (  # noqa: E402
    unified_preprocess_to_1000, crop_signal_tc,
)
from scripts.crosscenter_tierM.refs_per_center import (  # noqa: E402
    sample_refs_per_center, DEFAULT_PN2021_DIR, DEFAULT_REF_CACHE_DIR,
    MAIN_CENTERS_4,
)
from scripts.crosscenter_tierM.eval_crosscenter_tierM import (  # noqa: E402
    scan_center_records,
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/crosscenter_tierM_online"
BASELINE_EVAL_JSON = "/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json"
PTBXL_CACHE = "/root/autodl-tmp/crosscenter_v2/ptbxl_preprocessed.npy"
PTBXL_CSV = "/root/ECG_adv_Gen/datasets/PTBXL/ptbxl_database.csv"

TIERM_GEN_HYPERPARAMS_DEFAULT = {
    "num_inference_steps": 50,
    "adversarial_guidance_scale": 0.03,
    "noise_sampling_guidance_scale": 0.01,
    "num_noise_sampling_steps": 2,
    "guidance_start_fraction": 0.8,
    "acceptance_range": [0.5, 0.6],   # user-specified tight band
    "batch_size": 4,
}


# ------------------------------------------------------------------
# Quality-Aware Buffer (Tier-M: 6-class, (12, 250) tensors)
# ------------------------------------------------------------------

class QualityAwareBufferTierM:
    """FIFO + quality-score eviction buffer for Tier-M adv samples (12, 250)."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self.ecg_list: List[torch.Tensor] = []     # each (12, 250)
        self.label_list: List[torch.Tensor] = []   # each (6,)
        self.score_list: List[float] = []          # informativeness

    def __len__(self):
        return len(self.ecg_list)

    def add_one(self, ecg_ct: torch.Tensor, label_6: torch.Tensor, score: float):
        self.ecg_list.append(ecg_ct.detach().cpu())
        self.label_list.append(label_6.detach().cpu())
        self.score_list.append(float(score))
        self._evict()

    def _evict(self):
        while len(self.ecg_list) > self.max_size:
            j = int(np.argmin(self.score_list))
            self.ecg_list.pop(j)
            self.label_list.pop(j)
            self.score_list.pop(j)

    def to_dataset(self) -> Optional[TensorDataset]:
        if not self.ecg_list:
            return None
        return TensorDataset(
            torch.stack(self.ecg_list).float(),
            torch.stack(self.label_list).float(),
        )

    @torch.no_grad()
    def rescore(self, victim_model: nn.Module, device: str = "cuda"):
        if not self.ecg_list:
            return
        victim_model.eval()
        all_ecg = torch.stack(self.ecg_list).to(device)
        chunk = 128
        new_scores: List[float] = []
        for i in range(0, all_ecg.shape[0], chunk):
            logits = victim_model(all_ecg[i:i + chunk])
            # informativeness: smaller |logit| → closer to decision boundary → higher score
            scores = (1.0 / (1.0 + logits.abs().mean(dim=1))).cpu().tolist()
            new_scores.extend(scores)
        self.score_list = new_scores

    def get_sampling_weights(self) -> List[float]:
        return [max(s, 0.05) for s in self.score_list]


# ------------------------------------------------------------------
# Weak-spot budget
# ------------------------------------------------------------------

def build_budget(
    eval_json: Dict[str, Any],
    n_adv_total: int,
    centers: List[str],
    tier_m_classes: List[str],
    min_weight: float = 0.01,
    cell_skip_threshold: float = 1.1,   # default >1 disables skip; pass <1 to enable weak-only
    min_alloc_per_cell: int = 1,        # set to 0 to disable the "≥1 per valid cell" floor
) -> List[Dict[str, Any]]:
    """For each (center, class) with a finite AUROC in baseline eval, weight = max(min_weight, 1 - AUROC).
    Allocate n_adv_total across entries proportionally, rounding to integer with a post-hoc fix-up.

    Tunable gates for focusing attack budget:
      - cell_skip_threshold: skip cells whose baseline AUROC >= threshold (keep only weak cells)
      - min_alloc_per_cell: 1 guarantees every valid cell gets at least one attempt; 0 lets pure
        weight proportion dominate (so strong cells can get 0 allocations)
    """
    entries: List[Dict[str, Any]] = []
    for c in centers:
        center_info = eval_json.get("centers", {}).get(c)
        if center_info is None:
            continue
        per_class = center_info.get("per_class", {})
        for cls in tier_m_classes:
            auroc = per_class.get(cls, {}).get("auroc")
            if auroc is None:   # structural N/A, skip
                continue
            if float(auroc) >= cell_skip_threshold:
                continue        # skip already-strong cells
            weight = max(min_weight, 1.0 - float(auroc))
            entries.append({
                "center": c, "class": cls, "weight": weight,
                "baseline_auroc": float(auroc),
                "n_pos_in_center": int(per_class[cls].get("n_pos", 0)),
            })
    if not entries:
        raise RuntimeError(
            f"No valid (center, class) entries from baseline eval "
            f"(cell_skip_threshold={cell_skip_threshold})"
        )
    total_w = sum(e["weight"] for e in entries)
    n_remaining = n_adv_total
    for e in entries:
        n = int(round(n_adv_total * e["weight"] / total_w))
        e["n_alloc"] = max(min_alloc_per_cell, n)
        n_remaining -= e["n_alloc"]
    # Fix-up: if over-allocated, strip from lowest-weight entries; if under, add to highest.
    if n_remaining != 0:
        order = sorted(range(len(entries)),
                       key=lambda i: entries[i]["weight"],
                       reverse=(n_remaining > 0))
        i = 0
        while n_remaining != 0 and i < len(order) * 10:
            j = order[i % len(order)]
            delta = 1 if n_remaining > 0 else -1
            if delta == -1 and entries[j]["n_alloc"] <= min_alloc_per_cell:
                i += 1
                continue
            entries[j]["n_alloc"] += delta
            n_remaining -= delta
            i += 1
    # Drop zero-allocation entries (only relevant when min_alloc_per_cell=0)
    entries = [e for e in entries if e["n_alloc"] > 0]
    return entries


# ------------------------------------------------------------------
# Reference encoding + ref_items_by_center
# ------------------------------------------------------------------

def _to_ecgtwin_order(sig_ct_12x1000: np.ndarray) -> np.ndarray:
    """Canonical PTBXL order → ECGTwin order (involutive swap via lead_utils)."""
    from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES
    return sig_ct_12x1000[ECGTWIN_TO_PTBXL_INDICES, :]


def encode_refs_per_center(
    refs: Dict[str, Dict], ecgtwin: ECGTwinWrapper, device: str,
    precomputed_ref_text: str = "ecg record",
) -> Dict[str, List[Dict]]:
    """For each center, VAE-encode the K (1000, 12) signals → latent (4, 128) and
    build ref_items compatible with prepare_conditions_for_prompt."""
    assert ecgtwin.encoder is not None, "ECGTwinWrapper must be loaded with load_encoder=True"
    assert ecgtwin.embedding_model is not None, "ECGTwinWrapper must be loaded with load_text_model=True"

    # Precompute generic ref-side text embed so prepare_conditions avoids repeat work.
    with torch.no_grad():
        ref_text_embed = ecgtwin.get_text_embedding(precomputed_ref_text)   # (num_reports, 768)

    dev = torch.device(device)
    out: Dict[str, List[Dict]] = {}
    for center, data in refs.items():
        signals_tc = data["signals_tc"]       # (K, 1000, 12)
        labels_6 = data["labels_6"]           # (K, 6)
        record_ids = data["record_ids"]
        K = signals_tc.shape[0]

        # (K, 1000, 12) → (K, 12, 1000) canonical → reorder to ECGTwin → (K, 12, 1024) → (K, 1024, 12)
        sig_kct = np.transpose(signals_tc, (0, 2, 1)).astype(np.float32)    # (K, 12, 1000)
        # Reorder canonical → ECGTwin
        from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES
        sig_kct = sig_kct[:, ECGTWIN_TO_PTBXL_INDICES, :]                    # (K, 12, 1000) in ECGTwin order
        sig_kct_t = torch.from_numpy(sig_kct).float().to(dev)
        sig_kct_t = F.interpolate(sig_kct_t, size=1024, mode="linear", align_corners=True)  # (K, 12, 1024)
        sig_klc_t = sig_kct_t.transpose(1, 2).contiguous()                   # (K, 1024, 12)
        with torch.no_grad():
            latents = ecgtwin.encode_ecg(sig_klc_t)                          # (K, 4, 128)

        ref_items = []
        for i in range(K):
            ref_items.append({
                "data": latents[i].detach().cpu(),                           # (4, 128)
                "label": {
                    "hr": 70.0, "age": 50, "sex": 0,
                    "text": precomputed_ref_text,
                    "text_embed": ref_text_embed.detach().cpu(),
                },
                "true_label_6": labels_6[i].copy(),
                "record_id": record_ids[i],
                "center": center,
                # raw canonical signal @100Hz (for augmix chain basis)
                "ref_signal_ct_1000_canonical": np.transpose(signals_tc[i], (1, 0)).astype(np.float32),  # (12, 1000)
            })
        out[center] = ref_items
        print(f"  [encode_refs] {center}: K={K} refs encoded, "
              f"latent shape={tuple(latents.shape)}")
    return out


# ------------------------------------------------------------------
# Adversarial generation helper
# ------------------------------------------------------------------

def generate_adv_for_epoch(
    generator: BoundaryAdvDiffGenerator,
    ecgtwin: ECGTwinWrapper,
    budget: List[Dict[str, Any]],
    ref_items_by_center: Dict[str, List[Dict]],
    z_hist: Dict[Tuple[str, str], torch.Tensor],
    device: str,
    best_of_k: int = 5,
    max_attempts_per_entry: int = 50,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Generate adv samples for each budget entry. Returns list of per-sample dicts + per-entry stats."""
    all_items: List[Dict] = []
    stats: Dict[str, int] = {"attempted": 0, "accepted": 0, "entries_completed": 0, "entries_skipped": 0}

    for entry_i, entry in enumerate(budget):
        center = entry["center"]
        cls = entry["class"]
        n_alloc = entry["n_alloc"]
        refs = ref_items_by_center.get(center, [])
        if not refs:
            stats["entries_skipped"] += 1
            print(f"  [gen] ({entry_i+1}/{len(budget)}) {center:<22}×{cls:<5}  SKIP (no refs)",
                  flush=True)
            continue
        text_prompt = TIER_M_TEXT_PROMPT[cls]
        target_indices = get_target_indices_tierM(cls)
        key = (center, cls)

        cell_t0 = time.time()
        print(f"  [gen] ({entry_i+1}/{len(budget)}) {center:<22}×{cls:<5}  "
              f"target={n_alloc} cap_attempts={max_attempts_per_entry} ...",
              flush=True)

        n_collected = 0
        n_attempts = 0
        while n_collected < n_alloc and n_attempts < max_attempts_per_entry:
            n_attempts += 1
            stats["attempted"] += 1
            # Random ref pick from this center's pool
            ref_item = refs[np.random.randint(0, len(refs))]
            ref_latent = ref_item["data"]        # (4, 128)
            ref_label = ref_item["label"]
            conditions = ecgtwin.prepare_conditions(
                ref_latent=ref_latent,
                ref_label=ref_label,
                batch_size=generator.hp["batch_size"],
                target_text=text_prompt,
            )
            conditions["ref_latent_raw"] = ref_latent
            hist_lat = z_hist.get(key)
            clean_lat = ref_latent

            batch_result = generator.generate_batch(
                conditions=conditions,
                target_indices=target_indices,
                batch_size=generator.hp["batch_size"],
                historical_latent=hist_lat,
                clean_latent=clean_lat,
                best_of_k=best_of_k if hist_lat is not None else 0,
            )
            n_new = batch_result["ecg"].shape[0]
            if n_new == 0:
                continue

            z_hist[key] = batch_result["latents"][-1].cpu()

            kept_in_iter = 0
            for i in range(n_new):
                # probs here are (6,) for Tier-M because generate_batch uses victim.forward_from_latent
                # which returns sigmoid(model_output). Our victim model has 6 classes.
                all_items.append({
                    "z_adv": batch_result["latents"][i],         # (4, 128) CPU
                    "probs_6": batch_result["probs"][i],         # (6,)
                    "target_class": cls,
                    "target_idx": target_indices[0],
                    "center": center,
                    "ref_true_label_6": ref_item["true_label_6"],
                    "ref_signal_ct_1000_canonical": ref_item["ref_signal_ct_1000_canonical"],
                    "record_id": ref_item["record_id"],
                })
                n_collected += 1
                kept_in_iter += 1
                if n_collected >= n_alloc:
                    break
            stats["accepted"] += kept_in_iter
        stats["entries_completed"] += 1 if n_collected >= n_alloc else 0
        elapsed = time.time() - cell_t0
        rate = n_collected / max(1, n_attempts)
        print(f"      → accepted={n_collected}/{n_alloc}  attempts={n_attempts}  "
              f"rate={rate:.2f}  elapsed={elapsed:.0f}s",
              flush=True)

    return all_items, stats


# ------------------------------------------------------------------
# Augmix injection + buffer populate
# ------------------------------------------------------------------

def _center_crop_ct(sig_ct: np.ndarray, L: int) -> np.ndarray:
    T = sig_ct.shape[-1]
    if T == L:
        return sig_ct
    if T < L:
        pad = L - T
        left = pad // 2
        right = pad - left
        return np.pad(sig_ct, ((0, 0), (left, right)))
    start = (T - L) // 2
    return sig_ct[:, start:start + L]


def augmix_inject_and_buffer(
    adv_items: List[Dict],
    ecgtwin_encoder_wrapper: ECGTwinWrapper,
    buffer: QualityAwareBufferTierM,
    augmix_width: int,
    augmix_severity: int,
    soft_labels: bool,
    device: str,
    adv_label_mode: str = "target_only",       # 'ref' | 'target_only'
    tier_m_crop_len: int = TIERM_INPUT_LENGTH,
):
    """Adv sample labeling:
      - 'ref'         : keep ref's true 6-class multi-label, force target class = 1 (legacy v1-v4)
      - 'target_only' : only target class is labeled (=1); other 5 dims set to -1
                        → training loop MUST mask -1 entries from BCE loss
                        (prevents multi-label noise leaking across classes)
    """
    for item in adv_items:
        ref_sig_ct = item["ref_signal_ct_1000_canonical"]                   # (12, 1000)
        z_adv = item["z_adv"].unsqueeze(0) if item["z_adv"].dim() == 2 else item["z_adv"]  # (1, 4, 128)
        res = latent_augmix_on_signal(
            wrapper=ecgtwin_encoder_wrapper,
            signal_ct_100hz=ref_sig_ct,
            severity=augmix_severity,
            width=augmix_width,
            depth=-1,
            alpha=1.0,
            device=device,
            inject_latents=[z_adv],
        )
        augmix_ct_1000 = res.augmix_latent_ct                               # (12, 1000)
        augmix_ct_250 = _center_crop_ct(augmix_ct_1000, tier_m_crop_len)    # (12, 250)

        t_idx = item["target_idx"]
        if adv_label_mode == "target_only":
            lbl = torch.full((6,), -1.0)    # sentinel: non-target dims are masked out
        elif adv_label_mode == "ref":
            lbl = torch.from_numpy(np.ascontiguousarray(item["ref_true_label_6"])).float()
        else:
            raise ValueError(f"unknown adv_label_mode: {adv_label_mode}")

        if soft_labels:
            p = float(item["probs_6"][t_idx])
            lbl[t_idx] = max(0.6, p)
        else:
            lbl[t_idx] = 1.0

        # Informativeness: closer to 0.5 = higher score
        prob_t = float(item["probs_6"][t_idx])
        score = 1.0 - 2.0 * abs(prob_t - 0.5)

        buffer.add_one(
            torch.from_numpy(np.ascontiguousarray(augmix_ct_250)).float(),
            lbl,
            score,
        )


# ------------------------------------------------------------------
# Quick-eval stratified subset
# ------------------------------------------------------------------

def build_quick_eval_subset(
    centers: List[str],
    data_dir: str,
    n_per_center: int,
    cache_path: str,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Stratified-by-class-presence subsample (up to n_per_center per center)."""
    if os.path.exists(cache_path):
        if verbose:
            print(f"[quick_eval] cache hit: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        out = {}
        for c in centers:
            if f"{c}__signals" in data.files:
                out[c] = {
                    "signals_tc": data[f"{c}__signals"],
                    "labels_6":   data[f"{c}__labels6"],
                }
        if len(out) == len(centers):
            return out

    import wfdb
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}
    for center in centers:
        center_dir = os.path.join(data_dir, center)
        if not os.path.isdir(center_dir):
            continue
        t0 = time.time()
        paths, snomed_lists = scan_center_records(center_dir)
        if not paths:
            continue
        labels_26 = np.stack([snomed_to_26(s) for s in snomed_lists])
        labels_6 = labels_26[:, TIER_M_IDX].astype(np.float32)

        # Stratified: for each class with positives, ensure ≥ n_per_center/12 records,
        # then fill remainder randomly.
        if len(paths) <= n_per_center:
            selected = list(range(len(paths)))
        else:
            per_class_quota = max(10, n_per_center // 12)
            selected_set = set()
            for cls_i in range(NUM_CLASSES_TIER_M):
                pos = np.where(labels_6[:, cls_i] == 1.0)[0]
                pos = np.array([p for p in pos if int(p) not in selected_set])
                if len(pos) == 0:
                    continue
                n_take = min(per_class_quota, len(pos))
                chosen = rng.choice(pos, size=n_take, replace=False)
                selected_set.update(int(c) for c in chosen)
            deficit = n_per_center - len(selected_set)
            if deficit > 0:
                remaining = np.array([i for i in range(len(paths)) if int(i) not in selected_set])
                if len(remaining) > 0:
                    chosen = rng.choice(remaining, size=min(deficit, len(remaining)), replace=False)
                    selected_set.update(int(c) for c in chosen)
            selected = sorted(selected_set)

        signals = []
        kept_labels = []
        for i in selected:
            try:
                rec = wfdb.rdrecord(paths[i])
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
            kept_labels.append(labels_6[i])
        if not signals:
            continue
        out[center] = {
            "signals_tc": np.stack(signals).astype(np.float32),
            "labels_6":   np.stack(kept_labels).astype(np.float32),
        }
        if verbose:
            print(f"  [quick_eval] {center}: {out[center]['signals_tc'].shape[0]} records "
                  f"({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    dump = {}
    for c, d in out.items():
        dump[f"{c}__signals"] = d["signals_tc"]
        dump[f"{c}__labels6"] = d["labels_6"]
    np.savez_compressed(cache_path, **dump)
    print(f"[quick_eval] cached -> {cache_path}")
    return out


@torch.no_grad()
def quick_eval(victim_model: nn.Module, quick_subset: Dict[str, Dict], device: str,
               crop_len: int = TIERM_INPUT_LENGTH) -> Dict[str, Any]:
    """Forward the stratified subset, compute macro + per-class AUROC/AUPRC per center."""
    victim_model.eval()
    result: Dict[str, Any] = {"per_center": {}}
    all_macro_auroc = []
    all_macro_auprc = []
    for center, data in quick_subset.items():
        signals_tc = data["signals_tc"]    # (N, 1000, 12)
        labels_6 = data["labels_6"]        # (N, 6)
        N = signals_tc.shape[0]

        # (N, 1000, 12) → (N, 12, 1000) → center crop → (N, 12, 250)
        x_ct = np.transpose(signals_tc, (0, 2, 1)).astype(np.float32)  # (N, 12, 1000)
        start = (x_ct.shape[-1] - crop_len) // 2
        x_ct = x_ct[..., start:start + crop_len]
        x_t = torch.from_numpy(x_ct).to(device)

        chunk = 128
        all_logits = []
        for i in range(0, N, chunk):
            lg = victim_model(x_t[i:i + chunk])
            all_logits.append(lg.cpu().numpy())
        logits = np.concatenate(all_logits)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

        aurocs, auprcs, per_class = [], [], {}
        for i, name in enumerate(TIER_M_CLASSES):
            t = labels_6[:, i]
            s = probs[:, i]
            if len(np.unique(t)) < 2:
                per_class[name] = {"auroc": None, "auprc": None, "n_pos": int((t == 1.0).sum())}
                continue
            auc = float(roc_auc_score(t, s))
            ap = float(average_precision_score(t, s))
            per_class[name] = {"auroc": round(auc, 4), "auprc": round(ap, 4),
                               "n_pos": int((t == 1.0).sum())}
            aurocs.append(auc)
            auprcs.append(ap)
        center_macro_auroc = float(np.mean(aurocs)) if aurocs else float('nan')
        center_macro_auprc = float(np.mean(auprcs)) if auprcs else float('nan')
        result["per_center"][center] = {
            "n": N, "macro_auroc": round(center_macro_auroc, 4),
            "macro_auprc": round(center_macro_auprc, 4),
            "per_class": per_class,
        }
        if not np.isnan(center_macro_auroc):
            all_macro_auroc.append(center_macro_auroc)
            all_macro_auprc.append(center_macro_auprc)
    result["avg_macro_auroc"] = round(float(np.mean(all_macro_auroc)), 4) if all_macro_auroc else float('nan')
    result["avg_macro_auprc"] = round(float(np.mean(all_macro_auprc)), 4) if all_macro_auprc else float('nan')
    return result


# ------------------------------------------------------------------
# Quick eval → updated budget
# ------------------------------------------------------------------

def budget_from_quick_eval(quick: Dict[str, Any], n_adv_total: int,
                           cell_skip_threshold: float = 1.1,
                           min_alloc_per_cell: int = 1) -> List[Dict]:
    synthetic = {"centers": {}}
    for c, info in quick["per_center"].items():
        synthetic["centers"][c] = {"per_class": info["per_class"]}
    return build_budget(
        synthetic, n_adv_total, MAIN_CENTERS_4, TIER_M_CLASSES,
        cell_skip_threshold=cell_skip_threshold,
        min_alloc_per_cell=min_alloc_per_cell,
    )


# ------------------------------------------------------------------
# PTBXL training data
# ------------------------------------------------------------------

class PTBXLTierMDataset(Dataset):
    """Returns (sig_ct_250, label_6). Signals are 1000@100Hz pre-processed np arrays;
    label_6 is derived from CSV via the label_alignment_v2 pipeline."""

    def __init__(self, signals_1000, labels_6, crop_len=250, mode='train'):
        self.signals = signals_1000
        self.labels = labels_6
        self.crop_len = crop_len
        self.mode = mode

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig_tc = self.signals[idx]
        crop = crop_signal_tc(sig_tc, self.crop_len,
                              mode='random' if self.mode == 'train' else 'center')
        sig_ct = crop.T
        return (torch.from_numpy(np.ascontiguousarray(sig_ct)).float(),
                torch.from_numpy(self.labels[idx]).float())


def load_ptbxl_tierM_train_val():
    from scripts.crosscenter_v2.label_alignment_v2 import get_ptbxl_26_labels
    train_idx, train_labels_26, _ = get_ptbxl_26_labels(PTBXL_CSV, folds=list(range(1, 9)))
    val_idx,   val_labels_26,   _ = get_ptbxl_26_labels(PTBXL_CSV, folds=[9])
    all_sig = np.load(PTBXL_CACHE, mmap_mode='r')
    train_signals = np.asarray(all_sig[train_idx])
    val_signals   = np.asarray(all_sig[val_idx])
    train_labels_6 = train_labels_26[:, TIER_M_IDX].astype(np.float32)
    val_labels_6   = val_labels_26[:, TIER_M_IDX].astype(np.float32)
    train_ds = PTBXLTierMDataset(train_signals, train_labels_6, crop_len=TIERM_INPUT_LENGTH, mode='train')
    val_ds   = PTBXLTierMDataset(val_signals,   val_labels_6,   crop_len=TIERM_INPUT_LENGTH, mode='eval')
    return train_ds, val_ds, train_labels_6


def build_roundtrip_anchor_dataset(
    train_ds: "PTBXLTierMDataset",
    ecgtwin: ECGTwinWrapper,
    n_samples: int,
    device: str,
    crop_len: int,
    seed: int = 0,
    cache_path: Optional[str] = None,
) -> Optional[TensorDataset]:
    """Pre-compute VAE-roundtrip of a subset of PTBXL train samples with their true
    Tier-M 6-class labels. The resulting TensorDataset teaches the model
    "VAE-projected PTBXL preserves class" so subsequent fine-tuning on VAE-manifold
    adv samples doesn't drift the model off the raw-signal distribution.

    Cached to NPZ; first build ~3-5 min for 1500 samples on RTX 4090.
    """
    if n_samples <= 0:
        return None
    if cache_path and os.path.exists(cache_path):
        print(f"[roundtrip_anchor] cache hit: {cache_path}")
        data = np.load(cache_path)
        return TensorDataset(
            torch.from_numpy(data["signals_ct_250"]).float(),
            torch.from_numpy(data["labels_6"]).float(),
        )

    from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES

    N = min(n_samples, len(train_ds))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_ds), size=N, replace=False)

    dev = torch.device(device)
    batch = 32
    roundtrip_signals = []
    labels_list = []
    t0 = time.time()
    print(f"[roundtrip_anchor] building {N}-sample VAE-roundtrip anchor ...")
    for bi in range(0, N, batch):
        chunk_idx = indices[bi:bi + batch]
        # Load full-1000 signals from the dataset's underlying ndarray
        sigs_tc_1000 = np.stack([train_ds.signals[i] for i in chunk_idx], axis=0).astype(np.float32)  # (B,1000,12)
        lbls = np.stack([train_ds.labels[i] for i in chunk_idx], axis=0).astype(np.float32)          # (B,6)
        B = sigs_tc_1000.shape[0]

        # → (B, 12, 1000) canonical → reorder to ECGTwin → (B, 12, 1024) → (B, 1024, 12)
        sig_ct = np.transpose(sigs_tc_1000, (0, 2, 1))[:, ECGTWIN_TO_PTBXL_INDICES, :]   # (B, 12, 1000)
        sig_ct_t = torch.from_numpy(sig_ct).float().to(dev)
        sig_ct_t = F.interpolate(sig_ct_t, size=1024, mode="linear", align_corners=True)
        sig_lc_t = sig_ct_t.transpose(1, 2).contiguous()
        with torch.no_grad():
            latent = ecgtwin.encode_ecg(sig_lc_t)                                        # (B, 4, 128)
            rt_lc = ecgtwin.decode_latent(latent)                                        # (B, 1024, 12)
        rt_ct = rt_lc.transpose(1, 2)                                                    # (B, 12, 1024)
        rt_ct = rt_ct[:, ECGTWIN_TO_PTBXL_INDICES, :]                                    # back to canonical
        rt_ct = torch.clamp(rt_ct, min=-3.0, max=3.0)
        rt_ct = F.interpolate(rt_ct, size=1000, mode="linear", align_corners=True)       # (B, 12, 1000)
        # Global per-sample zscore (match preprocess_utils.per_sample_zscore)
        flat = rt_ct.reshape(B, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True).clamp(min=1e-8)
        rt_ct = (rt_ct - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        # Center crop to 250 to match training/eval input shape
        start = (1000 - crop_len) // 2
        rt_ct_250 = rt_ct[..., start:start + crop_len]                                   # (B, 12, 250)
        roundtrip_signals.append(rt_ct_250.cpu())
        labels_list.append(torch.from_numpy(lbls))

    sigs_tensor = torch.cat(roundtrip_signals, dim=0).float()   # (N, 12, 250)
    labels_tensor = torch.cat(labels_list, dim=0).float()       # (N, 6)
    print(f"[roundtrip_anchor] done in {time.time()-t0:.0f}s. shape={tuple(sigs_tensor.shape)}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path,
                            signals_ct_250=sigs_tensor.numpy(),
                            labels_6=labels_tensor.numpy())
        print(f"[roundtrip_anchor] cached -> {cache_path}")
    return TensorDataset(sigs_tensor, labels_tensor)


def compute_pos_weight_tierM(labels_6: np.ndarray, clip_max: float = 50.0) -> torch.Tensor:
    n = labels_6.shape[0]
    pw = np.zeros(NUM_CLASSES_TIER_M, dtype=np.float32)
    for i in range(NUM_CLASSES_TIER_M):
        n_pos = int((labels_6[:, i] == 1.0).sum())
        n_neg = n - n_pos
        pw[i] = min(max(n_neg / max(n_pos, 1), 1.0), clip_max)
    return torch.from_numpy(pw)


# ------------------------------------------------------------------
# Training / validation
# ------------------------------------------------------------------

def train_one_epoch_tierM(
    model: nn.Module, loader: DataLoader, optimizer, criterion, device,
    grad_clip: float, ewa_params: List[torch.Tensor], anchor_lambda: float, ewa_decay: float,
) -> float:
    """BCE with per-element mask for label == -1 sentinel (adv target_only mode).
    `criterion` must be nn.BCEWithLogitsLoss(pos_weight=..., reduction='none').
    """
    model.train()
    losses = []
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)                         # (B, 6)
        mask = (labels >= 0).float()                    # (B, 6)
        labels_clamp = labels.clamp(min=0.0)            # turn -1 → 0 for BCE math
        per_elem = criterion(logits, labels_clamp)      # (B, 6), reduction='none'
        denom = mask.sum().clamp(min=1.0)
        bce = (per_elem * mask).sum() / denom           # mean over unmasked entries
        if ewa_params is not None and anchor_lambda > 0:
            anchor = sum(
                (p - p_anchor.detach()).pow(2).sum()
                for p, p_anchor in zip(model.parameters(), ewa_params)
            )
            loss = bce + anchor_lambda * anchor
        else:
            loss = bce
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ewa_params is not None and ewa_decay > 0 and ewa_decay < 1.0:
            with torch.no_grad():
                for p, p_anchor in zip(model.parameters(), ewa_params):
                    p_anchor.mul_(ewa_decay).add_(p.data, alpha=1 - ewa_decay)
        losses.append(bce.item())
    return float(np.mean(losses)) if losses else float('nan')


# ------------------------------------------------------------------
# Visualization (for self-reflection)
# ------------------------------------------------------------------

def save_adv_viz(adv_items: List[Dict], augmix_subset: List[torch.Tensor], save_path: str,
                 n_examples: int = 4):
    if not adv_items or not augmix_subset:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(n_examples, len(adv_items), len(augmix_subset))
    fig, axes = plt.subplots(n, 2, figsize=(16, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for i in range(n):
        item = adv_items[i]
        ref = item["ref_signal_ct_1000_canonical"]      # (12, 1000)
        aug = augmix_subset[i].numpy() if torch.is_tensor(augmix_subset[i]) else augmix_subset[i]  # (12, 250)
        t_ref = np.arange(1000) / 100.0
        t_aug = np.arange(aug.shape[-1]) / 100.0
        ax0, ax1 = axes[i, 0], axes[i, 1]
        for lead in range(12):
            ax0.plot(t_ref, ref[lead] + lead * 2.0, linewidth=0.5, color='black')
            ax1.plot(t_aug, aug[lead] + lead * 2.0, linewidth=0.5, color='red')
        p = float(item['probs_6'][item['target_idx']])
        ax0.set_title(f"REF  [{item['center']}]  target={item['target_class']}", fontsize=8)
        ax1.set_title(f"AUGMIX-ADV  prob={p:.3f}  (target_class={item['target_class']})", fontsize=8)
        for ax in (ax0, ax1):
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--init_ckpt", default=DEFAULT_TIERM_CKPT)
    p.add_argument("--baseline_eval_json", default=BASELINE_EVAL_JSON)
    p.add_argument("--data_dir", default=DEFAULT_PN2021_DIR)
    p.add_argument("--centers", nargs='+', default=MAIN_CENTERS_4)
    p.add_argument("--n_epochs", type=int, default=25)
    p.add_argument("--n_adv_per_epoch", type=int, default=128)
    p.add_argument("--k_per_center", type=int, default=32)
    p.add_argument("--accept_prob_low", type=float, default=0.5)
    p.add_argument("--accept_prob_high", type=float, default=0.6)
    p.add_argument("--augmix_width", type=int, default=3)       # 1 adv + 2 time-domain
    p.add_argument("--augmix_severity", type=int, default=5)
    p.add_argument("--qab_size", type=int, default=2048)
    p.add_argument("--anchor_lambda", type=float, default=0.05)
    p.add_argument("--ewa_decay", type=float, default=0.999)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=3)
    p.add_argument("--rescore_interval", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--soft_labels", action="store_true",
                   help="Use max(0.6, prob) for target class (instead of hard 1.0). "
                        "Experiments show hard labels work better due to VAE manifold effects.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--quick_eval_n_per_center", type=int, default=1000)
    p.add_argument("--best_of_k", type=int, default=5)
    # Roundtrip-anchor: adds VAE-roundtripped PTBXL samples with true labels to
    # teach the model "VAE-projected PTBXL still has its original class". Addresses
    # the VAE-manifold distribution shift that hurt the v1 run.
    p.add_argument("--roundtrip_anchor_n", type=int, default=1500,
                   help="Number of PTBXL samples to pre-roundtrip-encode for manifold anchor. "
                        "0 disables.")
    p.add_argument("--roundtrip_weight", type=float, default=0.5,
                   help="Sampling weight for roundtrip anchor samples relative to real PTBXL=1.0")
    p.add_argument("--cell_skip_threshold", type=float, default=1.1,
                   help="Skip (center, class) cells whose baseline AUROC >= this. "
                        "Default 1.1 disables. Set e.g. 0.95 to focus adv-gen on weak cells only.")
    p.add_argument("--min_alloc_per_cell", type=int, default=1,
                   help="Min n_alloc per valid budget cell. Set 0 to disable the floor "
                        "and let weight-proportion pure-allocate.")
    p.add_argument("--adv_label_mode", choices=["ref", "target_only"], default="target_only",
                   help="'ref' = keep ref true multi-label + target=1 (v1-v4); "
                        "'target_only' = only target class gets label, other 5 masked (-1)")
    p.add_argument("--freeze_early_stages", type=int, default=0,
                   help="Number of initial_conv + features stages to freeze "
                        "(0 = train all, 5 = freeze initial_conv + features[:5] out of 7 stages)")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 72)
    print("Tier-M 在线对抗训练 (online_adv_train_tierM)")
    print("=" * 72)
    print(f"  init_ckpt:         {args.init_ckpt}")
    print(f"  output_dir:        {args.output_dir}")
    print(f"  n_epochs:          {args.n_epochs}")
    print(f"  n_adv/epoch:       {args.n_adv_per_epoch}")
    print(f"  k_per_center:      {args.k_per_center}")
    print(f"  accept_prob:       [{args.accept_prob_low}, {args.accept_prob_high}]")
    print(f"  augmix W:          {args.augmix_width}  severity={args.augmix_severity}")
    print(f"  qab_size:          {args.qab_size}")
    print(f"  anchor_lambda:     {args.anchor_lambda}")
    print(f"  lr:                {args.lr}")
    print(f"  soft_labels:       {args.soft_labels}")
    print("-" * 72)

    # ── Load ECGTwin (encoder + text model) ─────────────────────────────────
    print("[setup] Loading ECGTwin (encoder + text model)...")
    ecgtwin = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=True)

    # ── Load / build Tier-M victim ──────────────────────────────────────────
    print("[setup] Loading Tier-M victim...")
    victim = EfficientNetVictimTierM(
        weight_path=args.init_ckpt,
        device=args.device,
        ecgtwin_wrapper=ecgtwin,
    )

    # Optionally freeze early backbone layers
    if args.freeze_early_stages > 0:
        n_frozen_params = 0
        for p in victim.model.initial_conv.parameters():
            p.requires_grad = False
            n_frozen_params += p.numel()
        blocks = list(victim.model.features)
        n_to_freeze_blocks = min(args.freeze_early_stages, len(blocks))
        for blk in blocks[:n_to_freeze_blocks]:
            for p in blk.parameters():
                p.requires_grad = False
                n_frozen_params += p.numel()
        n_total = sum(p.numel() for p in victim.model.parameters())
        n_trainable = sum(p.numel() for p in victim.model.parameters() if p.requires_grad)
        print(f"[freeze] initial_conv + first {n_to_freeze_blocks}/{len(blocks)} features blocks frozen: "
              f"{n_frozen_params} / {n_total} ({100*n_frozen_params/n_total:.1f}%) "
              f"→ trainable = {n_trainable} ({100*n_trainable/n_total:.1f}%)")
    # EWA anchor snapshot of TRAINABLE params only (frozen params never move, no anchor needed)
    ewa_params = [p.data.clone().detach() for p in victim.model.parameters() if p.requires_grad]

    # ── Refs ────────────────────────────────────────────────────────────────
    refs_cache = os.path.join(args.output_dir, f"refs_cache_k{args.k_per_center}.npz")
    refs = sample_refs_per_center(
        k_per_center=args.k_per_center,
        centers=args.centers,
        data_dir=args.data_dir,
        cache_path=refs_cache,
        seed=args.seed,
        verbose=True,
    )
    print(f"[setup] Encoding refs → VAE latents...")
    ref_items_by_center = encode_refs_per_center(refs, ecgtwin, args.device)

    # ── Quick-eval stratified subset ────────────────────────────────────────
    qe_cache = os.path.join(args.output_dir,
                            f"quick_eval_subset_n{args.quick_eval_n_per_center}.npz")
    quick_subset = build_quick_eval_subset(
        centers=args.centers, data_dir=args.data_dir,
        n_per_center=args.quick_eval_n_per_center, cache_path=qe_cache,
        seed=args.seed,
    )
    baseline_quick = quick_eval(victim.model, quick_subset, args.device)
    print(f"[baseline] quick-eval avg macro AUROC={baseline_quick['avg_macro_auroc']}, "
          f"AUPRC={baseline_quick['avg_macro_auprc']}")
    for c, info in baseline_quick["per_center"].items():
        print(f"    {c}: AUROC={info['macro_auroc']}  AUPRC={info['macro_auprc']}  (n={info['n']})")

    # ── Baseline budget ─────────────────────────────────────────────────────
    with open(args.baseline_eval_json) as f:
        baseline_eval = json.load(f)
    budget = build_budget(
        eval_json=baseline_eval, n_adv_total=args.n_adv_per_epoch,
        centers=args.centers, tier_m_classes=TIER_M_CLASSES,
        cell_skip_threshold=args.cell_skip_threshold,
        min_alloc_per_cell=args.min_alloc_per_cell,
    )
    print(f"[budget] initial (from baseline eval):")
    for e in budget:
        print(f"    {e['center']:<22} × {e['class']:<6} "
              f"AUROC={e['baseline_auroc']:.3f} n_alloc={e['n_alloc']}")

    # ── PTBXL train / val ───────────────────────────────────────────────────
    print("[setup] Loading PTBXL train/val (via cache)...")
    train_ds, val_ds, train_labels_6 = load_ptbxl_tierM_train_val()
    pos_weight = compute_pos_weight_tierM(train_labels_6).to(args.device)
    # reduction='none' needed for per-element masking of label==-1 sentinel in adv samples
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"  PTBXL train n={len(train_ds)}, val n={len(val_ds)}")
    print(f"  pos_weight={pos_weight.cpu().tolist()}")

    # ── VAE-roundtrip PTBXL anchor (teaches model that VAE projection preserves class) ─
    roundtrip_ds: Optional[TensorDataset] = None
    if args.roundtrip_anchor_n > 0:
        rt_cache = os.path.join(args.output_dir,
                                f"roundtrip_anchor_n{args.roundtrip_anchor_n}.npz")
        roundtrip_ds = build_roundtrip_anchor_dataset(
            train_ds=train_ds, ecgtwin=ecgtwin, n_samples=args.roundtrip_anchor_n,
            device=args.device, crop_len=TIERM_INPUT_LENGTH,
            seed=args.seed, cache_path=rt_cache,
        )
        print(f"[setup] roundtrip-anchor ready: n={len(roundtrip_ds)}, weight={args.roundtrip_weight}")

    # ── Optimizer / scheduler ───────────────────────────────────────────────
    # Filter out frozen params (requires_grad=False) to avoid optimizer state overhead
    trainable_params = [p for p in victim.model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=args.lr * 0.01)

    # ── Adv generator + QAB ─────────────────────────────────────────────────
    gen_hp = dict(TIERM_GEN_HYPERPARAMS_DEFAULT)
    gen_hp["acceptance_range"] = [args.accept_prob_low, args.accept_prob_high]
    gen_hp["batch_size"] = args.gen_batch_size
    generator = BoundaryAdvDiffGenerator(
        ecgtwin_wrapper=ecgtwin, victim=victim, device=args.device, hyperparams=gen_hp,
    )
    buffer = QualityAwareBufferTierM(max_size=args.qab_size)
    z_hist: Dict[Tuple[str, str], torch.Tensor] = {}

    # ── Training log bookkeeping ────────────────────────────────────────────
    log: Dict[str, Any] = {
        "args": vars(args),
        "baseline_quick_eval": baseline_quick,
        "epochs": [],
    }
    best_avg_auroc = baseline_quick["avg_macro_auroc"]
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    # Save the init ckpt as "best" bootstrap so early quick evals always compare fairly
    torch.save(victim.model.state_dict(), best_ckpt_path)
    log_path = os.path.join(args.output_dir, "training_log.json")

    # ── Main loop ───────────────────────────────────────────────────────────
    for epoch in range(1, args.n_epochs + 1):
        epoch_t0 = time.time()

        # Phase A: adv generation (victim in eval during generation)
        victim.model.eval()
        adv_items, gen_stats = generate_adv_for_epoch(
            generator=generator, ecgtwin=ecgtwin, budget=budget,
            ref_items_by_center=ref_items_by_center, z_hist=z_hist,
            device=args.device, best_of_k=args.best_of_k,
        )

        # Phase B: augmix-inject + fill buffer
        augmix_inject_and_buffer(
            adv_items=adv_items, ecgtwin_encoder_wrapper=ecgtwin,
            buffer=buffer, augmix_width=args.augmix_width,
            augmix_severity=args.augmix_severity, soft_labels=args.soft_labels,
            device=args.device, adv_label_mode=args.adv_label_mode,
        )

        if epoch > 1 and (epoch - 1) % args.rescore_interval == 0:
            buffer.rescore(victim.model, args.device)

        # Phase C: build mixed loader
        # Streams: (A) real PTBXL, (B) VAE-roundtrip PTBXL anchor, (C) adv buffer
        buf_ds = buffer.to_dataset()
        streams: List[Tuple[Dataset, float, Optional[List[float]]]] = [
            (train_ds, 1.0, None),
        ]
        if roundtrip_ds is not None:
            streams.append((roundtrip_ds, args.roundtrip_weight, None))
        if buf_ds is not None and len(buf_ds) > 0:
            streams.append((buf_ds, 2.0, buffer.get_sampling_weights()))

        if len(streams) == 1:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True, drop_last=True)
        else:
            weights: List[float] = []
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
                                      num_workers=args.num_workers, pin_memory=True, drop_last=True)

        # Phase D: train
        train_loss = train_one_epoch_tierM(
            victim.model, train_loader, optimizer, criterion, args.device,
            grad_clip=1.0, ewa_params=ewa_params,
            anchor_lambda=args.anchor_lambda, ewa_decay=args.ewa_decay,
        )
        scheduler.step()

        # Phase E: val loss (on PTBXL val to monitor source performance)
        victim.model.eval()
        val_losses = []
        with torch.no_grad():
            for sigs, labels in val_loader:
                sigs = sigs.to(args.device)
                labels = labels.to(args.device)
                # Val set is real PTBXL (no -1 sentinels) so mask is all-ones → just mean
                vl = criterion(victim.model(sigs), labels).mean()
                val_losses.append(vl.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

        elapsed = time.time() - epoch_t0
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "n_adv_attempted": gen_stats["attempted"],
            "n_adv_accepted":  gen_stats["accepted"],
            "adv_accept_rate": round(gen_stats["accepted"] / max(1, gen_stats["attempted"]), 3),
            "buffer_size": len(buffer),
            "budget_size": sum(e["n_alloc"] for e in budget),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "time_s": round(elapsed, 1),
        }
        print(f"Ep {epoch:2d}/{args.n_epochs} | train={train_loss:.4f} val={val_loss:.4f} | "
              f"adv {gen_stats['accepted']}/{gen_stats['attempted']} "
              f"(rate={entry['adv_accept_rate']:.2f}) | buf={len(buffer)} | "
              f"{elapsed:.0f}s")

        # Phase F: quick eval (every eval_every; also last epoch)
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.n_epochs)
        if do_eval:
            qe = quick_eval(victim.model, quick_subset, args.device)
            entry["quick_eval"] = qe
            print(f"   quick eval: avg AUROC={qe['avg_macro_auroc']}  "
                  f"AUPRC={qe['avg_macro_auprc']}")
            for c, info in qe["per_center"].items():
                print(f"      {c}: AUROC={info['macro_auroc']}  AUPRC={info['macro_auprc']}")
            if qe["avg_macro_auroc"] > best_avg_auroc:
                best_avg_auroc = qe["avg_macro_auroc"]
                torch.save(victim.model.state_dict(), best_ckpt_path)
                print(f"   ** saved best @ ep{epoch}: avg AUROC {best_avg_auroc}")
                entry["best_update"] = True
            # Refresh budget from latest quick eval
            try:
                budget = budget_from_quick_eval(
                    qe, args.n_adv_per_epoch,
                    cell_skip_threshold=args.cell_skip_threshold,
                    min_alloc_per_cell=args.min_alloc_per_cell,
                )
            except Exception as e:
                print(f"   [warn] budget refresh failed: {e}")

            # Visualization: save first 4 augmix-adv ECGs from this epoch
            if adv_items and len(buffer) > 0:
                try:
                    # recover last-added augmix samples
                    recent = buffer.ecg_list[-min(4, len(buffer)):]
                    save_adv_viz(
                        adv_items[:4], recent,
                        save_path=os.path.join(args.output_dir, f"sample_adv_epoch_{epoch}.png"),
                    )
                except Exception as e:
                    print(f"   [warn] viz failed: {e}")

        log["epochs"].append(entry)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2, default=str)

    # ── Final train_result.json ─────────────────────────────────────────────
    final_result = {
        "args": vars(args),
        "baseline_quick_eval": baseline_quick,
        "best_avg_macro_auroc": best_avg_auroc,
        "n_epochs_run": len(log["epochs"]),
        "last_epoch_quick_eval": log["epochs"][-1].get("quick_eval") if log["epochs"] else None,
    }
    with open(os.path.join(args.output_dir, "train_result.json"), "w") as f:
        json.dump(final_result, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print(f"Training done. best_avg_macro_auroc={best_avg_auroc} ({best_ckpt_path})")
    print(f"Run full eval separately:")
    print(f"  /root/miniforge3/envs/ECGTwin/bin/python \\")
    print(f"    scripts/crosscenter_tierM/eval_crosscenter_tierM.py \\")
    print(f"    --model_dir {args.output_dir}")


if __name__ == "__main__":
    main()
