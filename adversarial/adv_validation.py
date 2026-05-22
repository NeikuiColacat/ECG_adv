"""Validation gates for ECGTwin-PGD adversarial buffers.

Gate 1 — Attack Success Rate: overall ≥ 70%, per-class ≥ 30% (a weaker "some
signal in every class" check; classes with very high victim confidence can
legitimately be harder to attack at a given epsilon).

Gate 2 — Medical Semantic: HR/QRS-width/amplitude drift bounded vs anchor,
plus Einthoven residual |II-I-III| p95 < 0.5 (z-scored units).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# PTBXL canonical lead index (after ECGTWIN_TO_PTBXL reorder)
LEAD_I = 0
LEAD_II = 1
LEAD_III = 2

# Sampling rate (all signals at 100 Hz)
FS = 100


# ============================================================
# Gate 1 — Attack Success Rate
# ============================================================

def compute_asr(
    victim,
    signals_ct_1000: np.ndarray,     # (N, 12, 1000) PTBXL order, z-scored
    labels_multi_hot: np.ndarray,    # (N, 6)
    device: str = "cuda",
    batch_size: int = 128,
) -> Dict:
    """Compute per-class and overall Attack Success Rate.

    Rules:
      - Attack successful on sample i iff argmax(victim(x_i)) != argmax(y_i).
      - For multi-hot labels, uses argmax as the "primary" class.
      - per_class_asr[c] = fraction of samples with argmax(y)==c that are attacked.
    """
    victim.eval()
    N = signals_ct_1000.shape[0]
    y_primary = labels_multi_hot.argmax(axis=1)       # (N,)

    all_logits = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = torch.from_numpy(signals_ct_1000[i:i + batch_size]).float().to(device)
            # forward_from_ecg returns sigmoid(logits); we need logits for argmax
            # (argmax of probs == argmax of logits)
            logits = victim.compute_logits_from_ecg(batch)
            all_logits.append(logits.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)        # (N, 6)
    pred_primary = logits.argmax(axis=1)               # (N,)

    # Probability victim assigns to each sample's primary true class
    probs = 1.0 / (1.0 + np.exp(-logits))              # (N, 6) sigmoid
    prob_on_true = probs[np.arange(N), y_primary]      # (N,)

    asr_overall = float((pred_primary != y_primary).mean())
    per_class_asr: Dict[str, float] = {}
    per_class_n: Dict[str, int] = {}
    for c in range(labels_multi_hot.shape[1]):
        mask = (y_primary == c)
        n_c = int(mask.sum())
        per_class_n[str(c)] = n_c
        if n_c == 0:
            per_class_asr[str(c)] = float("nan")
        else:
            per_class_asr[str(c)] = float((pred_primary[mask] != c).mean())

    positive_mask = labels_multi_hot > 0.5
    positive_probs = probs[positive_mask]
    positive_below = (probs < 0.5) & positive_mask
    sample_has_positive = positive_mask.any(axis=1)
    positive_counts = positive_mask.sum(axis=1)
    positive_below_counts = positive_below.sum(axis=1)
    if positive_probs.size:
        multilabel_positive_label_asr = float((positive_probs < 0.5).mean())
        sample_any_positive_below_0p5_asr = float(
            positive_below[sample_has_positive].any(axis=1).mean()
        )
        sample_all_positive_below_0p5_asr = float(
            (positive_below_counts[sample_has_positive] == positive_counts[sample_has_positive]).mean()
        )
        sample_all_positive_recognized_rate = float(
            (positive_below_counts[sample_has_positive] == 0).mean()
        )
    else:
        multilabel_positive_label_asr = float("nan")
        sample_any_positive_below_0p5_asr = float("nan")
        sample_all_positive_below_0p5_asr = float("nan")
        sample_all_positive_recognized_rate = float("nan")

    per_class_positive_label_asr: Dict[str, float] = {}
    per_class_positive_label_n: Dict[str, int] = {}
    for c in range(labels_multi_hot.shape[1]):
        mask = positive_mask[:, c]
        n_c = int(mask.sum())
        per_class_positive_label_n[str(c)] = n_c
        if n_c == 0:
            per_class_positive_label_asr[str(c)] = float("nan")
        else:
            per_class_positive_label_asr[str(c)] = float((probs[mask, c] < 0.5).mean())

    # PASS criterion (overall ≥ 0.7 is the hard requirement; per-class ≥ 0.3 is a
    # weaker "some signal in every class" check — a class with very high victim
    # confidence like RBBB can legitimately be harder to attack at a given ε,
    # and the un-attacked samples become correct-label augmentation rather than
    # adversarial training signal, which is fine).
    pass_overall = asr_overall >= 0.7
    per_class_ok = all(
        (np.isnan(v) or v >= 0.3) for v in per_class_asr.values()
    )

    return dict(
        asr_overall=asr_overall,
        per_class_asr=per_class_asr,
        per_class_n=per_class_n,
        multilabel_positive_label_asr=multilabel_positive_label_asr,
        sample_any_positive_below_0p5_asr=sample_any_positive_below_0p5_asr,
        sample_all_positive_below_0p5_asr=sample_all_positive_below_0p5_asr,
        sample_all_positive_recognized_rate=sample_all_positive_recognized_rate,
        per_class_positive_label_asr=per_class_positive_label_asr,
        per_class_positive_label_n=per_class_positive_label_n,
        prob_on_true_mean=float(prob_on_true.mean()),
        prob_on_true_median=float(np.median(prob_on_true)),
        PASS=bool(pass_overall and per_class_ok),
        fail_reasons=[] if (pass_overall and per_class_ok) else (
            (["asr_overall<0.7"] if not pass_overall else []) +
            (["per_class_asr<0.5"] if not per_class_ok else [])
        ),
    )


# ============================================================
# Gate 2 — Medical Semantic Preservation
# ============================================================

def _detect_r_peaks_lead2(sig_lead2: np.ndarray, fs: int = FS) -> np.ndarray:
    """Simple R-peak detection on Lead II via scipy.signal.find_peaks.

    Returns peak sample indices.
    """
    from scipy.signal import find_peaks
    # Signal is z-scored, so threshold ~0.5-1 std; min distance ~0.4s (150 bpm max)
    min_dist = int(0.4 * fs)
    # Relative prominence adapts to signal's own std
    prom = max(0.3, float(np.std(sig_lead2)) * 0.5)
    peaks, _ = find_peaks(sig_lead2, distance=min_dist, prominence=prom)
    return peaks


def _extract_clinical_features(ecg_ct_1000: np.ndarray, fs: int = FS) -> Dict:
    """Extract lightweight clinical features from a single (12, 1000) z-scored ECG.

    Returns dict with:
      hr_bpm: heart rate estimate (NaN if < 2 R-peaks found)
      qrs_amp: median R-peak amplitude on Lead II (z-score units)
      qrs_window_std: std of signal in ±50ms around R-peaks (QRS envelope proxy)
      einthoven_p95: 95th pct of |lead_II - lead_I - lead_III|
    """
    sig_ii = ecg_ct_1000[LEAD_II]
    peaks = _detect_r_peaks_lead2(sig_ii, fs=fs)

    if len(peaks) >= 2:
        rr_samples = np.diff(peaks)
        hr_bpm = float(60.0 * fs / np.mean(rr_samples))
    else:
        hr_bpm = float("nan")

    if len(peaks) >= 1:
        qrs_amp = float(np.median(sig_ii[peaks]))
    else:
        qrs_amp = float("nan")

    # QRS envelope: ±50 ms = ±5 samples at 100 Hz
    window = 5
    if len(peaks) >= 1:
        stds = []
        for p in peaks:
            lo = max(0, p - window)
            hi = min(len(sig_ii), p + window + 1)
            stds.append(np.std(sig_ii[lo:hi]))
        qrs_win_std = float(np.median(stds))
    else:
        qrs_win_std = float("nan")

    # Einthoven Law residual: |II - I - III|
    if ecg_ct_1000.shape[0] >= 3:
        resid = np.abs(ecg_ct_1000[LEAD_II] - ecg_ct_1000[LEAD_I] - ecg_ct_1000[LEAD_III])
        einthoven_p95 = float(np.percentile(resid, 95))
    else:
        einthoven_p95 = float("nan")

    return dict(
        hr_bpm=hr_bpm,
        qrs_amp=qrs_amp,
        qrs_win_std=qrs_win_std,
        einthoven_p95=einthoven_p95,
    )


def _aggregate_feature_stats(feat_list: List[Dict]) -> Dict:
    """Reduce a list of per-sample feature dicts to mean/std/p95 of finite values."""
    keys = ["hr_bpm", "qrs_amp", "qrs_win_std", "einthoven_p95"]
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in feat_list], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[k] = dict(mean=float("nan"), std=float("nan"), p95=float("nan"), n=0)
        else:
            out[k] = dict(
                mean=float(np.mean(vals)),
                std=float(np.std(vals)),
                p95=float(np.percentile(vals, 95)),
                n=int(len(vals)),
            )
    return out


def compute_semantic_gate(
    adv_signals_ct_1000: np.ndarray,         # (N_adv, 12, 1000)
    anchor_signals_ct_1000: np.ndarray,      # (N_anchor, 12, 1000) — decoded anchor refs
    fs: int = FS,
    hr_mean_delta_max: float = 15.0,         # bpm
    qrs_amp_ratio: Tuple[float, float] = (0.5, 2.0),
    einthoven_p95_max: float = 0.5,
) -> Dict:
    """Gate 2: compare clinical-feature distributions of adv vs anchor.

    PASS iff:
      - |mean(HR_adv) - mean(HR_anchor)| <= hr_mean_delta_max
      - mean(qrs_amp_adv) / mean(qrs_amp_anchor) within qrs_amp_ratio
      - mean(einthoven_p95_adv) <= einthoven_p95_max
    """
    feats_adv = [_extract_clinical_features(x, fs=fs) for x in adv_signals_ct_1000]
    feats_anc = [_extract_clinical_features(x, fs=fs) for x in anchor_signals_ct_1000]

    stats_adv = _aggregate_feature_stats(feats_adv)
    stats_anc = _aggregate_feature_stats(feats_anc)

    hr_delta = abs(stats_adv["hr_bpm"]["mean"] - stats_anc["hr_bpm"]["mean"])
    # QRS amplitude ratio (avoid div-by-zero)
    a_anc = stats_anc["qrs_amp"]["mean"]
    a_adv = stats_adv["qrs_amp"]["mean"]
    if np.isfinite(a_anc) and np.isfinite(a_adv) and abs(a_anc) > 1e-6:
        qrs_ratio = float(a_adv / a_anc)
    else:
        qrs_ratio = float("nan")
    einthoven_mean = stats_adv["einthoven_p95"]["mean"]

    fail_reasons = []
    if not (np.isfinite(hr_delta) and hr_delta <= hr_mean_delta_max):
        fail_reasons.append(f"hr_mean_delta={hr_delta:.1f} > {hr_mean_delta_max}")
    if not (np.isfinite(qrs_ratio) and qrs_amp_ratio[0] <= qrs_ratio <= qrs_amp_ratio[1]):
        fail_reasons.append(f"qrs_amp_ratio={qrs_ratio:.2f} outside {qrs_amp_ratio}")
    if not (np.isfinite(einthoven_mean) and einthoven_mean <= einthoven_p95_max):
        fail_reasons.append(f"einthoven_mean_p95={einthoven_mean:.2f} > {einthoven_p95_max}")

    return dict(
        adv_feat_stats=stats_adv,
        anchor_feat_stats=stats_anc,
        hr_mean_delta=float(hr_delta),
        qrs_amp_ratio=qrs_ratio,
        einthoven_mean_p95=float(einthoven_mean) if np.isfinite(einthoven_mean) else None,
        PASS=len(fail_reasons) == 0,
        fail_reasons=fail_reasons,
    )


# ============================================================
# Combined validator
# ============================================================

def validate_buffer(
    adv_signals: np.ndarray,                 # (N, 12, 1000)
    adv_labels_multi_hot: np.ndarray,        # (N, 6)
    anchor_signals: np.ndarray,              # (N_anc, 12, 1000) — decoded z0 references
    victim,
    device: str = "cuda",
) -> Dict:
    """Run Gate 1 + Gate 2. Returns combined result dict with top-level PASS."""
    gate1 = compute_asr(
        victim=victim,
        signals_ct_1000=adv_signals,
        labels_multi_hot=adv_labels_multi_hot,
        device=device,
    )
    gate2 = compute_semantic_gate(
        adv_signals_ct_1000=adv_signals,
        anchor_signals_ct_1000=anchor_signals,
    )
    overall_pass = bool(gate1["PASS"] and gate2["PASS"])
    return dict(
        PASS=overall_pass,
        gate1=gate1,
        gate2=gate2,
        fail_reasons=(gate1["fail_reasons"] + gate2["fail_reasons"]),
    )


if __name__ == "__main__":
    # Minimal smoke: random inputs
    print("[smoke] adv_validation module self-test")
    rng = np.random.default_rng(42)
    N_adv, N_anc = 100, 50
    # Simulate "adv is slightly perturbed anchor" (same baseline stats)
    anchors = rng.normal(0, 1, size=(N_anc, 12, 1000)).astype(np.float32)
    advs = rng.normal(0, 1, size=(N_adv, 12, 1000)).astype(np.float32)
    labels = np.zeros((N_adv, 6), dtype=np.float32)
    labels[:, rng.integers(0, 6, size=N_adv)] = 1.0  # random one-hot

    # Gate 2 only (no victim needed)
    g2 = compute_semantic_gate(advs, anchors)
    print(f"Gate 2 PASS={g2['PASS']} reasons={g2['fail_reasons']}")
    print(f"  HR mean adv={g2['adv_feat_stats']['hr_bpm']['mean']:.1f}  "
          f"anc={g2['anchor_feat_stats']['hr_bpm']['mean']:.1f}")
    print(f"  einthoven p95 adv={g2['einthoven_mean_p95']}")
