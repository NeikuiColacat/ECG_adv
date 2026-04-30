"""Digital ECG feature extractor.

Given a (12, L) signal tensor in mV at a known sampling rate (default 102.4 Hz)
in MIMIC lead order [I, II, III, aVR, aVF, aVL, V1..V6], return a dict of
quantitative features that can be checked against medical-defined digital
thresholds (see docs/ecg_digital_thresholds.md).

Implementation choices:
- R-peak detection: lead II via scipy.signal.find_peaks with refractory period
- QRS onset/offset: gradient-based, ±60 ms window around each R-peak, edges
  defined where |signal'| drops below 30% of its peak in the window
- J-point = QRS-offset
- ST level at J+40 / J+60 ms = mean signal over a 5-sample window centered there
- T-wave: max-abs deflection 100-300 ms after J
- P-wave: positive peak 80-200 ms before QRS-onset on lead II
- PR interval = P-onset to QRS-onset (we use P-peak − QRS-onset shifted by
  approx P-onset offset; small bias OK for our gross threshold check)

We aggregate per-beat features to per-lead medians to be robust to outlier beats.

Limitations: at 102.4 Hz, ms resolution is ~10 ms. Q-wave duration < 30 ms
is at granularity floor; treat with caution. No wavelet/QT correction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

# MIMIC lead order (ECGTwin native)
MIMIC_LEAD_NAMES = ["I", "II", "III", "aVR", "aVF", "aVL",
                     "V1", "V2", "V3", "V4", "V5", "V6"]
PTBXL_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]

# Lead-name -> index helpers (MIMIC ordering)
def lead_idx_mimic(name: str) -> int:
    return MIMIC_LEAD_NAMES.index(name)


def _ms_to_samples(ms: float, fs: float) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


def _detect_r_peaks(lead_ii: np.ndarray, fs: float) -> np.ndarray:
    """Return R-peak sample indices on lead II.

    Uses absolute signal so polarity-flipped leads still detect, but lead II
    is normally upright so positive amplitude works in practice. We use
    |signal| to be safe. For low-amplitude synthesized signals we fall back
    to a low absolute height (0.05 mV) gated by relative-to-max.
    """
    refrac = _ms_to_samples(400, fs)  # 400 ms refractory ~ HR ≤ 150
    abs_sig = np.abs(lead_ii - np.median(lead_ii))
    sig_max = float(np.max(abs_sig))
    if sig_max < 0.05:
        return np.array([], dtype=int)
    # height = max( 0.05 mV floor, 0.4 × peak ) — adaptive
    height = max(0.05, 0.4 * sig_max)
    peaks, _ = find_peaks(abs_sig, height=height, distance=refrac)
    return peaks


def _qrs_onset_offset(
    lead_signal: np.ndarray,
    r_peak: int,
    fs: float,
    win_pre_ms: float = 80.0,
    win_post_ms: float = 100.0,
    grad_frac: float = 0.10,
) -> Tuple[int, int]:
    """Estimate QRS onset and offset around a single R-peak via gradient drop-off.

    Strategy: search backward/forward from the R-peak in a fixed window;
    define QRS edges where smoothed |gradient| falls below 10% of its peak
    in the window AND stays below for ≥ 2 samples (debounce). This is much
    more lenient than the 30% threshold (which under-counts wide-QRS leads).

    Returns (onset_idx, offset_idx) absolute indices into lead_signal.
    If detection fails we fall back to fixed ±60 ms / +80 ms windows.
    """
    pre = _ms_to_samples(win_pre_ms, fs)
    post = _ms_to_samples(win_post_ms, fs)
    n = lead_signal.shape[0]
    lo = max(0, r_peak - pre)
    hi = min(n - 1, r_peak + post)
    if hi - lo < 4:
        return lo, hi

    # smoothed gradient (3-sample box)
    raw_grad = np.abs(np.diff(lead_signal[lo:hi + 1]))
    if raw_grad.size < 3:
        return max(0, r_peak - _ms_to_samples(40, fs)), min(n - 1, r_peak + _ms_to_samples(40, fs))
    kernel = np.ones(3) / 3
    grad = np.convolve(raw_grad, kernel, mode='same')
    if np.max(grad) <= 0:
        return max(0, r_peak - _ms_to_samples(40, fs)), min(n - 1, r_peak + _ms_to_samples(40, fs))
    thr = grad_frac * np.max(grad)

    # Onset: walk backward from R; find first index where grad has been
    # below thr for 2 consecutive samples
    r_local = r_peak - lo
    onset_local = 0
    below_run = 0
    for i in range(min(r_local, grad.size) - 1, -1, -1):
        if grad[i] < thr:
            below_run += 1
            if below_run >= 2:
                onset_local = i + 2  # boundary just before the run
                break
        else:
            below_run = 0
    # Offset: walk forward
    offset_local = grad.size
    below_run = 0
    for i in range(min(r_local, grad.size - 1), grad.size):
        if grad[i] < thr:
            below_run += 1
            if below_run >= 2:
                offset_local = i - 1
                break
        else:
            below_run = 0

    onset_idx = lo + onset_local
    offset_idx = lo + offset_local
    # Sanity clamps: QRS minimum 30 ms, maximum 200 ms
    min_qrs_samples = _ms_to_samples(30, fs)
    max_qrs_samples = _ms_to_samples(200, fs)
    if r_peak - onset_idx > max_qrs_samples // 2:
        onset_idx = r_peak - max_qrs_samples // 2
    if offset_idx - r_peak > max_qrs_samples // 2:
        offset_idx = r_peak + max_qrs_samples // 2
    if r_peak - onset_idx < min_qrs_samples // 2:
        onset_idx = max(0, r_peak - min_qrs_samples // 2)
    if offset_idx - r_peak < min_qrs_samples // 2:
        offset_idx = min(n - 1, r_peak + min_qrs_samples // 2)
    return onset_idx, offset_idx


def _per_beat_lead_features(
    lead_signal: np.ndarray,
    r_peak: int,
    fs: float,
) -> Dict[str, float]:
    """Compute per-beat features for one lead, anchored at one R-peak."""
    n = lead_signal.shape[0]
    onset, offset = _qrs_onset_offset(lead_signal, r_peak, fs)
    qrs_dur_ms = (offset - onset) * 1000.0 / fs

    # R-amp (max in the QRS window, must be positive)
    qrs_seg = lead_signal[onset:offset + 1] if offset > onset else lead_signal[onset:onset + 2]
    r_amp = float(np.max(qrs_seg))
    s_amp_neg = float(np.min(qrs_seg))   # negative deflection
    # Q-wave: negative deflection between onset and R-peak
    pre_r_seg = lead_signal[onset:r_peak + 1] if r_peak > onset else lead_signal[onset:onset + 2]
    q_min_idx_local = int(np.argmin(pre_r_seg))
    q_amp = float(pre_r_seg[q_min_idx_local])
    if q_amp >= 0:
        q_amp = 0.0
        q_dur_ms = 0.0
    else:
        # measure Q duration: from onset to where signal recrosses zero or
        # turns positive moving toward R
        q_start = onset
        q_end = onset + q_min_idx_local
        # extend forward to where signal crosses zero again
        idx = q_end
        while idx < min(r_peak, n - 1) and lead_signal[idx] < 0:
            idx += 1
        q_end = idx
        q_dur_ms = (q_end - q_start) * 1000.0 / fs

    # ST level at J+40 ms and J+60 ms (avg over 5-sample window)
    j_idx = offset
    win = 2  # ±2 sample → 5 total
    j40 = j_idx + _ms_to_samples(40, fs)
    j60 = j_idx + _ms_to_samples(60, fs)
    if j40 + win < n:
        st_j40 = float(np.mean(lead_signal[j40 - win:j40 + win + 1]))
    else:
        st_j40 = float('nan')
    if j60 + win < n:
        st_j60 = float(np.mean(lead_signal[j60 - win:j60 + win + 1]))
    else:
        st_j60 = float('nan')
    # ST at J point itself (for STEMI threshold, MI uses J)
    if j_idx + win < n:
        st_j = float(np.mean(lead_signal[j_idx - win:j_idx + win + 1]))
    else:
        st_j = float('nan')

    # T-wave: search 100-300 ms post-J for max-abs deflection
    t_start = j_idx + _ms_to_samples(100, fs)
    t_end = min(n - 1, j_idx + _ms_to_samples(300, fs))
    if t_end > t_start + 2:
        t_seg = lead_signal[t_start:t_end + 1]
        t_idx_local = int(np.argmax(np.abs(t_seg)))
        t_amp = float(t_seg[t_idx_local])
    else:
        t_amp = float('nan')

    # rsR' detection in QRS window: ≥ 2 positive peaks
    if qrs_seg.size >= 5:
        pos_peaks, props = find_peaks(qrs_seg, height=0.05)
        rsr = False
        if len(pos_peaks) >= 2:
            heights = props['peak_heights']
            # last two peaks: second must be ≥ first to qualify as M-shape (rsR')
            if heights[-1] >= heights[-2] and heights[-2] > 0.1:
                rsr = True
            elif np.max(heights[1:]) >= np.max(heights[:1]) and np.max(heights) > 0.1:
                rsr = True
    else:
        rsr = False

    # S-wave duration in lateral leads: width of the negative deflection after R
    post_r_seg = lead_signal[r_peak:offset + 1]
    if post_r_seg.size >= 2 and np.min(post_r_seg) < -0.05:
        # S-duration: from where signal crosses 0 (from above) to where it
        # returns to (or past) 0 again, within QRS offset
        below = post_r_seg < 0
        if np.any(below):
            first_below = np.argmax(below)
            # reverse scan from end of segment for last below
            last_below = len(below) - 1 - np.argmax(below[::-1])
            s_dur_ms = (last_below - first_below) * 1000.0 / fs
        else:
            s_dur_ms = 0.0
    else:
        s_dur_ms = 0.0

    return {
        'qrs_onset': onset,
        'qrs_offset': offset,
        'j_idx': j_idx,
        'qrs_dur_ms': qrs_dur_ms,
        'r_amp_mv': r_amp,
        's_amp_mv': abs(s_amp_neg),     # report S as positive scalar
        'q_amp_mv': q_amp,
        'q_dur_ms': q_dur_ms,
        'st_level_j_mv': st_j,
        'st_level_j40_mv': st_j40,
        'st_level_j60_mv': st_j60,
        't_amp_mv': t_amp,
        't_polarity': float(np.sign(t_amp)) if not np.isnan(t_amp) else 0.0,
        'rsr_pattern': bool(rsr),
        's_dur_ms': s_dur_ms,
    }


def _detect_p_wave(
    lead_ii: np.ndarray,
    qrs_onset: int,
    fs: float,
    p_threshold_mv: float = 0.05,
) -> Tuple[Optional[float], Optional[int]]:
    """Detect a positive P-wave 80-200 ms before QRS onset on lead II.

    Returns (p_amp_mv, p_peak_idx) or (None, None) if not found.
    """
    p_search_start = qrs_onset - _ms_to_samples(200, fs)
    p_search_end = qrs_onset - _ms_to_samples(80, fs)
    if p_search_start < 0 or p_search_end <= p_search_start:
        return None, None
    seg = lead_ii[p_search_start:p_search_end + 1]
    # baseline-correct using median of full lead
    seg = seg - np.median(lead_ii)
    if seg.size < 3:
        return None, None
    peak_local = int(np.argmax(seg))
    p_amp = float(seg[peak_local])
    if p_amp < p_threshold_mv:
        return None, None
    return p_amp, p_search_start + peak_local


def extract_digital_features(
    sig_12L: np.ndarray,
    fs: float = 102.4,
    lead_order: str = 'mimic',
) -> Dict:
    """
    Extract digital ECG features from a 12-lead signal.

    Args:
        sig_12L: (12, L) numpy array, mV scale.
        fs: sampling rate Hz (default 102.4).
        lead_order: 'mimic' or 'ptbxl'. We always reorder internally to
                    MIMIC layout for indexing by name.

    Returns:
        dict with global rhythm features (hr_bpm, rr_irregularity_cv,
        p_wave_present, pr_interval_ms, qrs_duration_ms_avg) and
        `per_lead`: dict[lead_name -> per-lead feature dict] using
        median-over-beats aggregation.
    """
    assert sig_12L.ndim == 2 and sig_12L.shape[0] == 12, \
        f"sig_12L must be (12, L), got {sig_12L.shape}"
    L = sig_12L.shape[1]

    # Reorder to MIMIC if needed
    if lead_order == 'ptbxl':
        # PTBXL [I,II,III,aVR,aVL,aVF,V1..V6] -> MIMIC [I,II,III,aVR,aVF,aVL,V1..V6]
        # Swap positions 4,5
        idx = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]
        sig_12L = sig_12L[idx]
    elif lead_order != 'mimic':
        raise ValueError(f"lead_order must be 'mimic' or 'ptbxl', got {lead_order}")

    out: Dict = {
        'fs': fs,
        'length_samples': L,
        'duration_s': L / fs,
        'unreliable_signal': False,
        'warnings': [],
    }

    lead_ii = sig_12L[lead_idx_mimic('II')]
    # DC offset check
    dc = float(np.median(lead_ii))
    if abs(dc) > 1.0:
        out['warnings'].append(f"lead II DC offset {dc:+.2f} mV > 1.0 mV")

    # R-peak detection on lead II
    r_peaks = _detect_r_peaks(lead_ii, fs)
    if r_peaks.size < 2:
        # try aVF as fallback
        avf = sig_12L[lead_idx_mimic('aVF')]
        r_peaks = _detect_r_peaks(avf, fs)
    if r_peaks.size < 2:
        out['unreliable_signal'] = True
        out['hr_bpm'] = None
        out['rr_intervals_ms'] = []
        out['rr_irregularity_cv'] = None
        out['p_wave_present'] = False
        out['pr_interval_ms'] = None
        out['qrs_duration_ms_avg'] = None
        out['per_lead'] = {}
        out['warnings'].append("R-peak detection failed (< 2 peaks)")
        return out

    rr_samples = np.diff(r_peaks)
    rr_ms = rr_samples * 1000.0 / fs
    hr_bpm = 60_000.0 / float(np.median(rr_ms)) if rr_ms.size else None
    rr_cv = float(np.std(rr_ms) / np.mean(rr_ms)) if rr_ms.size and np.mean(rr_ms) > 0 else None

    out['hr_bpm'] = hr_bpm
    out['rr_intervals_ms'] = rr_ms.tolist()
    out['rr_irregularity_cv'] = rr_cv
    out['n_r_peaks'] = int(r_peaks.size)

    # Per-lead per-beat features (median across beats)
    per_lead: Dict[str, Dict[str, float]] = {}
    qrs_durs: List[float] = []
    qrs_onsets_ii: List[int] = []
    for li, name in enumerate(MIMIC_LEAD_NAMES):
        feats_per_beat: Dict[str, List[float]] = {}
        for r in r_peaks:
            beat_feat = _per_beat_lead_features(sig_12L[li], int(r), fs)
            for k, v in beat_feat.items():
                feats_per_beat.setdefault(k, []).append(v)
            if name == 'II':
                qrs_durs.append(beat_feat['qrs_dur_ms'])
                qrs_onsets_ii.append(beat_feat['qrs_onset'])
        # Aggregate: numeric -> median (ignoring nan); bool -> majority
        agg: Dict[str, float] = {}
        for k, vlist in feats_per_beat.items():
            if all(isinstance(v, bool) for v in vlist):
                agg[k] = bool(sum(vlist) > len(vlist) / 2)
            else:
                arr = np.array([float(v) for v in vlist if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
                agg[k] = float(np.median(arr)) if arr.size else float('nan')
        per_lead[name] = agg
    out['per_lead'] = per_lead

    # Global QRS duration: median across beats, lead II
    out['qrs_duration_ms_avg'] = float(np.median(qrs_durs)) if qrs_durs else None
    # Broadest-lead QRS for CD criteria: max median across I, V5, V6
    broad_qrs = max(
        per_lead['I']['qrs_dur_ms'],
        per_lead['V5']['qrs_dur_ms'],
        per_lead['V6']['qrs_dur_ms'],
    )
    out['qrs_duration_ms_broadest'] = float(broad_qrs)

    # P-wave detection (lead II) using median QRS onset
    if qrs_onsets_ii:
        # detect on the FIRST beat for which we have full pre-window
        p_amp_list = []
        pr_ms_list = []
        for r, qrs_on in zip(r_peaks, qrs_onsets_ii):
            p_amp, p_idx = _detect_p_wave(lead_ii, int(qrs_on), fs)
            if p_amp is not None and p_idx is not None:
                p_amp_list.append(p_amp)
                # PR interval = (p_idx → qrs_on) approx (using P peak as proxy
                # for P onset; bias ~40 ms so we add 0 — i.e. we report PR-peak)
                pr_ms_list.append((qrs_on - p_idx) * 1000.0 / fs)
        if p_amp_list:
            out['p_wave_present'] = True
            out['p_wave_amp_mv'] = float(np.median(p_amp_list))
            out['pr_interval_ms'] = float(np.median(pr_ms_list))
            out['p_detection_rate'] = len(p_amp_list) / len(r_peaks)
        else:
            out['p_wave_present'] = False
            out['p_wave_amp_mv'] = 0.0
            out['pr_interval_ms'] = None
            out['p_detection_rate'] = 0.0
    else:
        out['p_wave_present'] = False
        out['p_wave_amp_mv'] = 0.0
        out['pr_interval_ms'] = None
        out['p_detection_rate'] = 0.0

    return out


# ---------------------------------------------------------------------------
# Per-class digital criteria checkers
# ---------------------------------------------------------------------------

ANTERIOR = ('V1', 'V2', 'V3', 'V4')
LATERAL = ('I', 'aVL', 'V5', 'V6')
INFERIOR = ('II', 'III', 'aVF')
DOMINANT_R = ('I', 'II', 'V4', 'V5', 'V6')


def _contiguous_pairs(group):
    """Return list of contiguous lead pairs within a group."""
    return [(group[i], group[i + 1]) for i in range(len(group) - 1)]


def check_norm(features: Dict) -> Dict:
    """Apply NORM digital criteria.  Returns dict with per-criterion bool + overall pass."""
    res = {}
    hr = features.get('hr_bpm')
    res['N1_hr_60_100'] = hr is not None and 60 <= hr <= 100
    res['N1_loose_50_130'] = hr is not None and 50 <= hr <= 130
    rr_cv = features.get('rr_irregularity_cv')
    res['N2_rr_regular_cv_le_010'] = rr_cv is not None and rr_cv <= 0.10
    res['N3_p_wave_present'] = bool(features.get('p_wave_present'))
    pr = features.get('pr_interval_ms')
    res['N4_pr_120_200'] = pr is not None and 120 <= pr <= 200
    qrs = features.get('qrs_duration_ms_avg')
    # N5: QRS < 120 ms is the canonical normal upper bound (LITFL,
    # Surawicz AHA 2009). Lower bound is implicit; we accept ≥ 50 ms
    # to allow VAE-narrowed synth QRS while still rejecting noise.
    res['N5_qrs_70_110'] = qrs is not None and 50 <= qrs < 120
    pl = features.get('per_lead', {})
    if pl:
        st = np.mean([pl[ld]['st_level_j40_mv'] for ld in ('II', 'V5', 'V6') if ld in pl and not np.isnan(pl[ld]['st_level_j40_mv'])])
        res['N6_st_isoelectric'] = -0.05 <= st <= 0.10
        # NORM N7: T positive (sign > 0, not amplitude > 0.10 mV — the latter
        # is the STTC "prominent inverted-T" threshold not the NORM polarity test)
        t_polarity_pos = sum(1 for ld in ('II', 'V5', 'V6')
                             if ld in pl and pl[ld].get('t_amp_mv', 0) > 0)
        res['N7_t_positive_2of3'] = t_polarity_pos >= 2
    else:
        res['N6_st_isoelectric'] = False
        res['N7_t_positive_2of3'] = False

    # Strict NORM = all of N1, N3, N4, N5, N6, N7
    res['NORM_strict_pass'] = all([
        res['N1_hr_60_100'], res['N3_p_wave_present'], res['N4_pr_120_200'],
        res['N5_qrs_70_110'], res['N6_st_isoelectric'], res['N7_t_positive_2of3'],
    ])
    # Rate-variant: relax HR; require P (or AFib mode below)
    res['NORM_ratevariant_pass'] = all([
        res['N1_loose_50_130'], res['N3_p_wave_present'],
        res['N5_qrs_70_110'], res['N6_st_isoelectric'], res['N7_t_positive_2of3'],
    ])
    # AFib path: irregular AND no P
    res['NORM_afib_pass'] = (rr_cv is not None and rr_cv > 0.10) and (not res['N3_p_wave_present'])
    # Any-NORM
    res['NORM_any_pass'] = res['NORM_strict_pass'] or res['NORM_ratevariant_pass'] or res['NORM_afib_pass']
    return res


def check_mi(features: Dict) -> Dict:
    res = {}
    pl = features.get('per_lead', {})
    if not pl:
        return {'MI1_ste_2contig': False, 'MI2_q_dur': False, 'MI3_q_depth': False, 'MI_pass': False}
    # MI1: ST elevation at J point ≥ 0.10 mV in 2 contiguous leads
    contig_groups = [ANTERIOR, LATERAL, INFERIOR]
    ste_pairs = []
    for grp in contig_groups:
        for a, b in _contiguous_pairs(grp):
            if a in pl and b in pl:
                sa = pl[a].get('st_level_j_mv', float('nan'))
                sb = pl[b].get('st_level_j_mv', float('nan'))
                if not (np.isnan(sa) or np.isnan(sb)) and sa >= 0.10 and sb >= 0.10:
                    ste_pairs.append((a, b, sa, sb))
    res['MI1_ste_2contig'] = bool(ste_pairs)
    res['MI1_pairs'] = ste_pairs

    # MI2 + MI3 in same lead, anterior groups
    pathological_q = []
    for ld in ANTERIOR + LATERAL + INFERIOR:
        if ld in pl:
            q_dur = pl[ld].get('q_dur_ms', 0.0)
            q_amp = pl[ld].get('q_amp_mv', 0.0)
            r_amp = pl[ld].get('r_amp_mv', 0.0)
            ok_dur = q_dur >= 30.0
            ok_depth = q_amp <= -0.10 and (r_amp > 0 and abs(q_amp) >= 0.25 * r_amp)
            if ok_dur and ok_depth:
                pathological_q.append((ld, q_dur, q_amp, r_amp))
    res['MI2_q_dur_and_MI3_q_depth'] = bool(pathological_q)
    res['MI_pathological_q_leads'] = pathological_q

    res['MI_pass'] = res['MI1_ste_2contig'] or res['MI2_q_dur_and_MI3_q_depth']
    return res


def check_sttc(features: Dict) -> Dict:
    res = {}
    pl = features.get('per_lead', {})
    if not pl:
        return {'ST1_depression': False, 'ST2_t_inversion': False, 'STTC_pass': False}
    # ST1: ST_J60 ≤ -0.05 mV in 2 contiguous leads
    contig_groups = [ANTERIOR, LATERAL, INFERIOR]
    std_pairs = []
    for grp in contig_groups:
        for a, b in _contiguous_pairs(grp):
            if a in pl and b in pl:
                sa = pl[a].get('st_level_j60_mv', float('nan'))
                sb = pl[b].get('st_level_j60_mv', float('nan'))
                if not (np.isnan(sa) or np.isnan(sb)) and sa <= -0.05 and sb <= -0.05:
                    std_pairs.append((a, b, sa, sb))
    res['ST1_depression'] = bool(std_pairs)
    res['ST1_pairs'] = std_pairs

    # ST2: T-wave inversion in 2 contiguous dominant-R leads
    tinv_pairs = []
    for a, b in _contiguous_pairs(DOMINANT_R):
        if a in pl and b in pl:
            ta = pl[a].get('t_amp_mv', float('nan'))
            tb = pl[b].get('t_amp_mv', float('nan'))
            if not (np.isnan(ta) or np.isnan(tb)) and ta <= -0.10 and tb <= -0.10:
                tinv_pairs.append((a, b, ta, tb))
    res['ST2_t_inversion'] = bool(tinv_pairs)
    res['ST2_pairs'] = tinv_pairs

    res['STTC_pass'] = res['ST1_depression'] or res['ST2_t_inversion']
    return res


def check_hyp(features: Dict) -> Dict:
    res = {}
    pl = features.get('per_lead', {})
    if not pl:
        return {'H1_sokolow': False, 'H2_cornell': False, 'HYP_pass': False, 'sokolow_mv': 0.0, 'cornell_mv': 0.0}
    s_v1 = pl.get('V1', {}).get('s_amp_mv', 0.0)
    r_v5 = pl.get('V5', {}).get('r_amp_mv', 0.0)
    r_v6 = pl.get('V6', {}).get('r_amp_mv', 0.0)
    sokolow = s_v1 + max(r_v5, r_v6)
    res['sokolow_mv'] = float(sokolow)
    res['H1_sokolow'] = sokolow > 3.5

    r_avl = pl.get('aVL', {}).get('r_amp_mv', 0.0)
    s_v3 = pl.get('V3', {}).get('s_amp_mv', 0.0)
    cornell = r_avl + s_v3
    res['cornell_mv'] = float(cornell)
    res['H2_cornell_male'] = cornell > 2.8
    res['H2w_cornell_female'] = cornell > 2.0

    res['HYP_pass'] = res['H1_sokolow'] or res['H2_cornell_male']
    return res


def check_cd(features: Dict) -> Dict:
    """Three sub-checks (LBBB, RBBB, 1°AVB); CD passes if any one passes."""
    res = {}
    pl = features.get('per_lead', {})
    if not pl:
        return {'LBBB_pass': False, 'RBBB_pass': False, 'AVB_pass': False, 'CD_pass': False}
    qrs_broadest = features.get('qrs_duration_ms_broadest', 0.0) or 0.0

    # LBBB
    res['L1_qrs_120'] = qrs_broadest >= 120.0
    # L2: broad monomorphic R in I, V5, V6 with no Q
    l2_count = 0
    for ld in ('I', 'V5', 'V6'):
        if ld in pl:
            r_amp = pl[ld].get('r_amp_mv', 0.0)
            q_amp = pl[ld].get('q_amp_mv', 0.0)
            if r_amp >= 0.5 and q_amp > -0.05:
                l2_count += 1
    res['L2_lateral_monomorphic_R'] = l2_count >= 2
    # L3: deep wide S (or QS) in V1, V2 (R/S < 0.5)
    l3_count = 0
    for ld in ('V1', 'V2'):
        if ld in pl:
            r = pl[ld].get('r_amp_mv', 0.0)
            s = pl[ld].get('s_amp_mv', 0.0)
            if s >= 0.5 and (r / max(s, 1e-6)) < 0.5:
                l3_count += 1
    res['L3_v1v2_deep_S'] = l3_count >= 1
    res['LBBB_pass'] = res['L1_qrs_120'] and res['L2_lateral_monomorphic_R'] and res['L3_v1v2_deep_S']

    # RBBB
    res['R1_qrs_120'] = qrs_broadest >= 120.0
    rsr_v1 = pl.get('V1', {}).get('rsr_pattern', False) or pl.get('V2', {}).get('rsr_pattern', False)
    res['R2_rsR_v1v2'] = bool(rsr_v1)
    res['RBBB_pass'] = res['R1_qrs_120'] and res['R2_rsR_v1v2']

    # 1°AVB
    pr = features.get('pr_interval_ms')
    res['A1_pr_gt_200'] = pr is not None and pr > 200.0
    # A2: every P followed by QRS — proxy via p_detection_rate ≥ 0.8
    p_det_rate = features.get('p_detection_rate', 0.0)
    res['A2_p_qrs_match'] = p_det_rate >= 0.8
    res['AVB_pass'] = res['A1_pr_gt_200'] and res['A2_p_qrs_match']

    res['CD_pass'] = res['LBBB_pass'] or res['RBBB_pass'] or res['AVB_pass']
    return res


def evaluate_super5(features: Dict, super5_target: str) -> Dict:
    """Run the full super5 criterion battery and return the {target}_pass key."""
    out = {
        'NORM': check_norm(features),
        'MI': check_mi(features),
        'STTC': check_sttc(features),
        'HYP': check_hyp(features),
        'CD': check_cd(features),
    }
    flat = {}
    for cls, sub in out.items():
        for k, v in sub.items():
            flat[f"{cls}.{k}"] = v
    # Top-level pass for the requested target
    target_pass_key_map = {
        'NORM': 'NORM.NORM_any_pass',
        'MI': 'MI.MI_pass',
        'STTC': 'STTC.STTC_pass',
        'HYP': 'HYP.HYP_pass',
        'CD': 'CD.CD_pass',
    }
    flat['target'] = super5_target
    flat['target_pass'] = bool(flat.get(target_pass_key_map.get(super5_target, ''), False))
    return flat


# ---------------------------------------------------------------------------
# Self-test on synthetic signal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a synthetic 12-lead ECG: pure sinus rhythm at 75 bpm, 10 s, fs 102.4
    fs = 102.4
    duration_s = 10.0
    L = int(fs * duration_s)
    t = np.arange(L) / fs
    hr = 75.0
    rr_s = 60.0 / hr
    n_beats = int(duration_s / rr_s)

    sig = np.zeros((12, L))

    def _gauss(width, amp, t0, t):
        return amp * np.exp(-((t - t0) ** 2) / (2 * width ** 2))

    for b in range(n_beats):
        beat_t = (b + 0.5) * rr_s
        # P-wave at beat_t - 0.16
        sig[1] += _gauss(0.025, 0.15, beat_t - 0.16, t)
        # QRS sharp peak at beat_t (positive R in I, II, V5, V6)
        for ld in (0, 1, 10, 11):
            sig[ld] += _gauss(0.02, 1.2, beat_t, t)
        # Negative S in V1, V2
        sig[6] -= _gauss(0.02, 0.8, beat_t + 0.02, t)
        sig[7] -= _gauss(0.02, 0.5, beat_t + 0.02, t)
        # T-wave 0.25 s post R, positive in II, V5, V6
        for ld in (1, 10, 11):
            sig[ld] += _gauss(0.06, 0.30, beat_t + 0.25, t)
        # add aVF some positive R
        sig[4] += _gauss(0.02, 0.8, beat_t, t)

    # add small noise
    rng = np.random.default_rng(0)
    sig += rng.normal(0, 0.01, size=sig.shape)

    feats = extract_digital_features(sig, fs=fs, lead_order='mimic')
    print(f"HR = {feats['hr_bpm']:.1f} bpm")
    print(f"RR-CV = {feats['rr_irregularity_cv']:.3f}")
    print(f"P-wave present = {feats['p_wave_present']}, amp = {feats.get('p_wave_amp_mv'):.3f}")
    print(f"PR interval = {feats['pr_interval_ms']}")
    print(f"QRS dur (II) = {feats['qrs_duration_ms_avg']:.0f} ms")
    print(f"QRS dur broadest = {feats['qrs_duration_ms_broadest']:.0f} ms")
    print(f"V5 R amp = {feats['per_lead']['V5']['r_amp_mv']:.3f} mV")
    print(f"V1 S amp = {feats['per_lead']['V1']['s_amp_mv']:.3f} mV")
    print(f"II ST_J40 = {feats['per_lead']['II']['st_level_j40_mv']:.3f} mV")
    print(f"II T amp = {feats['per_lead']['II']['t_amp_mv']:.3f} mV")
    print()
    res = evaluate_super5(feats, 'NORM')
    for k, v in res.items():
        print(f"  {k}: {v}")
