"""Digital ground-truth validation of ECGTwin super5 synth samples.

Reads tensor dumps from `outputs/sanity_super5_tensors/`, runs
`util.ecg_digital_features.extract_digital_features` and the per-class
threshold checkers, then writes a markdown report with:
  - per-sample row (cls, prompt, seed, hr, qrs, key metric, pass)
  - per-class summary (n_pass / n_total)
  - final verdict on which classes pass at 3/3, ≥1/3, 0/3 seeds

Usage:
  uv run python \
    scripts/ecgtwin_gen/digital_gt_validate.py \
    --tensor_dir outputs/sanity_super5_tensors \
    --out_md docs/ecgtwin_super5_digital_gt_validation.md
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from util.ecg_digital_features import (         # noqa: E402
    extract_digital_features, check_norm, check_mi, check_sttc, check_hyp, check_cd,
)


def fmt_metric(cls: str, feats: dict, scores: dict) -> str:
    """Compact 'key digital metric' string for the report."""
    pl = feats.get('per_lead', {})
    if cls == 'NORM':
        hr = feats.get('hr_bpm')
        rr_cv = feats.get('rr_irregularity_cv')
        p = feats.get('p_wave_present')
        qrs = feats.get('qrs_duration_ms_avg')
        return f"HR={hr:.0f}, RR-CV={rr_cv:.2f}, P={p}, QRS={qrs:.0f} ms" if hr else "no R-peaks"
    if cls == 'MI':
        ste_pairs = scores['MI'].get('MI1_pairs', [])
        if ste_pairs:
            a, b, sa, sb = ste_pairs[0]
            return f"STE J@({a},{b}) = {sa*1000:.0f}/{sb*1000:.0f} µV (need ≥100)"
        # Show max ST_J in any contiguous group
        max_st_j = max(
            (pl[ld]['st_level_j_mv'] for ld in pl if not np.isnan(pl[ld]['st_level_j_mv'])),
            default=float('nan')
        )
        return f"max ST_J = {max_st_j*1000:.0f} µV (need ≥100 in 2 contig)"
    if cls == 'STTC':
        std_pairs = scores['STTC'].get('ST1_pairs', [])
        tinv_pairs = scores['STTC'].get('ST2_pairs', [])
        if std_pairs:
            a, b, sa, sb = std_pairs[0]
            return f"STD J60@({a},{b}) = {sa*1000:.0f}/{sb*1000:.0f} µV"
        if tinv_pairs:
            a, b, ta, tb = tinv_pairs[0]
            return f"T-inv@({a},{b}) = {ta*1000:.0f}/{tb*1000:.0f} µV"
        if pl:
            min_st60 = min(
                (pl[ld]['st_level_j60_mv'] for ld in pl if not np.isnan(pl[ld]['st_level_j60_mv'])),
                default=float('nan')
            )
            min_t = min(
                (pl[ld]['t_amp_mv'] for ld in ('I', 'II', 'V4', 'V5', 'V6')
                 if ld in pl and not np.isnan(pl[ld]['t_amp_mv'])),
                default=float('nan')
            )
            return f"min ST_J60={min_st60*1000:.0f} µV, min T(dom-R)={min_t*1000:.0f} µV"
        return "n/a"
    if cls == 'HYP':
        sok = scores['HYP'].get('sokolow_mv', 0.0)
        cor = scores['HYP'].get('cornell_mv', 0.0)
        return f"Sokolow S(V1)+R(V5/6)={sok:.2f} mV (need >3.5), Cornell={cor:.2f} mV (need >2.8)"
    if cls == 'CD':
        qrs_b = feats.get('qrs_duration_ms_broadest', 0.0) or 0.0
        rsr = pl.get('V1', {}).get('rsr_pattern', False) or pl.get('V2', {}).get('rsr_pattern', False)
        pr = feats.get('pr_interval_ms')
        return (f"QRS_broad={qrs_b:.0f} ms (need ≥120), rsR'(V1/V2)={rsr}, "
                f"PR={pr if pr is None else f'{pr:.0f} ms'}")
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor_dir", default="outputs/sanity_super5_tensors")
    ap.add_argument("--out_md", default="docs/ecgtwin_super5_digital_gt_validation.md")
    ap.add_argument("--fs", type=float, default=102.4)
    ap.add_argument("--lead_order", default="mimic", choices=['mimic', 'ptbxl'])
    args = ap.parse_args()

    tensor_dir = Path(args.tensor_dir)
    npz_files = sorted(tensor_dir.rglob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No .npz files found under {tensor_dir}")
    print(f"[validate] found {len(npz_files)} samples under {tensor_dir}")

    rows = []
    for npz in npz_files:
        with np.load(npz, allow_pickle=True) as z:
            sig_ct = z['signal_ct']     # (12, 1024)
            prompt = str(z['prompt'])
            seed = int(z['seed'])
            cls = str(z['super5_target'])
            probs = z['super5_probs']
            top1 = str(z['super5_top1'])
        feats = extract_digital_features(sig_ct, fs=args.fs, lead_order=args.lead_order)
        scores = {
            'NORM': check_norm(feats),
            'MI': check_mi(feats),
            'STTC': check_sttc(feats),
            'HYP': check_hyp(feats),
            'CD': check_cd(feats),
        }
        target_pass_key = {
            'NORM': 'NORM_any_pass', 'MI': 'MI_pass',
            'STTC': 'STTC_pass', 'HYP': 'HYP_pass', 'CD': 'CD_pass',
        }[cls]
        target_pass = bool(scores[cls][target_pass_key])
        metric_str = fmt_metric(cls, feats, scores)
        rows.append({
            'cls': cls,
            'prompt': prompt,
            'seed': seed,
            'hr_bpm': feats.get('hr_bpm'),
            'rr_cv': feats.get('rr_irregularity_cv'),
            'qrs_dur_ms_ii': feats.get('qrs_duration_ms_avg'),
            'qrs_dur_ms_broad': feats.get('qrs_duration_ms_broadest'),
            'p_wave_present': feats.get('p_wave_present'),
            'pr_interval_ms': feats.get('pr_interval_ms'),
            'metric': metric_str,
            'target_pass': target_pass,
            'unreliable_signal': feats.get('unreliable_signal', False),
            'top1_victim': top1,
            'p_target_victim': float(probs[['NORM','MI','HYP','CD','STTC'].index(cls)]),
            # also save NORM-strict, NORM-rate, NORM-afib individually for diagnosis
            'norm_strict_pass': bool(scores['NORM']['NORM_strict_pass']),
            'norm_ratevariant_pass': bool(scores['NORM']['NORM_ratevariant_pass']),
            'norm_afib_pass': bool(scores['NORM']['NORM_afib_pass']),
            'mi1_ste': bool(scores['MI']['MI1_ste_2contig']),
            'sttc_st1': bool(scores['STTC']['ST1_depression']),
            'sttc_st2': bool(scores['STTC']['ST2_t_inversion']),
            'hyp_sokolow_mv': scores['HYP']['sokolow_mv'],
            'hyp_cornell_mv': scores['HYP']['cornell_mv'],
            'cd_lbbb': bool(scores['CD']['LBBB_pass']),
            'cd_rbbb': bool(scores['CD']['RBBB_pass']),
            'cd_avb': bool(scores['CD']['AVB_pass']),
        })

    # Aggregate per-class
    by_class = defaultdict(list)
    by_cell = defaultdict(list)
    for r in rows:
        by_class[r['cls']].append(r)
        by_cell[(r['cls'], r['prompt'])].append(r)

    # Compose markdown
    lines = []
    lines.append("# ECGTwin Super5 — Digital Ground-Truth Validation\n")
    lines.append(f"Date: 2026-04-27. Source: tensor dumps in `{args.tensor_dir}`. "
                  f"Sampling rate: **{args.fs} Hz**. Lead order: **{args.lead_order} (MIMIC)**. "
                  f"Total samples: **{len(rows)}** "
                  f"({len(by_cell)} (super5_target × prompt) cells × {len(rows) // max(len(by_cell),1)} seeds).\n")
    lines.append("Criteria reference: [`docs/ecg_digital_thresholds.md`](ecg_digital_thresholds.md). "
                  "All thresholds in µV / ms with 2-contiguous-leads requirement where applicable.\n")
    lines.append("---\n")

    # Per-sample table
    lines.append("## Per-sample digital validation\n")
    lines.append("| super5 | prompt | seed | HR | QRS_II | key digital metric | medical pass? | victim top1 (p_target) |")
    lines.append("|---|---|---:|---:|---:|---|:---:|---|")
    for r in rows:
        hr_s = f"{r['hr_bpm']:.0f}" if r['hr_bpm'] else "—"
        qrs_s = f"{r['qrs_dur_ms_ii']:.0f}" if r['qrs_dur_ms_ii'] is not None else "—"
        pass_s = "**PASS**" if r['target_pass'] else "FAIL"
        if r['unreliable_signal']:
            pass_s = "FAIL (unreliable)"
        lines.append(
            f"| {r['cls']} | `{r['prompt'][:42]}` | {r['seed']} | {hr_s} | {qrs_s} | "
            f"{r['metric']} | {pass_s} | {r['top1_victim']} (p={r['p_target_victim']:.2f}) |"
        )
    lines.append("")

    # Per-cell summary
    lines.append("## Per-cell pass rate (3 seeds per cell)\n")
    lines.append("| super5 | prompt | n_seeds | n_pass | pass rate |")
    lines.append("|---|---|---:|---:|---:|")
    for (cls, prompt), runs in sorted(by_cell.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n_pass = sum(r['target_pass'] for r in runs)
        lines.append(f"| {cls} | `{prompt[:55]}` | {len(runs)} | {n_pass} | {n_pass}/{len(runs)} |")
    lines.append("")

    # Per-class summary
    lines.append("## Per-class summary\n")
    lines.append("| super5 | n_samples | n_pass digital | pass rate | victim top1 hit | victim p_target avg |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cls in ('NORM', 'MI', 'STTC', 'HYP', 'CD'):
        runs = by_class.get(cls, [])
        if not runs:
            continue
        n = len(runs)
        n_pass = sum(r['target_pass'] for r in runs)
        n_top1 = sum(r['top1_victim'] == cls for r in runs)
        avg_p = float(np.mean([r['p_target_victim'] for r in runs]))
        lines.append(f"| {cls} | {n} | {n_pass} | {n_pass / n:.2f} | {n_top1}/{n} | {avg_p:.3f} |")
    lines.append("")

    # Verdict bucketing
    lines.append("## Verdict — which super5 classes meet medical-defined digital criteria?\n")
    cell_passrates = {}
    for (cls, prompt), runs in by_cell.items():
        n_pass = sum(r['target_pass'] for r in runs)
        cell_passrates[(cls, prompt)] = (n_pass, len(runs))

    def bucket(cls):
        # take best cell for this class (most permissive prompt)
        cells = [(p, *cell_passrates[(c, p)]) for (c, p) in cell_passrates if c == cls]
        if not cells:
            return None, []
        cells.sort(key=lambda x: -x[1])  # sort by n_pass desc
        return cells[0], cells

    full_pass = []
    partial = []
    zero = []
    for cls in ('NORM', 'MI', 'STTC', 'HYP', 'CD'):
        best, all_cells = bucket(cls)
        if best is None:
            continue
        prompt, n_pass, n_total = best
        if n_pass == n_total and n_total >= 3:
            full_pass.append((cls, prompt, n_pass, n_total))
        elif n_pass >= 1:
            partial.append((cls, prompt, n_pass, n_total, all_cells))
        else:
            zero.append((cls, prompt, n_pass, n_total, all_cells))

    lines.append("**Pass at 3/3 seeds (best cell):**")
    if full_pass:
        for cls, p, n, nt in full_pass:
            lines.append(f"- **{cls}** — best prompt `{p}` → {n}/{nt} seeds pass digital criteria.")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("**Pass at ≥ 1/3 seeds in best cell (partial):**")
    if partial:
        for cls, p, n, nt, _ in partial:
            lines.append(f"- **{cls}** — best prompt `{p}` → {n}/{nt} seeds pass.")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("**Pass at 0/3 seeds in any cell:**")
    if zero:
        for cls, p, n, nt, _ in zero:
            lines.append(f"- **{cls}** — best prompt `{p}` still 0/{nt} pass.")
    else:
        lines.append("- (none)")
    lines.append("")

    # Limitations
    lines.append("## Limitations\n")
    lines.append("1. **Synthesis-domain artifacts** — VAE-decoded ECGs may have spectral artifacts that interfere with R-peak detection or J-point localization. Samples flagged `unreliable_signal` had < 2 R-peaks detectable on lead II + aVF; we mark these FAIL.")
    lines.append("2. **102.4 Hz granularity** — `samples = ms × 0.1024`. So 30 ms ≈ 3 samples, 120 ms ≈ 12 samples; QRS-duration measurements are integer-quantized at ~10 ms. A 110-ms QRS may read as either 107 or 117 ms depending on edge-detection rounding.")
    lines.append("3. **Sample size** — 3 seeds per (super5_target × prompt) cell is `n=3`; verdicts are descriptive of modal behavior, not statistical tests. Author's gallery uses 50 PNGs per prompt; with 50 we would tighten the partial-pass band substantially.")
    lines.append("4. **Subjective criteria reduced** — pericarditis's \"concave-upward STE\" and LV-strain's \"asymmetric T inversion\" cannot be reduced to a single number. We report the simplest digital proxy (mean ST level + T sign) and accept the false-negative rate.")
    lines.append("5. **PR-interval is approximate** — we use P-peak → QRS-onset rather than P-onset → QRS-onset; this biases reported PR by ~30-40 ms (under-estimate). For 1° AVB detection (PR > 200 ms) the bias makes the test conservatively strict.")
    lines.append("6. **Q-wave detection** depends on a clean QRS-onset estimate. With our gradient-based onset, false negatives on Q-wave are likely on noisy synth; we accept this and treat MI2/MI3 as supportive only.")
    lines.append("7. **Single reference patient** (`normal_1.pt`) — measurements reflect prompt-driven changes from a single morphology baseline; not a generalization claim across patient phenotypes.")
    lines.append("8. **Voltage absolute scale assumes the VAE preserves mV.** If decoded amplitudes systematically under-shoot real ECG amplitudes (typical for variational reconstruction), Sokolow-Lyon will be falsely-negative even on visually clear LVH morphology.")
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"[validate] wrote {out_md}")

    # Also save detailed per-sample json for downstream use
    json_path = out_md.with_suffix(".json")
    json_path.write_text(json.dumps({
        'rows': rows,
        'per_class': {cls: {
            'n': len(runs),
            'n_pass': sum(r['target_pass'] for r in runs),
            'pass_rate': sum(r['target_pass'] for r in runs) / max(len(runs), 1),
        } for cls, runs in by_class.items()},
        'per_cell': {f"{c}|{p}": {
            'n_pass': cell_passrates[(c, p)][0],
            'n_total': cell_passrates[(c, p)][1],
        } for (c, p) in cell_passrates},
    }, indent=2, default=str))
    print(f"[validate] wrote {json_path}")

    # Console summary
    print()
    print("=" * 72)
    print("Per-class digital pass rate:")
    for cls in ('NORM', 'MI', 'STTC', 'HYP', 'CD'):
        runs = by_class.get(cls, [])
        if not runs:
            continue
        n = len(runs)
        n_pass = sum(r['target_pass'] for r in runs)
        print(f"  {cls:<5}  {n_pass}/{n} pass ({n_pass / n:.0%})")


if __name__ == "__main__":
    main()
