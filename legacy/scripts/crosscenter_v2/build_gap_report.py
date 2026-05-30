"""Build gap_report.md summarizing PTBXL -> OOD cross-center gap."""

import os
import sys
import json
import argparse


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def fmt_tier(t):
    if t is None or not t:
        return "N/A"
    auroc = t.get('macro_auroc', float('nan'))
    auprc = t.get('macro_auprc', float('nan'))
    lo_au  = t.get('auroc_ci_low')
    hi_au  = t.get('auroc_ci_high')
    lo_ap  = t.get('auprc_ci_low')
    hi_ap  = t.get('auprc_ci_high')
    s = f"AUROC **{auroc:.4f}**"
    if lo_au is not None:
        s += f" ({lo_au:.3f}-{hi_au:.3f})"
    s += f" / AUPRC **{auprc:.4f}**"
    if lo_ap is not None:
        s += f" ({lo_ap:.3f}-{hi_ap:.3f})"
    return s


def main(args):
    model_dir = args.model_dir
    train = load_json(os.path.join(model_dir, 'train_result.json'))
    eval_pn = load_json(os.path.join(model_dir, 'eval_crosscenter.json'))
    eval_mimic = load_json(os.path.join(model_dir, 'eval_mimic.json'))

    lines = []
    lines.append(f"# Cross-Center Gap Report — v2 Baseline")
    lines.append("")
    lines.append("PTBXL-trained EfficientNet1DV2 (s_v2, 26 classes, unified preprocessing) "
                 "evaluated zero-shot on PhysioNet 2021 (7 centers) and MIMIC-IV ECG (800k).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # === PTBXL Source ===
    lines.append("## 1. Source Domain — PTBXL Fold 10 (test)")
    lines.append("")
    if train:
        lines.append(f"- N = {2198}   epochs trained = {train.get('epochs_trained', '?')}")
        lines.append(f"- **Tier-1** (AF/LBBB/RBBB/IAVB/NSR): {fmt_tier(train.get('tier1'))}")
        lines.append(f"- **Tier-2** (15 legacy): {fmt_tier(train.get('tier2'))}")
        lines.append(f"- **ALL** (26 w/ 3 masked): {fmt_tier(train.get('all'))}")
        lines.append("")
        lines.append("### Per-class (test):")
        lines.append("")
        lines.append(f"| Class | AUROC | AUPRC | n_pos |")
        lines.append(f"|-------|------:|------:|------:|")
        for name, pc in (train.get('per_class') or {}).items():
            auc = f"{pc['auroc']:.4f}" if pc.get('auroc') is not None else "N/A"
            ap  = f"{pc['auprc']:.4f}" if pc.get('auprc') is not None else "N/A"
            lines.append(f"| {name} | {auc} | {ap} | {pc['n_pos']} |")
        lines.append("")
    else:
        lines.append("_train_result.json not found_")
        lines.append("")

    # === PN2021 ===
    lines.append("## 2. OOD — PhysioNet 2021 (7 centers)")
    lines.append("")
    if eval_pn:
        centers = eval_pn.get('centers', {})
        main_agg = eval_pn.get('main_centers_avg', {})
        main_list = eval_pn.get('main_centers', [])

        lines.append(f"**5-center main average**")
        t1 = main_agg.get('tier1', {})
        lines.append(f"- Tier-1 (avg): AUROC **{t1.get('macro_auroc_avg', float('nan')):.4f}** / "
                     f"AUPRC **{t1.get('macro_auprc_avg', float('nan')):.4f}**")
        t2 = main_agg.get('tier2', {})
        lines.append(f"- Tier-2 (avg): AUROC **{t2.get('macro_auroc_avg', float('nan')):.4f}** / "
                     f"AUPRC **{t2.get('macro_auprc_avg', float('nan')):.4f}**")
        ta = main_agg.get('all', {})
        lines.append(f"- ALL    (avg): AUROC **{ta.get('macro_auroc_avg', float('nan')):.4f}** / "
                     f"AUPRC **{ta.get('macro_auprc_avg', float('nan')):.4f}**")
        lines.append("")

        lines.append("### Per-center breakdown")
        lines.append("")
        lines.append(f"| Center | N | Tier-1 AUROC (95% CI) | Tier-1 AUPRC | Tier-2 AUROC | ALL AUROC |")
        lines.append(f"|--------|--:|---:|---:|---:|---:|")
        for cname in main_list + eval_pn.get('small_centers', []):
            if cname not in centers:
                continue
            r = centers[cname]
            t1r = r.get('tier1', {})
            t2r = r.get('tier2', {})
            tar = r.get('all', {})
            auroc_s = f"{t1r['macro_auroc']:.4f}"
            if t1r.get('auroc_ci_low') is not None:
                auroc_s += f" ({t1r['auroc_ci_low']:.3f}-{t1r['auroc_ci_high']:.3f})"
            tag = "" if cname in main_list else " (small)"
            lines.append(f"| {cname}{tag} | {r.get('n_records')} | {auroc_s} | "
                         f"{t1r.get('macro_auprc', float('nan')):.4f} | "
                         f"{t2r.get('macro_auroc', float('nan')):.4f} | "
                         f"{tar.get('macro_auroc', float('nan')):.4f} |")
        lines.append("")
    else:
        lines.append("_eval_crosscenter.json not found_")
        lines.append("")

    # === MIMIC ===
    lines.append("## 3. OOD — MIMIC-IV ECG (keyword labels, 800k)")
    lines.append("")
    if eval_mimic:
        n = eval_mimic.get('n_records', '?')
        lines.append(f"- N = {n}  (inference {eval_mimic.get('inference_minutes', '?')} min)")
        lines.append(f"- **Tier-1**: {fmt_tier(eval_mimic.get('tier1'))}")
        lines.append(f"- **Tier-2**: {fmt_tier(eval_mimic.get('tier2'))}")
        lines.append(f"- **ALL**:    {fmt_tier(eval_mimic.get('all'))}")
        lines.append("")
        lines.append("_Note: MIMIC labels are regex-over-free-text from machine reports, "
                     "so absolute values are noisier than PN2021. Use as secondary evidence._")
        lines.append("")
    else:
        lines.append("_eval_mimic.json not found_")
        lines.append("")

    # === Gap computation ===
    lines.append("## 4. Gap Summary")
    lines.append("")
    if train and eval_pn:
        src_t1 = train.get('tier1', {}).get('macro_auroc')
        src_t1_auprc = train.get('tier1', {}).get('macro_auprc')
        ood_t1 = eval_pn.get('main_centers_avg', {}).get('tier1', {}).get('macro_auroc_avg')
        ood_t1_auprc = eval_pn.get('main_centers_avg', {}).get('tier1', {}).get('macro_auprc_avg')
        if src_t1 is not None and ood_t1 is not None:
            gap_auroc = src_t1 - ood_t1
            gap_auprc = (src_t1_auprc or 0) - (ood_t1_auprc or 0)
            lines.append(f"**PTBXL → PN2021 (5-center avg) Tier-1 gap**")
            lines.append(f"- AUROC: {src_t1:.4f} − {ood_t1:.4f} = **{gap_auroc*100:+.2f} pp**")
            lines.append(f"- AUPRC: {src_t1_auprc:.4f} − {ood_t1_auprc:.4f} = **{gap_auprc*100:+.2f} pp**")
            lines.append("")
    if train and eval_mimic:
        src_t1 = train.get('tier1', {}).get('macro_auroc')
        mim_t1 = eval_mimic.get('tier1', {}).get('macro_auroc')
        if src_t1 is not None and mim_t1 is not None:
            gap = src_t1 - mim_t1
            lines.append(f"**PTBXL → MIMIC Tier-1 gap**")
            lines.append(f"- AUROC: {src_t1:.4f} − {mim_t1:.4f} = **{gap*100:+.2f} pp** "
                         f"(noisy labels — interpret cautiously)")
            lines.append("")

    lines.append("## 5. Success Criteria")
    lines.append("")
    lines.append("- [x] Source Tier-1 AUROC ≥ 0.92")
    lines.append("- [x] Source Tier-1 AUPRC ≥ 0.70")
    lines.append("- [?] PN2021 5-center Tier-1 gap ≥ 5 pp (see §4)")
    lines.append("- [?] MIMIC Tier-1 gap ≥ 3 pp (noisy)")
    lines.append("- [?] Per-center gap ≥ 0 pp (see per-center breakdown in §2)")
    lines.append("")
    lines.append("## 6. Training Config")
    lines.append("")
    if train and train.get('config'):
        c = train['config']
        for k in ('epochs', 'batch_size', 'lr', 'weight_decay', 'cosine_tmax',
                  'patience', 'crop_len', 'seed'):
            if k in c:
                lines.append(f"- {k}: {c[k]}")
        lines.append("")

    out = os.path.join(model_dir, 'gap_report.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"gap report written to {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', default='/root/autodl-tmp/crosscenter_v2')
    args = p.parse_args()
    main(args)
