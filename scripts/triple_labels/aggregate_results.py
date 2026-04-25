"""Aggregate eval results from 3 schemes into a single Markdown report.

Usage:
    /root/miniforge3/envs/ECGTwin/bin/python scripts/triple_labels/aggregate_results.py \
        --out_dir /root/autodl-tmp/triple_labels
"""

import os
import sys
import json
import argparse
from datetime import datetime


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (x != x)):
        return '—'
    return f"{x:.{nd}f}"


def fmt_pos(stats):
    if stats is None:
        return '—'
    auc = stats.get('auroc')
    n_pos = stats.get('n_pos', 0)
    if auc is None:
        return f"N/A (n={n_pos})"
    return f"{auc:.3f} (n={n_pos})"


def write_report(results, train_results, out_path):
    lines = []
    lines.append("# Triple-Label ECG Classifier Experiment\n")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n")
    lines.append("Three EfficientNet1DV2 (s_v2) classifiers trained on PTB-XL,")
    lines.append(" evaluated zero-shot on PhysioNet/CinC 2021 (7 centers, ptb-xl excluded)")
    lines.append(" and MIMIC-IV ECG (test split). Same backbone, same preprocessing,")
    lines.append(" different label heads.\n\n")

    lines.append("## Table A — Macro AUROC / AUPRC\n\n")
    lines.append("| Scheme | classes | PTB-XL test AUROC | PTB-XL test AUPRC | "
                 "PN2021 avg AUROC | PN2021 avg AUPRC | MIMIC test AUROC | MIMIC test AUPRC |\n")
    lines.append("|--------|---------|-------------------|-------------------|"
                 "------------------|------------------|------------------|------------------|\n")
    for scheme, r in results.items():
        ptbxl = r.get('ptbxl_test', {})
        pn = r.get('pn2021', {})
        mimic = r.get('mimic_test', {})
        lines.append(
            f"| **{scheme}** | {r['num_classes']} "
            f"| {fmt(ptbxl.get('macro_auroc'))} | {fmt(ptbxl.get('macro_auprc'))} "
            f"| {fmt(pn.get('avg_macro_auroc'))} | {fmt(pn.get('avg_macro_auprc'))} "
            f"| {fmt(mimic.get('macro_auroc'))} | {fmt(mimic.get('macro_auprc'))} |\n"
        )
    lines.append("\n")

    lines.append("## Table B — Per-class AUROC\n\n")
    for scheme, r in results.items():
        names = r['class_names']
        lines.append(f"### {scheme} ({r['num_classes']} classes)\n\n")
        lines.append("| Class | PTB-XL test AUROC (n_pos) | PN2021 avg AUROC | "
                     "MIMIC test AUROC (n_pos) |\n")
        lines.append("|-------|---------------------------|------------------|"
                     "--------------------------|\n")
        ptbxl_pc = r.get('ptbxl_test', {}).get('per_class', {})
        mimic_pc = r.get('mimic_test', {}).get('per_class', {})
        # PN2021 per-center average per class
        pn_per_center = r.get('pn2021', {}).get('per_center', {})
        for name in names:
            ptbxl_cell = fmt_pos(ptbxl_pc.get(name))
            mimic_cell = fmt_pos(mimic_pc.get(name))
            # Average per-class AUROC across centers (skip None)
            pn_aucs = []
            for cdata in pn_per_center.values():
                pc = cdata.get('per_class', {}).get(name, {})
                if pc.get('auroc') is not None:
                    pn_aucs.append(pc['auroc'])
            pn_cell = fmt(sum(pn_aucs) / len(pn_aucs)) if pn_aucs else '—'
            lines.append(f"| {name} | {ptbxl_cell} | {pn_cell} | {mimic_cell} |\n")
        lines.append("\n")

    lines.append("## Table C — PN2021 per-center macro AUROC\n\n")
    centers_seen = set()
    for r in results.values():
        centers_seen.update(r.get('pn2021', {}).get('per_center', {}).keys())
    centers_seen = sorted(centers_seen)
    if centers_seen:
        lines.append("| Center | " + " | ".join(results.keys()) + " |\n")
        lines.append("|--------|" + "|".join("------" for _ in results) + "|\n")
        for c in centers_seen:
            row = [c]
            for scheme, r in results.items():
                cdata = r.get('pn2021', {}).get('per_center', {}).get(c, {})
                row.append(fmt(cdata.get('macro_auroc')))
            lines.append("| " + " | ".join(row) + " |\n")
        lines.append("\n")

    lines.append("## Training summary\n\n")
    lines.append("| Scheme | epochs trained | best val AUROC | test AUROC | test AUPRC |\n")
    lines.append("|--------|----------------|----------------|------------|------------|\n")
    for scheme, tr in train_results.items():
        if tr is None:
            continue
        lines.append(
            f"| **{scheme}** | {tr.get('epochs_trained', '?')} "
            f"| {fmt(tr.get('best_val_macro_auroc'))} "
            f"| {fmt(tr.get('test_macro_auroc'))} "
            f"| {fmt(tr.get('test_macro_auprc'))} |\n"
        )
    lines.append("\n")

    # Cross-domain gap
    lines.append("## Cross-domain gap (PTB-XL test - PN2021 avg)\n\n")
    lines.append("| Scheme | gap to PN2021 | gap to MIMIC |\n")
    lines.append("|--------|---------------|---------------|\n")
    for scheme, r in results.items():
        p = r.get('ptbxl_test', {}).get('macro_auroc')
        n = r.get('pn2021', {}).get('avg_macro_auroc')
        m = r.get('mimic_test', {}).get('macro_auroc')
        gap_n = (p - n) if (p is not None and n is not None) else None
        gap_m = (p - m) if (p is not None and m is not None) else None
        lines.append(f"| **{scheme}** | {fmt(gap_n, 3)} | {fmt(gap_m, 3)} |\n")
    lines.append("\n")

    lines.append("---\n")
    lines.append(f"_Source JSONs: `{', '.join(results.keys())}` under "
                 "`/root/autodl-tmp/triple_labels/<scheme>/eval_result.json`_\n")

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f"[aggregate] wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='/root/autodl-tmp/triple_labels')
    p.add_argument('--report_path',
                   default='/root/ECG_adv_Gen/docs/triple_labels_experiment.md')
    args = p.parse_args()

    schemes = ['super5', 'sub23', 'pn26']
    results = {}
    train_results = {}
    for s in schemes:
        eval_path = os.path.join(args.out_dir, s, 'eval_result.json')
        train_path = os.path.join(args.out_dir, s, 'train_result.json')
        if not os.path.exists(eval_path):
            print(f"[aggregate] WARN: missing {eval_path}")
            continue
        with open(eval_path) as f:
            results[s] = json.load(f)
        if os.path.exists(train_path):
            with open(train_path) as f:
                train_results[s] = json.load(f)

    if not results:
        print("[aggregate] ERROR: no results found")
        sys.exit(1)

    write_report(results, train_results, args.report_path)


if __name__ == '__main__':
    main()
