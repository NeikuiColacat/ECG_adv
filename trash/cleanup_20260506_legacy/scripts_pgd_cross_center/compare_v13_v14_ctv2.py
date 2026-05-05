"""3-way comparison: super5 victim baseline vs v13 vanilla synth+AT vs v14 CT v2 sphere=5 synth+AT.

After pilot iter4 ctv2 finishes, this script:
  1. Loads baseline eval (super5 victim, no AT) — eval_with_exclude_3cells_k200.json
  2. Loads v13 vanilla cell evals — synth_anchored_super5_v13/{tag}_k200/eval_with_exclude.json
  3. Loads v14 CT v2 cell evals — synth_anchored_super5_v14_ctv2/{tag}_k200/eval_with_exclude.json
  4. Produces a markdown report comparing per-cell, per-center, per-class

Run AFTER pilot_iter4_ctv2.sh + aggregate_pilot_results.py --out_root v14_ctv2 finish.

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/pgd_cross_center/compare_v13_v14_ctv2.py \
      --output /root/autodl-tmp/synth_anchored_super5_v14_ctv2/v13_vs_v14_comparison.md
"""
import argparse
import json
import math
from pathlib import Path
from typing import Optional

CELLS = ["extra", "nin", "geo"]
TAG_TO_CENTER = {"extra": "cpsc_2018_extra", "nin": "ningbo", "geo": "georgia"}
SUPER5 = ["CD", "HYP", "MI", "NORM", "STTC"]


def fmt_pp(after, before):
    if after is None or before is None:
        return "n/a"
    if isinstance(after, float) and math.isnan(after):
        return "nan"
    if isinstance(before, float) and math.isnan(before):
        return "nan"
    pp = (after - before) * 100.0
    sign = "+" if pp >= 0 else ""
    return f"{sign}{pp:.2f}"


def fmt_val(x):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.4f}"


def load_json(p):
    if not Path(p).exists():
        print(f"[warn] missing: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_path",
                    default="/root/autodl-tmp/triple_labels/super5/eval_with_exclude_3cells_pilot4.json")
    ap.add_argument("--v13_root", default="/root/autodl-tmp/synth_anchored_super5_v13")
    ap.add_argument("--v14_root", default="/root/autodl-tmp/synth_anchored_super5_v14_ctv2")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base = load_json(args.baseline_path)
    if base is None:
        print(f"[err] baseline missing — run aggregate_pilot_results.py first")
        return

    v13_cells = {tag: load_json(f"{args.v13_root}/{tag}_k200/eval_with_exclude.json")
                 for tag in CELLS}
    v14_cells = {tag: load_json(f"{args.v14_root}/{tag}_k200/eval_with_exclude.json")
                 for tag in CELLS}

    centers_in_pn = list(base.get("pn2021", {}).get("per_center", {}).keys())

    L = []
    L.append("# CT v2 sphere=5 vs Vanilla Synth — Downstream Cross-Center AUROC Comparison\n")
    L.append("Compared variants (all use Super5 victim, K=200 ref pool, "
             "PGD pilot iter4 settings):\n")
    L.append("- **baseline**: super5 victim, NO adversarial training")
    L.append("- **v13 vanilla**: PGD AT with vanilla ECGTwin synth (Plan Rev 12 production)")
    L.append("- **v14 CT v2**: PGD AT with CenterToken v2 sphere=5 synth (Plan Rev 13)\n")

    # PN2021 macro AUROC + AUPRC delta vs baseline
    L.append("## PN2021 per-center macro AUROC delta (Δ vs baseline, in pp)\n")
    L.append("| center | baseline | v13_extra | v14_extra | v13_nin | v14_nin | v13_geo | v14_geo |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in centers_in_pn:
        b = base["pn2021"]["per_center"][c].get("macro_auroc")
        row = [c, fmt_val(b)]
        for tag in CELLS:
            for variant_dict in [v13_cells, v14_cells]:
                ev = variant_dict.get(tag) or {}
                v = ev.get("pn2021", {}).get("per_center", {}).get(c, {}).get("macro_auroc")
                row.append(fmt_pp(v, b))
        L.append("| " + " | ".join(row) + " |")
    base_avg = base["pn2021"].get("avg_macro_auroc")
    row = ["**avg**", fmt_val(base_avg)]
    for tag in CELLS:
        for variant_dict in [v13_cells, v14_cells]:
            ev = variant_dict.get(tag) or {}
            v = ev.get("pn2021", {}).get("avg_macro_auroc")
            row.append(fmt_pp(v, base_avg))
    L.append("| " + " | ".join(row) + " |")
    L.append("")

    L.append("## PN2021 per-center macro AUPRC delta (Δ vs baseline, in pp)\n")
    L.append("| center | baseline | v13_extra | v14_extra | v13_nin | v14_nin | v13_geo | v14_geo |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in centers_in_pn:
        b = base["pn2021"]["per_center"][c].get("macro_auprc")
        row = [c, fmt_val(b)]
        for tag in CELLS:
            for variant_dict in [v13_cells, v14_cells]:
                ev = variant_dict.get(tag) or {}
                v = ev.get("pn2021", {}).get("per_center", {}).get(c, {}).get("macro_auprc")
                row.append(fmt_pp(v, b))
        L.append("| " + " | ".join(row) + " |")
    base_avg = base["pn2021"].get("avg_macro_auprc")
    row = ["**avg**", fmt_val(base_avg)]
    for tag in CELLS:
        for variant_dict in [v13_cells, v14_cells]:
            ev = variant_dict.get(tag) or {}
            v = ev.get("pn2021", {}).get("avg_macro_auprc")
            row.append(fmt_pp(v, base_avg))
    L.append("| " + " | ".join(row) + " |")
    L.append("")

    # Per-cell head-to-head v13 vs v14
    L.append("## Head-to-head: v14 CT v2 minus v13 vanilla (Δ in pp; positive = v14 wins)\n")
    L.append("### PN2021 per-center macro AUROC")
    L.append("| center | extra v14-v13 | nin v14-v13 | geo v14-v13 |")
    L.append("|---|---|---|---|")
    for c in centers_in_pn:
        row = [c]
        for tag in CELLS:
            v13 = (v13_cells.get(tag) or {}).get("pn2021", {}).get("per_center", {}).get(c, {}).get("macro_auroc")
            v14 = (v14_cells.get(tag) or {}).get("pn2021", {}).get("per_center", {}).get(c, {}).get("macro_auroc")
            row.append(fmt_pp(v14, v13))
        L.append("| " + " | ".join(row) + " |")
    row = ["**avg**"]
    for tag in CELLS:
        v13 = (v13_cells.get(tag) or {}).get("pn2021", {}).get("avg_macro_auroc")
        v14 = (v14_cells.get(tag) or {}).get("pn2021", {}).get("avg_macro_auroc")
        row.append(fmt_pp(v14, v13))
    L.append("| " + " | ".join(row) + " |")
    L.append("")

    # Per-class breakdown for v13 vs v14
    L.append("## Per-cell per-class AUROC delta (v14 - v13 in pp)\n")
    for tag in CELLS:
        cn = TAG_TO_CENTER[tag]
        L.append(f"### Cell `{tag}` ({cn})")
        L.append("| center | " + " | ".join(SUPER5) + " |")
        L.append("|---|" + "|".join(["---"] * len(SUPER5)) + "|")
        for c in centers_in_pn:
            row = [c]
            v13_pc = ((v13_cells.get(tag) or {}).get("pn2021", {})
                      .get("per_center", {}).get(c, {}).get("per_class", {}))
            v14_pc = ((v14_cells.get(tag) or {}).get("pn2021", {})
                      .get("per_center", {}).get(c, {}).get("per_class", {}))
            for cls in SUPER5:
                v13v = v13_pc.get(cls, {}).get("auroc")
                v14v = v14_pc.get(cls, {}).get("auroc")
                row.append(fmt_pp(v14v, v13v))
            L.append("| " + " | ".join(row) + " |")
        L.append("")

    # MIMIC
    L.append("## MIMIC zero-shot delta (Δ vs baseline, in pp)\n")
    L.append("| metric | baseline | v13_extra | v14_extra | v13_nin | v14_nin | v13_geo | v14_geo |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in ["macro_auroc", "macro_auprc"]:
        b = base.get("mimic_test", {}).get(m)
        row = [m, fmt_val(b)]
        for tag in CELLS:
            for variant_dict in [v13_cells, v14_cells]:
                ev = variant_dict.get(tag) or {}
                v = ev.get("mimic_test", {}).get(m)
                row.append(fmt_pp(v, b))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # PTBXL
    L.append("## PTBXL fold10 in-domain delta (gate: ≥ -0.50pp)\n")
    L.append("| metric | baseline | v13_extra | v14_extra | v13_nin | v14_nin | v13_geo | v14_geo |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in ["macro_auroc", "macro_auprc"]:
        b = base.get("ptbxl_test", {}).get(m)
        row = [m, fmt_val(b)]
        for tag in CELLS:
            for variant_dict in [v13_cells, v14_cells]:
                ev = variant_dict.get(tag) or {}
                v = ev.get("ptbxl_test", {}).get(m)
                row.append(fmt_pp(v, b))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # Per-cell decision summary
    L.append("## Per-cell decision summary (v13 vs v14, gate ≥+0.30pp AUROC AND ≥+0.50pp AUPRC)\n")
    L.append("| cell | v13 vs base AUROC Δ | v14 vs base AUROC Δ | v13 vs base AUPRC Δ | v14 vs base AUPRC Δ | winner |")
    L.append("|---|---|---|---|---|---|")
    for tag in CELLS:
        b_a = base["pn2021"].get("avg_macro_auroc")
        b_p = base["pn2021"].get("avg_macro_auprc")
        v13a = (v13_cells.get(tag) or {}).get("pn2021", {}).get("avg_macro_auroc")
        v14a = (v14_cells.get(tag) or {}).get("pn2021", {}).get("avg_macro_auroc")
        v13p = (v13_cells.get(tag) or {}).get("pn2021", {}).get("avg_macro_auprc")
        v14p = (v14_cells.get(tag) or {}).get("pn2021", {}).get("avg_macro_auprc")
        d13a = (v13a - b_a) * 100 if v13a is not None and b_a is not None else None
        d14a = (v14a - b_a) * 100 if v14a is not None and b_a is not None else None
        winner = "tie"
        if v13a is not None and v14a is not None:
            winner = "v14_ctv2" if v14a > v13a else "v13_vanilla"
        L.append(f"| {tag} | {fmt_pp(v13a, b_a)} | {fmt_pp(v14a, b_a)} | "
                 f"{fmt_pp(v13p, b_p)} | {fmt_pp(v14p, b_p)} | **{winner}** |")
    L.append("")

    L.append("## Interpretation\n")
    L.append("- v14 CT v2 sphere=5 wins per-cell PN2021 AUROC if Δ > 0 in 'v14_ctv2 minus v13_vanilla' table above.")
    L.append("- Pay attention to extra cell — v13 trust mask was only STTC due to MI/NORM under-trust; v14 added NORM. Expect biggest delta there.")
    L.append("- PTBXL fold10 must stay > -0.50pp for both variants.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(L))
    print(f"[done] wrote → {args.output}")


if __name__ == "__main__":
    main()
