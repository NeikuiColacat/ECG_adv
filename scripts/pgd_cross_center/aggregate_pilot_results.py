"""Aggregate pilot iter 4 results (Plan Rev 13.2).

Runs Stage 3 eval (eval_crosscenter.py with --exclude_ref_ids) for each cell
+ baseline, then reports 5 tables and the decision-gate verdict:

  Table A — PN2021 per-center macro AUROC delta vs baseline
  Table B — Per-cell PN2021 per-class AUROC delta (per-center, per-class)
  Table C — MIMIC zero-shot delta
  Table D — PTBXL fold10 in-domain delta (must be >= -0.50pp)
  Table E — Per-cell training quick_eval ASR + Einthoven trace summary

Decision gate: ≥2/3 cells pass (macro AUROC Δ > +0.30pp AND macro AUPRC Δ > +0.50pp)
              AND PTBXL fold10 Δ ≥ -0.50pp.

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/pgd_cross_center/aggregate_pilot_results.py \
      --out_root /root/autodl-tmp/synth_anchored_super5_v13
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent.parent
PYTHON = "/root/miniforge3/envs/ECGTwin/bin/python"
EVAL_SCRIPT = str(REPO / "scripts/triple_labels/eval_crosscenter.py")

CELLS = ["extra", "nin", "geo"]
TAG_TO_CENTER = {"extra": "cpsc_2018_extra", "nin": "ningbo", "geo": "georgia"}
SUPER5 = ["CD", "HYP", "MI", "NORM", "STTC"]


def fmt_delta(after: Optional[float], before: Optional[float]) -> str:
    if after is None or before is None:
        return "  n/a"
    if isinstance(after, float) and math.isnan(after):
        return "  nan"
    if isinstance(before, float) and math.isnan(before):
        return "  nan"
    pp = (after - before) * 100.0
    sign = "+" if pp >= 0 else ""
    return f"{sign}{pp:.2f}pp"


def fmt_val(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.4f}"


def run_eval(model_dir: str, output_path: str,
              exclude_meta_paths: List[str]) -> Dict[str, Any]:
    if os.path.exists(output_path):
        print(f"[eval] cache hit: {output_path}")
    else:
        cmd = [PYTHON, EVAL_SCRIPT, "--scheme", "super5",
               "--model_dir", model_dir, "--output_path", output_path]
        meta_paths = [m for m in exclude_meta_paths if os.path.exists(m)]
        if meta_paths:
            cmd.extend(["--exclude_ref_ids"] + meta_paths)
        print(f"[eval] {' '.join(cmd)}", flush=True)
        subprocess.check_call(cmd)
    with open(output_path) as f:
        return json.load(f)


def load_training_log(cell_dir: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(cell_dir, "training_log.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_early_stop(cell_dir: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(cell_dir, "early_stop_info.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def best_qe_from_log(log: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best = None; best_v = -1.0
    for e in log.get("epochs", []):
        qe = e.get("quick_eval")
        if qe and qe.get("avg_macro_auroc") is not None:
            v = qe["avg_macro_auroc"]
            if isinstance(v, float) and (v != v):
                continue
            if v > best_v:
                best_v = v; best = (e["epoch"], qe)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--baseline_dir", default="/root/autodl-tmp/triple_labels/super5")
    ap.add_argument("--meta_dir", default="/root/autodl-tmp/center_token_super5")
    ap.add_argument("--report_md", default=None)
    ap.add_argument("--cell_suffix", default="k200",
                    help="Cell directory suffix (e.g. k200 for iter4, k400 for iter4b). "
                         "Used to find {tag}_{suffix} cells under out_root and meta files.")
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--skip_eval", action="store_true",
                    help="Use existing eval_with_exclude.json (don't rerun).")
    args = ap.parse_args()

    out_root = args.out_root

    # ── Step 1: baseline eval (with all 3 ref-pool meta exclusions) ─────────
    base_eval_path = os.path.join(args.baseline_dir,
                                   f"eval_with_exclude_3cells_{args.cell_suffix}.json")
    meta_paths_all = [os.path.join(args.meta_dir, f"{t}_{args.cell_suffix}.meta.json")
                      for t in CELLS]
    if args.skip_baseline and os.path.exists(base_eval_path):
        with open(base_eval_path) as f:
            base = json.load(f)
        print(f"[baseline] cache hit: {base_eval_path}")
    elif args.skip_eval and os.path.exists(base_eval_path):
        with open(base_eval_path) as f:
            base = json.load(f)
    else:
        base = run_eval(args.baseline_dir, base_eval_path, meta_paths_all)

    # ── Step 2: per-cell eval ───────────────────────────────────────────────
    cell_evals: Dict[str, Dict[str, Any]] = {}
    cell_logs: Dict[str, Dict[str, Any]] = {}
    cell_es: Dict[str, Dict[str, Any]] = {}
    for tag in CELLS:
        cell_dir = os.path.join(out_root, f"{tag}_{args.cell_suffix}")
        if not os.path.isdir(cell_dir):
            print(f"[skip] {tag}: dir missing — {cell_dir}")
            continue
        if not os.path.exists(os.path.join(cell_dir, "best_model.pt")):
            print(f"[skip] {tag}: best_model.pt missing")
            continue
        meta = os.path.join(args.meta_dir, f"{tag}_{args.cell_suffix}.meta.json")
        out_eval = os.path.join(cell_dir, "eval_with_exclude.json")
        if not args.skip_eval:
            cell_evals[tag] = run_eval(cell_dir, out_eval, [meta])
        elif os.path.exists(out_eval):
            with open(out_eval) as f:
                cell_evals[tag] = json.load(f)
        cell_logs[tag] = load_training_log(cell_dir)
        cell_es[tag] = load_early_stop(cell_dir)

    # ── Step 3: build markdown report ───────────────────────────────────────
    lines: List[str] = []
    lines.append("# Synth-anchored Super5 pilot iter 4 — Plan Rev 13.2 results\n")
    lines.append("Generation scope: NORM/MI/STTC (HYP/CD digital-GT 0/3 fail).")
    lines.append("Decision gate: ≥2/3 cells pass `macro AUROC Δ > +0.30pp` AND `macro AUPRC Δ > +0.50pp` "
                  "AND `PTBXL fold10 Δ ≥ -0.50pp`.\n")

    # Cell summary
    lines.append("## Cell summary\n")
    lines.append("| cell | center | epoch_run | early_stop | best_epoch | best_QE_AUROC |")
    lines.append("|---|---|---|---|---|---|")
    for tag in CELLS:
        cn = TAG_TO_CENTER[tag]
        es = cell_es.get(tag, {}) or {}
        n_ep = es.get("n_epochs_run", "n/a")
        early = es.get("early_stopped", "n/a")
        best_ep = es.get("best_epoch", "n/a")
        best_v = es.get("best_metric", "n/a")
        lines.append(f"| {tag} | {cn} | {n_ep} | {early} | {best_ep} | {best_v} |")
    lines.append("")

    # Table A — PN2021 per-center macro AUROC delta
    lines.append("## Table A — PN2021 per-center macro AUROC + AUPRC delta vs baseline\n")
    centers_in_pn = list(base.get("pn2021", {}).get("per_center", {}).keys())
    lines.append("### AUROC")
    lines.append("| center | baseline | extra | nin | geo |")
    lines.append("|---|---|---|---|---|")
    for c in centers_in_pn:
        b = base["pn2021"]["per_center"][c].get("macro_auroc")
        row = [c, fmt_val(b)]
        for tag in CELLS:
            ev = cell_evals.get(tag, {}).get("pn2021", {}).get("per_center", {}).get(c, {})
            row.append(fmt_delta(ev.get("macro_auroc"), b))
        lines.append("| " + " | ".join(row) + " |")
    base_avg_auroc = base["pn2021"].get("avg_macro_auroc")
    row = ["**avg**", fmt_val(base_avg_auroc)]
    for tag in CELLS:
        v = cell_evals.get(tag, {}).get("pn2021", {}).get("avg_macro_auroc")
        row.append(fmt_delta(v, base_avg_auroc))
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("### AUPRC")
    lines.append("| center | baseline | extra | nin | geo |")
    lines.append("|---|---|---|---|---|")
    for c in centers_in_pn:
        b = base["pn2021"]["per_center"][c].get("macro_auprc")
        row = [c, fmt_val(b)]
        for tag in CELLS:
            ev = cell_evals.get(tag, {}).get("pn2021", {}).get("per_center", {}).get(c, {})
            row.append(fmt_delta(ev.get("macro_auprc"), b))
        lines.append("| " + " | ".join(row) + " |")
    base_avg_auprc = base["pn2021"].get("avg_macro_auprc")
    row = ["**avg**", fmt_val(base_avg_auprc)]
    for tag in CELLS:
        v = cell_evals.get(tag, {}).get("pn2021", {}).get("avg_macro_auprc")
        row.append(fmt_delta(v, base_avg_auprc))
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Table B — per-cell per-class
    lines.append("## Table B — Per-cell PN2021 per-class AUROC delta\n")
    for tag in CELLS:
        if tag not in cell_evals:
            continue
        cn = TAG_TO_CENTER[tag]
        lines.append(f"### Cell `{tag}` (training center = {cn})")
        lines.append("")
        lines.append("| center | " + " | ".join(SUPER5) + " |")
        lines.append("|---|" + "|".join(["---"] * len(SUPER5)) + "|")
        for c in centers_in_pn:
            row = [c]
            base_pc = base["pn2021"]["per_center"].get(c, {}).get("per_class", {})
            cur_pc = cell_evals[tag].get("pn2021", {}).get("per_center", {}).get(c, {}).get("per_class", {})
            for cls in SUPER5:
                bv = base_pc.get(cls, {}).get("auroc")
                cv = cur_pc.get(cls, {}).get("auroc")
                row.append(fmt_delta(cv, bv))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Table C — MIMIC
    lines.append("## Table C — MIMIC zero-shot delta\n")
    lines.append("| metric | baseline | extra | nin | geo |")
    lines.append("|---|---|---|---|---|")
    for metric in ["macro_auroc", "macro_auprc"]:
        b = base.get("mimic_test", {}).get(metric)
        row = [metric, fmt_val(b)]
        for tag in CELLS:
            v = cell_evals.get(tag, {}).get("mimic_test", {}).get(metric)
            row.append(fmt_delta(v, b))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Table D — PTBXL fold10
    lines.append("## Table D — PTBXL fold10 in-domain delta (gate: ≥ -0.50pp)\n")
    lines.append("| metric | baseline | extra | nin | geo |")
    lines.append("|---|---|---|---|---|")
    for metric in ["macro_auroc", "macro_auprc"]:
        b = base.get("ptbxl_test", {}).get(metric)
        row = [metric, fmt_val(b)]
        for tag in CELLS:
            v = cell_evals.get(tag, {}).get("ptbxl_test", {}).get(metric)
            row.append(fmt_delta(v, b))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Table E — gate trace
    lines.append("## Table E — PGD gates trace (last 5 logged epochs / cell)\n")
    for tag in CELLS:
        log = cell_logs.get(tag)
        if log is None:
            continue
        lines.append(f"### `{tag}`")
        lines.append("")
        lines.append("| ep | asr | einthoven_p95 | medical_pass | buf | val | train |")
        lines.append("|---|---|---|---|---|---|---|")
        for e in log.get("epochs", [])[-5:]:
            lines.append(
                f"| {e.get('epoch')} | {e.get('asr_overall')} | "
                f"{e.get('einthoven_p95')} | {not e.get('buffer_skipped', False)} | "
                f"{e.get('buffer_size')} | {e.get('val_loss')} | {e.get('train_loss')} |"
            )
        lines.append("")

    # Decision gate
    lines.append("## Decision gate\n")
    pass_count = 0
    detail_rows = []
    for tag in CELLS:
        if tag not in cell_evals:
            detail_rows.append(f"- **{tag}**: not run")
            continue
        b_auroc = base["pn2021"].get("avg_macro_auroc")
        b_auprc = base["pn2021"].get("avg_macro_auprc")
        c_auroc = cell_evals[tag].get("pn2021", {}).get("avg_macro_auroc")
        c_auprc = cell_evals[tag].get("pn2021", {}).get("avg_macro_auprc")
        b_ptb = base.get("ptbxl_test", {}).get("macro_auroc")
        c_ptb = cell_evals[tag].get("ptbxl_test", {}).get("macro_auroc")

        def _pp(a, b):
            if a is None or b is None:
                return None
            if isinstance(a, float) and (a != a):
                return None
            if isinstance(b, float) and (b != b):
                return None
            return (a - b) * 100.0

        d_auroc_pp = _pp(c_auroc, b_auroc)
        d_auprc_pp = _pp(c_auprc, b_auprc)
        d_ptb_pp = _pp(c_ptb, b_ptb)
        cell_pass = (d_auroc_pp is not None and d_auroc_pp > 0.30
                     and d_auprc_pp is not None and d_auprc_pp > 0.50
                     and d_ptb_pp is not None and d_ptb_pp >= -0.50)
        if cell_pass:
            pass_count += 1
        verdict = "✅ PASS" if cell_pass else "❌ FAIL"
        detail_rows.append(
            f"- **{tag}**: AUROC Δ={d_auroc_pp:+.2f}pp" if d_auroc_pp is not None else f"- **{tag}**: AUROC Δ=n/a"
        )
        detail_rows[-1] += (
            f" | AUPRC Δ={d_auprc_pp:+.2f}pp" if d_auprc_pp is not None else " | AUPRC Δ=n/a"
        )
        detail_rows[-1] += (
            f" | PTBXL Δ={d_ptb_pp:+.2f}pp" if d_ptb_pp is not None else " | PTBXL Δ=n/a"
        )
        detail_rows[-1] += f" → {verdict}"

    lines.extend(detail_rows)
    lines.append("")
    lines.append(f"**Summary**: {pass_count}/3 cells pass.")
    if pass_count >= 2:
        lines.append("→ Recommended: 5-center confirmatory + Issue #24 ablations + memorization audit.")
    elif pass_count == 1:
        lines.append("→ Borderline. Per-center signal exists; consider 5-center confirmatory or pool scale 300→1200.")
    else:
        lines.append("→ STOP-AND-ANALYZE: scale Stage 1 pool 300→1200, rerun once. "
                     "If still fail, lock as ECGTwin vocab/manifold limitation.")
    lines.append("")

    out_md = args.report_md or os.path.join(out_root, "pilot_iter4_report.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[done] wrote → {out_md}")
    print()
    print("\n".join(lines[-(20):]))


if __name__ == "__main__":
    main()
