"""3-way diff of baseline / online / offline eval_crosscenter.json → markdown report.

Usage:
    python scripts/online_vs_offline/compare.py

Schema (from eval_crosscenter_tierM.py):
    d["centers"][name]["macro"]["macro_auroc" | "macro_auprc"]
    d["centers"][name]["per_class"][cls]["auroc"]
    d["centers"][name]["n_scored_records"]
    d["main_centers_avg"]["macro_auroc_avg" | "macro_auprc_avg"]
    d["main_centers"]                 # list of 5 center names
"""
import argparse
import json
from pathlib import Path


DEFAULT_BASELINE = "/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json"
# Online v4 chosen over v5: v5's best_model.pt is identical (MD5) to baseline
# (online training never exceeded baseline quick_eval so bootstrap was never
# overwritten). v4 is actually-trained and has the best MAIN5 among v3/v4.
DEFAULT_ONLINE   = "/root/autodl-tmp/crosscenter_tierM_online/eval_crosscenter_v4.json"
# Offline best_model.pt is byte-identical to baseline best_model.pt (confirmed
# via MD5 hash 8561f22563...). Offline's quick_eval never crossed baseline's
# 0.9483 floor during 20 epochs, so best-selection kept the bootstrap copy.
# We alias baseline eval JSON as "offline" for the comparison.
DEFAULT_OFFLINE  = "/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json"
DEFAULT_OUT      = "/root/ECG_adv_Gen/outputs/online_vs_offline/compare.md"

CLASSES = ["NSR", "STach", "AF", "IAVB", "LBBB", "RBBB"]


def _delta_pp(new, base):
    if new is None or base is None:
        return None, "  N/A"
    d = (new - base) * 100
    return d, f"{d:+.2f}pp"


def _find_train_result(eval_json_path, override_dir=None):
    """Look for train_result.json next to the eval json, or in override_dir."""
    if override_dir:
        p = Path(override_dir) / "train_result.json"
    else:
        p = Path(eval_json_path).parent / "train_result.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_json", default=DEFAULT_BASELINE)
    ap.add_argument("--online_json",   default=DEFAULT_ONLINE)
    ap.add_argument("--offline_json",  default=DEFAULT_OFFLINE)
    ap.add_argument("--offline_train_result",
                    default="/root/autodl-tmp/crosscenter_tierM_offline/train_result.json",
                    help="Separate path for offline's train_result.json (since offline's "
                         "eval json is aliased to baseline's).")
    ap.add_argument("--out",           default=DEFAULT_OUT)
    args = ap.parse_args()

    base = json.loads(Path(args.baseline_json).read_text())
    onl  = json.loads(Path(args.online_json).read_text())
    off  = json.loads(Path(args.offline_json).read_text())
    off_tr = None
    if Path(args.offline_train_result).exists():
        off_tr = json.loads(Path(args.offline_train_result).read_text())

    # Detect "offline is baseline alias" case: happens when offline's best_model never
    # beat baseline's quick_eval, so bootstrap-copy remained as best_model.pt.
    offline_is_baseline_alias = Path(args.offline_json).resolve() == Path(args.baseline_json).resolve()

    main_centers = base.get(
        "main_centers",
        ["chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "georgia", "ningbo"],
    )
    all_centers = list(base.get("centers", {}).keys())

    lines = []
    lines.append("# Online vs Offline Adversarial Training — 3-way Diff (tierM 6-class)\n")
    lines.append(f"- Baseline       : `{args.baseline_json}`")
    lines.append(f"- Online (v4)    : `{args.online_json}`")
    lines.append(f"- Offline        : `{args.offline_json}`")
    if offline_is_baseline_alias:
        lines.append(
            "  - **NOTE**: offline's `best_model.pt` is byte-identical (MD5) to baseline's — "
            "offline training never beat baseline quick_eval over 20 epochs, so the "
            "best-selection logic retained the bootstrap baseline copy. Eval is aliased "
            "to baseline's eval JSON because numbers would be bitwise identical."
        )
    if off_tr:
        lines.append(f"- Offline cfg    : n_adv_total={off_tr.get('args', {}).get('n_adv_total')} "
                     f"accept=[{off_tr.get('args', {}).get('accept_prob_low')},"
                     f"{off_tr.get('args', {}).get('accept_prob_high')}] "
                     f"n_epochs={off_tr.get('args', {}).get('n_epochs')}")
        lines.append(f"- Offline Phase0 : accepted={off_tr.get('phase0_gen_stats', {}).get('accepted')} "
                     f"attempted={off_tr.get('phase0_gen_stats', {}).get('attempted')} "
                     f"rate={off_tr.get('phase0_accept_rate')} "
                     f"buffer_size={off_tr.get('phase0_buffer_size')}")
        last_qe = off_tr.get('last_epoch_quick_eval')
        if last_qe:
            last_auroc = last_qe.get('avg_macro_auroc')
            best_auroc = off_tr.get('best_avg_macro_auroc')
            lines.append(f"- Offline quick  : best={best_auroc}  last_ep={last_auroc}  "
                         f"(last_ep Δ vs baseline = {(last_auroc - best_auroc)*100:+.2f}pp)")
    lines.append("")

    # ── Headline ────────────────────────────────────────────────────────────
    b_roc_avg = base["main_centers_avg"]["macro_auroc_avg"]
    o_roc_avg = onl["main_centers_avg"]["macro_auroc_avg"]
    f_roc_avg = off["main_centers_avg"]["macro_auroc_avg"]
    b_ap_avg  = base["main_centers_avg"]["macro_auprc_avg"]
    o_ap_avg  = onl["main_centers_avg"]["macro_auprc_avg"]
    f_ap_avg  = off["main_centers_avg"]["macro_auprc_avg"]

    d_on_roc, s_on_roc = _delta_pp(o_roc_avg, b_roc_avg)
    d_off_roc, s_off_roc = _delta_pp(f_roc_avg, b_roc_avg)
    d_on_ap, s_on_ap = _delta_pp(o_ap_avg, b_ap_avg)
    d_off_ap, s_off_ap = _delta_pp(f_ap_avg, b_ap_avg)
    # online vs offline direct diff
    d_on_vs_off_roc, s_on_vs_off_roc = _delta_pp(o_roc_avg, f_roc_avg)
    d_on_vs_off_ap, s_on_vs_off_ap = _delta_pp(o_ap_avg, f_ap_avg)

    lines.append("## Headline — MAIN5 average\n")
    lines.append("| Metric | Baseline | Online v4 | Offline | Δ Online | Δ Offline | Online − Offline |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| MAIN5 avg macro AUROC | {b_roc_avg:.4f} | {o_roc_avg:.4f} | {f_roc_avg:.4f} "
        f"| {s_on_roc} | {s_off_roc} | {s_on_vs_off_roc} |"
    )
    lines.append(
        f"| MAIN5 avg macro AUPRC | {b_ap_avg:.4f} | {o_ap_avg:.4f} | {f_ap_avg:.4f} "
        f"| {s_on_ap} | {s_off_ap} | {s_on_vs_off_ap} |"
    )
    lines.append("")

    # ── Per-Center ──────────────────────────────────────────────────────────
    lines.append("## Per-Center Macro AUROC / AUPRC\n")
    lines.append("| Center | N | Base AUROC | Onl AUROC | Off AUROC | ΔOn | ΔOff | Base AUPRC | Onl AUPRC | Off AUPRC | ΔOn | ΔOff |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in all_centers:
        if c not in base["centers"] or c not in onl["centers"] or c not in off["centers"]:
            continue
        b_c = base["centers"][c]
        o_c = onl["centers"][c]
        f_c = off["centers"][c]
        n = b_c.get("n_scored_records", "?")
        bmr = b_c["macro"]["macro_auroc"]
        omr = o_c["macro"]["macro_auroc"]
        fmr = f_c["macro"]["macro_auroc"]
        bmp = b_c["macro"]["macro_auprc"]
        omp = o_c["macro"]["macro_auprc"]
        fmp = f_c["macro"]["macro_auprc"]
        _, s_on_r = _delta_pp(omr, bmr)
        _, s_off_r = _delta_pp(fmr, bmr)
        _, s_on_p = _delta_pp(omp, bmp)
        _, s_off_p = _delta_pp(fmp, bmp)
        flag = " **★**" if c in main_centers else ""
        lines.append(
            f"| {c}{flag} | {n} "
            f"| {bmr:.4f} | {omr:.4f} | {fmr:.4f} | {s_on_r} | {s_off_r} "
            f"| {bmp:.4f} | {omp:.4f} | {fmp:.4f} | {s_on_p} | {s_off_p} |"
        )
    lines.append("")
    lines.append("*★ = main-5 center (averaged into headline MAIN5)*\n")

    # ── Per-Center × Per-Class AUROC ────────────────────────────────────────
    for tag, data in [("Online v4", onl), ("Offline", off)]:
        lines.append(f"## Per-Center × Per-Class AUROC — {tag} (value (Δ vs base in pp))\n")
        header = "| Center | " + " | ".join(CLASSES) + " |"
        sep = "|---|" + "---:|" * len(CLASSES)
        lines.append(header)
        lines.append(sep)
        for c in main_centers:
            if c not in base["centers"] or c not in data["centers"]:
                continue
            b_pc = base["centers"][c].get("per_class", {})
            d_pc = data["centers"][c].get("per_class", {})
            cells = []
            for cls in CLASSES:
                b = b_pc.get(cls, {}).get("auroc") if b_pc.get(cls) else None
                a = d_pc.get(cls, {}).get("auroc") if d_pc.get(cls) else None
                if b is None or a is None:
                    cells.append("N/A")
                else:
                    d = (a - b) * 100
                    sign = "+" if d >= 0 else ""
                    cells.append(f"{a:.3f} ({sign}{d:.2f})")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")

    # ── Verdict ─────────────────────────────────────────────────────────────
    lines.append("## Verdict\n")
    if d_on_roc is None or d_off_roc is None:
        lines.append("**INCONCLUSIVE** — missing MAIN5 metric in one of the runs.")
    else:
        gap = d_on_roc - d_off_roc    # positive = online ahead; negative = offline ahead
        summary_lines = [
            f"- Online (v4) Δ vs baseline : **{d_on_roc:+.2f}pp** MAIN5 AUROC",
            f"- Offline     Δ vs baseline : **{d_off_roc:+.2f}pp** MAIN5 AUROC",
            f"- Online − Offline gap      : **{gap:+.2f}pp** (positive = online ahead)",
            "",
        ]
        lines.extend(summary_lines)

        # Decision rules
        if offline_is_baseline_alias:
            last_qe_val = None
            if off_tr and off_tr.get('last_epoch_quick_eval'):
                last_qe_val = off_tr['last_epoch_quick_eval'].get('avg_macro_auroc')
            baseline_qe = off_tr.get('baseline_quick_eval', {}).get('avg_macro_auroc') if off_tr else None
            last_drop_pp = (last_qe_val - baseline_qe) * 100 if (last_qe_val and baseline_qe) else None
            lines.append(
                "**❌ BOTH INEFFECTIVE — adversarial training fails on tierM 6-class.** "
                f"Offline's `best_model.pt` never surpassed baseline quick_eval ({baseline_qe}) "
                f"over 20 epochs, ending at {last_qe_val} "
                f"(last-epoch Δ={last_drop_pp:+.2f}pp). Online v4 showed the same pattern "
                f"but narrowly crossed quick_eval threshold mid-training "
                f"(ended full-eval MAIN5 = baseline {d_on_roc:+.2f}pp). Neither strategy "
                f"produces a deployable improvement over baseline."
            )
            lines.append("")
            lines.append(
                "**Key learnings from this ablation**:"
            )
            lines.append(
                "1. *Labeling the ablation "
                "\"online vs offline\" is misleading* — both strategies fail here; the real "
                "question is *why AdvDiff+tierM fails* (likely: guidance_scale=0.03 too weak, "
                "acceptance_range=[0.5,0.6] too tight, or tierM 6-class is already well-covered "
                "by PTBXL without needing adversarial augmentation)."
            )
            lines.append(
                "2. *Online 3/5 runs (v1, v2, v5) also never beat "
                "baseline quick_eval* — deployed `best_model.pt` = bootstrap for those runs. "
                "Only v3/v4 actually output trained weights, both slightly regressing (-0.03 to "
                "-0.05pp MAIN5 on full eval)."
            )
            lines.append(
                "3. *Offline's last-epoch model regressed by ~1pp* "
                "on quick_eval — offline training actively hurts if checkpoint selection is "
                "disabled. Suggests the 1000 generated samples + augmix injection provides "
                "OOD noise rather than useful adversarial signal for this tierM setup."
            )
            lines.append(
                "4. *Next-step recommendation*: abandon this adversarial "
                "direction for tierM, pivot to ECGTwin center-conditioning (the direction "
                "discussed earlier) — that's a paradigm change more likely to improve cross-center."
            )
        elif (abs(d_on_roc) < 0.3) and (abs(d_off_roc) < 0.3):
            lines.append(
                "**❓ AMBIGUOUS — both conditions near baseline noise** "
                f"(|Δ_on|={abs(d_on_roc):.2f}pp, |Δ_off|={abs(d_off_roc):.2f}pp "
                f"both < 0.30pp). The adversarial strategy itself may be ineffective "
                f"on this tierM setup; single seed insufficient to differentiate "
                f"online vs offline."
            )
        elif gap >= 0.30:
            lines.append(
                f"**✅ ONLINE WINS** — gap {gap:+.2f}pp ≥ +0.30pp. "
                f"The per-epoch co-evolution of victim + generator yields measurably "
                f"more cross-center gain than one-shot offline gen."
            )
        elif gap <= -0.30:
            lines.append(
                f"**✅ OFFLINE WINS** — gap {gap:+.2f}pp ≤ -0.30pp. "
                f"Online's adaptive regeneration does NOT pay off; offline "
                f"(frozen baseline victim, static budget) matches or exceeds online."
            )
        else:
            lines.append(
                f"**≈ TIE** — gap {gap:+.2f}pp within ±0.30pp. Both strategies "
                f"deliver similar gain over baseline; single-seed noise insufficient "
                f"to declare a winner."
            )

    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[compare] wrote {out_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
