"""5-way diff for AugMix × Adv combo ablation → markdown report.

Columns:
  - Baseline              (no augmix, no adv)
  - AugMix-time           (augmix_validation/half_s5/)
  - AugMix-latent         (augmix_validation_latent/half_s5/)
  - AugMix-time + Adv     (this run, time mode)
  - AugMix-latent + Adv   (optional Phase 3)

Key metric: Δ(augmix+adv) − Δ(augmix_alone) on MAIN5 avg macro AUROC.
  - ≥ +0.15pp  → adv samples additively help AugMix
  - |Δ| < 0.15 → redundant
  - ≤ -0.15pp  → adv conflicts with AugMix training

Usage:
    python scripts/augmix_adv_combo/compare.py [--time_adv_json PATH] [--latent_adv_json PATH]
"""
import argparse
import json
from pathlib import Path


DEFAULT_BASELINE = "/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json"
DEFAULT_AUGMIX_TIME = "/root/autodl-tmp/crosscenter_tierM_augmix/half_s5/eval_crosscenter.json"
DEFAULT_AUGMIX_LATENT = "/root/autodl-tmp/crosscenter_tierM_augmix_latent/half_s5/eval_crosscenter.json"
DEFAULT_TIME_ADV = "/root/autodl-tmp/crosscenter_tierM_augmix_adv/time_s5/eval_crosscenter.json"
DEFAULT_LATENT_ADV = "/root/autodl-tmp/crosscenter_tierM_augmix_adv/latent_s5/eval_crosscenter.json"
DEFAULT_OUT = "/root/ECG_adv_Gen/outputs/augmix_adv_combo/compare.md"

CLASSES = ["NSR", "STach", "AF", "IAVB", "LBBB", "RBBB"]

COL_LABELS = {
    "baseline": "Base",
    "augmix_time": "AM-T",
    "augmix_latent": "AM-L",
    "time_adv": "AM-T+Adv",
    "latent_adv": "AM-L+Adv",
}


def _find_train_result(eval_json_path):
    p = Path(eval_json_path).parent / "train_result.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _load(paths):
    runs = {}
    for key, p in paths.items():
        if p is None or not Path(p).exists():
            continue
        try:
            d = json.loads(Path(p).read_text())
        except Exception as e:
            print(f"[warn] failed to load {p}: {e}")
            continue
        runs[key] = {"eval": d, "train": _find_train_result(p), "path": p}
    return runs


def _ptbxl(run, metric):
    tr = run.get("train")
    if tr is None:
        return None
    return tr.get(metric)


def _main5(run, metric):
    return run["eval"].get("main_centers_avg", {}).get(metric)


def _fmt(v, w=6):
    if v is None:
        return " " * w
    return f"{v:.4f}"


def _pp(new, base):
    if new is None or base is None:
        return None
    return (new - base) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_json", default=DEFAULT_BASELINE)
    ap.add_argument("--augmix_time_json", default=DEFAULT_AUGMIX_TIME)
    ap.add_argument("--augmix_latent_json", default=DEFAULT_AUGMIX_LATENT)
    ap.add_argument("--time_adv_json", default=DEFAULT_TIME_ADV)
    ap.add_argument("--latent_adv_json", default=DEFAULT_LATENT_ADV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    paths = {
        "baseline": args.baseline_json,
        "augmix_time": args.augmix_time_json,
        "augmix_latent": args.augmix_latent_json,
        "time_adv": args.time_adv_json,
        "latent_adv": args.latent_adv_json,
    }
    runs = _load(paths)
    assert "baseline" in runs, f"baseline missing: {args.baseline_json}"
    assert "augmix_time" in runs, f"augmix-time missing: {args.augmix_time_json}"

    present_combo = []
    if "time_adv" in runs:
        present_combo.append("time_adv")
    if "latent_adv" in runs:
        present_combo.append("latent_adv")
    assert present_combo, "At least one of time_adv / latent_adv must exist"

    has_latent = "augmix_latent" in runs
    col_order = ["baseline", "augmix_time"]
    if has_latent:
        col_order.append("augmix_latent")
    col_order.extend(present_combo)

    base = runs["baseline"]["eval"]
    main_centers = base.get("main_centers",
                            ["chapman_shaoxing", "cpsc_2018",
                             "cpsc_2018_extra", "georgia", "ningbo"])
    all_centers = list(base.get("centers", {}).keys())

    lines = []
    lines.append("# AugMix × Adv Combo — 5-Way Comparison\n")
    for k in col_order:
        lines.append(f"- {COL_LABELS[k]} : `{runs[k]['path']}`")
    # Training cfg summary
    for k in ("augmix_time", "augmix_latent", "time_adv", "latent_adv"):
        if k not in runs or runs[k]["train"] is None:
            continue
        cfg = runs[k]["train"].get("config") or runs[k]["train"].get("args") or {}
        bits = []
        if "augmix_mode" in cfg:
            bits.append(f"mode={cfg['augmix_mode']}")
        if "augmix_prob" in cfg:
            bits.append(f"prob={cfg['augmix_prob']}")
        if "augmix_severity" in cfg:
            bits.append(f"s={cfg['augmix_severity']}")
        if "augmix_width" in cfg:
            bits.append(f"w={cfg['augmix_width']}")
        if "synth_center_npz" in cfg and cfg["synth_center_npz"]:
            bits.append(f"synth_ratio={cfg.get('synth_ratio', '?')}")
        if "seed" in cfg:
            bits.append(f"seed={cfg['seed']}")
        if bits:
            lines.append(f"- {COL_LABELS[k]} cfg: " + " ".join(bits))
    lines.append("")

    # Headline
    lines.append("## Headline\n")
    hdr = "| Metric | " + " | ".join(COL_LABELS[k] for k in col_order) + " |"
    sep = "|---|" + "---:|" * len(col_order)
    lines.append(hdr)
    lines.append(sep)

    def _vals(metric_func, metric_name):
        return [metric_func(runs[k], metric_name) if k in runs else None for k in col_order]

    for label, vals in [
        ("PTBXL test macro AUROC", _vals(_ptbxl, "test_macro_auroc")),
        ("PTBXL test macro AUPRC", _vals(_ptbxl, "test_macro_auprc")),
        ("MAIN5 avg macro AUROC",  _vals(_main5, "macro_auroc_avg")),
        ("MAIN5 avg macro AUPRC",  _vals(_main5, "macro_auprc_avg")),
    ]:
        row = f"| {label} |"
        for v in vals:
            row += f" {v:.4f} |" if v is not None else "   N/A   |"
        lines.append(row)
    lines.append("")

    # Δ vs baseline
    lines.append("## Δ vs Baseline (pp on MAIN5 AUROC/AUPRC)\n")
    lines.append("| Variant | MAIN5 AUROC Δ | MAIN5 AUPRC Δ | PTBXL AUROC Δ |")
    lines.append("|---|---:|---:|---:|")
    b_roc5 = _main5(runs["baseline"], "macro_auroc_avg")
    b_ap5 = _main5(runs["baseline"], "macro_auprc_avg")
    b_ptroc = _ptbxl(runs["baseline"], "test_macro_auroc")
    for k in col_order[1:]:
        n_roc = _pp(_main5(runs[k], "macro_auroc_avg"), b_roc5)
        n_ap = _pp(_main5(runs[k], "macro_auprc_avg"), b_ap5)
        n_pt = _pp(_ptbxl(runs[k], "test_macro_auroc"), b_ptroc)
        def f(x):
            return f"{x:+.2f}pp" if x is not None else "N/A"
        lines.append(f"| {COL_LABELS[k]} | {f(n_roc)} | {f(n_ap)} | {f(n_pt)} |")
    lines.append("")

    # Combo delta: (+adv) vs (no adv)
    lines.append("## Combo Δ — Adv on top of AugMix (MAIN5 AUROC/AUPRC, pp)\n")
    lines.append("| Pair | MAIN5 AUROC Δ | MAIN5 AUPRC Δ | Verdict |")
    lines.append("|---|---:|---:|---|")

    def _verdict(delta):
        if delta is None:
            return "N/A"
        if delta >= 0.15:
            return f"**✅ 叠加有效** (Δ={delta:+.2f}pp ≥ +0.15)"
        if delta <= -0.15:
            return f"**❌ 互相冲突** (Δ={delta:+.2f}pp ≤ -0.15)"
        return f"**❓ 冗余** (|Δ|<0.15pp 噪声)"

    if "time_adv" in runs:
        d_roc = _pp(_main5(runs["time_adv"], "macro_auroc_avg"),
                    _main5(runs["augmix_time"], "macro_auroc_avg"))
        d_ap = _pp(_main5(runs["time_adv"], "macro_auprc_avg"),
                   _main5(runs["augmix_time"], "macro_auprc_avg"))
        lines.append(f"| AM-T+Adv − AM-T | {f(d_roc)} | {f(d_ap)} | {_verdict(d_roc)} |")
    if "latent_adv" in runs and "augmix_latent" in runs:
        d_roc = _pp(_main5(runs["latent_adv"], "macro_auroc_avg"),
                    _main5(runs["augmix_latent"], "macro_auroc_avg"))
        d_ap = _pp(_main5(runs["latent_adv"], "macro_auprc_avg"),
                   _main5(runs["augmix_latent"], "macro_auprc_avg"))
        lines.append(f"| AM-L+Adv − AM-L | {f(d_roc)} | {f(d_ap)} | {_verdict(d_roc)} |")
    lines.append("")

    # Per-center AUROC across all 5 columns
    lines.append("## Per-Center Macro AUROC\n")
    hdr = "| Center | N |" + "".join(f" {COL_LABELS[k]} |" for k in col_order)
    sep = "|---|---:|" + "---:|" * len(col_order)
    lines.append(hdr)
    lines.append(sep)
    for c in all_centers:
        if c not in base["centers"]:
            continue
        row_vals = []
        n = base["centers"][c].get("n_scored_records", "?")
        for k in col_order:
            if k not in runs:
                row_vals.append("N/A")
                continue
            ctr = runs[k]["eval"]["centers"].get(c)
            if ctr is None:
                row_vals.append("N/A")
            else:
                row_vals.append(f"{ctr['macro']['macro_auroc']:.4f}")
        flag = " **★**" if c in main_centers else ""
        lines.append(f"| {c}{flag} | {n} | " + " | ".join(row_vals) + " |")
    lines.append("")
    lines.append("*★ = main-5 center (averaged into headline MAIN5)*\n")

    # Per-class Δ on main centers
    lines.append("## Per-Center × Per-Class AUROC (Δ vs Baseline, pp)\n")
    hdr = "| Center | Variant |" + "".join(f" {cls} |" for cls in CLASSES)
    sep = "|---|---|" + "---:|" * len(CLASSES)
    lines.append(hdr)
    lines.append(sep)
    for c in main_centers:
        if c not in base["centers"]:
            continue
        b_pc = base["centers"][c].get("per_class", {})
        for k in col_order[1:]:
            if k not in runs:
                continue
            ctr = runs[k]["eval"]["centers"].get(c)
            if ctr is None:
                continue
            pc = ctr.get("per_class", {})
            cells = []
            for cls in CLASSES:
                b = (b_pc.get(cls) or {}).get("auroc")
                a = (pc.get(cls) or {}).get("auroc")
                if b is None or a is None:
                    cells.append("N/A")
                else:
                    d = (a - b) * 100
                    cells.append(f"{a:.3f}({d:+.2f})")
            lines.append(f"| {c} | {COL_LABELS[k]} | " + " | ".join(cells) + " |")
    lines.append("")

    # Final verdict
    lines.append("## Final Verdict\n")
    if "time_adv" in runs:
        d_t_base = _pp(_main5(runs["time_adv"], "macro_auroc_avg"), b_roc5)
        d_t_over_am = _pp(_main5(runs["time_adv"], "macro_auroc_avg"),
                          _main5(runs["augmix_time"], "macro_auroc_avg"))
        lines.append(f"**AM-T + Adv** Δ vs baseline = {d_t_base:+.2f}pp, "
                     f"Δ vs AM-T alone = {d_t_over_am:+.2f}pp → {_verdict(d_t_over_am)}")
    if "latent_adv" in runs and "augmix_latent" in runs:
        d_l_base = _pp(_main5(runs["latent_adv"], "macro_auroc_avg"), b_roc5)
        d_l_over_am = _pp(_main5(runs["latent_adv"], "macro_auroc_avg"),
                          _main5(runs["augmix_latent"], "macro_auroc_avg"))
        lines.append("")
        lines.append(f"**AM-L + Adv** Δ vs baseline = {d_l_base:+.2f}pp, "
                     f"Δ vs AM-L alone = {d_l_over_am:+.2f}pp → {_verdict(d_l_over_am)}")
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[compare] wrote {out_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
