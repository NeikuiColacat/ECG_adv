"""Diff baseline vs AugMix eval_crosscenter.json → markdown report.

Usage:
    python scripts/augmix_validation/compare_baseline.py \
        [--baseline_json PATH] [--augmix_json PATH] [--latent_json PATH] [--out PATH]

When `--latent_json` is provided (or its default path exists), produces a 3-way
table (baseline / time-domain / latent-space). Otherwise produces 2-way.

Schema (from eval_crosscenter_tierM.py):
    d["centers"][name]["macro"]["macro_auroc" | "macro_auprc"]
    d["centers"][name]["per_class"][cls]["auroc"]
    d["centers"][name]["n_scored_records"]
    d["main_centers_avg"]["macro_auroc_avg" | "macro_auprc_avg"]
    d["main_centers"]  # list of 5 center names
"""
import argparse
import json
from pathlib import Path


DEFAULT_BASELINE = "/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json"
DEFAULT_AUGMIX = "/root/autodl-tmp/crosscenter_tierM_augmix/half_s5/eval_crosscenter.json"
DEFAULT_LATENT = "/root/autodl-tmp/crosscenter_tierM_augmix_latent/half_s5/eval_crosscenter.json"
DEFAULT_OUT = "/root/ECG_adv_Gen/outputs/augmix_validation/compare.md"

CLASSES = ["NSR", "STach", "AF", "IAVB", "LBBB", "RBBB"]


def _delta_pp(new, base):
    if new is None or base is None:
        return None, "  N/A"
    d = (new - base) * 100
    return d, f"{d:+.2f}pp"


def _find_train_result(eval_json_path):
    p = Path(eval_json_path).parent / "train_result.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _load_all(paths_dict):
    """paths_dict: {'baseline': path, 'time': path, 'latent': path | None}
    Returns {'baseline': {'eval': d, 'train': t}, ...} (skipping None/missing)."""
    out = {}
    for key, p in paths_dict.items():
        if p is None or not Path(p).exists():
            continue
        try:
            d = json.loads(Path(p).read_text())
        except Exception as e:
            print(f"[warn] failed to load {p}: {e}")
            continue
        out[key] = {
            "eval": d,
            "train": _find_train_result(p),
            "path": p,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_json", default=DEFAULT_BASELINE)
    ap.add_argument("--augmix_json", default=DEFAULT_AUGMIX)
    ap.add_argument("--latent_json", default=DEFAULT_LATENT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    runs = _load_all({
        "baseline": args.baseline_json,
        "time":     args.augmix_json,
        "latent":   args.latent_json,
    })
    assert "baseline" in runs, f"baseline missing: {args.baseline_json}"
    assert "time" in runs, f"time-domain missing: {args.augmix_json}"
    has_latent = "latent" in runs

    base = runs["baseline"]["eval"]
    main_centers = base.get("main_centers",
                            ["chapman_shaoxing", "cpsc_2018",
                             "cpsc_2018_extra", "georgia", "ningbo"])
    all_centers = list(base.get("centers", {}).keys())

    # Headline numbers
    def _ptbxl(run_key, metric):
        tr = runs[run_key]["train"]
        return tr.get(metric) if tr else None

    def _main5(run_key, metric):
        return runs[run_key]["eval"]["main_centers_avg"].get(metric)

    lines = []
    title = "3-Way" if has_latent else "2-Way"
    lines.append(f"# AugMix Validation — {title} Comparison\n")
    lines.append(f"- Baseline : `{runs['baseline']['path']}`")
    lines.append(f"- Time-AugMix : `{runs['time']['path']}`")
    if has_latent:
        lines.append(f"- Latent-AugMix : `{runs['latent']['path']}`")
    for key in ("time", "latent"):
        if key in runs and runs[key]["train"]:
            cfg = runs[key]["train"].get("config", {})
            lines.append(f"- {key} cfg: mode={cfg.get('augmix_mode', 'time' if key=='time' else 'latent')} "
                         f"prob={cfg.get('augmix_prob')} "
                         f"severity={cfg.get('augmix_severity')} "
                         f"width={cfg.get('augmix_width')} "
                         f"seed={cfg.get('seed')}")
    lines.append("")

    # Headline table
    lines.append("## Headline\n")
    if has_latent:
        lines.append("| Metric | Baseline | Time | Latent | Δ Time | Δ Latent | Δ Latent − Time |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| Metric | Baseline | Time | Δ (pp) |")
        lines.append("|---|---:|---:|---:|")

    def _row(label, b_val, t_val, l_val=None):
        if b_val is None or t_val is None:
            return None
        bs = f"{b_val:.4f}"
        ts = f"{t_val:.4f}"
        dt = (t_val - b_val) * 100
        if has_latent and l_val is not None:
            ls = f"{l_val:.4f}"
            dl = (l_val - b_val) * 100
            dlt = (l_val - t_val) * 100
            return (f"| {label} | {bs} | {ts} | {ls} | "
                    f"{dt:+.2f}pp | {dl:+.2f}pp | {dlt:+.2f}pp |")
        return f"| {label} | {bs} | {ts} | {dt:+.2f}pp |"

    ptbxl_roc = [_ptbxl(k, "test_macro_auroc") for k in ["baseline", "time"]]
    ptbxl_ap = [_ptbxl(k, "test_macro_auprc") for k in ["baseline", "time"]]
    main5_roc = [_main5(k, "macro_auroc_avg") for k in ["baseline", "time"]]
    main5_ap = [_main5(k, "macro_auprc_avg") for k in ["baseline", "time"]]
    if has_latent:
        ptbxl_roc.append(_ptbxl("latent", "test_macro_auroc"))
        ptbxl_ap.append(_ptbxl("latent", "test_macro_auprc"))
        main5_roc.append(_main5("latent", "macro_auroc_avg"))
        main5_ap.append(_main5("latent", "macro_auprc_avg"))

    for label, vals in [
        ("PTBXL test macro AUROC", ptbxl_roc),
        ("PTBXL test macro AUPRC", ptbxl_ap),
        ("MAIN5 avg macro AUROC",  main5_roc),
        ("MAIN5 avg macro AUPRC",  main5_ap),
    ]:
        row = _row(label, vals[0], vals[1], vals[2] if has_latent else None)
        if row:
            lines.append(row)
    lines.append("")

    # Per-center
    lines.append("## Per-Center Macro AUROC\n")
    hdr = "| Center | N | Base | Time | "
    sep = "|---|---:|---:|---:|"
    if has_latent:
        hdr += "Latent | "
        sep += "---:|"
    hdr += "Δ Time | "
    sep += "---:|"
    if has_latent:
        hdr += "Δ Latent | Δ Latent − Time |"
        sep += "---:|---:|"
    else:
        hdr = hdr.rstrip(" | ") + " |"
    lines.append(hdr)
    lines.append(sep)
    for c in all_centers:
        if c not in base["centers"] or c not in runs["time"]["eval"]["centers"]:
            continue
        b_c = base["centers"][c]
        t_c = runs["time"]["eval"]["centers"][c]
        l_c = runs["latent"]["eval"]["centers"][c] if has_latent else None
        n = b_c.get("n_scored_records", "?")
        br = b_c["macro"]["macro_auroc"]
        tr = t_c["macro"]["macro_auroc"]
        lr = l_c["macro"]["macro_auroc"] if l_c else None
        dt_pp = (tr - br) * 100
        flag = " **★**" if c in main_centers else ""
        row = f"| {c}{flag} | {n} | {br:.4f} | {tr:.4f} |"
        if has_latent:
            row += f" {lr:.4f} |"
        row += f" {dt_pp:+.2f} |"
        if has_latent:
            dl_pp = (lr - br) * 100
            dlt_pp = (lr - tr) * 100
            row += f" {dl_pp:+.2f} | {dlt_pp:+.2f} |"
        lines.append(row)
    lines.append("")
    lines.append("*★ = main-5 center (averaged into headline MAIN5)*\n")

    # Per-center per-class AUROC deltas (3-way: Time vs Base, Latent vs Base)
    lines.append("## Per-Center × Per-Class AUROC (Δ vs Baseline in pp)\n")
    if has_latent:
        hdr = "| Center | Variant | " + " | ".join(CLASSES) + " |"
        sep = "|---|---|" + "---:|" * len(CLASSES)
        lines.append(hdr)
        lines.append(sep)
        for c in main_centers:
            if c not in base["centers"]:
                continue
            b_pc = base["centers"][c]["per_class"]
            for variant, run_key in [("Time", "time"), ("Latent", "latent")]:
                pc = runs[run_key]["eval"]["centers"][c].get("per_class", {})
                cells = []
                for cls in CLASSES:
                    b = b_pc.get(cls, {}).get("auroc") if b_pc.get(cls) else None
                    a = pc.get(cls, {}).get("auroc") if pc.get(cls) else None
                    if b is None or a is None:
                        cells.append("N/A")
                    else:
                        d = (a - b) * 100
                        cells.append(f"{a:.3f}({d:+.2f})")
                lines.append(f"| {c} | {variant} | " + " | ".join(cells) + " |")
    else:
        hdr = "| Center | " + " | ".join(CLASSES) + " |"
        sep = "|---|" + "---:|" * len(CLASSES)
        lines.append(hdr)
        lines.append(sep)
        for c in main_centers:
            b_pc = base["centers"][c].get("per_class", {})
            t_pc = runs["time"]["eval"]["centers"][c].get("per_class", {})
            cells = []
            for cls in CLASSES:
                b = b_pc.get(cls, {}).get("auroc") if b_pc.get(cls) else None
                a = t_pc.get(cls, {}).get("auroc") if t_pc.get(cls) else None
                if b is None or a is None:
                    cells.append("N/A")
                else:
                    d = (a - b) * 100
                    cells.append(f"{a:.3f}({d:+.2f})")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("")

    # Verdict
    lines.append("## Verdict\n")
    b_m5 = main5_roc[0]
    t_m5 = main5_roc[1]
    dt = (t_m5 - b_m5) * 100
    t_ptbxl_drop = (ptbxl_roc[0] - ptbxl_roc[1]) * 100 if ptbxl_roc[1] is not None else 0.0

    def _verdict(delta, drop, name):
        if delta >= 0.3 and drop <= 0.5:
            return f"**✅ {name} EFFECTIVE** — Δ={delta:+.2f}pp ≥ +0.30, PTBXL drop {drop:+.2f}pp within 0.5pp."
        elif abs(delta) < 0.3:
            return f"**❓ {name} AMBIGUOUS** — Δ={delta:+.2f}pp within ±0.30pp noise band."
        elif delta < 0:
            return f"**❌ {name} INEFFECTIVE** — Δ={delta:+.2f}pp regressed."
        else:
            return f"**⚠️ {name} PARTIAL** — Δ={delta:+.2f}pp but PTBXL drop {drop:+.2f}pp > 0.5pp."

    lines.append(_verdict(dt, t_ptbxl_drop, "Time-AugMix"))
    if has_latent:
        l_m5 = main5_roc[2]
        dl = (l_m5 - b_m5) * 100
        l_ptbxl_drop = (ptbxl_roc[0] - ptbxl_roc[2]) * 100 if ptbxl_roc[2] is not None else 0.0
        lines.append("")
        lines.append(_verdict(dl, l_ptbxl_drop, "Latent-AugMix"))

        # Head-to-head
        dlt = (l_m5 - t_m5) * 100
        lines.append("")
        lines.append("### Latent vs Time head-to-head\n")
        if dlt >= 0.2:
            lines.append(f"**Latent 胜出**（MAIN5 AUROC Δ={dlt:+.2f}pp vs time）。"
                         "推荐以 latent 为默认配置。")
        elif dlt <= -0.2:
            lines.append(f"**Time 胜出**（MAIN5 AUROC Δ={dlt:+.2f}pp vs time）。"
                         "latent 的 VAE 开销不划算。")
        else:
            lines.append(f"**两者等价**（MAIN5 AUROC Δ={dlt:+.2f}pp vs time，|Δ|<0.2pp）。"
                         "建议走 time-domain 版，无 VAE 开销。")
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[compare] wrote {out_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
