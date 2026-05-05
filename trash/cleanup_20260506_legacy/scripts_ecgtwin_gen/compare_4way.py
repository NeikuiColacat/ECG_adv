"""4-way comparison: Baseline / Time-AugMix / Latent-AugMix / CenterToken.

Focuses on the new CenterToken experiment — reuses schema from
``scripts/augmix_validation/compare_baseline.py`` but adds a 4th column.
"""
import argparse
import json
from pathlib import Path


DEFAULTS = {
    "baseline":    "/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json",
    "time":        "/root/autodl-tmp/crosscenter_tierM_augmix/half_s5/eval_crosscenter.json",
    "latent":      "/root/autodl-tmp/crosscenter_tierM_augmix_latent/half_s5/eval_crosscenter.json",
    "centertoken": "/root/autodl-tmp/crosscenter_tierM_centertoken/cpsc_2018_extra/eval_crosscenter.json",
}
DEFAULT_OUT = "/root/ECG_adv_Gen/outputs/augmix_validation/compare_4way.md"
CLASSES = ["NSR", "STach", "AF", "IAVB", "LBBB", "RBBB"]


def _train_json(eval_path: str):
    p = Path(eval_path).parent / "train_result.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load(paths):
    out = {}
    for key, p in paths.items():
        if p is None or not Path(p).exists():
            print(f"[compare] missing {key}: {p}")
            continue
        out[key] = {"eval": json.loads(Path(p).read_text()),
                    "train": _train_json(p), "path": p}
    return out


def main():
    ap = argparse.ArgumentParser()
    for k, default in DEFAULTS.items():
        ap.add_argument(f"--{k}_json", default=default)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--target_center", default="cpsc_2018_extra",
                    help="Center the CenterToken was trained for (for emphasis in output)")
    args = ap.parse_args()

    runs = _load({k: getattr(args, f"{k}_json") for k in DEFAULTS})
    assert "baseline" in runs, "baseline eval.json missing"
    assert "centertoken" in runs, f"centertoken eval.json missing: {args.centertoken_json}"

    variant_keys = [k for k in ["baseline", "time", "latent", "centertoken"] if k in runs]
    variant_label = {"baseline": "Baseline", "time": "Time", "latent": "Latent", "centertoken": "CenterToken"}

    base = runs["baseline"]["eval"]
    main_centers = base.get("main_centers",
                            ["chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "georgia", "ningbo"])
    all_centers = list(base.get("centers", {}).keys())

    lines = []
    lines.append("# CenterToken Validation — 4-Way Comparison\n")
    for k in variant_keys:
        lines.append(f"- {variant_label[k]}: `{runs[k]['path']}`")
    lines.append("")
    ct_cfg = runs["centertoken"]["train"].get("config", {}) if runs["centertoken"]["train"] else {}
    lines.append(f"CenterToken config: target center **{args.target_center}**, "
                 f"synth_ratio={ct_cfg.get('synth_ratio')}, "
                 f"synth_center_npz={ct_cfg.get('synth_center_npz')}, "
                 f"seed={ct_cfg.get('seed')}")
    lines.append("")

    # Headline numbers
    def _ptbxl(k, m):
        tr = runs[k]["train"]
        return tr.get(m) if tr else None

    def _main5(k, m):
        return runs[k]["eval"]["main_centers_avg"].get(m)

    # Headline table
    lines.append("## Headline Macro Metrics\n")
    header = "| Metric | " + " | ".join(variant_label[k] for k in variant_keys) + \
             " | Δ CT vs Base | Δ CT vs Time | Δ CT vs Latent |"
    sep = "|---|" + "---:|" * len(variant_keys) + "---:|---:|---:|"
    lines.append(header)
    lines.append(sep)

    for mkey, label in [("test_macro_auroc", "PTBXL test macro AUROC"),
                        ("test_macro_auprc", "PTBXL test macro AUPRC"),
                        ("macro_auroc_avg",  "MAIN5 avg macro AUROC"),
                        ("macro_auprc_avg",  "MAIN5 avg macro AUPRC")]:
        vals = {}
        for k in variant_keys:
            v = _ptbxl(k, mkey) if mkey.startswith("test") else _main5(k, mkey)
            vals[k] = v
        cells = [f"{vals[k]:.4f}" if vals[k] is not None else "N/A" for k in variant_keys]
        ct = vals.get("centertoken")
        d_base = (ct - vals.get("baseline")) * 100 if ct is not None and vals.get("baseline") is not None else None
        d_time = (ct - vals.get("time")) * 100 if ct is not None and vals.get("time") is not None else None
        d_latent = (ct - vals.get("latent")) * 100 if ct is not None and vals.get("latent") is not None else None
        fmt = lambda d: f"{d:+.2f}pp" if d is not None else "N/A"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {fmt(d_base)} | {fmt(d_time)} | {fmt(d_latent)} |")
    lines.append("")

    # Per-center macro AUROC
    lines.append("## Per-Center Macro AUROC\n")
    hdr = "| Center | N | " + " | ".join(variant_label[k] for k in variant_keys) + " | Δ CT vs Base |"
    lines.append(hdr)
    lines.append("|---|---:|" + "---:|" * len(variant_keys) + "---:|")
    for c in all_centers:
        if c not in base["centers"]:
            continue
        b_c = base["centers"][c]
        n = b_c.get("n_scored_records", "?")
        row_vals = {}
        for k in variant_keys:
            if c in runs[k]["eval"].get("centers", {}):
                row_vals[k] = runs[k]["eval"]["centers"][c]["macro"]["macro_auroc"]
            else:
                row_vals[k] = None
        cells = [f"{row_vals[k]:.4f}" if row_vals[k] is not None else "N/A" for k in variant_keys]
        ct = row_vals.get("centertoken")
        b_v = row_vals.get("baseline")
        d_base = (ct - b_v) * 100 if ct is not None and b_v is not None else None
        flag = " **★**" if c in main_centers else ""
        flag += " 🎯" if c == args.target_center else ""
        d_str = f"{d_base:+.2f}pp" if d_base is not None else "N/A"
        lines.append(f"| {c}{flag} | {n} | " + " | ".join(cells) + f" | {d_str} |")
    lines.append("")
    lines.append("*★ = main-5 center (in headline MAIN5 avg); 🎯 = target of CenterToken training*\n")

    # Target-center per-class AUROC change
    lines.append(f"## Target-Center ({args.target_center}) Per-Class AUROC\n")
    if args.target_center in base["centers"]:
        lines.append("| Class | Baseline | CenterToken | Δ (pp) |")
        lines.append("|---|---:|---:|---:|")
        b_pc = base["centers"][args.target_center].get("per_class", {})
        ct_pc = runs["centertoken"]["eval"]["centers"].get(args.target_center, {}).get("per_class", {})
        for cls in CLASSES:
            b = b_pc.get(cls, {}).get("auroc") if b_pc.get(cls) else None
            c = ct_pc.get(cls, {}).get("auroc") if ct_pc.get(cls) else None
            if b is None or c is None:
                lines.append(f"| {cls} | " + (f"{b:.4f}" if b else "N/A") + " | " + (f"{c:.4f}" if c else "N/A") + " | N/A |")
            else:
                lines.append(f"| {cls} | {b:.4f} | {c:.4f} | {(c - b) * 100:+.2f}pp |")
    lines.append("")

    # Verdict
    lines.append("## Verdict\n")
    b_m5 = _main5("baseline", "macro_auroc_avg")
    ct_m5 = _main5("centertoken", "macro_auroc_avg")
    d_m5 = (ct_m5 - b_m5) * 100 if ct_m5 is not None and b_m5 is not None else None
    ct_ptbxl = _ptbxl("centertoken", "test_macro_auroc")
    b_ptbxl = _ptbxl("baseline", "test_macro_auroc")
    drop_ptbxl = (b_ptbxl - ct_ptbxl) * 100 if ct_ptbxl and b_ptbxl else None

    # Target-center gain
    b_tc = base["centers"].get(args.target_center, {}).get("macro", {}).get("macro_auroc")
    ct_tc = runs["centertoken"]["eval"]["centers"].get(args.target_center, {}).get("macro", {}).get("macro_auroc")
    d_tc = (ct_tc - b_tc) * 100 if ct_tc is not None and b_tc is not None else None

    if d_tc is None:
        lines.append("**⚠️ Target-center delta unavailable.**")
    elif d_tc >= 1.0 and (drop_ptbxl is None or drop_ptbxl <= 0.5):
        lines.append(f"**✅ CenterToken EFFECTIVE for {args.target_center}** — target center Δ={d_tc:+.2f}pp ≥ +1.0, "
                     f"PTBXL drop {drop_ptbxl:+.2f}pp within 0.5. "
                     f"MAIN5 avg Δ={d_m5:+.2f}pp.")
    elif d_tc >= 0.5:
        lines.append(f"**⚠️ CenterToken PARTIAL** — target Δ={d_tc:+.2f}pp ∈ [+0.5, +1.0] — similar to AugMix; "
                     f"not clearly worth the ECGTwin-VAE complexity. MAIN5 Δ={d_m5:+.2f}pp.")
    elif abs(d_tc) < 0.5:
        lines.append(f"**❓ CenterToken AMBIGUOUS** — target Δ={d_tc:+.2f}pp within ±0.5 noise. "
                     f"Method not clearly helping.")
    else:
        lines.append(f"**❌ CenterToken INEFFECTIVE / REGRESSES** — target Δ={d_tc:+.2f}pp negative. "
                     f"Synth data may be degrading the classifier.")
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"[compare] wrote {out}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
