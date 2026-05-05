"""Stage-0 Style Translator ablation: Baseline vs A (target) vs B (null) vs C (wrong).

A: synth generated with correct target style_vec (cpsc_2018_extra).
B: synth generated with null style (vanilla ECGTwin baseline conditioning).
C: synth generated with wrong-style style_vec (georgia; stranger center).

Reports target center delta, MAIN5 delta, and ablation verdicts:
  - A > B : style vec adds value over vanilla conditioning
  - A > C : correct style beats wrong style
  - PTBXL ≥ 0.970 : no cross-domain collapse
"""
import argparse
import json
from pathlib import Path


DEFAULTS = {
    "baseline": "/root/autodl-tmp/crosscenter_tierM/eval_crosscenter.json",
    "A":        "/root/autodl-tmp/center_aware_ibe/retrain/stage0_A/eval_crosscenter.json",
    "B":        "/root/autodl-tmp/center_aware_ibe/retrain/stage0_B/eval_crosscenter.json",
    "C":        "/root/autodl-tmp/center_aware_ibe/retrain/stage0_C/eval_crosscenter.json",
}
LABELS = {
    "baseline": "Baseline",
    "A": "A: target style",
    "B": "B: null style",
    "C": "C: wrong (georgia)",
}
DEFAULT_OUT = "/root/ECG_adv_Gen/docs/ecgtwin_gen/style_translator_stage0_results.md"
CLASSES = ["NSR", "STach", "AF", "IAVB", "LBBB", "RBBB"]


def _train_json(eval_path: str):
    p = Path(eval_path).parent / "train_result.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load(paths):
    out = {}
    for k, p in paths.items():
        if p is None or not Path(p).exists():
            print(f"[compare] missing {k}: {p}")
            continue
        out[k] = {"eval": json.loads(Path(p).read_text()),
                  "train": _train_json(p), "path": p}
    return out


def main():
    ap = argparse.ArgumentParser()
    for k, default in DEFAULTS.items():
        ap.add_argument(f"--{k}_json", default=default)
    ap.add_argument("--target_center", default="cpsc_2018_extra")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    runs = _load({k: getattr(args, f"{k}_json") for k in DEFAULTS})
    assert "baseline" in runs
    variants = [k for k in ["baseline", "A", "B", "C"] if k in runs]

    base = runs["baseline"]["eval"]
    main5 = base.get("main_centers",
                     ["chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "georgia", "ningbo"])
    all_centers = list(base.get("centers", {}).keys())

    L = []
    L.append("# Stage 0 Style Translator — 3-center (cpsc_extra + chapman + cpsc)\n")
    L.append("**Ablation A / B / C** vs **baseline**\n")
    for k in variants:
        L.append(f"- {LABELS[k]}: `{runs[k]['path']}`")
    L.append("")

    def _ptbxl(k, m):
        tr = runs[k]["train"]
        return tr.get(m) if tr else None

    def _main5(k, m):
        return runs[k]["eval"]["main_centers_avg"].get(m)

    # ── Headline table ──
    L.append("## Headline Macro Metrics\n")
    hdr = "| Metric | " + " | ".join(LABELS[k] for k in variants) + " | Δ A vs B | Δ A vs C | Δ A vs Base |"
    L.append(hdr)
    L.append("|---|" + "---:|" * len(variants) + "---:|---:|---:|")
    for m, lbl in [
        ("test_macro_auroc", "PTBXL test macro AUROC"),
        ("test_macro_auprc", "PTBXL test macro AUPRC"),
        ("macro_auroc_avg",  "MAIN5 avg macro AUROC"),
        ("macro_auprc_avg",  "MAIN5 avg macro AUPRC"),
    ]:
        vals = {}
        for k in variants:
            vals[k] = _ptbxl(k, m) if m.startswith("test") else _main5(k, m)
        cells = [f"{vals[k]:.4f}" if vals[k] is not None else "N/A" for k in variants]
        A, B, C, Bl = vals.get("A"), vals.get("B"), vals.get("C"), vals.get("baseline")
        d_ab = (A - B) * 100 if A is not None and B is not None else None
        d_ac = (A - C) * 100 if A is not None and C is not None else None
        d_abl = (A - Bl) * 100 if A is not None and Bl is not None else None
        fmt = lambda d: f"{d:+.2f}pp" if d is not None else "N/A"
        L.append(f"| {lbl} | " + " | ".join(cells) + f" | {fmt(d_ab)} | {fmt(d_ac)} | {fmt(d_abl)} |")
    L.append("")

    # ── Per-center AUROC ──
    L.append("## Per-Center Macro AUROC\n")
    L.append("| Center | N | " + " | ".join(LABELS[k] for k in variants) + " | Δ A vs Base |")
    L.append("|---|---:|" + "---:|" * len(variants) + "---:|")
    for c in all_centers:
        if c not in base["centers"]:
            continue
        n = base["centers"][c].get("n_scored_records", "?")
        row = {}
        for k in variants:
            row[k] = runs[k]["eval"]["centers"].get(c, {}).get("macro", {}).get("macro_auroc")
        cells = [f"{row[k]:.4f}" if row[k] is not None else "N/A" for k in variants]
        A, Bl = row.get("A"), row.get("baseline")
        d = (A - Bl) * 100 if A is not None and Bl is not None else None
        flag = " ★" if c in main5 else ""
        flag += " 🎯" if c == args.target_center else ""
        L.append(f"| {c}{flag} | {n} | " + " | ".join(cells) + " | " + (f"{d:+.2f}pp" if d is not None else "N/A") + " |")
    L.append("")
    L.append("*★ = MAIN5 center; 🎯 = target of style enrollment*\n")

    # ── Target center per-class ──
    L.append(f"## Target Center ({args.target_center}) Per-Class AUROC\n")
    if args.target_center in base["centers"]:
        L.append("| Class | " + " | ".join(LABELS[k] for k in variants) + " | Δ A vs Base |")
        L.append("|---|" + "---:|" * len(variants) + "---:|")
        b_pc = base["centers"][args.target_center].get("per_class", {})
        for cls in CLASSES:
            cells = []
            b_v = b_pc.get(cls, {}).get("auroc")
            A_v = None
            for k in variants:
                pc = runs[k]["eval"]["centers"].get(args.target_center, {}).get("per_class", {})
                v = pc.get(cls, {}).get("auroc") if pc.get(cls) else None
                cells.append(f"{v:.4f}" if v is not None else "N/A")
                if k == "A":
                    A_v = v
            d = (A_v - b_v) * 100 if (A_v is not None and b_v is not None) else None
            L.append(f"| {cls} | " + " | ".join(cells) + f" | {f'{d:+.2f}pp' if d is not None else 'N/A'} |")
    L.append("")

    # ── Verdict ──
    L.append("## Ablation Verdict\n")
    def _tc(k):
        return runs[k]["eval"]["centers"].get(args.target_center, {}).get("macro", {}).get("macro_auroc") if k in runs else None

    A_tc, B_tc, C_tc, Bl_tc = _tc("A"), _tc("B"), _tc("C"), _tc("baseline")
    A_ptbxl = _ptbxl("A", "test_macro_auroc")

    tests = []
    if A_tc is not None and Bl_tc is not None:
        d = (A_tc - Bl_tc) * 100
        ok = d >= 0.5
        tests.append((f"A target AUROC ≥ +0.5pp vs Baseline (got {d:+.2f}pp)", ok))
    if A_tc is not None and B_tc is not None:
        d = (A_tc - B_tc) * 100
        ok = d >= 0.3
        tests.append((f"A > B target AUROC ≥ +0.3pp (got {d:+.2f}pp) — style vec adds value", ok))
    if A_tc is not None and C_tc is not None:
        d = (A_tc - C_tc) * 100
        ok = d >= 0.1
        tests.append((f"A > C target AUROC ≥ +0.1pp (got {d:+.2f}pp) — correct style beats wrong", ok))
    if A_ptbxl is not None:
        ok = A_ptbxl >= 0.970
        tests.append((f"PTBXL test AUROC ≥ 0.970 (got {A_ptbxl:.4f}) — no cross-domain collapse", ok))

    for msg, ok in tests:
        L.append(f"- {'✅' if ok else '❌'} {msg}")
    all_ok = all(ok for _, ok in tests) if tests else False
    L.append("")
    if all_ok:
        L.append("**🟢 Go for Stage 0-multi** (all four gates passed).")
    else:
        L.append("**🔴 No-go for Stage 0-multi** — pivot to root-cause investigation.")
    L.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"[compare] wrote {out}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
