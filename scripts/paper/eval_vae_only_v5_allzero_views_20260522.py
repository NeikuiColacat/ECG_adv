#!/usr/bin/env python3
"""Evaluate VAE-only K=500 LH-AT under PN2021 Super5 v5 mapping.

The evaluator reports two PN2021 metric views in one inference pass:

1. default: all-zero Super5 rows are kept as negatives for all five classes;
2. drop-all-zero: rows with no positive Super5 label are removed before metrics.

This script compares the PTB-XL baseline against the current VAE-only real-anchor
LH-AT K=500 runs on the four paper target centers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.triple_labels.label_schemes import get_super5_pn2021_mapping_metadata


PYTHON = "/root/miniforge3/envs/ECGTwin/bin/python"
BASELINE_DIR = Path("/root/autodl-tmp/triple_labels/super5_minresample_full10_perglobal_20260503")
PTBXL_PREP = Path(
    "/root/autodl-tmp/triple_labels/cache/"
    "ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"
)
PN2021_CACHE_DIR = Path("/root/autodl-tmp/triple_labels/pn2021_eval_cache_minresample_perglobal")
PN2021_MMAP_CACHE_DIR = Path("/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap_minresample_perglobal")
KCURVE_ROOT = Path("/root/autodl-tmp/paper_vae_only_lhat_kcurve_20260518")
OUT_ROOT = Path("/root/autodl-tmp/paper_vae_only_v5_allzero_eval_20260522")
DOC_OUT = Path("docs/reports/archive/20260522/vae_only_v5_allzero_eval_20260522.md")

CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]


def ref_meta(center: str) -> Path:
    return (
        KCURVE_ROOT
        / "subsets"
        / center
        / "k500_seed20260531"
        / f"{center}_real_k500_seed20260531.ref_meta.json"
    )


def vae_only_dir(center: str) -> Path:
    return (
        KCURVE_ROOT
        / "runs"
        / f"{center}_K500_lambda0p15_M20_ep10_seed20260531"
    )


def output_json(center: str, arm: str) -> Path:
    return OUT_ROOT / "evals" / f"{center}_{arm}_v5_allzero_views.json"


def is_current_eval(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    meta = payload.get("label_mapping", {}).get("pn2021_super5", {})
    expected = get_super5_pn2021_mapping_metadata()
    if meta.get("mapping_hash") != expected.get("mapping_hash"):
        return False
    pn = payload.get("pn2021", {})
    return "avg_drop_all_zero_macro_auroc" in pn and "per_center" in pn


def run_eval(center: str, arm: str, model_dir: Path, args: argparse.Namespace) -> Path:
    out = output_json(center, arm)
    log = out.with_suffix(".log")
    out.parent.mkdir(parents=True, exist_ok=True)
    if is_current_eval(out) and not args.force:
        print(f"[skip] {center} {arm}: {out}", flush=True)
        return out
    cmd = [
        PYTHON,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme",
        "super5",
        "--model_dir",
        str(model_dir),
        "--device",
        args.device,
        "--crop_len",
        "1000",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--ptbxl_cache",
        str(PTBXL_PREP),
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--pn2021_cache_dir",
        str(PN2021_CACHE_DIR),
        "--pn2021_mmap_cache_dir",
        str(PN2021_MMAP_CACHE_DIR),
        "--skip_mimic",
        "--exclude_ref_ids",
        str(ref_meta(center)),
        "--report_drop_all_zero_pn2021",
        "--output_path",
        str(out),
    ]
    print("[run] " + " ".join(cmd), flush=True)
    with log.open("w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    return out


def finite_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def fmt_metric(a: float, p: float) -> str:
    return f"{a:.4f} / {p:.4f}"


def fmt_delta(v: float, b: float) -> str:
    if not math.isfinite(v) or not math.isfinite(b):
        return "nan"
    return f"{(v - b) * 100:+.2f}pp"


def extract(path: Path, center: str, drop_all_zero: bool) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    pc = payload["pn2021"]["per_center"][center]
    if drop_all_zero:
        auroc_key = "drop_all_zero_macro_auroc"
        auprc_key = "drop_all_zero_macro_auprc"
        n_key = "drop_all_zero_n_records"
    else:
        auroc_key = "macro_auroc"
        auprc_key = "macro_auprc"
        n_key = "effective_n"
    return {
        "auroc": finite_float(pc.get(auroc_key)),
        "auprc": finite_float(pc.get(auprc_key)),
        "n": int(pc.get(n_key, 0)),
        "n_all_zero": int(pc.get("n_all_zero_labels", 0)),
        "n_nonzero": int(pc.get("n_nonzero_labels", 0)),
        "n_excluded_ref": int(pc.get("n_excluded_ref", 0)),
        "n_classes_used": int(pc.get("drop_all_zero_n_classes_used" if drop_all_zero else "n_classes_used", 0)),
    }


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def write_outputs(rows: list[dict[str, Any]], eval_paths: dict[tuple[str, str], Path]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "vae_only_v5_allzero_eval.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def table(view: str) -> list[str]:
        lines = [
            f"## {view}\n",
            "| center | baseline | VAE-only | delta | eval n | all-zero n |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        sub = [r for r in rows if r["view"] == view]
        for r in sub:
            lines.append(
                f"| {r['center']} | {fmt_metric(r['baseline_auroc'], r['baseline_auprc'])} | "
                f"{fmt_metric(r['vae_auroc'], r['vae_auprc'])} | "
                f"{fmt_delta(r['vae_auroc'], r['baseline_auroc'])} / "
                f"{fmt_delta(r['vae_auprc'], r['baseline_auprc'])} | "
                f"{r['eval_n']} | {r['all_zero_n']} |"
            )
        b_a = mean([r["baseline_auroc"] for r in sub])
        b_p = mean([r["baseline_auprc"] for r in sub])
        v_a = mean([r["vae_auroc"] for r in sub])
        v_p = mean([r["vae_auprc"] for r in sub])
        lines.append(
            f"| **4-center avg** | **{fmt_metric(b_a, b_p)}** | "
            f"**{fmt_metric(v_a, v_p)}** | "
            f"**{fmt_delta(v_a, b_a)} / {fmt_delta(v_p, b_p)}** |  |  |"
        )
        lines.append("")
        return lines

    mapping = get_super5_pn2021_mapping_metadata()
    md = [
        "# VAE-only v5 标签策略 all-zero 指标复核\n",
        f"- 生成时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- PN2021 Super5 mapping：`{mapping['mapping_version']}`，hash `{mapping['mapping_hash']}`",
        "- 模型：PTB-XL full baseline vs VAE-only real-anchor LH-AT K=500。",
        "- 评估：target center 指标；每个 target center 评估时排除对应 K=500 适配样本，避免测试污染。",
        "- 输入协议：minimal_resample、per_sample_global、100Hz、1000 samples。",
        "",
        "## 口径说明\n",
        "- `all-zero kept`：Super5 全 0 样本作为五个类别的负样本参与 AUROC/AUPRC。",
        "- `drop all-zero`：先删除 Super5 全 0 样本，只在至少一个 Super5 阳性的记录上计算指标。",
        "",
    ]
    md.extend(table("all-zero kept"))
    md.extend(table("drop all-zero"))
    md.extend([
        "## 我的判断\n",
        "主报告建议保留 `all-zero kept`。这些样本在 v5 中表示“没有足够证据归入 CD/HYP/MI/NORM/STTC，或属于 Super5 外异常”，真实部署时模型仍会遇到它们；把它们作为五类负样本可以测试模型是否会把 Super5 外记录误报成目标类。",
        "",
        "`drop all-zero` 可以作为敏感性分析。它回答的是“只在 Super5 覆盖范围内，模型区分五类的能力如何”，但会排除大量 PN2021 真实外部样本，外部有效性更弱，不建议作为主指标。",
        "",
        "## 产物\n",
        f"- CSV：`{csv_path}`",
        f"- JSON 目录：`{OUT_ROOT / 'evals'}`",
    ])
    for (center, arm), path in sorted(eval_paths.items()):
        md.append(f"- {center} {arm}: `{path}`")
    text = "\n".join(md) + "\n"
    report_path = OUT_ROOT / "vae_only_v5_allzero_eval.md"
    report_path.write_text(text)
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC_OUT.write_text(text)
    print(f"[done] wrote {report_path}")
    print(f"[done] wrote {DOC_OUT}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    eval_paths: dict[tuple[str, str], Path] = {}
    for center in CENTERS:
        if not ref_meta(center).exists():
            raise FileNotFoundError(ref_meta(center))
        if not (vae_only_dir(center) / "best_model.pt").exists():
            raise FileNotFoundError(vae_only_dir(center) / "best_model.pt")
        eval_paths[(center, "baseline")] = run_eval(center, "baseline", BASELINE_DIR, args)
        eval_paths[(center, "vae_only")] = run_eval(center, "vae_only", vae_only_dir(center), args)

    rows: list[dict[str, Any]] = []
    for center in CENTERS:
        b_path = eval_paths[(center, "baseline")]
        v_path = eval_paths[(center, "vae_only")]
        for view, drop in [("all-zero kept", False), ("drop all-zero", True)]:
            b = extract(b_path, center, drop)
            v = extract(v_path, center, drop)
            rows.append({
                "center": center,
                "view": view,
                "baseline_auroc": b["auroc"],
                "baseline_auprc": b["auprc"],
                "vae_auroc": v["auroc"],
                "vae_auprc": v["auprc"],
                "delta_auroc": v["auroc"] - b["auroc"],
                "delta_auprc": v["auprc"] - b["auprc"],
                "eval_n": v["n"],
                "all_zero_n": v["n_all_zero"],
                "nonzero_n": v["n_nonzero"],
                "excluded_ref_n": v["n_excluded_ref"],
                "classes_used": v["n_classes_used"],
            })
    write_outputs(rows, eval_paths)


if __name__ == "__main__":
    main()
