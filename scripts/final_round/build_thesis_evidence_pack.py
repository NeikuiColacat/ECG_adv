#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from util.ecg_viz import plot_comparison


CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
TMP_ROOT = Path("/root/autodl-tmp")
GRAD_ROOT = TMP_ROOT / "graduate_project"
FINAL_ROUND_ROOT = TMP_ROOT / "final_round_ablation_20260504"
STREAMLIT_ROOT = TMP_ROOT / "streamlit_ecg_demo"

SPLIT_JSON = GRAD_ROOT / "splits/ptbxl_super5_seed42_train2000_val2000.json"
LABEL_NPY = GRAD_ROOT / "method_a_real2000_seed42/ptbxl_labels.C5.all.npy"
PTBXL_CACHE = TMP_ROOT / "triple_labels/cache/ptbxl_minimal_resample_per_sample_global_fs100_len1000.npy"
SELECTED_SYNTH_NPZ = (
    FINAL_ROUND_ROOT / "thesis_paper_selected_v1/thesis_selected_samples.npz"
)
SELECTED_SYNTH_JSON = (
    FINAL_ROUND_ROOT / "thesis_paper_selected_v1/selected_examples.json"
)
MEDICAL_VALIDITY_JSON = FINAL_ROUND_ROOT / "medical_validity/medical_validity_summary.json"
MEDICAL_VALIDITY_CSV = FINAL_ROUND_ROOT / "medical_validity/per_sample_quality.csv"
BENCHMARK_JSON = STREAMLIT_ROOT / "reports/inference_benchmark.json"
TRT_VALIDATION_JSON = STREAMLIT_ROOT / "reports/tensorrt_validation.json"


@dataclass(frozen=True)
class RunSpec:
    key: str
    name: str
    path: Path
    group: str
    note: str = ""
    paper_main: bool = False


RUNS = [
    RunSpec(
        "real2000_original",
        "real2000 baseline",
        GRAD_ROOT / "method_a_real2000_seed42",
        "main",
        "Paper main baseline run paired with the original downstream comparison.",
        True,
    ),
    RunSpec(
        "no_token_hard_ft",
        "no-token hard pretrain -> real fine-tune",
        GRAD_ROOT
        / "self_distill_v2_e24_v46_no_token_hardlabel_r10_realfine_lr1e4_seed42_auroc",
        "main",
        "Strict no-token hard-label pretrain followed by real2000 fine-tune.",
        True,
    ),
    RunSpec(
        "center_token_hard_ft",
        "center-token hard pretrain -> real fine-tune",
        GRAD_ROOT
        / "self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc",
        "main",
        "Best thesis low-sample recipe.",
        True,
    ),
    RunSpec(
        "real2000_rerun_20260506",
        "real2000 baseline rerun 2026-05-06",
        GRAD_ROOT / "method_a_real2000_seed42_rerun_20260506",
        "reproduction",
        "Same split rerun; lower than original, so keep as reproducibility caveat.",
    ),
    RunSpec(
        "no_token_hard_ft_rerun_20260506",
        "no-token hard FT rerun 2026-05-06",
        GRAD_ROOT
        / "self_distill_v2_e24_v46_no_token_hardlabel_r10_realfine_lr1e4_seed42_auroc_rerun_20260506",
        "reproduction",
        "Rerun matches original within rounding.",
    ),
    RunSpec(
        "center_token_hard_ft_rerun_20260506",
        "center-token hard FT rerun 2026-05-06",
        GRAD_ROOT
        / "self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc_rerun_20260506",
        "reproduction",
        "Rerun matches original within rounding.",
    ),
    RunSpec(
        "method_b_real_synth_mv4",
        "real2000 + prompt-token synth",
        GRAD_ROOT / "method_b_real2000_plus_synth3364_mv4_seed42",
        "ablation",
        "Direct real+synth training from random init.",
    ),
    RunSpec(
        "method_b_actual_report",
        "real2000 + actual-report prompt synth",
        GRAD_ROOT
        / "method_b_real2000_plus_synth3313_actual_report_mv4_step2000_seed42",
        "ablation",
        "Actual-report prompt variant.",
    ),
    RunSpec(
        "method_b_classfallback",
        "real2000 + class-fallback new generation",
        GRAD_ROOT
        / "method_b_real2000_plus_synth3391_classfallback_newgen_sameclass_cap20_seed42",
        "ablation",
        "Class-fallback prompt variant.",
    ),
    RunSpec(
        "synthetic_only_no_token_20k",
        "no-token synthetic-only pretrain",
        GRAD_ROOT / "ablation_vanilla_no_token_synthonly_pretrain_n20000_seed42",
        "ablation",
        "Synthetic-only classifier is weak without real fine-tuning.",
    ),
    RunSpec(
        "synth_pretrain_no_token_ft",
        "no-token synth pretrain -> real fine-tune",
        GRAD_ROOT
        / "ablation_vanilla_no_token_synthpretrain_auprc_real_finetune_seed42",
        "ablation",
        "Synthetic pretraining followed by clean real2000 fine-tune.",
    ),
    RunSpec(
        "synth_pretrain_latenthull",
        "no-token synth pretrain + Latent-Hull AT",
        GRAD_ROOT
        / "ablation_vanilla_no_token_synthpretrain_real_latenthull_at_M10_n1000_seed42",
        "ablation",
        "Small Latent-Hull stabilization ablation.",
    ),
    RunSpec(
        "distill_synth_teacher",
        "synth-initialized teacher distillation",
        GRAD_ROOT / "self_distill_d1_teacher_c3_init_c0_synth20k_seed42",
        "ablation",
        "Teacher/self-distillation path.",
    ),
    RunSpec(
        "distill_no_token_teacher",
        "no-token teacher distillation",
        GRAD_ROOT / "self_distill_d1_teacher_no_token_c3_init_no_token_c0_synth20k_seed42",
        "ablation",
        "Strong distillation ablation.",
    ),
    RunSpec(
        "distill_real_only_teacher",
        "real-only teacher distillation",
        GRAD_ROOT / "self_distill_d2_teacher_c3_init_c0_realonly_seed42",
        "ablation",
        "Real-only distillation control.",
    ),
    RunSpec(
        "naked_no_token_soft",
        "naked no-token soft self-distill",
        GRAD_ROOT
        / "self_distill_v2_e8_v46_no_token_matched_filtered3734_gamma03_scratch_seed42_auroc",
        "ablation",
        "Earlier soft self-distillation no-token control.",
    ),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:
        return None


def jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown_table(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt4(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def fmt2(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def run_row(spec: RunSpec) -> dict:
    result_path = spec.path / "train_result.json"
    if not result_path.exists():
        return {
            "key": spec.key,
            "method": spec.name,
            "group": spec.group,
            "status": "missing",
            "run_dir": str(spec.path),
            "note": spec.note,
        }
    data = load_json(result_path)
    cfg = data.get("config", {})
    return {
        "key": spec.key,
        "method": spec.name,
        "group": spec.group,
        "status": "ok",
        "test_macro_auroc": data.get("test_macro_auroc"),
        "test_macro_auprc": data.get("test_macro_auprc"),
        "best_val_macro_auroc": data.get("best_val_macro_auroc"),
        "best_val_macro_auprc": data.get("best_val_macro_auprc"),
        "epochs_trained": data.get("epochs_trained"),
        "checkpoint_metric": data.get("checkpoint_metric"),
        "seed": cfg.get("seed"),
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
        "preprocess_mode": cfg.get("preprocess_mode"),
        "norm_mode": cfg.get("norm_mode"),
        "split_json": cfg.get("split_json"),
        "run_dir": str(spec.path),
        "result_json": str(result_path),
        "note": spec.note,
        "paper_main": spec.paper_main,
        "raw": data,
    }


def build_run_tables(tables_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    all_rows = [run_row(spec) for spec in RUNS]
    main_rows = [row for row in all_rows if row.get("paper_main")]
    if main_rows:
        base_auroc = float(main_rows[0]["test_macro_auroc"])
        base_auprc = float(main_rows[0]["test_macro_auprc"])
        for row in main_rows:
            row["delta_auroc_vs_real2000"] = float(row["test_macro_auroc"]) - base_auroc
            row["delta_auprc_vs_real2000"] = float(row["test_macro_auprc"]) - base_auprc

    display_fields = [
        "method",
        "test_macro_auroc",
        "test_macro_auprc",
        "delta_auroc_vs_real2000",
        "delta_auprc_vs_real2000",
        "epochs_trained",
        "run_dir",
        "note",
    ]
    csv_fields = [
        "key",
        "method",
        "group",
        "status",
        "test_macro_auroc",
        "test_macro_auprc",
        "best_val_macro_auroc",
        "best_val_macro_auprc",
        "delta_auroc_vs_real2000",
        "delta_auprc_vs_real2000",
        "epochs_trained",
        "checkpoint_metric",
        "seed",
        "lr",
        "batch_size",
        "preprocess_mode",
        "norm_mode",
        "run_dir",
        "result_json",
        "note",
    ]

    for rows, stem in [
        (main_rows, "low_sample_main_results"),
        ([r for r in all_rows if r["group"] == "ablation"], "low_sample_ablation_results"),
        ([r for r in all_rows if r["group"] == "reproduction"], "reproduction_check_results"),
        (all_rows, "all_low_sample_runs"),
    ]:
        write_csv(tables_dir / f"{stem}.csv", rows, csv_fields)
        md_rows = []
        for row in rows:
            md_rows.append({
                **row,
                "test_macro_auroc": fmt4(row.get("test_macro_auroc")),
                "test_macro_auprc": fmt4(row.get("test_macro_auprc")),
                "delta_auroc_vs_real2000": fmt4(row.get("delta_auroc_vs_real2000")),
                "delta_auprc_vs_real2000": fmt4(row.get("delta_auprc_vs_real2000")),
            })
        write_markdown_table(tables_dir / f"{stem}.md", md_rows, display_fields)

    per_class_rows: list[dict] = []
    for row in main_rows:
        per_class = row.get("raw", {}).get("test_per_class", {})
        for cls in CLASS_NAMES:
            rec = per_class.get(cls, {})
            per_class_rows.append({
                "method": row["method"],
                "class": cls,
                "auroc": rec.get("auroc"),
                "auprc": rec.get("auprc"),
                "n_pos": rec.get("n_pos"),
                "n_valid": rec.get("n_valid"),
                "run_dir": row["run_dir"],
            })
    write_csv(
        tables_dir / "per_class_main_results.csv",
        per_class_rows,
        ["method", "class", "auroc", "auprc", "n_pos", "n_valid", "run_dir"],
    )
    md_rows = [
        {**r, "auroc": fmt4(r.get("auroc")), "auprc": fmt4(r.get("auprc"))}
        for r in per_class_rows
    ]
    write_markdown_table(
        tables_dir / "per_class_main_results.md",
        md_rows,
        ["method", "class", "auroc", "auprc", "n_pos", "n_valid"],
    )
    return all_rows, main_rows, per_class_rows


def build_dataset_tables(tables_dir: Path, figures_dir: Path) -> dict:
    split = load_json(SPLIT_JSON)
    labels = np.load(LABEL_NPY, mmap_mode="r")
    subsets = {
        "all": np.arange(labels.shape[0]),
        "train2000": np.asarray(split["train_indices"], dtype=np.int64),
        "val2000": np.asarray(split["val_indices"], dtype=np.int64),
        "test_rest": np.asarray(split["test_indices"], dtype=np.int64),
    }
    rows = []
    for subset, indices in subsets.items():
        y = np.asarray(labels[indices])
        for j, cls in enumerate(CLASS_NAMES):
            count = int((y[:, j] > 0.5).sum())
            rows.append({
                "subset": subset,
                "n_records": int(len(indices)),
                "class": cls,
                "positive_count": count,
                "positive_rate": count / max(1, len(indices)),
            })
    write_csv(
        tables_dir / "dataset_split_label_distribution.csv",
        rows,
        ["subset", "n_records", "class", "positive_count", "positive_rate"],
    )
    md_rows = [{**r, "positive_rate": fmt4(r["positive_rate"])} for r in rows]
    write_markdown_table(
        tables_dir / "dataset_split_label_distribution.md",
        md_rows,
        ["subset", "n_records", "class", "positive_count", "positive_rate"],
    )

    cardinality_rows = []
    combo_rows = []
    for subset, indices in subsets.items():
        y = np.asarray(labels[indices] > 0.5, dtype=np.int64)
        for card in range(0, 6):
            count = int((y.sum(axis=1) == card).sum())
            if count:
                cardinality_rows.append({
                    "subset": subset,
                    "positive_label_count": card,
                    "n_records": count,
                    "rate": count / max(1, len(indices)),
                })
        combos: dict[str, int] = {}
        for row in y:
            names = [CLASS_NAMES[i] for i, v in enumerate(row) if v]
            key = "+".join(names) if names else "NONE"
            combos[key] = combos.get(key, 0) + 1
        for combo, count in sorted(combos.items(), key=lambda kv: (-kv[1], kv[0]))[:25]:
            combo_rows.append({
                "subset": subset,
                "label_combo": combo,
                "n_records": count,
                "rate": count / max(1, len(indices)),
            })
    write_csv(
        tables_dir / "label_cardinality_distribution.csv",
        cardinality_rows,
        ["subset", "positive_label_count", "n_records", "rate"],
    )
    write_csv(
        tables_dir / "label_combo_distribution_top25.csv",
        combo_rows,
        ["subset", "label_combo", "n_records", "rate"],
    )
    write_markdown_table(
        tables_dir / "label_cardinality_distribution.md",
        [{**r, "rate": fmt4(r["rate"])} for r in cardinality_rows],
        ["subset", "positive_label_count", "n_records", "rate"],
    )

    fig_path = figures_dir / "dataset_label_distribution.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(CLASS_NAMES))
    width = 0.2
    for i, subset in enumerate(["train2000", "val2000", "test_rest"]):
        vals = [
            next(
                row["positive_rate"]
                for row in rows
                if row["subset"] == subset and row["class"] == cls
            )
            for cls in CLASS_NAMES
        ]
        ax.bar(x + (i - 1) * width, vals, width=width, label=subset)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("positive rate")
    ax.set_title("PTB-XL super5 split label distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return {
        "split_json": str(SPLIT_JSON),
        "label_npy": str(LABEL_NPY),
        "n_total": int(labels.shape[0]),
        "figure": str(fig_path),
    }


def build_medical_validity_tables(tables_dir: Path, raw_dir: Path) -> dict:
    payload = load_json(MEDICAL_VALIDITY_JSON)
    rows = []
    for arm, rec in payload.get("summary_by_arm", {}).items():
        rows.append({"arm": arm, **rec})
    fields = [
        "arm",
        "n",
        "quality_pass_rate",
        "quality_warning_rate",
        "quality_fail_rate",
        "semantic_top1_rate",
        "semantic_any_positive_ge_05_rate",
        "mean_target_prob",
        "mean_einthoven_residual",
        "mean_avR_residual",
    ]
    write_csv(tables_dir / "medical_validity_summary.csv", rows, fields)
    md_rows = [
        {
            **row,
            "quality_pass_rate": fmt4(row.get("quality_pass_rate")),
            "quality_warning_rate": fmt4(row.get("quality_warning_rate")),
            "quality_fail_rate": fmt4(row.get("quality_fail_rate")),
            "semantic_top1_rate": fmt4(row.get("semantic_top1_rate")),
            "semantic_any_positive_ge_05_rate": fmt4(row.get("semantic_any_positive_ge_05_rate")),
            "mean_target_prob": fmt4(row.get("mean_target_prob")),
            "mean_einthoven_residual": fmt4(row.get("mean_einthoven_residual")),
            "mean_avR_residual": fmt4(row.get("mean_avR_residual")),
        }
        for row in rows
    ]
    write_markdown_table(tables_dir / "medical_validity_summary.md", md_rows, fields)
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MEDICAL_VALIDITY_JSON, raw_dir / "medical_validity_summary.json")
    shutil.copy2(MEDICAL_VALIDITY_CSV, raw_dir / "per_sample_quality.csv")
    return {"summary_json": str(MEDICAL_VALIDITY_JSON), "per_sample_csv": str(MEDICAL_VALIDITY_CSV)}


def build_benchmark_tables(tables_dir: Path, figures_dir: Path, raw_dir: Path) -> dict:
    payload = load_json(BENCHMARK_JSON)
    rows = payload.get("results", [])
    fields = [
        "backend",
        "device",
        "batch_size",
        "repeats",
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p95",
        "throughput_ecg_per_s",
    ]
    write_csv(tables_dir / "inference_benchmark.csv", rows, fields)
    md_rows = [
        {
            **row,
            "latency_ms_mean": fmt4(row.get("latency_ms_mean")),
            "latency_ms_p50": fmt4(row.get("latency_ms_p50")),
            "latency_ms_p95": fmt4(row.get("latency_ms_p95")),
            "throughput_ecg_per_s": fmt2(row.get("throughput_ecg_per_s")),
        }
        for row in rows
    ]
    write_markdown_table(tables_dir / "inference_benchmark.md", md_rows, fields)
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BENCHMARK_JSON, raw_dir / "inference_benchmark.json")
    if TRT_VALIDATION_JSON.exists():
        shutil.copy2(TRT_VALIDATION_JSON, raw_dir / "tensorrt_validation.json")

    selected = [row for row in rows if row.get("batch_size") in {1, 32}]
    labels = [f"{row['backend']}\nB{row['batch_size']}" for row in selected]
    lat = [row["latency_ms_p50"] for row in selected]
    thr = [row["throughput_ecg_per_s"] for row in selected]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, lat, color="#4C78A8")
    axes[0].set_ylabel("p50 latency (ms)")
    axes[0].set_title("Classifier inference latency")
    axes[0].tick_params(axis="x", labelrotation=35)
    axes[1].bar(labels, thr, color="#59A14F")
    axes[1].set_ylabel("ECG/s")
    axes[1].set_title("Classifier throughput")
    axes[1].tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    out = figures_dir / "inference_benchmark_bars.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"benchmark_json": str(BENCHMARK_JSON), "figure": str(out)}


def build_low_sample_figures(main_rows: list[dict], ablation_rows: list[dict], figures_dir: Path) -> dict:
    out: dict[str, str] = {}
    if main_rows:
        labels = [row["method"] for row in main_rows]
        auroc = [row["test_macro_auroc"] for row in main_rows]
        auprc = [row["test_macro_auprc"] for row in main_rows]
        x = np.arange(len(labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(x - width / 2, auroc, width, label="AUROC", color="#4C78A8")
        ax.bar(x + width / 2, auprc, width, label="AUPRC", color="#F58518")
        ax.set_ylim(0.5, 0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title("Low-sample PTB-XL custom test")
        ax.legend()
        for i, value in enumerate(auprc):
            ax.text(i + width / 2, value + 0.006, f"{value:.3f}", ha="center", fontsize=8)
        fig.tight_layout()
        path = figures_dir / "low_sample_main_results_bar.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        out["main_bar"] = str(path)

    rows = [row for row in ablation_rows if row.get("status") == "ok"]
    if rows:
        rows = sorted(rows, key=lambda r: float(r.get("test_macro_auprc") or 0), reverse=True)
        labels = [row["method"] for row in rows]
        values = [row["test_macro_auprc"] for row in rows]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.arange(len(labels)), values, color="#72B7B2")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("macro AUPRC")
        ax.set_title("Low-sample ablations ranked by AUPRC")
        fig.tight_layout()
        path = figures_dir / "low_sample_ablation_auprc_rank.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        out["ablation_rank"] = str(path)
    return out


def build_real_vs_synth_figures(tables_dir: Path, figures_dir: Path) -> dict:
    if not (PTBXL_CACHE.exists() and LABEL_NPY.exists() and SELECTED_SYNTH_NPZ.exists()):
        return {"status": "missing_inputs"}
    split = load_json(SPLIT_JSON)
    labels = np.load(LABEL_NPY, mmap_mode="r")
    real = np.load(PTBXL_CACHE, mmap_mode="r")
    synth = np.load(SELECTED_SYNTH_NPZ, allow_pickle=True)
    synth_signals = synth["signals"]
    synth_class_names = [str(x) for x in synth["class_names"]]
    rows = []
    out_dir = figures_dir / "real_vs_synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    for cls in CLASS_NAMES:
        class_idx = CLASS_NAMES.index(cls)
        y = np.asarray(labels[test_indices] > 0.5, dtype=np.int8)
        single = test_indices[(y[:, class_idx] == 1) & (y.sum(axis=1) == 1)]
        if len(single) == 0:
            single = test_indices[y[:, class_idx] == 1]
        if len(single) == 0:
            continue
        real_idx = int(single[0])
        synth_idx = synth_class_names.index(cls)
        path = out_dir / f"real_vs_synth_{cls}.png"
        plot_comparison(
            [real[real_idx], synth_signals[synth_idx]],
            [f"real {cls} idx={real_idx}", f"synthetic {cls}"],
            100.0,
            path,
            mode="stacked",
            lead_order="ptbxl",
            row_height=1.7,
        )
        rows.append({
            "class": cls,
            "real_ptbxl_index": real_idx,
            "synthetic_selected_index": synth_idx,
            "figure": str(path),
        })
    write_csv(
        tables_dir / "real_vs_synth_figure_manifest.csv",
        rows,
        ["class", "real_ptbxl_index", "synthetic_selected_index", "figure"],
    )
    write_markdown_table(
        tables_dir / "real_vs_synth_figure_manifest.md",
        rows,
        ["class", "real_ptbxl_index", "synthetic_selected_index", "figure"],
    )
    return {"status": "ok", "figures": [r["figure"] for r in rows]}


def draw_box(ax, xy, width, height, text, *, fc="#F4F6F8", ec="#4C78A8"):
    x, y = xy
    rect = plt.Rectangle((x, y), width, height, facecolor=fc, edgecolor=ec, linewidth=1.6)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)


def draw_arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#333333"},
    )


def build_design_diagrams(figures_dir: Path) -> dict:
    out: dict[str, str] = {}
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_axis_off()
    route_box_w = 1.55
    route = [
        ("PTB-XL\n12-lead ECG", 0.35),
        ("super5 labels\nseed42 split", 2.45),
        ("ECGTwin\nlatent diffusion", 4.55),
        ("quality gate\n+ visual audit", 6.65),
        ("EfficientNet1DV2\nlow-sample\ntraining", 8.75),
        ("TensorRT\nStreamlit demo", 10.85),
    ]
    for text, x in route:
        draw_box(ax, (x, 1.45), route_box_w, 1.1, text)
    for (_, x0), (_, x1) in zip(route, route[1:]):
        draw_arrow(ax, (x0 + route_box_w, 2.0), (x1, 2.0))
    ax.text(6.3, 0.6, "metrics: AUROC / AUPRC / per-class / latency / quality gates", ha="center", fontsize=10)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 3.2)
    fig.tight_layout()
    path = figures_dir / "technical_route_diagram.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    out["technical_route_diagram"] = str(path)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_axis_off()
    layers = [
        ("Data Layer\nPTB-XL cache, split, labels", 0.8, 4.4, "#E8F0FE"),
        ("Generation Layer\nECGTwin VAE + DiT\nprompt token", 3.8, 4.4, "#EAF7EA"),
        ("Detection Layer\nEfficientNet1DV2\nsuper5 classifier", 6.8, 4.4, "#FFF4E6"),
        ("Deployment Layer\nONNX / TensorRT / Streamlit", 3.8, 2.45, "#FDECEF"),
        ("Evidence Layer\nJSON, CSV, PNG, manifest", 3.8, 0.65, "#F5F0FF"),
    ]
    for text, x, y, fc in layers:
        draw_box(ax, (x, y), 2.4, 1.0, text, fc=fc)
    draw_arrow(ax, (3.2, 4.9), (3.8, 4.9))
    draw_arrow(ax, (6.2, 4.9), (6.8, 4.9))
    draw_arrow(ax, (8.0, 4.4), (5.0, 3.45))
    draw_arrow(ax, (5.0, 2.45), (5.0, 1.65))
    draw_arrow(ax, (2.0, 4.4), (5.0, 1.65))
    draw_arrow(ax, (5.0, 4.4), (5.0, 1.65))
    draw_arrow(ax, (8.0, 4.4), (5.0, 1.65))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.9)
    fig.tight_layout()
    path = figures_dir / "system_architecture_diagram.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    out["system_architecture_diagram"] = str(path)
    return out


def copy_static_artifacts(out_dir: Path) -> dict:
    copied: dict[str, str] = {}
    figures_dir = out_dir / "figures"
    generated_dir = figures_dir / "generated_ecg_examples"
    generated_dir.mkdir(parents=True, exist_ok=True)
    src_dir = FINAL_ROUND_ROOT / "thesis_paper_selected_v1"
    for name in [
        "thesis_CD_12lead.png",
        "thesis_HYP_12lead.png",
        "thesis_MI_12lead.png",
        "thesis_NORM_12lead.png",
        "thesis_STTC_12lead.png",
        "selected_examples.json",
        "selected_examples.md",
    ]:
        src = src_dir / name
        if src.exists():
            dst = generated_dir / name
            shutil.copy2(src, dst)
            copied[name] = str(dst)
    curve_dir = figures_dir / "training_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for src, name in [
        (Path("final/training_log_data/ibe_stage1/loss_curve.png"), "ibe_stage1_loss_curve.png"),
        (Path("final/training_log_data/dit_stage2/loss_curve.png"), "dit_stage2_loss_curve.png"),
    ]:
        abs_src = REPO / src
        if abs_src.exists():
            dst = curve_dir / name
            shutil.copy2(abs_src, dst)
            copied[name] = str(dst)
    return copied


def build_system_test_table(tables_dir: Path) -> list[dict]:
    checks = [
        {
            "test_item": "PTB-XL preprocessing cache",
            "input": str(PTBXL_CACHE),
            "expected": "(21799,1000,12) finite ECG array",
            "actual": "exists" if PTBXL_CACHE.exists() else "missing",
            "status": "pass" if PTBXL_CACHE.exists() else "fail",
        },
        {
            "test_item": "super5 label cache",
            "input": str(LABEL_NPY),
            "expected": "(21799,5), class order CD/HYP/MI/NORM/STTC",
            "actual": "exists" if LABEL_NPY.exists() else "missing",
            "status": "pass" if LABEL_NPY.exists() else "fail",
        },
        {
            "test_item": "low-sample split",
            "input": str(SPLIT_JSON),
            "expected": "train=2000, val=2000, test=17799",
            "actual": "exists" if SPLIT_JSON.exists() else "missing",
            "status": "pass" if SPLIT_JSON.exists() else "fail",
        },
        {
            "test_item": "selected generated ECG figures",
            "input": str(SELECTED_SYNTH_JSON),
            "expected": "five thesis-selected 12-lead PNGs and metadata",
            "actual": "exists" if SELECTED_SYNTH_JSON.exists() else "missing",
            "status": "pass" if SELECTED_SYNTH_JSON.exists() else "fail",
        },
        {
            "test_item": "selected generated ECG vectors",
            "input": str(SELECTED_SYNTH_NPZ),
            "expected": "signals shape (5,1000,12)",
            "actual": "exists" if SELECTED_SYNTH_NPZ.exists() else "missing",
            "status": "pass" if SELECTED_SYNTH_NPZ.exists() else "fail",
        },
        {
            "test_item": "medical validity summary",
            "input": str(MEDICAL_VALIDITY_JSON),
            "expected": "quality and semantic gate summary",
            "actual": "exists" if MEDICAL_VALIDITY_JSON.exists() else "missing",
            "status": "pass" if MEDICAL_VALIDITY_JSON.exists() else "fail",
        },
        {
            "test_item": "ONNX export",
            "input": str(STREAMLIT_ROOT / "models/efficientnetv2_super5.onnx"),
            "expected": "classifier ONNX model",
            "actual": "exists" if (STREAMLIT_ROOT / "models/efficientnetv2_super5.onnx").exists() else "missing",
            "status": "pass" if (STREAMLIT_ROOT / "models/efficientnetv2_super5.onnx").exists() else "fail",
        },
        {
            "test_item": "TensorRT engine",
            "input": str(STREAMLIT_ROOT / "models/efficientnetv2_super5_fp16.engine"),
            "expected": "FP16 TensorRT engine",
            "actual": "exists" if (STREAMLIT_ROOT / "models/efficientnetv2_super5_fp16.engine").exists() else "missing",
            "status": "pass" if (STREAMLIT_ROOT / "models/efficientnetv2_super5_fp16.engine").exists() else "fail",
        },
        {
            "test_item": "TensorRT numerical validation",
            "input": str(TRT_VALIDATION_JSON),
            "expected": "max_abs_prob_diff below 1e-3",
            "actual": "exists" if TRT_VALIDATION_JSON.exists() else "missing",
            "status": "pass" if TRT_VALIDATION_JSON.exists() else "fail",
        },
        {
            "test_item": "Streamlit application source",
            "input": "apps/streamlit_ecg_demo/app.py",
            "expected": "detection/generation/robustness/benchmark tabs",
            "actual": "exists" if (REPO / "apps/streamlit_ecg_demo/app.py").exists() else "missing",
            "status": "pass" if (REPO / "apps/streamlit_ecg_demo/app.py").exists() else "fail",
        },
    ]
    write_csv(
        tables_dir / "system_function_tests.csv",
        checks,
        ["test_item", "input", "expected", "actual", "status"],
    )
    write_markdown_table(
        tables_dir / "system_function_tests.md",
        checks,
        ["test_item", "input", "expected", "actual", "status"],
    )
    return checks


def write_readme(out_dir: Path, manifest: dict, main_rows: list[dict]) -> None:
    lines = [
        "# Thesis Evidence Pack",
        "",
        "This directory collects small, thesis-ready evidence files generated from the existing experiment artifacts.",
        "Large checkpoints and source experiment arrays remain under `/root/autodl-tmp/`.",
        "",
        "## Recommended Thesis Wording",
        "",
        "- Use PTB-XL custom seed42 low-sample test as the main undergraduate thesis evidence.",
        "- State that ECGTwin synthetic pretraining followed by real2000 fine-tuning improves low-sample classification.",
        "- Do not attribute the full gain solely to center prompt tokens, because the no-token hard-label control is strong.",
        "- Treat generated ECG figures as qualitative visualization and basic plausibility evidence, not clinical-grade validation.",
        "",
        "## Main Low-Sample Result",
        "",
        "| method | AUROC | AUPRC | delta AUROC | delta AUPRC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            "| {method} | {auroc} | {auprc} | {dauc} | {dprc} |".format(
                method=row["method"],
                auroc=fmt4(row.get("test_macro_auroc")),
                auprc=fmt4(row.get("test_macro_auprc")),
                dauc=fmt4(row.get("delta_auroc_vs_real2000")),
                dprc=fmt4(row.get("delta_auprc_vs_real2000")),
            )
        )
    lines.extend([
        "",
        "## Contents",
        "",
        "- `tables/low_sample_main_results.md`: Table 7.4 candidate.",
        "- `tables/per_class_main_results.md`: Table 7.3 / per-class detail candidate.",
        "- `tables/low_sample_ablation_results.md`: Table 7.5 candidate.",
        "- `tables/dataset_split_label_distribution.md`: Table 7.2 candidate.",
        "- `tables/medical_validity_summary.md`: Section 7.4 summary.",
        "- `tables/inference_benchmark.md`: Section 7.9 TensorRT benchmark.",
        "- `tables/system_function_tests.md`: Table 7.6 candidate.",
        "- `feature_distribution/feature_distribution_report.md`: real-vs-synth feature distribution analysis.",
        "- `figures/technical_route_diagram.png`: Figure 1.1 candidate.",
        "- `figures/system_architecture_diagram.png`: Figure 4.1 candidate.",
        "- `figures/`: thesis-ready figures and copied loss curves.",
        "- `raw/`: small JSON/CSV source summaries copied for traceability.",
        "",
        "## Manifest",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- implementation_repo_commit: `{manifest.get('implementation_repo_commit')}`",
        f"- thesis_repo_commit_before_pack: `{manifest.get('thesis_repo_commit_before_pack')}`",
        "",
    ])
    out_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final_root",
        default=str(REPO / "final"),
        help="Thesis repository root.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Evidence pack output directory. Default: <final_root>/artifacts/evidence_pack",
    )
    args = parser.parse_args()

    final_root = Path(args.final_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else final_root / "artifacts/evidence_pack"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, main_rows, per_class_rows = build_run_tables(tables_dir)
    dataset_info = build_dataset_tables(tables_dir, figures_dir)
    medical_info = build_medical_validity_tables(tables_dir, raw_dir)
    benchmark_info = build_benchmark_tables(tables_dir, figures_dir, raw_dir)
    low_sample_figures = build_low_sample_figures(
        main_rows,
        [r for r in all_rows if r.get("group") == "ablation"],
        figures_dir,
    )
    design_diagrams = build_design_diagrams(figures_dir)
    real_vs_synth_info = build_real_vs_synth_figures(tables_dir, figures_dir)
    copied = copy_static_artifacts(out_dir)
    system_tests = build_system_test_table(tables_dir)

    raw_train_dir = raw_dir / "train_results"
    raw_train_dir.mkdir(parents=True, exist_ok=True)
    for row in all_rows:
        result_json = row.get("result_json")
        src = Path(result_json) if result_json else None
        if src is not None and src.is_file():
            shutil.copy2(src, raw_train_dir / f"{row['key']}.train_result.json")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "implementation_repo": str(REPO),
        "implementation_repo_commit": git_value(["rev-parse", "HEAD"], REPO),
        "implementation_repo_branch": git_value(["branch", "--show-current"], REPO),
        "thesis_repo": str(final_root),
        "thesis_repo_commit_before_pack": git_value(["rev-parse", "HEAD"], final_root),
        "thesis_repo_branch": git_value(["branch", "--show-current"], final_root),
        "tables": sorted(str(p.relative_to(out_dir)) for p in tables_dir.glob("*")),
        "figures": sorted(str(p.relative_to(out_dir)) for p in figures_dir.rglob("*") if p.is_file()),
        "raw": sorted(str(p.relative_to(out_dir)) for p in raw_dir.rglob("*") if p.is_file()),
        "dataset_info": dataset_info,
        "medical_info": medical_info,
        "benchmark_info": benchmark_info,
        "low_sample_figures": low_sample_figures,
        "design_diagrams": design_diagrams,
        "real_vs_synth_info": real_vs_synth_info,
        "copied_static_artifacts": copied,
        "system_tests": system_tests,
        "runs": [{k: v for k, v in row.items() if k != "raw"} for row in all_rows],
        "selected_synth_npz_not_duplicated": str(SELECTED_SYNTH_NPZ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(out_dir, manifest, main_rows)
    print(json.dumps({"out_dir": str(out_dir), "manifest": str(out_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
