#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()
GRAD_ROOT = Path(os.environ.get("ECG_ADV_GRAD_ROOT", DATA_ROOT / "graduate_project")).expanduser()
STREAMLIT_ROOT = Path(os.environ.get("ECG_ADV_APP_DATA_ROOT", DATA_ROOT / "streamlit_ecg_demo")).expanduser()
FINAL_ROUND_ROOT = Path(
    os.environ.get("ECG_ADV_FINAL_ROUND_ROOT", DATA_ROOT / "final_round_ablation_20260504")
).expanduser()
EVIDENCE_PACK = REPO_ROOT / "artifacts" / "evidence_pack"


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def resolve_default_low_sample_results(
    *,
    grad_root: Path = GRAD_ROOT,
    evidence_pack: Path = EVIDENCE_PACK,
) -> dict[str, Path]:
    evidence_results = evidence_pack / "raw" / "train_results"
    return {
        "real_result": first_existing(
            grad_root / "method_a_real2000_seed42/train_result.json",
            evidence_results / "real2000_original.train_result.json",
        ),
        "no_token_result": first_existing(
            grad_root
            / "self_distill_v2_e24_v46_no_token_hardlabel_r10_realfine_lr1e4_seed42_auroc"
            / "train_result.json",
            evidence_results / "no_token_hard_ft.train_result.json",
        ),
        "center_token_result": first_existing(
            grad_root
            / "self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc"
            / "train_result.json",
            evidence_results / "center_token_hard_ft.train_result.json",
        ),
    }


DEFAULT_LOW_SAMPLE_RESULTS = resolve_default_low_sample_results()
DEFAULT_REAL = DEFAULT_LOW_SAMPLE_RESULTS["real_result"]
DEFAULT_NO_TOKEN = DEFAULT_LOW_SAMPLE_RESULTS["no_token_result"]
DEFAULT_CENTER_TOKEN = DEFAULT_LOW_SAMPLE_RESULTS["center_token_result"]


DEFAULT_BENCHMARK = first_existing(
    STREAMLIT_ROOT / "reports/inference_benchmark.json",
    EVIDENCE_PACK / "raw" / "inference_benchmark.json",
)
DEFAULT_FIGURES = first_existing(
    FINAL_ROUND_ROOT / "thesis_selected_ecg_examples/selected_examples.json",
    EVIDENCE_PACK / "figures" / "generated_ecg_examples" / "selected_examples.json",
)
DEFAULT_MEDICAL = first_existing(
    FINAL_ROUND_ROOT / "medical_validity" / "medical_validity_summary.json",
    EVIDENCE_PACK / "raw" / "medical_validity_summary.json",
)
DEFAULT_TABLE68 = EVIDENCE_PACK / "tables" / "low_sample_ablation_results.csv"
DEFAULT_OUT_DIR = FINAL_ROUND_ROOT / "final_evidence"


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct_delta(value: float, base: float) -> float:
    return float(value) - float(base)


def summarize_method(name: str, path: Path) -> dict:
    data = load_json(path)
    return {
        "name": name,
        "path": str(path),
        "test_macro_auroc": data["test_macro_auroc"],
        "test_macro_auprc": data["test_macro_auprc"],
        "best_val_macro_auroc": data.get("best_val_macro_auroc"),
        "best_val_macro_auprc": data.get("best_val_macro_auprc"),
        "checkpoint_metric": data.get("checkpoint_metric"),
        "checkpoint_path": data.get("checkpoint_path"),
        "epochs_trained": data.get("epochs_trained"),
        "per_class": data.get("test_per_class", {}),
        "config": data.get("config", {}),
    }


def summarize_benchmark(path: Path) -> dict:
    data = load_json(path)
    rows = []
    for row in data.get("results", []):
        if row.get("batch_size") in {1, 32}:
            rows.append({
                "backend": row.get("backend"),
                "device": row.get("device"),
                "batch_size": row.get("batch_size"),
                "latency_ms_p50": row.get("latency_ms_p50"),
                "latency_ms_p95": row.get("latency_ms_p95"),
                "throughput_ecg_per_s": row.get("throughput_ecg_per_s"),
            })
    return {
        "path": str(path),
        "rows": rows,
        "tensorrt_validation": data.get("tensorrt_validation", {}),
    }


def summarize_medical_validity(path: Path) -> dict:
    data = load_json(path)
    keep = ["target_token_s05", "no_token"]
    rows = []
    for arm in keep:
        row = data.get("summary_by_arm", {}).get(arm)
        if row:
            rows.append({"arm": arm, **row})
    return {"path": str(path), "rows": rows}


def summarize_table68(path: Path) -> dict:
    wanted = {
        "method_b_real_synth_mv4": "提示向量联合训练",
        "method_b_actual_report": "报告文本联合训练",
        "method_b_classfallback": "默认文本联合训练",
        "synthetic_only_no_token_20k": "无提示向量仅合成训练",
        "center_token_hard_ft": "中心提示向量预训练",
    }
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("key", "")
            if key in wanted:
                rows.append({
                    "key": key,
                    "paper_method": wanted[key],
                    "test_macro_auroc": float(row["test_macro_auroc"]),
                    "test_macro_auprc": float(row["test_macro_auprc"]),
                    "run_dir": row.get("run_dir", ""),
                    "result_json": row.get("result_json", ""),
                })
    if not any(row["key"] == "center_token_hard_ft" for row in rows):
        center_path = EVIDENCE_PACK / "raw" / "train_results" / "center_token_hard_ft.train_result.json"
        if center_path.exists():
            data = load_json(center_path)
            rows.append({
                "key": "center_token_hard_ft",
                "paper_method": "中心提示向量预训练",
                "test_macro_auroc": float(data["test_macro_auroc"]),
                "test_macro_auprc": float(data["test_macro_auprc"]),
                "run_dir": str(Path(data.get("config", {}).get("output_dir", ""))),
                "result_json": str(center_path),
            })
    return {"path": str(path), "rows": rows}


def write_markdown(out_path: Path, payload: dict) -> None:
    methods = payload["low_sample_methods"]
    real = methods[0]
    lines = [
        "# Final Thesis Evidence Summary",
        "",
        "## Low-Sample PTB-XL Custom Test",
        "",
        "| method | AUROC | AUPRC | delta AUROC vs real | delta AUPRC vs real | epochs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            "| {name} | {auroc:.4f} | {auprc:.4f} | {d_auroc:+.4f} | {d_auprc:+.4f} | {epochs} |".format(
                name=method["name"],
                auroc=method["test_macro_auroc"],
                auprc=method["test_macro_auprc"],
                d_auroc=pct_delta(method["test_macro_auroc"], real["test_macro_auroc"]),
                d_auprc=pct_delta(method["test_macro_auprc"], real["test_macro_auprc"]),
                epochs=method.get("epochs_trained"),
            )
        )

    lines.extend([
        "",
        "## Per-Class Custom Test",
        "",
        "| method | class | AUROC | AUPRC | n_pos |",
        "|---|---|---:|---:|---:|",
    ])
    for method in methods:
        for class_name, rec in method.get("per_class", {}).items():
            lines.append(
                "| {method} | {class_name} | {auroc:.4f} | {auprc:.4f} | {n_pos} |".format(
                    method=method["name"],
                    class_name=class_name,
                    auroc=rec["auroc"],
                    auprc=rec["auprc"],
                    n_pos=rec.get("n_pos", ""),
                )
            )

    lines.extend([
        "",
        "## Synthetic ECG Quality Proxy",
        "",
        "| arm | n | top1 consistency | mean target prob | Einthoven residual | aVR residual |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["medical_validity"]["rows"]:
        lines.append(
            "| {arm} | {n} | {semantic_top1_rate:.4f} | {mean_target_prob:.4f} | {mean_einthoven_residual:.4f} | {mean_avR_residual:.4f} |".format(**row)
        )

    lines.extend([
        "",
        "## Table 6.8 Ablation Rows",
        "",
        "| method | AUROC | AUPRC | source result |",
        "|---|---:|---:|---|",
    ])
    for row in payload["table68_ablation"]["rows"]:
        lines.append(
            "| {paper_method} | {test_macro_auroc:.4f} | {test_macro_auprc:.4f} | `{result_json}` |".format(**row)
        )

    lines.extend([
        "",
        "## TensorRT Classifier Inference",
        "",
        "| backend | device | batch | p50 latency ms | p95 latency ms | throughput ECG/s |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in payload["benchmark"]["rows"]:
        lines.append(
            "| {backend} | {device} | {batch_size} | {latency_ms_p50:.4f} | {latency_ms_p95:.4f} | {throughput_ecg_per_s:.2f} |".format(**row)
        )
    val = payload["benchmark"].get("tensorrt_validation", {})
    if val:
        lines.extend([
            "",
            "TensorRT probability drift versus PyTorch:",
            f"- max_abs_prob_diff: `{val.get('max_abs_prob_diff')}`",
            f"- mean_abs_prob_diff: `{val.get('mean_abs_prob_diff')}`",
        ])

    lines.extend([
        "",
        "## Five-Class Synthetic ECG Figures",
        "",
        "| class | source index | quality | target conf | figure |",
        "|---|---:|---|---:|---|",
    ])
    for row in payload["five_class_figures"].get("selected", []):
        quality_status = row.get("quality_status")
        if quality_status is None:
            quality_status = row.get("quality_gate", {}).get("status", "")
        lines.append(
            "| {class_name} | {source_index} | {quality_status} | {target_conf:.4f} | `{png}` |".format(
                class_name=row.get("class_name", ""),
                source_index=row.get("source_index", ""),
                quality_status=quality_status,
                target_conf=float(row.get("target_conf", 0.0)),
                png=row.get("png", ""),
            )
        )

    lines.extend([
        "",
        "## Thesis Wording Boundary",
        "",
        "- The low-sample result supports ECGTwin synthetic pretraining / real fine-tuning as useful for PTB-XL custom test.",
        "- The current strict no-token matched control is strong; do not attribute the full gain only to center token.",
        "- Five-class figures are qualitative visualization examples. HYP/CD still need cautious wording if digital clinical criteria are discussed.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_result", default=str(DEFAULT_REAL))
    ap.add_argument("--no_token_result", default=str(DEFAULT_NO_TOKEN))
    ap.add_argument("--center_token_result", default=str(DEFAULT_CENTER_TOKEN))
    ap.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    ap.add_argument("--figures", default=str(DEFAULT_FIGURES))
    ap.add_argument("--medical_validity", default=str(DEFAULT_MEDICAL))
    ap.add_argument("--table68", default=str(DEFAULT_TABLE68))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        summarize_method("real2000 baseline", Path(args.real_result)),
        summarize_method("no-token hard pretrain -> real fine-tune", Path(args.no_token_result)),
        summarize_method("center-token hard pretrain -> real fine-tune", Path(args.center_token_result)),
    ]
    payload = {
        "low_sample_methods": methods,
        "medical_validity": summarize_medical_validity(Path(args.medical_validity)),
        "table68_ablation": summarize_table68(Path(args.table68)),
        "benchmark": summarize_benchmark(Path(args.benchmark)),
        "five_class_figures": load_json(args.figures),
    }
    json_path = out_dir / "final_thesis_evidence_summary.json"
    md_path = out_dir / "final_thesis_evidence_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
