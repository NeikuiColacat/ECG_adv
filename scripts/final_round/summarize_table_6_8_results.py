#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()
GRAD_ROOT = Path(os.environ.get("ECG_ADV_GRAD_ROOT", DATA_ROOT / "graduate_project")).expanduser()
PTBXL_ROOT = Path(os.environ.get("ECG_ADV_PTBXL_ROOT", DATA_ROOT / "ptbxl")).expanduser()
TRIPLE_ROOT = Path(os.environ.get("ECG_ADV_TRIPLE_ROOT", DATA_ROOT / "triple_labels")).expanduser()
APP_DATA_ROOT = Path(os.environ.get("ECG_ADV_APP_DATA_ROOT", DATA_ROOT / "streamlit_ecg_demo")).expanduser()
FINAL_ROUND_ROOT = Path(
    os.environ.get("ECG_ADV_FINAL_ROUND_ROOT", DATA_ROOT / "final_round_ablation_20260504")
).expanduser()


ABLATION_SPECS = [
    {
        "key": "method_b_real_synth_mv4",
        "paper_method": "提示向量联合训练",
        "run_subdir": "center_token_joint_seed42",
        "note": "Center/prompt-token generated pool jointly trained with real2000.",
    },
    {
        "key": "method_b_actual_report",
        "paper_method": "报告文本联合训练",
        "run_subdir": "report_text_joint_seed42",
        "note": "Actual-report text generated pool jointly trained with real2000.",
    },
    {
        "key": "method_b_classfallback",
        "paper_method": "默认文本联合训练",
        "run_subdir": "default_text_joint_seed42",
        "note": "Default class-text generated pool jointly trained with real2000.",
    },
    {
        "key": "synthetic_only_no_token_20k",
        "paper_method": "无提示向量仅合成训练",
        "run_subdir": "no_token_synthetic_only_seed5042",
        "note": "No-token ECGTwin synthetic-only classifier training.",
    },
    {
        "key": "center_token_hard_ft",
        "paper_method": "中心提示向量预训练",
        "run_subdir": "center_token_pretrain_realfine_seed42",
        "note": "Center-token synthetic pretraining checkpoint fine-tuned on real2000.",
    },
]

FIELDNAMES = [
    "key",
    "paper_method",
    "run_subdir",
    "status",
    "test_macro_auroc",
    "test_macro_auprc",
    "best_val_macro_auroc",
    "best_val_macro_auprc",
    "epochs_trained",
    "checkpoint_metric",
    "seed",
    "lr",
    "batch_size",
    "preprocess_mode",
    "norm_mode",
    "synth_npz",
    "init_ckpt",
    "run_dir",
    "result_json",
    "note",
]


def _relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def portable_path(path: str | Path | None, *, run_root: Path) -> str:
    if not path:
        return ""
    path_obj = Path(path).expanduser()
    roots = [
        (run_root, "${TABLE_6_8_RUN_ROOT}"),
        (GRAD_ROOT, "${ECG_ADV_GRAD_ROOT}"),
        (PTBXL_ROOT, "${ECG_ADV_PTBXL_ROOT}"),
        (TRIPLE_ROOT, "${ECG_ADV_TRIPLE_ROOT}"),
        (APP_DATA_ROOT, "${ECG_ADV_APP_DATA_ROOT}"),
        (FINAL_ROUND_ROOT, "${ECG_ADV_FINAL_ROUND_ROOT}"),
        (DATA_ROOT, "${ECG_ADV_DATA_ROOT}"),
    ]
    for root, marker in roots:
        rel = _relative_to(path_obj, root)
        if rel is not None:
            rel_text = rel.as_posix()
            return marker if rel_text == "." else f"{marker}/{rel_text}"
    return str(path)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def summarize_run_root(run_root: str | Path) -> list[dict[str, str]]:
    run_root = Path(run_root).expanduser()
    rows: list[dict[str, str]] = []
    for spec in ABLATION_SPECS:
        run_dir = run_root / spec["run_subdir"]
        result_json = run_dir / "train_result.json"
        row: dict[str, str] = {
            "key": spec["key"],
            "paper_method": spec["paper_method"],
            "run_subdir": spec["run_subdir"],
            "status": "missing",
            "test_macro_auroc": "",
            "test_macro_auprc": "",
            "best_val_macro_auroc": "",
            "best_val_macro_auprc": "",
            "epochs_trained": "",
            "checkpoint_metric": "",
            "seed": "",
            "lr": "",
            "batch_size": "",
            "preprocess_mode": "",
            "norm_mode": "",
            "synth_npz": "",
            "init_ckpt": "",
            "run_dir": portable_path(run_dir, run_root=run_root),
            "result_json": portable_path(result_json, run_root=run_root),
            "note": spec["note"],
        }
        if result_json.exists():
            data = json.loads(result_json.read_text(encoding="utf-8"))
            config = data.get("config", {})
            row.update({
                "status": "ok",
                "test_macro_auroc": _cell(data.get("test_macro_auroc")),
                "test_macro_auprc": _cell(data.get("test_macro_auprc")),
                "best_val_macro_auroc": _cell(data.get("best_val_macro_auroc")),
                "best_val_macro_auprc": _cell(data.get("best_val_macro_auprc")),
                "epochs_trained": _cell(data.get("epochs_trained")),
                "checkpoint_metric": _cell(data.get("checkpoint_metric", config.get("checkpoint_metric"))),
                "seed": _cell(config.get("seed")),
                "lr": _cell(config.get("lr")),
                "batch_size": _cell(config.get("batch_size")),
                "preprocess_mode": _cell(config.get("preprocess_mode")),
                "norm_mode": _cell(config.get("norm_mode")),
                "synth_npz": portable_path(config.get("synth_npz"), run_root=run_root),
                "init_ckpt": portable_path(config.get("init_ckpt"), run_root=run_root),
            })
        rows.append(row)
    return rows


def write_summary(run_root: str | Path, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    run_root = Path(run_root).expanduser()
    out_dir = Path(out_dir).expanduser() if out_dir else run_root
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = summarize_run_root(run_root)
    csv_path = out_dir / "table_6_8_summary.csv"
    json_path = out_dir / "table_6_8_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": portable_path(run_root, run_root=run_root),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    csv_path, json_path = write_summary(args.run_root, args.out_dir)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
