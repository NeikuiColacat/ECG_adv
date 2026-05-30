#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DATA_ROOT = Path(os.environ.get("ECG_ADV_DATA_ROOT", Path.home() / "autodl-tmp")).expanduser()
GRAD_ROOT = Path(os.environ.get("ECG_ADV_GRAD_ROOT", DATA_ROOT / "graduate_project")).expanduser()
FINAL_ROUND_ROOT = Path(
    os.environ.get("ECG_ADV_FINAL_ROUND_ROOT", DATA_ROOT / "final_round_ablation_20260504")
).expanduser()

DEFAULT_METHODS = {
    "real2000_baseline": GRAD_ROOT / "method_a_real2000_seed42" / "train_result.json",
    "no_token_pretrain_finetune": (
        GRAD_ROOT
        / "self_distill_v2_e24_v46_no_token_hardlabel_r10_realfine_lr1e4_seed42_auroc"
        / "train_result.json"
    ),
    "center_token_pretrain_finetune": (
        GRAD_ROOT
        / "self_distill_v2_e23_v46_class_oracle_hardlabel_r10_realfine_lr1e4_seed42_auroc"
        / "train_result.json"
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_one(name: str, path: Path) -> dict[str, Any]:
    data = load_json(path)
    split = data.get("split") or {}
    return {
        "name": name,
        "path": str(path),
        "test_macro_auroc": data["test_macro_auroc"],
        "test_macro_auprc": data["test_macro_auprc"],
        "best_val_macro_auroc": data.get("best_val_macro_auroc"),
        "best_val_macro_auprc": data.get("best_val_macro_auprc"),
        "epochs_trained": data.get("epochs_trained"),
        "checkpoint_metric": data.get("checkpoint_metric"),
        "checkpoint_path": data.get("checkpoint_path"),
        "class_names": data.get("class_names"),
        "split_counts": {
            "train": len(split.get("train_indices", [])),
            "val": len(split.get("val_indices", [])),
            "test": len(split.get("test_indices", [])),
        },
        "per_class": data.get("test_per_class", {}),
    }


def write_markdown(out_path: Path, methods: list[dict[str, Any]]) -> None:
    baseline = methods[0]
    lines = [
        "# Low-Sample PTB-XL Summary",
        "",
        "Source: archived `train_result.json` files from the fixed PTB-XL custom split.",
        "",
        "| method | AUROC | AUPRC | delta AUROC | delta AUPRC | train/val/test | epochs |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in methods:
        counts = row["split_counts"]
        lines.append(
            "| {name} | {auroc:.4f} | {auprc:.4f} | {d_auroc:+.4f} | {d_auprc:+.4f} | "
            "{train}/{val}/{test} | {epochs} |".format(
                name=row["name"],
                auroc=float(row["test_macro_auroc"]),
                auprc=float(row["test_macro_auprc"]),
                d_auroc=float(row["test_macro_auroc"]) - float(baseline["test_macro_auroc"]),
                d_auprc=float(row["test_macro_auprc"]) - float(baseline["test_macro_auprc"]),
                train=counts["train"],
                val=counts["val"],
                test=counts["test"],
                epochs=row.get("epochs_trained", ""),
            )
        )

    lines.extend(["", "## Artifact Paths", ""])
    for row in methods:
        lines.append(f"- `{row['name']}`: `{row['path']}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(FINAL_ROUND_ROOT / "low_sample_summary"))
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Override or add a method as name=/path/to/train_result.json",
    )
    args = parser.parse_args()

    methods = dict(DEFAULT_METHODS)
    for item in args.method:
        name, path = item.split("=", 1)
        methods[name] = Path(path).expanduser()

    missing = [str(path) for path in methods.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing low-sample result file(s):\n" + "\n".join(missing))

    rows = [summarize_one(name, path) for name, path in methods.items()]
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "archived train_result.json files",
        "ptbxl_protocol": "super5 train=2000 val=2000 test=17799 custom split",
        "methods": rows,
    }
    (out_dir / "low_sample_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "low_sample_summary.md", rows)
    print(f"[done] wrote {out_dir / 'low_sample_summary.json'}")
    print(f"[done] wrote {out_dir / 'low_sample_summary.md'}")


if __name__ == "__main__":
    main()
