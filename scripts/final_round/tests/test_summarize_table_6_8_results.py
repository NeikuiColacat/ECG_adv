import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.final_round import summarize_table_6_8_results as table68


def _write_result(run_root: Path, subdir: str, *, auroc: float, auprc: float, config: dict) -> None:
    output_dir = run_root / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_macro_auroc": auroc,
        "test_macro_auprc": auprc,
        "best_val_macro_auroc": auroc - 0.01,
        "best_val_macro_auprc": auprc - 0.01,
        "checkpoint_metric": "auroc",
        "epochs_trained": config.get("epochs", 50),
        "config": config,
    }
    (output_dir / "train_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_table68_summary_writes_paper_order_csv_and_json(tmp_path):
    run_root = tmp_path / "table_6_8_ablation_demo"
    common = {
        "seed": 42,
        "lr": 0.01,
        "batch_size": 64,
        "preprocess_mode": "minimal_resample",
        "norm_mode": "per_sample_global",
        "checkpoint_metric": "auroc",
    }
    _write_result(
        run_root,
        "center_token_joint_seed42",
        auroc=0.8477,
        auprc=0.6419,
        config={**common, "synth_npz": "/data/center/gated_samples.npz"},
    )
    _write_result(
        run_root,
        "report_text_joint_seed42",
        auroc=0.8412,
        auprc=0.6239,
        config={**common, "synth_npz": "/data/report/gated_samples.npz"},
    )
    _write_result(
        run_root,
        "default_text_joint_seed42",
        auroc=0.8432,
        auprc=0.6065,
        config={**common, "synth_npz": "/data/default/gated_samples.npz"},
    )
    _write_result(
        run_root,
        "no_token_synthetic_only_seed5042",
        auroc=0.6023,
        auprc=0.3851,
        config={**common, "seed": 5042, "epochs": 30, "synth_npz": "/data/no_token/samples.npz"},
    )
    _write_result(
        run_root,
        "center_token_pretrain_realfine_seed42",
        auroc=0.8735,
        auprc=0.7013,
        config={**common, "seed": 9042, "lr": 0.0001, "epochs": 25, "init_ckpt": "/data/center/best_model.pt"},
    )

    csv_path, json_path = table68.write_summary(run_root)

    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert [row["key"] for row in rows] == [
        "method_b_real_synth_mv4",
        "method_b_actual_report",
        "method_b_classfallback",
        "synthetic_only_no_token_20k",
        "center_token_hard_ft",
    ]
    assert rows[0]["paper_method"] == "提示向量联合训练"
    assert rows[0]["run_dir"] == "${TABLE_6_8_RUN_ROOT}/center_token_joint_seed42"
    assert rows[0]["result_json"] == "${TABLE_6_8_RUN_ROOT}/center_token_joint_seed42/train_result.json"
    assert rows[-1]["init_ckpt"] == "/data/center/best_model.pt"
    assert float(rows[-1]["test_macro_auprc"]) == 0.7013

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_root"] == "${TABLE_6_8_RUN_ROOT}"
    assert len(payload["rows"]) == 5
