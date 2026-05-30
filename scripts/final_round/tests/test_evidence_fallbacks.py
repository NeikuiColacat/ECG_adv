import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.final_round import summarize_final_thesis_evidence as final_summary
from scripts.final_round import summarize_low_sample_results as low_sample


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def _write_complete_table68(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "key,status,test_macro_auroc,test_macro_auprc,run_dir,result_json",
            "method_b_real_synth_mv4,ok,0.8477,0.6419,,",
            "method_b_actual_report,ok,0.8412,0.6239,,",
            "method_b_classfallback,ok,0.8432,0.6065,,",
            "synthetic_only_no_token_20k,ok,0.6023,0.3851,,",
            "center_token_hard_ft,ok,0.8735,0.7013,,",
        ])
        + "\n",
        encoding="utf-8",
    )
    return path


def test_low_sample_defaults_fall_back_to_repo_evidence_pack(tmp_path):
    evidence_pack = tmp_path / "repo" / "artifacts" / "evidence_pack"
    real = _touch(evidence_pack / "raw/train_results/real2000_original.train_result.json")
    no_token = _touch(evidence_pack / "raw/train_results/no_token_hard_ft.train_result.json")
    center = _touch(evidence_pack / "raw/train_results/center_token_hard_ft.train_result.json")

    methods = low_sample.resolve_default_methods(
        grad_root=tmp_path / "missing_grad_root",
        evidence_pack=evidence_pack,
    )

    assert methods == {
        "real2000_baseline": real,
        "no_token_pretrain_finetune": no_token,
        "center_token_pretrain_finetune": center,
    }


def test_final_summary_defaults_fall_back_to_repo_evidence_pack(tmp_path):
    evidence_pack = tmp_path / "repo" / "artifacts" / "evidence_pack"
    real = _touch(evidence_pack / "raw/train_results/real2000_original.train_result.json")
    no_token = _touch(evidence_pack / "raw/train_results/no_token_hard_ft.train_result.json")
    center = _touch(evidence_pack / "raw/train_results/center_token_hard_ft.train_result.json")

    paths = final_summary.resolve_default_low_sample_results(
        grad_root=tmp_path / "missing_grad_root",
        evidence_pack=evidence_pack,
    )

    assert paths == {
        "real_result": real,
        "no_token_result": no_token,
        "center_token_result": center,
    }


def test_final_summary_prefers_latest_table68_rerun_over_archived_csv(tmp_path):
    grad_root = tmp_path / "graduate_project"
    evidence_pack = tmp_path / "repo" / "artifacts" / "evidence_pack"
    archived = _touch(evidence_pack / "tables/low_sample_ablation_results.csv")
    old = _write_complete_table68(grad_root / "table_6_8_ablation_20260501_010101/table_6_8_summary.csv")
    latest = _write_complete_table68(grad_root / "table_6_8_ablation_20260530_020202/table_6_8_summary.csv")

    resolved = final_summary.resolve_default_table68(
        grad_root=grad_root,
        evidence_pack=evidence_pack,
    )

    assert resolved == latest
    assert resolved != old
    assert resolved != archived


def test_final_summary_ignores_incomplete_table68_rerun(tmp_path):
    grad_root = tmp_path / "graduate_project"
    evidence_pack = tmp_path / "repo" / "artifacts" / "evidence_pack"
    archived = _touch(evidence_pack / "tables/low_sample_ablation_results.csv")
    incomplete = grad_root / "table_6_8_ablation_20260530_020202/table_6_8_summary.csv"
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.write_text(
        "\n".join([
            "key,status,test_macro_auroc,test_macro_auprc",
            "method_b_real_synth_mv4,missing,,",
        ])
        + "\n",
        encoding="utf-8",
    )

    resolved = final_summary.resolve_default_table68(
        grad_root=grad_root,
        evidence_pack=evidence_pack,
    )

    assert resolved == archived


def test_final_summary_sanitizes_table68_archived_host_paths(tmp_path):
    table68 = tmp_path / "low_sample_ablation_results.csv"
    table68.write_text(
        "\n".join([
            "key,test_macro_auroc,test_macro_auprc,run_dir,result_json",
            "method_b_real_synth_mv4,0.8477,0.6419,"
            "/root/autodl-tmp/graduate_project/method_b,"
            "/root/autodl-tmp/graduate_project/method_b/train_result.json",
        ])
        + "\n",
        encoding="utf-8",
    )

    rows = final_summary.summarize_table68(table68)["rows"]

    assert rows[0]["run_dir"] == "${ECG_ADV_GRAD_ROOT}/method_b"
    assert rows[0]["result_json"] == "${ECG_ADV_GRAD_ROOT}/method_b/train_result.json"


def test_final_summary_sanitizes_nested_host_paths():
    payload = {
        "figure": "/home/neiku/autodl-tmp/final_round_ablation_20260504/figures/example.png",
        "rows": [
            {"model": "/root/autodl-tmp/streamlit_ecg_demo/models/efficientnetv2_super5.onnx"}
        ],
    }

    sanitized = final_summary.sanitize_paths(payload)

    assert sanitized["figure"] == "${ECG_ADV_FINAL_ROUND_ROOT}/figures/example.png"
    assert sanitized["rows"][0]["model"] == "${ECG_ADV_APP_DATA_ROOT}/models/efficientnetv2_super5.onnx"
