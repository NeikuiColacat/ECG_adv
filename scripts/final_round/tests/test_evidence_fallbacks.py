import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.final_round import summarize_final_thesis_evidence as final_summary
from scripts.final_round import summarize_low_sample_results as low_sample


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
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
