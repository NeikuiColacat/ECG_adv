from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from scripts.agent import pn2021c_dual_model_sweep as sweep


def test_load_candidate_profiles_infers_single_operator(tmp_path: Path):
    profile_file = tmp_path / "profiles.yaml"
    profile_file.write_text(
        """
profiles:
  emg_amp2p3:
    emg_noise:
      5:
        max_amplitude: 2.3
  mask_p0p57:
    random_leads_masking:
      5:
        mask_leads_prob: 0.57
""",
        encoding="utf-8",
    )

    profiles = sweep.load_candidate_profiles(profile_file)

    assert profiles["emg_amp2p3"].operator == "emg_noise"
    assert profiles["mask_p0p57"].operator == "random_leads_masking"


def test_corruption_input_mapping_keeps_powerline_branches_explicit():
    assert sweep.corruption_input_for("effnet", "emg_amp2p3") == "raw_first"
    assert sweep.corruption_input_for("ecgfounder", "emg_amp2p3") == "bottleneck5000"
    assert sweep.corruption_input_for("effnet", "power_locked_amp8") == "raw_first"
    assert sweep.corruption_input_for("ecgfounder", "power_locked_amp8") == "bottleneck5000"
    assert sweep.corruption_input_for("effnet", "power_native_amp8") == "native_raw_first"
    assert sweep.corruption_input_for("ecgfounder", "power_native_amp8") == "native_raw_first"


def test_build_frozen_manifest_rows_uses_profile_input_modes(tmp_path: Path):
    profile_file = tmp_path / "v1.yaml"
    profile_file.write_text(
        """
profiles:
  dual_model_10to15pp_v1:
    emg_noise:
      5:
        max_amplitude: 2.3
    powerline_noise:
      5:
        max_amplitude: 20.0
metadata:
  input_modes:
    emg_noise:
      effnet: raw_first
      ecgfounder: bottleneck5000
    powerline_noise:
      effnet: native_raw_first
      ecgfounder: native_raw_first
""",
        encoding="utf-8",
    )

    rows = sweep.build_frozen_manifest_rows(
        stage_dir=tmp_path / "four_center_confirmation",
        profile_file=profile_file,
        profile_name="dual_model_10to15pp_v1",
        centers=["cpsc_2018"],
        models=sweep.MODELS,
    )

    assert len(rows) == 4
    by_key = {(row["model"], row["operator"]): row for row in rows}
    assert by_key[("effnet", "emg_noise")]["candidate"] == "dual_model_10to15pp_v1"
    assert by_key[("ecgfounder", "emg_noise")]["corruption_input"] == "bottleneck5000"
    assert by_key[("effnet", "powerline_noise")]["corruption_input"] == "native_raw_first"
    assert by_key[("ecgfounder", "powerline_noise")]["corruption_input"] == "native_raw_first"


def _write_result(path: Path, *, center: str, operator: str, clean: float, corrupt: float, clean_pr: float, corrupt_pr: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "severity_profile": "custom",
        "severity_params_name": "emg_amp2p3",
        "severity_profile_metadata": {
            "severity_params_file_sha256": "abc123",
        },
        "per_center": {
            center: {
                operator: {
                    "5": {
                        "clean_macro_auroc": clean,
                        "macro_auroc": corrupt,
                        "auroc_drop_vs_clean": clean - corrupt,
                        "clean_macro_auprc": clean_pr,
                        "macro_auprc": corrupt_pr,
                        "auprc_drop_vs_clean": clean_pr - corrupt_pr,
                        "n_records": 42,
                        "n_all_zero_labels": 7,
                        "corruption": {
                            "resolved_params": {
                                "max_amplitude": 2.3,
                                "p": 1.0,
                            },
                        },
                    }
                }
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_manifest_outputs_gap_and_score(tmp_path: Path):
    eff_json = tmp_path / "eff.json"
    ecg_json = tmp_path / "ecg.json"
    _write_result(
        eff_json,
        center="cpsc_2018",
        operator="emg_noise",
        clean=0.90,
        corrupt=0.78,
        clean_pr=0.70,
        corrupt_pr=0.58,
    )
    _write_result(
        ecg_json,
        center="cpsc_2018",
        operator="emg_noise",
        clean=0.92,
        corrupt=0.81,
        clean_pr=0.72,
        corrupt_pr=0.60,
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "center",
                "operator",
                "candidate",
                "corruption_input",
                "output_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "effnet",
                "center": "cpsc_2018",
                "operator": "emg_noise",
                "candidate": "emg_amp2p3",
                "corruption_input": "raw_first",
                "output_json": str(eff_json),
            }
        )
        writer.writerow(
            {
                "model": "ecgfounder",
                "center": "cpsc_2018",
                "operator": "emg_noise",
                "candidate": "emg_amp2p3",
                "corruption_input": "bottleneck5000",
                "output_json": str(ecg_json),
            }
        )
    out_csv = tmp_path / "candidate_scores.csv"

    rows = sweep.aggregate_manifest(manifest, out_csv)

    assert len(rows) == 2
    by_model = {row["model"]: row for row in rows}
    assert by_model["effnet"]["drop_auroc_pp"] == "12.0000"
    assert by_model["ecgfounder"]["drop_auroc_pp"] == "11.0000"
    assert by_model["effnet"]["inter_model_gap_pp"] == "1.0000"
    assert by_model["ecgfounder"]["inter_model_gap_pp"] == "1.0000"
    assert by_model["effnet"]["accepted_on_two_center"] == "true"
    assert by_model["effnet"]["severity_profile"] == "custom"
    assert by_model["effnet"]["severity_params_name"] == "emg_amp2p3"
    assert by_model["effnet"]["n_records"] == "42"
    assert by_model["effnet"]["n_all_zero_labels"] == "7"
    assert '"max_amplitude":2.3' in by_model["effnet"]["resolved_params_json"]
    assert out_csv.exists()


def test_aggregate_manifest_can_use_realistic_hospital_score_profile(tmp_path: Path):
    eff_json = tmp_path / "eff.json"
    ecg_json = tmp_path / "ecg.json"
    _write_result(
        eff_json,
        center="cpsc_2018",
        operator="baseline_wander",
        clean=0.90,
        corrupt=0.84,
        clean_pr=0.70,
        corrupt_pr=0.64,
    )
    _write_result(
        ecg_json,
        center="cpsc_2018",
        operator="baseline_wander",
        clean=0.92,
        corrupt=0.84,
        clean_pr=0.72,
        corrupt_pr=0.62,
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "center",
                "operator",
                "candidate",
                "corruption_input",
                "output_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "effnet",
                "center": "cpsc_2018",
                "operator": "baseline_wander",
                "candidate": "wander_amp1p0_k3_f0p6",
                "corruption_input": "raw_first",
                "output_json": str(eff_json),
            }
        )
        writer.writerow(
            {
                "model": "ecgfounder",
                "center": "cpsc_2018",
                "operator": "baseline_wander",
                "candidate": "wander_amp1p0_k3_f0p6",
                "corruption_input": "bottleneck5000",
                "output_json": str(ecg_json),
            }
        )

    default_rows = sweep.aggregate_manifest(manifest, tmp_path / "stress_scores.csv")
    realistic_rows = sweep.aggregate_manifest(
        manifest,
        tmp_path / "realistic_scores.csv",
        score_profile="realistic_hospital",
    )

    assert default_rows[0]["accepted_on_two_center"] == "false"
    assert realistic_rows[0]["accepted_on_two_center"] == "true"
    assert realistic_rows[0]["score_profile"] == "realistic_hospital"


def test_powerline_branch_uses_input_mode_for_frozen_profile_rows():
    assert (
        sweep.powerline_branch_for(
            "powerline_noise",
            "dual_model_10to15pp_v1",
            "native_raw_first",
        )
        == "native_raw_first"
    )
    assert (
        sweep.powerline_branch_for(
            "powerline_noise",
            "dual_model_10to15pp_v1",
            "bottleneck5000",
        )
        == "locked_path"
    )


def test_write_command_scripts_can_split_rows_into_shards(tmp_path: Path):
    rows = [
        {
            "model": "effnet",
            "center": "cpsc_2018",
            "operator": "emg_noise",
            "candidate": f"emg_amp{i}",
            "corruption_input": "raw_first",
            "output_json": str(tmp_path / f"out{i}.json"),
        }
        for i in range(5)
    ]

    paths = sweep.write_command_scripts(
        rows,
        tmp_path / "commands.sh",
        profile_file=tmp_path / "profiles.yaml",
        limit=16,
        device="cpu",
        num_shards=2,
    )

    assert [path.name for path in paths] == ["commands_shard0.sh", "commands_shard1.sh"]
    assert paths[0].read_text().count("[run]") == 3
    assert paths[1].read_text().count("[run]") == 2
    script_text = paths[0].read_text()
    assert "micromamba run" not in script_text
    assert "/home/linbinhao/micromamba/envs/ECGTwin/bin/python" in script_text


def test_parser_no_limit_clears_default_limit():
    parser = sweep.build_parser()

    args = parser.parse_args(["plan", "--no_limit"])

    assert args.limit is None


def test_select_best_candidates_prefers_accepted_then_lowest_score(tmp_path: Path):
    scores = tmp_path / "candidate_scores.csv"
    with scores.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator",
                "candidate",
                "model",
                "center",
                "score",
                "accepted_on_two_center",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "operator": "emg_noise",
                    "candidate": "emg_amp2p0",
                    "model": "effnet",
                    "center": "cpsc_2018",
                    "score": "4.0",
                    "accepted_on_two_center": "false",
                    "status": "ok",
                },
                {
                    "operator": "emg_noise",
                    "candidate": "emg_amp2p3",
                    "model": "effnet",
                    "center": "cpsc_2018",
                    "score": "9.0",
                    "accepted_on_two_center": "true",
                    "status": "ok",
                },
                {
                    "operator": "baseline_shift",
                    "candidate": "shift_amp2p4_r0p90_seg6",
                    "model": "effnet",
                    "center": "cpsc_2018",
                    "score": "1.5",
                    "accepted_on_two_center": "false",
                    "status": "ok",
                },
            ]
        )

    selected = sweep.select_best_candidates(scores)

    assert selected["emg_noise"] == "emg_amp2p3"
    assert selected["baseline_shift"] == "shift_amp2p4_r0p90_seg6"


def test_write_frozen_profile_copies_selected_operator_profiles(tmp_path: Path):
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text(
        """
profiles:
  emg_amp2p3:
    emg_noise:
      5:
        max_amplitude: 2.3
  shift_amp2p4_r0p90_seg6:
    baseline_shift:
      5:
        max_amplitude: 2.4
""",
        encoding="utf-8",
    )
    out = tmp_path / "v1.yaml"

    sweep.write_frozen_profile(
        selected={
            "emg_noise": "emg_amp2p3",
            "baseline_shift": "shift_amp2p4_r0p90_seg6",
        },
        candidate_profile_file=candidates,
        output_file=out,
        profile_name="dual_model_10to15pp_v1",
    )

    data = yaml.safe_load(out.read_text())
    profile = data["profiles"]["dual_model_10to15pp_v1"]
    assert profile["emg_noise"][5]["max_amplitude"] == 2.3
    assert profile["baseline_shift"][5]["max_amplitude"] == 2.4
    assert data["selection"]["emg_noise"] == "emg_amp2p3"


def test_write_frozen_profile_records_input_modes_for_selected_candidates(tmp_path: Path):
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text(
        """
profiles:
  power_native_amp20:
    powerline_noise:
      5:
        max_amplitude: 20.0
  mask_p0p57:
    random_leads_masking:
      5:
        mask_leads_prob: 0.57
""",
        encoding="utf-8",
    )
    out = tmp_path / "v1.yaml"

    sweep.write_frozen_profile(
        selected={
            "powerline_noise": "power_native_amp20",
            "random_leads_masking": "mask_p0p57",
        },
        candidate_profile_file=candidates,
        output_file=out,
        profile_name="dual_model_10to15pp_v1",
    )

    metadata = yaml.safe_load(out.read_text())["metadata"]
    assert metadata["input_modes"]["powerline_noise"] == {
        "effnet": "native_raw_first",
        "ecgfounder": "native_raw_first",
    }
    assert metadata["input_modes"]["random_leads_masking"] == {
        "effnet": "raw_first",
        "ecgfounder": "bottleneck5000",
    }


def test_write_selection_report_includes_selected_candidates_and_params(tmp_path: Path):
    scores = tmp_path / "candidate_scores.csv"
    with scores.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator",
                "candidate",
                "model",
                "center",
                "drop_auroc_pp",
                "drop_auprc_pp",
                "inter_model_gap_pp",
                "score",
                "accepted_on_two_center",
                "powerline_branch",
                "resolved_params_json",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "operator": "emg_noise",
                "candidate": "emg_amp2p3",
                "model": "effnet",
                "center": "cpsc_2018",
                "drop_auroc_pp": "11.0",
                "drop_auprc_pp": "13.0",
                "inter_model_gap_pp": "1.0",
                "score": "0.0",
                "accepted_on_two_center": "true",
                "powerline_branch": "",
                "resolved_params_json": '{"max_amplitude":2.3}',
                "status": "ok",
            }
        )
    frozen = tmp_path / "v1.yaml"
    frozen.write_text(
        """
profiles:
  dual_model_10to15pp_v1:
    emg_noise:
      5:
        max_amplitude: 2.3
selection:
  emg_noise: emg_amp2p3
metadata:
  profile_name: dual_model_10to15pp_v1
""",
        encoding="utf-8",
    )
    report = tmp_path / "selection.md"

    sweep.write_selection_report(scores, frozen, report)

    text = report.read_text()
    assert "dual_model_10to15pp_v1" in text
    assert "emg_noise" in text
    assert "emg_amp2p3" in text
    assert "11.00" in text
    assert "max_amplitude" in text


def test_write_confirmation_summary_marks_in_band_operator(tmp_path: Path):
    scores = tmp_path / "four_center_scores.csv"
    with scores.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator",
                "candidate",
                "model",
                "center",
                "drop_auroc_pp",
                "drop_auprc_pp",
                "corruption_input",
                "powerline_branch",
                "status",
            ],
        )
        writer.writeheader()
        for model, drop in [("effnet", "12.0"), ("ecgfounder", "11.0")]:
            for center in ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]:
                writer.writerow(
                    {
                        "operator": "powerline_noise",
                        "candidate": "dual_model_10to15pp_v1",
                        "model": model,
                        "center": center,
                        "drop_auroc_pp": drop,
                        "drop_auprc_pp": "13.0",
                        "corruption_input": "native_raw_first",
                        "powerline_branch": "native_raw_first",
                        "status": "ok",
                    }
                )
    out_csv = tmp_path / "confirmation.csv"
    out_md = tmp_path / "confirmation.md"

    rows = sweep.write_confirmation_summary(scores, out_csv, out_md)

    assert rows[0]["four_center_status"] == "in_band"
    assert rows[0]["effnet_mean_drop_auroc_pp"] == "12.0000"
    assert rows[0]["ecgfounder_mean_drop_auroc_pp"] == "11.0000"
    assert rows[0]["corruption_inputs"] == "native_raw_first"
    assert rows[0]["powerline_branch"] == "native_raw_first"
    assert "in_band" in out_md.read_text()


def test_write_confirmation_summary_uses_realistic_hospital_status(tmp_path: Path):
    scores = tmp_path / "four_center_scores.csv"
    with scores.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator",
                "candidate",
                "model",
                "center",
                "drop_auroc_pp",
                "drop_auprc_pp",
                "corruption_input",
                "powerline_branch",
                "status",
            ],
        )
        writer.writeheader()
        for model, drop in [("effnet", "6.0"), ("ecgfounder", "8.0")]:
            for center in ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]:
                writer.writerow(
                    {
                        "operator": "baseline_wander",
                        "candidate": "realistic_hospital_v1",
                        "model": model,
                        "center": center,
                        "drop_auroc_pp": drop,
                        "drop_auprc_pp": "8.0",
                        "corruption_input": "raw_first",
                        "powerline_branch": "",
                        "status": "ok",
                    }
                )
    out_csv = tmp_path / "confirmation.csv"
    out_md = tmp_path / "confirmation.md"

    rows = sweep.write_confirmation_summary(
        scores,
        out_csv,
        out_md,
        score_profile="realistic_hospital",
    )

    assert rows[0]["four_center_status"] == "realistic_main"
    assert "realistic_hospital" in out_md.read_text()
