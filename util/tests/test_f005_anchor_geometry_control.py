"""CPU-only contract tests for the isolated F005 anchor-geometry control."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from adversarial.latent_hull_pgd import LatentHullPGDGenerator
from ecg_adv_gen.adaptation.lhat import SameLabelLatentIndex, build_exact_eligibility_manifest
from ecg_adv_gen.adaptation.latent_hull_torch import identity_collapsed_weight_metrics
from ecg_adv_gen.config import ConfigError, build_runner_commands, load_experiment_config
from ecg_adv_gen.f005_control import (
    F005_BASE_TOPOLOGY,
    F005_STUDY_SCOPE,
    F005_VARIANTS,
    build_f005_control_run_leaf,
    validate_f005_case,
    validate_f005_paired_bundle,
)
from ecg_adv_gen.matched_effnet import MATCHED_EFFNET_ARMS
from ecg_adv_gen.reporting import PaperTableError, export_paper_table
from ecg_adv_gen.training.resume_contract import RESUME_CONTRACT_KEYS, resume_contract_mismatches


REPO = Path(__file__).resolve().parents[2]
LOCAL = REPO / "configs/local/linbinhao_server.example.yaml"
TRAIN = REPO / "configs/studies/f005_anchor_geometry_control_train.yaml"
SMOKE = REPO / "configs/studies/f005_anchor_geometry_control_smoke.yaml"
CLEAN = REPO / "configs/studies/f005_anchor_geometry_control_pn2021_clean.yaml"
S5 = REPO / "configs/studies/f005_anchor_geometry_control_pn2021c_s5.yaml"
DEPTH23 = REPO / "configs/studies/f005_anchor_geometry_control_pn2021c_depth23.yaml"


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _load(path: Path, run_id: str = "pytest_f005") -> dict:
    return load_experiment_config(path, LOCAL, runtime_context={"run_id": run_id})


def _study_case(seed: int, variant: str) -> dict[str, object]:
    return {
        "arm": "a5",
        "role": "vae_lhat_threechain_full",
        "vae_lhat": True,
        "raw_augmix": True,
        "augmix_view_bce": True,
        "jsd": True,
        "rho": 0.5,
        "third_chain_route": "vae_lhat_adversarial_waveform",
        "study_scope": F005_STUDY_SCOPE,
        "base_topology": F005_BASE_TOPOLOGY,
        "variant": variant,
        "seed": seed,
        "primary_comparison_eligible": False,
    }


def test_f005_variant_table_is_exact_pair_and_only_lambda_differs() -> None:
    assert tuple(F005_VARIANTS) == ("balanced_l060", "neighbor_only_l100")
    balanced, neighbor = (F005_VARIANTS[name] for name in F005_VARIANTS)
    assert balanced.hull_lambda == pytest.approx(0.6)
    assert neighbor.hull_lambda == pytest.approx(1.0)
    assert balanced.label_mode == neighbor.label_mode == "exact"
    assert balanced.include_anchor is neighbor.include_anchor is False
    assert balanced.base_topology == neighbor.base_topology == "a5"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"variant": "unknown"}, "unknown F005 variant"),
        ({"arm": "a3", "role": "vae_lhat_only"}, "base topology"),
        ({"primary_comparison_eligible": True}, "primary_comparison_eligible"),
        ({"hull_lambda": 0.2}, "case-level hull_lambda"),
    ],
)
def test_f005_case_validation_fails_closed(mutation: dict[str, object], match: str) -> None:
    case = _study_case(20260601, "balanced_l060")
    case.update(mutation)
    with pytest.raises(ValueError, match=match):
        validate_f005_case(case)


def test_exact_singleton_never_silently_falls_back_to_anchor() -> None:
    latents = np.zeros((3, 4, 128), dtype=np.float32)
    labels = np.asarray([[1, 0], [0, 1], [0, 1]], dtype=np.float32)
    index = SameLabelLatentIndex(latents, labels, label_mode="exact", include_self=False)

    with pytest.raises(ValueError, match=r"anchor 0.*no non-self"):
        index.candidates_for(np.asarray([0]), 2)


def test_exact_eligibility_manifest_is_deterministic_and_requires_two_nonself() -> None:
    labels = np.asarray(
        [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.float32
    )
    record_ids = np.asarray([f"r{i}" for i in range(len(labels))])
    first = build_exact_eligibility_manifest(labels, record_ids, min_nonself=2)
    second = build_exact_eligibility_manifest(labels, record_ids, min_nonself=2)

    assert first == second
    assert first["eligible_pool_indices"] == [0, 1, 2]
    assert first["eligible_record_ids"] == ["r0", "r1", "r2"]
    assert first["eligible_count"] == 3 and first["total_count"] == 6
    assert first["exact_label_sets"]["10"]["eligible_count"] == 3
    assert first["exact_label_sets"]["01"]["eligible_count"] == 0
    assert len(first["ordered_record_ids_sha256"]) == 64
    assert len(first["manifest_sha256"]) == 64


def test_exact_candidates_have_no_anchor_and_at_least_two_unique_nonself_ids() -> None:
    latents = np.arange(3 * 4 * 128, dtype=np.float32).reshape(3, 4, 128)
    labels = np.ones((3, 1), dtype=np.float32)
    index = SameLabelLatentIndex(latents, labels, label_mode="exact", include_self=False)

    candidates = index.candidates_for(np.asarray([0]), 20)

    assert candidates.shape == (1, 20, 4, 128)
    identities = index.last_candidate_indices[0]
    assert 0 not in identities
    assert set(identities) == {1, 2}


def test_identity_metrics_collapse_duplicate_positions_before_entropy() -> None:
    metrics = identity_collapsed_weight_metrics(
        candidate_pool_indices=torch.tensor([[4, 4, 9, 9]]),
        weights=torch.full((1, 4), 0.25),
        candidate_is_anchor=torch.zeros((1, 4), dtype=torch.bool),
    )

    assert metrics["candidate_unique_nonself_count"] == [2]
    assert metrics["candidate_duplicate_fraction"] == pytest.approx([0.5])
    assert metrics["identity_top1_weight"] == pytest.approx([0.5])
    assert metrics["identity_entropy"] == pytest.approx([np.log(2.0)])
    assert metrics["effective_candidate_count"] == pytest.approx([2.0])
    assert metrics["candidate_anchor_count"] == [0]


class _ToyVictim(torch.nn.Module):
    num_classes = 1

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Identity()

    def forward_from_latent_to_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.flatten(1).mean(dim=1, keepdim=True)


class _ToyGenerator(LatentHullPGDGenerator):
    def _decode_to_ptbxl_1000(self, latent: torch.Tensor) -> torch.Tensor:
        value = latent.flatten(1).mean(dim=1).view(-1, 1, 1)
        return value.expand(-1, 12, 1000)


def test_inner_attack_scores_projected_states_and_atk_init_snapshot_is_projected() -> None:
    generator = _ToyGenerator(
        ecgtwin_wrapper=object(), victim=_ToyVictim(), epsilon=0.25,
        hull_lambda=1.0, hull_steps=3, hull_lr=0.5, init_logit_gap=0.0, device="cpu",
    )
    z0 = torch.zeros(1, 4, 128)
    candidates = torch.stack([torch.ones_like(z0), torch.full_like(z0, 4.0)], dim=1)

    _, delta = generator.attack_from_latent(z0, torch.zeros(1, 1), candidates)

    assert delta.flatten(1).norm(dim=1).max().item() <= 0.250001
    assert generator.last_info["inner_post_projection_norm_max"] <= 0.250001
    assert generator.last_initial_signals[0, 0, 0].item() == pytest.approx(
        generator.last_info["initial_projected_latent_mean"][0]
    )
    assert generator.last_info["projected_final_bce_mean"] >= (
        generator.last_info["projected_initial_bce_mean"] - 1e-7
    )


def test_resume_contract_rejects_study_variant_or_eligibility_drift() -> None:
    keys = {"study_scope", "mechanism_variant", "eligibility_manifest_sha256"}
    assert keys <= set(RESUME_CONTRACT_KEYS)
    for key in keys:
        assert resume_contract_mismatches({key: "saved"}, {key: "current"}) == [
            {"key": key, "saved": "saved", "current": "current"}
        ]


def test_f005_run_leaf_is_unique_by_scope_variant_seed_and_lambda() -> None:
    balanced = build_f005_control_run_leaf("ningbo", 20260601, "balanced_l060", epochs=30)
    neighbor = build_f005_control_run_leaf("ningbo", 20260601, "neighbor_only_l100", epochs=30)
    other_seed = build_f005_control_run_leaf("ningbo", 20260611, "balanced_l060", epochs=30)
    assert len({balanced, neighbor, other_seed}) == 3
    assert F005_STUDY_SCOPE in balanced
    assert "balanced_l060" in balanced and "lam0p6" in balanced
    assert "neighbor_only_l100" in neighbor and "lam1" in neighbor


def test_smoke_and_full_training_expand_exact_paired_cells() -> None:
    smoke = build_runner_commands(_load(SMOKE, "pytest_f005_smoke"))
    full = build_runner_commands(_load(TRAIN, "pytest_f005_full"))
    assert len(smoke) == 2
    assert len(full) == 24
    assert {(c["matrix"]["center"], c["matrix"]["case"]["seed"], c["matrix"]["case"]["variant"])
            for c in full} == {
        (center, seed, variant)
        for center in ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
        for seed in (20260531, 20260601, 20260611)
        for variant in F005_VARIANTS
    }


def test_balanced_and_neighbor_commands_differ_only_by_declared_treatment() -> None:
    commands = build_runner_commands(_load(SMOKE, "pytest_f005_equal"))
    by_variant = {c["matrix"]["case"]["variant"]: c["argv"] for c in commands}

    def scrub(argv: list[str]) -> list[str]:
        out = list(argv)
        for option in ("--mechanism_variant", "--hull_lambda"):
            out[out.index(option) + 1] = f"<{option}>"
        return out

    assert scrub(by_variant["balanced_l060"]) == scrub(by_variant["neighbor_only_l100"])
    assert _option(by_variant["balanced_l060"], "--hull_lambda") == "0.6"
    assert _option(by_variant["neighbor_only_l100"], "--hull_lambda") == "1.0"
    for argv in by_variant.values():
        assert _option(argv, "--study_scope") == F005_STUDY_SCOPE
        assert _option(argv, "--hull_label_mode") == "exact"
        assert "--hull_include_anchor" not in argv
        assert _option(argv, "--comparison_arm") == "a5"
        assert "paper_matched_effnet_k500_v7_fixedk_three_seed_20260711/subsets" in _option(
            argv, "--anchor_base"
        )


@pytest.mark.parametrize("path", [CLEAN, S5, DEPTH23])
def test_each_consumer_expands_24_and_points_to_exact_variant_producer(path: Path) -> None:
    commands = build_runner_commands(_load(path, f"pytest_{path.stem}"))
    assert len(commands) == 24
    for command in commands:
        case = command["matrix"]["case"]
        model_dir = _option(command["argv"], "--model_dir")
        assert case["variant"] in model_dir
        assert f"seed{case['seed']}" in model_dir
        assert F005_STUDY_SCOPE in model_dir
        assert _option(command["argv"], "--checkpoint_name") == "best_model.pt"


def test_study_matrix_rejects_duplicate_or_missing_cells_before_adapter() -> None:
    duplicate = _load(TRAIN, "pytest_f005_duplicate")
    duplicate["runner"]["matrix"]["case"].append(dict(duplicate["runner"]["matrix"]["case"][0]))
    with pytest.raises(ConfigError, match="duplicate.*seed.*variant"):
        build_runner_commands(duplicate)

    missing = _load(TRAIN, "pytest_f005_missing")
    missing["runner"]["matrix"]["case"].pop()
    with pytest.raises(ConfigError, match="complete seed x variant"):
        build_runner_commands(missing)


def test_primary_paper_table_rejects_mechanism_study_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run_id", "dataset", "view", "canonical_view", "scope", "center",
            "class_name", "metric", "value", "mapping_version", "mapping_hash",
            "class_order", "source_file", "study_scope",
        ])
        writer.writeheader()
        for metric in ("macro_auroc", "macro_auprc"):
            writer.writerow({
                "run_id": "balanced_l060", "dataset": "pn2021", "view": "all_zero_kept_refexcluded",
                "canonical_view": "all_zero_kept_refexcluded", "scope": "center", "center": "ningbo",
                "metric": metric, "value": "0.8", "mapping_version": "v7", "mapping_hash": "hash",
                "class_order": "CD|HYP|MI|NORM|STTC", "source_file": "/tmp/source.json",
                "study_scope": F005_STUDY_SCOPE,
            })
    with pytest.raises(PaperTableError, match="mechanism-study"):
        export_paper_table(path, tmp_path / "table", view="all_zero_kept_refexcluded")


def _bundle_records() -> list[dict[str, object]]:
    records = []
    for center in ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"):
        for seed in (20260531, 20260601, 20260611):
            for variant, spec in F005_VARIANTS.items():
                records.append({
                    "center": center, "seed": seed, "study_scope": F005_STUDY_SCOPE,
                    "base_topology": "a5", "variant": variant,
                    "primary_comparison_eligible": False, "hull_lambda": spec.hull_lambda,
                    "hull_label_mode": "exact", "hull_include_anchor": False,
                    "git_sha": "abc123", "source_checkpoint_sha256": "source-sha",
                    "k500_train_ids_sha256": f"k500-{center}-{seed}",
                    "eligibility_manifest_sha256": f"eligible-{center}-{seed}",
                    "selection_metric": "macro_auprc", "source_floor_max_drop": 0.02,
                    "realized_optimizer_steps": 300, "scheduler_steps": 30,
                    "consumers": {name: {"status": "complete"} for name in ("clean", "s5", "depth23")},
                    "attack_audit": {
                        "training_steps": 5, "audit_steps": 20,
                        "training_loss_gain_median": 0.8, "audit_loss_gain_median": 1.0,
                    },
                })
    return records


def test_dedicated_paired_bundle_accepts_only_complete_exact_matched_surface() -> None:
    bundle = validate_f005_paired_bundle(_bundle_records())
    assert bundle["n_records"] == 24
    assert bundle["n_pairs"] == 12
    assert bundle["study_scope"] == F005_STUDY_SCOPE
    assert bundle["five_step_retention_min"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda rows: rows.pop(), "complete paired"),
        (lambda rows: rows[0].update(hull_label_mode="compatible"), "exact"),
        (lambda rows: rows[0].update(git_sha="other"), "mixed git"),
        (lambda rows: rows[0].update(consumers={"clean": {"status": "complete"}}), "consumers"),
        (lambda rows: rows[0]["attack_audit"].update(training_loss_gain_median=0.7), "80%"),
    ],
)
def test_dedicated_paired_bundle_fails_closed(mutate, match: str) -> None:
    records = _bundle_records()
    mutate(records)
    with pytest.raises(ValueError, match=match):
        validate_f005_paired_bundle(records)


def test_canonical_surface_and_partner_policy_remain_unchanged_until_data_preflight_passes() -> None:
    assert MATCHED_EFFNET_ARMS == ("a0", "a2", "a3", "a4", "a5")
    canonical = yaml.safe_load((REPO / "configs/defaults/vae_lhat_defaults.yaml").read_text())
    assert canonical["adaptation"]["hull"]["label_mode"] == "compatible"
    assert "f005_anchor_geometry_control" not in (
        REPO / "configs/golden/latest_mainline_contract_v1.json"
    ).read_text(encoding="utf-8")


def test_separate_f005_golden_is_generated_and_current() -> None:
    golden = json.loads((REPO / "configs/golden/f005_anchor_geometry_control_v1.json").read_text())
    assert golden["study_scope"] == F005_STUDY_SCOPE
    assert golden["command_counts"] == {"clean": 24, "depth23": 24, "s5": 24, "smoke": 2, "train": 24}
    assert golden["variants"] == list(F005_VARIANTS)
