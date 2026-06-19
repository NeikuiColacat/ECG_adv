#!/usr/bin/env python3
"""Plan and aggregate PN2021-C dual-model corruption calibration sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/linbinhao/ECG_adv_data")
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "runs/pn2021c_dual_model_10to15pp_sweep_20260618"
DEFAULT_PROFILE_FILE = (
    REPO_ROOT
    / "configs/corruption_profiles/pn2021c_dual_model_10to15pp_candidates_20260618.yaml"
)
DEFAULT_CENTERS = ("cpsc_2018", "georgia")
FOUR_CENTERS = ("ningbo", "chapman_shaoxing", "cpsc_2018", "georgia")
MODELS = ("effnet", "ecgfounder")
PUBLIC_SEVERITY = 5
DEFAULT_SCORE_PROFILE = "stress_10to15"
SCORE_PROFILES = {
    "stress_10to15": {
        "score_lo": 10.0,
        "score_hi": 15.0,
        "accept_lo": 10.0,
        "accept_hi": 15.0,
        "gap_hi": 3.0,
    },
    "realistic_hospital": {
        "score_lo": 5.0,
        "score_hi": 10.0,
        "accept_lo": 5.0,
        "accept_hi": 12.0,
        "gap_hi": 3.0,
    },
}
PN2021_REQUIRED_CACHE_VERSION = "v7_refexcluded_100hz1000"
PN2021_CLEAN_MMAP_CACHE_DIR = DATA_ROOT / "triple_labels/pn2021_eval_cache_mmap_minresample_perglobal"
PN2021_CLEAN_CACHE_DIR = DATA_ROOT / "triple_labels/pn2021_eval_cache_minresample_perglobal"
PN2021_ROOT = DATA_ROOT / "physionet2021"
REF_META_ROOT = DATA_ROOT / "paper_vae_only_latenthull_sweep_20260516_v7_sjr_rgq/subsets"
EFFNET_ROOT = (
    DATA_ROOT
    / "runs/effnet_direct_k500_v7_sjr_rgq_matrix/mainline_v7_k500_4gpu_20260605"
)
ECGFOUNDER_ROOT = (
    DATA_ROOT / "runs/ecgfounder_k500_fullft_locked/locked_rawfirst_20260618/runs"
)
ECGFOUNDER_CHECKPOINT = DATA_ROOT / "ecgfounder/checkpoint/12_lead_ECGFounder.pth"
PYTHON_EXECUTABLE = Path("/home/linbinhao/micromamba/envs/ECGTwin/bin/python")


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    operator: str
    params: dict


def load_candidate_profiles(profile_file: Path) -> dict[str, CandidateProfile]:
    data = yaml.safe_load(profile_file.read_text())
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict):
        raise ValueError(f"{profile_file} must contain a top-level profiles mapping")
    out: dict[str, CandidateProfile] = {}
    for name, body in profiles.items():
        if not isinstance(body, dict) or len(body) != 1:
            raise ValueError(f"profile {name!r} must define exactly one corruption operator")
        operator = next(iter(body))
        out[str(name)] = CandidateProfile(name=str(name), operator=str(operator), params=dict(body))
    return out


def corruption_input_for(model: str, candidate: str) -> str:
    model = str(model)
    candidate = str(candidate)
    if candidate.startswith("power_native_"):
        return "native_raw_first"
    if model == "effnet":
        return "raw_first"
    if model == "ecgfounder":
        return "bottleneck5000"
    raise ValueError(f"unknown model: {model}")


def effnet_model_dir(center: str) -> Path:
    return (
        EFFNET_ROOT
        / center
        / "runs"
        / f"{center}_K500_direct_ft_ep30_seed20260531_val0.2"
    )


def effnet_clean_eval_json(center: str) -> Path:
    return effnet_model_dir(center) / "eval_result_v7_super5_sjr_rgq_review_exclrefs_crop1000.json"


def ecgfounder_run_dir(center: str) -> Path:
    return ECGFOUNDER_ROOT / f"{center}_k500_fullft_locked"


def ref_meta_json(center: str) -> Path:
    return REF_META_ROOT / center / "k500_seed20260531" / f"{center}_real_k500_seed20260531.ref_meta.json"


def shell_join(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _output_json(stage_dir: Path, model: str, center: str, candidate: str, operator: str) -> Path:
    return stage_dir / model / center / operator / f"{candidate}.json"


def build_manifest_rows(
    *,
    stage_dir: Path,
    centers: Iterable[str],
    candidate_profiles: dict[str, CandidateProfile],
    models: Iterable[str] = MODELS,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for center in centers:
        for candidate_name, profile in candidate_profiles.items():
            for model in models:
                corruption_input = corruption_input_for(model, candidate_name)
                rows.append(
                    {
                        "model": model,
                        "center": center,
                        "operator": profile.operator,
                        "candidate": candidate_name,
                        "corruption_input": corruption_input,
                        "output_json": str(
                            _output_json(stage_dir, model, center, candidate_name, profile.operator)
                        ),
                    }
                )
    return rows


def _default_input_mode_for_frozen(model: str, operator: str) -> str:
    if model == "effnet":
        return "raw_first"
    if model == "ecgfounder":
        return "bottleneck5000"
    raise ValueError(f"unknown model: {model}")


def build_frozen_manifest_rows(
    *,
    stage_dir: Path,
    profile_file: Path,
    profile_name: str,
    centers: Iterable[str],
    models: Iterable[str] = MODELS,
) -> list[dict[str, str]]:
    data = yaml.safe_load(profile_file.read_text())
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError(f"{profile_file} must define profiles.{profile_name}")
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(f"profiles.{profile_name} must be a corruption-operator mapping")
    input_modes = (data.get("metadata", {}) or {}).get("input_modes", {}) or {}

    rows: list[dict[str, str]] = []
    for center in centers:
        for operator in profile:
            for model in models:
                corruption_input = input_modes.get(operator, {}).get(model)
                if not corruption_input:
                    corruption_input = _default_input_mode_for_frozen(model, operator)
                rows.append(
                    {
                        "model": model,
                        "center": str(center),
                        "operator": str(operator),
                        "candidate": str(profile_name),
                        "corruption_input": str(corruption_input),
                        "output_json": str(
                            _output_json(stage_dir, model, str(center), str(profile_name), str(operator))
                        ),
                    }
                )
    return rows


def _effnet_command(row: dict[str, str], *, profile_file: Path, limit: int | None, device: str) -> list[object]:
    center = row["center"]
    cmd: list[object] = [
        PYTHON_EXECUTABLE,
        "-u",
        "scripts/triple_labels/eval_pn2021_corruptions.py",
        "--mode",
        "stream",
        "--scheme",
        "super5",
        "--model_dir",
        effnet_model_dir(center),
        "--clean_eval_json",
        effnet_clean_eval_json(center),
        "--checkpoint_name",
        "best_model.pt",
        "--clean_mmap_cache_dir",
        PN2021_CLEAN_MMAP_CACHE_DIR,
        "--clean_cache_dir",
        PN2021_CLEAN_CACHE_DIR,
        "--required_cache_version",
        PN2021_REQUIRED_CACHE_VERSION,
        "--centers",
        center,
        "--corruptions",
        row["operator"],
        "--severities",
        PUBLIC_SEVERITY,
        "--severity_profile",
        "custom",
        "--severity_params_file",
        profile_file,
        "--severity_params_name",
        row["candidate"],
        "--corruption_input",
        row["corruption_input"],
        "--pn2021_root",
        PN2021_ROOT,
        "--exclude_ref_ids",
        ref_meta_json(center),
        "--device",
        device,
        "--crop_len",
        1000,
        "--batch_size",
        192 if device == "cuda" else 16,
        "--num_workers",
        0,
        "--min_pos",
        10,
        "--seed",
        20260501,
        "--output_path",
        row["output_json"],
    ]
    if limit is not None:
        cmd.extend(["--limit", int(limit)])
    return cmd


def _ecgfounder_command(row: dict[str, str], *, profile_file: Path, limit: int | None, device: str) -> list[object]:
    center = row["center"]
    cmd: list[object] = [
        PYTHON_EXECUTABLE,
        "-u",
        "scripts/triple_labels/eval_ecgfounder_pn2021_corruptions.py",
        "--scheme",
        "super5",
        "--run_dir",
        ecgfounder_run_dir(center),
        "--variant",
        "ecgfounder_k500_fullft_locked",
        "--checkpoint",
        ECGFOUNDER_CHECKPOINT,
        "--clean_mmap_cache_dir",
        PN2021_CLEAN_MMAP_CACHE_DIR,
        "--clean_cache_dir",
        PN2021_CLEAN_CACHE_DIR,
        "--required_cache_version",
        PN2021_REQUIRED_CACHE_VERSION,
        "--centers",
        center,
        "--corruptions",
        row["operator"],
        "--severities",
        PUBLIC_SEVERITY,
        "--severity_profile",
        "custom",
        "--severity_params_file",
        profile_file,
        "--severity_params_name",
        row["candidate"],
        "--corruption_input",
        row["corruption_input"],
        "--pn2021_root",
        PN2021_ROOT,
        "--device",
        device,
        "--crop_len",
        1000,
        "--batch_size",
        96 if device == "cuda" else 8,
        "--num_workers",
        0,
        "--min_pos",
        10,
        "--seed",
        20260501,
        "--output_path",
        row["output_json"],
    ]
    if limit is not None:
        cmd.extend(["--limit", int(limit)])
    return cmd


def build_command(row: dict[str, str], *, profile_file: Path, limit: int | None, device: str) -> str:
    if row["model"] == "effnet":
        cmd = _effnet_command(row, profile_file=profile_file, limit=limit, device=device)
    elif row["model"] == "ecgfounder":
        cmd = _ecgfounder_command(row, profile_file=profile_file, limit=limit, device=device)
    else:
        raise ValueError(f"unknown model: {row['model']}")
    env = [
        "TMPDIR=/home/linbinhao/tmp",
        "ECG_ADV_GEN_DATA_ROOT=/home/linbinhao/ECG_adv_data",
    ]
    if row["model"] == "ecgfounder":
        env.append("ECGFOUNDER_ROOT=/home/linbinhao/ECG_adv_data/ecgfounder")
    if device == "cuda":
        env.insert(0, 'CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"')
    return " ".join(env + [shell_join(cmd)])


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
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
        writer.writerows(rows)


def write_command_script(rows: list[dict[str, str]], path: Path, *, profile_file: Path, limit: int | None, device: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "mkdir -p /home/linbinhao/tmp",
        "",
    ]
    for row in rows:
        out = shlex.quote(row["output_json"])
        lines.extend(
            [
                f"mkdir -p {shlex.quote(str(Path(row['output_json']).parent))}",
                f"if [ -s {out} ]; then",
                f"  echo '[skip] {row['model']} {row['center']} {row['candidate']} -> {row['output_json']}'",
                "else",
                f"  echo '[run] {row['model']} {row['center']} {row['candidate']}'",
                "  " + build_command(row, profile_file=profile_file, limit=limit, device=device),
                "fi",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return path


def write_command_scripts(
    rows: list[dict[str, str]],
    path: Path,
    *,
    profile_file: Path,
    limit: int | None,
    device: str,
    num_shards: int = 1,
) -> list[Path]:
    num_shards = int(num_shards)
    if num_shards <= 1:
        return [
            write_command_script(
                rows,
                path,
                profile_file=profile_file,
                limit=limit,
                device=device,
            )
        ]
    paths: list[Path] = []
    stem = path.stem
    suffix = path.suffix or ".sh"
    for shard_idx in range(num_shards):
        shard_rows = [row for idx, row in enumerate(rows) if idx % num_shards == shard_idx]
        shard_path = path.with_name(f"{stem}_shard{shard_idx}{suffix}")
        paths.append(
            write_command_script(
                shard_rows,
                shard_path,
                profile_file=profile_file,
                limit=limit,
                device=device,
            )
        )
    return paths


def _metric_row(payload: dict, *, center: str, operator: str) -> dict:
    return payload["per_center"][center][operator][str(PUBLIC_SEVERITY)]


def _drop_pp(row: dict, clean_key: str, metric_key: str, drop_key: str) -> float | None:
    if row.get(drop_key) is not None:
        return float(row[drop_key]) * 100.0
    if row.get(clean_key) is None or row.get(metric_key) is None:
        return None
    return (float(row[clean_key]) - float(row[metric_key])) * 100.0


def _score_profile_config(score_profile: str) -> dict[str, float]:
    if score_profile not in SCORE_PROFILES:
        raise ValueError(f"unknown score_profile {score_profile!r}; choices={sorted(SCORE_PROFILES)}")
    return SCORE_PROFILES[score_profile]


def _band_penalty_pp(value: float, lo: float = 10.0, hi: float = 15.0) -> float:
    if lo <= value <= hi:
        return 0.0
    if value < lo:
        return (lo - value) ** 2
    return (value - hi) ** 2


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _json_compact(value: object) -> str:
    if value is None or value == "":
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def powerline_branch_for(operator: str, candidate: str, corruption_input: str = "") -> str:
    if str(operator) != "powerline_noise":
        return ""
    if str(corruption_input) == "native_raw_first":
        return "native_raw_first"
    if str(corruption_input) in {"raw_first", "bottleneck5000"}:
        return "locked_path"
    candidate = str(candidate)
    if candidate.startswith("power_native_"):
        return "native_raw_first"
    if candidate.startswith("power_locked_"):
        return "locked_path"
    return "unknown"


def aggregate_manifest(
    manifest_path: Path,
    output_csv: Path,
    *,
    score_profile: str = DEFAULT_SCORE_PROFILE,
) -> list[dict[str, str]]:
    score_cfg = _score_profile_config(score_profile)
    with manifest_path.open() as f:
        manifest_rows = list(csv.DictReader(f))
    parsed: list[dict] = []
    for item in manifest_rows:
        output_json = Path(item["output_json"])
        if not output_json.exists():
            parsed.append({**item, "status": "missing"})
            continue
        payload = json.loads(output_json.read_text())
        result = _metric_row(payload, center=item["center"], operator=item["operator"])
        label_mapping = payload.get("label_mapping", {}) or {}
        pn2021c_protocol = payload.get("pn2021c_protocol", {}) or {}
        drop_auroc_pp = _drop_pp(
            result,
            "clean_macro_auroc",
            "macro_auroc",
            "auroc_drop_vs_clean",
        )
        drop_auprc_pp = _drop_pp(
            result,
            "clean_macro_auprc",
            "macro_auprc",
            "auprc_drop_vs_clean",
        )
        parsed.append(
            {
                **item,
                "status": "ok",
                "severity_profile": payload.get("severity_profile"),
                "severity_params_name": payload.get("severity_params_name"),
                "severity_params_file": payload.get("severity_params_file"),
                "severity_params_file_sha256": (
                    payload.get("severity_profile_metadata", {}) or {}
                ).get("severity_params_file_sha256"),
                "required_cache_version": payload.get("required_cache_version"),
                "pn2021_c_cache_version": payload.get("pn2021_c_cache_version"),
                "pn2021_mapping_version": label_mapping.get("version"),
                "pn2021_mapping_hash": label_mapping.get("hash"),
                "label_mapping_json": _json_compact(payload.get("label_mapping")),
                "pn2021c_input_order_id": pn2021c_protocol.get("input_order_id"),
                "pn2021c_protocol_json": _json_compact(pn2021c_protocol),
                "exclude_ref_ids_by_center_counts_json": _json_compact(
                    payload.get("exclude_ref_ids_by_center_counts")
                ),
                "selected_ref_record_ids_count": payload.get("selected_ref_record_ids_count"),
                "clean_auroc": result.get("clean_macro_auroc"),
                "corrupted_auroc": result.get("macro_auroc"),
                "drop_auroc_pp_float": drop_auroc_pp,
                "clean_auprc": result.get("clean_macro_auprc"),
                "corrupted_auprc": result.get("macro_auprc"),
                "drop_auprc_pp_float": drop_auprc_pp,
                "n_records": result.get("n_records"),
                "n_all_zero_labels": result.get("n_all_zero_labels"),
                "drop_all_zero_n_records": result.get("drop_all_zero_n_records"),
                "drop_all_zero_macro_auroc": result.get("drop_all_zero_macro_auroc"),
                "drop_all_zero_macro_auprc": result.get("drop_all_zero_macro_auprc"),
                "n_excluded_ref_ids_for_center": result.get("n_excluded_ref_ids_for_center"),
                "ref_record_ids_sha256": result.get("ref_record_ids_sha256"),
                "cache_source": result.get("cache_source"),
                "clean_metric_source": result.get("clean_metric_source"),
                "resolved_params_json": _json_compact(
                    (result.get("corruption", {}) or {}).get("resolved_params")
                ),
            }
        )

    group_stats: dict[tuple[str, str], dict[str, float | bool]] = {}
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in parsed:
        if row.get("status") != "ok":
            continue
        groups.setdefault((row["operator"], row["candidate"]), []).append(row)
    for key, rows in groups.items():
        by_model: dict[str, list[float]] = {"effnet": [], "ecgfounder": []}
        for row in rows:
            drop = row.get("drop_auroc_pp_float")
            if drop is not None and row["model"] in by_model:
                by_model[row["model"]].append(float(drop))
        eff_mean = sum(by_model["effnet"]) / len(by_model["effnet"]) if by_model["effnet"] else None
        ecg_mean = sum(by_model["ecgfounder"]) / len(by_model["ecgfounder"]) if by_model["ecgfounder"] else None
        if eff_mean is None or ecg_mean is None:
            group_stats[key] = {"gap": None, "score": None, "accepted": False}
            continue
        gap = abs(eff_mean - ecg_mean)
        score = (
            _band_penalty_pp(eff_mean, score_cfg["score_lo"], score_cfg["score_hi"])
            + _band_penalty_pp(ecg_mean, score_cfg["score_lo"], score_cfg["score_hi"])
            + 3.0 * max(0.0, gap - score_cfg["gap_hi"]) ** 2
        )
        group_stats[key] = {
            "gap": gap,
            "score": score,
            "accepted": (
                score_cfg["accept_lo"] <= eff_mean <= score_cfg["accept_hi"]
                and score_cfg["accept_lo"] <= ecg_mean <= score_cfg["accept_hi"]
                and gap <= score_cfg["gap_hi"]
            ),
        }

    out_rows: list[dict[str, str]] = []
    for row in parsed:
        stats = group_stats.get((row["operator"], row["candidate"]), {})
        out_rows.append(
            {
                "operator": row["operator"],
                "candidate": row["candidate"],
                "model": row["model"],
                "center": row["center"],
                "clean_auroc": _fmt(row.get("clean_auroc")),
                "corrupted_auroc": _fmt(row.get("corrupted_auroc")),
                "drop_auroc_pp": _fmt(row.get("drop_auroc_pp_float")),
                "clean_auprc": _fmt(row.get("clean_auprc")),
                "corrupted_auprc": _fmt(row.get("corrupted_auprc")),
                "drop_auprc_pp": _fmt(row.get("drop_auprc_pp_float")),
                "inter_model_gap_pp": _fmt(stats.get("gap")),
                "score": _fmt(stats.get("score")),
                "score_profile": score_profile,
                "accepted_on_two_center": "true" if stats.get("accepted") else "false",
                "powerline_branch": powerline_branch_for(
                    row["operator"],
                    row["candidate"],
                    row.get("corruption_input", ""),
                ),
                "severity_profile": str(row.get("severity_profile") or ""),
                "severity_params_name": str(row.get("severity_params_name") or ""),
                "severity_params_file": str(row.get("severity_params_file") or ""),
                "severity_params_file_sha256": str(row.get("severity_params_file_sha256") or ""),
                "required_cache_version": str(row.get("required_cache_version") or ""),
                "pn2021_c_cache_version": str(row.get("pn2021_c_cache_version") or ""),
                "pn2021_mapping_version": str(row.get("pn2021_mapping_version") or ""),
                "pn2021_mapping_hash": str(row.get("pn2021_mapping_hash") or ""),
                "label_mapping_json": str(row.get("label_mapping_json") or ""),
                "pn2021c_input_order_id": str(row.get("pn2021c_input_order_id") or ""),
                "pn2021c_protocol_json": str(row.get("pn2021c_protocol_json") or ""),
                "exclude_ref_ids_by_center_counts_json": str(
                    row.get("exclude_ref_ids_by_center_counts_json") or ""
                ),
                "selected_ref_record_ids_count": str(
                    row.get("selected_ref_record_ids_count") or ""
                ),
                "n_records": str(row.get("n_records") or ""),
                "n_all_zero_labels": str(row.get("n_all_zero_labels") or ""),
                "drop_all_zero_n_records": str(row.get("drop_all_zero_n_records") or ""),
                "drop_all_zero_macro_auroc": _fmt(row.get("drop_all_zero_macro_auroc")),
                "drop_all_zero_macro_auprc": _fmt(row.get("drop_all_zero_macro_auprc")),
                "n_excluded_ref_ids_for_center": str(row.get("n_excluded_ref_ids_for_center") or ""),
                "ref_record_ids_sha256": str(row.get("ref_record_ids_sha256") or ""),
                "cache_source": str(row.get("cache_source") or ""),
                "clean_metric_source": str(row.get("clean_metric_source") or ""),
                "resolved_params_json": str(row.get("resolved_params_json") or ""),
                "status": row.get("status", "ok"),
                "corruption_input": row.get("corruption_input", ""),
                "output_json": row["output_json"],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            writer.writeheader()
            writer.writerows(out_rows)
    return out_rows


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: object) -> float:
    if value is None or str(value).strip() == "":
        return float("inf")
    return float(value)


def select_best_candidates(candidate_scores_csv: Path) -> dict[str, str]:
    """Select one candidate profile per operator from an aggregate score CSV."""

    with candidate_scores_csv.open() as f:
        rows = [row for row in csv.DictReader(f) if row.get("status") == "ok"]

    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        operator = row["operator"]
        candidate = row["candidate"]
        key = (operator, candidate)
        item = grouped.setdefault(
            key,
            {
                "operator": operator,
                "candidate": candidate,
                "score": float("inf"),
                "accepted": False,
            },
        )
        item["score"] = min(float(item["score"]), _as_float(row.get("score")))
        item["accepted"] = bool(item["accepted"]) or _as_bool(row.get("accepted_on_two_center"))

    by_operator: dict[str, list[dict[str, object]]] = {}
    for item in grouped.values():
        by_operator.setdefault(str(item["operator"]), []).append(item)

    selected: dict[str, str] = {}
    for operator, items in by_operator.items():
        best = sorted(
            items,
            key=lambda item: (
                not bool(item["accepted"]),
                float(item["score"]),
                str(item["candidate"]),
            ),
        )[0]
        selected[operator] = str(best["candidate"])
    return selected


def write_frozen_profile(
    *,
    selected: dict[str, str],
    candidate_profile_file: Path,
    output_file: Path,
    profile_name: str,
) -> None:
    """Write a single frozen custom profile from selected candidate profiles."""

    data = yaml.safe_load(candidate_profile_file.read_text())
    profiles = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(profiles, dict):
        raise ValueError(f"{candidate_profile_file} must contain a top-level profiles mapping")

    frozen: dict[str, object] = {}
    input_modes: dict[str, dict[str, str]] = {}
    for operator, candidate in selected.items():
        body = profiles.get(candidate)
        if not isinstance(body, dict) or len(body) != 1:
            raise ValueError(f"profile {candidate!r} must define exactly one corruption operator")
        candidate_operator = next(iter(body))
        if candidate_operator != operator:
            raise ValueError(
                f"selected operator {operator!r} does not match profile {candidate!r} "
                f"operator {candidate_operator!r}"
            )
        frozen[operator] = body[candidate_operator]
        input_modes[operator] = {
            model: corruption_input_for(model, candidate)
            for model in MODELS
        }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        yaml.safe_dump(
            {
                "profiles": {profile_name: frozen},
                "selection": selected,
                "metadata": {
                    "profile_name": profile_name,
                    "public_severity": PUBLIC_SEVERITY,
                    "calibration_centers": list(DEFAULT_CENTERS),
                    "input_modes": input_modes,
                    "candidate_profile_file": str(candidate_profile_file),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _mean(values: Iterable[str]) -> float | None:
    nums: list[float] = []
    for value in values:
        if value is None or str(value).strip() == "":
            continue
        nums.append(float(value))
    if not nums:
        return None
    return sum(nums) / len(nums)


def _selected_score_rows(score_rows: list[dict[str, str]], selection: dict[str, str]) -> list[dict[str, str]]:
    return [
        row
        for row in score_rows
        if row.get("status") == "ok"
        and selection.get(row.get("operator", "")) == row.get("candidate")
    ]


def write_selection_report(
    candidate_scores_csv: Path,
    frozen_profile_file: Path,
    output_md: Path,
) -> None:
    score_rows = _read_csv_rows(candidate_scores_csv)
    frozen = yaml.safe_load(frozen_profile_file.read_text())
    selection = {
        str(operator): str(candidate)
        for operator, candidate in (frozen.get("selection", {}) or {}).items()
    }
    profile_name = str(
        (frozen.get("metadata", {}) or {}).get("profile_name")
        or next(iter((frozen.get("profiles", {}) or {}).keys()), "dual_model_10to15pp_v1")
    )
    selected_rows = _selected_score_rows(score_rows, selection)

    lines = [
        f"# {profile_name} Selection Report",
        "",
        f"Candidate scores: `{candidate_scores_csv}`",
        f"Frozen profile: `{frozen_profile_file}`",
        "",
        "| Operator | Selected candidate | Model | Centers | Mean AUROC drop pp | Mean AUPRC drop pp | Gap pp | Accepted | Branch | Params |",
        "|---|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for operator in sorted(selection):
        candidate = selection[operator]
        rows = [row for row in selected_rows if row["operator"] == operator]
        by_model = sorted({row.get("model", "") for row in rows if row.get("model")})
        for model in by_model or [""]:
            model_rows = [row for row in rows if row.get("model", "") == model]
            centers = ",".join(sorted({row.get("center", "") for row in model_rows if row.get("center")}))
            auroc = _mean(row.get("drop_auroc_pp", "") for row in model_rows)
            auprc = _mean(row.get("drop_auprc_pp", "") for row in model_rows)
            gap = _mean(row.get("inter_model_gap_pp", "") for row in model_rows)
            accepted = "true" if any(row.get("accepted_on_two_center") == "true" for row in model_rows) else "false"
            branch = next((row.get("powerline_branch", "") for row in model_rows if row.get("powerline_branch")), "")
            params = next((row.get("resolved_params_json", "") for row in model_rows if row.get("resolved_params_json")), "")
            lines.append(
                "| "
                + " | ".join(
                    [
                        operator,
                        candidate,
                        model,
                        centers,
                        _fmt(auroc),
                        _fmt(auprc),
                        _fmt(gap),
                        accepted,
                        branch,
                        params,
                    ]
                )
                + " |"
            )
        if not rows:
            lines.append(
                f"| {operator} | {candidate} |  |  |  |  |  | false |  | no score rows found |"
            )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _four_center_status(
    eff_mean: float | None,
    ecg_mean: float | None,
    gap: float | None,
    *,
    score_profile: str = DEFAULT_SCORE_PROFILE,
) -> str:
    values = [v for v in (eff_mean, ecg_mean) if v is not None]
    if len(values) < 2:
        return "missing"
    score_cfg = _score_profile_config(score_profile)
    if any(v < score_cfg["accept_lo"] for v in values):
        return "miss_low"
    if any(v > score_cfg["accept_hi"] for v in values):
        return "miss_high"
    if gap is not None and gap > score_cfg["gap_hi"]:
        return "gap_high"
    if score_profile == "realistic_hospital":
        if all(score_cfg["score_lo"] <= v <= score_cfg["score_hi"] for v in values):
            return "realistic_main"
        return "realistic_strong"
    return "in_band"


def write_confirmation_summary(
    candidate_scores_csv: Path,
    output_csv: Path,
    output_md: Path,
    *,
    score_profile: str = DEFAULT_SCORE_PROFILE,
) -> list[dict[str, str]]:
    _score_profile_config(score_profile)
    rows = [row for row in _read_csv_rows(candidate_scores_csv) if row.get("status") == "ok"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["operator"], row["candidate"]), []).append(row)

    out_rows: list[dict[str, str]] = []
    for (operator, candidate), items in sorted(grouped.items()):
        eff_auroc = _mean(row.get("drop_auroc_pp", "") for row in items if row.get("model") == "effnet")
        ecg_auroc = _mean(row.get("drop_auroc_pp", "") for row in items if row.get("model") == "ecgfounder")
        eff_auprc = _mean(row.get("drop_auprc_pp", "") for row in items if row.get("model") == "effnet")
        ecg_auprc = _mean(row.get("drop_auprc_pp", "") for row in items if row.get("model") == "ecgfounder")
        gap = abs(eff_auroc - ecg_auroc) if eff_auroc is not None and ecg_auroc is not None else None
        centers = ",".join(sorted({row.get("center", "") for row in items if row.get("center")}))
        corruption_inputs = ",".join(
            sorted({row.get("corruption_input", "") for row in items if row.get("corruption_input")})
        )
        powerline_branches = ",".join(
            sorted({row.get("powerline_branch", "") for row in items if row.get("powerline_branch")})
        )
        out_rows.append(
            {
                "operator": operator,
                "candidate": candidate,
                "centers": centers,
                "corruption_inputs": corruption_inputs,
                "powerline_branch": powerline_branches,
                "effnet_mean_drop_auroc_pp": _fmt(eff_auroc),
                "ecgfounder_mean_drop_auroc_pp": _fmt(ecg_auroc),
                "effnet_mean_drop_auprc_pp": _fmt(eff_auprc),
                "ecgfounder_mean_drop_auprc_pp": _fmt(ecg_auprc),
                "inter_model_gap_auroc_pp": _fmt(gap),
                "four_center_status": _four_center_status(
                    eff_auroc,
                    ecg_auroc,
                    gap,
                    score_profile=score_profile,
                ),
                "score_profile": score_profile,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            writer.writeheader()
            writer.writerows(out_rows)

    lines = [
        f"# {score_profile} Four-Center Confirmation",
        "",
        f"Source scores: `{candidate_scores_csv}`",
        f"CSV: `{output_csv}`",
        f"Score profile: `{score_profile}`",
        "",
        "| Operator | Candidate | Input modes | Powerline branch | EffNet AUROC drop pp | ECGFounder AUROC drop pp | Gap pp | Status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in out_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["operator"],
                    row["candidate"],
                    row["corruption_inputs"],
                    row["powerline_branch"],
                    row["effnet_mean_drop_auroc_pp"],
                    row["ecgfounder_mean_drop_auroc_pp"],
                    row["inter_model_gap_auroc_pp"],
                    row["four_center_status"],
                ]
            )
            + " |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_rows


def cmd_plan(args: argparse.Namespace) -> None:
    profiles = load_candidate_profiles(args.profile_file)
    stage_dir = args.output_root / args.stage
    rows = build_manifest_rows(
        stage_dir=stage_dir,
        centers=args.centers,
        candidate_profiles=profiles,
        models=args.models,
    )
    manifest = stage_dir / "manifest.csv"
    commands = stage_dir / "commands.sh"
    write_manifest(rows, manifest)
    command_paths = write_command_scripts(
        rows,
        commands,
        profile_file=args.profile_file,
        limit=args.limit,
        device=args.device,
        num_shards=args.num_shards,
    )
    print(f"[plan] rows={len(rows)} manifest={manifest}")
    for command_path in command_paths:
        print(f"[plan] commands={command_path}")


def cmd_plan_frozen(args: argparse.Namespace) -> None:
    stage_dir = args.output_root / args.stage
    rows = build_frozen_manifest_rows(
        stage_dir=stage_dir,
        profile_file=args.profile_file,
        profile_name=args.profile_name,
        centers=args.centers,
        models=args.models,
    )
    manifest = stage_dir / "manifest.csv"
    commands = stage_dir / "commands.sh"
    write_manifest(rows, manifest)
    command_paths = write_command_scripts(
        rows,
        commands,
        profile_file=args.profile_file,
        limit=args.limit,
        device=args.device,
        num_shards=args.num_shards,
    )
    print(f"[plan-frozen] rows={len(rows)} manifest={manifest}")
    for command_path in command_paths:
        print(f"[plan-frozen] commands={command_path}")


def cmd_aggregate(args: argparse.Namespace) -> None:
    rows = aggregate_manifest(
        args.manifest,
        args.output_csv,
        score_profile=args.score_profile,
    )
    ok = sum(1 for row in rows if row["status"] == "ok")
    missing = sum(1 for row in rows if row["status"] == "missing")
    print(f"[aggregate] rows={len(rows)} ok={ok} missing={missing} output={args.output_csv}")


def cmd_select(args: argparse.Namespace) -> None:
    selected = select_best_candidates(args.candidate_scores)
    write_frozen_profile(
        selected=selected,
        candidate_profile_file=args.profile_file,
        output_file=args.output_profile,
        profile_name=args.profile_name,
    )
    for operator, candidate in selected.items():
        print(f"[select] {operator}={candidate}")
    print(f"[select] output={args.output_profile}")


def cmd_selection_report(args: argparse.Namespace) -> None:
    write_selection_report(args.candidate_scores, args.frozen_profile, args.output_md)
    print(f"[selection-report] output={args.output_md}")


def cmd_confirmation_summary(args: argparse.Namespace) -> None:
    rows = write_confirmation_summary(
        args.candidate_scores,
        args.output_csv,
        args.output_md,
        score_profile=args.score_profile,
    )
    print(
        f"[confirmation-summary] rows={len(rows)} csv={args.output_csv} md={args.output_md}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--stage", default="smoke")
    p_plan.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_plan.add_argument("--profile_file", type=Path, default=DEFAULT_PROFILE_FILE)
    p_plan.add_argument("--centers", nargs="+", default=list(DEFAULT_CENTERS))
    p_plan.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    limit_group = p_plan.add_mutually_exclusive_group()
    limit_group.add_argument("--limit", type=int, default=2000)
    limit_group.add_argument("--no_limit", action="store_const", const=None, dest="limit")
    p_plan.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p_plan.add_argument("--num_shards", type=int, default=1)
    p_plan.set_defaults(func=cmd_plan)

    p_plan_frozen = sub.add_parser("plan-frozen")
    p_plan_frozen.add_argument("--stage", default="four_center_confirmation")
    p_plan_frozen.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_plan_frozen.add_argument("--profile_file", type=Path, required=True)
    p_plan_frozen.add_argument("--profile_name", default="dual_model_10to15pp_v1")
    p_plan_frozen.add_argument("--centers", nargs="+", default=list(FOUR_CENTERS))
    p_plan_frozen.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    frozen_limit_group = p_plan_frozen.add_mutually_exclusive_group()
    frozen_limit_group.add_argument("--limit", type=int, default=None)
    frozen_limit_group.add_argument("--no_limit", action="store_const", const=None, dest="limit")
    p_plan_frozen.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p_plan_frozen.add_argument("--num_shards", type=int, default=1)
    p_plan_frozen.set_defaults(func=cmd_plan_frozen)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--manifest", type=Path, required=True)
    p_agg.add_argument("--output_csv", type=Path, required=True)
    p_agg.add_argument(
        "--score_profile",
        choices=sorted(SCORE_PROFILES),
        default=DEFAULT_SCORE_PROFILE,
    )
    p_agg.set_defaults(func=cmd_aggregate)

    p_select = sub.add_parser("select")
    p_select.add_argument("--candidate_scores", type=Path, required=True)
    p_select.add_argument("--profile_file", type=Path, default=DEFAULT_PROFILE_FILE)
    p_select.add_argument("--output_profile", type=Path, required=True)
    p_select.add_argument("--profile_name", default="dual_model_10to15pp_v1")
    p_select.set_defaults(func=cmd_select)

    p_sel_report = sub.add_parser("selection-report")
    p_sel_report.add_argument("--candidate_scores", type=Path, required=True)
    p_sel_report.add_argument("--frozen_profile", type=Path, required=True)
    p_sel_report.add_argument("--output_md", type=Path, required=True)
    p_sel_report.set_defaults(func=cmd_selection_report)

    p_conf = sub.add_parser("confirmation-summary")
    p_conf.add_argument("--candidate_scores", type=Path, required=True)
    p_conf.add_argument("--output_csv", type=Path, required=True)
    p_conf.add_argument("--output_md", type=Path, required=True)
    p_conf.add_argument(
        "--score_profile",
        choices=sorted(SCORE_PROFILES),
        default=DEFAULT_SCORE_PROFILE,
    )
    p_conf.set_defaults(func=cmd_confirmation_summary)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
