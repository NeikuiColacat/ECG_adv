#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from apps.streamlit_ecg_demo.services.preprocessing import CLASS_NAMES, to_signal_ct
from apps.streamlit_ecg_demo.services.quality_gate import run_quality_gate
from util.ecg_digital_features import evaluate_super5, extract_digital_features
from util.ecg_viz import plot_ecg_ecgtwin_gallery


DEFAULT_SYNTH_NPZ = (
    "/root/autodl-tmp/graduate_project/"
    "self_distill_v2_filtered_v46_ptbxl_contrast_seed42/"
    "synth_v2_filtered_top4000_gamma03.npz"
)
DEFAULT_OUT_DIR = "/root/autodl-tmp/final_round_ablation_20260504/thesis_selected_ecg_examples"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _load_source_metadata(data: np.lib.npyio.NpzFile) -> dict:
    if "metadata_json" not in data.files:
        return {}
    raw = str(data["metadata_json"].item())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_metadata_json": raw}


def _parse_manual_indices(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    out: dict[str, int] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        out[key.strip().upper()] = int(value)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def _base_score(gate: dict, conf: float, teacher: float, features: dict) -> float:
    status = str(gate.get("status", "fail"))
    score = {"pass": 80.0, "warning": 25.0, "fail": -120.0}.get(status, -120.0)
    warnings = gate.get("warnings", []) or []
    score -= 7.5 * len(warnings)
    score += 22.0 * conf + 10.0 * teacher

    einthoven = _safe_float(gate.get("einthoven_residual"), 1.0)
    avr = _safe_float(gate.get("avR_residual"), 1.0)
    score -= 80.0 * min(einthoven + avr, 1.0)

    rr_cv = _safe_float(features.get("rr_irregularity_cv"), 0.5)
    score -= 10.0 * min(rr_cv, 0.5)
    return score


def _score_norm(criteria: dict, features: dict) -> float:
    score = 0.0
    norm = {k.split(".", 1)[1]: v for k, v in criteria.items() if k.startswith("NORM.")}
    if norm.get("NORM_strict_pass"):
        score += 120.0
    elif norm.get("NORM_ratevariant_pass"):
        score += 70.0
    elif norm.get("NORM_any_pass"):
        score += 45.0

    hr = _safe_float(features.get("hr_bpm"), 0.0)
    if 60 <= hr <= 100:
        score += 60.0
    elif 50 <= hr <= 110:
        score += 35.0
    elif 50 <= hr <= 130:
        score += 10.0
    else:
        score -= 70.0

    if norm.get("N3_p_wave_present"):
        score += 25.0
    if norm.get("N5_qrs_70_110"):
        score += 20.0
    if norm.get("N6_st_isoelectric"):
        score += 18.0
    if norm.get("N7_t_positive_2of3"):
        score += 18.0

    # A normal example for the thesis should not strongly satisfy abnormal rules.
    for key in ("MI.MI_pass", "STTC.STTC_pass", "HYP.HYP_pass", "CD.CD_pass"):
        if criteria.get(key):
            score -= 35.0
    return score


def _score_mi(criteria: dict) -> float:
    score = 0.0
    if criteria.get("MI.MI_pass"):
        score += 100.0
    if criteria.get("MI.MI2_q_dur_and_MI3_q_depth"):
        score += 45.0
    if criteria.get("MI.MI1_ste_2contig"):
        score += 30.0
    score += 6.0 * len(criteria.get("MI.MI_pathological_q_leads", []) or [])
    return score


def _score_sttc(criteria: dict) -> float:
    score = 0.0
    if criteria.get("STTC.STTC_pass"):
        score += 100.0
    if criteria.get("STTC.ST1_depression"):
        score += 45.0
    if criteria.get("STTC.ST2_t_inversion"):
        score += 45.0
    score += 7.0 * len(criteria.get("STTC.ST1_pairs", []) or [])
    score += 7.0 * len(criteria.get("STTC.ST2_pairs", []) or [])
    return score


def _score_hyp(criteria: dict) -> float:
    score = 0.0
    if criteria.get("HYP.HYP_pass"):
        score += 100.0
    if criteria.get("HYP.H1_sokolow"):
        score += 50.0
    if criteria.get("HYP.H2_cornell_male") or criteria.get("HYP.H2w_cornell_female"):
        score += 45.0
    score += min(_safe_float(criteria.get("HYP.sokolow_mv")), 5.0) * 8.0
    score += min(_safe_float(criteria.get("HYP.cornell_mv")), 4.0) * 8.0
    return score


def _score_cd(criteria: dict, features: dict) -> float:
    score = 0.0
    if criteria.get("CD.CD_pass"):
        score += 110.0
    if criteria.get("CD.LBBB_pass"):
        score += 50.0
    if criteria.get("CD.RBBB_pass"):
        score += 50.0
    if criteria.get("CD.AVB_pass"):
        score += 40.0
    qrs = _safe_float(features.get("qrs_duration_ms_broadest"), 0.0)
    if qrs >= 120:
        score += 35.0
    elif qrs >= 110:
        score += 15.0
    score += min(max(qrs - 80.0, 0.0), 80.0) * 0.35
    return score


def _class_score(class_name: str, criteria: dict, features: dict) -> float:
    if class_name == "NORM":
        return _score_norm(criteria, features)
    if class_name == "MI":
        return _score_mi(criteria)
    if class_name == "STTC":
        return _score_sttc(criteria)
    if class_name == "HYP":
        return _score_hyp(criteria)
    if class_name == "CD":
        return _score_cd(criteria, features)
    return 0.0


def _candidate_metrics(
    signal_ct: np.ndarray,
    class_name: str,
    conf: float,
    teacher: float,
) -> dict:
    gate = run_quality_gate(signal_ct, sample_rate=100.0)
    features = extract_digital_features(signal_ct, fs=100.0, lead_order="ptbxl")
    criteria = evaluate_super5(features, class_name)
    score = _base_score(gate, conf, teacher, features) + _class_score(class_name, criteria, features)
    return {
        "score": float(score),
        "quality_gate": gate,
        "digital_features": features,
        "criteria": criteria,
    }


def _candidate_indices_for_class(
    labels: np.ndarray,
    target_class: np.ndarray | None,
    target_conf: np.ndarray | None,
    teacher_probs: np.ndarray | None,
    class_idx: int,
    limit: int,
) -> np.ndarray:
    if target_class is not None:
        idx = np.where(np.asarray(target_class) == class_idx)[0]
    else:
        idx = np.where(labels[:, class_idx] > 0.5)[0]
    if target_conf is not None:
        conf = np.asarray(target_conf)[idx]
    elif teacher_probs is not None:
        conf = np.asarray(teacher_probs)[idx, class_idx]
    else:
        conf = labels[idx, class_idx]
    order = np.argsort(conf)[::-1]
    return idx[order[: max(1, limit)]]


def rank_candidates(
    data: np.lib.npyio.NpzFile,
    class_names: list[str],
    scan_per_class: int,
    manual_indices: dict[str, int],
) -> dict[str, list[dict]]:
    signals = data["signals"]
    labels = data["labels"]
    teacher_probs = data["teacher_probs"] if "teacher_probs" in data.files else None
    target_conf = data["target_conf"] if "target_conf" in data.files else None
    source_indices = data["source_indices"] if "source_indices" in data.files else None
    target_class = data["target_class"] if "target_class" in data.files else None

    rankings: dict[str, list[dict]] = {}
    for class_idx, class_name in enumerate(class_names):
        if class_name in manual_indices:
            candidate_indices = np.asarray([manual_indices[class_name]], dtype=int)
        else:
            candidate_indices = _candidate_indices_for_class(
                labels, target_class, target_conf, teacher_probs, class_idx, scan_per_class
            )

        rows: list[dict] = []
        for idx in candidate_indices:
            idx = int(idx)
            signal_ct = to_signal_ct(signals[idx])
            conf = float(target_conf[idx]) if target_conf is not None else float(labels[idx, class_idx])
            teacher = float(teacher_probs[idx, class_idx]) if teacher_probs is not None else conf
            metrics = _candidate_metrics(signal_ct, class_name, conf, teacher)
            rows.append({
                "class_name": class_name,
                "class_index": class_idx,
                "source_index": idx,
                "original_source_index": int(source_indices[idx]) if source_indices is not None else idx,
                "label": labels[idx].astype(float).tolist(),
                "target_conf": conf,
                "teacher_target_prob": teacher,
                **metrics,
            })
        rows.sort(key=lambda x: x["score"], reverse=True)
        rankings[class_name] = rows
    return rankings


def _record_for_json(row: dict, png: Path | None = None) -> dict:
    keys = [
        "class_name", "class_index", "source_index", "original_source_index",
        "score", "label", "target_conf", "teacher_target_prob",
        "quality_gate", "criteria",
    ]
    out = {k: row[k] for k in keys if k in row}
    features = row.get("digital_features", {})
    out["digital_summary"] = {
        "hr_bpm": features.get("hr_bpm"),
        "rr_irregularity_cv": features.get("rr_irregularity_cv"),
        "p_wave_present": features.get("p_wave_present"),
        "pr_interval_ms": features.get("pr_interval_ms"),
        "qrs_duration_ms_avg": features.get("qrs_duration_ms_avg"),
        "qrs_duration_ms_broadest": features.get("qrs_duration_ms_broadest"),
    }
    if png is not None:
        out["png"] = str(png)
    return out


def _write_png(signal: np.ndarray, png: Path, class_name: str, title_suffix: str) -> None:
    plot_ecg_ecgtwin_gallery(
        signal,
        png,
        sample_rate=100.0,
        lead_order="ptbxl",
        title=f"Synthetic {class_name} ECG example ({title_suffix})",
        row_gap=8.2,
        target_peak=1.45,
        scale_mode="global_p95",
        figsize=(18.0, 14.5),
        linewidth=0.78,
        show_grid=True,
    )


def write_outputs(
    data: np.lib.npyio.NpzFile,
    source_npz: Path,
    out_dir: Path,
    class_names: list[str],
    rankings: dict[str, list[dict]],
    candidate_pngs: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = out_dir / "candidate_pngs"
    selected_rows = [rankings[name][0] for name in class_names]
    signals_arr = data["signals"]
    labels_arr = data["labels"]
    teacher_probs_arr = data["teacher_probs"] if "teacher_probs" in data.files else None
    soft_labels_arr = data["soft_labels"] if "soft_labels" in data.files else None

    final_records = []
    selected_signals = []
    selected_labels = []
    selected_teacher_probs = []
    selected_soft_labels = []
    selected_target_conf = []
    selected_source_indices = []
    selected_original_source_indices = []

    for row in selected_rows:
        class_name = row["class_name"]
        idx = int(row["source_index"])
        signal_ct = to_signal_ct(signals_arr[idx])
        png = out_dir / f"thesis_{class_name}_12lead.png"
        _write_png(signal_ct, png, class_name, "selected v46 center-token")
        final_records.append(_record_for_json(row, png))

        selected_signals.append(signals_arr[idx])
        selected_labels.append(labels_arr[idx])
        selected_target_conf.append(row["target_conf"])
        selected_source_indices.append(idx)
        selected_original_source_indices.append(row["original_source_index"])
        if teacher_probs_arr is not None:
            selected_teacher_probs.append(teacher_probs_arr[idx])
        if soft_labels_arr is not None:
            selected_soft_labels.append(soft_labels_arr[idx])

    for class_name, rows in rankings.items():
        for rank, row in enumerate(rows[:candidate_pngs], start=1):
            idx = int(row["source_index"])
            signal_ct = to_signal_ct(signals_arr[idx])
            png = candidates_dir / class_name / f"rank{rank:02d}_idx{idx:04d}_score{row['score']:.1f}.png"
            _write_png(signal_ct, png, class_name, f"candidate rank {rank}")
            row["candidate_png"] = str(png)

    source_metadata = _load_source_metadata(data)
    payload = {
        "source_npz": str(source_npz),
        "source_metadata": source_metadata,
        "class_order": class_names,
        "selected": final_records,
        "ranking_top": {
            cls: [_record_for_json(row, Path(row["candidate_png"])) for row in rows[:candidate_pngs]]
            for cls, rows in rankings.items()
        },
        "note": (
            "These examples are curated thesis visualization artifacts. The saved "
            "NPZ contains raw generated ECG vectors in PTB-XL lead order, shape "
            "(5,1000,12). Clinical validity should still be described cautiously."
        ),
    }
    metadata_json = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2)
    (out_dir / "selected_examples.json").write_text(metadata_json, encoding="utf-8")

    lines = [
        "# Thesis-Selected Synthetic ECG Examples",
        "",
        f"Source NPZ: `{source_npz}`",
        "",
        "| class | source index | quality | score | target conf | HR | target digital pass | figure |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for row in final_records:
        gate = row["quality_gate"]
        digital = row["digital_summary"]
        lines.append(
            "| {class_name} | {source_index} | {quality} | {score:.1f} | "
            "{target_conf:.4f} | {hr:.1f} | {target_pass} | `{png}` |".format(
                class_name=row["class_name"],
                source_index=row["source_index"],
                quality=gate.get("status"),
                score=row["score"],
                target_conf=row["target_conf"],
                hr=_safe_float(digital.get("hr_bpm"), float("nan")),
                target_pass=row["criteria"].get("target_pass"),
                png=row["png"],
            )
        )
    lines.extend([
        "",
        "Vector sample NPZ:",
        f"`{out_dir / 'thesis_selected_samples.npz'}`",
        "",
        "Saved arrays: `signals` (5,1000,12), `labels` (5,5), "
        "`class_names`, `target_class`, `source_indices`, and metadata JSON.",
        "",
    ])
    (out_dir / "selected_examples.md").write_text("\n".join(lines), encoding="utf-8")

    npz_kwargs: dict[str, Any] = {
        "signals": np.stack(selected_signals).astype(np.float32),
        "labels": np.stack(selected_labels).astype(np.float32),
        "class_names": np.asarray(class_names),
        "target_class": np.asarray([row["class_index"] for row in selected_rows], dtype=np.int64),
        "target_conf": np.asarray(selected_target_conf, dtype=np.float32),
        "source_indices": np.asarray(selected_source_indices, dtype=np.int64),
        "original_source_indices": np.asarray(selected_original_source_indices, dtype=np.int64),
        "metadata_json": np.asarray(metadata_json),
    }
    if selected_teacher_probs:
        npz_kwargs["teacher_probs"] = np.stack(selected_teacher_probs).astype(np.float32)
    if selected_soft_labels:
        npz_kwargs["soft_labels"] = np.stack(selected_soft_labels).astype(np.float32)
    np.savez_compressed(out_dir / "thesis_selected_samples.npz", **npz_kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_npz", default=DEFAULT_SYNTH_NPZ)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--scan_per_class", type=int, default=800)
    ap.add_argument("--candidate_pngs", type=int, default=5)
    ap.add_argument(
        "--manual_indices",
        default=None,
        help="Optional overrides like 'CD=495,HYP=1442,MI=1890,NORM=2631,STTC=3763'.",
    )
    args = ap.parse_args()

    source_npz = Path(args.synth_npz)
    out_dir = Path(args.out_dir)
    data = np.load(source_npz, allow_pickle=True)
    if "signals" not in data.files or "labels" not in data.files:
        raise ValueError(f"{source_npz} must contain signals and labels")

    class_names = [str(x) for x in data["class_names"]] if "class_names" in data.files else CLASS_NAMES
    if class_names != CLASS_NAMES:
        raise ValueError(f"Unexpected class order: {class_names}; expected {CLASS_NAMES}")

    rankings = rank_candidates(
        data,
        class_names,
        scan_per_class=args.scan_per_class,
        manual_indices=_parse_manual_indices(args.manual_indices),
    )
    write_outputs(data, source_npz, out_dir, class_names, rankings, args.candidate_pngs)
    print(json.dumps({
        "out_dir": str(out_dir),
        "selected_npz": str(out_dir / "thesis_selected_samples.npz"),
        "selected_figures": [str(out_dir / f"thesis_{name}_12lead.png") for name in class_names],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
