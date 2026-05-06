#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from apps.streamlit_ecg_demo.services.preprocessing import CLASS_NAMES, to_signal_ct
from apps.streamlit_ecg_demo.services.quality_gate import run_quality_gate
from util.ecg_viz import plot_ecg_ecgtwin_gallery


DEFAULT_SYNTH_NPZ = (
    "/root/autodl-tmp/graduate_project/"
    "self_distill_v2_filtered_v46_ptbxl_contrast_seed42/"
    "synth_v2_filtered_top4000_gamma03.npz"
)
DEFAULT_OUT_DIR = "/root/autodl-tmp/final_round_ablation_20260504/five_class_visuals"


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _load_metadata(data: np.lib.npyio.NpzFile) -> dict:
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


def _candidate_rank(
    gate: dict,
    target_conf: float,
    target_teacher_prob: float,
    *,
    class_name: str,
    prefer_pass: bool,
) -> tuple:
    status = str(gate.get("status", "fail"))
    status_rank = {"pass": 0, "warning": 1, "fail": 2}.get(status, 2)
    warnings = gate.get("warnings", []) or []
    einthoven = gate.get("einthoven_residual")
    avr = gate.get("avR_residual")
    einthoven = float(einthoven) if einthoven is not None and np.isfinite(einthoven) else 999.0
    avr = float(avr) if avr is not None and np.isfinite(avr) else 999.0
    hr = gate.get("hr_estimate_bpm")
    hr = float(hr) if hr is not None and np.isfinite(hr) else 999.0
    if class_name == "NORM":
        if 60.0 <= hr <= 100.0:
            hr_rank = 0
        elif 50.0 <= hr <= 110.0:
            hr_rank = 1
        elif 50.0 <= hr <= 130.0:
            hr_rank = 2
        else:
            hr_rank = 3
    else:
        hr_rank = 0
    if prefer_pass:
        return (status_rank, hr_rank, len(warnings), einthoven + avr, -target_conf, -target_teacher_prob)
    return (hr_rank, -target_conf, status_rank, len(warnings), einthoven + avr, -target_teacher_prob)


def select_examples(
    data: np.lib.npyio.NpzFile,
    class_names: list[str],
    max_scan_per_class: int,
    prefer_pass: bool,
    manual_indices: dict[str, int] | None = None,
) -> list[dict]:
    manual_indices = manual_indices or {}
    signals = data["signals"]
    labels = data["labels"]
    teacher_probs = data["teacher_probs"] if "teacher_probs" in data.files else None
    target_conf = data["target_conf"] if "target_conf" in data.files else None
    target_class = data["target_class"] if "target_class" in data.files else None

    selected: list[dict] = []
    for class_idx, class_name in enumerate(class_names):
        if class_name in manual_indices:
            candidates = np.asarray([manual_indices[class_name]], dtype=np.int64)
        elif target_class is not None:
            candidates = np.where(np.asarray(target_class) == class_idx)[0]
        else:
            candidates = np.where(labels[:, class_idx] > 0.5)[0]
        if len(candidates) == 0:
            raise RuntimeError(f"No candidate found for class {class_name}")

        if target_conf is not None:
            candidates = candidates[np.argsort(np.asarray(target_conf)[candidates])[::-1]]
        candidates = candidates[: max(1, int(max_scan_per_class))]

        best = None
        best_rank = None
        for idx in candidates:
            signal_ct = to_signal_ct(signals[int(idx)])
            gate = run_quality_gate(signal_ct)
            conf = float(target_conf[int(idx)]) if target_conf is not None else float(labels[int(idx), class_idx])
            teacher = (
                float(teacher_probs[int(idx), class_idx])
                if teacher_probs is not None
                else conf
            )
            rank = _candidate_rank(gate, conf, teacher, class_name=class_name, prefer_pass=prefer_pass)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = {
                    "class_name": class_name,
                    "class_index": class_idx,
                    "source_index": int(idx),
                    "label": labels[int(idx)].astype(float).tolist(),
                    "target_conf": conf,
                    "teacher_target_prob": teacher,
                    "quality_gate": gate,
                    "signal": signal_ct,
                }

        if best is None:
            raise RuntimeError(f"Could not select candidate for class {class_name}")
        selected.append(best)
    return selected


def write_report(out_dir: Path, source_npz: Path, metadata: dict, selected: list[dict]) -> None:
    records = []
    for item in selected:
        gate = item["quality_gate"]
        records.append({
            "class_name": item["class_name"],
            "source_npz": str(source_npz),
            "source_index": item["source_index"],
            "target_conf": item["target_conf"],
            "teacher_target_prob": item["teacher_target_prob"],
            "quality_status": gate.get("status"),
            "warnings": gate.get("warnings", []),
            "einthoven_residual": gate.get("einthoven_residual"),
            "avR_residual": gate.get("avR_residual"),
            "hr_estimate_bpm": gate.get("hr_estimate_bpm"),
            "png": str(item["png"]),
        })
    payload = {
        "source_npz": str(source_npz),
        "metadata": metadata,
        "selected": records,
        "note": (
            "These figures are visualization examples. HYP/CD are retained as "
            "cautious qualitative examples if their digital gates contain warnings."
        ),
    }
    (out_dir / "selected_examples.json").write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Five-Class Synthetic ECG Visualization Examples",
        "",
        f"Source NPZ: `{source_npz}`",
        "",
        "| class | source index | quality | target conf | teacher target prob | figure |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in records:
        lines.append(
            "| {class_name} | {source_index} | {quality_status} | "
            "{target_conf:.4f} | {teacher_target_prob:.4f} | `{png}` |".format(**row)
        )
    lines.extend([
        "",
        "HYP/CD figures should be described as qualitative visualization examples "
        "unless their downstream digital criteria are separately validated.",
        "",
    ])
    (out_dir / "selected_examples.md").write_text("\n".join(lines), encoding="utf-8")


def write_selected_npz(
    out_dir: Path,
    source_npz: Path,
    data: np.lib.npyio.NpzFile,
    selected: list[dict],
) -> Path:
    source_indices = np.asarray([item["source_index"] for item in selected], dtype=np.int64)
    signals = np.asarray(data["signals"][source_indices], dtype=np.float32)
    labels = np.asarray(data["labels"][source_indices], dtype=np.float32)
    class_names = np.asarray([item["class_name"] for item in selected])
    metadata = {
        "source_npz": str(source_npz),
        "class_order": [item["class_name"] for item in selected],
        "source_indices": source_indices.tolist(),
        "note": "Selected generated ECG vectors for thesis figures; signals shape is (5,1000,12) in PTB-XL lead order.",
    }
    payload = {
        "signals": signals,
        "labels": labels,
        "class_names": class_names,
        "target_class": np.asarray([item["class_index"] for item in selected], dtype=np.int64),
        "target_conf": np.asarray([item["target_conf"] for item in selected], dtype=np.float32),
        "source_indices": source_indices,
        "metadata_json": np.asarray(json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2)),
    }
    if "teacher_probs" in data.files:
        payload["teacher_probs"] = np.asarray(data["teacher_probs"][source_indices], dtype=np.float32)
    if "soft_labels" in data.files:
        payload["soft_labels"] = np.asarray(data["soft_labels"][source_indices], dtype=np.float32)
    if "source_indices" in data.files:
        payload["original_source_indices"] = np.asarray(data["source_indices"][source_indices], dtype=np.int64)
    out_path = out_dir / "thesis_selected_samples.npz"
    np.savez_compressed(out_path, **payload)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth_npz", default=DEFAULT_SYNTH_NPZ)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--max_scan_per_class", type=int, default=800)
    ap.add_argument("--scale_mode", default="global_p95",
                    choices=["global_p95", "global_peak", "per_lead_peak", "none"])
    ap.add_argument("--row_gap", type=float, default=7.0)
    ap.add_argument("--target_peak", type=float, default=2.0)
    ap.add_argument("--prefer_confidence", action="store_true",
                    help="Rank by target confidence before signal-gate cleanliness.")
    ap.add_argument("--save_selected_npz", action="store_true",
                    help="Save the selected generated ECG vectors as thesis_selected_samples.npz.")
    ap.add_argument(
        "--manual_indices",
        default=None,
        help="Optional class-specific overrides, for example 'NORM=3042,HYP=1442'.",
    )
    args = ap.parse_args()

    source_npz = Path(args.synth_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(source_npz, allow_pickle=True)
    if "signals" not in data.files or "labels" not in data.files:
        raise ValueError(f"{source_npz} must contain signals and labels")
    class_names = [str(x) for x in data["class_names"]] if "class_names" in data.files else CLASS_NAMES
    if class_names != CLASS_NAMES:
        raise ValueError(f"Unexpected class order: {class_names}; expected {CLASS_NAMES}")

    metadata = _load_metadata(data)
    selected = select_examples(
        data,
        class_names,
        max_scan_per_class=args.max_scan_per_class,
        prefer_pass=not args.prefer_confidence,
        manual_indices=_parse_manual_indices(args.manual_indices),
    )

    for item in selected:
        class_name = item["class_name"]
        png = out_dir / f"synthetic_{class_name}_12lead.png"
        title = f"Synthetic {class_name} ECG example (v46 center-token)"
        plot_ecg_ecgtwin_gallery(
            item["signal"],
            png,
            sample_rate=100.0,
            lead_order="ptbxl",
            title=title,
            row_gap=args.row_gap,
            target_peak=args.target_peak,
            scale_mode=args.scale_mode,
            figsize=(18.0, 14.5),
            linewidth=0.78,
            show_grid=True,
        )
        item["png"] = png

    write_report(out_dir, source_npz, metadata, selected)
    selected_npz = write_selected_npz(out_dir, source_npz, data, selected) if args.save_selected_npz else None
    print(json.dumps({
        "out_dir": str(out_dir),
        "figures": [str(item["png"]) for item in selected],
        "selected_npz": str(selected_npz) if selected_npz is not None else None,
    }, indent=2))


if __name__ == "__main__":
    main()
