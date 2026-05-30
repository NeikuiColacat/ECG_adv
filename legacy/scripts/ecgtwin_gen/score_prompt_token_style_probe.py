"""Score prompt-token synthetic ECGs with the real-only center style classifier.

This is a fast first-pass probe for center-token effectiveness. The classifier
is trained on real PN2021 ECG only; synthetic ECGs are query/eval samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model" / "DeepECG" / "notebooks"))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402


PN2021_CENTERS = [
    "chapman_shaoxing",
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "ningbo",
    "ptb",
    "st_petersburg_incart",
]


def load_center_names(ckpt_dir: str) -> List[str]:
    result_path = Path(ckpt_dir) / "train_result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
            names = result.get("center_names")
            if names:
                return [str(x) for x in names]
        except Exception:
            pass
    return list(PN2021_CENTERS)


def load_classifier(ckpt_dir: str, device: str) -> tuple[torch.nn.Module, List[str]]:
    center_names = load_center_names(ckpt_dir)
    model = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=len(center_names),
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    )
    sd = torch.load(Path(ckpt_dir) / "best_model.pt", map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, center_names


@torch.no_grad()
def classify(model: torch.nn.Module, signals: np.ndarray, device: str, batch_size: int, crop_len: int) -> np.ndarray:
    start = max(0, (signals.shape[-1] - crop_len) // 2)
    crops = signals[:, :, start:start + crop_len]
    out = []
    for i in range(0, len(crops), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(crops[i:i + batch_size])).float().to(device)
        out.append(F.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _record_class(record: Dict, label: np.ndarray) -> str:
    if record.get("class"):
        return str(record["class"])
    return CLASS_NAMES_SUPER5[int(np.argmax(label))]


def load_records(input_dir: Path, source_indices: np.ndarray | None) -> List[Dict]:
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        return []
    records = json.loads(summary_path.read_text()).get("records", [])
    if source_indices is not None and len(records) >= int(source_indices.max()) + 1:
        return [records[int(i)] for i in source_indices]
    return records


def summarize_rows(rows: List[Dict], center_names: List[str]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["center"], row["arm"], row["class"])].append(row)
    summary = []
    for (center, arm, cls), rs in sorted(grouped.items()):
        expected_idx = center_names.index(center)
        probs = np.asarray([r["prob_expected"] for r in rs], dtype=np.float64)
        top1 = np.asarray([r["pred_center"] == center for r in rs], dtype=np.float64)
        summary.append({
            "center": center,
            "arm": arm,
            "class": cls,
            "n": len(rs),
            "mean_prob_expected": float(probs.mean()),
            "median_prob_expected": float(np.median(probs)),
            "top1_acc_expected": float(top1.mean()),
            "pred_breakdown": dict(Counter(r["pred_center"] for r in rs)),
            "expected_center_index": expected_idx,
        })
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier_dir", default="/root/autodl-tmp/per_center_style_classifier")
    ap.add_argument("--input_dirs", nargs="+", required=True)
    ap.add_argument("--npz_name", default="samples.npz")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--crop_len", type=int, default=250)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, center_names = load_classifier(args.classifier_dir, args.device)

    rows: List[Dict] = []
    for raw_dir in args.input_dirs:
        input_dir = Path(raw_dir)
        npz_path = input_dir / args.npz_name
        if not npz_path.exists():
            print(f"[skip] missing {npz_path}")
            continue
        with np.load(npz_path, allow_pickle=True) as z:
            signals = z["signals"].astype(np.float32)
            labels = z["labels"].astype(np.float32)
            center = str(z["center_name"]) if "center_name" in z.files else input_dir.name
            source_indices = z["source_indices"] if "source_indices" in z.files else None
        if center not in center_names:
            print(f"[skip] center {center} is not a PN2021 style-classifier class")
            continue
        records = load_records(input_dir.parent if args.npz_name.startswith("gated_") else input_dir, source_indices)
        probs = classify(model, signals, args.device, args.batch_size, args.crop_len)
        expected_idx = center_names.index(center)
        preds = probs.argmax(axis=1)
        for i in range(len(signals)):
            rec = records[i] if i < len(records) else {}
            cls = _record_class(rec, labels[i])
            arm = str(rec.get("arm") or input_dir.parent.name)
            row = {
                "input_dir": input_dir.as_posix(),
                "center": center,
                "arm": arm,
                "class": cls,
                "sample_index": i,
                "ref_record_id": rec.get("ref_record_id"),
                "prompt_token": rec.get("prompt_token"),
                "token_center": rec.get("token_center"),
                "token_class": rec.get("token_class"),
                "pred_center": center_names[int(preds[i])],
                "prob_expected": float(probs[i, expected_idx]),
            }
            for j, c in enumerate(center_names):
                row[f"prob_{c}"] = float(probs[i, j])
            rows.append(row)
        print(f"[score] {npz_path}: n={len(signals)} center={center}")

    if not rows:
        raise RuntimeError("no rows scored")

    sample_path = out_dir / f"style_probe_samples_{args.npz_name.replace('.', '_')}.csv"
    with sample_path.open("w", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = summarize_rows(rows, center_names)
    summary_path = out_dir / f"style_probe_summary_{args.npz_name.replace('.', '_')}.csv"
    with summary_path.open("w", newline="") as f:
        fields = ["center", "arm", "class", "n", "mean_prob_expected", "median_prob_expected",
                  "top1_acc_expected", "pred_breakdown", "expected_center_index"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    json_path = out_dir / f"style_probe_summary_{args.npz_name.replace('.', '_')}.json"
    json_path.write_text(json.dumps({"summary": summary, "n_samples": len(rows)}, indent=2))
    print(f"[done] wrote {sample_path}")
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {json_path}")


if __name__ == "__main__":
    main()
