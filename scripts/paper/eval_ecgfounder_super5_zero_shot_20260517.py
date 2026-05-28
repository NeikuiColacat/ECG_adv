#!/usr/bin/env python3
"""Evaluate ECGFounder out-of-box predictions under this repo's Super5 protocol.

This is a foundation-model baseline, not a target-adaptation method:

* ECGFounder official Net1D, preprocessing, and 12-lead checkpoint are reused
  from /root/autodl-tmp/ecgfounder.
* ECGFounder 150-class logits are projected into PTB-XL Super5
  classes: CD / HYP / MI / NORM / STTC. The default mapping is a strict,
  explicit, versioned task list; the older broad keyword mapping is retained
  only as a legacy weak-baseline option.
* PN2021 labels use scripts.triple_labels.label_schemes.snomed_list_to_super5,
  matching the current repo Super5 evaluation policy.
* For fair target-center comparison, this script reports metrics with the same
  K=500 ref ids excluded for ningbo, chapman_shaoxing, cpsc_2018, and georgia.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATED_DATA_ROOT = Path(
    "/home/linbinhao/ECG/ecg_paper_migration_full_20260522_extract/root/autodl-tmp"
)
DATA_ROOT = Path(
    os.environ.get(
        "ECG_ADV_GEN_DATA_ROOT",
        str(_MIGRATED_DATA_ROOT if _MIGRATED_DATA_ROOT.exists() else Path("/root/autodl-tmp")),
    )
)
ECGFOUNDER_ROOT = Path(os.environ.get("ECGFOUNDER_ROOT", str(DATA_ROOT / "ecgfounder")))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ECGFOUNDER_ROOT))

from scripts.triple_labels.label_schemes import (  # noqa: E402
    CLASS_NAMES_SUPER5,
    snomed_list_to_super5,
)
from ecg_adv_gen.evaluation import MetricRow, compute_macro_metrics  # noqa: E402
from eval_physionet2021 import load_model  # noqa: E402
from physionet2021_dataset import (  # noqa: E402
    PhysioNet2021Dataset,
    scan_records,
)


PN2021_CENTERS = [
    "chapman_shaoxing",
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "ningbo",
    "ptb",
    "st_petersburg_incart",
]
TARGET_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]
PN2021_FORBIDDEN = {"ptb-xl", "ptbxl"}
ECGFOUNDER150_SUPER5_STRICT_VERSION = "ecgfounder150_to_super5_v1_strict_20260522"
ECGFOUNDER150_SUPER5_LEGACY_VERSION = "ecgfounder150_to_super5_legacy_keyword_max_20260517"


STRICT_SUPER5_TASKS = {
    # Only diagnostic normal statements. "SINUS RHYTHM" and
    # "NORMAL SINUS RHYTHM" are intentionally not direct NORM positives because
    # sinus rhythm can coexist with abnormalities.
    "NORM": [
        "NORMAL ECG",
        "otherwise normal ecg",
    ],
    # Explicit infarction / acute MI statements. "CONSIDER ..." and injury
    # pattern labels are kept out of the strict main projection.
    "MI": [
        "SEPTAL INFARCT",
        "ANTERIOR INFARCT",
        "LATERAL INFARCT",
        "ANTEROSEPTAL INFARCT",
        "ANTEROLATERAL INFARCT",
        "INFERIOR INFARCT",
        "ACUTE MI / STEMI",
        "INFERIOR-POSTERIOR INFARCT",
        "ACUTE MI",
        "POSTERIOR INFARCT",
    ],
    # BBB, fascicular block, AV block, IVCD, WPW, and clear conduction delay.
    # Pacemaker/device rhythm labels are NORM suppressors, not direct CD.
    "CD": [
        "RIGHT BUNDLE BRANCH BLOCK",
        "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK",
        "LEFT BUNDLE BRANCH BLOCK",
        "LEFT ANTERIOR FASCICULAR BLOCK",
        "RSR' OR QR PATTERN IN V1 SUGGESTS RIGHT VENTRICULAR CONDUCTION DELAY",
        "INCOMPLETE LEFT BUNDLE BRANCH BLOCK",
        "BIFASCICULAR BLOCK",
        "LEFT POSTERIOR FASCICULAR BLOCK",
        "WITH 1ST DEGREE AV BLOCK",
        "WITH PROLONGED AV CONDUCTION",
        "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY",
        "WOLFF-PARKINSON-WHITE",
        "NONSPECIFIC INTRAVENTRICULAR BLOCK",
        "WITH COMPLETE HEART BLOCK",
        "RBBB AND LEFT ANTERIOR FASCICULAR BLOCK",
        "RBBB AND LEFT POSTERIOR FASCICULAR BLOCK",
        "WITH 2ND DEGREE AV BLOCK MOBITZ I",
        "WITH 2:1 AV CONDUCTION",
    ],
    # Explicit hypertrophy/enlargement only. Voltage criteria is excluded from
    # strict HYP and left for sensitivity analysis.
    "HYP": [
        "LEFT ATRIAL ENLARGEMENT",
        "LEFT VENTRICULAR HYPERTROPHY",
        "RIGHT ATRIAL ENLARGEMENT",
        "BIATRIAL ENLARGEMENT",
        "RIGHT VENTRICULAR HYPERTROPHY",
        "BIVENTRICULAR HYPERTROPHY",
    ],
    # Current-state ST/T/QT abnormalities. Temporal comparison phrases
    # ("now evident", "no longer", "less/more") are excluded from strict main.
    "STTC": [
        "NONSPECIFIC T WAVE ABNORMALITY",
        "NONSPECIFIC ST ABNORMALITY",
        "NONSPECIFIC ST AND T WAVE ABNORMALITY",
        "NON-SPECIFIC CHANGE IN ST SEGMENT IN",
        "WITH REPOLARIZATION ABNORMALITY",
        "PROLONGED QT",
    ],
}


STRICT_NORM_SUPPRESS_TASKS = [
    # Generic abnormal/borderline labels.
    "ABNORMAL ECG",
    "BORDERLINE ECG",
    # Sinus rhythm variants and arrhythmias are not PTB-XL diagnostic NORM.
    "NORMAL SINUS RHYTHM",
    "SINUS RHYTHM",
    "SINUS BRADYCARDIA",
    "SINUS TACHYCARDIA",
    "MARKED SINUS BRADYCARDIA",
    "WITH SINUS ARRHYTHMIA",
    "WITH MARKED SINUS ARRHYTHMIA",
    "WITH SINUS PAUSE",
    "ATRIAL FIBRILLATION",
    "ATRIAL FLUTTER",
    "WITH RAPID VENTRICULAR RESPONSE",
    "WITH SLOW VENTRICULAR RESPONSE",
    "SUPRAVENTRICULAR TACHYCARDIA",
    "VENTRICULAR TACHYCARDIA",
    "MULTIFOCAL ATRIAL TACHYCARDIA",
    "JUNCTIONAL RHYTHM",
    "JUNCTIONAL BRADYCARDIA",
    "ECTOPIC ATRIAL RHYTHM",
    "UNDETERMINED RHYTHM",
    "WITH UNDETERMINED RHYTHM IRREGULARITY",
    "NO P-WAVES FOUND",
    "IDIOVENTRICULAR RHYTHM",
    "WITH JUNCTIONAL ESCAPE COMPLEXES",
    # Ectopy / complexes.
    "PREMATURE VENTRICULAR COMPLEXES",
    "PREMATURE ATRIAL COMPLEXES",
    "PREMATURE SUPRAVENTRICULAR COMPLEXES",
    "FUSION COMPLEXES",
    "WITH PREMATURE VENTRICULAR OR ABERRANTLY CONDUCTED COMPLEXES",
    "PREMATURE ECTOPIC COMPLEXES",
    "PREMATURE VENTRICULAR AND FUSION COMPLEXES",
    "IN A PATTERN OF BIGEMINY",
    "SUPRAVENTRICULAR COMPLEXES",
    "WITH VENTRICULAR ESCAPE COMPLEXES",
    # Axis/voltage/non-super5 morphology.
    "LEFT AXIS DEVIATION",
    "RIGHT AXIS DEVIATION",
    "RIGHTWARD AXIS",
    "RIGHT SUPERIOR AXIS DEVIATION",
    "LEFTWARD AXIS",
    "ABNORMAL LEFT AXIS DEVIATION",
    "ABNORMAL RIGHT AXIS DEVIATION",
    "LOW VOLTAGE QRS",
    "PULMONARY DISEASE PATTERN",
    "PEDIATRIC ECG ANALYSIS",
    # Device/pacing labels suppress NORM but are not strict CD positives.
    "ELECTRONIC ATRIAL PACEMAKER",
    "ELECTRONIC VENTRICULAR PACEMAKER",
    "WITH A COMPETING JUNCTIONAL PACEMAKER",
    "VENTRICULAR-PACED RHYTHM",
    "ATRIAL-PACED RHYTHM",
    "ATRIAL-SENSED VENTRICULAR-PACED RHYTHM",
    "AV SEQUENTIAL OR DUAL CHAMBER ELECTRONIC PACEMAKER",
    "AV DUAL-PACED RHYTHM",
    "VENTRICULAR-PACED COMPLEXES",
    "ELECTRONIC DEMAND PACING",
    "BIVENTRICULAR PACEMAKER DETECTED",
    "SUSPECT UNSPECIFIED PACEMAKER FAILURE",
    "AV DUAL-PACED COMPLEXES",
    "ATRIAL-PACED COMPLEXES",
    # Strict HYP/STTC/MI/CD sensitivity-only labels.
    "VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY",
    "WITH QRS WIDENING",
    "WITH QRS WIDENING AND REPOLARIZATION ABNORMALITY",
    "RSR' PATTERN IN V1",
    "WIDE QRS RHYTHM",
    "WIDE QRS TACHYCARDIA",
    "ABERRANT CONDUCTION",
    "WITH SHORT PR",
    "WITH RETROGRADE CONDUCTION",
    "WITH AV DISSOCIATION",
    "BLOCKED",
    "WITH 2ND DEGREE SA BLOCK MOBITZ I",
    "WITH 2ND DEGREE SA BLOCK MOBITZ II",
    "WITH VARIABLE AV BLOCK",
    "QT HAS SHORTENED",
    "QT HAS LENGTHENED",
    "NONSPECIFIC T WAVE ABNORMALITY NOW EVIDENT IN",
    "NONSPECIFIC T WAVE ABNORMALITY NO LONGER EVIDENT IN",
    "T WAVE INVERSION NOW EVIDENT IN",
    "T WAVE INVERSION NO LONGER EVIDENT IN",
    "ST NOW DEPRESSED IN",
    "ST NO LONGER DEPRESSED IN",
    "INVERTED T WAVES HAVE REPLACED NONSPECIFIC T WAVE ABNORMALITY IN",
    "NONSPECIFIC T WAVE ABNORMALITY HAS REPLACED INVERTED T WAVES IN",
    "T WAVE INVERSION LESS EVIDENT IN",
    "T WAVE INVERSION MORE EVIDENT IN",
    "OR DIGITALIS EFFECT",
    "ST NO LONGER ELEVATED IN",
    "ST ELEVATION NOW PRESENT IN",
    "T WAVE AMPLITUDE HAS DECREASED IN",
    "T WAVE AMPLITUDE HAS INCREASED IN",
    "ST LESS DEPRESSED IN",
    "EARLY REPOLARIZATION",
    "ST MORE DEPRESSED IN",
    "ST ELEVATION HAS REPLACED ST DEPRESSION IN",
    "ST LESS ELEVATED IN",
    "ST MORE ELEVATED IN",
    "ACUTE PERICARDITIS",
    "LATERAL INJURY PATTERN",
    "INFERIOR INJURY PATTERN",
    "ANTERIOR INJURY PATTERN",
    "INFEROLATERAL INJURY PATTERN",
    "ANTEROLATERAL INJURY PATTERN",
    "CONSIDER RIGHT VENTRICULAR INVOLVEMENT IN ACUTE INFERIOR INFARCT",
]


def load_task_names(tasks_path: Path) -> list[str]:
    with open(tasks_path) as f:
        return [line.strip() for line in f if line.strip()]


def _contains_any(name: str, needles: tuple[str, ...]) -> bool:
    return any(n in name for n in needles)


def _task_lookup(task_names: list[str]) -> dict[str, int]:
    return {name.upper(): i for i, name in enumerate(task_names)}


def _indices_from_names(task_names: list[str], names: list[str]) -> list[int]:
    lookup = _task_lookup(task_names)
    missing = [name for name in names if name.upper() not in lookup]
    if missing:
        raise ValueError(f"ECGFounder task names missing from tasks.txt: {missing}")
    return [lookup[name.upper()] for name in names]


def _mapping_hash(pools: dict[str, list[int]], suppress: list[int], task_names: list[str], policy: str) -> str:
    payload = {
        "policy": policy,
        "pools": {cls: [task_names[i] for i in idx] for cls, idx in pools.items()},
        "norm_suppress": [task_names[i] for i in suppress],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def build_legacy_keyword_founder_indices(task_names: list[str]) -> dict[str, list[int]]:
    """Build the historical broad keyword pools for each Super5 class."""
    pools = {c: [] for c in CLASS_NAMES_SUPER5}
    for i, raw_name in enumerate(task_names):
        name = raw_name.upper()

        if name in {"NORMAL SINUS RHYTHM", "NORMAL ECG", "SINUS RHYTHM", "OTHERWISE NORMAL ECG"}:
            pools["NORM"].append(i)

        if _contains_any(name, ("INFARCT", "ACUTE MI", "STEMI")):
            pools["MI"].append(i)

        if (
            _contains_any(name, ("BUNDLE BRANCH BLOCK", "FASCICULAR BLOCK", "INTRAVENTRICULAR CONDUCTION"))
            or _contains_any(name, ("INTRAVENTRICULAR BLOCK", "ABERRANT CONDUCTION", "QRS WIDENING"))
            or _contains_any(name, ("RSR'", "WOLFF-PARKINSON-WHITE", "PACEMAKER", "PACED"))
            or _contains_any(name, ("AV BLOCK", "AV CONDUCTION", "COMPLETE HEART BLOCK"))
        ):
            pools["CD"].append(i)

        if _contains_any(name, ("HYPERTROPHY", "ENLARGEMENT", "VOLTAGE CRITERIA")):
            pools["HYP"].append(i)

        if (
            _contains_any(name, ("T WAVE", "REPOLARIZATION", "PROLONGED QT", "QT HAS"))
            or _contains_any(name, ("NONSPECIFIC ST", "ST NOW", "ST NO LONGER", "ST ELEVATION"))
            or _contains_any(name, ("ST LESS", "ST MORE", "ST DEPRESSION", "ST SEGMENT"))
            or _contains_any(name, ("NON-SPECIFIC CHANGE IN ST", "DIGITALIS EFFECT", "PERICARDITIS"))
        ):
            pools["STTC"].append(i)

    return pools


def build_super5_founder_indices(task_names: list[str], policy: str = "strict_v1") -> dict[str, list[int]]:
    """Build 0-based ECGFounder task-index pools for each Super5 class."""
    if policy == "legacy_keyword":
        return build_legacy_keyword_founder_indices(task_names)
    if policy != "strict_v1":
        raise ValueError(f"unknown ECGFounder Super5 mapping policy: {policy}")
    return {
        cls: _indices_from_names(task_names, STRICT_SUPER5_TASKS[cls])
        for cls in CLASS_NAMES_SUPER5
    }


def build_super5_founder_mapping(task_names: list[str], policy: str = "strict_v1") -> dict:
    pools = build_super5_founder_indices(task_names, policy=policy)
    if policy == "legacy_keyword":
        suppress: list[int] = []
        version = ECGFOUNDER150_SUPER5_LEGACY_VERSION
        description = "historical broad keyword pool with per-class max pooling"
    elif policy == "strict_v1":
        suppress = sorted(set(
            _indices_from_names(task_names, STRICT_NORM_SUPPRESS_TASKS)
            + [i for cls in CLASS_NAMES_SUPER5 if cls != "NORM" for i in pools[cls]]
        ))
        version = ECGFOUNDER150_SUPER5_STRICT_VERSION
        description = (
            "explicit strict ECGFounder-150 to PTB-XL Super5 projection; "
            "device/pacing, voltage-only, temporal-comparison, rhythm-only, "
            "axis/ectopy, and boundary labels suppress NORM but are not direct positives"
        )
    else:
        raise ValueError(f"unknown ECGFounder Super5 mapping policy: {policy}")
    return {
        "policy": policy,
        "version": version,
        "description": description,
        "hash": _mapping_hash(pools, suppress, task_names, policy),
        "pools": pools,
        "norm_suppress_indices": suppress,
        "pool_task_names": {cls: [task_names[i] for i in pools[cls]] for cls in CLASS_NAMES_SUPER5},
        "norm_suppress_task_names": [task_names[i] for i in suppress],
    }


def logits_to_super5(
    logits: torch.Tensor,
    pools: dict[str, list[int]],
    norm_suppress_indices: list[int] | None = None,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    outs = []
    for cls in CLASS_NAMES_SUPER5:
        idx = pools[cls]
        if idx:
            outs.append(probs[:, idx].max(dim=1).values)
        else:
            outs.append(torch.zeros(probs.shape[0], device=probs.device))
    out = torch.stack(outs, dim=1)
    if norm_suppress_indices:
        suppress_prob = probs[:, norm_suppress_indices].max(dim=1).values
        norm_idx = list(CLASS_NAMES_SUPER5).index("NORM")
        out[:, norm_idx] = out[:, norm_idx] * (1.0 - suppress_prob)
    return out


def center_from_record_path(path: str) -> str:
    parts = Path(path).parts
    if "training" in parts:
        idx = parts.index("training")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(path).parent.name


def record_id_from_record_path(path: str) -> str:
    return Path(path).name


def load_or_scan_records(data_root: Path, manifest_cache: Path) -> list[dict]:
    if manifest_cache.exists():
        with open(manifest_cache) as f:
            return json.load(f)
    records = scan_records(str(data_root))
    manifest_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_cache, "w") as f:
        json.dump(records, f)
    return records


class Super5FounderDataset(Dataset):
    def __init__(self, data_root: Path, records: list[dict]):
        self.inner = PhysioNet2021Dataset(str(data_root), records=records)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        signal, _ = self.inner[idx]
        rec = self.records[idx]
        label = torch.from_numpy(snomed_list_to_super5(rec.get("snomed", []))).float()
        return signal, label, center_from_record_path(rec["path"]), record_id_from_record_path(rec["path"])


def compute_macro(y_true: np.ndarray, y_score: np.ndarray, min_pos: int = 10) -> MetricRow:
    return compute_macro_metrics(
        y_true,
        y_score,
        class_names=CLASS_NAMES_SUPER5,
        min_pos=min_pos,
    )


def load_ref_ids(ref_root: Path, centers: list[str]) -> dict[str, set[str]]:
    out = {}
    for center in centers:
        p = ref_root / center / "k500_seed20260531" / f"{center}_real_k500_seed20260531.ref_meta.json"
        with open(p) as f:
            meta = json.load(f)
        out[center] = set(meta["ref_record_ids"])
    return out


def evaluate_views(
    labels: np.ndarray,
    scores: np.ndarray,
    centers: np.ndarray,
    record_ids: np.ndarray,
    ref_ids_by_center: dict[str, set[str]],
) -> dict:
    views = {}
    for target_center in TARGET_CENTERS:
        ref_ids = ref_ids_by_center[target_center]
        per_center = {}
        avg_aurocs, avg_auprcs = [], []
        for center in PN2021_CENTERS:
            if center.lower() in PN2021_FORBIDDEN:
                continue
            mask = centers == center
            n_total = int(mask.sum())
            n_excluded = 0
            if center == target_center:
                exclude = np.array([rid in ref_ids for rid in record_ids], dtype=bool)
                n_excluded = int((mask & exclude).sum())
                mask = mask & ~exclude
            if int(mask.sum()) == 0:
                continue
            m = compute_macro(labels[mask], scores[mask])
            per_center[center] = {
                "n_records": n_total,
                "n_excluded_ref": n_excluded,
                "effective_n": int(mask.sum()),
                "macro_auroc": m.macro_auroc,
                "macro_auprc": m.macro_auprc,
                "n_classes_used": m.n_classes_used,
                "per_class": m.per_class,
            }
            if m.macro_auroc is not None:
                avg_aurocs.append(m.macro_auroc)
                avg_auprcs.append(m.macro_auprc)
        views[target_center] = {
            "target_center": target_center,
            "avg_macro_auroc": float(np.mean(avg_aurocs)),
            "avg_macro_auprc": float(np.mean(avg_auprcs)),
            "per_center": per_center,
        }
    return views


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/root/autodl-tmp/physionet2021/training")
    ap.add_argument("--ecgfounder_root", default=str(ECGFOUNDER_ROOT))
    ap.add_argument("--checkpoint", default="/root/autodl-tmp/ecgfounder/checkpoint/12_lead_ECGFounder.pth")
    ap.add_argument("--manifest_cache", default="/root/autodl-tmp/ecgfounder/physionet2021_manifest.json")
    ap.add_argument("--ref_root", default="/root/autodl-tmp/paper_vae_only_latenthull_sweep_20260516/subsets")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/paper_foundation_baselines_20260517/ecgfounder_super5_zeroshot")
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit_per_center", type=int, default=0, help="debug only")
    ap.add_argument(
        "--mapping_policy",
        choices=["strict_v1", "legacy_keyword"],
        default="strict_v1",
        help="ECGFounder 150-label to Super5 projection policy.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_or_scan_records(Path(args.data_root), Path(args.manifest_cache))
    records = [r for r in records if center_from_record_path(r["path"]) in PN2021_CENTERS]
    if args.limit_per_center:
        limited, counts = [], {c: 0 for c in PN2021_CENTERS}
        for r in records:
            c = center_from_record_path(r["path"])
            if counts[c] < args.limit_per_center:
                limited.append(r)
                counts[c] += 1
        records = limited

    task_names = load_task_names(Path(args.ecgfounder_root) / "tasks.txt")
    mapping = build_super5_founder_mapping(task_names, policy=args.mapping_policy)
    pools = mapping["pools"]
    print(
        f"[mapping] ECGFounder -> Super5 {mapping['version']} "
        f"hash={mapping['hash']}"
    )
    for cls in CLASS_NAMES_SUPER5:
        names = mapping["pool_task_names"][cls]
        print(f"  {cls}: {len(names)} tasks -> {names[:12]}{' ...' if len(names) > 12 else ''}")
    if mapping["norm_suppress_indices"]:
        print(f"  NORM suppress: {len(mapping['norm_suppress_indices'])} tasks")

    model = load_model(args.checkpoint, args.device)
    dataset = Super5FounderDataset(Path(args.data_root), records)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )

    labels_all, scores_all, centers_all, record_ids_all = [], [], [], []
    t0 = time.time()
    for signals, labels, batch_centers, batch_record_ids in tqdm(loader, desc="ECGFounder Super5"):
        signals = signals.to(args.device, non_blocking=True)
        with torch.no_grad():
            logits = model(signals)
            scores = logits_to_super5(logits, pools, mapping["norm_suppress_indices"])
        labels_all.append(labels.numpy())
        scores_all.append(scores.cpu().numpy())
        centers_all.extend(list(batch_centers))
        record_ids_all.extend(list(batch_record_ids))
    elapsed = time.time() - t0

    labels = np.concatenate(labels_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    centers = np.asarray(centers_all)
    record_ids = np.asarray(record_ids_all)
    ref_ids_by_center = load_ref_ids(Path(args.ref_root), TARGET_CENTERS)
    views = evaluate_views(labels, scores, centers, record_ids, ref_ids_by_center)

    output = {
        "method": f"ECGFounder_12lead_out_of_box_super5_{args.mapping_policy}",
        "checkpoint": args.checkpoint,
        "n_records": int(len(records)),
        "elapsed_s": elapsed,
        "class_names": CLASS_NAMES_SUPER5,
        "ecgfounder_super5_mapping": {
            "policy": mapping["policy"],
            "version": mapping["version"],
            "description": mapping["description"],
            "hash": mapping["hash"],
            "pool_sizes": {cls: len(mapping["pool_task_names"][cls]) for cls in CLASS_NAMES_SUPER5},
            "norm_suppress_pool_size": len(mapping["norm_suppress_task_names"]),
            "pool_task_names": mapping["pool_task_names"],
            "norm_suppress_task_names": mapping["norm_suppress_task_names"],
        },
        "founder_task_pools": mapping["pool_task_names"],
        "views": views,
    }
    json_path = out_dir / f"ecgfounder_super5_{args.mapping_policy}_ref_excluded.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    csv_path = out_dir / f"ecgfounder_super5_{args.mapping_policy}_ref_excluded.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_center_view",
                "center",
                "n_records",
                "n_excluded_ref",
                "effective_n",
                "macro_auroc",
                "macro_auprc",
                "n_classes_used",
            ],
        )
        writer.writeheader()
        for view_name, view in views.items():
            for center, row in view["per_center"].items():
                writer.writerow({
                    "target_center_view": view_name,
                    "center": center,
                    "n_records": row["n_records"],
                    "n_excluded_ref": row["n_excluded_ref"],
                    "effective_n": row["effective_n"],
                    "macro_auroc": row["macro_auroc"],
                    "macro_auprc": row["macro_auprc"],
                    "n_classes_used": row["n_classes_used"],
                })

    print(f"[done] inference {elapsed:.1f}s records={len(records)}")
    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {csv_path}")
    for target in TARGET_CENTERS:
        row = views[target]["per_center"][target]
        ta = "NA" if row["macro_auroc"] is None else f"{row['macro_auroc']:.4f}"
        tp = "NA" if row["macro_auprc"] is None else f"{row['macro_auprc']:.4f}"
        print(
            f"{target:18s} target AUROC={ta} "
            f"AUPRC={tp} "
            f"avg={views[target]['avg_macro_auroc']:.4f}/{views[target]['avg_macro_auprc']:.4f}"
        )


if __name__ == "__main__":
    main()
