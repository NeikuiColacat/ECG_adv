#!/usr/bin/env python3
"""Paper-safe EfficientNet K500-val multi-candidate selector.

Selection uses only the known K=500 target-center internal validation split.
The ref-excluded PN2021 target center is evaluated after the selector is fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.paper.eval_effnet_k500val_classwise_blend_20260524 import (  # noqa: E402
    K500VAL_VAE_RUNS,
    build_k500_val_dataset,
)
from scripts.paper.eval_effnet_score_blend_20260524 import (  # noqa: E402
    ALPHAS,
    CENTERS,
    DATA_ROOT,
    load_center_dataset,
    load_model,
    metric_view,
    sigmoid,
    target_paths,
)
from scripts.triple_labels.label_schemes import get_scheme, get_super5_pn2021_mapping_metadata  # noqa: E402
from scripts.triple_labels.model_zoo import normalize_model_name  # noqa: E402
from scripts.triple_labels.train_ptbxl import compute_macro_auroc_auprc  # noqa: E402


DEFAULT_CANDIDATE_ROOTS = {
    "old_lam015_wlat030": DATA_ROOT / "paper_effnet_k500val_latent_augmix_stage3_v6_20260524",
    "lowadv_localstd": DATA_ROOT / "paper_effnet_localstd_anchorwarm_stage3_v6_20260525",
    "highadv20warm20": DATA_ROOT / "paper_effnet_highadv_stage3_v6_20260525",
}


def parse_candidate_roots(items: list[str]) -> dict[str, Path]:
    out = dict(DEFAULT_CANDIDATE_ROOTS)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected NAME=/path, got {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name or name == "direct":
            raise ValueError(f"invalid candidate name {name!r}")
        out[name] = Path(raw_path).expanduser()
    return out


def single_existing_dir(paths: list[Path], *, label: str) -> Path:
    dirs = sorted({p.parent for p in paths if p.exists()})
    if len(dirs) != 1:
        found = "\n".join(str(p) for p in dirs[:20])
        raise FileNotFoundError(f"expected exactly one {label}, found {len(dirs)}\n{found}")
    return dirs[0]


def find_candidate_dir(name: str, root: Path, center: str, data_root: Path) -> Path:
    if name == "old_lam015_wlat030":
        path = data_root / K500VAL_VAE_RUNS[center]
        if not (path / "best_model.pt").exists():
            raise FileNotFoundError(f"missing old K500-val candidate: {path}")
        return path
    if (root / "best_model.pt").exists():
        return root
    hits = list(root.glob(f"{center}_*/best_model.pt"))
    hits.extend((root / "runs").glob(f"{center}_*/best_model.pt"))
    if name == "lowadv_localstd":
        redist_hits = [p for p in hits if "rare_redist" in str(p)]
        if redist_hits:
            hits = redist_hits
    return single_existing_dir(hits, label=f"{name} candidate for {center} under {root}")


@torch.no_grad()
def infer_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    amp = device.type == "cuda"
    for signals, y in loader:
        signals = signals.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(signals)
        labels.append(y.numpy())
        logits.append(out.float().cpu().numpy())
    return np.concatenate(labels, axis=0), np.concatenate(logits, axis=0)


def blend_scores(direct_logits: np.ndarray, cand_logits: np.ndarray, alpha: float, blend_space: str) -> np.ndarray:
    if alpha <= 0.0:
        return sigmoid(direct_logits)
    if blend_space == "logit":
        return sigmoid((1.0 - alpha) * direct_logits + alpha * cand_logits)
    return (1.0 - alpha) * sigmoid(direct_logits) + alpha * sigmoid(cand_logits)


def choose_global(
    y_true: np.ndarray,
    logits: dict[str, np.ndarray],
    class_names: list[str],
    alphas: list[float],
    blend_space: str,
    min_pos: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in [n for n in logits if n != "direct"]:
        for alpha in alphas:
            scores = blend_scores(logits["direct"], logits[name], alpha, blend_space)
            metrics = metric_view(y_true, scores, class_names, min_pos)
            rows.append({"candidate": name, "alpha": float(alpha), "metrics": metrics})
    best = max(
        rows,
        key=lambda r: (
            float(r["metrics"]["macro_auprc"]),
            float(r["metrics"]["macro_auroc"]),
            -float(r["alpha"]),
        ),
    )
    return {"best": {"candidate": best["candidate"], "alpha": float(best["alpha"])}, "grid": rows}


def choose_classwise(
    y_true: np.ndarray,
    logits: dict[str, np.ndarray],
    class_names: list[str],
    alphas: list[float],
    blend_space: str,
    min_pos: int,
    min_val_auprc_gain: float,
    min_val_auroc_gain: float,
) -> dict[str, Any]:
    choices: dict[str, Any] = {}
    candidate_names = [n for n in logits if n != "direct"]
    for idx, cls in enumerate(class_names):
        y = y_true[:, idx : idx + 1]
        n_pos = int((y[:, 0] > 0.5).sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < min_pos or n_neg < 1:
            choices[cls] = {
                "candidate": "direct",
                "alpha": 0.0,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "selection": "direct_fallback_insufficient_val_positives",
            }
            continue
        direct_scores = sigmoid(logits["direct"][:, idx : idx + 1])
        direct_m = compute_macro_auroc_auprc(y, direct_scores, [cls], min_pos=1)["per_class"][cls]
        best: dict[str, Any] = {
            "candidate": "direct",
            "alpha": 0.0,
            "auroc": float(direct_m["auroc"]),
            "auprc": float(direct_m["auprc"]),
            "direct_auroc": float(direct_m["auroc"]),
            "direct_auprc": float(direct_m["auprc"]),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "selection": "direct",
        }
        for name in candidate_names:
            for alpha in alphas:
                scores = blend_scores(
                    logits["direct"][:, idx : idx + 1],
                    logits[name][:, idx : idx + 1],
                    alpha,
                    blend_space,
                )
                m = compute_macro_auroc_auprc(y, scores, [cls], min_pos=1)["per_class"][cls]
                row = {
                    "candidate": name,
                    "alpha": float(alpha),
                    "auroc": float(m["auroc"]),
                    "auprc": float(m["auprc"]),
                    "direct_auroc": float(direct_m["auroc"]),
                    "direct_auprc": float(direct_m["auprc"]),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "selection": "candidate",
                }
                if (
                    row["auprc"] > best["auprc"]
                    or (row["auprc"] == best["auprc"] and row["auroc"] > best["auroc"])
                    or (
                        row["auprc"] == best["auprc"]
                        and row["auroc"] == best["auroc"]
                        and row["alpha"] < best["alpha"]
                    )
                ):
                    best = row
        if best["candidate"] != "direct":
            if best["auprc"] - best["direct_auprc"] < min_val_auprc_gain:
                best = {**best, "candidate": "direct", "alpha": 0.0, "selection": "direct_min_auprc_gain_gate"}
            elif best["auroc"] - best["direct_auroc"] < min_val_auroc_gain:
                best = {**best, "candidate": "direct", "alpha": 0.0, "selection": "direct_min_auroc_gain_gate"}
        choices[cls] = best
    return choices


def apply_classwise(
    direct_logits: np.ndarray,
    logits: dict[str, np.ndarray],
    choices: dict[str, Any],
    class_names: list[str],
    blend_space: str,
) -> np.ndarray:
    out = sigmoid(direct_logits)
    for idx, cls in enumerate(class_names):
        choice = choices[cls]
        cand = str(choice["candidate"])
        alpha = float(choice["alpha"])
        if cand == "direct" or alpha <= 0.0:
            continue
        out[:, idx : idx + 1] = blend_scores(
            direct_logits[:, idx : idx + 1],
            logits[cand][:, idx : idx + 1],
            alpha,
            blend_space,
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=CENTERS)
    p.add_argument("--candidate_roots", nargs="*", default=[])
    p.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    p.add_argument("--blend_space", choices=["prob", "logit"], default="prob")
    p.add_argument("--data_root", type=Path, default=DATA_ROOT)
    p.add_argument("--out_dir", type=Path, default=DATA_ROOT / "paper_effnet_k500val_multicandidate_selector_v6_20260525")
    p.add_argument("--model_name", default="efficientnet1dv2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--min_pos", type=int, default=10)
    p.add_argument("--preprocess_mode", default="minimal_resample")
    p.add_argument("--norm_mode", default="per_sample_global")
    p.add_argument("--target_real_val_fraction", type=float, default=0.2)
    p.add_argument("--target_real_val_seed", type=int, default=20260531)
    p.add_argument("--min_val_auprc_gain", type=float, default=0.0)
    p.add_argument("--min_val_auroc_gain", type=float, default=0.0)
    args = p.parse_args()

    args.model_name = normalize_model_name(args.model_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_roots = parse_candidate_roots(args.candidate_roots)
    device = torch.device(args.device)
    scheme = get_scheme("super5")
    class_names = list(scheme["class_names"])

    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "method": "EfficientNet direct/VAE multi-candidate score fusion selected on K500-internal validation",
        "class_names": class_names,
        "candidate_roots": {k: str(v) for k, v in candidate_roots.items()},
        "alphas": args.alphas,
        "blend_space": args.blend_space,
        "target_real_val_fraction": args.target_real_val_fraction,
        "target_real_val_seed": args.target_real_val_seed,
        "min_val_auprc_gain": args.min_val_auprc_gain,
        "min_val_auroc_gain": args.min_val_auroc_gain,
        "label_mapping": get_super5_pn2021_mapping_metadata(),
        "centers": {},
    }

    for center in args.centers:
        print(f"[center] {center}")
        paths = target_paths(args.data_root, center)
        candidate_dirs = {
            name: find_candidate_dir(name, root, center, args.data_root)
            for name, root in candidate_roots.items()
        }
        print(f"  direct={paths['direct_dir']}")
        for name, path in candidate_dirs.items():
            print(f"  {name}={path}")

        val_ds, val_meta = build_k500_val_dataset(center, args)
        heldout_ds, heldout_meta = load_center_dataset(center, scheme, args)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        heldout_loader = DataLoader(
            heldout_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        all_dirs = {"direct": paths["direct_dir"], **candidate_dirs}
        val_logits: dict[str, np.ndarray] = {}
        heldout_logits: dict[str, np.ndarray] = {}
        y_val: np.ndarray | None = None
        y_heldout: np.ndarray | None = None
        for name, model_dir in all_dirs.items():
            model = load_model(model_dir, args.model_name, scheme["num_classes"], device)
            yv, lv = infer_logits(model, val_loader, device)
            yh, lh = infer_logits(model, heldout_loader, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if y_val is None:
                y_val = yv
                y_heldout = yh
            val_logits[name] = lv
            heldout_logits[name] = lh
        assert y_val is not None and y_heldout is not None

        global_choice = choose_global(
            y_val,
            val_logits,
            class_names,
            args.alphas,
            args.blend_space,
            args.min_pos,
        )
        class_choices = choose_classwise(
            y_val,
            val_logits,
            class_names,
            args.alphas,
            args.blend_space,
            args.min_pos,
            args.min_val_auprc_gain,
            args.min_val_auroc_gain,
        )
        direct_scores = sigmoid(heldout_logits["direct"])
        direct_m = metric_view(y_heldout, direct_scores, class_names, args.min_pos)
        metrics: dict[str, Any] = {"direct": direct_m}
        for name in candidate_dirs:
            metrics[name] = metric_view(y_heldout, sigmoid(heldout_logits[name]), class_names, args.min_pos)
        best_global = global_choice["best"]
        global_scores = blend_scores(
            heldout_logits["direct"],
            heldout_logits[best_global["candidate"]],
            float(best_global["alpha"]),
            args.blend_space,
        )
        classwise_scores = apply_classwise(
            heldout_logits["direct"],
            heldout_logits,
            class_choices,
            class_names,
            args.blend_space,
        )
        metrics["global_k500val_multicandidate"] = metric_view(y_heldout, global_scores, class_names, args.min_pos)
        metrics["classwise_k500val_multicandidate"] = metric_view(y_heldout, classwise_scores, class_names, args.min_pos)

        for method, m in metrics.items():
            rows.append(
                {
                    "center": center,
                    "method": method,
                    "target_auroc": m["macro_auroc"],
                    "target_auprc": m["macro_auprc"],
                    "delta_vs_direct_auroc": m["macro_auroc"] - direct_m["macro_auroc"],
                    "delta_vs_direct_auprc": m["macro_auprc"] - direct_m["macro_auprc"],
                    "global_candidate": best_global["candidate"],
                    "global_alpha": best_global["alpha"],
                    "classwise_choices": json.dumps(
                        {k: {"candidate": v["candidate"], "alpha": v["alpha"]} for k, v in class_choices.items()},
                        sort_keys=True,
                    ),
                }
            )
        payload["centers"][center] = {
            "paths": {
                "direct_dir": str(paths["direct_dir"]),
                "candidate_dirs": {k: str(v) for k, v in candidate_dirs.items()},
                "ref_meta": str(paths["ref_meta"]),
            },
            "k500_val": val_meta,
            "heldout": heldout_meta,
            "global_selection": global_choice,
            "classwise_selection": class_choices,
            "heldout_metrics": metrics,
        }
        best_report = max(
            [r for r in rows if r["center"] == center and "multicandidate" in r["method"]],
            key=lambda r: (float(r["target_auprc"]), float(r["target_auroc"])),
        )
        print(
            f"  best {best_report['method']} "
            f"{float(best_report['target_auroc']):.6f}/"
            f"{float(best_report['target_auprc']):.6f} "
            f"delta={float(best_report['delta_vs_direct_auroc']):+.6f}/"
            f"{float(best_report['delta_vs_direct_auprc']):+.6f}"
        )

    csv_path = args.out_dir / "effnet_k500val_multicandidate_selector.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.out_dir / "effnet_k500val_multicandidate_selector.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[summary] wrote {csv_path}")
    print(f"[summary] wrote {json_path}")


if __name__ == "__main__":
    main()
