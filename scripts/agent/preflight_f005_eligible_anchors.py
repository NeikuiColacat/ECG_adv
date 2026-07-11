#!/usr/bin/env python3
"""CPU/data preflight for all predeclared F005 exact partner pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ecg_adv_gen.adaptation.lhat import build_exact_eligibility_manifest  # noqa: E402
from ecg_adv_gen.config import build_runner_commands, load_experiment_config  # noqa: E402
from ecg_adv_gen.data.kshot import filter_latent_candidates, matched_k500_split  # noqa: E402
from ecg_adv_gen.data.latent_pools import load_synth_pool  # noqa: E402


CONFIG = REPO / "configs/studies/f005_anchor_geometry_control_train.yaml"
LOCAL = REPO / "configs/local/linbinhao_server.example.yaml"


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/ecg-f005-eligible-preflight.json"))
    args = parser.parse_args()
    config = load_experiment_config(CONFIG, LOCAL, runtime_context={"run_id": "F005_PREFLIGHT"})
    commands = build_runner_commands(config)
    cells = {}
    for command in commands:
        matrix, argv = command["matrix"], command["argv"]
        case = matrix["case"]
        key = (matrix["center"], int(case["seed"]))
        cells.setdefault(key, Path(_option(argv, "--anchor_base")))

    rows, blockers = [], []
    for (center, seed), base in sorted(cells.items()):
        latent_path, signal_path = base.with_suffix(".latent.npz"), base.with_suffix(".raw1000.npz")
        missing = [str(path) for path in (latent_path, signal_path) if not path.exists()]
        if missing:
            rows.append({"center": center, "seed": seed, "status": "missing", "missing": missing})
            blockers.extend(missing)
            continue
        latents, labels, _, source_meta = load_synth_pool(latent_path)
        with np.load(signal_path, allow_pickle=True) as data:
            signal_labels = np.asarray(data["labels"], dtype=np.float32)
            signal_ids = np.asarray(data["record_ids"]).astype(str)
        split = matched_k500_split(
            signal_ids,
            signal_labels,
            val_fraction=float(config["paper_protocol"]["selection"]["validation_fraction"]),
            seed=seed,
        )
        _, filtered_labels, filtered_meta = filter_latent_candidates(
            latents,
            labels,
            source_meta,
            train_record_ids=split["train_record_ids"],
            validation_record_ids=split["val_record_ids"],
        )
        manifest = build_exact_eligibility_manifest(
            filtered_labels,
            filtered_meta["record_ids"],
            min_nonself=2,
        )
        rows.append({
            "center": center,
            "seed": seed,
            "status": "pass" if manifest["eligible_count"] > 0 else "blocked",
            "train_count": len(filtered_labels),
            "eligible_count": manifest["eligible_count"],
            "excluded_insufficient_nonself_count": manifest["ineligible_count"],
            "manifest_sha256": manifest["manifest_sha256"],
            "exact_label_sets": manifest["exact_label_sets"],
        })
        if manifest["eligible_count"] == 0:
            blockers.append(f"{center}/seed{seed}: no eligible exact anchors")
    report = {
        "schema_version": 1,
        "study_scope": config["study"]["scope"],
        "canonical_exact_promotion_safe": not blockers and len(rows) == 12,
        "n_expected_cells": 12,
        "n_checked_cells": len(rows),
        "blockers": blockers,
        "cells": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if report["canonical_exact_promotion_safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
