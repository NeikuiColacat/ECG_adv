"""One-way scalar mirror in the isolated Trackio environment; no ECG uploads."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def scalar_rows(result, directory):
    if result["artifact_type"] == "pulse_train_result":
        protocol = result["protocol"]
        config = {"center": protocol["center"], "width": protocol["width"],
                  "training": protocol["training"], "development_only": True}
        history = json.loads((directory / "history.json").read_text())
        rows = [(r["step"], {k: r[k] for k in ("loss", "clean_answer_ce", "answer_token_jsd", "seconds", "peak_allocated_bytes") if k in r}) for r in history]
    elif result["artifact_type"] == "pulse_hybrid_result" and result["mode"] == "evaluate":
        config = {k: result["details"][k] for k in ("center", "phase", "records", "development_only")}
        rows = [(0, {f"{arm}/{family}/macro_f1": value for arm, families in result["details"]["families"].items()
                     for family, value in families.items()})]
    else:
        raise ValueError("unsupported scalar tracking source")
    return config, rows


def main():
    parser = argparse.ArgumentParser()
    for name in ("result", "storage", "name", "project"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    path, storage = Path(args.result).resolve(), Path(args.storage).resolve()
    roots = (Path("/home/linbinhao/ECG_adv_data"), Path("/dev/shm/linbinhao-pulse-hybrid"))
    if any(not any(p.is_relative_to(r) for r in roots) for p in (path, storage)):
        raise ValueError("tracking paths outside the experiment roots")
    os.environ.update(TRACKIO_DIR=str(storage), HF_HUB_OFFLINE="1", DO_NOT_TRACK="1")
    import trackio
    result = json.loads(path.read_text())
    config, rows = scalar_rows(result, path.parent)
    trackio.init(project=args.project, name=args.name, config=config,
                 auto_log_cpu=False, auto_log_gpu=False, resume="never")
    for step, row in rows:
        trackio.log(row, step=step)
    trackio.finish()


if __name__ == "__main__":
    main()
