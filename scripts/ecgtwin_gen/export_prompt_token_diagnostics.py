"""Export prompt-token bank diagnostics.

This script is intentionally lightweight: it only reads saved token banks and
writes CSV summaries for token norms, deltas, and cosine similarities. It does
not load ECGTwin or use the GPU.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F


DEFAULT_PATTERNS = [
    "/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs/*/prompt_token_bank.pt",
]


def _collect_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern))
    return sorted(set(paths))


def _as_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(v) for v in value]


def _token_labels(centers: List[str], class_names: List[str], embeddings: torch.Tensor) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if embeddings.ndim == 3:
        for cidx, center in enumerate(centers):
            for kidx, cls in enumerate(class_names):
                rows.append({
                    "center": center,
                    "class": cls,
                    "slot": 0,
                    "index": (cidx, kidx),
                    "name": f"{center}_{cls}",
                })
    elif embeddings.ndim == 4:
        for cidx, center in enumerate(centers):
            for kidx, cls in enumerate(class_names):
                for slot in range(embeddings.shape[2]):
                    rows.append({
                        "center": center,
                        "class": cls,
                        "slot": slot,
                        "index": (cidx, kidx, slot),
                        "name": f"{center}_{cls}_{slot}",
                    })
    else:
        raise ValueError(f"unsupported embeddings shape: {tuple(embeddings.shape)}")
    return rows


def _flatten_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim == 3:
        return embeddings.reshape(-1, embeddings.shape[-1])
    if embeddings.ndim == 4:
        return embeddings.reshape(-1, embeddings.shape[-1])
    raise ValueError(f"unsupported embeddings shape: {tuple(embeddings.shape)}")


def _extract_tokens(bank: Dict[str, Any]) -> tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor | None]:
    centers = _as_list(bank["centers"])
    class_names = _as_list(bank["class_names"])
    if "embeddings" in bank:
        embeddings = bank["embeddings"].float()
        init_embeddings = bank.get("init_embeddings")
        labels = _token_labels(centers, class_names, embeddings)
        flat = _flatten_embeddings(embeddings)
        flat_init = _flatten_embeddings(init_embeddings.float()) if init_embeddings is not None else None
        return labels, flat, flat_init

    if {"center_embeddings", "class_embeddings", "residual_embeddings"}.issubset(bank):
        labels: List[Dict[str, Any]] = []
        vectors: List[torch.Tensor] = []
        init_vectors: List[torch.Tensor] = []

        center_embeddings = bank["center_embeddings"].float()
        class_embeddings = bank["class_embeddings"].float()
        residual_embeddings = bank["residual_embeddings"].float()
        init_center = bank.get("init_center_embeddings")
        init_class = bank.get("init_class_embeddings")
        init_residual = bank.get("init_residual_embeddings")

        for cidx, center in enumerate(centers):
            for slot in range(center_embeddings.shape[1]):
                labels.append({
                    "center": center,
                    "class": "*",
                    "slot": slot,
                    "name": f"{center}_CENTER_{slot}",
                })
                vectors.append(center_embeddings[cidx, slot])
                if init_center is not None:
                    init_vectors.append(init_center.float()[cidx, slot])
        for kidx, cls in enumerate(class_names):
            for slot in range(class_embeddings.shape[1]):
                labels.append({
                    "center": "*",
                    "class": cls,
                    "slot": slot,
                    "name": f"CLASS_{cls}_{slot}",
                })
                vectors.append(class_embeddings[kidx, slot])
                if init_class is not None:
                    init_vectors.append(init_class.float()[kidx, slot])
        for cidx, center in enumerate(centers):
            for kidx, cls in enumerate(class_names):
                for slot in range(residual_embeddings.shape[2]):
                    labels.append({
                        "center": center,
                        "class": cls,
                        "slot": slot,
                        "name": f"{center}_{cls}_RESIDUAL_{slot}",
                    })
                    vectors.append(residual_embeddings[cidx, kidx, slot])
                    if init_residual is not None:
                        init_vectors.append(init_residual.float()[cidx, kidx, slot])

        flat = torch.stack(vectors, dim=0)
        flat_init = torch.stack(init_vectors, dim=0) if init_vectors else None
        return labels, flat, flat_init

    raise KeyError("unsupported prompt-token bank schema")


def _safe_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def export_bank(path: Path, out_dir: Path, root: Path) -> Dict[str, int]:
    bank = torch.load(path, map_location="cpu")
    labels, flat, flat_init = _extract_tokens(bank)
    flat_norm = F.normalize(flat, dim=-1)
    cosine = flat_norm @ flat_norm.T

    try:
        run_name = path.parent.relative_to(root).as_posix()
    except ValueError:
        run_name = path.parent.as_posix()

    safe_name = f"{run_name}__{path.stem}".replace("/", "__")
    token_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for i, label in enumerate(labels):
        vec = flat[i]
        row = {
            "run": run_name,
            "bank_path": path.as_posix(),
            "token": label["name"],
            "center": label["center"],
            "class": label["class"],
            "slot": label["slot"],
            "token_norm": _safe_float(vec.norm()),
        }
        if flat_init is not None:
            init = flat_init[i]
            row.update({
                "init_norm": _safe_float(init.norm()),
                "delta_norm": _safe_float((vec - init).norm()),
                "cos_to_init": _safe_float(F.cosine_similarity(vec[None], init[None]).squeeze()),
            })
        token_rows.append(row)

    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if j <= i:
                continue
            pair_rows.append({
                "run": run_name,
                "bank_path": path.as_posix(),
                "left_token": left["name"],
                "right_token": right["name"],
                "left_center": left["center"],
                "right_center": right["center"],
                "left_class": left["class"],
                "right_class": right["class"],
                "left_slot": left["slot"],
                "right_slot": right["slot"],
                "same_center": int(left["center"] == right["center"]),
                "same_class": int(left["class"] == right["class"]),
                "cosine": float(cosine[i, j].item()),
            })

    token_path = out_dir / f"{safe_name}.token_summary.csv"
    pair_path = out_dir / f"{safe_name}.token_cosine.csv"
    with token_path.open("w", newline="") as f:
        fields = [
            "run", "bank_path", "token", "center", "class", "slot",
            "token_norm", "init_norm", "delta_norm", "cos_to_init",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(token_rows)
    with pair_path.open("w", newline="") as f:
        fields = [
            "run", "bank_path", "left_token", "right_token",
            "left_center", "right_center", "left_class", "right_class",
            "left_slot", "right_slot", "same_center", "same_class", "cosine",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)

    return {"token_rows": len(token_rows), "pair_rows": len(pair_rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", nargs="*", default=DEFAULT_PATTERNS)
    parser.add_argument("--out_dir", default="/root/autodl-tmp/ecgtwin_prompt_token_super5/diagnostics")
    parser.add_argument("--root", default="/root/autodl-tmp/ecgtwin_prompt_token_super5/prompt_token_runs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    total_banks = 0
    total_tokens = 0
    total_pairs = 0
    token_files: List[Path] = []
    pair_files: List[Path] = []
    for path in _collect_paths(args.patterns):
        counts = export_bank(path, out_dir=out_dir, root=root)
        total_banks += 1
        total_tokens += counts["token_rows"]
        total_pairs += counts["pair_rows"]
        try:
            run_name = path.parent.relative_to(root).as_posix()
        except ValueError:
            run_name = path.parent.as_posix()
        safe_name = f"{run_name}__{path.stem}".replace("/", "__")
        token_files.append(out_dir / f"{safe_name}.token_summary.csv")
        pair_files.append(out_dir / f"{safe_name}.token_cosine.csv")
        print(f"[export] {path}: tokens={counts['token_rows']} pairs={counts['pair_rows']}")

    for files, combined_name in [
        (token_files, "token_summary.csv"),
        (pair_files, "token_cosine.csv"),
    ]:
        combined_path = out_dir / combined_name
        writer = None
        out_f = combined_path.open("w", newline="")
        try:
            for file_path in files:
                with file_path.open() as in_f:
                    reader = csv.DictReader(in_f)
                    if writer is None:
                        writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    for row in reader:
                        writer.writerow(row)
        finally:
            out_f.close()
        print(f"[export] combined {combined_name} -> {combined_path}")
    print(f"[done] banks={total_banks} tokens={total_tokens} pairs={total_pairs} out_dir={out_dir}")


if __name__ == "__main__":
    main()
