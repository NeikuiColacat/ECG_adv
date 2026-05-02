"""Compose a prompt-token bank from class-specific checkpoint banks.

This is a small utility for gate-aware token checkpoint selection. A common
failure mode is that the best training step differs by class, so this tool
copies selected `(center, class)` token vectors from checkpoint banks into a
base bank.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def parse_class_bank(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"bad --class_bank item {item!r}; expected CLASS=/path/bank.pt")
        cls, path = item.split("=", 1)
        cls = cls.strip().upper()
        if not cls:
            raise ValueError(f"empty class in --class_bank item {item!r}")
        out[cls] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_bank", required=True)
    ap.add_argument("--out_bank", required=True)
    ap.add_argument("--center", required=True)
    ap.add_argument("--class_bank", nargs="+", required=True,
                    help="Items like MI=/path/prompt_token_bank_step01000.pt")
    args = ap.parse_args()

    base = torch.load(args.base_bank, map_location="cpu", weights_only=False)
    centers = list(base["centers"])
    classes = list(base["class_names"])
    if args.center not in centers:
        raise ValueError(f"center {args.center!r} not in base centers={centers}")
    center_idx = centers.index(args.center)

    if "embeddings" not in base:
        raise ValueError("only direct/legacy banks with 'embeddings' are supported")
    base_emb = base["embeddings"].clone()
    if base_emb.dim() not in {3, 4}:
        raise ValueError(f"unsupported base embeddings shape {tuple(base_emb.shape)}")

    replacements = parse_class_bank(args.class_bank)
    provenance = dict(base.get("composition_provenance", {}))
    provenance.setdefault("base_bank", args.base_bank)
    provenance.setdefault("replacements", [])

    for cls, bank_path in replacements.items():
        if cls not in classes:
            raise ValueError(f"class {cls!r} not in base classes={classes}")
        cls_idx = classes.index(cls)
        other = torch.load(bank_path, map_location="cpu", weights_only=False)
        if list(other["centers"]) != centers:
            raise ValueError(f"{bank_path} centers differ from base")
        if list(other["class_names"]) != classes:
            raise ValueError(f"{bank_path} classes differ from base")
        other_emb = other["embeddings"].float()
        if other_emb.shape != base_emb.shape:
            raise ValueError(
                f"{bank_path} embeddings shape {tuple(other_emb.shape)} "
                f"!= base {tuple(base_emb.shape)}"
            )
        base_emb[center_idx, cls_idx] = other_emb[center_idx, cls_idx]
        provenance["replacements"].append({
            "center": args.center,
            "class": cls,
            "source_bank": bank_path,
        })
        print(f"[compose] {args.center} {cls} <- {bank_path}")

    base["embeddings"] = base_emb
    base["composition_provenance"] = provenance
    base["version"] = str(base.get("version", "prompt_token_bank")) + "_composed"

    out_path = Path(args.out_bank)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out_path)
    print(f"[compose] wrote {out_path}")


if __name__ == "__main__":
    main()
