#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from apps.streamlit_ecg_demo.services.classifier_backend import (
    DEFAULT_CKPT,
    DEFAULT_ONNX,
    build_efficientnet_super5,
)


def load_model(ckpt: str, device: str):
    model = build_efficientnet_super5()
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {str(k).removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_ONNX)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(args.ckpt, device)
    dummy = torch.randn(1, 12, 1000, device=device)
    with torch.no_grad():
        ref = model(dummy).detach().cpu()
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["ecg"],
        output_names=["logits"],
        dynamic_axes={"ecg": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=False,
    )
    meta = {
        "checkpoint": args.ckpt,
        "onnx": str(out),
        "input_shape": [1, 12, 1000],
        "output_shape": list(ref.shape),
        "opset": args.opset,
    }
    meta_path = out.with_suffix(".export.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
