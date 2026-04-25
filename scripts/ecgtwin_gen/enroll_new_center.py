"""Enroll a new center into the Style Translator: K unlabeled ECGs → 256-d style vec.

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \\
    scripts/ecgtwin_gen/enroll_new_center.py \\
      --center_dir /root/autodl-tmp/physionet2021/training/georgia \\
      --translator_ckpt /root/autodl-tmp/center_aware_ibe/ckpts/stage0_style_translator_3c/translator_best.pth \\
      --center_name georgia \\
      --n_support 50 \\
      --out /root/autodl-tmp/center_aware_ibe/styles/georgia_style.pt

Stores: {"style_vec": (256,) tensor, "center_name": str, "n_support": int, "records_used": list}
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from methods.ecgtwin_gen.style_translator.prototype_encoder import PrototypeEncoder  # noqa: E402
from scripts.crosscenter_tierM.eval_crosscenter_tierM import scan_center_records  # noqa: E402
from scripts.crosscenter_v2.preprocess_utils import unified_preprocess_to_1000  # noqa: E402
from scripts.ecgtwin_gen.prep_center_dataset import _encode_batch  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center_dir", required=True, help="PN2021 center dir")
    ap.add_argument("--translator_ckpt", required=True,
                    help="translator_best.pth — we only need prototype_encoder state")
    ap.add_argument("--center_name", required=True)
    ap.add_argument("--n_support", type=int, default=50)
    ap.add_argument("--out", required=True, help=".pt output path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) scan records
    print(f"[enroll] scanning {args.center_dir}...")
    record_paths, _ = scan_center_records(args.center_dir)
    print(f"[enroll] found {len(record_paths)} records")
    if len(record_paths) < args.n_support:
        raise SystemExit(f"need ≥ {args.n_support} records; only {len(record_paths)} found")

    # 2) preprocess K records (with retries on failures)
    import wfdb
    signals, used_paths = [], []
    indices = list(range(len(record_paths)))
    random.shuffle(indices)
    t0 = time.time()
    for ri in indices:
        if len(signals) >= args.n_support:
            break
        rp = record_paths[ri]
        try:
            rec = wfdb.rdrecord(rp)
        except Exception:
            continue
        sig = rec.p_signal
        if sig is None or sig.shape[1] < 12:
            continue
        sig_names = [s.strip() for s in rec.sig_name] if getattr(rec, "sig_name", None) else None
        proc = unified_preprocess_to_1000(
            sig.astype(np.float32), fs=rec.fs, source_leads=sig_names,
            target_fs=100, target_len=1000,
            apply_filter=True, apply_zscore=True,
        )
        if proc is None:
            continue
        signals.append(proc.T.astype(np.float32))  # (12, 1000)
        used_paths.append(rp)

    if len(signals) < args.n_support:
        raise SystemExit(f"only {len(signals)} records passed preprocessing")
    print(f"[enroll] preprocessed {len(signals)} records ({time.time() - t0:.1f}s)")

    signals = np.stack(signals, axis=0)  # (K, 12, 1000)

    # 3) VAE encode → (K, 4, 128) latents
    print(f"[enroll] loading ECGTwin VAE on {args.device}...")
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=True, load_text_model=False)
    t0 = time.time()
    latents = _encode_batch(wrapper, signals, batch=args.n_support)  # (K, 4, 128) on CPU
    print(f"[enroll] VAE latents {tuple(latents.shape)} ({time.time() - t0:.1f}s)")
    assert latents.shape == (args.n_support, 4, 128)

    # 4) Load prototype_encoder from translator checkpoint
    print(f"[enroll] loading prototype_encoder from {args.translator_ckpt}")
    ckpt = torch.load(args.translator_ckpt, map_location="cpu")
    proto = PrototypeEncoder()
    proto.load_state_dict(ckpt["prototype_encoder"])
    proto.to(args.device).eval()

    # 5) Forward → style_vec
    with torch.no_grad():
        style_vec = proto(latents.to(args.device))  # (256,)
    style_vec = style_vec.detach().cpu()
    print(f"[enroll] style_vec: shape={tuple(style_vec.shape)}  norm={style_vec.norm().item():.3f}")

    # 6) Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "style_vec": style_vec,
            "center_name": args.center_name,
            "n_support": args.n_support,
            "records_used": [str(Path(p).name) for p in used_paths],
            "translator_ckpt": str(args.translator_ckpt),
        },
        out,
    )
    print(f"[enroll] wrote → {out}")


if __name__ == "__main__":
    main()
