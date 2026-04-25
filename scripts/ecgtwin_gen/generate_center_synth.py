"""Generate N synthetic ECGs per Tier-M class using a trained center token.

Inputs:
  - A center .pt file from prep_center_dataset.py (provides reference latents
    per-class, plus per-class canonical text_embed tensors).
  - A trained center-token checkpoint (center_token_best.pth).

Pipeline per class:
  1. Pick a reference sample from that class's pool (cycle with RNG).
  2. prepare_conditions with the canonical class text_embed.
  3. ddpm_sample (hooks inject center_token into DiT AdaX c path).
  4. decode_latent → (B, 1024, 12) ECGTwin order.
  5. interp 1024→1000, lead-swap back to PTBXL, per-sample z-score.

Output .npz:
  - signals: (N_total, 12, 1000) float32 in PTBXL lead order
  - labels:  (N_total, 6) float32 one-hot Tier-M
  - center_name: str

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/generate_center_synth.py \
      --ref_pt /root/autodl-tmp/center_token/cpsc_2018_extra.pt \
      --token_ckpt /root/autodl-tmp/center_token/cpsc_2018_extra_ckpt/center_token_best.pth \
      --center_name cpsc_2018_extra \
      --out /root/autodl-tmp/center_token/cpsc_2018_extra_synth.npz \
      --n_per_class 1000 --batch_size 50 --num_steps 50
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from methods.ecgtwin_gen.center_token.model import CenterToken  # noqa: E402
from scripts.crosscenter_v2.label_alignment_v2 import TIER_M  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402


def _register_hooks(wrapper, center_token):
    def make_hook():
        def hook_fn(module, args):
            x, c, c2 = args[0], args[1], args[2]
            rest = args[3:]
            c = c + center_token(c.size(0))
            return (x, c, c2) + rest
        return hook_fn

    hooks = []
    for block in wrapper.noise_predictor.blocks:
        hooks.append(block.register_forward_pre_hook(make_hook()))
    return hooks


def _index_by_class(samples: List[dict]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {c: [] for c in TIER_M}
    for i, s in enumerate(samples):
        cls = s["label"].get("diagnostic_class")
        if cls in out:
            out[cls].append(i)
    return out


def _postprocess(ecg_et_tc: torch.Tensor) -> torch.Tensor:
    """(B, 1024, 12) ECGTwin order → (B, 12, 1000) PTBXL order, per-sample z-scored."""
    B = ecg_et_tc.shape[0]
    ecg_ct_1024 = ecg_et_tc.transpose(1, 2)                             # (B, 12, 1024)
    ecg_ct_1000 = F.interpolate(ecg_ct_1024, size=1000, mode="linear", align_corners=True)
    ecg_ptbxl = ecg_ct_1000[:, ECGTWIN_TO_PTBXL_INDICES, :]              # involutive swap
    flat = ecg_ptbxl.reshape(B, -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp(min=1e-8)
    ecg_norm = (ecg_ptbxl - mean.unsqueeze(-1)) / std.unsqueeze(-1)
    return ecg_norm.to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_pt", required=True)
    ap.add_argument("--token_ckpt", required=True)
    ap.add_argument("--center_name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    samples = torch.load(args.ref_pt, map_location="cpu")
    print(f"[gen] loaded {len(samples)} reference samples from {args.ref_pt}")

    class_idx = _index_by_class(samples)
    print(f"[gen] per-class reference counts:")
    for c, idx in class_idx.items():
        print(f"  {c}: {len(idx)} refs")

    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)

    center_token = CenterToken(dim=256)
    ckpt = torch.load(args.token_ckpt, map_location="cpu")
    center_token.load_state_dict(ckpt["center_token"])
    center_token.to(args.device)
    center_token.eval()
    print(f"[gen] center_token loaded (norm={center_token.norm:.4f})")

    hooks = _register_hooks(wrapper, center_token)
    print(f"[gen] registered {len(hooks)} forward_pre_hooks on DiT blocks")

    all_signals = []
    all_labels = []

    t_start = time.time()
    for cls_i, cls in enumerate(TIER_M):
        refs = class_idx[cls]
        if not refs:
            print(f"[gen] [{cls}] no reference samples — skipping")
            continue

        n_batches = (args.n_per_class + args.batch_size - 1) // args.batch_size
        produced = 0
        t0 = time.time()
        for b in range(n_batches):
            bs = min(args.batch_size, args.n_per_class - produced)
            ref_sample = samples[int(rng.choice(refs))]
            ref_latent = ref_sample["data"]  # (4, 128)
            ref_label = {
                "hr": float(ref_sample["label"].get("hr", 75.0)),
                "age": float(ref_sample["label"].get("age", 60.0)),
                "sex": ref_sample["label"].get("sex", "U"),
                "text_embed": ref_sample["label"]["text_embed"],
            }
            target_text_embed = ref_sample["label"]["text_embed"]  # canonical class embedding

            conditions = wrapper.prepare_conditions(
                ref_latent=ref_latent,
                ref_label=ref_label,
                batch_size=bs,
                target_text_embed=target_text_embed,
                target_hr=ref_label["hr"],
                target_age=ref_label["age"],
            )
            with torch.no_grad():
                latent_gen = wrapper.ddpm_sample(
                    conditions=conditions,
                    batch_size=bs,
                    num_inference_steps=args.num_steps,
                )
                ecg_et = wrapper.decode_latent(latent_gen)  # (bs, 1024, 12)
                signals = _postprocess(ecg_et)              # (bs, 12, 1000)
            all_signals.append(signals.cpu().numpy())
            lbl = np.zeros((bs, 6), dtype=np.float32)
            lbl[:, cls_i] = 1.0
            all_labels.append(lbl)
            produced += bs

            if (b + 1) % 5 == 0 or b == n_batches - 1:
                print(f"  [{cls}] {produced}/{args.n_per_class} samples, elapsed {time.time() - t0:.1f}s", flush=True)
        print(f"[gen] [{cls}] done in {time.time() - t0:.1f}s")

    for h in hooks:
        h.remove()

    signals = np.concatenate(all_signals, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.float32)
    print(f"[gen] total: signals {signals.shape}, labels {labels.shape}, elapsed {time.time() - t_start:.1f}s")

    assert np.isfinite(signals).all(), "non-finite in generated signals!"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, signals=signals, labels=labels, center_name=args.center_name)
    print(f"[gen] wrote → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
