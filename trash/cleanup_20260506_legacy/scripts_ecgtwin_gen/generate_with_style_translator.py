"""Generate synthetic ECGs using a Style Translator-conditioned base_vector.

Pipeline per class:
  1. pick ref sample (cycle through class's reference pool).
  2. prepare_conditions → base_vector (from vanilla IBE).
  3. apply style fusion:  base_vector' = base_vector + style_fusion([base_vector, style_vec])
  4. ddpm_sample with replaced base_vector, decode, postprocess.

Modes:
  --style_vec_path  : use enrolled style vec (e.g. georgia_style.pt)
  --null_style      : skip style fusion (== vanilla ECGTwin baseline)

Output .npz: signals (N,12,1000), labels (N,6), center_name
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

from methods.ecgtwin_gen.style_translator.model import StyleTranslatorIBE  # noqa: E402
from methods.ecgtwin_gen.style_translator.prototype_encoder import PrototypeEncoder  # noqa: E402
from scripts.crosscenter_v2.label_alignment_v2 import TIER_M  # noqa: E402
from scripts.ecgtwin_gen.generate_center_synth import _postprocess  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402


def _index_by_class(samples: List[dict]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {c: [] for c in TIER_M}
    for i, s in enumerate(samples):
        cls = s["label"].get("diagnostic_class")
        if cls in out:
            out[cls].append(i)
    return out


def _build_style_translator(wrapper, translator_ckpt_path: str, device: str):
    """Wrap wrapper.ibe_model with StyleTranslatorIBE and load trained state."""
    ckpt = torch.load(translator_ckpt_path, map_location="cpu")
    proto = PrototypeEncoder()
    proto.load_state_dict(ckpt["prototype_encoder"])
    sibe = StyleTranslatorIBE(
        base_ibe=wrapper.ibe_model, prototype_encoder=proto, embed_dim=256,
    ).to(device).eval()
    # load style_fusion weights
    sibe.style_fusion.load_state_dict(ckpt["style_fusion"])
    return sibe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_pt", required=True, help="reference .pt (e.g., cpsc_2018_extra.pt)")
    ap.add_argument("--translator_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--center_name", required=True, help="label for output (e.g., 'target_style')")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--style_vec_path", default=None, help="enrolled style vec .pt")
    group.add_argument("--null_style", action="store_true", help="bypass style fusion (baseline)")
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load refs
    samples = torch.load(args.ref_pt, map_location="cpu")
    print(f"[gen] loaded {len(samples)} references from {args.ref_pt}")
    class_idx = _index_by_class(samples)
    for c, idx in class_idx.items():
        print(f"  {c}: {len(idx)} refs")

    # Load ECGTwin (DiT + VAE + IBE)
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)

    # Build StyleTranslator (unless null-style mode)
    style_vec = None
    sibe = None
    if not args.null_style:
        sibe = _build_style_translator(wrapper, args.translator_ckpt, args.device)
        sv_data = torch.load(args.style_vec_path, map_location="cpu")
        style_vec = sv_data["style_vec"].to(args.device)
        print(f"[gen] loaded style_vec from {args.style_vec_path}  (center={sv_data['center_name']}, norm={style_vec.norm().item():.3f})")
    else:
        print("[gen] NULL STYLE MODE — skipping style fusion (baseline)")

    all_signals, all_labels = [], []
    t_start = time.time()

    for cls_i, cls in enumerate(TIER_M):
        refs = class_idx[cls]
        if not refs:
            print(f"[gen] [{cls}] no refs — skip")
            continue

        n_batches = (args.n_per_class + args.batch_size - 1) // args.batch_size
        produced = 0
        t0 = time.time()
        for b in range(n_batches):
            bs = min(args.batch_size, args.n_per_class - produced)
            ref = samples[int(rng.choice(refs))]
            ref_latent = ref["data"]  # (4, 128)
            ref_label = {
                "hr": float(ref["label"].get("hr", 75.0)),
                "age": float(ref["label"].get("age", 60.0)),
                "sex": ref["label"].get("sex", "U"),
                "text_embed": ref["label"]["text_embed"],
            }
            target_text_embed = ref["label"]["text_embed"]

            conditions = wrapper.prepare_conditions(
                ref_latent=ref_latent, ref_label=ref_label, batch_size=bs,
                target_text_embed=target_text_embed,
                target_hr=ref_label["hr"], target_age=ref_label["age"],
            )

            # Apply style fusion to base_vector (overwrite conditions['base_vector']).
            if sibe is not None and style_vec is not None:
                base_v = conditions["base_vector"]  # (bs, 256)
                with torch.no_grad():
                    style_b = style_vec.unsqueeze(0).expand(base_v.shape[0], -1)
                    delta = sibe.style_fusion(torch.cat([base_v, style_b], dim=-1))
                    conditions["base_vector"] = base_v + delta

            with torch.no_grad():
                latent_gen = wrapper.ddpm_sample(
                    conditions=conditions, batch_size=bs,
                    num_inference_steps=args.num_steps,
                )
                ecg_et = wrapper.decode_latent(latent_gen)  # (bs, 1024, 12)
                signals = _postprocess(ecg_et)               # (bs, 12, 1000)

            all_signals.append(signals.cpu().numpy())
            lbl = np.zeros((bs, 6), dtype=np.float32)
            lbl[:, cls_i] = 1.0
            all_labels.append(lbl)
            produced += bs

            if (b + 1) % 5 == 0 or b == n_batches - 1:
                print(f"  [{cls}] {produced}/{args.n_per_class}, {time.time() - t0:.1f}s", flush=True)
        print(f"[gen] [{cls}] done in {time.time() - t0:.1f}s")

    signals = np.concatenate(all_signals, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.float32)
    assert np.isfinite(signals).all(), "non-finite signal!"
    print(f"[gen] total signals {signals.shape}, labels {labels.shape}, "
          f"elapsed {time.time() - t_start:.1f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, signals=signals, labels=labels, center_name=args.center_name)
    print(f"[gen] wrote → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
