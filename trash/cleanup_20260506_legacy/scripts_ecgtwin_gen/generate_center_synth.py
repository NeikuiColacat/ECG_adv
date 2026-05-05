"""Generate N synthetic ECGs per class using a trained center token.

Supports two label schemes:
  - tierM  (default; 6 classes — Tier-M, backward compatible)
  - super5 (5 classes — CD/HYP/MI/NORM/STTC; Plan Rev 8)

Inputs:
  - A center .pt file from prep_center_dataset[_super5].py (provides reference
    latents per-class, plus per-record / per-class text_embed tensors).
  - (Optional) A trained center-token checkpoint. **Plan Rev 12 (2026-04-27)
    drops CenterToken** — default omits --token_ckpt for vanilla ECGTwin + IBE
    base_vector path. Only pass --token_ckpt for ablation/historical
    reproduction; see memory/centertoken_actual_mechanism.md.

Pipeline per class:
  1. Pick a reference sample from that class's pool (cycle with RNG).
  2. prepare_conditions with the per-record text_embed (super5) or shared
     canonical text_embed (tierM).
  3. ddpm_sample (if --token_ckpt given, hooks inject token into DiT AdaLN
     driver; otherwise vanilla DiT forward).
  4. decode_latent → (B, 1024, 12) ECGTwin order.
  5. interp 1024→1000, lead-swap back to PTBXL, per-sample z-score.

Output .npz:
  - signals: (N_total, 12, 1000) float32 in PTBXL lead order
  - labels:  (N_total, num_classes) float32 one-hot
  - center_name: str

If --save_latent is set, also writes {out}.latent.npz with:
  - latents: (N_total, 4, 128) float32   (pre-decode VAE latent from DDPM)
  - labels:  (N_total, num_classes) float32 (same as signals)
  - center_name: str

Usage (Plan Rev 12, no CenterToken):
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/generate_center_synth.py \
      --scheme super5 \
      --ref_pt /root/autodl-tmp/center_token_super5/extra_k200.pt \
      --center_name cpsc_2018_extra \
      --out /root/autodl-tmp/synth_anchored_super5/synth_latents/extra_k200.npz \
      --n_per_class 100 --batch_size 100 --num_steps 50 --save_latent

Legacy (with CT hook, for ablation reproductions only):
  ... --token_ckpt /path/to/center_token_best.pth ...
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

from methods.ecgtwin_gen.center_token.model import CenterToken, CenterTokenPerBlock  # noqa: E402
from scripts.crosscenter_v2.label_alignment_v2 import TIER_M  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402

SCHEME_TO_CLASSES = {
    "tierM":  TIER_M,
    "super5": CLASS_NAMES_SUPER5,
}

# Plan Rev 11 / Rev 13.2 (2026-04-27): for super5, generate ONLY these classes.
# HYP/CD are excluded because the digital-GT validation
# (`docs/ecgtwin_super5_digital_gt_validation.md`) shows ECGTwin generation
# fails 0/3 best-cell medical thresholds for those classes. The eval-time
# class scheme remains 5-class (CD/HYP/MI/NORM/STTC) — only the synth
# generation scope is narrowed. Labels still have 5 dims; CD/HYP positions
# stay zero in the synth pool.
SUPER5_GEN_SUBSET = {"NORM", "MI", "STTC"}


def _register_hooks(wrapper, center_token):
    """Single 256-d CT (v1) — same vector added to all blocks."""
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


def _register_hooks_per_block(wrapper, center_token):
    """Per-block CT (v2) — each block gets its own 256-d vector via index closure."""
    hooks = []
    for i, block in enumerate(wrapper.noise_predictor.blocks):
        def make_fn(idx):
            def fn(module, args):
                x, c, c2 = args[0], args[1], args[2]
                rest = args[3:]
                c = c + center_token(idx, c.size(0))
                return (x, c, c2) + rest
            return fn
        hooks.append(block.register_forward_pre_hook(make_fn(i)))
    return hooks


def _index_by_class(samples: List[dict], class_names: List[str],
                    use_super5_multi_hot: bool,
                    label_key: str = "diagnostic_class") -> Dict[str, List[int]]:
    """Map class_name → list of sample indices that are positive for that class.

    For Tier-M scheme each ref has a single `diagnostic_class` string.
    For Super5 scheme we honor multi-positive labels via `super5_multi_hot`,
    so a single CD+STTC ref can serve both class loops (Issue #13).
    """
    out: Dict[str, List[int]] = {c: [] for c in class_names}
    for i, s in enumerate(samples):
        if use_super5_multi_hot and "super5_multi_hot" in s["label"]:
            mh = s["label"]["super5_multi_hot"]
            for j, c in enumerate(class_names):
                if float(mh[j]) > 0.5:
                    out[c].append(i)
        else:
            cls = s["label"].get(label_key)
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
    ap.add_argument("--scheme", default="tierM", choices=("tierM", "super5"),
                    help="Class scheme; default tierM for backward compat")
    ap.add_argument("--ref_pt", required=True)
    ap.add_argument("--token_ckpt", default=None,
                    help="Optional CenterToken ckpt. Plan Rev 12 (2026-04-27) "
                         "drops CT v1 — default is None (no hook). Pass for "
                         "either v1 ablation reproductions or v2 per-block CT.")
    ap.add_argument("--token_arch", default="single",
                    choices=("single", "per_block"),
                    help="single (v1, broadcast 256-d) | "
                         "per_block (v2, CenterTokenPerBlock with one 256-d per DiT block)")
    ap.add_argument("--center_name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Explicit generation classes. Default preserves legacy scope.")
    ap.add_argument("--ref_index_mode", choices=("auto", "diagnostic_class", "multi_hot"),
                    default="auto",
                    help="How refs are indexed for super5. auto preserves legacy behavior.")
    ap.add_argument("--save_ref_trace", action="store_true",
                    help="Save per-generated-sample reference metadata in {out}.ref_trace.json")
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_latent", action="store_true",
                    help="Also save (B, 4, 128) DDPM latents to {out}.latent.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    class_names = list(SCHEME_TO_CLASSES[args.scheme])
    num_classes = len(class_names)
    if args.classes:
        unknown = [c for c in args.classes if c not in class_names]
        if unknown:
            raise ValueError(f"unknown classes {unknown}; valid={class_names}")
        generation_classes = list(args.classes)
    elif args.scheme == "super5":
        generation_classes = [c for c in class_names if c in SUPER5_GEN_SUBSET]
    else:
        generation_classes = list(class_names)

    samples = torch.load(args.ref_pt, map_location="cpu", weights_only=False)
    print(f"[gen] loaded {len(samples)} reference samples from {args.ref_pt} "
          f"(scheme={args.scheme}, {num_classes} classes)")

    if args.ref_index_mode == "auto":
        use_multi_hot = args.scheme == "super5"
    else:
        use_multi_hot = args.ref_index_mode == "multi_hot"
    class_idx = _index_by_class(samples, class_names,
                                use_super5_multi_hot=use_multi_hot)
    print(f"[gen] per-class reference counts:")
    for c, idx in class_idx.items():
        print(f"  {c}: {len(idx)} refs")

    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)

    hooks = []
    if args.token_ckpt is not None:
        ckpt = torch.load(args.token_ckpt, map_location="cpu", weights_only=False)
        if args.token_arch == "single":
            center_token = CenterToken(dim=256)
            center_token.load_state_dict(ckpt["center_token"])
            center_token.to(args.device).eval()
            print(f"[gen] CenterToken (v1, single) loaded "
                  f"(norm={center_token.norm:.4f})")
            hooks = _register_hooks(wrapper, center_token)
        elif args.token_arch == "per_block":
            sd = ckpt["center_token"]
            n_blocks = len([k for k in sd.keys() if k.startswith("embeddings.")])
            dim = sd["embeddings.0"].shape[0]
            center_token = CenterTokenPerBlock(dim=dim, num_blocks=n_blocks)
            center_token.load_state_dict(sd)
            center_token.to(args.device).eval()
            print(f"[gen] CenterTokenPerBlock (v2, per-block) loaded: "
                  f"{n_blocks} blocks × {dim}d, "
                  f"norms=[{','.join(f'{n:.2f}' for n in center_token.norms)}], "
                  f"total={center_token.total_norm:.3f}")
            hooks = _register_hooks_per_block(wrapper, center_token)
        else:
            raise ValueError(f"unknown --token_arch {args.token_arch}")
        print(f"[gen] registered {len(hooks)} forward_pre_hooks on DiT blocks")
    else:
        print("[gen] CenterToken disabled (Plan Rev 12 default). "
              "Vanilla ECGTwin + IBE base_vector path.")

    all_signals = []
    all_latents = []   # only filled if --save_latent
    all_labels = []
    ref_trace = []

    t_start = time.time()
    for cls in generation_classes:
        cls_i = class_names.index(cls)
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
            ref_label_full = ref_sample["label"]
            ref_latent = ref_sample["data"]  # (4, 128)
            ref_label = {
                "hr": float(ref_sample["label"].get("hr", 75.0)),
                "age": float(ref_sample["label"].get("age", 60.0)),
                "sex": ref_sample["label"].get("sex", "U"),
                "text_embed": ref_sample["label"]["text_embed"],
            }
            target_text_embed = ref_sample["label"]["text_embed"]  # per-record / per-class embedding

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
                if args.save_latent:
                    all_latents.append(latent_gen.detach().cpu().numpy())
                ecg_et = wrapper.decode_latent(latent_gen)  # (bs, 1024, 12)
                signals = _postprocess(ecg_et)              # (bs, 12, 1000)
            all_signals.append(signals.cpu().numpy())
            lbl = np.zeros((bs, num_classes), dtype=np.float32)
            lbl[:, cls_i] = 1.0
            all_labels.append(lbl)
            if args.save_ref_trace:
                ref_trace.extend([
                    {
                        "class": cls,
                        "ref_record_id": str(ref_label_full.get("record_id", ref_label_full.get("ecg_id", ""))),
                        "ref_ecg_id": ref_label_full.get("ecg_id"),
                        "ref_patient_id": ref_label_full.get("patient_id"),
                        "ref_source_index": ref_label_full.get("source_index"),
                        "ref_fold": ref_label_full.get("strat_fold"),
                        "ref_diagnostic_class": ref_label_full.get("diagnostic_class"),
                        "ref_text": ref_label_full.get("text"),
                    }
                    for _ in range(bs)
                ])
            produced += bs

            if (b + 1) % 5 == 0 or b == n_batches - 1:
                print(f"  [{cls}] {produced}/{args.n_per_class} samples, "
                      f"elapsed {time.time() - t0:.1f}s", flush=True)
        print(f"[gen] [{cls}] done in {time.time() - t0:.1f}s")

    for h in hooks:
        h.remove()

    signals = np.concatenate(all_signals, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.float32)
    print(f"[gen] total: signals {signals.shape}, labels {labels.shape}, "
          f"elapsed {time.time() - t_start:.1f}s")

    assert np.isfinite(signals).all(), "non-finite in generated signals!"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, signals=signals, labels=labels, center_name=args.center_name)
    print(f"[gen] wrote → {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    if args.save_ref_trace:
        import json
        trace_out = out.with_suffix(".ref_trace.json")
        with trace_out.open("w") as f:
            json.dump(ref_trace, f, indent=2, default=str)
        print(f"[gen] wrote ref trace → {trace_out}")

    if args.save_latent:
        latents = np.concatenate(all_latents, axis=0).astype(np.float32)
        assert latents.shape[1:] == (4, 128), f"bad latent shape: {latents.shape}"
        latent_out = out.with_suffix(".latent.npz")
        np.savez_compressed(latent_out,
                            latents=latents, labels=labels, center_name=args.center_name)
        print(f"[gen] wrote latents → {latent_out}  "
              f"({latent_out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
