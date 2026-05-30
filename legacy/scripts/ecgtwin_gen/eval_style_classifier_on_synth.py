"""Run pre-trained per-center style classifier on CT-conditioned synth ECGs.

Iter 1.5 corroboration: independent validation that CT v2 conditioned synth
is encoding per-center style. Tests:
  - Generate ~K synth ECGs with CT_v2_extra hooked → run through style
    classifier → top-1 predicted center should be 'cpsc_2018_extra'
    significantly above chance (1/7 = 14.3%).
  - Same for CT_v2_nin → 'ningbo', CT_v2_geo → 'georgia'.

If classifier predicts 'extra' on extra-hooked synth at >38% (matching its
real-test accuracy), CT v2 has fully transferred per-center vendor style.
If at chance (~14%), CT v2 doesn't encode style despite signal moves.

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/eval_style_classifier_on_synth.py \
      --classifier_dir /root/autodl-tmp/per_center_style_classifier \
      --synth_root /root/autodl-tmp/synth_anchored_super5_v14_ctv2/synth_latents \
      --centers extra:cpsc_2018_extra,nin:ningbo,geo:georgia \
      --output /root/autodl-tmp/ct_v2_smoking_gun_sphere5/style_classifier_eval.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model" / "DeepECG" / "notebooks"))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402

PN2021_CENTERS = [
    "chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra",
    "georgia", "ningbo", "ptb", "st_petersburg_incart",
]


def load_classifier(ckpt_dir: str, num_classes: int = 7, device: str = "cuda:0"):
    model = EfficientNet1DV2(
        variant="s_v2", input_channels=12, num_classes=num_classes,
        activation="leaky_relu", stochastic_depth_prob=0.304,
        dropout_rate=0.0, use_se=True, norm_type="batch",
    )
    sd = torch.load(Path(ckpt_dir) / "best_model.pt", map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


@torch.no_grad()
def classify_signals(model, signals_array, device, batch_size=64, crop_len=250):
    """signals: (N, 12, 1000) PTBXL order, z-scored. Center-crop to 250."""
    N = signals_array.shape[0]
    start = (1000 - crop_len) // 2
    end = start + crop_len
    crops = signals_array[:, :, start:end]  # (N, 12, 250)

    all_probs = []
    for i in range(0, N, batch_size):
        x = torch.from_numpy(crops[i:i+batch_size]).float().to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)  # (N, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier_dir", required=True)
    ap.add_argument("--synth_root", required=True,
                    help="Directory containing {center}_k200.npz synth files")
    ap.add_argument("--centers", required=True,
                    help="Comma-separated short:full pairs, "
                         "e.g. 'extra:cpsc_2018_extra,nin:ningbo,geo:georgia'")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap each synth file to N samples (None = all)")
    args = ap.parse_args()

    pairs = []
    for s in args.centers.split(","):
        short, full = s.split(":")
        pairs.append((short.strip(), full.strip()))

    model = load_classifier(args.classifier_dir, num_classes=7, device=args.device)
    print(f"[classifier] loaded from {args.classifier_dir}")
    print(f"[classifier] center_names: {PN2021_CENTERS}")
    print()

    results = {"per_center": {}, "expected_centers": {}, "config": vars(args)}
    chance = 1.0 / len(PN2021_CENTERS)

    for short, full in pairs:
        synth_path = Path(args.synth_root) / f"{short}_k200.npz"
        if not synth_path.exists():
            print(f"[skip] {synth_path} not found")
            continue

        d = np.load(synth_path)
        signals = d["signals"]
        if args.max_samples and len(signals) > args.max_samples:
            idx = np.random.RandomState(42).choice(
                len(signals), args.max_samples, replace=False)
            signals = signals[idx]
        print(f"[{short}] synth loaded: {signals.shape}, expected center: {full}")

        probs = classify_signals(model, signals, args.device)
        preds = probs.argmax(axis=1)
        expected_idx = PN2021_CENTERS.index(full)

        # Per-center prediction breakdown
        per_pred_count = {}
        for i, c in enumerate(PN2021_CENTERS):
            per_pred_count[c] = int((preds == i).sum())
        top1_acc = float((preds == expected_idx).mean())
        mean_prob_expected = float(probs[:, expected_idx].mean())
        # Top-3 accuracy
        top3_idx = np.argsort(probs, axis=1)[:, -3:]
        top3_acc = float((top3_idx == expected_idx).any(axis=1).mean())

        print(f"  top-1 acc on {full}: {top1_acc:.4f}  "
              f"(chance={chance:.4f}, classifier real-test acc varied)")
        print(f"  top-3 acc on {full}: {top3_acc:.4f}")
        print(f"  mean prob({full}): {mean_prob_expected:.4f}")
        print(f"  prediction breakdown:")
        for c, n in sorted(per_pred_count.items(), key=lambda x: -x[1]):
            pct = n / len(preds) * 100
            marker = " ←" if c == full else ""
            print(f"    {c:<22} {n:>4d}  ({pct:.1f}%){marker}")

        results["per_center"][short] = {
            "expected_center": full,
            "n_samples": len(signals),
            "top1_acc": top1_acc,
            "top3_acc": top3_acc,
            "mean_prob_expected": mean_prob_expected,
            "prediction_breakdown": per_pred_count,
            "mean_per_center_prob": probs.mean(axis=0).tolist(),
        }
        print()

    # Aggregate
    n_centers_above_chance = sum(
        1 for v in results["per_center"].values()
        if v["top1_acc"] > chance * 1.5
    )
    n_centers_above_25pct = sum(
        1 for v in results["per_center"].values()
        if v["top1_acc"] > 0.25
    )
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  centers above chance × 1.5 ({chance*1.5:.3f}): "
          f"{n_centers_above_chance}/{len(pairs)}")
    print(f"  centers above 25% top-1: {n_centers_above_25pct}/{len(pairs)}")
    print(f"  chance: {chance:.4f}")
    results["summary"] = {
        "n_above_chance_x1.5": n_centers_above_chance,
        "n_above_25pct": n_centers_above_25pct,
        "chance": chance,
        "n_centers_evaluated": len(pairs),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] saved to {args.output}")


if __name__ == "__main__":
    main()
