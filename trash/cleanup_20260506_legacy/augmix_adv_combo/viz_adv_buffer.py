"""Visualize generated adv samples from the buffer .npz for sanity check.

Produces one PNG per Tier-M class, each showing 3 adv samples (12-lead each)
stacked vertically for physiological inspection. Samples are picked as those
where the target-dim soft label sits in the accept range [0.50, 0.60].
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from util.ecg_viz import plot_ecg  # noqa: E402

NPZ = "/root/autodl-tmp/adv_buffer_tierM/adv_buffer_n1800_soft.npz"
OUT_DIR = Path("/root/ECG_adv_Gen/outputs/augmix_adv_combo/adv_samples_viz")
CLASSES = ["NSR", "STach", "AF", "IAVB", "LBBB", "RBBB"]
N_PER_CLASS = 3
SEED = 42


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(NPZ, allow_pickle=True)
    signals = d["signals"]   # (N, 12, 1000) float32, 100 Hz canonical leads
    labels = d["labels"]     # (N, 6) float32
    meta = json.loads(str(d["meta"]))
    print(f"[load] {signals.shape[0]} samples from {NPZ}")
    print(f"[load] label scheme: {meta['label_scheme']}")

    rng = np.random.default_rng(SEED)
    summary = []
    for i, cls in enumerate(CLASSES):
        # Target-dim soft label in [0.5, 0.6] = target class for this sample
        target_mask = (labels[:, i] >= 0.5) & (labels[:, i] <= 0.6)
        idx_pool = np.where(target_mask)[0]
        if idx_pool.size == 0:
            print(f"[warn] no samples with target={cls}")
            continue
        pick = rng.choice(idx_pool, size=min(N_PER_CLASS, idx_pool.size), replace=False)
        for j, k in enumerate(pick):
            sig = signals[int(k)]               # (12, 1000)
            lbl = labels[int(k)]                # (6,)
            p_t = float(lbl[i])
            ref_others = [f"{CLASSES[q]}={lbl[q]:.0f}" for q in range(6) if q != i]
            title = (f"target={cls}  victim_p={p_t:.3f}  "
                     f"ref_labels=[{','.join(ref_others)}]")
            out_png = OUT_DIR / f"adv_{cls}_{j:02d}_idx{k}.png"
            plot_ecg(
                signal=sig, sample_rate=100, save_path=out_png,
                title=title, lead_order="ecgtwin", engine="matplotlib",
            )
            summary.append({
                "class": cls, "buffer_idx": int(k),
                "target_soft_label": round(p_t, 4),
                "ref_others": dict(zip(
                    [c for c in CLASSES if c != cls],
                    [float(lbl[q]) for q in range(6) if q != i],
                )),
                "png": str(out_png),
            })
            print(f"  [viz] {cls:<6} sample #{j+1} → {out_png.name}")

    # Overlay: 3 samples from same class stacked (one per row)
    from util.ecg_viz import plot_comparison
    for i, cls in enumerate(CLASSES):
        target_mask = (labels[:, i] >= 0.5) & (labels[:, i] <= 0.6)
        idx_pool = np.where(target_mask)[0]
        if idx_pool.size < 2:
            continue
        pick = rng.choice(idx_pool, size=min(3, idx_pool.size), replace=False)
        sigs = [signals[int(k)] for k in pick]
        lbls_txt = [f"adv#{int(k)} p_t={labels[int(k), i]:.3f}" for k in pick]
        out_png = OUT_DIR / f"overlay_{cls}.png"
        plot_comparison(
            signals=sigs, labels=lbls_txt, sample_rate=100,
            save_path=out_png, mode="overlay", lead_order="ecgtwin",
        )
        print(f"  [overlay] {cls:<6} 3-way → {out_png.name}")

    # Save summary JSON
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps({
        "n_samples_viz": len(summary),
        "per_class": summary,
        "buffer_npz": NPZ,
        "label_scheme": meta["label_scheme"],
        "n_total_in_buffer": int(signals.shape[0]),
    }, indent=2))
    print(f"\n[done] {len(summary)} PNGs + {len(CLASSES)} overlays in {OUT_DIR}")


if __name__ == "__main__":
    main()
