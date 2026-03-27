#!/usr/bin/env python
"""
Plot comparison ECGs: real PTBXL MI vs synthetic adversarial samples (fixed code).
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ecg_plot

OUTPUT_DIR = Path("/root/ECG_adv_Gen/outputs/mi_experiment/ecg_plots_fixed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_INDEX = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def plot_and_save(ecg_np, filename, sample_rate=250):
    """Plot ECG and save as PNG."""
    ecg_plot.plot(ecg_np, sample_rate=sample_rate, lead_index=LEAD_INDEX, columns=1, row_height=4)
    ecg_plot.save_as_png(file_name=filename, path=str(OUTPUT_DIR) + "/")
    print(f"  Saved → {OUTPUT_DIR / (filename + '.png')}")


def main():
    # Load generated samples
    for prompt_key in ["MIMIC_MI_inferior", "MIMIC_MI_acute", "MIMIC_MI_acute_inferior", "MIMIC_MI_inferolateral"]:
        path = Path(f"/root/ECG_adv_Gen/outputs/mi_experiment/{prompt_key}.pt")
        if not path.exists():
            print(f"[SKIP] {path}")
            continue

        data = torch.load(path, map_location="cpu")
        ecg = data["ecg"]  # (N, 12, 2500)
        probs = data["probs"]  # (N, 77)

        print(f"\n{prompt_key}: {ecg.shape[0]} samples")
        print(f"  ECG amplitude: min={ecg.min():.3f}, max={ecg.max():.3f}, std={ecg.std():.4f}")

        # Plot first 3 samples
        for i in range(min(3, ecg.shape[0])):
            ecg_np = ecg[i].numpy()  # (12, 2500)
            plot_and_save(ecg_np, f"synthetic_{prompt_key}_{i}")

    # Load and plot real PTBXL MI samples for comparison
    print("\n--- Real PTBXL MI samples ---")
    from util.get_PTBXL import get_ecg_dataset
    import torch.nn.functional as F

    train_loader, _, test_loader, scaler = get_ecg_dataset()

    # Get a batch of MI samples from test set
    for batch_ecg, batch_labels in test_loader:
        # batch_ecg: (B, 1000, 12), batch_labels: (B,) superdiag class indices
        # MI = class 2
        mi_mask = (batch_labels == 2)
        if mi_mask.sum() >= 3:
            mi_ecg = batch_ecg[mi_mask][:3]  # (3, 1000, 12)
            # Transpose to (B, 12, 1000), resample to 2500
            mi_ecg = mi_ecg.transpose(1, 2)  # (3, 12, 1000)
            mi_ecg = F.interpolate(mi_ecg, size=2500, mode='linear', align_corners=True)

            for i in range(3):
                ecg_np = mi_ecg[i].numpy()  # (12, 2500)
                print(f"  Real MI sample {i}: min={ecg_np.min():.3f}, max={ecg_np.max():.3f}")
                plot_and_save(ecg_np, f"real_MI_{i}")
            break

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
