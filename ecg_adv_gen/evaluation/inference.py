from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def infer_dataset(model, loader, device):
    model.eval()
    all_labels, all_logits = [], []
    for signals, labels in loader:
        signals = signals.to(device)
        with torch.cuda.amp.autocast(enabled="cuda" in str(device)):
            logits = model(signals)
        all_labels.append(labels.detach().cpu().numpy())
        all_logits.append(logits.float().cpu().numpy())
    y_true = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)
    y_score = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    return y_true, y_score
