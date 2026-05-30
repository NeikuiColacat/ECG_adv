"""Fine-tune EfficientNet1DV2 on K-shot PN2021 target-center real + synth data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "model" / "DeepECG" / "notebooks"))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402
from scripts.crosscenter_v2.preprocess_utils import crop_signal_tc, unified_preprocess_to_1000  # noqa: E402
from scripts.triple_labels.eval_crosscenter import parse_header_snomed  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5, NUM_SUPER5, snomed_list_to_super5  # noqa: E402
from scripts.triple_labels.train_ptbxl import (  # noqa: E402
    SynthNPZDataset,
    compute_macro_auroc_auprc,
    compute_pos_weight,
    masked_bce_with_logits,
)


class CenterRealDataset(Dataset):
    def __init__(self, records: list[dict], crop_len: int = 250, mode: str = "train"):
        self.signals = [r["signal"] for r in records]
        self.labels = np.stack([r["label"] for r in records]).astype(np.float32)
        self.record_ids = [r["record_id"] for r in records]
        self.crop_len = crop_len
        self.mode = mode

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        crop = crop_signal_tc(self.signals[idx], self.crop_len, mode="random" if self.mode == "train" else "center")
        return (
            torch.from_numpy(np.ascontiguousarray(crop.T)).float(),
            torch.from_numpy(self.labels[idx]).float(),
        )


def scan_center_records(center: str, pn2021_root: str, allowed_ids: set[str] | None = None) -> list[tuple[str, str]]:
    center_dir = os.path.join(pn2021_root, "training", center)
    out = []
    for root, _, files in os.walk(center_dir):
        for name in files:
            if not name.endswith(".hea"):
                continue
            rec_id = name[:-4]
            if allowed_ids is not None and rec_id not in allowed_ids:
                continue
            out.append((os.path.join(root, name), rec_id))
    return sorted(out)


def load_center_real_records(args) -> list[dict]:
    import wfdb

    allowed = None
    if args.ref_meta:
        with open(args.ref_meta) as f:
            meta = json.load(f)
        allowed = set(meta.get("ref_record_ids", []))
    pairs = scan_center_records(args.center, args.pn2021_root, allowed)
    records = []
    for hea_path, rec_id in pairs:
        codes = parse_header_snomed(hea_path)
        y = snomed_list_to_super5(codes).astype(np.float32)
        if y.sum() <= 0:
            continue
        try:
            rec = wfdb.rdrecord(hea_path[:-4])
        except Exception:
            continue
        sig = rec.p_signal
        if sig is None or sig.shape[1] < 12:
            continue
        names = [s.strip() for s in rec.sig_name] if getattr(rec, "sig_name", None) else None
        proc = unified_preprocess_to_1000(
            sig.astype(np.float32),
            fs=rec.fs,
            source_leads=names,
            target_fs=100,
            target_len=1000,
            apply_filter=True,
            apply_zscore=True,
        )
        if proc is None:
            continue
        records.append({"record_id": rec_id, "signal": proc, "label": y})
        if args.max_real_records and len(records) >= args.max_real_records:
            break
    if not records:
        raise RuntimeError(f"No usable real records for center={args.center}")
    return records


def split_records(records: list[dict], val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n_val = max(1, int(round(len(records) * val_fraction))) if len(records) >= 10 else 0
    val_idx = set(order[:n_val].tolist())
    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [r for i, r in enumerate(records) if i in val_idx]
    if not val:
        val = train
    return train, val


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, labels_all, logits_all = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        losses.append(float(loss.item()))
        labels_all.append(y.cpu().numpy())
        logits_all.append(logits.float().cpu().numpy())
    y_true = np.concatenate(labels_all)
    logits = np.concatenate(logits_all)
    y_score = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    metrics = compute_macro_auroc_auprc(y_true, y_score, CLASS_NAMES_SUPER5, min_pos=1)
    return float(np.mean(losses)), metrics


def load_model(args, device):
    model = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=NUM_SUPER5,
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    ).to(device)
    state = torch.load(args.init_checkpoint, map_location=device)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


def main(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    records = load_center_real_records(args)
    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    train_real = CenterRealDataset(train_records, crop_len=args.crop_len, mode="train")
    val_ds = CenterRealDataset(val_records, crop_len=args.crop_len, mode="eval")
    synth_ds = SynthNPZDataset(args.synth_npz, crop_len=args.crop_len, mode="train") if args.synth_npz else None

    if synth_ds is not None:
        combo = ConcatDataset([train_real, synth_ds])
        real_weight = np.ones(len(train_real), dtype=np.float64) / max(len(train_real), 1)
        synth_weight = np.ones(len(synth_ds), dtype=np.float64) * args.synth_ratio / max(len(synth_ds), 1)
        sampler = WeightedRandomSampler(
            np.concatenate([real_weight, synth_weight]),
            num_samples=max(args.epoch_samples, len(train_real)),
            replacement=True,
        )
        train_loader = DataLoader(combo, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                                  num_workers=args.num_workers, pin_memory=device.type == "cuda")
    else:
        train_loader = DataLoader(train_real, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")

    labels_for_weight = train_real.labels
    if synth_ds is not None:
        labels_for_weight = np.concatenate([labels_for_weight, synth_ds.labels], axis=0)
    pos_weight = torch.tensor(compute_pos_weight(labels_for_weight, NUM_SUPER5, clip_max=args.pos_weight_clip_max),
                              dtype=torch.float32, device=device)

    def criterion(logits, labels):
        return masked_bce_with_logits(logits, labels, pos_weight)

    model = load_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_score = -float("inf")
    log = []
    print(f"[data] center={args.center} real_train={len(train_real)} real_val={len(val_ds)} synth={0 if synth_ds is None else len(synth_ds)}")
    print(f"[init] {args.init_checkpoint}")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        score = val_metrics["macro_auroc"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_loss": val_loss,
            "val_macro_auroc": score,
            "val_macro_auprc": val_metrics["macro_auprc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        log.append(row)
        print(
            f"[epoch {epoch}] loss={row['train_loss']:.5f} val={val_loss:.5f} "
            f"auroc={score:.4f} auprc={val_metrics['macro_auprc']:.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        with (out_dir / "training_log.json").open("w") as f:
            json.dump(log, f, indent=2)

    if not (out_dir / "best_model.pt").exists():
        torch.save(model.state_dict(), out_dir / "best_model.pt")
    with (out_dir / "train_result.json").open("w") as f:
        json.dump({
            "center": args.center,
            "best_val_macro_auroc": best_score,
            "epochs_trained": len(log),
            "real_train": len(train_real),
            "real_val": len(val_ds),
            "synth": 0 if synth_ds is None else len(synth_ds),
            "class_names": CLASS_NAMES_SUPER5,
            "pos_weight": pos_weight.detach().cpu().tolist(),
            "config": vars(args),
        }, f, indent=2)
    print(f"[done] saved {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--center", required=True)
    parser.add_argument("--ref_meta", default="")
    parser.add_argument("--pn2021_root", default="/root/autodl-tmp/physionet2021")
    parser.add_argument("--synth_npz", default="")
    parser.add_argument("--init_checkpoint", default="/root/autodl-tmp/triple_labels/super5/best_model.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop_len", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--synth_ratio", type=float, default=5.0)
    parser.add_argument("--epoch_samples", type=int, default=1024)
    parser.add_argument("--max_real_records", type=int, default=0)
    parser.add_argument("--pos_weight_clip_max", type=float, default=50.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
