"""Train a no-leak real-only center-style auxiliary classifier.

This classifier is used as a differentiable style loss for prompt-token
training. It is not the final evidence that synthetic ECGs match a center; the
post-hoc no-leak probes in `run_prompt_token_noleak_style_validation.py` remain
the acceptance gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO = Path(__file__).resolve().parent.parent.parent
DEEPECG_NB = REPO / "model" / "DeepECG" / "notebooks"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DEEPECG_NB))

from EfficientNetv2 import EfficientNet1DV2  # noqa: E402


CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
DEFAULT_CENTERS = ["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"]


def primary_class(label: np.ndarray, allowed: set[str]) -> str | None:
    vals = {CLASS_NAMES[i]: float(label[i]) for i in range(len(CLASS_NAMES))}
    for cls in ["MI", "STTC", "CD", "HYP"]:
        if cls in allowed and vals.get(cls, 0.0) > 0.5:
            return cls
    if "NORM" in allowed and vals.get("NORM", 0.0) > 0.5:
        return "NORM"
    return None


def load_anchor_ids(cache_root: Path, center: str, k: int, seed: int) -> set[str]:
    path = cache_root / "ref_selection" / f"{center}_k{k}_seed{seed}.json"
    if not path.exists():
        return set()
    with path.open() as f:
        data = json.load(f)
    ids = data.get("ref_record_ids") or data.get("selected_record_ids") or []
    return {str(x) for x in ids}


def center_dir(mmap_root: Path, center: str) -> Path:
    return mmap_root / f"super5_{center}_100hz1000_v3_super5_normsuppress"


class MMapCenterStore:
    def __init__(self, root: Path, center: str) -> None:
        d = center_dir(root, center)
        if not d.exists():
            raise FileNotFoundError(d)
        self.center = center
        self.signals = np.load(d / "signals.npy", mmap_mode="r")
        self.labels = np.load(d / "labels.npy", mmap_mode="r")
        self.record_ids = np.load(d / "record_ids.npy", mmap_mode="r", allow_pickle=True)


def make_items(
    stores: Dict[str, MMapCenterStore],
    centers: List[str],
    classes: List[str],
    anchor_cache_root: Path,
    k: int,
    selection_seed: int,
    max_per_center_class: int,
    seed: int,
) -> tuple[List[dict], dict]:
    rng = np.random.default_rng(seed)
    allowed = set(classes)
    items: List[dict] = []
    stats = {}
    for center_idx, center in enumerate(centers):
        store = stores[center]
        anchor_ids = load_anchor_ids(anchor_cache_root, center, k, selection_seed)
        by_class: Dict[str, List[int]] = defaultdict(list)
        excluded = 0
        for i, rid in enumerate(store.record_ids):
            if str(rid) in anchor_ids:
                excluded += 1
                continue
            cls = primary_class(np.asarray(store.labels[i]), allowed)
            if cls is not None:
                by_class[cls].append(i)
        stats[center] = {
            "excluded_anchor_count": int(excluded),
            "by_class_counts_before_cap": {c: len(v) for c, v in by_class.items()},
        }
        for cls, idxs in by_class.items():
            idx_arr = np.asarray(idxs, dtype=np.int64)
            rng.shuffle(idx_arr)
            if max_per_center_class > 0:
                idx_arr = idx_arr[:max_per_center_class]
            for idx in idx_arr:
                items.append({
                    "center": center,
                    "center_idx": int(center_idx),
                    "record_idx": int(idx),
                    "primary_class": cls,
                    "record_id": str(store.record_ids[idx]),
                })
        stats[center]["by_class_counts_after_cap"] = Counter(
            x["primary_class"] for x in items if x["center"] == center
        )
        stats[center]["by_class_counts_after_cap"] = dict(stats[center]["by_class_counts_after_cap"])
    return items, stats


def split_items(items: List[dict], seed: int) -> tuple[List[dict], List[dict], List[dict]]:
    rng = np.random.default_rng(seed)
    groups: Dict[tuple[str, str], List[dict]] = defaultdict(list)
    for item in items:
        groups[(item["center"], item["primary_class"])].append(item)
    train, val, test = [], [], []
    for group_items in groups.values():
        group_items = list(group_items)
        rng.shuffle(group_items)
        n = len(group_items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train.extend(group_items[:n_train])
        val.extend(group_items[n_train:n_train + n_val])
        test.extend(group_items[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class StyleAuxDataset(Dataset):
    def __init__(self, stores: Dict[str, MMapCenterStore], items: List[dict]) -> None:
        self.stores = stores
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        sig_tc = np.asarray(
            self.stores[item["center"]].signals[item["record_idx"]],
            dtype=np.float32,
        )
        sig_ct = np.ascontiguousarray(sig_tc.T)
        return torch.from_numpy(sig_ct).float(), torch.tensor(item["center_idx"], dtype=torch.long)


def make_sampler(items: List[dict]) -> WeightedRandomSampler:
    keys = [(x["center"], x["primary_class"]) for x in items]
    counts = Counter(keys)
    weights = torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        ys.append(y.numpy())
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def train_one(args) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    centers = list(args.centers)
    classes = list(args.classes)
    mmap_root = Path(args.mmap_root)
    stores = {center: MMapCenterStore(mmap_root, center) for center in centers}
    items, stats = make_items(
        stores=stores,
        centers=centers,
        classes=classes,
        anchor_cache_root=Path(args.anchor_cache_root),
        k=args.K,
        selection_seed=args.selection_seed,
        max_per_center_class=args.max_per_center_class,
        seed=args.seed,
    )
    train_items, val_items, test_items = split_items(items, args.seed)
    (out / "split_record_ids.json").write_text(json.dumps({
        "train": [x["record_id"] for x in train_items],
        "val": [x["record_id"] for x in val_items],
        "test": [x["record_id"] for x in test_items],
    }, indent=2))

    train_loader = DataLoader(
        StyleAuxDataset(stores, train_items),
        batch_size=args.batch_size,
        sampler=make_sampler(train_items),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        StyleAuxDataset(stores, val_items),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        StyleAuxDataset(stores, test_items),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    device = torch.device(args.device)
    model = EfficientNet1DV2(
        variant="s_v2",
        input_channels=12,
        num_classes=len(centers),
        activation="leaky_relu",
        stochastic_depth_prob=0.304,
        dropout_rate=0.0,
        use_se=True,
        norm_type="batch",
    ).to(device)
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    config = vars(args)
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"[data] items={len(items)} train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    print(f"[data] stats={json.dumps(stats, sort_keys=True)}")
    print(f"[model] centers={centers} classes={classes}")

    best_val_f1 = -1.0
    patience = 0
    log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        scheduler.step()
        vy, vp = evaluate(model, val_loader, device)
        val_acc = accuracy_score(vy, vp)
        val_f1 = f1_score(vy, vp, average="macro")
        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            patience = 0
            torch.save(model.state_dict(), out / "best_model.pt")
        else:
            patience += 1
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, n_batches),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - t0,
        }
        log.append(row)
        (out / "training_log.json").write_text(json.dumps(log, indent=2))
        print(
            f"[ep {epoch:03d}] loss={row['train_loss']:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"lr={row['lr']:.5g} elapsed={row['elapsed_sec']:.0f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if patience >= args.patience:
            print(f"[early-stop] epoch={epoch} patience={args.patience}")
            break

    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    ty, tp = evaluate(model, test_loader, device)
    result = {
        "center_names": centers,
        "classes": classes,
        "n_total": len(items),
        "split_sizes": {
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items),
        },
        "stats": stats,
        "best_val_macro_f1": float(best_val_f1),
        "test_acc": float(accuracy_score(ty, tp)),
        "test_macro_f1": float(f1_score(ty, tp, average="macro")),
        "test_confusion_matrix": confusion_matrix(ty, tp).tolist(),
        "test_classification_report": classification_report(
            ty, tp, target_names=centers, output_dict=True, zero_division=0
        ),
        "epochs_trained": len(log),
        "config": config,
    }
    (out / "train_result.json").write_text(json.dumps(result, indent=2))
    print(
        f"[test] acc={result['test_acc']:.4f} macro_f1={result['test_macro_f1']:.4f} "
        f"saved={out}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mmap_root", default="/root/autodl-tmp/triple_labels/pn2021_eval_cache_mmap_minresample_perglobal")
    parser.add_argument("--anchor_cache_root", default="/root/autodl-tmp/ecgtwin_prompt_token_super5/cache_v1")
    parser.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    parser.add_argument("--classes", nargs="+", default=["NORM", "MI", "STTC"])
    parser.add_argument("--K", type=int, default=500)
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument("--max_per_center_class", type=int, default=3000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_one(args)


if __name__ == "__main__":
    main()
