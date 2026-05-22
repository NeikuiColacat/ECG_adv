#!/usr/bin/env python3
"""CDAN-style K=500 target-center adaptation baseline for Super5 PN2021.

This is a multi-label ECG adaptation of THUML Transfer-Learning-Library's CDAN:
we reuse the official GRL and DomainDiscriminator modules and the randomized
multi-linear conditioning idea, but condition on sigmoid Super5 probabilities
instead of softmax probabilities because Super5 is multi-label.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper.run_dann_k500_20260517 import (  # noqa: E402
    DOMAIN_DISC_FILE,
    GRL_FILE,
    load_class_from_file,
)
from scripts.paper.run_deep_coral_k500_20260517 import (  # noqa: E402
    BASELINE_CKPT,
    CLASS_NAMES_SUPER5,
    NUM_SUPER5,
    PN2021_CACHE_DIR,
    PN2021_MMAP_CACHE_DIR,
    PTBXL_PREP,
    PYTHON,
    evaluate,
    forward_feature_map,
    load_model,
    make_loaders,
    pooled_features,
    run_cmd,
    subset_paths,
    TargetNPZDataset,
)
from scripts.triple_labels.train_ptbxl import compute_pos_weight, masked_bce_with_logits  # noqa: E402


DEFAULT_OUT_ROOT = Path("/root/autodl-tmp/paper_uda_baselines_20260517/cdan_k500")


class RandomizedMultiLinearMap(nn.Module):
    def __init__(self, features_dim: int, num_classes: int, output_dim: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        self.register_buffer("Rf", torch.randn(features_dim, output_dim, generator=gen))
        self.register_buffer("Rg", torch.randn(num_classes, output_dim, generator=gen))
        self.output_dim = int(output_dim)

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f_proj = torch.mm(f, self.Rf.to(f.device, dtype=f.dtype))
        g_proj = torch.mm(g, self.Rg.to(g.device, dtype=g.dtype))
        return torch.mul(f_proj, g_proj) / math.sqrt(float(self.output_dim))


def cdan_tag(args: argparse.Namespace) -> str:
    return f"cdan_lam{args.cdan_lambda:g}_d{args.randomized_dim}_h{args.domain_hidden}".replace(".", "p")


def cdan_loss(
    logits_s: torch.Tensor,
    feat_s: torch.Tensor,
    logits_t: torch.Tensor,
    feat_t: torch.Tensor,
    mapper: nn.Module,
    discriminator: nn.Module,
    grl: nn.Module,
) -> tuple[torch.Tensor, float]:
    p_s = torch.sigmoid(logits_s).detach()
    p_t = torch.sigmoid(logits_t).detach()
    h = mapper(torch.cat([feat_s, feat_t], dim=0), torch.cat([p_s, p_t], dim=0))
    d = discriminator(grl(h)).float()
    d_s, d_t = d.chunk(2, dim=0)
    y_s = torch.ones_like(d_s)
    y_t = torch.zeros_like(d_t)
    loss = 0.5 * (
        F.binary_cross_entropy(d_s, y_s) +
        F.binary_cross_entropy(d_t, y_t)
    )
    with torch.no_grad():
        acc_s = ((d_s >= 0.5) == (y_s >= 0.5)).float().mean()
        acc_t = ((d_t >= 0.5) == (y_t >= 0.5)).float().mean()
        acc = float((0.5 * (acc_s + acc_t)).item())
    return loss, acc


def train_one(center: str, args: argparse.Namespace) -> Path:
    paths = subset_paths(center, args.k, args.subset_seed)
    if not paths["signals"].exists() or not paths["meta"].exists():
        raise FileNotFoundError(f"missing K-shot subset for {center}: {paths}")

    tag = cdan_tag(args)
    out_dir = Path(args.out_root) / "runs" / f"{center}_K{args.k}_{tag}_ep{args.epochs}_seed{args.seed}"
    eval_path = out_dir / "eval_result_v3_super5_normsuppress_exclrefs_crop1000.json"
    if eval_path.exists() and not args.force:
        print(f"[skip] {center} {tag} already evaluated")
        return eval_path
    if args.dry_run:
        print(f"[dry-run] would train {center} {tag} into {out_dir}")
        return eval_path
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    source_train_ds, source_val_ds, train_labels = make_loaders(args)
    with np.load(paths["signals"], allow_pickle=True) as data:
        target_signals = data["signals"].astype(np.float32, copy=False)
    target_ds = TargetNPZDataset(target_signals, crop_len=args.crop_len, mode="train")

    source_loader = DataLoader(
        source_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        source_val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    pos_weight = torch.tensor(
        compute_pos_weight(train_labels, NUM_SUPER5, clip_max=args.pos_weight_clip_max),
        dtype=torch.float32,
        device=device,
    )

    def criterion(logits, y):
        return masked_bce_with_logits(logits, y, pos_weight)

    model = load_model(device)
    DomainDiscriminator = load_class_from_file(DOMAIN_DISC_FILE, "DomainDiscriminator")
    WarmStartGRL = load_class_from_file(GRL_FILE, "WarmStartGradientReverseLayer")
    mapper = RandomizedMultiLinearMap(
        features_dim=args.feature_dim,
        num_classes=NUM_SUPER5,
        output_dim=args.randomized_dim,
        seed=args.seed,
    ).to(device)
    discriminator = DomainDiscriminator(
        in_feature=args.randomized_dim,
        hidden_size=args.domain_hidden,
        batch_norm=True,
        sigmoid=True,
    ).to(device)
    grl = WarmStartGRL(
        alpha=1.0,
        lo=0.0,
        hi=1.0,
        max_iters=max(len(source_loader) * args.epochs, 1),
        auto_step=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    with (out_dir / "run_config.json").open("w") as f:
        json.dump(
            {
                **vars(args),
                "center": center,
                "method": "cdan_multilabel_sigmoid",
                "method_tag": tag,
                "source_protocol": "PTB-XL folds 1-8 labeled",
                "target_protocol": "K=500 target-center refs, labels ignored",
                "checkpoint_selection": "PTB-XL fold9 macro AUPRC",
                "init_checkpoint": BASELINE_CKPT,
                "official_grl": str(GRL_FILE),
                "official_domain_discriminator": str(DOMAIN_DISC_FILE),
                "conditional_map": "CDAN randomized multilinear map using sigmoid Super5 probabilities",
                "subset_signals": str(paths["signals"]),
                "ref_meta": str(paths["meta"]),
                "class_names": list(CLASS_NAMES_SUPER5),
                "pos_weight": pos_weight.detach().cpu().tolist(),
            },
            f,
            indent=2,
        )

    best_score = -float("inf")
    rows = []
    t0 = time.time()
    print(
        f"[data] center={center} source_train={len(source_train_ds)} "
        f"source_val={len(source_val_ds)} target={len(target_ds)} "
        f"cdan_lambda={args.cdan_lambda}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        discriminator.train()
        target_iter = itertools.cycle(target_loader)
        cls_losses, domain_losses, total_losses, domain_accs = [], [], [], []
        for x_s, y_s in source_loader:
            x_t = next(target_iter)
            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                feat_s_map = forward_feature_map(model, x_s)
                feat_t_map = forward_feature_map(model, x_t)
                logits_s = model.classifier(feat_s_map)
                logits_t = model.classifier(feat_t_map)
                loss_cls = criterion(logits_s, y_s)
            feat_s = pooled_features(feat_s_map).float()
            feat_t = pooled_features(feat_t_map).float()
            loss_domain, domain_acc = cdan_loss(
                logits_s.float(), feat_s, logits_t.float(), feat_t, mapper, discriminator, grl
            )
            loss = loss_cls + float(args.cdan_lambda) * loss_domain
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(discriminator.parameters()), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            cls_losses.append(float(loss_cls.detach().item()))
            domain_losses.append(float(loss_domain.detach().item()))
            total_losses.append(float(loss.detach().item()))
            domain_accs.append(domain_acc)

        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device)
        score = float(val_metrics["macro_auprc"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(total_losses)),
            "train_cls_loss": float(np.mean(cls_losses)),
            "train_domain_loss": float(np.mean(domain_losses)),
            "train_domain_acc": float(np.mean(domain_accs)),
            "val_loss": float(val_metrics["loss"]),
            "val_macro_auroc": float(val_metrics["macro_auroc"]),
            "val_macro_auprc": float(val_metrics["macro_auprc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(
            f"Ep {epoch:02d}/{args.epochs} cls={row['train_cls_loss']:.4f} "
            f"domain={row['train_domain_loss']:.4f} dacc={row['train_domain_acc']:.3f} "
            f"val={row['val_loss']:.4f} auroc={row['val_macro_auroc']:.4f} "
            f"auprc={row['val_macro_auprc']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            torch.save(discriminator.state_dict(), out_dir / "best_domain_discriminator.pt")
        with (out_dir / "training_log.json").open("w") as f:
            json.dump(rows, f, indent=2)

    with (out_dir / "train_result.json").open("w") as f:
        json.dump(
            {
                "center": center,
                "method": "cdan_multilabel_sigmoid",
                "method_tag": tag,
                "best_val_macro_auprc": best_score,
                "epochs_trained": args.epochs,
                "n_source_train": int(len(source_train_ds)),
                "n_source_val": int(len(source_val_ds)),
                "n_target_unlabeled": int(len(target_ds)),
                "pos_weight": pos_weight.detach().cpu().tolist(),
            },
            f,
            indent=2,
        )

    eval_cmd = [
        PYTHON,
        "-u",
        "scripts/triple_labels/eval_crosscenter.py",
        "--scheme",
        "super5",
        "--model_dir",
        str(out_dir),
        "--device",
        args.device,
        "--crop_len",
        str(args.crop_len),
        "--batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
        "--ptbxl_cache",
        PTBXL_PREP,
        "--preprocess_mode",
        "minimal_resample",
        "--norm_mode",
        "per_sample_global",
        "--pn2021_cache_dir",
        PN2021_CACHE_DIR,
        "--pn2021_mmap_cache_dir",
        PN2021_MMAP_CACHE_DIR,
        "--skip_mimic",
        "--exclude_ref_ids",
        str(paths["meta"]),
        "--output_path",
        str(eval_path),
    ]
    run_cmd(eval_cmd, out_dir / "eval_full.log", dry_run=args.dry_run)
    return eval_path


def parse_eval(center: str, args: argparse.Namespace, eval_path: Path) -> dict:
    with eval_path.open() as f:
        data = json.load(f)
    return {
        "method": "cdan_multilabel_sigmoid",
        "tag": cdan_tag(args),
        "center": center,
        "K": int(args.k),
        "epochs": int(args.epochs),
        "cdan_lambda": float(args.cdan_lambda),
        "target_auroc": float(data["pn2021"]["per_center"][center]["macro_auroc"]),
        "target_auprc": float(data["pn2021"]["per_center"][center]["macro_auprc"]),
        "pn2021_avg_auroc": float(data["pn2021"]["avg_macro_auroc"]),
        "pn2021_avg_auprc": float(data["pn2021"]["avg_macro_auprc"]),
        "ptbxl_auroc": float(data["ptbxl_test"]["macro_auroc"]),
        "ptbxl_auprc": float(data["ptbxl_test"]["macro_auprc"]),
        "eval_path": str(eval_path),
    }


def load_existing_rows(out_root: Path) -> list[dict]:
    rows = []
    for eval_path in sorted((out_root / "runs").glob("*/eval_result_v3_super5_normsuppress_exclrefs_crop1000.json")):
        cfg_path = eval_path.parent / "run_config.json"
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open() as f:
                cfg = json.load(f)
            rows.append(parse_eval(str(cfg["center"]), argparse.Namespace(**cfg), eval_path))
        except Exception as exc:
            print(f"[warn] failed to parse {eval_path}: {exc}", flush=True)
    return rows


def write_summary(rows: list[dict], out_root: Path) -> None:
    summary_dir = out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r["center"], r["tag"]))
    csv_path = summary_dir / "cdan_k500.csv"
    fields = [
        "method",
        "tag",
        "center",
        "K",
        "epochs",
        "cdan_lambda",
        "target_auroc",
        "target_auprc",
        "pn2021_avg_auroc",
        "pn2021_avg_auprc",
        "ptbxl_auroc",
        "ptbxl_auprc",
        "eval_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"[summary] wrote {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--centers", nargs="+", default=["ningbo", "chapman_shaoxing", "cpsc_2018", "georgia"])
    p.add_argument("--k", type=int, default=500)
    p.add_argument("--subset_seed", type=int, default=20260531)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--cdan_lambda", type=float, default=0.1)
    p.add_argument("--domain_hidden", type=int, default=256)
    p.add_argument("--feature_dim", type=int, default=640)
    p.add_argument("--randomized_dim", type=int, default=1024)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=192)
    p.add_argument("--crop_len", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pos_weight_clip_max", type=float, default=50.0)
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = load_existing_rows(out_root)
    for center in args.centers:
        eval_path = train_one(center, args)
        if not args.dry_run:
            rows = [r for r in rows if not (r["center"] == center and r["tag"] == cdan_tag(args))]
            rows.append(parse_eval(center, args, eval_path))
        write_summary(rows, out_root)
    write_summary(rows, out_root)


if __name__ == "__main__":
    main()
