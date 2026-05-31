#!/usr/bin/env python
"""Audit reconstruction quality for a trained 500 Hz PTB-XL VAE."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_adv_gen.vae import build_vae500, lead_consistency_residual


class SplitDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str, *, limit: int | None = None):
        self.path = cache_dir / split / "signals_norm.npy"
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.signals = np.load(self.path, mmap_mode="r")
        self.limit = min(int(limit), len(self.signals)) if limit else len(self.signals)

    def __len__(self) -> int:
        return int(self.limit)

    def __getitem__(self, idx: int) -> torch.Tensor:
        signal = np.array(self.signals[idx], dtype=np.float32, copy=True)
        return torch.from_numpy(np.ascontiguousarray(signal))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _pearson(a: np.ndarray, b: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    a0 = a - a.mean(axis=axis, keepdims=True)
    b0 = b - b.mean(axis=axis, keepdims=True)
    denom = np.sqrt((a0 * a0).sum(axis=axis) * (b0 * b0).sum(axis=axis)) + 1e-8
    return (a0 * b0).sum(axis=axis) / denom


def _invalid_mask(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x).all(axis=(1, 2))
    p2p = np.ptp(x, axis=1).max(axis=1)
    flat = p2p < 1e-4
    extreme = p2p > 100.0
    return (~finite) | flat | extreme


@torch.no_grad()
def _audit_split(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    xs, rs = [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        noise = torch.zeros((batch.shape[0], model.latent_channels, model.latent_length), device=device)
        out = model(batch, noise=noise)
        xs.append(batch.detach().cpu().numpy().astype(np.float32))
        rs.append(out.recon.detach().cpu().numpy().astype(np.float32))
    x = np.concatenate(xs, axis=0)
    recon = np.concatenate(rs, axis=0)

    invalid = _invalid_mask(recon)
    global_corr = _pearson(recon.reshape(recon.shape[0], -1), x.reshape(x.shape[0], -1), axis=1)
    lead_corr = _pearson(recon, x, axis=1)
    diff_corr = _pearson(
        (recon[:, 1:] - recon[:, :-1]).reshape(recon.shape[0], -1),
        (x[:, 1:] - x[:, :-1]).reshape(x.shape[0], -1),
        axis=1,
    )
    recon_p2p = np.ptp(recon, axis=1).max(axis=1)
    x_p2p = np.ptp(x, axis=1).max(axis=1)
    p2p_ratio = recon_p2p / (x_p2p + 1e-8)

    x_res = lead_consistency_residual(torch.from_numpy(x)).numpy()
    r_res = lead_consistency_residual(torch.from_numpy(recon)).numpy()
    x_res_mae = np.mean(np.abs(x_res), axis=(1, 2))
    r_res_mae = np.mean(np.abs(r_res), axis=(1, 2))
    residual_ratio = r_res_mae / (x_res_mae + 1e-8)

    return {
        "n": int(x.shape[0]),
        "invalid_decode_rate": float(invalid.mean()),
        "median_global_pearson": float(np.nanmedian(global_corr)),
        "median_first_diff_pearson": float(np.nanmedian(diff_corr)),
        "median_lead_pearson": np.nanmedian(lead_corr, axis=0).astype(float).tolist(),
        "n_leads_median_pearson_ge_0p90": int((np.nanmedian(lead_corr, axis=0) >= 0.90).sum()),
        "min_median_lead_pearson": float(np.nanmin(np.nanmedian(lead_corr, axis=0))),
        "p2p_ratio_median": float(np.nanmedian(p2p_ratio)),
        "p2p_ratio_p05": float(np.nanquantile(p2p_ratio, 0.05)),
        "p2p_ratio_p95": float(np.nanquantile(p2p_ratio, 0.95)),
        "lead_consistency_residual_ratio_median": float(np.nanmedian(residual_ratio)),
        "recon_mae": float(np.mean(np.abs(recon - x))),
        "recon_mse": float(np.mean((recon - x) ** 2)),
    }


def _passes_hard_gate(val: dict[str, Any], audit: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if val["invalid_decode_rate"] >= 0.01:
        reasons.append("fold9 invalid decode rate >= 1%")
    if audit["invalid_decode_rate"] >= 0.01:
        reasons.append("fold10 invalid decode rate >= 1%")
    if val["median_global_pearson"] < 0.95:
        reasons.append("fold9 median global Pearson < 0.95")
    if audit["median_global_pearson"] < 0.93:
        reasons.append("fold10 median global Pearson < 0.93")
    if audit["n_leads_median_pearson_ge_0p90"] < 10:
        reasons.append("fewer than 10/12 leads have median Pearson >= 0.90")
    if audit["min_median_lead_pearson"] < 0.80:
        reasons.append("a lead has median Pearson < 0.80")
    if audit["median_first_diff_pearson"] < 0.80:
        reasons.append("fold10 first-difference Pearson < 0.80")
    if not (0.80 <= audit["p2p_ratio_median"] <= 1.25):
        reasons.append("fold10 median p2p ratio outside [0.80, 1.25]")
    if audit["p2p_ratio_p05"] < 0.50 or audit["p2p_ratio_p95"] > 2.00:
        reasons.append("fold10 p2p p5-p95 outside [0.50, 2.00]")
    if audit["lead_consistency_residual_ratio_median"] > 1.20:
        reasons.append("fold10 lead residual worsens by >20%")
    return len(reasons) == 0, reasons


def audit(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    meta = ckpt.get("model_metadata", {})
    variant = args.model_variant or meta.get("variant") or ckpt_args.get("model_variant", "diffusets500_v1_dynamic")
    model = build_vae500(
        variant,
        base_channels=int(ckpt_args.get("base_channels", args.base_channels)),
        use_attention=not bool(ckpt_args.get("no_attention", args.no_attention)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(args.device)
    model.to(device)

    results: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "cache_dir": str(cache_dir),
        "variant": variant,
    }
    split_results = {}
    for split in args.splits.split(","):
        split = split.strip()
        dataset = SplitDataset(cache_dir, split, limit=args.limit)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        split_results[split] = _audit_split(model, loader, device)
        print(f"[{split}] {json.dumps(split_results[split], default=_json_default)}", flush=True)
    results["splits"] = split_results
    if "val" in split_results and "audit" in split_results:
        passed, reasons = _passes_hard_gate(split_results["val"], split_results["audit"])
        results["hard_gate_pass"] = passed
        results["hard_gate_reasons"] = reasons
    out_json = output_dir / "reconstruction_audit.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"[done] wrote {out_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_variant", default=None)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--splits", default="val,audit")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
