"""Generate ECGTwin samples with learned 768-d center-class prompt tokens.

This is the textual-inversion-style generation path:

  base diagnosis text_embed + learned <center_CLASS> embedding -> ECGTwin text path
  reference ECG/text/patient info -> IBE base_vector

The original ECGTwin tokenizer and model weights are not modified.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from apps.streamlit_ecg_demo.services.paths import DATA_ROOT  # noqa: E402
from methods.ecgtwin_gen.prompt_token.trainer import _load_text_embed  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402
from util.super5_victim import EfficientNetVictimTierM  # noqa: E402


DEFAULT_CACHE_ROOT = str(DATA_ROOT / "ecgtwin_prompt_token_super5/cache_v1")
DEFAULT_TOKEN_BANK = (
    str(DATA_ROOT / "ecgtwin_prompt_token_super5/prompt_token_runs/v1_all4_steps2000/prompt_token_bank.pt")
)
DEFAULT_PROMPT_BANK = f"{DEFAULT_CACHE_ROOT}/text_prompt_bank.pt"
DEFAULT_OUT = str(DATA_ROOT / "ecgtwin_prompt_token_super5/generated_smoke")
DEFAULT_VICTIM_CKPT = str(DATA_ROOT / "triple_labels/super5_minresample_full10_perglobal_20260503/best_model.pt")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def postprocess_for_classifier(ecg_tc: torch.Tensor) -> torch.Tensor:
    """(B,1024,12) ECGTwin order -> (B,12,1000) PTBXL order, global z-score."""
    bsz = ecg_tc.shape[0]
    ecg_ct = ecg_tc.transpose(1, 2)
    ecg_ct = F.interpolate(ecg_ct, size=1000, mode="linear", align_corners=True)
    ecg_ct = ecg_ct[:, ECGTWIN_TO_PTBXL_INDICES, :]
    flat = ecg_ct.reshape(bsz, -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp(min=1e-8)
    return (ecg_ct - mean.unsqueeze(-1)) / std.unsqueeze(-1)


def choose_ref_indices(cache, selection, cls: str, n: int, rng: np.random.Generator):
    selected = selection["selected_indices_in_full_cache"]
    idx = [i for i in selected if cache["primary_class"][i] == cls]
    if not idx:
        return []
    replace = len(idx) < n
    return rng.choice(np.asarray(idx, dtype=np.int64), size=n, replace=replace).tolist()


def token_sequence_from_bank(token_blob, center_idx: int, cls_idx: int) -> torch.Tensor:
    """Return a `(M,768)` prompt-token sequence from legacy or v2 banks."""
    mode = str(token_blob.get("token_mode", "direct"))
    if mode == "factorized" or "center_embeddings" in token_blob:
        center = token_blob["center_embeddings"][center_idx].float()
        cls = token_blob["class_embeddings"][cls_idx].float()
        pieces = [center, cls]
        residual = token_blob.get("residual_embeddings")
        if residual is not None and residual.shape[2] > 0:
            pieces.append(residual[center_idx, cls_idx].float())
        return torch.cat(pieces, dim=0)

    emb = token_blob["embeddings"].float()
    if emb.dim() == 3:
        return emb[center_idx, cls_idx].unsqueeze(0)
    if emb.dim() == 4:
        return emb[center_idx, cls_idx]
    raise ValueError(f"unsupported token embedding shape: {tuple(emb.shape)}")


def init_token_sequence_from_bank(token_blob, center_idx: int, cls_idx: int) -> torch.Tensor | None:
    """Return the prompt-token initialization sequence when the bank stores it."""
    mode = str(token_blob.get("token_mode", "direct"))
    if mode == "factorized" or "center_embeddings" in token_blob:
        if "init_center_embeddings" not in token_blob or "init_class_embeddings" not in token_blob:
            return None
        center = token_blob["init_center_embeddings"][center_idx].float()
        cls = token_blob["init_class_embeddings"][cls_idx].float()
        pieces = [center, cls]
        residual = token_blob.get("init_residual_embeddings")
        if residual is not None and residual.shape[2] > 0:
            pieces.append(residual[center_idx, cls_idx].float())
        return torch.cat(pieces, dim=0)

    init = token_blob.get("init_embeddings")
    if init is None:
        return None
    init = init.float()
    if init.dim() == 3:
        return init[center_idx, cls_idx].unsqueeze(0)
    if init.dim() == 4:
        return init[center_idx, cls_idx]
    return None


def scale_token_sequence(
    token_blob,
    token_seq: torch.Tensor,
    center_idx: int,
    cls_idx: int,
    scale: float,
) -> torch.Tensor:
    """Scale learned prompt-token strength while preserving initialization semantics."""
    if scale == 1.0:
        return token_seq
    init_seq = init_token_sequence_from_bank(token_blob, center_idx, cls_idx)
    if init_seq is not None and init_seq.shape == token_seq.shape:
        return init_seq + float(scale) * (token_seq - init_seq)
    return token_seq * float(scale)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", default="ningbo")
    ap.add_argument("--classes", nargs="+", default=["NORM", "MI", "STTC", "HYP", "CD"])
    ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--token_bank", default=DEFAULT_TOKEN_BANK)
    ap.add_argument("--token_center", default=None,
                    help="Center token to append. Default: same as --center.")
    ap.add_argument("--token_class", default=None,
                    help="Class token to append. Default: same as generated class.")
    ap.add_argument("--no_token", action="store_true",
                    help="Generate with diagnosis prompt only, no learned center token.")
    ap.add_argument("--arm", default=None,
                    help="Optional ablation arm label written into summary records.")
    ap.add_argument("--prompt_bank", default=DEFAULT_PROMPT_BANK)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--selection_seed", type=int, default=42)
    ap.add_argument("--n_per_class", type=int, default=2)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--victim_ckpt", default=DEFAULT_VICTIM_CKPT)
    ap.add_argument("--victim_crop_len", type=int, default=250)
    ap.add_argument("--token_repeat", type=int, default=1,
                    help="Repeat the learned center-class token embedding in text_embed.")
    ap.add_argument("--token_scale", type=float, default=1.0,
                    help="Scale learned token delta as init + scale * (learned - init).")
    args = ap.parse_args()
    if args.token_repeat < 1:
        raise ValueError(f"token_repeat must be >= 1, got {args.token_repeat}")
    if args.token_scale < 0:
        raise ValueError(f"token_scale must be >= 0, got {args.token_scale}")
    set_all_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir) / args.center
    out_dir.mkdir(parents=True, exist_ok=True)

    token_blob = None if args.no_token else torch.load(args.token_bank, map_location="cpu", weights_only=False)
    prompt_bank = torch.load(args.prompt_bank, map_location="cpu", weights_only=False)
    if token_blob is not None:
        centers = list(token_blob["centers"])
        class_names = list(token_blob["class_names"])
        token_center = args.token_center or args.center
        if token_center not in centers:
            raise ValueError(f"token_center {token_center!r} not in token bank centers={centers}")
        token_center_idx = centers.index(token_center)
    else:
        centers = []
        class_names = list(CLASS_NAMES_SUPER5)
        token_center = None
        token_center_idx = None

    cache_root = Path(args.cache_root)
    cache = torch.load(
        cache_root / "center_full_latents" / f"{args.center}.pt",
        map_location="cpu",
        weights_only=False,
    )
    with open(cache_root / "ref_selection" / f"{args.center}_k{args.K}_seed{args.selection_seed}.json") as f:
        selection = json.load(f)

    print(f"[setup] ECGTwin wrapper + Super5 victim on {args.device}")
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)
    victim = EfficientNetVictimTierM(
        weight_path=args.victim_ckpt,
        device=args.device,
        ecgtwin_wrapper=wrapper,
        num_classes=len(CLASS_NAMES_SUPER5),
        crop_len=args.victim_crop_len,
    )

    all_signals, all_raw, all_latents, all_labels = [], [], [], []
    records = []
    t0 = time.time()
    for cls in args.classes:
        if cls not in class_names:
            raise ValueError(f"unknown class {cls}; token classes={class_names}")
        cls_idx = class_names.index(cls)
        token_cls = args.token_class or cls
        if token_cls not in class_names:
            raise ValueError(f"unknown token_class {token_cls}; token classes={class_names}")
        token_cls_idx = class_names.index(token_cls)
        ref_indices = choose_ref_indices(cache, selection, cls, args.n_per_class, rng)
        if not ref_indices:
            print(f"[skip] {args.center} {cls}: no selected references")
            continue

        for k, ref_idx in enumerate(ref_indices):
            seed = args.seed + cls_idx * 1000 + k
            set_all_seeds(seed)
            latent_ref = cache["latents"][ref_idx].float()
            primary_snomed = cache["primary_snomed"][ref_idx]
            base_text = _load_text_embed(prompt_bank, primary_snomed, cls)
            if args.no_token:
                token_seq = None
                text_aug = base_text
            else:
                token_seq = token_sequence_from_bank(token_blob, token_center_idx, token_cls_idx)
                token_seq = scale_token_sequence(
                    token_blob,
                    token_seq,
                    token_center_idx,
                    token_cls_idx,
                    args.token_scale,
                )
                token_seq = token_seq.repeat(args.token_repeat, 1)
                text_aug = torch.cat([base_text, token_seq], dim=0)
            mask_aug = torch.ones(1, text_aug.shape[0], dtype=torch.float32, device=wrapper.device)

            sex = cache["sex"][ref_idx]
            ref_label = {
                "hr": float(cache["hr"][ref_idx]),
                "age": float(cache["age"][ref_idx]),
                "sex": sex,
                "text_embed": base_text,
            }
            cond = wrapper.prepare_conditions(
                ref_latent=latent_ref,
                ref_label=ref_label,
                batch_size=1,
                target_text_embed=text_aug,
                target_hr=ref_label["hr"],
                target_age=ref_label["age"],
            )
            cond["text_embed_mask"] = mask_aug
            x_init = torch.randn(1, 4, 128, device=wrapper.device)
            with torch.no_grad():
                latent_gen = wrapper.ddpm_sample(
                    cond, batch_size=1, num_inference_steps=args.steps, x_init=x_init
                )
                ecg_tc = wrapper.decode_latent(latent_gen)
                signal_ct = postprocess_for_classifier(ecg_tc)
                logits = victim.compute_logits_from_ecg(signal_ct)
                probs = torch.sigmoid(logits)[0].cpu().numpy()

            label = np.zeros((1, len(CLASS_NAMES_SUPER5)), dtype=np.float32)
            label[0, CLASS_NAMES_SUPER5.index(cls)] = 1.0
            all_signals.append(signal_ct.cpu().numpy().astype(np.float32))
            all_raw.append(ecg_tc.transpose(1, 2).cpu().numpy().astype(np.float32))
            all_latents.append(latent_gen.cpu().numpy().astype(np.float32))
            all_labels.append(label)

            rec = {
                "center": args.center,
                "class": cls,
                "arm": args.arm or ("vanilla_no_token" if args.no_token else "target_token"),
                "token_center": token_center,
                "token_class": None if args.no_token else token_cls,
                "token_scale": None if args.no_token else float(args.token_scale),
                "seed": seed,
                "ref_record_id": cache["record_ids"][ref_idx],
                "ref_primary_snomed": primary_snomed,
                "prompt_token": "none" if args.no_token else f"<{token_center}_{token_cls}>",
                "p_target": float(probs[CLASS_NAMES_SUPER5.index(cls)]),
                "top1": CLASS_NAMES_SUPER5[int(np.argmax(probs))],
                "victim_probs": {CLASS_NAMES_SUPER5[i]: float(probs[i]) for i in range(len(CLASS_NAMES_SUPER5))},
                "finite": bool(np.isfinite(all_signals[-1]).all()),
            }
            records.append(rec)
            print(
                f"[gen] {cls:<4} ref={rec['ref_record_id']} seed={seed} "
                f"top1={rec['top1']:<4} p_target={rec['p_target']:.3f}",
                flush=True,
            )

    if not all_signals:
        raise RuntimeError("no samples generated")
    signals = np.concatenate(all_signals, axis=0)
    raw_signal_ct = np.concatenate(all_raw, axis=0)
    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    npz_path = out_dir / "samples.npz"
    np.savez_compressed(
        npz_path,
        signals=signals,
        raw_signal_ct=raw_signal_ct,
        latents=latents,
        labels=labels,
        center_name=args.center,
        class_names=np.asarray(CLASS_NAMES_SUPER5),
    )
    summary = {
        "config": vars(args),
        "npz": str(npz_path),
        "n_samples": int(signals.shape[0]),
        "signal_shape": list(signals.shape),
        "raw_signal_shape": list(raw_signal_ct.shape),
        "latents_shape": list(latents.shape),
        "elapsed_sec": round(time.time() - t0, 2),
        "records": records,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote {npz_path}")
    print(f"[done] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
