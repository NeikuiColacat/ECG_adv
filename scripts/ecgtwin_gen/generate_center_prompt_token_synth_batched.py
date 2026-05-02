"""Batched ECGTwin prompt-token generation for graduate-project synthetic pools.

This is output-compatible with generate_center_prompt_token_synth.py but avoids
the one-sample-at-a-time loop. It supports true multi-vector prompt-token banks,
for example 5 x 4 x 768 PTB-XL class tokens.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from adversarial.efficientnet_victim_tierM import EfficientNetVictimTierM  # noqa: E402
from methods.ecgtwin_gen.prompt_token.trainer import (  # noqa: E402
    _actual_report_text_embed,
    _load_text_embed,
)
from scripts.ecgtwin_gen.generate_center_prompt_token_synth import (  # noqa: E402
    choose_ref_indices,
    postprocess_for_classifier,
    token_sequence_from_bank,
)
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402
from model.ECGTwin.utils.data_utils import _pad_text_embed  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper, process_pat_info, sex_transform  # noqa: E402


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _chunks(items: list[int], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield start, items[start:start + chunk_size]


def _sex_tensor(values: list[str]) -> torch.Tensor:
    return torch.tensor([sex_transform(v) for v in values], dtype=torch.float32)


def _choose_refs(
    cache: dict,
    selection: dict,
    cls: str,
    n: int,
    rng: np.random.Generator,
    policy: str,
    per_ref_cap: int,
) -> list[int]:
    selected = [int(i) for i in selection["selected_indices_in_full_cache"]]
    if policy == "same_class":
        eligible = [i for i in selected if cache["primary_class"][i] == cls]
    elif policy == "normal_ref":
        eligible = [i for i in selected if cache["primary_class"][i] == "NORM"]
    elif policy == "random_any":
        eligible = list(selected)
    elif policy == "cross_non_target":
        eligible = [i for i in selected if cache["primary_class"][i] != cls]
    elif policy == "mixed":
        chunks = []
        for sub_policy in ["same_class", "normal_ref", "random_any"]:
            chunks.extend(_choose_refs(
                cache, selection, cls, int(np.ceil(n / 3)), rng, sub_policy, per_ref_cap
            ))
        rng.shuffle(chunks)
        return chunks[:n]
    else:
        raise ValueError(f"unknown ref_class_policy={policy!r}")

    if not eligible:
        return []
    if per_ref_cap > 0:
        expanded = np.asarray(
            [idx for idx in eligible for _ in range(per_ref_cap)],
            dtype=np.int64,
        )
        if expanded.size >= n:
            chosen = rng.choice(expanded, size=n, replace=False)
        else:
            extra = rng.choice(np.asarray(eligible, dtype=np.int64), size=n - expanded.size, replace=True)
            chosen = np.concatenate([expanded, extra])
            rng.shuffle(chosen)
        return chosen.astype(int).tolist()

    replace = len(eligible) < n
    return rng.choice(np.asarray(eligible, dtype=np.int64), size=n, replace=replace).astype(int).tolist()


def _ref_text_for(
    cache: dict,
    idx: int,
    prompt_bank: dict,
    cls: str,
    mode: str,
    ref_prompt_embeds: dict | None = None,
) -> torch.Tensor:
    target_text = _load_text_embed(prompt_bank, None, cls).float()
    if mode == "actual_report":
        actual = _actual_report_text_embed(cache, idx)
        return actual if actual is not None else target_text
    if mode == "translated_report":
        if ref_prompt_embeds is None:
            raise ValueError("--ref_prompt_embed_cache is required for ref_text_mode=translated_report")
        source_idx = int(cache["source_indices"][idx])
        emb = ref_prompt_embeds.get(source_idx)
        if emb is None:
            emb = ref_prompt_embeds.get(str(source_idx))
        return emb.detach().float() if torch.is_tensor(emb) else target_text
    if mode == "normal":
        return prompt_bank["by_class"]["NORM"].detach().float()
    if mode == "class_fallback":
        return target_text
    raise ValueError(f"unknown ref_text_mode={mode!r}")


@torch.no_grad()
def generate_batch(
    wrapper: ECGTwinWrapper,
    victim: EfficientNetVictimTierM,
    cache: dict,
    prompt_bank: dict,
    token_blob: dict,
    center_idx: int,
    cls: str,
    cls_idx: int,
    ref_indices: list[int],
    steps: int,
    seed_base: int,
    token_repeat: int,
    ref_text_mode: str,
    ref_class_policy: str,
    ref_prompt_embeds: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    device = wrapper.device
    bsz = len(ref_indices)
    base_text = _load_text_embed(prompt_bank, None, cls).float()
    token_seq = token_sequence_from_bank(token_blob, center_idx, cls_idx).repeat(token_repeat, 1)
    text_aug = torch.cat([base_text, token_seq], dim=0).to(device)
    text_aug_b = text_aug.unsqueeze(0).repeat(bsz, 1, 1)
    mask_aug = torch.ones(bsz, text_aug.shape[0], dtype=torch.float32, device=device)

    ref_latents = torch.stack([cache["latents"][i].float() for i in ref_indices], dim=0).to(device)
    hr = torch.tensor([float(cache["hr"][i]) for i in ref_indices], dtype=torch.float32)
    age = torch.tensor([float(cache["age"][i]) for i in ref_indices], dtype=torch.float32)
    sex = _sex_tensor([cache["sex"][i] for i in ref_indices])

    ref_text_list = [
        _ref_text_for(cache, idx, prompt_bank, cls, ref_text_mode, ref_prompt_embeds)
        for idx in ref_indices
    ]
    ref_text_b, ref_text_mask = _pad_text_embed(ref_text_list)
    ref_text_b = ref_text_b.to(device)
    ref_text_mask = ref_text_mask.to(device)
    pat_info_ref = process_pat_info(normalize=True, hr=hr, age=age, sex=sex).to(device)
    pat_info_tar = process_pat_info(normalize=True, add_token=False, hr=hr, age=age, sex=sex).to(device)
    base_vector = wrapper.ibe_model.extract_features(
        ref_latents.transpose(2, 1), ref_text_b, ref_text_mask, pat_info_ref, reduce=True
    )

    cond = {
        "base_vector": base_vector,
        "text_embed": text_aug_b,
        "text_embed_mask": mask_aug,
        "pat_info": pat_info_tar,
        "ref_latent": ref_latents,
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_base)
    x_init = torch.randn(bsz, 4, 128, device=device, generator=generator)
    latent_gen = wrapper.ddpm_sample(
        cond,
        batch_size=bsz,
        num_inference_steps=steps,
        x_init=x_init,
    )
    ecg_tc = wrapper.decode_latent(latent_gen)
    signal_ct = postprocess_for_classifier(ecg_tc)
    logits = victim.compute_logits_from_ecg(signal_ct)
    probs = torch.sigmoid(logits).detach().cpu().numpy()

    labels = np.zeros((bsz, len(CLASS_NAMES_SUPER5)), dtype=np.float32)
    labels[:, cls_idx] = 1.0
    records = []
    for j, ref_idx in enumerate(ref_indices):
        seed = seed_base + j
        records.append({
            "center": token_blob["centers"][center_idx],
            "class": cls,
            "seed": int(seed),
            "ref_record_id": str(cache["record_ids"][ref_idx]),
            "ref_source_index": int(cache["source_indices"][ref_idx]),
            "ref_primary_class": str(cache["primary_class"][ref_idx]),
            "ref_primary_snomed": cache["primary_snomed"][ref_idx],
            "ref_text_mode": ref_text_mode,
            "ref_class_policy": ref_class_policy,
            "prompt_token": f"<{token_blob['centers'][center_idx]}_{cls}>",
            "prompt_token_vectors": int(token_seq.shape[0]),
            "p_target": float(probs[j, cls_idx]),
            "top1": CLASS_NAMES_SUPER5[int(np.argmax(probs[j]))],
            "victim_probs": {
                CLASS_NAMES_SUPER5[i]: float(probs[j, i])
                for i in range(len(CLASS_NAMES_SUPER5))
            },
            "finite": bool(torch.isfinite(signal_ct[j]).all().item()),
        })

    return (
        signal_ct.detach().cpu().numpy().astype(np.float32),
        ecg_tc.transpose(1, 2).detach().cpu().numpy().astype(np.float32),
        latent_gen.detach().cpu().numpy().astype(np.float32),
        labels,
        records,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", default="ptbxl")
    ap.add_argument("--classes", nargs="+", default=["NORM", "MI", "STTC", "HYP", "CD"])
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--token_bank", required=True)
    ap.add_argument("--prompt_bank", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--selection_seed", type=int, default=42)
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--gen_batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--victim_ckpt", required=True)
    ap.add_argument("--victim_crop_len", type=int, default=1000)
    ap.add_argument("--token_repeat", type=int, default=1)
    ap.add_argument("--ref_text_mode", choices=["class_fallback", "actual_report", "translated_report", "normal"],
                    default="class_fallback")
    ap.add_argument("--ref_prompt_embed_cache", default="",
                    help="PTB-XL index -> text_embed cache for ref_text_mode=translated_report")
    ap.add_argument("--ref_class_policy",
                    choices=["same_class", "normal_ref", "random_any", "cross_non_target", "mixed"],
                    default="same_class")
    ap.add_argument("--per_ref_cap", type=int, default=0,
                    help="If >0, cap repeats per reference before replacement fallback.")
    args = ap.parse_args()
    if args.gen_batch_size < 1:
        raise ValueError("--gen_batch_size must be >= 1")
    if args.token_repeat < 1:
        raise ValueError("--token_repeat must be >= 1")

    set_all_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir) / args.center
    out_dir.mkdir(parents=True, exist_ok=True)

    token_blob = torch.load(args.token_bank, map_location="cpu", weights_only=False)
    prompt_bank = torch.load(args.prompt_bank, map_location="cpu", weights_only=False)
    ref_prompt_embeds = None
    if args.ref_prompt_embed_cache:
        ref_prompt_blob = torch.load(args.ref_prompt_embed_cache, map_location="cpu", weights_only=False)
        ref_prompt_embeds = ref_prompt_blob.get("index_to_embed", ref_prompt_blob)
        print(f"[setup] loaded translated ref prompt embeds: {len(ref_prompt_embeds)}", flush=True)
    if args.ref_text_mode == "translated_report" and ref_prompt_embeds is None:
        raise ValueError("--ref_prompt_embed_cache is required for --ref_text_mode translated_report")
    centers = list(token_blob["centers"])
    class_names = list(token_blob["class_names"])
    if args.center not in centers:
        raise ValueError(f"center {args.center!r} not in token bank centers={centers}")
    center_idx = centers.index(args.center)

    cache_root = Path(args.cache_root)
    cache = torch.load(cache_root / "center_full_latents" / f"{args.center}.pt",
                       map_location="cpu", weights_only=False)
    with open(cache_root / "ref_selection" / f"{args.center}_k{args.K}_seed{args.selection_seed}.json") as f:
        selection = json.load(f)

    print(f"[setup] ECGTwin wrapper + victim on {args.device}", flush=True)
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=False)
    victim = EfficientNetVictimTierM(
        weight_path=args.victim_ckpt,
        device=args.device,
        ecgtwin_wrapper=wrapper,
        num_classes=len(CLASS_NAMES_SUPER5),
        crop_len=args.victim_crop_len,
    )
    victim.eval()

    all_signals, all_raw, all_latents, all_labels, records = [], [], [], [], []
    t0 = time.time()
    for cls in args.classes:
        if cls not in class_names:
            raise ValueError(f"unknown class {cls}; token classes={class_names}")
        cls_idx = class_names.index(cls)
        ref_indices = _choose_refs(
            cache=cache,
            selection=selection,
            cls=cls,
            n=args.n_per_class,
            rng=rng,
            policy=args.ref_class_policy,
            per_ref_cap=args.per_ref_cap,
        )
        if not ref_indices:
            print(f"[skip] {args.center} {cls}: no selected references", flush=True)
            continue
        for offset, chunk in _chunks(ref_indices, args.gen_batch_size):
            seed_base = args.seed + cls_idx * 1_000_000 + offset
            sig, raw, lat, lab, rec = generate_batch(
                wrapper=wrapper,
                victim=victim,
                cache=cache,
                prompt_bank=prompt_bank,
                token_blob=token_blob,
                center_idx=center_idx,
                cls=cls,
                cls_idx=cls_idx,
                ref_indices=chunk,
                steps=args.steps,
                seed_base=seed_base,
                token_repeat=args.token_repeat,
                ref_text_mode=args.ref_text_mode,
                ref_class_policy=args.ref_class_policy,
                ref_prompt_embeds=ref_prompt_embeds,
            )
            all_signals.append(sig)
            all_raw.append(raw)
            all_latents.append(lat)
            all_labels.append(lab)
            records.extend(rec)
            p_mean = float(np.mean([r["p_target"] for r in rec]))
            print(
                f"[gen] {cls:<4} {offset + len(chunk):>5}/{len(ref_indices):<5} "
                f"batch={len(chunk):>3} p_target_mean={p_mean:.3f}",
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
    print(f"[done] wrote {npz_path}", flush=True)
    print(f"[done] wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
