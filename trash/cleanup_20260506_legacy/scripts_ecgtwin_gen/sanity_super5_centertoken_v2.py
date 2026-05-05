"""CenterToken v2 smoking-gun gates G1-G4.

Per Plan Rev 13 verification spec. Validates whether CT v2 actually moves signal
(unlike v1 which had rel-diff 0.006-0.012 = noise floor).

Required gates:
  G1 — AdaLN move:        per-block rel-diff > 0.05 in ≥2 of 7 blocks
  G2 — Decoded swap:      ‖x_extra - x_nin‖ rel > 0.10 (vs v1 0.006-0.012)
  G3 — Inter-center cos:  median off-diag < 0.5, distinguishable from Gaussian
  G4 — Victim shift:      ≥ 1 super5 class shifts > 0.05 between CT_i / CT_j

G1, G2, G4 are MANDATORY (signal-level verification).
G3 is data-driven informational (per-center-style, not loss-driven).

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/sanity_super5_centertoken_v2.py \
      --ckpt_root /root/autodl-tmp/center_token_super5_v2 \
      --ref_pt /root/autodl-tmp/center_token_super5/extra_k200.pt \
      --centers extra,nin,geo \
      --output_dir /root/autodl-tmp/ct_v2_smoking_gun
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from methods.ecgtwin_gen.center_token.model import CenterTokenPerBlock  # noqa: E402
from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402
from util.lead_utils import ECGTWIN_TO_PTBXL_INDICES  # noqa: E402
from scripts.triple_labels.label_schemes import CLASS_NAMES_SUPER5  # noqa: E402

sys.path.insert(0, str(REPO / "model" / "DeepECG" / "notebooks"))
from EfficientNetv2 import EfficientNet1DV2  # noqa: E402


def load_ct_v2(ckpt_path: str, num_blocks_expected: int = None) -> CenterTokenPerBlock:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["center_token"]
    n_blocks = len([k for k in sd.keys() if k.startswith("embeddings.")])
    if num_blocks_expected is not None:
        assert n_blocks == num_blocks_expected, \
            f"ckpt has {n_blocks} blocks, expected {num_blocks_expected}"
    dim = sd["embeddings.0"].shape[0]
    ct = CenterTokenPerBlock(dim=dim, num_blocks=n_blocks)
    ct.load_state_dict(sd)
    ct.eval()
    return ct


def register_per_block_hooks(wrapper, center_token: CenterTokenPerBlock):
    """Register pre-hooks that add per-block CT to AdaLN driver `c`."""
    hooks = []
    for i, block in enumerate(wrapper.noise_predictor.blocks):
        def make_fn(idx):
            def fn(module, args):
                x, c, c2 = args[0], args[1], args[2]
                rest = args[3:]
                c = c + center_token(idx, c.size(0)).to(c.device)
                return (x, c, c2) + rest
            return fn
        hooks.append(block.register_forward_pre_hook(make_fn(i)))
    return hooks


# ─────────────────────────────────────────────
# G1 — Per-block AdaLN move
# ─────────────────────────────────────────────

def gate_g1(wrapper, ct_dict, ref_pt, n_probes=20, device="cuda:0"):
    """For each center, compute per-block rel-diff = ||CT_block|| / mean ||c_block||.

    c_block at each DiT block = t_emb + ib_projector(base_vector). For per-block
    CT, the additive perturbation is fixed regardless of input — so rel-diff is
    constant per (center, block) and we just need a representative ||c_block||
    for the denominator.

    We capture c_block by hooking each DiTBlock during a vanilla forward (no
    CT added).
    """
    samples = torch.load(ref_pt, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(42)
    captured_cs = [[] for _ in wrapper.noise_predictor.blocks]

    def make_capture(idx):
        def fn(module, args):
            x, c, c2 = args[0], args[1], args[2]
            captured_cs[idx].append(c.detach().clone())
        return fn

    capture_hooks = [
        block.register_forward_pre_hook(make_capture(i))
        for i, block in enumerate(wrapper.noise_predictor.blocks)
    ]

    n_total = len(samples)
    indices = rng.choice(n_total, min(n_probes, n_total), replace=False)
    with torch.no_grad():
        for idx in indices:
            entry = samples[int(idx)]
            ref_latent = entry["data"]
            lab = entry["label"]
            ref_label = {
                "hr": float(lab.get("hr", 75.0)),
                "age": float(lab.get("age", 60.0)),
                "sex": lab.get("sex", "M"),
                "text_embed": lab["text_embed"],
            }
            target_text_embed = lab["text_embed"]
            conditions = wrapper.prepare_conditions(
                ref_latent=ref_latent, ref_label=ref_label, batch_size=1,
                target_text_embed=target_text_embed,
                target_hr=ref_label["hr"], target_age=ref_label["age"],
            )
            _ = wrapper.ddpm_sample(conditions=conditions, batch_size=1,
                                    num_inference_steps=5)  # short, just need c
    for h in capture_hooks:
        h.remove()

    c_norms_per_block = []
    for cs in captured_cs:
        if not cs:
            c_norms_per_block.append(0.0)
            continue
        # cs is list of (B=1, hidden=256) tensors over many steps & samples
        all_c = torch.cat([c for c in cs], dim=0)  # (N, 256)
        c_norms_per_block.append(all_c.norm(dim=1).mean().item())

    results = {}
    for center, ct in ct_dict.items():
        per_block_rel = []
        for b in range(ct.num_blocks):
            ct_norm_b = ct.embeddings[b].data.norm(p=2).item()
            c_norm_b = c_norms_per_block[b]
            rel = ct_norm_b / max(c_norm_b, 1e-8)
            per_block_rel.append(rel)
        results[center] = {
            "per_block_rel_diff": per_block_rel,
            "blocks_above_0.05": int(sum(1 for r in per_block_rel if r > 0.05)),
            "max_rel_diff": float(max(per_block_rel)),
            "min_rel_diff": float(min(per_block_rel)),
        }
    results["_c_norms_per_block_baseline"] = c_norms_per_block
    return results


# ─────────────────────────────────────────────
# Generate K signals with a given CT
# ─────────────────────────────────────────────

def generate_with_ct(wrapper, samples, ref_indices, prompt_embed,
                     center_token, n_per_setup, num_steps, device, seed=42):
    """For each (ref_idx, seed), generate one signal. Returns (N, 12, 1000) PTBXL."""
    hooks = []
    if center_token is not None:
        hooks = register_per_block_hooks(wrapper, center_token)

    out_signals, out_latents = [], []
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        for ref_idx in ref_indices[:n_per_setup]:
            entry = samples[int(ref_idx)]
            ref_latent = entry["data"]
            lab = entry["label"]
            ref_label = {
                "hr": float(lab.get("hr", 75.0)),
                "age": float(lab.get("age", 60.0)),
                "sex": lab.get("sex", "M"),
                "text_embed": lab["text_embed"],
            }
            conditions = wrapper.prepare_conditions(
                ref_latent=ref_latent, ref_label=ref_label, batch_size=1,
                target_text_embed=prompt_embed,
                target_hr=ref_label["hr"], target_age=ref_label["age"],
            )
            latent_gen = wrapper.ddpm_sample(
                conditions=conditions, batch_size=1,
                num_inference_steps=num_steps,
            )
            ecg_et = wrapper.decode_latent(latent_gen)  # (1, 1024, 12)
            ecg_ct_1024 = ecg_et.transpose(1, 2)  # (1, 12, 1024)
            ecg_ct_1000 = F.interpolate(ecg_ct_1024, size=1000, mode="linear",
                                        align_corners=True)
            ecg_ptbxl = ecg_ct_1000[:, ECGTWIN_TO_PTBXL_INDICES, :]
            flat = ecg_ptbxl.reshape(1, -1)
            mean = flat.mean(dim=1, keepdim=True)
            std = flat.std(dim=1, keepdim=True).clamp(min=1e-8)
            ecg_norm = (ecg_ptbxl - mean.unsqueeze(-1)) / std.unsqueeze(-1)
            out_signals.append(ecg_norm.cpu().numpy())
            out_latents.append(latent_gen.cpu().numpy())

    for h in hooks:
        h.remove()
    return (np.concatenate(out_signals, axis=0).astype(np.float32),
            np.concatenate(out_latents, axis=0).astype(np.float32))


# ─────────────────────────────────────────────
# G2 — Decoded signal swap
# ─────────────────────────────────────────────

def gate_g2(wrapper, ct_dict, ref_pt, n_per_setup=8, num_steps=50,
            device="cuda:0", seed=42):
    """For each pair (CT_i, CT_j), generate same-ref+same-prompt+same-seed
    signals and measure rel-diff = ||x_i - x_j|| / ||x_i||.
    """
    samples = torch.load(ref_pt, map_location="cpu", weights_only=False)
    # shared ref pool — pick first n_per_setup STTC-positive (or any) refs
    rng = np.random.default_rng(seed)
    ref_indices = rng.choice(len(samples), n_per_setup, replace=False).tolist()

    # Use a generic STTC prompt (can use any; comparison is per-pair)
    prompt = "t wave inversion|repolarization abnormality"
    prompt_embed = wrapper.get_text_embedding(prompt)

    # Also include "vanilla" (no CT) as a baseline
    setups = {"vanilla": None, **ct_dict}

    sigs_by_setup = {}
    for name, ct in setups.items():
        ct_dev = ct.to(device) if ct is not None else None
        sigs, _ = generate_with_ct(wrapper, samples, ref_indices, prompt_embed,
                                   ct_dev, n_per_setup, num_steps, device, seed=seed)
        sigs_by_setup[name] = sigs
        print(f"  [G2 gen] {name}: {sigs.shape}")

    pair_results = {}
    setup_names = list(setups.keys())
    for i in range(len(setup_names)):
        for j in range(i + 1, len(setup_names)):
            a_name = setup_names[i]
            b_name = setup_names[j]
            a = sigs_by_setup[a_name]
            b = sigs_by_setup[b_name]
            diff = a - b
            rel_per_sample = (np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
                              / np.linalg.norm(a.reshape(a.shape[0], -1), axis=1).clip(min=1e-8))
            key = f"{a_name}_vs_{b_name}"
            pair_results[key] = {
                "mean_rel_diff": float(rel_per_sample.mean()),
                "median_rel_diff": float(np.median(rel_per_sample)),
                "min_rel_diff": float(rel_per_sample.min()),
                "max_rel_diff": float(rel_per_sample.max()),
                "n_samples": int(len(rel_per_sample)),
            }
    return pair_results, sigs_by_setup


# ─────────────────────────────────────────────
# G3 — Inter-center cosine matrix vs Gaussian noise
# ─────────────────────────────────────────────

def gate_g3(ct_dict, n_random=1000, seed=42):
    """Pairwise cosine of trained CTs (flattened) vs Gaussian baseline of
    matched norm.
    """
    rng = np.random.default_rng(seed)
    centers = list(ct_dict.keys())
    flats = {c: ct_dict[c].flat().cpu().numpy() for c in centers}
    norm = np.linalg.norm(flats[centers[0]])
    dim = flats[centers[0]].shape[0]
    print(f"  [G3] flat dim={dim}, ref norm={norm:.4f}, n_centers={len(centers)}")

    # Pairwise cosine of trained CTs
    cos_matrix = np.zeros((len(centers), len(centers)))
    pair_cos = []
    for i, ci in enumerate(centers):
        for j, cj in enumerate(centers):
            v1, v2 = flats[ci], flats[cj]
            c = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            cos_matrix[i, j] = c
            if i < j:
                pair_cos.append(c)

    # Random Gaussian baseline
    rand_pair_cos = []
    for _ in range(n_random):
        v1 = rng.normal(size=dim).astype(np.float32)
        v1 = v1 * (norm / np.linalg.norm(v1))
        v2 = rng.normal(size=dim).astype(np.float32)
        v2 = v2 * (norm / np.linalg.norm(v2))
        c = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        rand_pair_cos.append(c)

    abs_pair_cos = [abs(c) for c in pair_cos]
    abs_rand = [abs(c) for c in rand_pair_cos]
    mw = mannwhitneyu(abs_pair_cos, abs_rand, alternative="greater")

    return {
        "centers": centers,
        "cos_matrix": cos_matrix.tolist(),
        "off_diag_pair_cos": pair_cos,
        "off_diag_median": float(np.median(pair_cos)),
        "off_diag_mean": float(np.mean(pair_cos)),
        "off_diag_max_abs": float(max(abs(c) for c in pair_cos)),
        "random_pair_cos_median": float(np.median(rand_pair_cos)),
        "random_pair_cos_max_abs": float(max(abs(c) for c in rand_pair_cos)),
        "mann_whitney_u_stat": float(mw.statistic),
        "mann_whitney_p_value": float(mw.pvalue),
        "interpretation": (
            "trained > random" if mw.pvalue < 0.05
            else "trained ~ random (CTs not separable from noise)"
        ),
    }


# ─────────────────────────────────────────────
# G4 — Victim probability shift
# ─────────────────────────────────────────────

def load_super5_victim(ckpt_path: str, device: str):
    model = EfficientNet1DV2(
        variant="s_v2", input_channels=12, num_classes=5,
        activation="leaky_relu", stochastic_depth_prob=0.304,
        dropout_rate=0.0, use_se=True, norm_type="batch",
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def gate_g4(victim_model, sigs_by_setup, ct_centers, device="cuda:0"):
    """For each pair (CT_i, CT_j), measure max per-class probability shift on
    super5 victim.
    """
    probs_by_setup = {}
    for name, sigs in sigs_by_setup.items():
        # sigs: (N, 12, 1000) PTBXL order, z-scored
        # super5 victim takes (B, 12, 250) at 100 Hz — center crop
        crop = sigs[:, :, 375:625]  # 250 samples around center
        x = torch.from_numpy(crop).float().to(device)
        with torch.no_grad():
            logits = victim_model(x)
            probs = torch.sigmoid(logits).cpu().numpy()  # (N, 5)
        probs_by_setup[name] = probs

    pair_shifts = {}
    for i, ci in enumerate(ct_centers):
        for j in range(i + 1, len(ct_centers)):
            cj = ct_centers[j]
            pi = probs_by_setup[ci]
            pj = probs_by_setup[cj]
            per_sample_class_shift = np.abs(pi - pj)  # (N, 5)
            mean_shift = per_sample_class_shift.mean(axis=0)  # (5,)
            max_shift_class = int(np.argmax(mean_shift))
            pair_shifts[f"{ci}_vs_{cj}"] = {
                "mean_per_class_shift": mean_shift.tolist(),
                "class_names": list(CLASS_NAMES_SUPER5),
                "max_shift_class": CLASS_NAMES_SUPER5[max_shift_class],
                "max_shift_value": float(mean_shift.max()),
            }

    # Also vanilla vs each CT
    if "vanilla" in probs_by_setup:
        for ci in ct_centers:
            pi = probs_by_setup[ci]
            pv = probs_by_setup["vanilla"]
            per_sample_class_shift = np.abs(pi - pv)
            mean_shift = per_sample_class_shift.mean(axis=0)
            max_shift_class = int(np.argmax(mean_shift))
            pair_shifts[f"{ci}_vs_vanilla"] = {
                "mean_per_class_shift": mean_shift.tolist(),
                "class_names": list(CLASS_NAMES_SUPER5),
                "max_shift_class": CLASS_NAMES_SUPER5[max_shift_class],
                "max_shift_value": float(mean_shift.max()),
            }
    return {"per_setup_mean_probs": {k: v.mean(axis=0).tolist() for k, v in probs_by_setup.items()},
            "pair_shifts": pair_shifts,
            "class_names": list(CLASS_NAMES_SUPER5)}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="/root/autodl-tmp/center_token_super5_v2")
    ap.add_argument("--ref_pt",
                    default="/root/autodl-tmp/center_token_super5/extra_k200.pt")
    ap.add_argument("--centers", default="extra,nin,geo")
    ap.add_argument("--victim_ckpt",
                    default="/root/autodl-tmp/triple_labels/super5/best_model.pt")
    ap.add_argument("--output_dir", default="/root/autodl-tmp/ct_v2_smoking_gun")
    ap.add_argument("--n_g1_probes", type=int, default=20)
    ap.add_argument("--n_g2_per_setup", type=int, default=8)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    centers = args.centers.split(",")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nCT v2 SMOKING GUN GATES G1-G4\n{'='*60}")

    # Load wrapper once
    print("\n[init] loading ECGTwinWrapper")
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=True,
                             load_text_model=True)
    num_blocks = len(wrapper.noise_predictor.blocks)
    print(f"[init] DiT blocks: {num_blocks}")

    # Load all 3 CTs
    print(f"\n[init] loading {len(centers)} CT v2 ckpts")
    ct_dict = {}
    for center in centers:
        ckpt_path = Path(args.ckpt_root) / f"{center}_k200" / "center_token_smoothed.pth"
        ct = load_ct_v2(str(ckpt_path), num_blocks_expected=num_blocks)
        ct_dict[center] = ct
        print(f"  {center}: norms=[{','.join(f'{n:.3f}' for n in ct.norms)}] "
              f"total={ct.total_norm:.3f}")

    results = {}

    # G1
    print("\n[G1] AdaLN move (per-block rel-diff > 0.05 in ≥2 of 7 blocks)")
    t0 = time.time()
    g1 = gate_g1(wrapper, ct_dict, args.ref_pt, n_probes=args.n_g1_probes,
                 device=args.device)
    print(f"  ({time.time()-t0:.0f}s)")
    print(f"  baseline c_norms_per_block: "
          f"{[round(n, 2) for n in g1['_c_norms_per_block_baseline']]}")
    for center in centers:
        r = g1[center]
        print(f"  {center}: blocks_above_0.05={r['blocks_above_0.05']}/{num_blocks}  "
              f"max_rel={r['max_rel_diff']:.4f}  "
              f"per_block_rel={[round(x, 3) for x in r['per_block_rel_diff']]}")
    results["G1"] = g1

    # G2
    print(f"\n[G2] Decoded swap (rel-diff > 0.10 vs v1 0.006-0.012)")
    t0 = time.time()
    g2_pair_results, g2_sigs = gate_g2(wrapper, ct_dict, args.ref_pt,
                                       n_per_setup=args.n_g2_per_setup,
                                       num_steps=args.num_steps,
                                       device=args.device, seed=args.seed)
    print(f"  ({time.time()-t0:.0f}s)")
    for k, v in g2_pair_results.items():
        marker = " ✓" if v["mean_rel_diff"] > 0.10 else " ✗"
        print(f"  {k}: mean_rel={v['mean_rel_diff']:.4f}  "
              f"median={v['median_rel_diff']:.4f}{marker}")
    results["G2"] = g2_pair_results

    # G3
    print("\n[G3] Inter-center cosine matrix (off-diag < 0.5, p<0.01 vs Gaussian)")
    g3 = gate_g3(ct_dict, n_random=1000, seed=args.seed)
    print(f"  off-diag median: {g3['off_diag_median']:.4f}")
    print(f"  off-diag max |cos|: {g3['off_diag_max_abs']:.4f}")
    print(f"  random_pair median: {g3['random_pair_cos_median']:.4f}")
    print(f"  random_pair max |cos|: {g3['random_pair_cos_max_abs']:.4f}")
    print(f"  Mann-Whitney p(trained_abs > random_abs): {g3['mann_whitney_p_value']:.4e}")
    print(f"  → {g3['interpretation']}")
    results["G3"] = g3

    # G4
    print("\n[G4] Victim shift (≥1 super5 class shifts > 0.05 between CTs)")
    if Path(args.victim_ckpt).exists():
        victim = load_super5_victim(args.victim_ckpt, args.device)
        g4 = gate_g4(victim, g2_sigs, centers, device=args.device)
        print(f"  per-setup mean probs ({CLASS_NAMES_SUPER5}):")
        for setup, probs in g4["per_setup_mean_probs"].items():
            print(f"    {setup:<12}: " + " ".join(f"{p:.3f}" for p in probs))
        for k, v in g4["pair_shifts"].items():
            marker = " ✓" if v["max_shift_value"] > 0.05 else " ✗"
            print(f"  {k}: max_shift={v['max_shift_value']:.4f} on "
                  f"'{v['max_shift_class']}'{marker}")
        results["G4"] = g4
    else:
        print(f"  [skip] victim ckpt missing at {args.victim_ckpt}")
        results["G4"] = None

    # Aggregate
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    g1_pass = all(g1[c]["blocks_above_0.05"] >= 2 for c in centers)
    g2_pass = all(v["mean_rel_diff"] > 0.10
                  for k, v in g2_pair_results.items()
                  if "vanilla" not in k)
    g3_pass = (g3["off_diag_median"] < 0.5 and g3["mann_whitney_p_value"] < 0.01)
    g4_pass = (results["G4"] is not None and any(
        v["max_shift_value"] > 0.05 for v in results["G4"]["pair_shifts"].values()
    ))
    print(f"  G1 (AdaLN move):     {'PASS' if g1_pass else 'FAIL'}")
    print(f"  G2 (decoded swap):   {'PASS' if g2_pass else 'FAIL'}")
    print(f"  G3 (cos-matrix):     {'PASS' if g3_pass else 'FAIL'}  (data-driven)")
    print(f"  G4 (victim shift):   {'PASS' if g4_pass else 'FAIL'}")
    mandatory_pass = g1_pass and g2_pass and g4_pass
    print(f"\n  MANDATORY (G1+G2+G4): {'GREEN' if mandatory_pass else 'RED'}")
    print(f"  → {'proceed to Iter 2 downstream eval' if mandatory_pass else 'CT v2 fails signal-level — accept Plan Rev 12'}")

    # Save
    summary = {
        "centers": centers,
        "g1_pass": g1_pass,
        "g2_pass": g2_pass,
        "g3_pass": g3_pass,
        "g4_pass": g4_pass,
        "mandatory_pass": mandatory_pass,
        "results": {
            "G1": g1,
            "G2": g2_pair_results,
            "G3": g3,
            "G4": results["G4"],
        },
    }
    with open(Path(args.output_dir) / "smoking_gun_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(Path(args.output_dir) / "g2_signals.npz",
                        **{k: v for k, v in g2_sigs.items()})
    print(f"\n[done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
