"""Generate Super5 SNOMED-keyed text embeddings via ECGTwin nomic encoder.

Builds a SNOMED-code-keyed dict of pre-encoded text embeddings, aligned with
ECGTwin's training distribution (lowercase, pipe-separated short phrases —
see model/ECGTwin/data/store_embedding_nomic.py + generation_result_by_disease/).

Output schema:
  torch.save({
      "by_snomed":   {snomed_code: tensor (num_reports, 768)},
      "by_class":    {super5_class: tensor (num_reports, 768)},   # fallback
      "prompts_by_snomed": {snomed_code: prompt_str},             # for audit
      "prompts_by_class":  {super5_class: prompt_str},
  }, out_path)

Phase 0.A 12-lead viz sanity is performed via --viz (requires --token_ckpt to
override center_token to zero, then 1 sample/class is generated and rendered).

Usage:
  /root/miniforge3/envs/ECGTwin/bin/python \
    scripts/ecgtwin_gen/super5_text_embeds.py \
    --out /root/autodl-tmp/center_token_super5/super5_text_embeds.pt
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402


# Each SNOMED code → ECGTwin-aligned pipe-separated prompt.
# Vocabulary chosen to match
#   model/ECGTwin/generation_result_by_disease/{normal_ecg, myocardial_infarction,
#     left_bundle_branch_block, right_bundle_branch_block, nstemi, st_elevation_mi,
#     left_ventricular_hypertrophy, atrial_fibrillation, sinus_bradycardia,
#     sinus_tachycardia, atrioventricular_block}/features.json
# and the prompt_propcess() pipeline at
#   model/ECGTwin/data/store_embedding_nomic.py:10
SNOMED_TO_PROMPT = {
    # Plan Rev 11 / Rev 13.2 (2026-04-27): scope narrowed to NORM/MI/STTC only.
    # HYP (10) + CD (23) entries removed because ECGTwin generation fails the
    # digital-GT validation for those classes — see CLAUDE.md "ECGTwin super5
    # generation scope". Records whose only super5-positive class is HYP/CD
    # are filtered out by SUPER5_PRIORITY=["MI","STTC","NORM"] in the prep
    # script (`scripts/ecgtwin_gen/prep_center_dataset_super5.py`).

    # ─── NORM (4 codes) ─────────────────────────────────────────────
    426783006: "sinus rhythm|normal ecg",
    426177001: "sinus bradycardia|slow heart rate",
    427084000: "sinus tachycardia|fast heart rate",
    427393009: "sinus arrhythmia|normal ecg",

    # ─── MI (7 codes) ───────────────────────────────────────────────
    164865005: "myocardial infarction|st elevation|anterior wall",
    164867002: "old myocardial infarction|abnormal ecg",
    57054005:  "acute myocardial infarction|st elevation|anterior wall",
    54329005:  "anterior myocardial infarction|st elevation|anterior wall",
    22298006:  "subacute myocardial infarction|abnormal ecg",
    401303003: "acute anterior myocardial infarction|st elevation|anterior wall",
    233843008: "inferior myocardial infarction|st elevation|inferior wall",

    # ─── STTC (14 codes) ────────────────────────────────────────────
    # Bug-1 fix v2 (S4 panel-tested cleanest):
    #   • DROP "abnormal ecg" suffix (it appears in 355k MIMIC rows so dragging
    #     all embeds toward a common centroid; cos +0.10 across the board).
    #   • DROP any "myocardial" / "infarct" / "ischemia" lexeme (literal MI cluster).
    #   • DROP "nstemi" / "non st elevation" / generic "st elevation" (MI manifold).
    #   • Use morphology-only words: "t wave inversion", "repolarization abnormality",
    #     "st depression". S4 measured "t wave inversion|repolarization abnormality"
    #     vs MI cos = 0.634 (panel min). Note this is still > 0.5 — DiT cross-attn
    #     entanglement makes prompt change only marginally helpful; H4 trust gate
    #     remains the structural fallback if STTC AUROC < 0.55 in Phase 0.4.
    164934002: "t wave inversion|repolarization abnormality",
    164917005: "q wave abnormal",
    111975006: "prolonged qt interval|long qt",
    164931005: "st elevation",
    429622005: "st depression|t wave inversion",
    164930006: "st interval abnormal",
    59931005:  "t wave inversion|repolarization abnormality",
    164861001: "st-t segment changes|repolarization abnormality",
    428750005: "nonspecific st-t changes",
    55930002:  "st changes",
    425623009: "lateral t wave changes|repolarization abnormality",
    425419005: "inferior t wave changes|repolarization abnormality",
    426434006: "anterior t wave changes|repolarization abnormality",
    428417006: "early repolarization",
}


# Per-primary-class fallback prompt.
# Used when a record's primary super5 class has no SNOMED code in
# SNOMED_TO_PROMPT (very rare since the table covers the entire
# SNOMED_TO_SUPER5 keyset, but kept for safety).
SUPER5_FALLBACK_PROMPT = {
    # Plan Rev 11 / Rev 13.2 (2026-04-27): scope = NORM/MI/STTC only.
    # HYP/CD fallback prompts removed; refs whose only super5-positive class
    # is HYP/CD are filtered out at prep time by SUPER5_PRIORITY.
    "NORM": "sinus rhythm|normal ecg",
    "MI":   "myocardial infarction|st elevation|anterior wall",
    "STTC": "t wave inversion|repolarization abnormality",
}


def _encode_unique_prompts(wrapper: ECGTwinWrapper, prompts: dict) -> dict:
    """Encode each unique prompt string ONCE; share the tensor across keys."""
    unique_prompts = sorted(set(prompts.values()))
    print(f"[super5_text_embeds] encoding {len(unique_prompts)} unique prompts "
          f"(over {len(prompts)} keys) ...", flush=True)
    text_to_emb = {}
    for p in unique_prompts:
        emb = wrapper.get_text_embedding(p).detach().cpu()
        text_to_emb[p] = emb
        print(f"  '{p}' → shape {tuple(emb.shape)}")
    out = {}
    for k, p in prompts.items():
        out[k] = text_to_emb[p].clone()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/center_token_super5/super5_text_embeds.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"[super5_text_embeds] loading ECGTwin (text_model only) on {args.device} ...",
          flush=True)
    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=True)

    by_snomed = _encode_unique_prompts(wrapper, SNOMED_TO_PROMPT)
    by_class = _encode_unique_prompts(wrapper, SUPER5_FALLBACK_PROMPT)

    bundle = {
        "by_snomed": by_snomed,
        "by_class":  by_class,
        "prompts_by_snomed": dict(SNOMED_TO_PROMPT),
        "prompts_by_class":  dict(SUPER5_FALLBACK_PROMPT),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"[super5_text_embeds] wrote {out_path}")
    print(f"  by_snomed: {len(by_snomed)} entries")
    print(f"  by_class:  {len(by_class)} entries")
    print(f"  unique by_snomed shapes: "
          f"{sorted(set(tuple(t.shape) for t in by_snomed.values()))}")


if __name__ == "__main__":
    main()
