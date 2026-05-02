"""Build ECGTwin text embeddings for prompt JSONL records.

Input JSONL rows must contain `ptbxl_index` and `ecgtwin_prompt_en`.
The output maps PTB-XL row indices to ECGTwin text embeddings `(L, 768)`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from util.ecgtwin_utils import ECGTwinWrapper  # noqa: E402


DEFAULT_JSONL = "/root/autodl-tmp/graduate_project/ptbxl_train2000_ecgtwin_prompts_seed42.jsonl"
DEFAULT_OUT = "/root/autodl-tmp/graduate_project/ptbxl_train2000_ecgtwin_prompt_embeds_seed42.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_jsonl", default=DEFAULT_JSONL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rows = []
    with open(args.prompt_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {args.prompt_jsonl}")

    unique_prompts = sorted({str(r["ecgtwin_prompt_en"]).strip() for r in rows if str(r.get("ecgtwin_prompt_en", "")).strip()})
    print(f"[load] rows={len(rows)} unique_prompts={len(unique_prompts)}", flush=True)

    wrapper = ECGTwinWrapper(device=args.device, load_encoder=False, load_text_model=True)
    prompt_to_embed = {}
    for i, prompt in enumerate(unique_prompts, start=1):
        prompt_to_embed[prompt] = wrapper.get_text_embedding(prompt).detach().cpu().float()
        if i % 100 == 0 or i == len(unique_prompts):
            print(f"[embed] {i}/{len(unique_prompts)}", flush=True)

    index_to_embed = {}
    metadata = {}
    for r in rows:
        idx = int(r["ptbxl_index"])
        prompt = str(r["ecgtwin_prompt_en"]).strip()
        if not prompt:
            continue
        index_to_embed[idx] = prompt_to_embed[prompt].clone()
        metadata[idx] = {
            "ecg_id": int(r["ecg_id"]),
            "prompt": prompt,
            "source": r.get("source"),
            "confidence": r.get("confidence"),
            "super5": r.get("super5"),
            "original_report": r.get("original_report"),
        }

    out = {
        "prompt_jsonl": args.prompt_jsonl,
        "n_rows": len(rows),
        "n_unique_prompts": len(unique_prompts),
        "index_to_embed": index_to_embed,
        "metadata": metadata,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"[done] wrote {len(index_to_embed)} embeddings -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
