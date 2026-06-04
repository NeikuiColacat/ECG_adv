"""Merge gated ECGTwin prompt-token pools.

Each input directory should contain outputs from `gate_prompt_token_synth.py`:

  gated_samples.npz
  gated_samples.latent.npz
  gated_samples.class_trust.json
  gated_samples.ref_meta.json

The merged output keeps the same file naming convention so it can be passed
directly to `scripts/pgd_cross_center/synth_online_at_super5.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from ecg_adv_gen.data.gated_pools import merge_gated_pool_artifacts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dirs", nargs="+", required=True,
                    help="Directories containing gated_samples.* outputs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    result = merge_gated_pool_artifacts(args.input_dirs, args.out_dir, tag=args.tag)

    print(f"[merge] wrote {result.paths.samples}")
    print(f"[merge] wrote {result.paths.latents}")
    print(f"[merge] counts: {result.counts}")
    print(f"[merge] trust: {result.class_trust}")
    print(f"[merge] ref ids: {len(result.ref_record_ids)}")


if __name__ == "__main__":
    main()
