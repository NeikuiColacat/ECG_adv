# Legacy Archive 2026-05-06

This directory contains code and notes that were outside the final
graduation-project branch mainline.

Archived groups:

| group | reason |
|---|---|
| `methods_ecgtwin_gen/center_token/` | old 256-d AdaLN/base-vector center-token hook, not the active textual-inversion prompt-token method |
| `methods_ecgtwin_gen/style_translator/` | historical style-translator attempt |
| `scripts_ecgtwin_gen/` | old center-token/style-translator entrypoints and sanity scripts |
| `scripts_top/` | old top-level AdvDiff/fine-tune/eval runners |
| `augmix_adv_combo/`, `augmix_validation/` | AugMix side-route runners, not used by the final thesis pipeline |
| `scripts_pgd_cross_center/` | old CT-v2 synth-anchor pilots |
| `tsne_clustering_v1/` | side experiment |
| `tmp_md/` | temporary experiment notes |

Active graduation-project paths remain in `scripts/final_round/`,
`scripts/triple_labels/`, `scripts/ecgtwin_gen/`, `scripts/pgd_cross_center/`,
`methods/ecgtwin_gen/prompt_token/`, `apps/streamlit_ecg_demo/`, and
`scripts/deploy/`. Shared helper directories `scripts/crosscenter_v2/`,
`scripts/crosscenter_tierM/`, and `methods/augmix/` are still kept in the
active tree because current code imports them.
