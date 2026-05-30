# Thesis Evidence Pack

This directory collects small, thesis-ready evidence files used by `thesis.md`.
Large checkpoints and source experiment arrays stay outside git and are listed in
`docs/artifact_manifest.json`.

## Recommended Thesis Wording

- Use PTB-XL custom seed42 low-sample test as the main undergraduate thesis evidence.
- State that ECGTwin synthetic pretraining followed by real2000 fine-tuning improves low-sample classification.
- Do not attribute the full gain solely to center prompt tokens, because the no-token hard-label control is strong.
- Treat generated ECG figures as qualitative visualization and basic plausibility evidence, not clinical-grade validation.

## Main Low-Sample Result

| method | AUROC | AUPRC | delta AUROC | delta AUPRC |
|---|---:|---:|---:|---:|
| real2000 baseline | 0.8433 | 0.6234 | 0.0000 | 0.0000 |
| no-token hard pretrain -> real fine-tune | 0.8669 | 0.6828 | 0.0236 | 0.0594 |
| center-token hard pretrain -> real fine-tune | 0.8735 | 0.7013 | 0.0302 | 0.0779 |

## Contents

- `tables/low_sample_main_results.md`: Table 6.6 evidence.
- `tables/per_class_main_results.md`: Table 6.7 evidence.
- `tables/low_sample_ablation_results.md`: Table 6.8 evidence.
- `tables/dataset_split_label_distribution.md`: Table 6.2 evidence.
- `raw/ecgtwin_author_repro_summary.json`: Table 6.3 IBE/DiT metric summary.
- `tables/medical_validity_summary.md`: Table 6.4 evidence.
- `tables/inference_benchmark.md`: Table 5.2 evidence.
- `tables/system_function_tests.md`: Streamlit/system test evidence.
- `feature_distribution/feature_distribution_report.md`: real-vs-synth feature distribution analysis.
- `figures/technical_route_diagram.png`: Figure 1.1 candidate.
- `figures/system_architecture_diagram.png`: Figure 4.1 candidate.
- `figures/`: thesis-ready figures and copied loss curves.
- `raw/`: small JSON/CSV source summaries copied for traceability.

Legacy Latent-Hull, teacher-distillation, soft self-distillation, and rerun-only
rows were removed from this committed evidence pack because they are not part of
the current thesis mainline.

## Manifest

- generated_at: `2026-05-05T22:32:55.273763+00:00`
- implementation_repo_commit: `4e565a733e69bebce28ed0fd37b86ae9bd8784bf`
- thesis_repo_commit_before_pack: `f75462b6065c1811f9f0eb09df66fe17c321f75f`
