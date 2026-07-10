# ECGFounder Decoupled VAE-LHAT Four-Center Depth23 Result

Date: 2026-07-09

Status: deprecated protocol-invalid historical observation. The absolute
metrics remain registered for traceability, but this result is prohibited from
paper comparison because its target-adapted initialization used a different
K500 identity.

## Method

The candidate keeps the thesis method family while routing the two robustness
sources separately:

- ECGFounder 12-lead model, full-model fine-tuning, 30,670,389 trainable
  parameters.
- Four target centers: Ningbo, Chapman-Shaoxing, CPSC 2018, and Georgia.
- K=500 current target records per center, seed `20260531`; those current refs
  are excluded from evaluation.
- Fifteen training epochs; only `last_model.pt` is evaluated.
- VAE-LH latent-hull search: M=20, lambda=0.05, 5 optimization steps,
  latent learning rate 0.25, PGD epsilon 2.0, and 160 searched anchors.
- Three clean-anchor AugMix chains, width=3, random depth, official standard
  severity=5, 24 copies, and the five locked PN2021-C operators.
- VAE-LH adversarial waveforms are a separate supervised stream with sample
  scale 0.10. They are not used as the base waveform of the AugMix chains.
- JSD/VAE consistency is disabled for this run.

The implementation surface is commit `b57ee1600e468ce7bc0765708c9d78d7f8f65217`
(`Tune VAE-LHAT AugMix chain-base routing`). The tracked managed config records
the same routing topology, but its default `epochs=10` and `copies=1` do not
exactly replay this one-off `epochs=15`, `copies=24` result.

## Evaluation Protocol

- Mapping: `v7_super5_sjr_rgq_review_20260528` / `555ec85d5b51`.
- Class order: `CD,HYP,MI,NORM,STTC`.
- Corruption profile: official standard severity 5.
- Depth 2: all 10 unordered pairs from the five locked operators.
- Depth 3: all 10 unordered triples from the five locked operators.
- Aggregation: four centers times 20 combinations, 80 center-combination rows
  for the combined depth2+3 mean.
- Primary metric view in this report: ref-excluded, all-zero-kept macro AUROC
  and macro AUPRC.

## Four-Center Descriptive Result

| Composite depth | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop from clean | Observed unmatched difference vs direct fullFT (not for claim) |
|---|---:|---:|---:|---:|
| 2 | 0.9004 / 0.6440 | 0.8307 / 0.5385 | 6.97 / 10.55 pp | +4.33 / +7.55 pp |
| 3 | 0.9004 / 0.6440 | 0.8015 / 0.4942 | 9.89 / 14.98 pp | +4.62 / +7.50 pp |
| 2+3 | 0.9004 / 0.6440 | 0.8161 / 0.5163 | 8.43 / 12.76 pp | +4.48 / +7.53 pp |

For the combined depth2+3 view, the separately observed direct fullFT baseline
is `0.7714 / 0.4411` on PN2021-C and `0.8879 / 0.6103` on clean data. Because
the K500 identities are unmatched, the numeric differences in the last column
are descriptive arithmetic only and are not evidence of improvement. The
previously recorded unmatched comparisons against other baselines are
removed because their unmatched identities do not support comparative claims.

## K500 Lineage Audit

The current run trained on and excluded seed `20260531`, but every center was
initialized from the target-adapted `asrboost_g9` K500 full-FT checkpoint using
seed `20260601`. A target-adapted initialization is already exposed to its K500
records; excluding only the current seed does not remove those records from the
evaluation set.

| Center | Current/evaluation seed | Initialization seed | K500 intersection | Initialization records left in evaluation |
|---|---:|---:|---:|---:|
| Ningbo | 20260531 | 20260601 | 12 | 488 |
| Chapman-Shaoxing | 20260531 | 20260601 | 44 | 456 |
| CPSC 2018 | 20260531 | 20260601 | 55 | 445 |
| Georgia | 20260531 | 20260601 | 34 | 466 |
| Total | — | — | 145 | 1855 / 2000 |

This is not repaired by taking a union exclusion larger than K=500. The locked
protocol instead rejects a different target-adapted initialization and permits
only the same K500 identity or an explicitly source-only initialization.

## Per-Center Combined Depth2+3

| Center | Clean AUROC / AUPRC | PN2021-C AUROC / AUPRC | Drop from clean |
|---|---:|---:|---:|
| Ningbo | 0.9085 / 0.5716 | 0.8258 / 0.4492 | 8.26 / 12.24 pp |
| Chapman-Shaoxing | 0.9171 / 0.5762 | 0.8428 / 0.4726 | 7.42 / 10.36 pp |
| CPSC 2018 | 0.9093 / 0.7359 | 0.8229 / 0.5795 | 8.64 / 15.64 pp |
| Georgia | 0.8668 / 0.6922 | 0.7730 / 0.5639 | 9.38 / 12.82 pp |

## Evidence

Current host resolves `${paths.output_root}` to
`/home/linbinhao/ECG_adv_data/runs`.

- Training root:
  `${paths.output_root}/ecgfounder_vae_lhat_augmix_threechain_locked_k500/ecgf_decoupled_scale0p1_c24_ep15_fourcenter_20260709`
- Evaluation root:
  `${paths.output_root}/pn2021c_ecgfounder_official_s5_depth23_composite/ecgf_decoupled_scale0p1_c24_ep15_fourcenter_20260709`
- Summary CSV:
  `${paths.output_root}/pn2021c_ecgfounder_official_s5_depth23_composite/ecgf_decoupled_scale0p1_c24_ep15_fourcenter_20260709/summary/ecgfounder_decoupled_scale0p1_c24_ep15_depth23_summary.csv`
  (`sha256:32c2c37f6814c69d12a270ec4f3b9dc690c86f3ab0e2d4798407b8f866c17f84`)
- Center CSV:
  `${paths.output_root}/pn2021c_ecgfounder_official_s5_depth23_composite/ecgf_decoupled_scale0p1_c24_ep15_fourcenter_20260709/summary/ecgfounder_decoupled_scale0p1_c24_ep15_depth23_center_summary.csv`
  (`sha256:11c78537769b104c2c2d21e130a4d77dad9659074190e8f8fd65ee30f740534e`)
- Summary JSON:
  `${paths.output_root}/pn2021c_ecgfounder_official_s5_depth23_composite/ecgf_decoupled_scale0p1_c24_ep15_fourcenter_20260709/summary/ecgfounder_decoupled_scale0p1_c24_ep15_depth23_summary.json`
  (`sha256:7b7c22b7fdd1db737253fdb792ce215adebf08aac3a8cc4085f8487d3fbc67a0`)
- Artifact integrity manifest:
  `${paths.output_root}/evidence_records/ecgfounder_decoupled_fourcenter_depth23_20260709/artifact_integrity.json`
  (`33` selected input/output artifacts, `1,129,751,679` bytes;
  manifest `sha256:f4a102770b673d1074540688400f9f2630e3b863908b78a19ea7df66dbe4e9ce`).
- Provenance-gap record:
  `${paths.output_root}/evidence_records/ecgfounder_decoupled_fourcenter_depth23_20260709/provenance_gap.json`
  (`sha256:a28b084a5a6e5eb8b89caa7e5c0df54750988998240f8041f526360fd7c1d3f4`).

The integrity scope includes selected and initialization checkpoints, K500
reference metadata/raw1000 subsets, training logs, clean and PN2021-C result
JSON, the PN2021 manifest, and summary files. It intentionally excludes the
regenerable 20 GiB waveform cache and does not convert this one-off run into a
managed replay.

Each center's `eval_result.json` records the resolved training arguments,
mapping, trainable scope, selected `last_model.pt`, ref-excluded clean metrics,
and source PTB-XL sanity metrics.

## Traceability Limits

- The run predates the current managed run-card contract and does not contain a
  `run_card.json`, `run_file_index.json`, or persisted exact launch command.
  The new provenance-gap/integrity records preserve what can be verified now;
  they are not substitutes for those missing launch records.
- The artifacts do not record an exact Git SHA; `b57ee16` is the matching
  implementation commit and predates the run, but the artifact itself does not
  independently prove code provenance.
- The tracked managed config has the correct model-aware routing but not the
  exact copies/epoch overrides or the one-off `asrboost_g9` initialization.
- The K500 lineage mismatch is protocol-invalid, not merely a missing
  replication. A new managed run with matching current/init/evaluation identity
  (or a source-only initialization) is required before any paper comparison;
  multi-seed replication remains a separate later requirement.
