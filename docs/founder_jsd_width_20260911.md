# ECGFounder JSD width 1 / 2 / 3

User-authorized 2026-09-11. Single-seed development comparison, not paper-final.

Startup verified: launcher PID `931665`, start ticks `277805209`, boot ID
`b895ba8c-06e5-42d7-829a-08f6e9e08769`. Initial status is
`waiting_for_predecessor`, with zero new GPU allocations. This is a startup
snapshot, not a live progress assertion. Validation: 532 tests passed,
14 skipped; 34-file portable parent config closure dry-run passed.
Receipt: `/home/linbinhao/ECG_adv_data/runs/founder_jsd_width_20260911/launch_receipt.json`.

| Width | Work |
| --- | --- |
| 1 | New training and full evaluation in each of four logical centers |
| 2 | Reuse frozen managed `dual_jsd_20260911/founder_objective_jsd_{center}_eval` results |
| 3 | New training and full evaluation in each of four logical centers |

Only the number of ECG corruption chains *inside each AugMix strong view*
changes. Keep clean Beta residual, Dirichlet/Beta alpha 0.5, depths 2/3,
clean + two independently sampled AugMix views, Bernoulli JSD weight 12,
source-logit anchor 5, frozen Stage1 head, source checkpoint, K500 identities,
seed, optimizer-step budgets and Stage2 VAE-LHAT unchanged. No SimCLR.
Single-chain is not the old no-clean-mixing control. Three-chain means three
corruption chains, not two corruption chains plus a VAE-generated chain.

Full evaluation: four logical centers, K500 excluded, mapping `555ec85d5b51`,
class order `CD,HYP,MI,NORM,STTC`, clean + 20 PN2021-C views,
drop-all-zero macro AUROC / sklearn AP; center-equal and corruption-view-equal
aggregation, paired deltas against width2.

## Isolated launch surface

- Source: `/home/linbinhao/ECG_manual_refactor_jsd_width_20260911`
- Managed YAML: `configs/experiments/founder_jsd_width_workflow.yaml`
- Output root: `/home/linbinhao/ECG_adv_data/runs/founder_jsd_width_20260911`
- Status: `founder_jsd_width_workflow/training/status.json` under output root.
- Final comparison: `founder_jsd_width_workflow/training/width_comparison.json`.
- Source parity receipt: `source_isolation_proof.json` under output root.

No modifications to the original active PULSE runtime. The successor waits for
`dual_jsd_20260911/dual_jsd_workflow_handoff` to complete and for both coordinator
processes to exit before using up to the same four authorized devices 0,1,5,7.
All launches re-check availability and reserve a device even during CPU model
loading. Failed or interrupted children are preserved, not overwritten/retried.
The background coordinator performs bounded checks; no ongoing LLM polling.
This document is a plan/provenance pointer, not proof that the new jobs finished.

Existing width2 center-equal metrics (proportions): clean AUROC 0.9167521591,
clean AP 0.7179699188, corrupted AUROC 0.8967984437, corrupted AP 0.6826762110.
No width1/3 performance conclusion is available at queue preparation time.
