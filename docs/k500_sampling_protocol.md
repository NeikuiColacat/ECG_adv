# K500 sampling sensitivity, fixed training randomness

The user authorized resampling target-center K500 and two GPUs on 2026-11-04.
This isolated experiment permits adaptation on the three newly drawn K500s;
the original locked splits and all existing experiment outputs stay unchanged.

Frozen grid: three independent uniform-without-replacement K500 draws, four
logical centers, EfficientNet1DV2 and ECGFounder, classic two-stage SimCLR with
contracted VAE-LHAT versus its matched no-VAE arm. 48 training + 48 full
evaluations. The draw namespaces are `k500_sampling_20261104_r0:draw0/1/2`;
base sampling seed 20260501. Nonzero-label eligibility and original clean/cache
quality rules are unchanged. No draw is selected using downstream performance.

Training RNG namespace/comparison group `k500_sampling_20261104_r0`, replicate
0 for every draw and both methods. Effective training seed is fixed within
each backbone/center. Sampling namespaces are separate from training RNG.
Independent draws may overlap; this is not a disjoint-fold experiment.

Each method uses precisely the same 500 records within each center/draw.
The per-center evaluation set is frozen before training as all center records
minus the union of all three K500 draws; the primary view additionally drops
all-zero labels. All draws/models/methods therefore use identical evaluation
records, labels and GPU corruption microbatches. Each run separately records
its own 500 training exclusions and the additional common-cohort exclusions.
No training record from any draw appears in this common evaluation set.

Both arms retain the classic source checkpoint, source-logit anchor Stage 1,
head freeze/unfreeze, two-chain AugMix and Stage-2 clean/rotating-four
corruption supervision. Only the full arm uses the locked VAE-LHAT path.
EfficientNet: Stage 1 1024 steps then 25 epochs/100 supervised steps.
ECGFounder: Stage 1 256 steps then 30 epochs/240 supervised steps.
All training uses only the selected K500, no validation split, last checkpoint.

Clean and 20 depth2/depth3 severity-5 PN2021-C views use existing Torch GPU
operators, raw100 -> GPU500 -> GPU100 and the same model adapters as before.
No new corrupted-waveform cache. Original source/mapping/cache metadata are
verified and new split roots have independent full-content ledgers.

Primary: macro sklearn average precision (`AUPRC` field), drop_all_zero;
also report clean/corrupt macro AUROC/AP. Equal views then equal four centers,
then mean/sample SD across the three sampling-draw means. Preserve per-center
draw values and paired full-minus-no-VAE deltas. This SD measures sampling
sensitivity at a fixed training seed, not training-seed variation. Three draws
and historically tuned development centers do not constitute a paper-final
independent test; no statistical significance or universal robustness claim.

Managed entry: `configs/experiments/k500_sampling/workflow.yaml`.
CPU-only coordinator: `CUDA_VISIBLE_DEVICES=` and `K500_SAMPLING_GPUS=0,1`;
at most one child on each physical GPU 0/1, bound by UUID. Admission checks
idle GPUs, host RAM/load and disk; it waits for foreign work to leave, and
never overwrites, retries or terminates foreign jobs. Source/config guards,
four real-model engineering admissions and a fixed finite grid are required.

External output: `/home/linbinhao/ECG_adv_data/runs/k500_sampling_20261104_r0`.
Status/control: `workflow/training/status.json` and `control.json`.
Final report: `workflow/training/sampling_comparison.json` and
`sampling_metrics.csv`. Record-specific last checkpoints, source snapshots,
hashes and VAE-LHAT diagnostics are preserved in their managed run directories.

## Stopped queue and metadata repair, 2026-11-04

The original queue stopped after two successful EfficientNet/Ningbo/draw0
training children (full and no-VAE), before any evaluation completed. The first
no-VAE evaluation rejected its new ledger because it counted eleven metadata
members: the original ten files plus `RETIREMENT_20261017.md`. All ten original
file hashes and sizes agree with the historical ledger. This was an online
metadata validation bug, not a training failure or insufficient GPU memory.

The repaired online verifier requires the ten canonical paths explicitly,
fully hashes every required file, and reports additional ledger sidecars as
unverified. A missing/replaced path, modified content, symlink, or wrong ledger
hash still fails. The retired waveform is neither read nor certified; existing
full-cache inventory checks still reject inventory differences.

The failed evaluation/logs, two trained checkpoints, four engineering runs,
old source snapshots and original engineering admission remain intact. The
original workflow has no final sampling comparison. A separate managed
evaluation checks the fix without overwriting that failure; see
[handoff validation](simclr_handoff_validation.md) for its exact identity.

The complete 96-child queue is not resumed by this repair. Its admission binds
the old source bytes, so the repaired checkout requires a new admission and an
explicit recovery configuration/receipt defining checkpoint reuse and fresh
evaluation directories. Do not edit the old admission/source hash, erase failed
directories, or relaunch the original queue with implicit retries. The three
new draw ledgers are retained as `content_ledger_final.jsonl`; preliminary
unreferenced ledger drafts are not part of this delivery.
