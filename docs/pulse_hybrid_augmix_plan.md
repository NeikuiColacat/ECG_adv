# PULSE hybrid AugMix — development experiment

Authorized 2026-09-14: four fixed center-specific K500 training sets; CLIP
blocks 19–22 q/v LoRA (r8/alpha16), full projector training, frozen language
model. At most four free GPUs, one task per GPU; no wall-time limit. No Hydra,
SimCLR or adversarial objective in this first ablation.

## Evidence boundary

The user explicitly redesignated non-K500 PN2021 as development validation
for Optuna. This supersedes the no-target-heldout-selection rule **only for
this new development study**, never retrospectively for historical evidence.
PN2021-C views of these records are also development data when used for tuning.
Do not advertise these scores as an independent external test or source-only
domain generalization. Verify record exclusion; patient-level independence
requires actual patient identifiers and must not be inferred from record hashes.

## Inputs and objectives

- Keep native500 physical-mV waveform, lead mapping and original renderer/processor.
- Clean-only, single-image-chain and three-image-chain arms use identical K500
  identities, initialization, seed, record schedule and optimizer-step budget.
- Two independent augmented views for JSD. In each view, sample 1–3 waveform
  operators with replacement, render once, then sample 1–3 image operators per
  chain. All branches in that view share the same corrupted waveform.
- Dirichlet(1) chain weights, Beta(1,1) residual against that view's waveform
  render. This is a project-specific hybrid topology, not original AugMix parity.
- Image pool: mild paper texture/tint, red-grid fading, tone, smooth shadow.
  No geometric displacement, cross-patient blending, pasted text or lead swaps.
- Clean answer-token CE + configurable JSD across three aligned teacher-forced
  vocabulary distributions; not five-class probability JSD or SimCLR.
- Historical schema 1–3 and its JSD=12 remain unchanged; new schema 4 is explicit.

## Search and reporting

Plan: eight common Optuna trial recipes, equal single/three search budgets;
clean control shares learning-rate candidates. Search LR, projector LR,
JSD and mild augmentation strength; width is an ablation, not a tuned parameter.
Screening uses 25 optimizer steps, then refits the selected shared recipe from
the source weights for 100 steps; all arms use effective batch 16. Search bounds
live in `configs/train/pulse_hybrid_search.yaml`; search helpers remain separate
from GPU scheduling and generation. Launch with the managed
`configs/experiments/pulse_hybrid_search.yaml` entrypoint.
Selection score: per-center 0.5 clean macro-F1 + 0.25 mean waveform-corruption
macro-F1 + 0.25 mean image-corruption macro-F1; then equal centers and equal
clean/single/three arms. Report the three families separately, including clean
regressions. This balanced development objective is not a clinical utility score.
Freeze and report the search space and objective before the first trial.
Distinguish matched-recipe width comparison from independently selected best arms.
Report macro-F1, exact match, Hamming and parser failure, clean/corrupt separately,
equal view/center weighting, paired uncertainty and actual GPU-hours.

## Execution gates

CPU contracts → dry-run → real native500 image visual review → single-GPU
gradient/packing/checkpoint smoke → development evaluator admission → managed
finite HPO and four-center evaluation. Full benchmark does not start before these
gates. Recompute trained vision features, never use frozen-feature caches.

Augmented tensors remain online; temporary renderer output goes to the scoped
RAM directory. Keep indispensable adapters, predictions and provenance in the
existing external data root, never in Git. Trackio is local, one-way metrics only;
no automatic image/data upload. After a real background launch, verify progress
and hand off without agent polling.

## Implementation checkpoint — 2026-09-14

Completed: schema-4 training integration, four pure-Torch image operators,
shared-waveform image-branch topology, configurable JSD, finite YAML search
space, Optuna ask/tell/scoring helpers, CPU contracts and training dry-run.
Full project tests: **647 passed, 14 skipped**. PULSE focused tests: **34 passed**.
The actual GPU gradient/checkpoint smoke has **not** run; all eight devices were
occupied by other users at the final resource check. No real HPO trial started.

The fixed four-center 512-record development cohort and original predictions
passed the retained parent hash checks. Each center's K500 record overlap is zero.
Receipt: `/home/linbinhao/ECG_adv_data/runs/pulse_hybrid_20260914/admissions/cohort_admission.json`.
Patient IDs are absent in these manifests: patient-level independence is not proven.
One real Ningbo K500 ECG clean/single/three preview was visually checked; it shows
expected noise/lead masking, unchanged layout, and no independent-waveform blending.
Preview is ephemeral: `/dev/shm/linbinhao-pulse-hybrid/hybrid_preview.png`.

Optuna 5.0.0 is installed in ECGTwin. Trackio 0.37.1 is in the isolated
`/home/linbinhao/ECG_adv_data/envs/pulse_tracking` environment (188 MB); local
scalar-only write admission passed in RAM, no server or external upload requested.
PULSE torch/transformers packages are unchanged.

Full CPU test command uses process-local environment settings:

```bash
LD_LIBRARY_PATH=/home/linbinhao/miniforge3/envs/ECGTwin/lib \
TMPDIR=/dev/shm/linbinhao-pulse-hybrid \
PYTHONPYCACHEPREFIX=/dev/shm/linbinhao-pulse-hybrid/pycache \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/linbinhao/miniforge3/envs/ECGTwin/bin/python -m pytest -q
```

The private RAM temp directory must exist. This avoids stale pre-move bytecode
paths and SSD-dependent temp-disk test failures. The library path fixes the
Optuna/SQLite ICU versus old system libstdc++ load-order conflict when Torch is
imported first. Do not change system libraries or blindly inherit this setting
into the separate PULSE environment.

## Executable workflow checkpoint

The managed evaluator, finite scheduler, local scalar mirror and final paired
report are now connected. Updated full CPU suite: **657 passed, 14 skipped**;
PULSE focused suite: **36 passed**. This does not replace the GPU admission.

Execution is strictly gated:

1. One GPU at a time: clean/single/three two-step training, three-chain exact
   fresh-process resume, then four-record four-arm author-generation admission.
2. Eight shared recipes: 25-step K500-pool screening; 128 fixed non-K500 records
   per center, clean + four fixed waveform composites + four image operators.
3. Best shared recipe: retrain all three arms from source for 100 steps, then
   evaluate 512 records/center on clean + all 20 waveform composites + four
   image operators. Four centers, not a full PN2021 population claim.
4. Persist per-condition metrics and paired record-bootstrap intervals. These
   intervals do not account for hyperparameter selection or training-seed variance.

Visual token packing and generated token IDs must match the author's path for
all four arms. Fresh original predictions must also match the frozen parent on
clean/waveform views. LoRA weights are **not merged** in this version: the only
LoRA modules are in CLIP; the large language decoder has no LoRA overhead.
Any future weight-merging optimization needs its own numerical admission.

Screening checkpoints live in RAM; final checkpoints and all evaluation
predictions are durable. Screening recipes, result hashes and scalar histories
are mirrored durably, but RAM loss can require rerunning an incomplete trial.
No automatic deletion or failed-run overwrite. Exact live children may be
adopted by PID/start-time/boot identity; failed partial children require review.

The workflow uses the existing launcher and resource primitives. Runtime-generated
child YAMLs may change only the finite search fields and declared arm/center/budget;
they are preserved with the full copied closure. They are not one-off entrypoints.

Control/state root: `/home/linbinhao/ECG_adv_data/runs/pulse_hybrid_20260914/state`.
`control.json` supports `paused`, `max_workers` (1–4) and the candidate GPU list.
Candidates are not allocations: only devices with no compute PID, sufficient
VRAM and at least 30 seconds idle are admitted. All smoke jobs remain serial.
SIGTERM drains the coordinator without killing its running children. Never edit
Python sources while this workflow is live; admitted source hashes are enforced.

GPU admission completed on 2026-09-15: clean/single/three two-step runs passed,
with nonzero visual LoRA/projector updates and unchanged frozen parameters.
The three-chain fresh-process resume matched exactly (maximum trainable delta 0).
The four-record, nine-condition, four-arm evaluation passed its result contract;
all arms matched author packing and generated tokens, and original clean/waveform
predictions matched the frozen parent. These are engineering gates, not evidence
of an augmentation benefit. Artifacts: `<run_root>/smoke/`.

The finite Optuna study has started after these gates. Live control is restricted
to candidate GPUs 1, 3, 4, 6 with a four-worker maximum; every new worker still
requires a free device. Check `state/status.json` and each training `progress.json`
for real optimizer progress before claiming the formal launch is verified.

Formal launch verified on 2026-09-15: trial 00 has four live workers on GPUs
1/3/4/6 (Ningbo clean/single/three and Chapman-Shaoxing clean). Every worker has
completed at least one real optimizer step. Single/three first-step time was
67.7/69.8 seconds at effective batch 16, with finite nonzero LoRA/projector
gradients; these startup timings are not a whole-study duration estimate.
The implementation/admission/startup goal is handed off; the eight trials and
final four-center development evaluations remain running, not completed results.

Local Trackio dashboard: `http://127.0.0.1:9091/?project=pulse-hybrid-development`.
The isolated CPU-only server binds localhost, disables sharing/MCP, and uses
`<run_root>/trackio`; launch identity is in `trackio_dashboard_launch.json`.
HTTP availability was verified before any model run. An empty project is expected
until a completed child result is mirrored; this is not a live per-step feed.
The dashboard is not the scheduler status: use `state/status.json` for the queue.
