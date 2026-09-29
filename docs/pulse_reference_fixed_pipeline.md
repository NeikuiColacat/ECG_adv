# Fixed PULSE reference-AugMix pipeline

Authorized on 2026-09-19: no Optuna; four original center-specific K500 sets;
single/three-chain training; original/single/three comparison on the fixed
512 non-K500 records per center; image severity3; at most four free GPUs.

## Recipe

The source is the locked PULSE-7B checkpoint. Train CLIP blocks 19-22 q/v LoRA
rank8/alpha16 and the projector, keeping the language model frozen. The visual
LoRA learning rate is 2e-5 and projector LR is 1e-5. Use effective batch16,
200 optimizer steps (3200 record exposures, 6.4 K500 passes), ten warmup steps,
cosine decay, gradient clipping1, weight decay0, JSD weight3, seed20260919.
Select the last checkpoint, with no target-test checkpoint/hyperparameter
selection. This is a fixed engineering starting point, not proven convergence.

Keep the hybrid topology explicit: 1-3 waveform operators shared by all image
branches in each view, waveform strength0.5, nine original PIL operator formulas
at training severity3, image chain depth1-3, Dirichlet/Beta alpha1, and clean CE
plus two teacher-forced answer-token JSD views. This is not a claim of complete
original AugMix experimental parity.

Evaluation has exactly 1 clean + 5 single-wave + 15 image + 75 joint conditions
per record, three model arms, 512 records on each of four centers: 589,824 arm
predictions in total. Use the same record identities, waveform realizations and
paired image realizations across arms. Keep the historical development exposure
and unknown patient-level independence disclosed. Report per-center and
clean/wave/image/joint metrics, with equal-center aggregation and paired
record-bootstrap intervals; generated labels are not continuous AUROC scores.

## Automatic stages

The single managed entry is `configs/experiments/pulse_reference_fixed_pipeline.yaml`.
The coordinator creates immutable child YAML closures and uses the existing
launcher and resource checks. It does not import or run Optuna.

1. Serial GPU admission: two-step single and three training on four K500
   records, exact fresh-process three-chain resume, then four-record/96-condition
   original/single/three author-packing and token-equivalence evaluation.
2. Eight matched 200-step runs, at most four in parallel, each on a confirmed
   free GPU UUID. Every final model starts again from source, not smoke weights.
3. After all eight training results validate, automatically run the four center
   evaluations, at most four in parallel, 512 records each and all 96 conditions.
4. Automatically validate prediction rectangles and hashes, aggregate metrics,
   bootstrap paired differences, and write the managed workflow result.

An engineering gate failure prevents formal work. A nonfinite update, data or
source-hash mismatch, insufficient resource floor, or failed child is never
handled by silently changing settings, omitting views or overwriting outputs.
Fatal failures leave `failed_or_interrupted` status and the exact error. The
pipeline is automatic on a healthy host, not a promise to recover transparently
from arbitrary hardware/reboot failures. Existing completed work is preserved.

## State and operation

Run root: `/home/linbinhao/ECG_adv_data/runs/pulse_reference_fixed_s3_steps200_20260919`.

- `state/status.json`: phase, live child identities, completed/total jobs, failure.
- `state/control.json`: pause new launches and set worker/candidate-GPU limits.
- `state/launches/`: each child's dry-run, command, PID/start-time/boot identity, log.
- `smoke/`: disposable-in-purpose but protected admission evidence and checkpoints.
- `final/<center>_{single,three}/training/`: progress, history, source and checkpoint.
- `final/<center>_eval/evaluation/`: predictions, metrics and author admission.
- `workflow/training/comparison.json` and `per_condition.csv`: final summary.
- `workflow/training/hybrid_result.json`: completed, integrity-checked workflow receipt.

The top-level launcher is detached from SSH/Codex with stdin closed and logs
redirected; children are likewise independent sessions. Closing the interactive
terminal does not terminate this process tree. Do not edit runtime sources while
it is active. SIGTERM to the coordinator drains existing children; it does not
kill them. Pausing the control file stops new launches, not already-running jobs.
Do not run a second coordinator or use `--force` on a failed directory.

No expanded image dataset is persisted. Temporary files, checkpoints and logs
remain under the home-owned data root; no new `/dev/shm` or system-wide writes.
Minimum available RAM is 64 GiB and free disk is 40 GiB before new admissions.

## Reference performance admission

The pinned imagecorruptions glass loop uses ordered NumPy view assignment.
The accelerated version bulk-draws the same NumPy RNG stream and performs the
same ordered uint8 assignments with CPU-only Numba, preserving the reference's
aliasing semantics rather than replacing them with a different pixel swap.
Tests compare float outputs and post-operation RNG state across every severity
and multiple seeds. Native-size parity and measured timings are required before
the long pipeline is launched. Gaussian filters, severity tables and every other
C15 operator remain unchanged. No Torch, CUDA or driver package was replaced.

Verified before launch: 1700x2200 severity3 float outputs and post-operation
NumPy RNG state were exactly equal. Reference glass blur took 113.98 seconds;
the accelerated path took 2.75 seconds including first JIT compilation (41.5x).
All-severity/multiple-seed small-image equivalence tests also passed.

Live data preflight verified all four centers have exactly 500 unique adaptation
records and 512 evaluation records with zero record-hash overlap. The receipt is
`/home/linbinhao/ECG_adv_data/previews/pulse_reference_fixed_20260919/cohort_preflight.json`,
SHA256 `db5fc94fe7c4a9b6943e38eb265a68d612dfb01205ffcfea71c457360417c2af`.
The adjacent `native500_views.png` was visually checked on a real Ningbo K500
record; solarization/darkening and geometric mixed traces are expected AugMix
effects, not evidence of diagnostic-label preservation.

CPU verification: 793 passed, 14 skipped; the same three unrelated legacy
archive-path/disk-reserve failures remain documented in the September 19 audit.
The fixed DAG, three-arm evaluator/aggregation and reference acceleration tests
passed. Managed dry-run and `git diff --check` passed. GPU admission is performed
automatically as stage 1, not claimed by these CPU checks.

Launch snapshot, 2026-09-19: detached launcher PID2180736 and coordinator
PID2180781 are live. The first single-chain smoke completed both optimizer
steps with finite, nonzero LoRA/projector gradients, and the coordinator then
automatically launched the three-chain smoke. Formal eight-model training had
not started at this snapshot; the remaining resume/evaluation admissions gate
it automatically. All 91 guarded runtime source files were rechecked unchanged
after launch. Use the live state file, not this snapshot, for subsequent status.
