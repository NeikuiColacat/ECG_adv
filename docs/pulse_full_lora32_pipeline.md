# PULSE full-LoRA managed pipeline

This development workflow is launched through
`configs/experiments/pulse_full_lora32_pipeline.yaml`. The user explicitly
authorized RAM staging followed by HDD archival on September 27. Runtime output
is `/dev/shm/linbinhao-pulse-hybrid/full_lora32_20260927_audited4gpu/run`; temporary
files use its sibling `tmp/`. No large runtime output is redirected to the SSD.
The final archive is
`/data/linbinhao/ecg_llm_runs/pulse_full_lora32_pipeline_20260927_audited4gpu`, with
the unchanged output tree under `run/` and `archive_receipt.json` alongside it.

The coordinator runs a bounded four-stage sequence:

1. four-center smoke and fresh-process resume admission
   (four optimizer steps for each trained arm);
2. eight matched trainings (four centers, width 1 and width 3);
3. one full-cohort evaluation per center after excluding that center's fixed
   K500 records;
4. four-center comparison and immutable result validation.

The training scope is `llm_all_linear_lora32_clip_last4_qv_lora8_projector`.
The evaluation suite is GPU C5 severity 5 with clean plus five image
conditions and original/single/three model arms. The scheduler admits only
devices in the explicit `PULSE_GPU_ALLOWLIST`, rejects devices with active
compute processes, and caps concurrent children at four for this RAM-staged run.
Each child uses one explicit device; dispatch admits at most one new loader per
30-second poll, rechecking RAM and cgroup headroom before the next launch.
The single-GPU performance smoke must pass before the four-device workflow is
started. Every center repeats native token parity during its own preflight.

## September 27 integrity and performance audit

The former preflight was drained after 11 of 16 smoke jobs, before formal
training. Its 714 files (18,209,393,834 bytes) were byte/SHA256 verified in
`/data/linbinhao/ecg_llm_runs/pulse_full_lora32_pipeline_20260927_repaired_retry2_audit_drained`.
Its original failed/interrupted status is preserved; archival does not promote
that run to complete experimental evidence. Its task RAM was released only
after archive publication and verification.

The evaluator now checks the locked parent protocol and cohort SHA256, exact
four-center population sizes, binary labels versus label names, parsed fields
against original responses, matched training protocols and actual K500 IDs,
metric-file integrity coverage, cohort digests and processor recipe digests.
An independent canonical-loader audit rechecked all 39,879 native500 source
records, ref-excluded sets, source identities and labels. The audit receipt is
`/home/linbinhao/ECG_adv_data/audits/pulse_four_center_repair_20260927/independent-data-audit.json`.

The optional `arm_major_cached_prompt_v1` execution path retains FP16 B1
generation and recomputes each adapted vision tower for every view. It caches
only frozen text embeddings, batches exact adapter copies, and evaluates the
six views before switching arms. `performance_smoke: true` compares all raw
token IDs and decoded answers against the reference path on all four smoke
records, alternates timing order, and exposes a bounded cudaProfilerApi trace
after warmup. Final evaluation repeats the six-view token check on its first
record. A mismatch fails admission. No power limit or driver setting is changed.

`arm_major_fp16_adapters_v2` additionally retains temporary FP16 inference
storage for LoRA/projector linear operands. This avoids PEFT repeatedly
promoting activations to the FP32 checkpoint storage type only for autocast to
convert them back. The exact original FP32 parameter buffers are restored in
`finally`, including exceptions; training/checkpoint precision is unchanged.
Each smoke record must match the original FP32-storage/autocast implementation
at every generated token before this path is admitted. This is not a LoRA
merge or quantization change. The managed four-center pipeline selects v2.

Native admission completed on September 27: all 72 answer pairs (four records,
six views, three arms) matched at every raw generated token. Alternating-order
timing totaled 64.98 s for reference generation and 56.54 s for v2: 13.0% lower
latency / 14.9% higher throughput in this small smoke. In the captured final
three records, kernels decreased from 939,783 to 594,663 (36.7%). CUDA kernel,
memcpy and memset union occupancy improved from 59.0% to 60.7%; the maximum
idle gap decreased from 7.95 ms to 4.72 ms. Nsight instrumentation is included.
The 200 ms power sampler observed a 392.36 W peak and about 305 W mean for
samples with SM utilization above 50%, across both methods. These measurements
do not establish sustained 450 W, zero idle gaps or full-cohort throughput.

Receipts and analysis: `profile-v2-summary.json`, `independent-data-audit.json`,
`raw-header-audit.json`, `fp16-inference-tests.log` and `fp16-project-tests.log`
under `/home/linbinhao/ECG_adv_data/audits/pulse_four_center_repair_20260927/`.
All 478,548 selected waveform channels declare mV. The current CPU project
suite reports 838 passed, 5 pre-existing failures and 233 skipped; the focused
C5/workflow suite reports 47 passed. The five failures remain the historical
dual-JSD/visual paths, an elastic-report source-inspection fixture, low SSD
reserve in a cache-staging test and the Optuna/SQLite environment dependency.

The new four-device run is queued with GPU allowlist 2,3,4,5. It still requires
the full four-center smoke/resume grid, eight newly identified 200-step training
runs and complete evaluation rectangles. A running queue is not a completed
experiment, and no final metric is claimed by this audit.

## RAM admission and final archival

Startup requires 40 GiB output budget plus 8 GiB free tmpfs reserve, host
MemAvailable of at least 64 GiB, inherited cgroup reclaimable headroom of at
least 32 GiB, bounded CPU load, and HDD space for the 40 GiB budget above its
40 GiB durable reserve. Runtime dispatch rechecks these floors. GPU allowlist
and idle admission still apply independently. Existing tmpfs files are untouched.

Only after all evaluations and the outer launcher have finalized their result,
closed their logs and validated the file index does the launcher archive. It
copies to a fresh `.partial` directory, fsyncs files/directories, compares every
file's bytes and SHA256, verifies the unchanged indexes and relocated result
references, and publishes the archive. It then releases this task's RAM run
directory. Copy/verification failure leaves RAM intact and preserves the partial
copy for diagnosis. No pre-existing destination is overwritten.

Absolute source references and scientific hashes are preserved. The external
receipt maps the original RAM prefix to the archived `run/` tree; the validators
resolve those references locally and recheck their recorded hashes. The sibling
RAM `archive_status.json` records copying, failure, or completed release. Small
launcher logs and temporary interpreter caches remain outside the run directory.
RAM is volatile until archival succeeds; a host restart can lose an active run.

`processor_input_identity_sha256` is a deterministic `view_identity_v1` digest
over the source record, condition, renderer identity and processor shape/dtype.
It is an input-recipe identity, not a SHA-256 over CUDA tensor bytes. It
replaces per-tile host hashing so the model input remains on the selected GPU;
the result explicitly records this hash mode.

This is development evidence only. It is not an independent external-test or
paper claim. Training results without `model_asset_sha256` are rejected by the
evaluation loader and must be retrained; the old full-run outputs that stopped
on original-parent answer drift are therefore not final evidence.

## September 27 repair and baseline definition

The September 25 retry4 run is preserved as failed evidence. Its eight trainings
completed 200 steps, but CPSC stopped after 280 records and Chapman after 347
when batch-one C5 generation was compared literally with the historical paired
parent generation. Those checkpoints also lack the complete model asset
manifest now required for admission. Do not rewrite their identity or resume
them against changed source.

The repaired C5 YAML explicitly selects `original_baseline_mode: current_protocol_v1`.
Original, single and three arms all run with the same batch-one, FP16, deterministic
SDPA generation settings and current model assets. Historical clean predictions
are diagnostic comparisons only; they never replace the newly generated original
answers in the metric rectangle. No permission is inferred from prose in the
parent handoff. Frozen-parent mode still rejects changed labels, while allowing
valid, untruncated label-order or formatting differences.

Each complete C5 result must contain `original_baseline.json`, binding the
generation settings, environment, prompt digest, full model asset manifest and
all original record/view predictions. The validator recomputes the digest from
the prediction batches and requires original-state restoration admission after
switching all adapters. A missing or changed receipt prevents completion.

The smoke grid covers all four centers and includes the first unfinished
CPSC A3512 and Chapman JS09211 records from retry4. These are regression
admission records, not recipe-selection or checkpoint-selection data. Final
cohorts remain the full fixed K500-excluded cohorts.

On a child failure the coordinator verifies launch receipts, same-user PID
start times, boot identity and the launcher session before signalling only
captured task members. It records `state/failure_cleanup.json` and reports
survivors rather than leaving historical PIDs labelled active.

The repair requires new outputs and retraining. CPU tests and launcher dry-runs
do not establish GPU admission or four-center performance. Preserve the 40 GiB
HDD free-disk floor; the eight final checkpoints alone require about 9.1 GiB based
on retry4 file sizes, in addition to smoke checkpoints, temporary saves and
evaluation outputs. The prior SSD blocker is addressed by the explicit RAM/HDD
storage contract, rather than lowering the durable reserve.

Earlier repair validation on 2026-09-27: all 88 focused PULSE cases passed, including live
task-owned subprocess cleanup and stale-PID refusal. The full CPU suite reported
829 passed, 5 failed and 233 skipped. The remaining failures concern two historical
`/data` archive paths, a source-inspection fixture, the cache disk reserve and
the Optuna/SQLite `CXXABI_1.3.15` environment dependency. The launcher dry-run
and whitespace checks passed. GPU admission had not executed at that earlier
checkpoint; the later native admission and queue status are recorded above.
Logs: `/home/linbinhao/ECG_adv_data/audits/pulse_four_center_repair_20260927/`.

RAM/HDD follow-up validation on September 27: 114 focused tests passed; full
CPU suite 833 passed, the same 5 unrelated failures and 233 skipped. The final
launcher dry-run and live RAM/HDD resource admission passed. This establishes
the storage and CPU contracts, not completed GPU admission or final metrics.
Logs: `ram-hdd-focused-tests.log` and `ram-hdd-project-tests.log` in the audit
directory above. Live progress comes from the new RAM run's `state/status.json`.

The first RAM smoke attempt stopped before CUDA on `config.json`: its expected
SHA256 literal had 63 hex digits, missing the `9` after `a1fcab355134208`. The
local file's Git blob `d617980063c5c32e156554c4b441ffd81065ff76` exactly matches
its original Hugging Face download receipt at revision
`3b497b425ade2717f76ba0b14202823b83668011`. The corrected full digest is
`a1fcab3551342089a3893d6fc3897388aaffa8ea54bb2014af875a350bb262f3`. No model file
or failed result was altered. The hash-format test now covers all nine assets;
the stopped attempt remains in its original RAM directory and retry2 is fresh.
All nine real model assets subsequently passed full SHA256 verification, and
all 56 focused training/workflow/evaluation tests passed again after the fix.
See `ram-retry2-model-assets.json` and `ram-retry2-tests.log` in the audit directory.
