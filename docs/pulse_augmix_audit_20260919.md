# PULSE AugMix implementation audit, 2026-09-19

## Scope and verdict

Reviewed the September 17-18 uncommitted PULSE image augmentation and joint
evaluation changes on `direction/ecg-llm`, based at `f31ed4a`. The September 18
user request was to use original AugMix training operators and ImageNet-C or
CIFAR-C evaluation operators on top of the five existing waveform operators.

The old C15 implementation did **not** establish that original AugMix fails.
It used faulty custom approximations, not the named reference operators.
Historical checkpoints and run files remain untouched. No training or model
inference was started by this audit. No source/hash guard was removed.

## Confirmed findings

| Priority | Finding in the September 18 source | Consequence and repair |
| --- | --- | --- |
| P1 | `pixelate` used `max(1, scale)` for a scale below one | Exactly identical input/output at every severity; replaced by reference BOX downsampling and NEAREST upsampling. |
| P1 | `equalize` ranked individual pixels rather than equalizing an intensity histogram | A constant image produced 1536 different intensities in a 32x48 fixture; replaced by Pillow `ImageOps.equalize`. |
| P1 | `shot_noise` did not pass its caller's generator to Poisson sampling | Same seed did not replay, and global Torch RNG changed; reference corruptions now have seeded, restored NumPy randomness, including a separately seeded skimage impulse-noise RNG. |
| P1 | JPEG was scalar quantization; glass blur was an elastic warp; defocus was a box filter; weather operators were arbitrary overlays | Replaced all 15 proxies with `imagecorruptions==1.1.2`, including its frost assets and actual JPEG codec. |
| P1 | Training operators used extra residual blending, different posterize bit depths/solarization thresholds and hand-shrunken geometry | Nine PIL operations now use the upstream formulas with explicit severity 3, without a second blend inside each operator. |
| P1 | C15 single-wave condition IDs were looked up in a parent containing only composite-wave predictions | The evaluator would fail on the first new wave condition. Clean parent equivalence remains mandatory; new single-wave IDs are evaluated fresh. |
| P2 | Image RNG keys included the joint-wave condition ID | Image-only and joint conditions used different noise realizations. Reference evaluation keys image RNG by record, operator and severity, shared across all five waves. |
| P2 | No C15-specific tests; launcher inventory still expected 1047 experiments when 1059 existed | Added reference/operator/RNG/config/evaluator tests and restored the exact inventory, including four new pilot configurations. |
| P2 | C15 plan claimed q/k/v/o CLIP LoRA, while runtime selects q/v | Corrected the documentation, not the matched training architecture. |

Affected historical run root:
`/home/linbinhao/ECG_adv_data/runs/pulse_c15_augmix_20260918`.
Ningbo clean/single/three-retry result files report 100 steps with the same old
implementation hashes. Width 0 did not execute image augmentation, but must
not be silently mixed with a newly identified matched experiment.
The three actual K500 manifests are identical as parsed records, each
contains 500 unique hashes, and their intersection with the frozen Ningbo
512-record development cohort is empty. Patient independence is not established.

The earlier `pulse_joint_singleop_20260918` experiment uses the separate
three-effect `image_stress.py` suite, not C15; these findings must not be
retroactively attributed to that operator implementation.

## Corrected protocol

- Training: native500 -> unchanged ECG renderer -> original nine PIL operator
  formulas -> image-branch Dirichlet(1)/Beta(1,1) mixing -> unchanged PULSE
  processor. Protocol ID: `augmix_pil_reference_v1`.
- The existing hybrid topology is deliberately retained: per augmented view,
  sample 1-3 waveform operators from the existing five, render once, and share
  that waveform across image branches. The image mixing residual is this
  waveform render, not the original clean ECG. Clean CE and two augmented
  teacher-forced answer-token JSD views remain unchanged. This is still a
  hybrid ECG adaptation, not full original AugMix experimental parity.
- CLIP blocks 19-22 q/v LoRA rank8/alpha16 plus full projector; frozen LLM.
  Same K500, seed 20260918, 100 optimizer steps, effective batch16, learning
  rates and JSD12. No new model-capacity or tuning variable was introduced.
- Evaluation: `joint_c15_reference_v1`, 1 clean + 5 single waveform + 15 image
  + 75 Cartesian joint conditions. The pilot remains severity3 and 128 fixed
  non-K500 Ningbo records, not a full five-severity/four-center benchmark.
- C15 uses the ImageNet-C parameter tables in the rectangular-image extension
  `imagecorruptions==1.1.2`, not CIFAR-C's different tables. Its Python motion
  blur and generalized elastic transform are upstream extension choices,
  not byte parity with the original fixed-224 ImageNet-C files.
- Preserve the full rectangular ECG canvas. AugMix transforms use each axis's
  actual size and the original black fill. Uint8 rounding is explicit. These
  severities are not proven to preserve diagnostic labels on ECG paper.
  Native-resolution blur/geometry also cannot be equated with natural-image
  benchmark severity at 224x224 or 32x32.
- Version/hash the C backend, frost assets, Pillow and numerical dependencies
  in the evaluation artifact. Training records the PIL/upstream identity.
- Old approximate configs remain historical records and fail before model
  loading if executed. There is no automatic old-checkpoint migration.

## Entry points and verification

New train configs are
`configs/experiments/pulse_augmix_reference_{clean,single,three}_ningbo.yaml`;
evaluation is `configs/experiments/pulse_c15_reference_ningbo.yaml`.
All use the existing managed launcher and fresh output root
`/home/linbinhao/ECG_adv_data/runs/pulse_augmix_reference_20260919`.

```bash
/home/linbinhao/miniforge3/envs/ECGTwin/bin/python boot_scripts/run_experiment.py \
  --config configs/experiments/pulse_augmix_reference_three_ningbo.yaml --dry-run
```

CPU dependencies are fixed in `environments/pulse-image-requirements.txt` and
were added with `--no-deps` to ECGTwin and `pulse_llava_infer`. Existing Torch,
CUDA, NumPy, SciPy and Pillow were not upgraded. These CPU transforms trade
speed for reference fidelity; particularly native-canvas glass blur needs a
runtime-cost admission before authorizing a long evaluation.

The original model loader, source weights, renderer, processor, class mapping,
K500 split, generation and source/hash guards are unchanged. Fresh real-image
visual review and GPU gradient/packing/resume admission are still required
before a new performance claim or long training run. CPU tests do not prove
PULSE performance or clinical label preservation. Retain development-only
status and report clean/wave/image/joint metrics separately.

Focused verification: 158 tests passed, covering all 15 C operators at all five
severities, nine training PIL formulas, real codecs, failure/RNG restoration,
protocol isolation, matched configs, result acceptance and launcher inventory.
Managed train/evaluation dry-runs resolve without loading data/model/GPU.
The final full CPU run was **773 passed, 3 failed, 14 skipped**. The 15 C
operators also replayed exactly in the actual `pulse_llava_infer` environment
at severity3, with `torch.cuda.is_initialized()` still false. Its Torch
2.1.1+cu118, Transformers 4.37.2, NumPy 1.26.4 and Pillow 10.1.0 remain intact.
`git diff --check` passed; no files were staged or committed.

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  TMPDIR=/home/linbinhao/.cache \
  PYTHONPYCACHEPREFIX=/home/linbinhao/.cache/pulse-audit-pycache \
  LD_LIBRARY_PATH=/home/linbinhao/miniforge3/envs/ECGTwin/lib \
  /home/linbinhao/miniforge3/envs/ECGTwin/bin/python -m pytest -q --tb=short
```

The first full CPU run exposed three unrelated host-dependent failures in old
tests, in addition to the new inventory entries now fixed:

- `test_visual_configs_and_finite_grid`: September 16 archive symlink makes
  `dual_jsd_20260911` resolve outside the permitted home-root run tree.
- `test_fixed_workflow_has_three_admissions_eight_trains_and_one_evaluation`:
  the old pixel-mixing resume path similarly resolves into `/data`.
- `test_cache_staging_persists_only_the_canonical_100hz_waveform`: a 96 KB
  synthetic cache still triggers the real disk-reserve guard (about 86 GB free
  versus a required 98 GB reserve on the test filesystem).

No archive symlink, production path validator or disk-reserve threshold was
changed to bypass these failures. They are not new operator test failures and
do not establish that the entire project suite is green.

## Reference provenance

- Google AugMix `augmentations.py`, commit
  `9b9824c7c19bf7e72df2d085d97b99b3bfb00ba4`, retrieved from the author's
  `google-research/augmix` repository; Apache-2.0 notice retained in code.
- `imagecorruptions==1.1.2`, authors' `bethgelab/imagecorruptions` package;
  wheel SHA256 `d0e4bf529ba6fbb27d8aba1bde7c5db43bf2b969b2c767c11419e939e3e365b8`.
  Its source and assets are runtime-fingerprinted. Local adapters only handle
  skimage's `multichannel` -> `channel_axis` API and explicit impulse RNG.
- Relevant prior request: local project conversation
  `01a05a89-fdfa-7941-85bb-6e3631c168cc`, September 18 at 16:44 China time.
  Recalled intent was checked against actual source and run protocols.

## Severity-3 HPO and 24-hour budget assessment

Requested after the audit: severity is fixed at 3; assess Optuna, trainable
capacity, and full versus sampled four-center evaluation. The following is a
measured feasibility assessment and proposed protocol, not a launched study.

### Measured capacity

The existing Ningbo three-chain `model_audit.json` reports 21,110,784 trainable
parameters. Its eight CLIP q/v projections have hidden size 1024 and rank 8:
`8 * (1024*8 + 8*1024) = 131,072` visual LoRA parameters. The two-layer
1024->4096->4096 projector has 20,979,712 parameters, including biases.
Thus about 99.38% of trainables are in the projector, not the vision tower.

The historical visual admission recorded nonzero updates for both groups.
This establishes that the old optimization path learned, not that its visual
capacity or corrected AugMix performance is sufficient. On the old three-chain
run, mean clean CE over steps 1-10 was 0.9052 and over steps 91-100 was 0.7188;
the last three ten-step means were 0.7619, 0.6957, 0.7188. These noisy training
values do not establish heldout convergence, particularly after changing the
augmentation implementation.

A bounded capacity comparison should keep the last four effective CLIP blocks
and frozen LLM, comparing q/v rank8 against q/k/v/out rank16. The latter has
524,288 visual LoRA parameters. Both widths must use the same selected capacity
and matched optimizer budget. Neither this comparison nor a claim that either
capacity is sufficient has been executed. Current training contracts still
admit q/v rank8 only; do not silently override them.

### Measured throughput

The four completed `pulse_joint_singleop_20260918_full512_retry` evaluations
each contained 512 records, 24 conditions and four model arms. Runtime seconds
for Ningbo/Chapman/CPSC/Georgia were 21510.75/21315.01/20996.31/21427.47.
The average is approximately 0.4336 seconds per arm prediction, or 2.306
predictions/second/GPU, inclusive of that old pipeline's overhead.

At the same throughput, with four perfectly balanced GPUs and three arms:

| Records per center | Severity-3 conditions | Extrapolated evaluation wall time |
| --- | --- | --- |
| 128 | 96 | 4.44 hours |
| 256 | 96 | 8.88 hours |
| 512 | 96 | 17.76 hours |
| All 39,879 across centers | 96 | 345.83 hours, about 14.4 days |

These are extrapolations, **not** updated end-to-end ETAs. They exclude HPO,
training and the incremental cost of the new CPU reference corruptions. Full
evaluation in 24 hours would require 33.23 predictions/second/GPU, about 14.4x
the measured old throughput. The eight historical 100-step augmented training
cells would themselves take roughly 3.8 wall hours on four GPUs, before HPO
and before accounting for the corrected PIL operators.

A fresh CPU-only probe used a synthetic 1700x2200 RGB paper canvas, severity3,
two Torch threads, one process, and no GPU initialization. Times in seconds:

| Operator | Seconds |
| --- | --- |
| gaussian_noise | 1.220 |
| shot_noise | 2.297 |
| impulse_noise | 1.073 |
| defocus_blur | 1.084 |
| glass_blur | 120.201 |
| motion_blur | 3.491 |
| zoom_blur | 8.941 |
| snow | 2.472 |
| frost | 0.961 |
| fog | 2.000 |
| brightness | 5.294 |
| contrast | 1.227 |
| elastic_transform | 3.615 |
| pixelate | 0.448 |
| jpeg_compression | 0.372 |

The 15 operations total about 154.7 seconds. Each appears six times per
record in the 96-condition suite (image-only plus five joint waves). A serial
worker extrapolation gives about 33 hours of CPU transformation alone for
128 records on each of four workers. This is a synthetic, single-pass timing,
not a clinical-image or GPU throughput admission, but it rules out promising
a 24-hour run with the current unoptimized reference path. Optimize only after
byte-level equivalence tests against the pinned reference; do not substitute
the rejected approximate corruptions to improve this number.

### Proposed search boundary

- Keep training/evaluation image severity3, waveform strength0.5, native500,
  original PIL families and frozen LLM fixed for this study.
- Split each existing K500 into fixed inner training/validation identities
  (initial proposal: 400/100). Use no non-K500 record for new HPO selection.
  Refit each final model on all 500 original K500 records from source weights.
- Use a small matched Optuna study: initially four shared candidate recipes,
  with intermediate validation and pruning. Search separate CLIP/projector
  learning rates and JSD weight, not chain width. Capacity comparison is a
  matched development diagnostic, not an independently favorable setting per arm.
- Do not rank trials by total `CE + lambda*JSD`: lowering lambda would change
  the objective scale. Track clean/augmented validation CE, generated-label
  macro-F1, JSD, gradients and finite updates separately. A training-loss decrease
  is not a convergence certificate.
- Training rungs and maximum budget must be explicit; 25-step screening plus
  a fixed 100-step refit is not convergence-controlled training. A proposed
  50/100/200-step schedule requires actual intermediate validation/pruning
  integration and bounded GPU-hour accounting, not just a YAML pruner name.
- A short fixed subset on every center is the first practical stage. After
  reference-preserving performance work, target 128 fresh non-K500 records per
  center with all 96 conditions and three arms; expand to 512 or full only
  after a new measured throughput/cost gate. This sample-size reduction is a
  recommendation pending user scope confirmation, not an executed replacement
  of the requested full benchmark.
- Preserve per-center/family metrics and paired record-bootstrap uncertainty.
  Do not select easier records/operators from observed scores. Historical
  heldout-development exposure remains disclosed; new inner-K500 tuning does
  not make previously observed data an independent external test.

The existing `core/pulse_hpo.py` / `util/pulse_hybrid_workflow.py` are still the
old mild-image study: they search `image_strength`, expect a trained clean arm,
use the non-K500 development subset and fixed 25/100-step budgets. They are
not an admitted launcher for this proposed reference/three-arm/inner-K500 HPO.
No Optuna GPU study, capacity sweep, retraining or full test was launched during
this assessment. The native glass-blur acceleration has not yet been implemented
or admitted. All current benchmark probes completed and no GPU was allocated.
