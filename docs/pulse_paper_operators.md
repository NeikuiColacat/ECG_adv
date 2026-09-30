# PULSE paper-image operators and performance audit

Status: bounded engineering admission complete. Clinical calibration and
robustness benefits have not been established.

Verification so far: 1266 CPU tests passed, 237 skipped; all six managed paper
declarations dry-run successfully. The actual PULSE environment also passes 141
CPU checks with one native-GPU test skipped. Native-size tests in that environment
now pass all 361 paper/legacy checks with GPU admission enabled on one RTX 4090.
A 24-case legacy comparison preserves
exact pixels, augmentation traces and global RNG; 15 JPEG cases remain pixel exact
after constant caching. At 2200 x 1700, batch one, FP16, the 16 paper operators
take 0.28--5.12 ms each after warmup (three CUDA-event repeats, validation excluded).
These are operator timings, not whole-pipeline speedups. Both single- and
three-chain LoRA smokes completed four optimizer steps with finite nonzero
language/vision LoRA and projector gradients, frozen-parameter preservation,
and exact saved trainables. Four K500-excluded records then completed all 17
conditions across three arms: all 204 generated answers and raw token sequences
match the reference path. Result contracts, file indexes and source hashes pass.
Native GPU previews were inspected using a synthetic ECG-like fixture, not
patient data. A new-process checkpoint resume was not run for this new recipe;
saved optimizer/RNG states and disk trainables were verified. Clinical fidelity
and robustness benefits remain unvalidated.

## Delivery plan

- [x] Separate measured model inference costs from rendering and serialization.
- [x] Verify paper-ECG artifacts against PULSE and ECG-Image-Kit primary sources.
- [x] Define a versioned training pool and independently reported stress suites.
- [x] Implement device-resident operators with PyTorch and bounded allocations.
- [x] Simplify the affected owners and document the public execution path.
- [x] Verify deterministic replay, source identity, native-size execution,
      numerical parity where applicable, and measured throughput.

## Protocol boundary

The proposed paper-image protocol acts on native-500-Hz ECG renders before
model-specific image preprocessing. It does not redefine the historical
waveform PN2021-C protocol, CPU ImageNet-C reference, or the retained GPU C5
visual-approximation protocol. New results require a new implementation and
configuration identity. Running experiments retain their frozen source.

Training augmentation, seen-family stress, and held-out-family stress must be
reported separately. Severity parameters are engineering specifications, not
clinically calibrated units. Do not select them using target evaluation labels.
Severe occlusion or geometric distortion may destroy diagnostic information;
such conditions must not be described as guaranteed label preserving.

## Primary sources under review

- PULSE: [published paper](https://www.nature.com/articles/s41746-026-02551-3),
  [preprint](https://arxiv.org/abs/2410.19008), and
  [author repository](https://github.com/AIMedLab/PULSE).
- [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit): the image
  synthesis framework used by PULSE; implementation details require independent
  verification rather than assumptions from artifact names.
- [ECG-Image-Database](https://arxiv.org/abs/2409.16612): synthetic and real
  imaging/scanning artifact evidence for evaluating paper-ECG processing.

### What the sources actually establish

PULSE Appendix C describes folds/wrinkles, rotations, resolution and background
color variation, noise, layout variation, and occasional grid removal. Its
header augmentation can include metadata; we do not add diagnostic text,
patient attributes, or label-derived annotations to this pool. See the
[full appendix](https://arxiv.org/html/2410.19008v1#A3).

ECG-Image-Kit v1.0.0 is pinned at
`27b90f56896c9fc78b05a83ca14844ea2637aa0b` (BSD-3-Clause). In the original
`CreasesWrinkles/creases.py`, wrinkles use stored paper textures and quilting;
creases use blurred line masks. `ImageAugmentation/augment.py` composes rotation,
additive Gaussian noise, crop, and color-temperature change. Our implementations
are independently written procedural approximations, not ports of its quilting
algorithm or reproduced author pixels. Upstream code was inspected, not imported
as a runtime dependency.

ECG-Image-Database Section 5 distinguishes synthetic artifacts from real printing,
scanning, photography, soaking/staining and mold exposure. This supports including
paper damage in a stress suite, but does not validate any particular synthetic
stain, parameter range, or clinical label preservation claim. See its
[methods](https://arxiv.org/html/2409.16612v1). The
[PhysioNet 2024 Challenge](https://moody-challenge.physionet.org/2024/) is a useful
external reference for image digitization and classification evaluation.

## Proposed pools: paper_ecg_torch_v1

These are engineering choices to freeze before new training. They are not
selected from PN2021 target-evaluation performance. Training uses severity 2
initially, with the existing clean branch, JSD objective and K500 restriction.
This choice is provisional until native-size visual inspection and runtime
checks complete; it is not a published PULSE setting.

| Operator | Mechanism | Train | Stress interpretation |
| --- | --- | --- | --- |
| yellowing | Multiplicative warm paper tint | Yes | Seen family; artificial aging appearance |
| exposure | Gamma and illumination attenuation | Yes | Seen family; scanner/camera exposure |
| shadow | Smooth spatial attenuation | Yes | Seen family; uneven lighting |
| crease | Paired narrow dark/highlight bands | Yes | Seen family; fold appearance without signal warping |
| wrinkle | Nonperiodic height-field light/shadow shading | Yes | Seen family; wrinkle appearance proxy |
| ink_fade | Attenuate dark achromatic ink | Yes | Seen family; trace/text fading proxy |
| defocus | Separable Gaussian blur | Yes | Seen family; loss of sharpness |
| sensor_noise | Additive RGB Gaussian noise | Yes | Seen family; imaging noise approximation |
| low_resolution | Antialiased downsample and resize | Yes | Seen family; reduced sampling resolution |
| grid_fade | Attenuate red grid while preserving gray ink | Yes | Seen family; grid visibility |
| rotate | Whole-page rotation with white fill and fit | No | Held-out family; scan orientation |
| perspective | Whole-page projective sampling | No | Held-out family; camera viewpoint |
| stain | Translucent localized brown field | No | Held-out family; appearance proxy, not liquid physics |
| glare | Local saturation toward white | No | Held-out family; information may be lost |
| occlusion | Small white rectangular mask | No | Held-out family; explicit missing information |
| jpeg_compression | Shared block-DCT quantization approximation | No | Held-out family; not a JPEG encoder |

Geometric operators are excluded from the default training pool: independently
rotated AugMix branches can form ghost traces when averaged. No vertical/horizontal
flips, lead permutation, free elastic waveform deformation, diagnosis text, or
unbounded cutout is admitted. Handwriting and mold are literature-supported
artifact categories, but convincing simulation needs separate assets and review;
this implementation does not claim to model them.

### Stress design and aggregation

`paper_conditions()` defines 16 operators at five severities (80 image views),
plus a separately evaluated clean view. Evaluate one corruption at a time first.
Report clean, seen-family stress, and held-out-family stress separately, with
per-operator and per-severity results. Average views and centers equally, not by
their record counts. A seen-family score is not an unseen-corruption result.
Future composed damage should be a separate frozen suite, not mixed into this
single-operator average. The retained waveform PN2021-C and image C5 numbers
remain separate reference protocols.

The managed evaluator runs one fixed severity per declaration: clean plus 16
paper views (17 conditions, 51 generated answers per ECG across the three arms).
Its family keys are `clean`, `image_seen` and `image_held_out`. To evaluate
all five levels, freeze five declarations with fresh output roots and aggregate
each family across severity levels; do not count repeated clean views as extra
corruptions. The library also exposes all 80 operator/level combinations through
`paper_conditions()` for external integrations.

### Parameter and replay contract

Severity zero returns an exact clone without consuming RNG. For positive levels,
`s = severity / 5`; most amplitudes use an independent per-image factor in
`[0.75s, s]`. At severity five, tint gains are bounded by 0.96/0.82/0.52 (RGB),
shadow attenuation by 0.70, ink/grid fading by 0.85, Gaussian noise standard
deviation by 0.10, and rotation by 10 degrees. Resolution can fall to 25 percent
per dimension. Blur sigma scales with the short edge relative to 1700 pixels.
All exact formulas live in `core/paper_ecg.py`; these numerical ranges are ours,
not author-derived clinical calibration. Increasing severity is not a guarantee
of monotonic model error.

The caller owns a generator on the input device. Same input, generator state,
batch grouping, software, device and deterministic backend settings give replay; CPU/CUDA bitwise
equivalence and invariance to regrouping records are not claimed. Evaluate each
record with its identity-derived seed and reuse its corrupted view across all
model arms. Never reuse trainable visual features across adapters.

The implementation has no per-image Python loop or per-image tensor readback.
The public validator intentionally synchronizes for range/finite checks; a
trusted chain validates its anchor once and uses `validate=False` internally.
Only O(H+W) coordinate vectors are cached, with eight shape/device entries.
Random fields are never cached. Conversion to display images belongs to audit
export, outside the training/inference hot path.
Managed augmentation runs outside model autocast. Standalone callers should
also keep these operators outside autocast to retain the stated FP32 working
arithmetic; ambient autocast can change convolution/matrix precision.

## Use and review

From the repository root, the standalone API needs PyTorch:

```python
from core.paper_ecg import apply_paper_operator
import torch

# images: floating [batch, 3, height, width] RGB in [0, 1].
generator = torch.Generator(device=images.device).manual_seed(7)
augmented = apply_paper_operator(images, "crease", severity=2, rng=generator)
```

The managed Ningbo examples have explicit paired configurations:

| Purpose | Experiment declaration | Scope |
| --- | --- | --- |
| Single-chain admission | `pulse_paper_single_ningbo_smoke.yaml` | Four optimizer steps, K500 only |
| Three-chain admission | `pulse_paper_three_ningbo_smoke.yaml` | Same budget, three image branches |
| Paired stress admission | `pulse_paper_stress_ningbo_smoke.yaml` | Four K500-excluded records, S5, three arms |
| Candidate training | `pulse_paper_single_ningbo.yaml`, `pulse_paper_three_ningbo.yaml` | 200 matched optimizer steps |
| Development stress screen | `pulse_paper_stress_ningbo.yaml` | 16 seeded K500-excluded records; never a paper-final result |

All declarations are under `configs/experiments/`. For example:

```bash
/home/linbinhao/miniforge3/envs/ECGTwin/bin/python \
  boot_scripts/run_experiment.py \
  --config configs/experiments/pulse_paper_single_ningbo_smoke.yaml --dry-run
```

The paired stress example consumes both matching smoke training results. Actual
execution needs one admitted free GPU and the model-specific environment. Do not
substitute archived adapters from another recipe or bypass their source guards.
Four-center work requires matching per-center training/evaluation declarations.

CPU checks: `util/tests/test_paper_ecg.py`, the generic GPU-image result contract
matrix in `test_pulse_joint.py`, and the retained training/launcher suites.
Native operator checks are opt-in via `PULSE_GPU_OPS_NATIVE=1`, with exactly one
admitted device selected in `CUDA_VISIBLE_DEVICES`. CPU fixtures and synthetic
previews establish neither GPU throughput nor clinical fidelity.

## Architecture changes

- Training implementations share one configuration/dispatch/identity owner;
  the trainer no longer repeats its protocol dictionary and source-file switch.
- Image evaluation shares admission and sampling code across C5 and paper views,
  while each suite retains its own condition generator and implementation ID.
- Five fresh source-copy loops use `util.run_record.capture_source_snapshot`;
  locked resume paths still verify existing snapshots without overwriting them.
- Both AugMix views transfer their small diagnostic weights together once.
  Image tensors are never copied for this logging operation.
- JPEG DCT and quantization constants are reused per device, avoiding repeated
  allocations and constant-table transfers after warmup.
- FP16 inference storage is allocated without converting the previous arm: the
  next arm overwrites every parameter before use. Exact FP32 reference buffers
  are restored on both success and failure. This removes redundant work; no
  separate end-to-end speedup is attributed to this small change.

Existing replay recipes require their recorded source checkout. The initial
Torch profiling screen was frozen at `63f6909` before training-owner changes;
its old adapters must be profiled from that checkout. The separate live runtime
continues with its original source identity.

## Performance acceptance

Use seconds per complete ECG (all required arms and conditions), peak allocated
memory, and bounded CPU/CUDA traces. GPU-utilization percentage alone is not an
efficiency score. Keep data, checkpoints, decoding, precision, batch, and random
seeds matched when evaluating a runtime optimization. Profile overhead is
excluded from speed claims. Use PyTorch operators; no custom CUDA implementation
is part of this work.

### Measured Torch breakdown

The frozen `63f6909` screen completed on an RTX 4090 with CUDA activities present
and exact generated tokens/answers. Four Georgia records, six conditions and
three arms produced 72 matched outputs. Unprofiled generation totals were
53.443 s for the author-reference path, 47.866 s for v2, and 43.545 s for v3:
v3 reduced latency by 9.03% versus v2 in this bounded screen. This is neither a
full-cohort result nor the speedup from the later storage allocation cleanup.

For one clean view across the three arms, CUDA kernels were correlated with
their host launch and the innermost module hook; durations are non-overlapping:

| Stage | Kernel time | Share of kernel time |
| --- | ---: | ---: |
| LLM prefill | 1036.504 ms | 79.97% |
| LLM decode | 133.430 ms | 10.29% |
| Vision tower | 97.840 ms | 7.55% |
| Language head | 15.496 ms | 1.20% |
| Projector | 2.402 ms | 0.19% |
| Outside these hooks | 10.412 ms | 0.80% |

There were 25,100 kernels. They occupied 1296.084 ms of a 2048.874 ms kernel
span. Profiling changes timing, so this ratio is not a production utilization
claim. Host `cudaLaunchKernel` calls accumulated 275.691 ms. Long host
`cudaMemcpyAsync` calls accumulated 545.057 ms, but actual host/device transfers
were only 107 bytes D2H and 24 bytes H2D in this generation-only trace. The three
longest waits belonged to tensor-to-Boolean scalar reads. This identifies
synchronization with queued GPU work, not a measured PCIe-bandwidth shortage.

The principal compute cost is the 2880-token visual prompt passing through the
LLM for every view/arm. Active visual adapters require separate vision forwards.
The next potentially material Torch experiments are admitted larger batches
and a fixed-shape compiled/captured generation path to reduce launch gaps.
Both need fresh memory, token-parity and wall-clock admission; neither is enabled
here. Changing image tiling, generation length, adapter merging, quantization,
or attention numerics would change more than scheduling and needs a separate
research identity. No promise of sustained 100% utilization is supported.

Evidence: external `pulse_torch_profile_20260930_r0/evaluation/` contains the
trace, stage summary, paired timings and validated result; the operations folder
`pulse_torch_paper_20260930/` contains `torch_trace_breakdown.json`,
`profile_performance_summary.json`, and `gpu_operator_admission.json`.
The paired four-step training and 204-output evaluation are under
`pulse_paper_ecg_v1_20260930/`; their independent revalidation is recorded in
`model_paper_admission.json` in the same operations folder. None of these bounded
admissions establish the benefit of the unexecuted 200-step candidate recipe.

### Code size and scope

Against pre-task cleanup commit `a6753b7`, retained production Python
(`git ls-files '*.py'`, excluding `util/tests/` and `agent_workspace/`) changes
from 92 files / 37,613 lines to 93 files / 37,897 lines, a net addition of 284.
The new pool, profiler and explicit protocol integration exceed the lines
removed from existing owners; total repository code size has not decreased.
The simplification is in shared ownership and fewer duplicate paths. No legacy
file deletion or wholesale rewrite is part of this change.
