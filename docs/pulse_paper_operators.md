# PULSE paper-image operators and performance audit

Status: design and implementation in progress; no new training result is claimed.

## Delivery plan

- [ ] Separate measured model inference costs from rendering and serialization.
- [x] Verify paper-ECG artifacts against PULSE and ECG-Image-Kit primary sources.
- [ ] Define a versioned training pool and independently reported stress suites.
- [ ] Implement device-resident operators with PyTorch and bounded allocations.
- [ ] Simplify the affected owners and document the public execution path.
- [ ] Verify deterministic replay, source identity, native-size execution,
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
| wrinkle | Smooth texture plus directional shading | Yes | Seen family; wrinkle appearance proxy |
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
batch grouping, software and device give deterministic replay; CPU/CUDA bitwise
equivalence and invariance to regrouping records are not claimed. Evaluate each
record with its identity-derived seed and reuse its corrupted view across all
model arms. Never reuse trainable visual features across adapters.

The implementation has no per-image Python loop or per-image tensor readback.
The public validator intentionally synchronizes for range/finite checks; a
trusted chain validates its anchor once and uses `validate=False` internally.
Only O(H+W) coordinate vectors are cached, with eight shape/device entries.
Random fields are never cached. Conversion to display images belongs to audit
export, outside the training/inference hot path.

## Performance acceptance

Use seconds per complete ECG (all required arms and conditions), peak allocated
memory, and bounded CPU/CUDA traces. GPU-utilization percentage alone is not an
efficiency score. Keep data, checkpoints, decoding, precision, batch, and random
seeds matched when evaluating a runtime optimization. Profile overhead is
excluded from speed claims. Use PyTorch operators; no custom CUDA implementation
is part of this work.
