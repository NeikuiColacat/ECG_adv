# Paper ECG printing and quilting ports

Implementation: `paper_ecg_torch_v3`; evaluation: `paper_ecg_gpu_v3`.
This is a new development protocol. Existing v1/v2 lists and identities retain
their meanings. Operator tests do not establish model robustness gains.

## Scope

| Added operator | Implemented RGB path | Boundary |
| --- | --- | --- |
| dirty_rollers | Six gradient-bar patterns, random concatenation, meta-mask, both background modes, orientation | Private versioned sampler |
| dirty_drum | Gaussian clusters in strips, last-write rasterization, both directions, blur, darken blend | Small ragged plans on CPU; fields and pixels on GPU |
| folding | Two-sided warps, darkening, noise, repeated folds, rotation | White ECG backdrop and explicit interpolation/severity policy |
| faxify | Downsampling, five thresholds, rotated block halftone, inversion, upsampling | Mean/Otsu/Li/triangle/Sauvola; no dynamic expressions |
| bad_photocopy | Five noise families, ten locations, optional blur/wave/edge effects, multiplication | Versioned sampling and guarded degenerate masks |
| quilting_wrinkle | Original texture patches, resize, mean shift, thresholded overlay | Author default is one block; multi-block quilting is separately exposed |

Augraphy: sparkfish/augraphy commit
`ed4dcbdaf7b1da6ef59ac60816f96ea253da4c9b` (MIT). Noise families are Gaussian
clusters, multiscale Gaussian, Perlin, Worley and rectangular clusters.
ECG-Image-Kit: alphanumericslab/ecg-image-kit commit
`27b90f56896c9fc78b05a83ca14844ea2637aa0b` (BSD-3-Clause). Notices are retained
under `docs/licenses/`. RGB images are supported; annotation/keypoint/alpha
wrappers and every arbitrary upstream API argument are outside this port.
Kornia is not a new dependency.

Owners: `core/paper_ecg_print.py` and `core/paper_ecg_quilting.py`.
Small shape/algorithm plans use a private generator derived from the caller's
Torch seed/counter. CUDA Philox offset access reads host metadata, not a CUDA
scalar. The sampler consumes caller state explicitly. Ragged allocation sizes
are known before launching kernels; repeat-interleave receives output sizes.
No image returns to the CPU in the warmed operator path.

## Quilting and textures

`quilt_texture(texture, block_size, num_blocks)` implements exhaustive overlap
search and minimum-cost seams. Candidate extents retain the author's excluded
last row/column. Integer byte-domain SSE avoids floating tie ambiguity; equal
seam costs use lexicographically first paths. Search is chunked; seams retain
sequential row dependencies. Multi-block synthesis is for bank preparation.

The author `get_creased` calls `quilt(..., 250, (1,1), ...)`. With no overlap,
all first-patch errors are zero and argmin chooses the top-left patch. Loading
that patch directly preserves this default without exhaustive zero-cost search.
All nineteen author texture files are included in the external bank; selection
does not use target scores. Encoded JPEG and decoded RGB bank hashes are checked.
The author's RGB-through-PIL then BGR-to-gray convention is kept deliberately.

Textures remain outside Git under
`/home/linbinhao/ECG_adv_data/assets/paper_ecg_wrinkles_27b90f56896c`.
The source repository has a root BSD license and no nearer license at this
asset path; repository provenance is not independent photography ownership.
Fresh-checkout setup: explicitly call `fetch_original_textures()` from
`core.paper_ecg_quilting` before launching. It downloads only pinned files,
checks hashes and refuses to overwrite mismatched files.
`load_original_texture_bank(device)` verifies/uploads once per device; callers
may also provide their own device-resident bank directly.

## Protocol and validation

The candidate adds dirty_rollers, dirty_drum and quilting_wrinkle to the ten
v2 training families: thirteen total. Folding, bad_photocopy and faxify join
the six held-out families: nine total. Each stress level has clean plus
twenty-two individual artifacts. Six `pulse_paper_print_*` declarations retain
matched K500, optimizer budget, seed and JSD settings; they are unexecuted
candidates, not accuracy evidence.

Severity scales appearances and fold geometry under a project-owned policy.
Independent folding is excluded from training branches. These are not clinical
severity units. Fixed-parameter tolerances are recorded per kernel; full
author pixel or seed equivalence is not claimed. Affine interpolation, byte
rounding, cluster sampling and degenerate-noise fixes can differ. Upstream
tuple-shaped OpenCV dilation is preserved as column dilation, not silently
reinterpreted as square kernels.

Source checksums and validation records are outside Git under
`/home/linbinhao/ECG_adv_data/operations/pulse_paper_complete_20260930_154836`.
CPU oracles, native CUDA admission and real-model admission are separate.
No full training run is part of this migration.

### Migration verification

- Final project regression: 1579 passed, 240 skipped, zero failures; three
  upstream deprecation warnings. Skipped tests are not counted as passes.
- Actual PULSE environment (Torch 2.1.1+cu118), paper CPU/GPU suites:
  314 passed, zero skipped or failed. GPU 0 was explicitly admitted and free.
  Strict deterministic algorithms were enabled and TF32 disabled.
- All eighteen v1/v2/v3 declarations pass managed dry-run, including six new
  v3 declarations; all six v3 declarations were rechecked after the GPU fixes.
  Paired training differs only in width and temporary path.
- All nineteen source texture hashes and the decoded bank hash match.
- The first GPU check exposed unsupported deterministic floating cumsum in
  Otsu on Torch 2.1. Prefix sums now use exact integers before division.
- Li uses a finite histogram transition map instead of 100 scalar iterations.
  Eight pointer-doubling gathers resolve its 256 partitions. Independent
  skimage checks cover 84 random sparse, dense and constant histograms.
- Tiled integer histograms avoid atomic contention on white pixels. The
  second optimization preserves all 100 recorded CPU and GPU image hashes.
- The protected live inference checkout remains on its original source.
  New full-model training/evaluation admission has not been executed.

Current evidence, logs, source hashes and the local interactive HTML report:

    /home/linbinhao/ECG_adv_data/operations/pulse_paper_gpu0_admission_20260930_170834

The report is paired_native_v3/index.html. Generated ECG images remain outside
Git. They contain a real training ECG with lead names and calibration marks,
but no patient metadata. They have not been published to an external service.

### CPU/GPU numerical comparison

One fixed, previously recorded Ningbo K500 training exposure was chosen before
comparison, without model-score selection. The native 1700 by 2200 renderer
and K500/exposure files were checked against their recorded hashes. Twenty-five
operators (the v1/v2/v3 union), two severities (S2/S5), and FP32/FP16 produce
100 paired cases. The CPU reference is the same Torch port with identical
GPU-generated parameter plans and random fields replayed on CPU; original
OpenCV/skimage/quilting formula checks are separate. This does not establish
pixel equality for every option of the complete upstream libraries.

Forty-one pairs are bitwise identical. Ninety-three have maximum error within
approximately one 8-bit gray level; seven exceed it. FP32's largest error is
1.878 gray levels (Folding), with mean error 0.0276 and at most 0.018 percent
of pixels above one level. A diagnostic using FP64 sampling reduces Folding's
CPU/GPU maximum to 0.0000304 gray levels, identifying interpolation and
subsequent byte rounding as the main source. The production path retains FP32.

The largest FP16 difference is 6.101 gray levels in JPEG S5, affecting
0.00131 percent of pixels above one level. Exactly one of 11,246,400 DCT
coefficients crosses a quantization boundary, within 0.000000477 of a half
integer. Wrinkle/Faxify FP16 also show sparse rounding differences up to
1.121 gray levels. Therefore the audit does not claim cross-device bitwise
equivalence or identical downstream generated answers. The actual renderer
supplies FP32 images; model inputs are converted after image preprocessing.

The HTML shows original/GPU, CPU/GPU, magnified differences, waveform crops,
and the actual five PULSE input tiles. General 3 by 3 quilting is checked
separately from the author's default one-block path. Neither visual review nor
these arithmetic tests establish clinical label preservation or an AugMix gain.

### Bounded performance measurements

RTX 4090, native B1 RGB FP32, S2, same real ECG and seed, warm operator only,
three CUDA-event repetitions; comparison copies and file IO are excluded:

| Operator | Milliseconds |
| --- | ---: |
| quilting_wrinkle | 2.188 |
| dirty_rollers | 2.456 |
| folding | 3.414 |
| bad_photocopy | 6.699 |
| faxify | 7.223 |
| dirty_drum | 8.070 |

On this matched input, histogram tiling reduces Faxify from 60.133 to 7.223 ms
(8.33 times faster) without changing its output. This is an operator-level
measurement, not a whole-training or whole-inference speedup.

### Proposed next pool (not activated)

The registered v3 recipe remains thirteen training and nine held-out operators.
A future separately versioned recipe can consider these twelve training
families: yellowing, exposure, shadow, crease, quilting_wrinkle, ink_fade,
ink_bleed, low_ink_lines, dirty_rollers, defocus, sensor_noise, low_resolution.
Start with mild train-side visual calibration; independent branches should
preserve waveform coordinates.

Ten candidate held-out operators are rotate, perspective, folding, stain,
glare, occlusion, jpeg_compression, dirty_drum, bad_photocopy and faxify. Report
these separately from clean and seen-operator severity sweeps. Operator
holdout does not imply disjoint physical mechanisms. Retain old C5 for
longitudinal comparison, explicitly noting noise/brightness overlap with the
new training pool. Include a matched-budget clean-only LoRA control and freeze
the recipe before looking at held-out metrics. No candidate was activated here.

The opt-in native test covers all twenty-two operators, every fax threshold
and photocopy noise family, original textures and multi-block quilting. Run
it on an explicitly admitted free GPU using PULSE_GPU_OPS_NATIVE=1 and the
actual PULSE Python, with CUBLAS_WORKSPACE_CONFIG=:4096:8. Unit-test admission
does not authorize a full training/evaluation run.
