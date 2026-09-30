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

- Final project regression: 1575 passed, 239 skipped, zero failures.
- Actual PULSE environment (Torch 2.1.1+cu118), paper CPU suites: 306 passed,
  three native-GPU checks skipped.
- All eighteen v1/v2/v3 declarations pass managed dry-run, including six new
  v3 declarations. Paired training differs only in width and temporary path.
- All nineteen source texture hashes and the decoded bank hash match.
- Synthetic full-canvas S2/S5 previews reviewed; no patient images published.
- New native CUDA admission was not executed: available devices remained
  occupied, including the graphics process on GPU 7. No other job was changed.
  GPU throughput and real-model training/evaluation admission remain pending.

The opt-in native test covers all twenty-two operators, every fax threshold
and photocopy noise family, original textures and multi-block quilting. Run
it on an explicitly admitted free GPU using PULSE_GPU_OPS_NATIVE=1 and the
actual PULSE Python, with CUBLAS_WORKSPACE_CONFIG=:4096:8. Unit-test admission
does not authorize a full training/evaluation run.
