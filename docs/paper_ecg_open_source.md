# Open-source paper ECG corruption audit

This document records the v2 migration. The later six-family printing/quilting
extension and its separate validation status are in `docs/paper_ecg_printing.md`.

Status: source audit complete; version-two ports wired to training/evaluation.
Focused CPU integration passed 378 tests with two explicit GPU skips. Native
GPU operator admission passed all 58 checks in the actual PULSE environment.
Full CPU regression passes 1386 tests with 238 skips and no failures. Model
training/evaluation admission and robustness benefits remain unvalidated.
No new robustness or full-training result is claimed.

## Pinned sources

| Source | License | Pinned commit | Role |
| --- | --- | --- | --- |
| ECG-Image-Kit | BSD-3-Clause | 27b90f56896c9fc78b05a83ca14844ea2637aa0b | ECG layout, creases, wrinkles and paper-image reference |
| Augraphy | MIT | ed4dcbdaf7b1da6ef59ac60816f96ea253da4c9b | Ink, paper and post-printing degradation algorithms |
| Kornia | Apache-2.0 | 4f5e92706ee09966e299057b9f2fdf35697a858e | Existing Torch/GPU building blocks; no environment upgrade here |

Primary references:
- https://github.com/alphanumericslab/ecg-image-kit
- https://github.com/sparkfish/augraphy
- https://github.com/kornia/kornia
- https://arxiv.org/abs/2208.14558
- https://arxiv.org/abs/2307.01946

Inspected files, licenses and SHA-256 values are in the external operations
folder pulse_paper_sources_20260930_113246. Code licenses do not by themselves
resolve the provenance of externally obtained textures, fonts or patient data.

## Useful operators

ECG-Image-Kit supplies ECG rendering, creases, wrinkles, rotation, noise,
color-temperature changes and cropping. Its wrinkle path uses textures and
quilting. A procedural shading field is not that algorithm. Preparing licensed
textures once, then sampling a GPU-resident bank, is a later implementation
option after asset provenance is cleared.

Augraphy separates ink, paper and post-printing effects. Relevant classes include
ColorPaper, InkBleed, InkMottling, LowInkPeriodicLines, LowInkRandomLines,
LightingGradient, ShadowCast, Stains, ReflectedLight, Folding, DirtyDrum,
DirtyRollers, BadPhotoCopy, Faxify, Jpeg and BookBinding. Their default document
settings are not clinically calibrated ECG settings.

Kornia is GPU infrastructure, not an ECG benchmark. Its geometry, blur, noise
and color primitives do not establish paper realism by themselves.

## First migration boundary

First port ColorPaper and InkBleed with fixed-parameter OpenCV comparisons.
Implement a low-ink row variant following the upstream lighten-line idea,
without reproducing LowInkLine's bottom-neighbor indexing behavior. Record
sampling, severity scaling and uint8 rounding differences explicitly.

The selected MIT notice is retained in docs/licenses/augraphy-MIT.txt.
ColorPaper replaces hue/saturation while retaining HSV value. It also changes
colored ink and can remove a red grid with the same value as the background;
it is not a background-only physical aging model. InkBleed follows upstream
Scharr dx-dy, dilation, erosion, blur and blending. Three fixed-parameter
OpenCV fixtures pass with at most one uint8 level of error; this is not a
universal bitwise-equivalence guarantee. Private Torch RNG and severity
sampling differ from upstream. Low-ink lines only lighten selected rows;
the upstream neighboring-row and color-channel behavior is not reproduced.

The new owner is core/paper_ecg_upstream.py. It adds three selected adaptations
and reuses existing procedural effects, rather than importing Augraphy into
the PULSE environment. Torch kernels operate on the existing device tensor;
operator internals contain no image CPU readback, PIL/OpenCV runtime call,
or custom CUDA. Validation at the public boundary can synchronize; training
and evaluation use the already validated path. At BCHW [1,3,1700,2200], FP16
input, severity 2, deterministic mode and five warmed CUDA-event repeats,
ColorPaper takes 1.446 ms, InkBleed 2.990 ms and low-ink lines 0.542 ms on
GPU 2. Across all sixteen effects the observed range is 0.415--4.954 ms.
Each timed call creates a private generator. CUDA-event intervals exclude
public validation and model execution; they are not whole-training speedups. The CUDA test checks
finite bounded output and exact same-seed replay. CPU OpenCV comparisons
remain separate from GPU replay; no cross-device bitwise claim is made.

## Candidate operator allocation

| Artifact | Open-source reference | Initial role | Constraint |
| --- | --- | --- | --- |
| Paper color / yellow tint | Augraphy ColorPaper | Train | Alters colored grid too; not physical aging |
| Ink spread / thickening | Augraphy InkBleed | Train | Keep amplitude small; inspect fine ECG morphology |
| Printer dropout lines | Augraphy LowInkPeriodicLines / LowInkRandomLines | Train variant | Row-only lightening; no false new dark trace |
| Uneven illumination | LightingGradient / ShadowCast | Train approximation | Smooth attenuation and retained clean branch |
| Crease / wrinkle appearance | ECG-Image-Kit CreasesWrinkles, Augraphy Folding | Train shading approximation | Upstream texture quilting still pending |
| Defocus / sensor noise / resolution | Kornia and ECG-Image-Kit primitives | Train | Calibrate after model preprocessing, not only at canvas size |
| Stains / liquid damage | Augraphy Stains | Held-out appearance approximation | Never infer simulated chemistry or diagnosis preservation |
| Glare / reflected light | Augraphy ReflectedLight | Held-out | Saturation may destroy diagnostic information |
| Camera perspective / rotation | Kornia geometry, ECG-Image-Kit | Held-out | Independent branch warps may produce ghost traces |
| Scanner dirt / photocopy / fax | DirtyDrum, DirtyRollers, BadPhotoCopy, Faxify | Later separate stress candidates | Strong defaults can alter waveforms; no new implementation yet |
| Compression / occlusion | Augraphy Jpeg; project mask | Held-out | Current JPEG is block-DCT approximation, not real encoding |

Version two trains with color_paper, ink_bleed, low_ink_lines, exposure, shadow,
crease, wrinkle, defocus, sensor_noise and low_resolution. Its six held-out
families are rotate, perspective, stain, glare, occlusion and jpeg_compression.
This keeps ten training operators and the old depth/Beta/Dirichlet/JSD settings
fixed, but the changed pool is a separate development intervention.

Existing paper_ecg_torch_v1 effects remain independent appearance approximations
with their original identity. No OpenCV/PIL execution or image readback belongs
inside a prevalidated GPU operator.

## Matched AugMix study

Use mild appearance/printing effects inside independent branches. Camera/page
geometry should be shared before branching or applied after mixing under a
separate identity, because independently displaced traces may be superposed.
Freeze the pool and severity from source semantics and K500 visual checks,
not the already observed target evaluation scores. Match single/three-chain
budgets, seeds, K500 identity, loss and evaluation views. A no-AugMix LoRA control
is needed to isolate augmentation from the overall adaptation gain.

Unit tests and short model smokes establish engineering admission only. A claim
that three-chain AugMix has recovered requires completed matched evaluation
and independent training seeds. Current development feedback cannot become
an untouched final test retrospectively.

## Execution and acceptance

Six managed declarations use the prefix pulse_paper_upstream_: single/three
Ningbo four-step smokes, paired four-record stress admission, two 200-step
candidate trainings, and a development stress screen. Each has its own YAML
closure and fresh external output under pulse_paper_upstream_ecg_v2_20260930.
No overwrite or implicit switch of a live run is allowed.
All six declarations pass managed dry-run. The short GPU admission completed
while GPU 2 was free; the following resource check found it occupied again.
No model training, evaluation or 200-step candidate was launched in this audit.

Run the four-step admissions before the 200-step candidates. Require native
1700-by-2200 device replay, finite model gradients and losses, frozen-parameter
preservation, checkpoint/result hash checks and paired reference-token
agreement before interpreting a smoke as executable. Dry-runs alone do not
establish this. The source identity changes intentionally reject adapters
trained under a different source snapshot; do not disable that guard.

For the scientific follow-up, freeze the recipe before new target scoring,
retain the existing clean assessment and unchanged C5 reference, add a
same-budget clean-only LoRA control, and compare single/three over independent
seeds. The current strict three-arm paper evaluator does not yet implement
that four-arm or cross-training-pool C5 experiment; it must be versioned
explicitly rather than relaxing the existing lineage checks. Report clean,
seen-family and held-out-family results separately.

Hypotheses to test, not conclusions: domain mismatch in the generic pool;
ghost traces from independent geometry; changed effective severity after
mixing; and an unsuitable JSD weight. First change only the pool. A later
severity-matched or JSD ablation is a separate experiment, not an explanation
selected after seeing which configuration wins.
