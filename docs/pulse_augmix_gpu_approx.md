# PULSE GPU image operator protocol

This is an opt-in development throughput path for native-500 PULSE image
training and image-only evaluation. Waveform corruption is disabled in the
new training protocol. The CPU reference path remains separate.

## Operators

`augmix_torch_gpu_v2` implements the nine training operators:

`autocontrast`, `equalize`, `posterize`, `rotate`, `solarize`, `shear_x`,
`shear_y`, `translate_x`, and `translate_y`.

`image_c5_torch_gpu_v1` implements the five evaluation operators:

`gaussian_noise`, `motion_blur`, `brightness`, `elastic_transform`, and
`jpeg_compression`.

The evaluation suite is `image_c5_gpu_v1`: one clean view plus those five
single image corruptions at severity 5. It has no waveform or joint views.
In this approximation, C5 severity bounds a per-image random level sampled
uniformly from `[0.1, 2 * severity)`. It is not an ImageNet-C severity table,
and severity 5 does not mean every image receives the maximum perturbation.

## Implementation boundary

The operators use floating BCHW Torch tensors and explicit device-local random generators,
rectangular-canvas affine correction, per-channel histogram equalization,
device-side motion kernels, elastic displacement grids, and block DCT JPEG
style quantization. They do not call Pillow, NumPy, or CPU image codecs inside
the operator.

The output is intended to be visually close enough for the throughput lane.
It is not pixel-equivalent to Pillow, `imagecorruptions`, or the retained
`augmix_pil_reference_v1` protocol. GPU results must retain their own
implementation identity and must not be merged into reference metrics.

## Current configs

- `configs/experiments/pulse_augmix_gpu_single_ningbo.yaml`
- `configs/experiments/pulse_augmix_gpu_three_ningbo.yaml`
- `configs/experiments/pulse_image_c5_gpu_ningbo.yaml`

These configs only declare managed runs. No PULSE model training or inference
has been launched for this protocol. Operator tests below are not model admission.

## Operator validation, 2026-09-24

Fixed the C5 implementation-name import and reused the operator owner's pool.
Motion blur now spreads all 21 taps over the sampled radius; the previous
two-pixel tap spacing silently produced identity images for radii below two.
AugMix rejects boolean/string severities, and C5 config/result validation
requires integer severity 5 and the original/single/three arms.
The three new configurations are included in the exact launcher test inventories.

- Project environment: 242 operator/formula/integration checks passed on an
  exclusively selected GPU. CUDA and native-size tests are explicitly opt-in.
- Actual `pulse_llava_infer` environment: 243 checks passed, including an
  additional synthetic waveform -> retained renderer -> C5 -> PULSE processor
  -> small downstream gradient check. No PULSE weights or patient data loaded.
  A second run with a fresh Python cache prefix also passed all 243 checks.
- Coverage: all 9+5 operators; train severity 0.1/3/10; all five C5 severities;
  FP16/BF16/FP32; noncontiguous batches; invalid inputs; input/global RNG
  immutability; explicit-generator replay and trusted/public path agreement;
  smallest supported images; autocast; 1700x2200 batch-2 FP16/FP32 images.
- Focused CPU regression: 130 passed; 219 explicitly gated GPU cases skipped.
  C5 condition identities, metric families, training hashes, required source
  snapshots, result round-trip and tamper rejection are covered.
- Static undefined-name/export/import checks passed. These results establish
  implementation contracts, not CPU pixel parity, clinical label preservation,
  a measured throughput advantage or PULSE end-to-end training/generation parity.

Full CPU regression with a fresh Python cache prefix: **820 passed, 4 failed,
233 skipped**. The remaining failures are the hard-coded registry date in
`test_data_contracts.py`, archived `/data` paths in `test_dual_jsd.py` and
`test_pulse_pixel_mixing.py`, and the live disk-reserve gate in
`test_pn2021_corruptions.py`. No archive paths or disk guards were changed.
A fifth, stale-bytecode source-location failure disappeared with the fresh
cache prefix, without editing its test. The project as a whole is not green.

External JUnit receipts (no model artifacts):

```text
/home/linbinhao/ECG_adv_data/tmp/gpu_ops_pulse_cuda_20260924_a3/junit.xml
/home/linbinhao/ECG_adv_data/tmp/gpu_ops_full_cpu_20260924_a3/junit.xml
```

Runtime source SHA256 values for these checks:

```text
core/image_augmix_gpu.py 8058e83647d9d40d0d31d4e62a677c6f2c99ff88e40a7b873900c4fe7062e56b
util/evaluation/pulse_hybrid_development.py a4befa2ec14811c69e8723836039faaa280d36f6b8f83e000cff0f0700543351
util/pulse_hybrid_contract.py 4299f1c15f49409920d02b2623db072b7f810036abdf21558a0494d72a7efbe3
```

After shared-server resource/ownership preflight, select one free GPU UUID
and run the retained test file with `PULSE_GPU_OPS_TEST=1` and
`PULSE_GPU_OPS_NATIVE=1`. Without those flags the normal CPU suite does not
allocate a GPU. Keep pytest temporary files under the user's home tree.

## Rewrite difficulty

The nine training operators are low to medium difficulty. Motion blur, elastic
transform, and block DCT JPEG are medium difficulty because their geometry or
transform-domain details affect both speed and appearance. Protocol integration
and source-hash/result contracts are medium difficulty. Exact reference parity
would be a separate high-difficulty project and is outside this approximation
lane.
