# PULSE C15-style AugMix pilot

Historical September 18 protocol. The implementation was audited on September
19 and contained incorrect operators and an evaluator condition-ID bug. It is
not evidence against original AugMix. Live execution of these old approximate
configs is disabled; see [the audit and corrected protocol](pulse_augmix_audit_20260919.md).
Old checkpoints and source snapshots are preserved, not migrated.

This is a development pilot, not an ImageNet-C reproduction and not an
independent external-test claim.

## Protocol

- Native 500 Hz ECG -> the locked PULSE renderer/processor; no raw100 path.
- Five existing waveform operators remain unchanged:
  `powerline_noise`, `emg_noise`, `baseline_wander`, `baseline_shift`,
  `random_leads_masking`.
- Training image pool uses the original AugMix operator families:
  `autocontrast`, `equalize`, `posterize`, `rotate`, `solarize`, `shear_x`,
  `shear_y`, `translate_x`, `translate_y`.
- Training topology is the existing PULSE visual protocol: clean + two
  stochastic augmented views, JSD over aligned teacher-forced answer-token
  distributions, three chains for the three-chain arm, depth 1--3, Dirichlet
  alpha 1 and Beta alpha 1. JSD weight is 12, CLIP blocks 19--22 use q/v
  LoRA rank 8/alpha 16, the projector is trained, and the language model is
  frozen.
- Operator magnitudes were unvalidated ECG-specific remappings, not the natural-image numeric
  severities. Training image strength is 0.35 and waveform strength is 0.5.
- PN2021-C pilot uses the 15 ImageNet-C/CIFAR-C operator families at fixed
  middle severity 3: 1 clean + 5 waveform + 15 image + 75 waveform/image
  Cartesian joint conditions = 96 conditions per record.
- The pilot uses 128 non-K500 records from Ningbo. A four-center and five-level
  severity expansion is only warranted after this engineering/feasibility run.

## CLIP/projector rationale

AugMix does not prescribe a special CLIP learning rate. We keep the matched
PULSE visual adaptation budget and train only the last four CLIP blocks through
LoRA plus the multimodal projector. This isolates the effect of changing the
operator family and avoids confounding it with language-model LoRA or a new
optimizer budget. The original AugMix defaults (three chains, stochastic depth
1--3, Dirichlet/Beta alpha 1 and three-view JSD) are retained at the topology
level; the ECG-safe magnitude remapping is recorded explicitly.

## Evidence boundary

ImageNet-C/CIFAR-C GPU operators such as JPEG, glass blur and elastic transform
are implemented as documented GPU approximations for this pilot. They are not
byte-identical to the NumPy natural-image benchmark. Any paper result would
need a parameter calibration against real print/scan/photo ECGs and all five
severity levels, with per-operator results and paired record bootstrap.

Outputs are external to Git at
`/home/linbinhao/ECG_adv_data/runs/pulse_c15_augmix_20260918`; augmented images
are generated online and not persisted.
