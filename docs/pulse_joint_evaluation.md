# PULSE waveform + image-v2 development evaluation

Reuse the completed 20260914 original/clean/single/three models. No retraining,
checkpoint selection or augmentation strength search. This is PN2021 development
validation, not an independent hospital test or a new training-seed repeat.

Each center uses the first 128 records from its frozen 512-record non-K500 cohort.
All four arms see identical tensors. The 84 conditions are: clean, 20 unchanged
PN2021-C depth-2/3 waveforms, three image-only profiles, and their 60 joint views.
The four-record admission uses four waveform conditions and 20 total views.
Waveform-only and joint conditions share the exact waveform RNG identity;
image-only and joint conditions share image parameters per record/profile.

Image-v2 is evaluation-only, in `core/image_stress.py`, not a change to the
training pool: perspective +/-4 degrees and projective coefficients +/-0.035
on a 0.84-scale white canvas (retain corners); 0.4x resolution then bilinear
upsampling and 3x3 box blur; gamma 0.8–1.2 and smooth local shade depth 0.25–0.40.
No fake JPEG, cutout, lead permutation, annotations or diagnostic text. These
are synthetic acquisition stresses, not medically validated hospital simulators.
Inspect smoke previews before launching the four-center pilot; do not tune these
parameters against model scores. Low-resolution images may lose diagnostic detail;
report that limitation, not automatic clinical label-preservation guarantees.

Launch only through `boot_scripts/run_experiment.py` with
`configs/experiments/pulse_joint_{smoke,ningbo,chapman_shaoxing,cpsc_2018,georgia}.yaml`.
GPU UUIDs remain external; at most four exclusive devices. Images are generated
online; previews only in owned RAM, predictions/configs/hashes in the external run
root `/home/linbinhao/ECG_adv_data/runs/pulse_joint_20260917`.

Report macro-F1, exact match, Hamming loss, parse/truncation rates, per center and
family. Average views then centers equally. Primary contrast: three minus single
on joint views, alongside clean-only and original controls. Report incremental
joint minus waveform-only degradation on identical records. Bootstrap original
records within center, not individual views; this does not capture training-seed
uncertainty or correct development selection bias. Patient independence is unverified.
