# Graduate Project Pipeline Pointer

This archive branch keeps only short pipeline pointers under `docs/pipelines/`.
The authoritative graduation-thesis reproduction route is:

- `README.md`
- `docs/thesis_reproduction.md`
- `docs/thesis_repro_manifest.json`
- `docs/artifact_manifest.json`
- `docs/thesis_archive_completion_audit.md`

The active thesis protocol is:

```text
PTB-XL super5 fixed low-sample split
-> EfficientNetV2 real2000 baseline
-> ECGTwin no-token synthetic pretrain + real2000 fine-tune
-> ECGTwin center-class 768-d prompt-token synthetic pretrain + real2000 fine-tune
-> synthetic ECG quality proxy analysis
-> ONNX/TensorRT inference benchmark
-> Streamlit demonstration system
```

Fixed classification split:

```text
train_real = 2000
val_real   = 2000
test_real  = 17799
seed       = 42
classes    = CD, HYP, MI, NORM, STTC
input      = 100 Hz, 1000 points, PTB-XL lead order
preprocess = minimal_resample + per_sample_global
```

Use `scripts/final_round/run_thesis_reproduction.sh` for all thesis-facing
commands. Historical pipeline writeups, PN2021 experiments, Latent-Hull/PGD
work, TA-OMAT, AdvDiff, AugMix, and self-distillation explorations live under
`legacy/` and are not the final undergraduate-thesis route.

