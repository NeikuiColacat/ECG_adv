# External Model Repositories

This directory is intentionally not used for tracked code. The thesis archive
keeps external model repositories outside git and recreates local links with:

```bash
bash scripts/bootstrap_model_repos.sh
```

Required runtime links:

- `model/DeepECG` -> EfficientNetV2 implementation used by the classifier.
- `model/ECGTwin` -> ECGTwin VAE, IBE, DiT, configs, and demo reference files.

Do not commit host-specific symlink targets from this directory.
