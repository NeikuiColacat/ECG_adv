"""Manually rebuilt ECG model definitions and their single factory interface."""

from models.checkpoints import (
    CheckpointIdentity,
    extract_state_dict,
    load_model_checkpoint,
)
from models.contracts import (
    CLASS_ORDER,
    ECGFOUNDER_SPEC,
    EFFICIENTNET1DV2_SPEC,
    ModelSpec,
    validate_model_input,
    validate_model_output,
)
from models.ecgfounder import ECGFounderNet1D, build_ecgfounder
from models.efficientnet1d import EfficientNet1DV2, build_efficientnet1dv2
from models.factory import available_models, build_model, get_model_spec
from models.input_adapter import prepare_canonical_model_input
from models.vae import (
    ECGTWIN_TO_PTBXL_INDICES,
    LATENT_SCALE,
    VAEConfig,
    VAEDecoder,
    VAEEncoder,
    build_ecgtwin_vae,
    decode_to_ptbxl_waveform,
    load_vae_config,
    prepare_ecgtwin_encoder_input,
)


__all__ = [
    "CLASS_ORDER",
    "ECGFOUNDER_SPEC",
    "EFFICIENTNET1DV2_SPEC",
    "CheckpointIdentity",
    "ECGFounderNet1D",
    "EfficientNet1DV2",
    "ECGTWIN_TO_PTBXL_INDICES",
    "LATENT_SCALE",
    "ModelSpec",
    "VAEConfig",
    "VAEDecoder",
    "VAEEncoder",
    "available_models",
    "build_ecgfounder",
    "build_efficientnet1dv2",
    "build_ecgtwin_vae",
    "decode_to_ptbxl_waveform",
    "build_model",
    "extract_state_dict",
    "get_model_spec",
    "load_model_checkpoint",
    "load_vae_config",
    "prepare_canonical_model_input",
    "prepare_ecgtwin_encoder_input",
    "validate_model_input",
    "validate_model_output",
]
