"""Style Translator IBE — counterfactual center-style mapping via Prototype Encoder."""
from .prototype_encoder import PrototypeEncoder
from .model import StyleTranslatorIBE
from .losses import SupConLoss, style_translator_loss

__all__ = ["PrototypeEncoder", "StyleTranslatorIBE", "SupConLoss", "style_translator_loss"]
