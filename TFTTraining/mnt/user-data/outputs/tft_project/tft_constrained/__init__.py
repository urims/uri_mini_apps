"""
TFT Constrained — Temporal Fusion Transformer with α-lag contract floor enforcement.

Extends the standard TFT with:
    - AlphaShiftLayer: floor(t) = contract(t − α) × (1 + margin)
    - SoftConstraintLayer: LogSumExp (training) / hard max (inference)
    - constrained_quantile_loss: pinball loss with optional violation penalty
"""

from tft_constrained.constraint_layers import AlphaShiftLayer, SoftConstraintLayer
from tft_constrained.model import build_constrained_tft
from tft_constrained.losses import constrained_quantile_loss

__all__ = [
    "AlphaShiftLayer",
    "SoftConstraintLayer",
    "build_constrained_tft",
    "constrained_quantile_loss",
]
