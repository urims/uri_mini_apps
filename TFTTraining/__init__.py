"""
TFT Core — Standard Temporal Fusion Transformer for Multi-Component Forecasting.

Implements:
    - GatedResidualNetwork (GRN)
    - VariableSelectionNetwork (VSN)
    - InterpretableMultiHeadAttention (IMHA)
    - build_tft() model factory
    - quantile_loss
    - Data preparation pipeline
"""

from tft_core.layers import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
)
from tft_core.model import build_tft
from tft_core.losses import quantile_loss

__all__ = [
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "InterpretableMultiHeadAttention",
    "build_tft",
    "quantile_loss",
]
