"""
Constrained TFT Model — TFT + α-lag contract floor enforcement.

Extends the standard TFT by adding two layers after the quantile output:
    1. AlphaShiftLayer → computes effective floor from known future contracts
    2. SoftConstraintLayer → clamps all quantiles above the floor

The constraint layers sit AFTER the TFT — they don't interfere with how
the TFT learns temporal patterns. The TFT produces unconstrained predictions,
then the constraint layers enforce the business rule.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    LayerNormalization,
    Concatenate,
    Lambda,
    Embedding,
)
from tensorflow.keras.models import Model

from tft_core.layers import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
)
from tft_constrained.constraint_layers import AlphaShiftLayer, SoftConstraintLayer


def build_constrained_tft(
    n_static: int = 4,
    n_hist: int = 5,
    n_future: int = 5,
    n_components: int = 5,
    T: int = 24,
    H: int = 6,
    d_model: int = 64,
    n_heads: int = 4,
    n_quantiles: int = 3,
    alpha: int = 3,
    margin: float = 0.02,
    soft_temp: float = 10.0,
    dropout: float = 0.1,
    contract_feature_index: int = -1,
) -> Model:
    """
    Build a Constrained Temporal Fusion Transformer.

    Same architecture as standard TFT, plus:
        - AlphaShiftLayer: floor(t) = contract(t−α) × (1+margin)
        - SoftConstraintLayer: ensures all quantiles >= floor

    Args:
        n_static: Number of static covariate features.
        n_hist: Number of historical features per timestep.
        n_future: Number of known future features per timestep.
        n_components: Number of unique components.
        T: Lookback window.
        H: Forecast horizon.
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        n_quantiles: Number of output quantiles.
        alpha: Contract-to-sell propagation lag (timesteps).
        margin: Minimum margin above α-shifted contract floor (e.g., 0.02 = 2%).
        soft_temp: LogSumExp temperature for soft constraint during training.
        dropout: Dropout rate.
        contract_feature_index: Index of contract_value in the future input
            features (default -1 = last feature).

    Returns:
        tf.keras.Model with inputs [static, historical, known_future]
        and output shape (batch, H, n_quantiles), constrained.
    """

    # ═══ INPUTS ═══
    static_input = Input(shape=(n_static,), name="static")
    hist_input = Input(shape=(T, n_hist), name="historical")
    future_input = Input(shape=(H, n_future), name="known_future")

    # Extract contract values for constraint (bypasses TFT entirely)
    if contract_feature_index == -1:
        future_contract = Lambda(
            lambda x: x[:, :, -1:], name="extract_future_contract"
        )(future_input)
    else:
        idx = contract_feature_index
        future_contract = Lambda(
            lambda x: x[:, :, idx : idx + 1], name="extract_future_contract"
        )(future_input)

    # ═══ STATIC PROCESSING ═══
    comp_embedding = Embedding(
        n_components, d_model, name="comp_embedding"
    )(tf.cast(static_input[:, 0], tf.int32))

    static_vsn = VariableSelectionNetwork(
        n_static, d_model, dropout, name="static_vsn"
    )
    static_selected, static_var_weights = static_vsn(static_input)

    c_enrichment = GatedResidualNetwork(d_model, name="c_enrichment")(static_selected)
    c_encoder = GatedResidualNetwork(d_model, name="c_encoder")(static_selected)
    c_decoder = GatedResidualNetwork(d_model, name="c_decoder")(static_selected)
    c_state_h = Dense(d_model, name="c_state_h")(static_selected)
    c_state_c = Dense(d_model, name="c_state_c")(static_selected)

    # ═══ TEMPORAL VARIABLE SELECTION ═══
    hist_vsn = VariableSelectionNetwork(
        n_hist, d_model, dropout, name="hist_vsn"
    )
    hist_selected, hist_var_weights = hist_vsn(hist_input, context=c_encoder)

    future_vsn = VariableSelectionNetwork(
        n_future, d_model, dropout, name="future_vsn"
    )
    future_selected, future_var_weights = future_vsn(future_input, context=c_decoder)

    # ═══ LSTM ENCODER ═══
    enc_lstm = LSTM(
        d_model, return_sequences=True, return_state=True,
        dropout=dropout, name="enc_lstm",
    )
    enc_out, enc_h, enc_c = enc_lstm(
        hist_selected, initial_state=[c_state_h, c_state_c]
    )

    # ═══ LSTM DECODER ═══
    dec_lstm = LSTM(
        d_model, return_sequences=True, dropout=dropout, name="dec_lstm",
    )
    dec_out = dec_lstm(future_selected, initial_state=[enc_h, enc_c])

    # ═══ GATED SKIP (LSTM) ═══
    lstm_all = Concatenate(axis=1, name="cat_lstm")([enc_out, dec_out])
    vsn_all = Concatenate(axis=1, name="cat_vsn")([hist_selected, future_selected])
    gated_lstm = GatedResidualNetwork(d_model, name="glu_lstm")
    temporal = gated_lstm(lstm_all)
    temporal = LayerNormalization(name="ln_lstm")(temporal + vsn_all)

    # ═══ STATIC ENRICHMENT ═══
    enrichment = GatedResidualNetwork(d_model, name="static_enrichment")
    enriched = enrichment(temporal, context=c_enrichment)

    # ═══ INTERPRETABLE MULTI-HEAD ATTENTION ═══
    attn_layer = InterpretableMultiHeadAttention(
        n_heads, d_model, dropout, name="imha"
    )
    decoder_enriched = enriched[:, T:, :]
    attn_out, attn_weights = attn_layer(decoder_enriched, enriched, enriched)

    gated_attn = GatedResidualNetwork(d_model, name="glu_attn")
    attn_gated = gated_attn(attn_out)
    attn_gated = LayerNormalization(name="ln_attn")(attn_gated + decoder_enriched)

    # ═══ POSITIONWISE FEEDFORWARD ═══
    ff = GatedResidualNetwork(d_model, name="ff_grn")
    ff_out = ff(attn_gated)
    ff_out = LayerNormalization(name="ln_ff")(ff_out + attn_gated)

    # ═══ RAW QUANTILE PREDICTION (unconstrained) ═══
    raw_quantiles = Dense(
        n_quantiles, activation="linear", dtype="float32",
        name="raw_quantiles",
    )(ff_out)

    # ═══ α-SHIFT CONTRACT FLOOR ═══
    alpha_shift = AlphaShiftLayer(alpha=alpha, margin=margin, name="alpha_shift")
    floor_values = alpha_shift(future_contract)

    # ═══ APPLY CONSTRAINT ═══
    constraint = SoftConstraintLayer(temperature=soft_temp, name="floor_constraint")
    constrained_quantiles = constraint([raw_quantiles, floor_values])

    # ═══ ASSEMBLE ═══
    model = Model(
        inputs=[static_input, hist_input, future_input],
        outputs=constrained_quantiles,
        name="Constrained_TFT",
    )

    return model
