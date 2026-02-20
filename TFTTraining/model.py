"""
Standard TFT Model — Multi-Component Global Forecasting.

Builds the full Temporal Fusion Transformer:
    Input Processing → VSN → LSTM Enc/Dec → Static Enrichment →
    Interpretable Multi-Head Attention → Quantile Output

Supports:
    - Static covariates (component_id, category, supplier, warehouse)
    - Historical observed features (sales, price, contract, margin, lead_time)
    - Known future features (promo, month, quarter, day_of_week, contract)
    - Multi-component global training via static embeddings
    - Quantile outputs (P10, P50, P90) with pinball loss
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    LayerNormalization,
    Concatenate,
    Embedding,
)
from tensorflow.keras.models import Model

from tft_core.layers import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
)


def build_tft(
    n_static: int = 4,
    n_hist: int = 5,
    n_future: int = 5,
    n_components: int = 5,
    T: int = 24,
    H: int = 6,
    d_model: int = 64,
    n_heads: int = 4,
    n_quantiles: int = 3,
    dropout: float = 0.1,
) -> Model:
    """
    Build a standard Temporal Fusion Transformer.

    Args:
        n_static: Number of static covariate features.
        n_hist: Number of historical (observed) features per timestep.
        n_future: Number of known future features per timestep.
        n_components: Number of unique components (for embedding).
        T: Lookback window length.
        H: Forecast horizon length.
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        n_quantiles: Number of output quantiles (default 3: P10, P50, P90).
        dropout: Dropout rate.

    Returns:
        tf.keras.Model with inputs [static, historical, known_future]
        and output shape (batch, H, n_quantiles).
    """

    # ═══ INPUTS ═══
    static_input = Input(shape=(n_static,), name="static")
    hist_input = Input(shape=(T, n_hist), name="historical")
    future_input = Input(shape=(H, n_future), name="known_future")

    # ═══ STATIC PROCESSING ═══
    # Embed component_id (first static feature)
    comp_embedding = Embedding(
        n_components, d_model, name="comp_embedding"
    )(tf.cast(static_input[:, 0], tf.int32))

    # Static variable selection
    static_vsn = VariableSelectionNetwork(
        n_static, d_model, dropout, name="static_vsn"
    )
    static_selected, static_var_weights = static_vsn(static_input)

    # 4 context vectors derived from static features
    # Each conditions a different part of the architecture
    c_enrichment = GatedResidualNetwork(d_model, name="c_enrichment")(
        static_selected
    )
    c_encoder = GatedResidualNetwork(d_model, name="c_encoder")(static_selected)
    c_decoder = GatedResidualNetwork(d_model, name="c_decoder")(static_selected)
    c_state_h = Dense(d_model, name="c_state_h")(static_selected)
    c_state_c = Dense(d_model, name="c_state_c")(static_selected)

    # ═══ TEMPORAL VARIABLE SELECTION ═══
    hist_vsn = VariableSelectionNetwork(
        n_hist, d_model, dropout, name="hist_vsn"
    )
    hist_selected, hist_var_weights = hist_vsn(
        hist_input, context=c_encoder
    )

    future_vsn = VariableSelectionNetwork(
        n_future, d_model, dropout, name="future_vsn"
    )
    future_selected, future_var_weights = future_vsn(
        future_input, context=c_decoder
    )

    # ═══ LSTM ENCODER ═══
    enc_lstm = LSTM(
        d_model,
        return_sequences=True,
        return_state=True,
        dropout=dropout,
        name="enc_lstm",
    )
    enc_out, enc_h, enc_c = enc_lstm(
        hist_selected, initial_state=[c_state_h, c_state_c]
    )

    # ═══ LSTM DECODER ═══
    dec_lstm = LSTM(
        d_model,
        return_sequences=True,
        dropout=dropout,
        name="dec_lstm",
    )
    dec_out = dec_lstm(future_selected, initial_state=[enc_h, enc_c])

    # ═══ GATED SKIP CONNECTION (LSTM) ═══
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
    decoder_enriched = enriched[:, T:, :]  # queries: decoder positions only
    attn_out, attn_weights = attn_layer(
        decoder_enriched, enriched, enriched
    )

    # Gated skip connection (attention)
    gated_attn = GatedResidualNetwork(d_model, name="glu_attn")
    attn_gated = gated_attn(attn_out)
    attn_gated = LayerNormalization(name="ln_attn")(
        attn_gated + decoder_enriched
    )

    # ═══ POSITIONWISE FEEDFORWARD ═══
    ff = GatedResidualNetwork(d_model, name="ff_grn")
    ff_out = ff(attn_gated)
    ff_out = LayerNormalization(name="ln_ff")(ff_out + attn_gated)

    # ═══ QUANTILE OUTPUT ═══
    output = Dense(
        n_quantiles,
        activation="linear",
        dtype="float32",
        name="quantile_output",
    )(ff_out)
    # Shape: (batch, H, n_quantiles)

    model = Model(
        inputs=[static_input, hist_input, future_input],
        outputs=output,
        name="TemporalFusionTransformer",
    )

    return model
