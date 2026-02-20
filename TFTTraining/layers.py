"""
TFT Core Layers — Building blocks for the Temporal Fusion Transformer.

Contains:
    - GatedResidualNetwork (GRN): Core nonlinear processing unit with gating
    - VariableSelectionNetwork (VSN): Learns per-variable importance via softmax
    - InterpretableMultiHeadAttention (IMHA): Shared-V attention for interpretability
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
    Softmax,
)


class GatedResidualNetwork(tf.keras.layers.Layer):
    """
    Gated Residual Network — the fundamental building block of TFT.

    GRN(a, c) = LayerNorm(a + GLU(η))
    where η = W₂ · ELU(W₁ · a + W₃ · c + b) + b₂

    Used in:
        - Variable selection (per-variable transform + selection weights)
        - Static enrichment (inject component identity into temporal states)
        - Positionwise feedforward (final nonlinear transform before output)

    Args:
        d_model: Hidden dimension.
        dropout: Dropout rate applied after gating.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self._dropout_rate = dropout

    def build(self, input_shape):
        self.dense1 = Dense(self.d_model, activation="elu", name="grn_dense1")
        self.dense_ctx = Dense(self.d_model, use_bias=False, name="grn_ctx")
        self.dense2 = Dense(self.d_model, name="grn_dense2")
        self.gate = Dense(self.d_model, activation="sigmoid", name="grn_gate")
        self.drop = Dropout(self._dropout_rate)
        self.norm = LayerNormalization()
        self.project = Dense(self.d_model, name="grn_project")
        super().build(input_shape)

    def call(self, x, context=None, training=None):
        residual = self.project(x)

        if context is not None:
            # Broadcast context to match x dimensions
            if len(context.shape) < len(x.shape):
                context = tf.expand_dims(context, 1)
                context = tf.repeat(context, tf.shape(x)[1], axis=1)
            x = self.dense1(x + self.dense_ctx(context))
        else:
            x = self.dense1(x)

        x = self.dense2(x)
        x = self.drop(x, training=training)

        # GLU: element-wise gating
        gate_val = self.gate(x)
        x = gate_val * x

        return self.norm(x + residual)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "dropout": self._dropout_rate})
        return config


class VariableSelectionNetwork(tf.keras.layers.Layer):
    """
    Variable Selection Network — learns WHICH input features matter.

    For each input variable:
        1. Transform via independent GRN
        2. Compute softmax selection weights (interpretable!)
        3. Weighted combination → selected representation

    The selection weights are directly interpretable as feature importance.
    Can be conditioned on static context (component identity).

    Args:
        n_vars: Number of input variables.
        d_model: Hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_vars = n_vars
        self.d_model = d_model
        self._dropout_rate = dropout

    def build(self, input_shape):
        self.var_transforms = [
            Dense(self.d_model, name=f"vsn_transform_{i}")
            for i in range(self.n_vars)
        ]
        self.var_grns = [
            GatedResidualNetwork(self.d_model, self._dropout_rate, name=f"vsn_grn_{i}")
            for i in range(self.n_vars)
        ]
        self.selection_weights = Dense(
            self.n_vars, activation="softmax", name="vsn_softmax"
        )
        self.flattened_grn = GatedResidualNetwork(
            self.d_model, self._dropout_rate, name="vsn_flattened_grn"
        )
        super().build(input_shape)

    def call(self, inputs, context=None, training=None):
        """
        Args:
            inputs: Tensor of shape (batch, [T,] n_vars * var_dim).
            context: Optional static context tensor.

        Returns:
            selected: Weighted combination, shape (batch, [T,] d_model).
            weights: Softmax selection weights, shape (batch, [T,] n_vars).
        """
        var_dim = inputs.shape[-1] // self.n_vars

        transformed = []
        for i in range(self.n_vars):
            var_i = inputs[..., i * var_dim : (i + 1) * var_dim]
            var_i = self.var_transforms[i](var_i)
            var_i = self.var_grns[i](var_i, context=context, training=training)
            transformed.append(var_i)

        # Stack: (..., d_model, n_vars)
        stacked = tf.stack(transformed, axis=-1)

        # Selection weights via softmax
        weights = self.selection_weights(inputs)  # (..., n_vars)
        weights_expanded = tf.expand_dims(weights, -2)  # (..., 1, n_vars)

        # Weighted sum
        selected = tf.reduce_sum(stacked * weights_expanded, axis=-1)

        return selected, weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_vars": self.n_vars,
            "d_model": self.d_model,
            "dropout": self._dropout_rate,
        })
        return config


class InterpretableMultiHeadAttention(tf.keras.layers.Layer):
    """
    TFT's modified Multi-Head Attention with shared Values.

    Standard multi-head attention: each head has independent Q, K, V.
    TFT modification: all heads SHARE the same V projection.

    Why? Because shared V means each head's attention weights have
    the same semantic meaning — they can be averaged across heads
    for a single, directly interpretable temporal importance score.

    Args:
        n_heads: Number of attention heads.
        d_model: Model dimension (must be divisible by n_heads).
        dropout: Dropout rate on attention weights.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self._dropout_rate = dropout

    def build(self, input_shape):
        # Per-head Q, K projections
        self.W_q = [
            Dense(self.d_k, name=f"attn_q_{i}") for i in range(self.n_heads)
        ]
        self.W_k = [
            Dense(self.d_k, name=f"attn_k_{i}") for i in range(self.n_heads)
        ]
        # SHARED V across all heads — the key TFT innovation
        self.W_v = Dense(self.d_model, name="attn_v_shared")
        self.W_out = Dense(self.d_model, name="attn_out")
        self.drop = Dropout(self._dropout_rate)
        super().build(input_shape)

    def call(self, queries, keys, values, mask=None, training=None):
        """
        Args:
            queries: (batch, H, d_model) — decoder positions.
            keys: (batch, T+H, d_model) — all positions.
            values: (batch, T+H, d_model) — all positions.
            mask: Optional causal mask.

        Returns:
            output: (batch, H, d_model).
            avg_attn_weights: (batch, H, T+H) — interpretable importance.
        """
        V = self.W_v(values)  # Shared: (batch, T+H, d_model)

        head_outputs = []
        attn_weights_list = []

        for i in range(self.n_heads):
            Q_i = self.W_q[i](queries)   # (batch, H, d_k)
            K_i = self.W_k[i](keys)      # (batch, T+H, d_k)

            # Scaled dot-product attention
            scores = tf.matmul(Q_i, K_i, transpose_b=True)
            scores = scores / tf.sqrt(tf.cast(self.d_k, tf.float32))

            if mask is not None:
                scores += mask * -1e9

            attn_w = tf.nn.softmax(scores, axis=-1)
            attn_w = self.drop(attn_w, training=training)
            attn_weights_list.append(attn_w)

            # Each head attends to SHARED V
            head_out = tf.matmul(attn_w, V)
            head_outputs.append(head_out)

        # Average heads (not concat) — preserves interpretability
        combined = tf.reduce_mean(tf.stack(head_outputs), axis=0)
        output = self.W_out(combined)

        # Average attention weights across heads
        avg_attn = tf.reduce_mean(tf.stack(attn_weights_list), axis=0)

        return output, avg_attn

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "dropout": self._dropout_rate,
        })
        return config
