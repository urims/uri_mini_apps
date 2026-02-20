"""
Constraint Layers — α-lag floor enforcement for sell price forecasting.

The business rule:
    sell_price(t) >= contract(t − α) × (1 + margin)

Two layers enforce this:
    1. AlphaShiftLayer: Computes the effective floor by shifting the known
       future contract backward by α timesteps and adding a margin.
    2. SoftConstraintLayer: Clamps all quantile predictions above the floor.
       Uses differentiable LogSumExp during training, hard max at inference.
"""

import tensorflow as tf


class AlphaShiftLayer(tf.keras.layers.Layer):
    """
    Computes the effective contract floor with α-lag adjustment.

    The sell price at forecast time t is not constrained by the contract
    at time t, but by the contract that was effective α steps earlier —
    because it takes α periods for contract changes to propagate into
    procurement, inventory, and pricing systems.

    floor(t) = contract(t − α) × (1 + margin)

    Args:
        alpha: Number of timesteps for contract propagation lag.
        margin: Minimum margin above the shifted contract (e.g., 0.02 = 2%).
    """

    def __init__(self, alpha: int = 3, margin: float = 0.02, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.margin = margin

    def call(self, future_contracts: tf.Tensor) -> tf.Tensor:
        """
        Args:
            future_contracts: Known future contract values, shape (batch, H, 1).

        Returns:
            floor: Effective floor values, shape (batch, H, 1).
        """
        # Shift backward by alpha: pad start with the first value
        padding = tf.repeat(future_contracts[:, :1, :], self.alpha, axis=1)
        shifted = tf.concat(
            [padding, future_contracts[:, : -self.alpha, :]], axis=1
        )
        # Apply margin
        floor = shifted * (1.0 + self.margin)
        return floor

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "margin": self.margin})
        return config


class SoftConstraintLayer(tf.keras.layers.Layer):
    """
    Enforces floor constraint on all quantile predictions.

    During training:
        Uses LogSumExp smooth approximation of max():
            softmax(a, b) = log(exp(τa) + exp(τb)) / τ
        This is differentiable everywhere, so gradients flow through
        the constraint and the TFT learns to stay above the floor.

    During inference:
        Uses hard max() for exact guarantee:
            constrained(t) = max(raw_pred(t), floor(t))

    Applied independently to each quantile (P10, P50, P90).
    P10 is the binding constraint — if P10 >= floor, then P50 and P90
    are automatically above floor too (since P10 <= P50 <= P90 by
    the quantile ordering property learned during training).

    Args:
        temperature: Sharpness of the soft approximation.
            Higher τ → closer to hard max, but steeper gradients.
            Recommended: 5.0–20.0.
    """

    def __init__(self, temperature: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs, training=None):
        """
        Args:
            inputs: Tuple of (raw_quantiles, floor).
                raw_quantiles: (batch, H, n_quantiles) — unconstrained TFT output.
                floor: (batch, H, 1) — from AlphaShiftLayer.

        Returns:
            Constrained quantiles, shape (batch, H, n_quantiles).
        """
        raw_quantiles, floor = inputs
        n_quantiles = raw_quantiles.shape[-1]

        # Broadcast floor to all quantiles
        floor_broadcast = tf.repeat(floor, n_quantiles, axis=-1)

        if training:
            # Differentiable smooth max via LogSumExp
            stacked = tf.stack(
                [
                    raw_quantiles * self.temperature,
                    floor_broadcast * self.temperature,
                ],
                axis=-1,
            )
            constrained = (
                tf.reduce_logsumexp(stacked, axis=-1) / self.temperature
            )
        else:
            # Hard max for exact guarantee at inference
            constrained = tf.maximum(raw_quantiles, floor_broadcast)

        return constrained

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config
