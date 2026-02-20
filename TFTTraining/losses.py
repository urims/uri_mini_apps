"""
TFT Loss Functions — Quantile (Pinball) Loss for probabilistic forecasting.
"""

import tensorflow as tf


def quantile_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Standard quantile (pinball) loss for TFT.

    Computes the average quantile loss across P10, P50, P90.

    L_q(y, ŷ) = q · max(y − ŷ, 0) + (1 − q) · max(ŷ − y, 0)

    Args:
        y_true: Actual values, shape (batch, H, 1).
        y_pred: Predicted quantiles, shape (batch, H, n_quantiles).

    Returns:
        Scalar loss averaged across all quantiles, timesteps, and batch.
    """
    quantiles = [0.1, 0.5, 0.9]

    total_loss = 0.0
    for i, q in enumerate(quantiles):
        pred_q = y_pred[:, :, i : i + 1]
        error = y_true - pred_q
        ql = tf.maximum(q * error, (q - 1.0) * error)
        total_loss += tf.reduce_mean(ql)

    return total_loss / len(quantiles)


def weighted_quantile_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    quantile_weights: tuple = (1.0, 2.0, 1.0),
) -> tf.Tensor:
    """
    Weighted quantile loss — upweight P50 (median) for better point accuracy.

    Args:
        y_true: Actual values, shape (batch, H, 1).
        y_pred: Predicted quantiles, shape (batch, H, n_quantiles).
        quantile_weights: Per-quantile weights (P10, P50, P90).

    Returns:
        Scalar weighted loss.
    """
    quantiles = [0.1, 0.5, 0.9]

    total_loss = 0.0
    total_weight = 0.0
    for i, (q, w) in enumerate(zip(quantiles, quantile_weights)):
        pred_q = y_pred[:, :, i : i + 1]
        error = y_true - pred_q
        ql = tf.maximum(q * error, (q - 1.0) * error)
        total_loss += w * tf.reduce_mean(ql)
        total_weight += w

    return total_loss / total_weight
