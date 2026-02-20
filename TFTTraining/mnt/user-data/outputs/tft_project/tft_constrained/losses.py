"""
Constrained Quantile Loss — Pinball loss + optional floor violation penalty.
"""

import tensorflow as tf


def constrained_quantile_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    violation_weight: float = 0.0,
    floor_value: tf.Tensor = None,
) -> tf.Tensor:
    """
    Quantile loss for constrained TFT.

    Even though the SoftConstraintLayer prevents violations architecturally,
    an optional penalty term provides gradient pressure to keep raw predictions
    above the floor naturally — reducing reliance on the constraint correction.

    Args:
        y_true: Actual values, shape (batch, H, 1).
        y_pred: Constrained quantile predictions, shape (batch, H, 3).
        violation_weight: Penalty weight for floor violations (default 0).
        floor_value: Floor tensor for penalty computation (optional).

    Returns:
        Scalar loss.
    """
    quantiles = [0.1, 0.5, 0.9]

    total_loss = 0.0
    for i, q in enumerate(quantiles):
        pred_q = y_pred[:, :, i : i + 1]
        error = y_true - pred_q
        ql = tf.maximum(q * error, (q - 1.0) * error)
        total_loss += tf.reduce_mean(ql)

    loss = total_loss / len(quantiles)

    # Optional violation penalty
    if violation_weight > 0 and floor_value is not None:
        for i in range(len(quantiles)):
            pred_q = y_pred[:, :, i : i + 1]
            violation = tf.maximum(0.0, floor_value - pred_q)
            loss += violation_weight * tf.reduce_mean(tf.square(violation))

    return loss


def make_constrained_quantile_loss(
    violation_weight: float = 0.0,
):
    """
    Factory for a constrained quantile loss compatible with model.compile().

    When violation_weight=0 (default), this is identical to standard
    quantile loss — the constraint is handled entirely by the architecture.

    Args:
        violation_weight: Penalty multiplier for floor violations.

    Returns:
        Loss function with signature (y_true, y_pred) -> scalar.
    """

    def loss_fn(y_true, y_pred):
        quantiles = [0.1, 0.5, 0.9]
        total_loss = 0.0
        for i, q in enumerate(quantiles):
            pred_q = y_pred[:, :, i : i + 1]
            error = y_true - pred_q
            ql = tf.maximum(q * error, (q - 1.0) * error)
            total_loss += tf.reduce_mean(ql)
        return total_loss / len(quantiles)

    loss_fn.__name__ = "constrained_quantile_loss"
    return loss_fn
