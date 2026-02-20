"""
Strategy 1: Single Server — 72 CPU Cores, 148 GB RAM.

Best for: Large datasets that fit in RAM, no GPU available.
Key optimizations:
    - 72 intra-op threads for maximum CPU parallelism
    - MKL/oneDNN optimized math kernels
    - Large batch size (2048) to fully utilize cores
    - Full dataset cached in RAM (148 GB)
    - Warmup + Cosine Decay learning rate schedule

Estimated timing:
    - ~45 min/epoch with 300K samples
    - 8-15 hours total (early stopping ~50-60 epochs)
"""

import os
import tensorflow as tf
from typing import List
import numpy as np


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""

    def __init__(self, peak_lr: float, warmup_steps: int, decay_steps: int):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=peak_lr,
            decay_steps=decay_steps,
            alpha=1e-5,
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.peak_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        decayed = self.decay_schedule(step - tf.cast(self.warmup_steps, tf.float32))
        return tf.where(step < self.warmup_steps, warmup, decayed)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
        }


def configure_cpu_72core():
    """Set environment variables for 72-core CPU optimization."""
    os.environ["OMP_NUM_THREADS"] = "72"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "72"
    os.environ["TF_NUM_INTEROP_THREADS"] = "12"
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

    tf.config.threading.set_intra_op_parallelism_threads(72)
    tf.config.threading.set_inter_op_parallelism_threads(12)


def train_cpu_72core(
    model: tf.keras.Model,
    X_train: List[np.ndarray],
    Y_train: np.ndarray,
    X_val: List[np.ndarray],
    Y_val: np.ndarray,
    loss_fn,
    batch_size: int = 2048,
    max_epochs: int = 100,
    patience: int = 15,
    checkpoint_path: str = "checkpoints/tft_cpu72_best.keras",
    log_dir: str = "logs/cpu72",
) -> tf.keras.callbacks.History:
    """
    Train TFT on a 72-core CPU server with 148 GB RAM.

    Args:
        model: Compiled or uncompiled TFT model.
        X_train: [X_static, X_hist, X_future] training arrays.
        Y_train: Training targets.
        X_val: Validation inputs.
        Y_val: Validation targets.
        loss_fn: Loss function (quantile_loss or constrained_quantile_loss).
        batch_size: Batch size (large for CPU parallelism).
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        checkpoint_path: Path for best model checkpoint.
        log_dir: TensorBoard log directory.

    Returns:
        Training history.
    """
    configure_cpu_72core()

    # ─── Build tf.data pipelines ───
    dataset_train = tf.data.Dataset.from_tensor_slices((
        {"static": X_train[0], "historical": X_train[1], "known_future": X_train[2]},
        Y_train,
    ))
    dataset_train = (
        dataset_train
        .cache()
        .shuffle(buffer_size=50_000)
        .batch(batch_size)
        .prefetch(4)
    )

    dataset_val = tf.data.Dataset.from_tensor_slices((
        {"static": X_val[0], "historical": X_val[1], "known_future": X_val[2]},
        Y_val,
    )).batch(batch_size).cache().prefetch(4)

    # ─── Learning rate schedule ───
    lr_schedule = WarmupCosineSchedule(
        peak_lr=1e-3, warmup_steps=500, decay_steps=10_000
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, clipnorm=1.0
    )
    model.compile(optimizer=optimizer, loss=loss_fn)

    # ─── Callbacks ───
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]

    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history
