"""
Strategy 4: Local CPU Training (Laptop/Desktop).

Best for: Prototyping, small datasets, no GPU available.
Key adaptations:
    - Reduced model size (d_model=32, n_heads=2)
    - Shorter sequences (T=12, H=4)
    - Data subsampling (max 50K samples)
    - Aggressive early stopping (patience=8)

Estimated timing (8-core laptop):
    - ~15-40 min/epoch
    - 3-12 hours total (early stopping ~20-30 epochs)
    - Good for prototyping, NOT production quality
"""

import os
import tensorflow as tf
from typing import List
import numpy as np


def configure_cpu_local():
    """Configure local CPU with available cores."""
    n_cores = os.cpu_count() or 4
    print(f"CPU cores available: {n_cores}")

    tf.config.threading.set_intra_op_parallelism_threads(n_cores)
    tf.config.threading.set_inter_op_parallelism_threads(max(2, n_cores // 4))

    return n_cores


def train_cpu_local(
    model: tf.keras.Model,
    X_train: List[np.ndarray],
    Y_train: np.ndarray,
    X_val: List[np.ndarray],
    Y_val: np.ndarray,
    loss_fn,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 8,
    max_train_samples: int = 50_000,
    checkpoint_path: str = "checkpoints/tft_cpu_local_best.keras",
) -> tf.keras.callbacks.History:
    """
    Train TFT on a local CPU (laptop/desktop).

    Automatically subsamples data if too large for CPU training.

    Args:
        model: TFT model (smaller config recommended: d_model=32, n_heads=2).
        X_train, Y_train: Training data.
        X_val, Y_val: Validation data.
        loss_fn: Loss function.
        batch_size: Small batch for limited RAM.
        max_epochs: Fewer epochs for CPU.
        patience: Aggressive early stopping.
        max_train_samples: Cap training samples for feasibility.
        checkpoint_path: Best model save path.

    Returns:
        Training history.
    """
    configure_cpu_local()

    # ─── Subsample if too large ───
    n = len(X_train[0])
    if n > max_train_samples:
        print(f"Subsampling {n} → {max_train_samples} training samples")
        idx = np.random.choice(n, max_train_samples, replace=False)
        X_train = [x[idx] for x in X_train]
        Y_train = Y_train[idx]

    # ─── Lightweight data pipeline ───
    dataset_train = (
        tf.data.Dataset.from_tensor_slices((
            {"static": X_train[0], "historical": X_train[1], "known_future": X_train[2]},
            Y_train,
        ))
        .shuffle(5_000)
        .batch(batch_size)
        .prefetch(2)
    )

    dataset_val = (
        tf.data.Dataset.from_tensor_slices((
            {"static": X_val[0], "historical": X_val[1], "known_future": X_val[2]},
            Y_val,
        ))
        .batch(batch_size)
        .prefetch(2)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss=loss_fn,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
    ]

    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history
