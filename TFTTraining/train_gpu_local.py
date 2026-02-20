"""
Strategy 3: Local GPU Training (Single GPU).

Best for: RTX 3080/3090/4090 or A-series GPUs.
Key optimizations:
    - Mixed FP16 precision for 2x throughput on Tensor Cores
    - Memory growth enabled (don't grab all VRAM upfront)
    - TensorBoard profiling to identify bottlenecks
    - Auto-tuned prefetch buffer

Estimated timing (RTX 3090):
    - ~2-5 min/epoch
    - 1-3 hours to convergence
"""

import tensorflow as tf
from typing import List
import numpy as np


def configure_gpu_local():
    """Configure local GPU with memory growth and mixed precision."""
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)}")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Mixed precision: FP16 compute, FP32 accumulate
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


def train_gpu_local(
    model: tf.keras.Model,
    X_train: List[np.ndarray],
    Y_train: np.ndarray,
    X_val: List[np.ndarray],
    Y_val: np.ndarray,
    loss_fn,
    batch_size: int = 512,
    max_epochs: int = 100,
    patience: int = 12,
    checkpoint_path: str = "checkpoints/tft_gpu_best.keras",
    log_dir: str = "logs/gpu_local",
) -> tf.keras.callbacks.History:
    """
    Train TFT on a local GPU.

    Batch size guide:
        RTX 3080 (10 GB): 256-512
        RTX 3090 (24 GB): 512-1024
        RTX 4090 (24 GB): 512-1024

    Args:
        model: TFT model (uncompiled — will be compiled here with LossScale).
        X_train, Y_train: Training data.
        X_val, Y_val: Validation data.
        loss_fn: Loss function.
        batch_size: Per-GPU batch size.
        max_epochs: Maximum epochs.
        patience: Early stopping patience.
        checkpoint_path: Best model save path.
        log_dir: TensorBoard directory.

    Returns:
        Training history.
    """
    configure_gpu_local()

    # ─── GPU-optimized data pipeline ───
    dataset_train = (
        tf.data.Dataset.from_tensor_slices((
            {"static": X_train[0], "historical": X_train[1], "known_future": X_train[2]},
            Y_train,
        ))
        .shuffle(buffer_size=20_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    dataset_val = (
        tf.data.Dataset.from_tensor_slices((
            {"static": X_val[0], "historical": X_val[1], "known_future": X_val[2]},
            Y_val,
        ))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ─── Optimizer with loss scaling for FP16 stability ───
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer=optimizer, loss=loss_fn)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, profile_batch=(10, 20),
        ),
    ]

    # Monitor GPU: run `nvidia-smi -l 1` in separate terminal
    # Target: >85% GPU utilization, >80% memory usage

    history = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history
