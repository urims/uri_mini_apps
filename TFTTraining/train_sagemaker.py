"""
Strategy 2: Distributed GPU Training — Amazon SageMaker.

Best for: Production training, large datasets, multi-GPU.
Key optimizations:
    - MirroredStrategy for synchronous data parallelism across GPUs
    - Mixed FP16 precision for 2x throughput on Tensor Cores
    - Linear learning rate scaling with GPU count
    - Automatic S3 data loading

Estimated timing (ml.g5.12xlarge, 4× A10G):
    - ~30 sec/epoch
    - 30-50 min total to convergence
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy


def train_sagemaker_script():
    """
    SageMaker training script entry point.

    This function is called inside the SageMaker container.
    Data is pre-loaded to /opt/ml/input/data/ from S3.
    """
    # ─── SageMaker environment ───
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_dir = os.environ.get("SM_CHANNEL_VAL", "/opt/ml/input/data/val")
    n_gpus = int(os.environ.get("SM_NUM_GPUS", 1))

    print(f"GPUs available: {n_gpus}")

    # ─── Multi-GPU strategy ───
    if n_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    print(f"Replicas: {strategy.num_replicas_in_sync}")

    # ─── Scale batch size and LR with GPU count ───
    PER_GPU_BATCH = 512
    GLOBAL_BATCH = PER_GPU_BATCH * strategy.num_replicas_in_sync
    BASE_LR = 1e-3
    SCALED_LR = BASE_LR * strategy.num_replicas_in_sync

    # ─── Mixed precision ───
    set_global_policy("mixed_float16")

    # ─── Load data ───
    X_static = np.load(f"{train_dir}/X_static.npy")
    X_hist = np.load(f"{train_dir}/X_hist.npy")
    X_future = np.load(f"{train_dir}/X_future.npy")
    Y_train = np.load(f"{train_dir}/Y_train.npy")

    X_val_s = np.load(f"{val_dir}/X_static.npy")
    X_val_h = np.load(f"{val_dir}/X_hist.npy")
    X_val_f = np.load(f"{val_dir}/X_future.npy")
    Y_val = np.load(f"{val_dir}/Y_val.npy")

    # ─── Datasets ───
    train_ds = (
        tf.data.Dataset.from_tensor_slices((
            {"static": X_static, "historical": X_hist, "known_future": X_future},
            Y_train,
        ))
        .shuffle(100_000)
        .batch(GLOBAL_BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((
            {"static": X_val_s, "historical": X_val_h, "known_future": X_val_f},
            Y_val,
        ))
        .batch(GLOBAL_BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ─── Build model inside strategy scope ───
    with strategy.scope():
        # Import here to avoid issues outside SageMaker
        from tft_core.model import build_tft
        from tft_core.losses import quantile_loss

        model = build_tft(
            n_static=4, n_hist=5, n_future=5, n_components=5,
            T=24, H=6, d_model=128, n_heads=4, n_quantiles=3, dropout=0.1,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=SCALED_LR, clipnorm=1.0)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.compile(optimizer=optimizer, loss=quantile_loss)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"{model_dir}/tft_best.keras", monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=f"{model_dir}/logs"),
    ]

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=100, callbacks=callbacks, verbose=2,
    )

    model.save(f"{model_dir}/tft_final")
    print(f"Model saved to {model_dir}/tft_final")


def launch_sagemaker_job(
    entry_point: str = "training/train_sagemaker.py",
    source_dir: str = ".",
    instance_type: str = "ml.g5.12xlarge",
    train_s3_uri: str = None,
    val_s3_uri: str = None,
):
    """
    Launch a SageMaker training job from your local machine.

    Requires: pip install sagemaker boto3

    Args:
        entry_point: Path to training script.
        source_dir: Source directory to upload.
        instance_type: SageMaker instance type.
            - ml.g5.12xlarge:  4× A10G (96 GB VRAM total) — recommended
            - ml.p3.8xlarge:   4× V100 (64 GB)
            - ml.p4d.24xlarge: 8× A100 (640 GB) — massive jobs
        train_s3_uri: S3 URI for training data.
        val_s3_uri: S3 URI for validation data.
    """
    import sagemaker
    from sagemaker.tensorflow import TensorFlow

    role = sagemaker.get_execution_role()
    session = sagemaker.Session()

    if train_s3_uri is None:
        train_s3_uri = session.upload_data("data/train", key_prefix="tft/train")
    if val_s3_uri is None:
        val_s3_uri = session.upload_data("data/val", key_prefix="tft/val")

    estimator = TensorFlow(
        entry_point=entry_point,
        source_dir=source_dir,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        framework_version="2.15",
        py_version="py310",
        hyperparameters={"epochs": 100, "batch_size": 512, "d_model": 128},
        volume_size=100,
        max_run=86400,
        metric_definitions=[
            {"Name": "val_loss", "Regex": r"val_loss: ([0-9\\.]+)"},
        ],
    )

    estimator.fit({"train": train_s3_uri, "val": val_s3_uri})
    return estimator


if __name__ == "__main__":
    train_sagemaker_script()
