"""
CLI Entry Point — Train TFT with automatic strategy selection.

Usage:
    python scripts/run_training.py --strategy auto
    python scripts/run_training.py --strategy gpu_local --constrained --alpha 3
    python scripts/run_training.py --strategy cpu_72core --d-model 64
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf


def detect_strategy() -> str:
    """Auto-detect the best training strategy for current hardware."""
    gpus = tf.config.list_physical_devices("GPU")
    n_cpus = os.cpu_count() or 1
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3) if hasattr(os, "sysconf") else 16

    if gpus:
        print(f"Detected {len(gpus)} GPU(s) → using gpu_local strategy")
        return "gpu_local"
    elif n_cpus >= 64 and ram_gb >= 100:
        print(f"Detected {n_cpus} CPUs, {ram_gb:.0f} GB RAM → using cpu_72core strategy")
        return "cpu_72core"
    else:
        print(f"Detected {n_cpus} CPUs, {ram_gb:.0f} GB RAM → using cpu_local strategy")
        return "cpu_local"


def main():
    parser = argparse.ArgumentParser(description="Train TFT Multi-Component Forecaster")
    parser.add_argument("--strategy", type=str, default="auto",
                        choices=["auto", "cpu_72core", "sagemaker", "gpu_local", "cpu_local"],
                        help="Training strategy")
    parser.add_argument("--constrained", action="store_true",
                        help="Use constrained TFT with α-lag floor enforcement")
    parser.add_argument("--alpha", type=int, default=3,
                        help="Contract propagation lag (α)")
    parser.add_argument("--margin", type=float, default=0.02,
                        help="Minimum margin above floor")
    parser.add_argument("--d-model", type=int, default=64,
                        help="Model hidden dimension")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--T", type=int, default=24, help="Lookback window")
    parser.add_argument("--H", type=int, default=6, help="Forecast horizon")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (auto-set per strategy if None)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing .npy data files")

    args = parser.parse_args()

    # ─── Strategy selection ───
    strategy = args.strategy if args.strategy != "auto" else detect_strategy()

    # ─── Adjust model size for CPU local ───
    d_model = args.d_model
    n_heads = args.n_heads
    T, H = args.T, args.H
    if strategy == "cpu_local":
        d_model = min(d_model, 32)
        n_heads = min(n_heads, 2)
        T = min(T, 12)
        H = min(H, 4)
        print(f"CPU local: reduced model to d={d_model}, heads={n_heads}, T={T}, H={H}")

    # ─── Load data ───
    data_dir = args.data_dir
    try:
        X_static = np.load(f"{data_dir}/X_static.npy")
        X_hist = np.load(f"{data_dir}/X_hist.npy")
        X_future = np.load(f"{data_dir}/X_future.npy")
        Y = np.load(f"{data_dir}/Y.npy")
    except FileNotFoundError:
        print(f"Data not found in {data_dir}/. Run: python scripts/generate_sample_data.py")
        sys.exit(1)

    # ─── Split ───
    n = len(X_static)
    t_end = int(n * 0.7)
    v_end = int(n * 0.85)

    X_train = [X_static[:t_end], X_hist[:t_end], X_future[:t_end]]
    Y_train = Y[:t_end]
    X_val = [X_static[t_end:v_end], X_hist[t_end:v_end], X_future[t_end:v_end]]
    Y_val = Y[t_end:v_end]

    print(f"Train: {t_end}, Val: {v_end - t_end}, Test: {n - v_end}")

    # ─── Build model ───
    if args.constrained:
        from tft_constrained.model import build_constrained_tft
        from tft_constrained.losses import make_constrained_quantile_loss

        model = build_constrained_tft(
            n_static=X_static.shape[-1], n_hist=X_hist.shape[-1],
            n_future=X_future.shape[-1], n_components=int(X_static[:, 0].max()) + 1,
            T=T, H=H, d_model=d_model, n_heads=n_heads,
            alpha=args.alpha, margin=args.margin,
        )
        loss_fn = make_constrained_quantile_loss()
        print(f"Built Constrained TFT (α={args.alpha}, margin={args.margin})")
    else:
        from tft_core.model import build_tft
        from tft_core.losses import quantile_loss

        model = build_tft(
            n_static=X_static.shape[-1], n_hist=X_hist.shape[-1],
            n_future=X_future.shape[-1], n_components=int(X_static[:, 0].max()) + 1,
            T=T, H=H, d_model=d_model, n_heads=n_heads,
        )
        loss_fn = quantile_loss
        print("Built Standard TFT")

    model.summary()

    # ─── Train with selected strategy ───
    if strategy == "cpu_72core":
        from training.train_cpu_72core import train_cpu_72core
        history = train_cpu_72core(
            model, X_train, Y_train, X_val, Y_val, loss_fn,
            batch_size=args.batch_size or 2048, max_epochs=args.epochs,
        )
    elif strategy == "gpu_local":
        from training.train_gpu_local import train_gpu_local
        history = train_gpu_local(
            model, X_train, Y_train, X_val, Y_val, loss_fn,
            batch_size=args.batch_size or 512, max_epochs=args.epochs,
        )
    elif strategy == "cpu_local":
        from training.train_cpu_local import train_cpu_local
        history = train_cpu_local(
            model, X_train, Y_train, X_val, Y_val, loss_fn,
            batch_size=args.batch_size or 64, max_epochs=min(args.epochs, 50),
        )
    elif strategy == "sagemaker":
        print("Use `python training/train_sagemaker.py` or launch_sagemaker_job()")
        sys.exit(0)

    print(f"\nTraining complete. Best val_loss: {min(history.history['val_loss']):.6f}")


if __name__ == "__main__":
    main()
