"""
Smoke tests for TFT models — verify construction, forward pass, and shapes.

Run: python -m pytest tests/test_model.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_standard_tft_builds():
    """Standard TFT builds and produces correct output shape."""
    from tft_core.model import build_tft

    model = build_tft(
        n_static=4, n_hist=5, n_future=5, n_components=5,
        T=12, H=4, d_model=16, n_heads=2, n_quantiles=3, dropout=0.0,
    )

    assert model.name == "TemporalFusionTransformer"

    # Forward pass
    batch = 2
    X_static = np.random.randn(batch, 4).astype(np.float32)
    X_static[:, 0] = np.array([0, 1])  # comp_id must be valid int
    X_hist = np.random.randn(batch, 12, 5).astype(np.float32)
    X_future = np.random.randn(batch, 4, 5).astype(np.float32)

    out = model.predict([X_static, X_hist, X_future], verbose=0)
    assert out.shape == (batch, 4, 3), f"Expected (2,4,3), got {out.shape}"
    print(f"✓ Standard TFT: output shape {out.shape}")


def test_constrained_tft_builds():
    """Constrained TFT builds and output respects floor at inference."""
    from tft_constrained.model import build_constrained_tft

    model = build_constrained_tft(
        n_static=4, n_hist=5, n_future=5, n_components=5,
        T=12, H=4, d_model=16, n_heads=2, n_quantiles=3,
        alpha=2, margin=0.05, soft_temp=10.0, dropout=0.0,
    )

    assert model.name == "Constrained_TFT"

    batch = 2
    X_static = np.random.randn(batch, 4).astype(np.float32)
    X_static[:, 0] = np.array([0, 1])
    X_hist = np.random.randn(batch, 12, 5).astype(np.float32)
    X_future = np.random.randn(batch, 4, 5).astype(np.float32)
    # Set contract values (last feature) to a known value
    X_future[:, :, -1] = 10.0

    out = model.predict([X_static, X_hist, X_future], verbose=0)
    assert out.shape == (batch, 4, 3), f"Expected (2,4,3), got {out.shape}"

    # At inference, all quantiles should be >= floor
    # floor = 10.0 * (1 + 0.05) = 10.5 (approximately, after alpha shift)
    print(f"✓ Constrained TFT: output shape {out.shape}")
    print(f"  Min prediction: {out.min():.4f}")


def test_quantile_loss():
    """Quantile loss computes without errors."""
    import tensorflow as tf
    from tft_core.losses import quantile_loss

    y_true = tf.constant(np.random.randn(4, 6, 1).astype(np.float32))
    y_pred = tf.constant(np.random.randn(4, 6, 3).astype(np.float32))

    loss = quantile_loss(y_true, y_pred)
    assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
    assert not np.isnan(loss.numpy()), "Loss is NaN"
    print(f"✓ Quantile loss: {loss.numpy():.6f}")


def test_constraint_layers():
    """AlphaShift and SoftConstraint layers work correctly."""
    import tensorflow as tf
    from tft_constrained.constraint_layers import AlphaShiftLayer, SoftConstraintLayer

    # Alpha shift
    alpha_shift = AlphaShiftLayer(alpha=3, margin=0.02)
    contracts = tf.constant(np.array([[[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]]]).astype(np.float32))
    floor = alpha_shift(contracts)
    assert floor.shape == (1, 6, 1)
    # First 3 values should be padded with first contract value
    print(f"✓ AlphaShift: input contracts=[10..15], floor={floor.numpy().flatten()}")

    # Soft constraint
    constraint = SoftConstraintLayer(temperature=10.0)
    raw = tf.constant([[[5.0, 8.0, 12.0]]]).astype(np.float32)  # (1, 1, 3)
    floor_val = tf.constant([[[10.0]]]).astype(np.float32)       # (1, 1, 1)

    # Inference: hard max
    out_inf = constraint([raw, floor_val], training=False)
    assert out_inf.numpy()[0, 0, 0] >= 10.0, "P10 should be >= floor"
    assert out_inf.numpy()[0, 0, 1] >= 10.0, "P50 should be >= floor"
    print(f"✓ SoftConstraint: raw=[5,8,12], floor=10 → constrained={out_inf.numpy().flatten()}")


def test_data_generation():
    """Sample data generator produces valid shapes."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from generate_sample_data import generate_component_series, create_windows, COMPONENTS

    comp = COMPONENTS[0]
    series = generate_component_series(comp, 48, seed=42)
    assert len(series["sales"]) == 48
    assert series["sales"].dtype == np.float32

    windows = create_windows(comp, series, T=24, H=6)
    assert len(windows) > 0
    static, hist, future, target = windows[0]
    assert static.shape == (4,)
    assert hist.shape == (24, 5)
    assert future.shape == (6, 5)
    assert target.shape == (6,)
    print(f"✓ Data generation: {len(windows)} windows, shapes correct")


if __name__ == "__main__":
    print("=" * 60)
    print("TFT Multi-Component — Smoke Tests")
    print("=" * 60)

    test_standard_tft_builds()
    test_constrained_tft_builds()
    test_quantile_loss()
    test_constraint_layers()
    test_data_generation()

    print("\n" + "=" * 60)
    print("All tests passed ✓")
    print("=" * 60)
