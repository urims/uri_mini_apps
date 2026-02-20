"""
Generate synthetic multi-component sales data for TFT training.

Creates 5 electronic components with:
    - 48 months of sales history
    - Manufacturing contracts with periodic renewals
    - Seasonal patterns, promotions, and noise
    - Known future covariates (calendar + contracts + promos)

Output: .npy files in data/ directory ready for run_training.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


COMPONENTS = [
    {"id": 0, "cat": 0, "supplier": 0, "wh": 0, "base_sales": 12000, "base_price": 0.042, "base_contract": 0.031},
    {"id": 1, "cat": 0, "supplier": 1, "wh": 1, "base_sales": 8500, "base_price": 0.19, "base_contract": 0.14},
    {"id": 2, "cat": 1, "supplier": 2, "wh": 2, "base_sales": 3500, "base_price": 2.50, "base_contract": 1.90},
    {"id": 3, "cat": 2, "supplier": 3, "wh": 3, "base_sales": 5800, "base_price": 0.86, "base_contract": 0.62},
    {"id": 4, "cat": 3, "supplier": 4, "wh": 0, "base_sales": 15000, "base_price": 0.12, "base_contract": 0.08},
]

T = 24    # Lookback
H = 6     # Horizon
N_MONTHS = 48  # Total history length


def generate_component_series(comp: dict, n_months: int, seed: int):
    """Generate sales, price, contract, promo, lead_time for one component."""
    rng = np.random.RandomState(seed)

    months = np.arange(n_months)
    trend = months * 0.005 * comp["base_sales"]
    seasonal = comp["base_sales"] * 0.1 * np.sin(2 * np.pi * months / 12)
    noise = rng.normal(0, comp["base_sales"] * 0.08, n_months)
    promo = (rng.random(n_months) > 0.75).astype(float)
    promo_effect = promo * comp["base_sales"] * 0.15

    sales = comp["base_sales"] + trend + seasonal + noise + promo_effect
    sales = np.maximum(sales, 100)

    # Contracts step up every 6 months
    contracts = np.array([
        comp["base_contract"] * (1.0 + 0.05 * (m // 6))
        for m in range(n_months)
    ])
    contracts += rng.normal(0, comp["base_contract"] * 0.01, n_months)

    # Price follows contract with margin + noise
    margin = comp["base_price"] - comp["base_contract"]
    prices = contracts + margin + rng.normal(0, margin * 0.15, n_months)

    lead_time = np.full(n_months, 14.0) + rng.normal(0, 2, n_months)

    return {
        "sales": sales.astype(np.float32),
        "price": prices.astype(np.float32),
        "contract_value": contracts.astype(np.float32),
        "margin": (prices - contracts).astype(np.float32),
        "lead_time": lead_time.astype(np.float32),
        "promo": promo.astype(np.float32),
        "month": ((months % 12) + 1).astype(np.float32),
        "quarter": ((months // 3) % 4 + 1).astype(np.float32),
        "day_of_week": rng.randint(0, 7, n_months).astype(np.float32),
    }


def create_windows(comp: dict, series: dict, T: int, H: int):
    """Create sliding windows for one component."""
    n = len(series["sales"])
    static_feats = np.array([comp["id"], comp["cat"], comp["supplier"], comp["wh"]], dtype=np.float32)

    hist_keys = ["sales", "price", "contract_value", "margin", "lead_time"]
    future_keys = ["promo", "month", "quarter", "day_of_week", "contract_value"]

    samples = []
    for i in range(T, n - H + 1):
        hist = np.stack([series[k][i - T : i] for k in hist_keys], axis=-1)
        future = np.stack([series[k][i : i + H] for k in future_keys], axis=-1)
        target = series["sales"][i : i + H]
        samples.append((static_feats, hist, future, target))

    return samples


def main():
    os.makedirs("data", exist_ok=True)

    all_samples = []
    for ci, comp in enumerate(COMPONENTS):
        series = generate_component_series(comp, N_MONTHS, seed=ci * 42 + 7)
        windows = create_windows(comp, series, T, H)
        all_samples.extend(windows)
        print(f"Component {ci} ({comp['base_sales']:.0f} base): {len(windows)} windows")

    # Shuffle (but keep temporal ordering within each component)
    rng = np.random.RandomState(123)
    rng.shuffle(all_samples)

    X_static = np.array([s[0] for s in all_samples])
    X_hist = np.array([s[1] for s in all_samples])
    X_future = np.array([s[2] for s in all_samples])
    Y = np.array([s[3] for s in all_samples])[..., np.newaxis]

    np.save("data/X_static.npy", X_static)
    np.save("data/X_hist.npy", X_hist)
    np.save("data/X_future.npy", X_future)
    np.save("data/Y.npy", Y)

    print(f"\nTotal samples: {len(all_samples)}")
    print(f"  X_static:  {X_static.shape}")
    print(f"  X_hist:    {X_hist.shape}")
    print(f"  X_future:  {X_future.shape}")
    print(f"  Y:         {Y.shape}")
    print(f"\nSaved to data/")


if __name__ == "__main__":
    main()
