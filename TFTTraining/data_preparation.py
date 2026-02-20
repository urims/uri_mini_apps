"""
Data Preparation — From raw sales/contract tables to TFT-ready tensors.

Pipeline:
    1. Load sales_history + manufacturing_contracts tables
    2. Merge via merge_asof (attach active contract to each sale)
    3. Feature engineering (margin, calendar features)
    4. Encode categoricals (LabelEncoder)
    5. Create sliding windows per component
    6. Pool across all components into single dataset
    7. Normalize features (StandardScaler, fit on train only)
    8. Temporal train/val/test split (NO random shuffle)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TFTDataConfig:
    """Configuration for TFT data preparation."""

    T: int = 24                  # Lookback window
    H: int = 6                   # Forecast horizon
    train_frac: float = 0.70     # Training fraction
    val_frac: float = 0.15       # Validation fraction (test = 1 - train - val)

    static_features: List[str] = field(default_factory=lambda: [
        "comp_enc", "cat_enc", "supplier_enc", "wh_enc",
    ])
    historical_features: List[str] = field(default_factory=lambda: [
        "sales", "price", "contract_value", "margin", "lead_time",
    ])
    known_future_features: List[str] = field(default_factory=lambda: [
        "promo", "month", "quarter", "day_of_week", "contract_value",
    ])
    target: str = "sales"


class TFTDataPipeline:
    """
    End-to-end data preparation for multi-component TFT training.

    Usage:
        pipeline = TFTDataPipeline(config)
        pipeline.fit(sales_df, contracts_df, components_df)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = pipeline.get_splits()
    """

    def __init__(self, config: Optional[TFTDataConfig] = None):
        self.config = config or TFTDataConfig()
        self.cat_encoders: Dict[str, LabelEncoder] = {}
        self.scaler_hist: Optional[StandardScaler] = None
        self.scaler_target: Optional[StandardScaler] = None
        self._fitted = False

    def fit(
        self,
        sales_df: pd.DataFrame,
        contracts_df: pd.DataFrame,
        components_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Fit the pipeline: merge tables, engineer features, create windows.

        Args:
            sales_df: Sales history with columns:
                [date, comp, sales, price, promo, lead_time].
            contracts_df: Manufacturing contracts with columns:
                [comp, start, end, value, moq].
            components_df: Optional component metadata with columns:
                [comp, cat, supplier, wh].
        """
        cfg = self.config

        # ─── Merge sales + contracts ───
        sales_df = sales_df.copy()
        sales_df["date"] = pd.to_datetime(sales_df["date"])
        contracts_df = contracts_df.copy()
        contracts_df["start"] = pd.to_datetime(contracts_df["start"])

        merged = pd.merge_asof(
            sales_df.sort_values("date"),
            contracts_df[["comp", "start", "value"]].rename(
                columns={"value": "contract_value"}
            ).sort_values("start"),
            by="comp",
            left_on="date",
            right_on="start",
            direction="backward",
        )

        # ─── Join component metadata ───
        if components_df is not None:
            merged = merged.merge(components_df, on="comp", how="left")

        # ─── Feature engineering ───
        merged["margin"] = merged["price"] - merged["contract_value"]
        merged["margin_pct"] = merged["margin"] / merged["contract_value"].clip(1e-8)
        merged["month"] = merged["date"].dt.month
        merged["quarter"] = merged["date"].dt.quarter
        merged["day_of_week"] = merged["date"].dt.dayofweek

        # ─── Encode categoricals ───
        for col in ["comp", "cat", "supplier", "wh"]:
            if col in merged.columns:
                le = LabelEncoder()
                merged[col + "_enc"] = le.fit_transform(merged[col].astype(str))
                self.cat_encoders[col] = le

        # ─── Create sliding windows per component ───
        all_samples = []
        for comp_id in merged["comp_enc"].unique():
            comp_df = merged[merged["comp_enc"] == comp_id].sort_values("date")
            if len(comp_df) < cfg.T + cfg.H:
                continue

            for i in range(cfg.T, len(comp_df) - cfg.H + 1):
                static = comp_df.iloc[i][cfg.static_features].values.astype(np.float32)
                hist = comp_df.iloc[i - cfg.T : i][cfg.historical_features].values.astype(
                    np.float32
                )
                future = comp_df.iloc[i : i + cfg.H][
                    cfg.known_future_features
                ].values.astype(np.float32)
                target = comp_df.iloc[i : i + cfg.H][cfg.target].values.astype(
                    np.float32
                )
                all_samples.append((static, hist, future, target))

        if not all_samples:
            raise ValueError(
                f"No valid windows found. Need at least T+H={cfg.T + cfg.H} "
                "timesteps per component."
            )

        # ─── Unzip into arrays ───
        self.X_static = np.array([s[0] for s in all_samples])
        self.X_hist = np.array([s[1] for s in all_samples])
        self.X_future = np.array([s[2] for s in all_samples])
        self.Y = np.array([s[3] for s in all_samples])[..., np.newaxis]

        # ─── Normalize (fit on training portion only) ───
        n = len(self.X_static)
        train_end = int(n * cfg.train_frac)

        self.scaler_hist = StandardScaler()
        hist_flat = self.X_hist.reshape(-1, self.X_hist.shape[-1])
        self.scaler_hist.fit(hist_flat[:train_end * cfg.T])
        self.X_hist = self.scaler_hist.transform(hist_flat).reshape(self.X_hist.shape)

        self.scaler_target = StandardScaler()
        y_flat = self.Y.reshape(-1, 1)
        self.scaler_target.fit(y_flat[:train_end * cfg.H])
        self.Y = self.scaler_target.transform(y_flat).reshape(self.Y.shape)

        self._fitted = True

    def get_splits(
        self,
    ) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Return temporal train/val/test splits.

        CRITICAL: Split by time, not random, to prevent data leakage.

        Returns:
            X_train, Y_train, X_val, Y_val, X_test, Y_test
            where X_* = [X_static, X_hist, X_future].
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .get_splits()")

        cfg = self.config
        n = len(self.X_static)
        train_end = int(n * cfg.train_frac)
        val_end = int(n * (cfg.train_frac + cfg.val_frac))

        def _split(start, end):
            return (
                [self.X_static[start:end], self.X_hist[start:end], self.X_future[start:end]],
                self.Y[start:end],
            )

        X_train, Y_train = _split(0, train_end)
        X_val, Y_val = _split(train_end, val_end)
        X_test, Y_test = _split(val_end, n)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled predictions back to original scale."""
        if self.scaler_target is None:
            raise RuntimeError("Pipeline not fitted.")
        shape = y_scaled.shape
        return self.scaler_target.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).reshape(shape)
