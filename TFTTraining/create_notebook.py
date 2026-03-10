#!/usr/bin/env python3
"""
Generates: nixtla_tft_expertfloor_sagemaker.ipynb
Run with: python create_notebook.py
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

cells = []

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("""\
# Nixtla TFT · ExpertFloorLoss · AWS SageMaker
### Global Multi-Series Forecasting with Contract Floor Constraints

**Architecture overview**
```
Raw CSV (20K+ ts_ids)
  │
  ▼
DataLoader  ──►  ContractTypeClassifier  ──►  FeatureEngineer
                                                   │
                                                   ▼
                                         SyntheticDataGenerator
                                                   │
                                                   ▼
                                         NixtlaDataPreparer
                                         (unique_id / ds / y + exog)
                                                   │
                            ┌──────────────────────┘
                            ▼
                     CustomTFT (NeuralForecast)
                       │  loss: NixtlaFloorLoss
                       │    └─ ExpertFloorLoss
                       │         (Type 1/2/3 contracts)
                       │
                       ▼
              FloorConstrainedForecaster
                       │
                       ▼
               ModelEvaluator  +  HyperparamTuner
                       │
                       ▼
            SageMakerDeployer (S3 + Training Job)
```

**Contract floor logic**
| Type | Formula | Description |
|------|---------|-------------|
| 1 | `v_initial × (1.054)^años` | Standard 5.4 % annual capitalisation |
| 2 | `v_initial` | Fixed price |
| 3 | `v_initial × (1 + tasa_custom)^años` | Custom capitalisation rate |
"""))

# ─────────────────────────────────────────────
# CELL 0: Environment Setup
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("## Part 0 · Environment Setup"))

cells.append(new_code_cell("""\
import sys, os, subprocess

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# Core forecasting stack
_pip(
    "neuralforecast>=1.7.4",
    "utilsforecast>=0.1.12",
    "datasetsforecast>=0.0.8",
)
# Data & viz
_pip("pandas>=2.0", "numpy>=1.24", "matplotlib>=3.7", "seaborn>=0.12")
# Tuning
_pip("optuna>=3.5", "ray[tune]>=2.9")
# AWS
_pip("boto3>=1.34", "sagemaker>=2.200")

print("✓ All packages installed")
"""))

cells.append(new_code_cell("""\
import os, warnings, threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import MAE, MSE, BasePointLoss

warnings.filterwarnings("ignore")

# ── Environment detection ──────────────────────────────────────────────────────
IS_SAGEMAKER = os.path.exists("/opt/ml/input")
IS_GPU       = torch.cuda.is_available()
DEVICE       = "gpu" if IS_GPU else "cpu"
N_GPUS       = torch.cuda.device_count() if IS_GPU else 0

print(f"Running on: {'SageMaker' if IS_SAGEMAKER else 'Local'}")
print(f"GPU available: {IS_GPU}  |  GPU count: {N_GPUS}")
print(f"PyTorch: {torch.__version__}")

# ── SageMaker paths (fall back to local paths when running locally) ────────────
SM_INPUT   = os.environ.get("SM_CHANNEL_TRAIN", str(Path.home() / "data"))
SM_OUTPUT  = os.environ.get("SM_MODEL_DIR",     "output/model")
SM_RESULTS = os.environ.get("SM_OUTPUT_DATA_DIR", "output/results")

os.makedirs(SM_OUTPUT,  exist_ok=True)
os.makedirs(SM_RESULTS, exist_ok=True)

SAMPLE_CSV = Path(__file__).parent / "timeseries_sample.csv" \\
    if "__file__" in dir() else Path("timeseries_sample.csv")
"""))

# ─────────────────────────────────────────────
# CELL 1: Project Configuration
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("## Part 1 · Project Configuration\n\n> **Edit this cell** to match your data and compute budget."))

cells.append(new_code_cell("""\
@dataclass
class ProjectConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_path: str          = str(SAMPLE_CSV)   # local CSV or s3:// URI
    target_col: str         = "actual_cost_paid"
    ts_id_col: str          = "ts_id"
    date_col: str           = "date"            # YYYYMM integer format
    contract_value_col: str = "contract_value"

    # ── Forecast ──────────────────────────────────────────────────────────────
    horizon: int            = 12                # months ahead
    input_size: int         = 24                # lookback window
    val_months: int         = 12                # months held-out for validation
    test_months: int        = 12                # months held-out for test

    # ── Synthetic data ────────────────────────────────────────────────────────
    add_synthetic: bool     = True
    n_synthetic_per_type: int = 20              # per contract type (1/2/3)

    # ── Feature lists ─────────────────────────────────────────────────────────
    stat_exog_list: List[str] = field(default_factory=lambda: [
        "tipo", "v_initial", "tasa_custom",
    ])
    futr_exog_list: List[str] = field(default_factory=lambda: [
        "años", "floor_value", "month", "quarter",
    ])
    hist_exog_list: List[str] = field(default_factory=lambda: [
        "lag_1", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_std_3", "rolling_mean_12",
    ])

    # ── Model ─────────────────────────────────────────────────────────────────
    hidden_size: int        = 128
    n_head: int             = 4
    dropout: float          = 0.1
    attn_dropout: float     = 0.05
    max_steps: int          = 1000
    batch_size: int         = 32
    windows_batch_size: int = 1024
    learning_rate: float    = 1e-3
    val_check_steps: int    = 50
    early_stop_patience: int = 10
    scaler_type: str        = "robust"

    # ── Custom loss ───────────────────────────────────────────────────────────
    floor_loss_weight: float = 0.5             # weight of floor penalty vs base loss
    window_size: int         = 12              # rolling window for violation count

    # ── SageMaker ─────────────────────────────────────────────────────────────
    instance_type: str      = "ml.g4dn.xlarge"  # 1× T4 (16 GB VRAM)
    s3_bucket: str          = "your-sagemaker-bucket"
    s3_prefix: str          = "tft-expertfloor"
    framework_version: str  = "2.1"            # PyTorch version on SageMaker
    py_version: str         = "py310"


CFG = ProjectConfig()
print("ProjectConfig loaded:")
for k, v in CFG.__dict__.items():
    print(f"  {k:30s} = {v}")
"""))

# ─────────────────────────────────────────────
# CELL 2: Data Loading & EDA
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("## Part 2 · Data Loading & EDA"))

cells.append(new_code_cell("""\
class DataLoader:
    \"\"\"
    Loads and validates the raw CSV/S3 file into NeuralForecast format.

    Expected input columns:
        date            : int YYYYMM
        ts_id           : str  unique series identifier
        actual_cost_paid: float target variable
        contract_value  : float optional contract reference (may have NaN)

    Output columns:
        ds, unique_id, y, contract_value
    \"\"\"

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    # ── Load ──────────────────────────────────────────────────────────────────
    def load(self, path: Optional[str] = None) -> pd.DataFrame:
        path = path or self.cfg.data_path

        if str(path).startswith("s3://"):
            import boto3, io
            s3 = boto3.client("s3")
            bucket, key = str(path)[5:].split("/", 1)
            obj = s3.get_object(Bucket=bucket, Key=key)
            raw = pd.read_csv(io.BytesIO(obj["Body"].read()))
        else:
            raw = pd.read_csv(path)

        return self._transform(raw)

    def _transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns={
            self.cfg.ts_id_col:          "unique_id",
            self.cfg.target_col:         "y",
            self.cfg.contract_value_col: "contract_value",
        })

        # Parse YYYYMM → datetime (period start)
        df["ds"] = pd.to_datetime(df[self.cfg.date_col].astype(str), format="%Y%m")
        df = df.drop(columns=[self.cfg.date_col], errors="ignore")

        df["contract_value"] = pd.to_numeric(df["contract_value"], errors="coerce")
        df["y"]              = pd.to_numeric(df["y"],              errors="coerce")

        df = df.dropna(subset=["y"])
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        return df

    # ── EDA ───────────────────────────────────────────────────────────────────
    def eda_report(self, df: pd.DataFrame) -> None:
        n_ts        = df["unique_id"].nunique()
        ts_lengths  = df.groupby("unique_id").size()
        cv_missing  = df["contract_value"].isna().mean() * 100

        print("=" * 60)
        print("  DATASET OVERVIEW")
        print("=" * 60)
        print(f"  Unique series   : {n_ts:,}")
        print(f"  Total rows      : {len(df):,}")
        print(f"  Date range      : {df['ds'].min():%Y-%m} → {df['ds'].max():%Y-%m}")
        print(f"  Avg ts length   : {ts_lengths.mean():.1f} months")
        print(f"  Missing cv      : {cv_missing:.1f}%")
        print()

        print("  TARGET (y) per series")
        print(df.groupby("unique_id")["y"]
              .agg(["min", "mean", "max", "std"])
              .round(2).to_string())
        print()

        # Quick plot
        fig, axes = plt.subplots(
            n_ts, 1, figsize=(14, 3 * n_ts), sharex=False, squeeze=False
        )
        for ax, (uid, grp) in zip(axes.flatten(), df.groupby("unique_id")):
            grp = grp.sort_values("ds")
            ax.plot(grp["ds"], grp["y"], label="actual", lw=1.5)
            if grp["contract_value"].notna().any():
                ax.plot(grp["ds"], grp["contract_value"],
                        "--", alpha=0.6, label="contract")
            ax.set_title(uid, fontsize=9)
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{SM_RESULTS}/eda_series.png", dpi=100, bbox_inches="tight")
        plt.show()
        print(f"\\nEDA plot saved → {SM_RESULTS}/eda_series.png")
"""))

cells.append(new_code_cell("""\
loader = DataLoader(CFG)
raw_df = loader.load()
loader.eda_report(raw_df)
raw_df.head()
"""))

# ─────────────────────────────────────────────
# CELL 3: Contract Classification & Feature Engineering
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Part 3 · Feature Engineering

### 3.1 Contract Type Classifier

Assigns each `ts_id` to one of three contract types:

| Type | Signal | Rule |
|------|--------|------|
| 2 (Fixed) | CV < 0.5 %, |CAGR| < 1 % | Virtually constant |
| 1 (Std 5.4 %) | CAGR within ±2 pp of 5.4 % | Matches standard capitalisation |
| 3 (Custom rate) | Everything else | Rate detected from CAGR |
"""))

cells.append(new_code_cell("""\
class ContractTypeClassifier:
    \"\"\"
    Classifies each unique_id into a contract type and estimates parameters.
    \"\"\"
    RATE_T1       = 0.054     # Standard capitalisation rate
    FIXED_CV_THR  = 0.005     # CV below this → fixed price
    T1_BAND       = 0.020     # ±2 pp tolerance for Type-1 detection

    def classify_series(
        self,
        y: pd.Series,
        cv: pd.Series,
        ds: pd.Series,
    ) -> dict:
        \"\"\"Classify a single series. Returns tipo, v_initial, tasa_custom.\"\"\"

        y    = y.sort_index()
        ds   = ds.sort_index()
        cv_s = cv.sort_index()

        # v_initial: first non-NaN contract_value, else first y
        v_initial = float(cv_s.dropna().iloc[0]) if cv_s.notna().any() else float(y.iloc[0])
        v_initial = max(v_initial, 1e-9)

        # CAGR over the full available period
        v_final = float(y.iloc[-1])
        n_years = max((ds.iloc[-1] - ds.iloc[0]).days / 365.25, 1e-3)
        cagr    = (v_final / v_initial) ** (1.0 / n_years) - 1.0

        # Coefficient of variation
        coeff_v = y.std() / y.mean() if y.mean() != 0 else 0.0

        # Classification
        if coeff_v < self.FIXED_CV_THR and abs(cagr) < 0.01:
            tipo        = 2
            tasa_custom = 0.0
        elif abs(cagr - self.RATE_T1) <= self.T1_BAND:
            tipo        = 1
            tasa_custom = 0.0
        else:
            tipo        = 3
            tasa_custom = max(round(cagr, 4), 0.0)

        return {
            "tipo":        tipo,
            "v_initial":   v_initial,
            "tasa_custom": tasa_custom,
            "detected_cagr": round(cagr, 4),
            "detected_cv":   round(coeff_v, 4),
        }

    def classify_all(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Return a DataFrame with one row per unique_id.\"\"\"
        rows = []
        for uid, grp in df.groupby("unique_id"):
            grp = grp.set_index("ds").sort_index()
            info = self.classify_series(
                y=grp["y"],
                cv=grp.get("contract_value", pd.Series(dtype=float)),
                ds=grp.reset_index()["ds"],
            )
            info["unique_id"] = uid
            rows.append(info)
        return pd.DataFrame(rows).set_index("unique_id")
"""))

cells.append(new_markdown_cell("### 3.2 Feature Engineer"))

cells.append(new_code_cell("""\
class FeatureEngineer:
    \"\"\"
    Adds all features required by ExpertFloorLoss + NeuralForecast:

    Static (per series, time-invariant):
        tipo, v_initial, tasa_custom

    Future-known (known for entire horizon):
        años, floor_value, month, quarter

    Historical (observed only up to current time):
        lag_1, lag_3, lag_6, lag_12
        rolling_mean_3, rolling_std_3, rolling_mean_12
    \"\"\"

    LAG_WINDOWS     = [1, 3, 6, 12]
    ROLLING_WINDOWS = [3, 6, 12]

    def __init__(self, contract_info: pd.DataFrame):
        # contract_info has index=unique_id, cols: tipo, v_initial, tasa_custom
        self.contract_info = contract_info

    # ── Static features ───────────────────────────────────────────────────────
    def _add_static(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["tipo", "v_initial", "tasa_custom"]:
            df[col] = df["unique_id"].map(self.contract_info[col])
        return df

    # ── Time features ────────────────────────────────────────────────────────
    def _add_time(self, df: pd.DataFrame) -> pd.DataFrame:
        start = df.groupby("unique_id")["ds"].transform("min")
        df["años"]   = (df["ds"] - start).dt.days / 365.25
        df["month"]  = df["ds"].dt.month.astype(float)
        df["quarter"] = df["ds"].dt.quarter.astype(float)
        return df

    # ── Floor value ──────────────────────────────────────────────────────────
    def _add_floor(self, df: pd.DataFrame) -> pd.DataFrame:
        vi    = df["v_initial"]
        t     = df["tipo"]
        tc    = df["tasa_custom"]
        años  = df["años"]

        floor = torch.zeros(len(df))
        df["floor_value"] = np.where(
            t == 1, vi * (1 + 0.054) ** años,
            np.where(t == 2, vi,
                     vi * (1 + tc) ** años)
        )
        # Fill any edge-case NaN
        df["floor_value"] = df["floor_value"].fillna(df["v_initial"])
        return df

    # ── Lag features ─────────────────────────────────────────────────────────
    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["unique_id", "ds"])
        grp = df.groupby("unique_id")["y"]
        for lag in self.LAG_WINDOWS:
            df[f"lag_{lag}"] = grp.shift(lag)
        return df

    # ── Rolling features ─────────────────────────────────────────────────────
    def _add_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["unique_id", "ds"])
        for w in self.ROLLING_WINDOWS:
            shifted = df.groupby("unique_id")["y"].shift(1)
            df[f"rolling_mean_{w}"] = shifted.groupby(
                df["unique_id"]).transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f"rolling_std_{w}"]  = shifted.groupby(
                df["unique_id"]).transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        return df

    # ── Fill NaN lags (first rows of each series) ─────────────────────────────
    def _fill_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        lag_cols = [f"lag_{l}" for l in self.LAG_WINDOWS] + \\
                   [f"rolling_mean_{w}" for w in self.ROLLING_WINDOWS] + \\
                   [f"rolling_std_{w}"  for w in self.ROLLING_WINDOWS]
        for col in lag_cols:
            if col in df.columns:
                df[col] = df.groupby("unique_id")[col].transform(
                    lambda x: x.fillna(x.median())
                )
        return df

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_static(df)
        df = self._add_time(df)
        df = self._add_floor(df)
        df = self._add_lags(df)
        df = self._add_rolling(df)
        df = self._fill_nan(df)
        return df
"""))

# ─────────────────────────────────────────────
# CELL 4: Synthetic Data Generation
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("### 3.3 Synthetic Data Generator\n\nCreates labelled examples for each contract type to augment training."))

cells.append(new_code_cell("""\
class SyntheticDataGenerator:
    \"\"\"
    Generates synthetic time series following the three contract type rules.
    Useful for testing and for augmenting real data during early training.

    Scale guidelines (for 20 K+ real series):
        n_per_type = 200–500   → moderate augmentation
        n_per_type = 1000+     → heavy augmentation for cold-start
    \"\"\"

    RATE_T1 = 0.054

    def __init__(
        self,
        n_per_type: int  = 20,
        start_date: str  = "2020-01",
        n_months: int    = 72,
        noise_pct: float = 0.02,       # Gaussian noise as % of floor value
        seed: int        = 42,
    ):
        self.n     = n_per_type
        self.start = start_date
        self.T     = n_months
        self.noise = noise_pct
        self.rng   = np.random.default_rng(seed)

    # ── Internal builders ────────────────────────────────────────────────────
    def _dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, periods=self.T, freq="MS")

    def _noisy(self, floor: np.ndarray) -> np.ndarray:
        return floor + self.rng.normal(0, floor * self.noise)

    def _make_type1(self, uid: str, v0: float) -> pd.DataFrame:
        dates = self._dates()
        años  = np.arange(self.T) / 12
        floor = v0 * (1 + self.RATE_T1) ** años
        return pd.DataFrame({
            "ds": dates, "unique_id": uid,
            "y": np.maximum(self._noisy(floor), 0),
            "contract_value": floor,
            "tipo": 1, "v_initial": v0, "tasa_custom": 0.0,
        })

    def _make_type2(self, uid: str, v0: float) -> pd.DataFrame:
        dates = self._dates()
        floor = np.full(self.T, v0)
        noise = self.rng.normal(0, v0 * self.noise * 0.5, self.T)
        return pd.DataFrame({
            "ds": dates, "unique_id": uid,
            "y": np.maximum(floor + noise, 0),
            "contract_value": floor,
            "tipo": 2, "v_initial": v0, "tasa_custom": 0.0,
        })

    def _make_type3(self, uid: str, v0: float, rate: float) -> pd.DataFrame:
        dates = self._dates()
        años  = np.arange(self.T) / 12
        floor = v0 * (1 + rate) ** años
        return pd.DataFrame({
            "ds": dates, "unique_id": uid,
            "y": np.maximum(self._noisy(floor), 0),
            "contract_value": floor,
            "tipo": 3, "v_initial": v0, "tasa_custom": rate,
        })

    # ── Public API ────────────────────────────────────────────────────────────
    def generate(self) -> pd.DataFrame:
        dfs = []

        # Type 1 – standard 5.4 %
        for i in range(self.n):
            v0 = float(self.rng.uniform(500, 150_000))
            dfs.append(self._make_type1(f"SYN_T1_{i:05d}", v0))

        # Type 2 – fixed
        for i in range(self.n):
            v0 = float(self.rng.uniform(100, 200_000))
            dfs.append(self._make_type2(f"SYN_T2_{i:05d}", v0))

        # Type 3 – custom rates spanning 2 % – 12 %
        custom_rates = np.linspace(0.02, 0.12, self.n)
        for i, rate in enumerate(custom_rates):
            v0 = float(self.rng.uniform(500, 100_000))
            dfs.append(self._make_type3(f"SYN_T3_{i:05d}", v0, round(float(rate), 4)))

        df = pd.concat(dfs, ignore_index=True)
        print(f"Synthetic data: {df['unique_id'].nunique()} series, "
              f"{len(df):,} rows, types={df['tipo'].value_counts().to_dict()}")
        return df
"""))

# ─────────────────────────────────────────────
# CELL 5: Run Full Data Pipeline
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("### 3.4 Run the Pipeline"))

cells.append(new_code_cell("""\
# ── Step 1: Classify real series ─────────────────────────────────────────────
classifier   = ContractTypeClassifier()
contract_info = classifier.classify_all(raw_df)
print("Contract type distribution:")
print(contract_info["tipo"].value_counts())
print()
print(contract_info[["tipo", "v_initial", "tasa_custom", "detected_cagr"]].head(10))

# ── Step 2: Synthetic data ────────────────────────────────────────────────────
if CFG.add_synthetic:
    syn_gen  = SyntheticDataGenerator(
        n_per_type=CFG.n_synthetic_per_type,
        n_months=raw_df.groupby("unique_id").size().max(),
        start_date=raw_df["ds"].min().strftime("%Y-%m"),
    )
    syn_df = syn_gen.generate()

    # Synthetic series already have tipo/v_initial/tasa_custom columns
    syn_contract_info = (
        syn_df[["unique_id", "tipo", "v_initial", "tasa_custom"]]
        .drop_duplicates("unique_id")
        .set_index("unique_id")
    )
    syn_df = syn_df.drop(columns=["tipo", "v_initial", "tasa_custom"])
    contract_info = pd.concat([contract_info, syn_contract_info])

    full_df = pd.concat([raw_df, syn_df], ignore_index=True)
else:
    full_df = raw_df.copy()

# ── Step 3: Feature engineering ───────────────────────────────────────────────
feat_eng = FeatureEngineer(contract_info)
feat_df  = feat_eng.transform(full_df)

print(f"\\nFeature-engineered dataframe: {feat_df.shape}")
print(f"Columns: {feat_df.columns.tolist()}")
feat_df.head(3)
"""))

# ─────────────────────────────────────────────
# CELL 6: Nixtla Data Preparation
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("## Part 4 · NeuralForecast Data Preparation"))

cells.append(new_code_cell("""\
class NixtlaDataPreparer:
    \"\"\"
    Converts the feature-engineered DataFrame into the format expected by
    NeuralForecast (unique_id / ds / y + exogenous columns).

    Train / Val / Test splits use a temporal cutoff strategy:
        test  = last `test_months` months
        val   = preceding `val_months` months
        train = everything before val
    \"\"\"

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    # ── Temporal split ────────────────────────────────────────────────────────
    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        max_date   = df["ds"].max()
        test_cut   = max_date - pd.DateOffset(months=self.cfg.test_months - 1)
        val_cut    = test_cut - pd.DateOffset(months=self.cfg.val_months)

        train = df[df["ds"] <  val_cut].copy()
        val   = df[df["ds"] <  test_cut].copy()
        test  = df.copy()

        print(f"Train: {train['ds'].max():%Y-%m} "
              f"| Val cutoff: {val_cut:%Y-%m} "
              f"| Test cutoff: {test_cut:%Y-%m}")
        print(f"Train rows: {len(train):,}  "
              f"| Val rows: {len(val):,}  "
              f"| Test rows: {len(test):,}")
        return train, val, test

    # ── Required columns check ────────────────────────────────────────────────
    def validate(self, df: pd.DataFrame) -> None:
        required = (
            ["unique_id", "ds", "y"]
            + self.cfg.stat_exog_list
            + self.cfg.futr_exog_list
            + self.cfg.hist_exog_list
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        print(f"✓ All {len(required)} required columns present")

    # ── Future exogenous for inference ────────────────────────────────────────
    def build_futr_df(
        self, df: pd.DataFrame, h: Optional[int] = None
    ) -> pd.DataFrame:
        \"\"\"
        Build the future exogenous DataFrame needed for forecasting.
        For each series, extend `futr_exog_list` features h steps beyond
        the last known date.

        Note: For production with 20 K+ series this should be pre-computed
        from your contract management system (años, floor_value are known).
        \"\"\"
        h = h or self.cfg.horizon
        rows = []
        for uid, grp in df.groupby("unique_id"):
            grp       = grp.sort_values("ds")
            last_ds   = grp["ds"].max()
            v_initial = float(grp["v_initial"].iloc[0])
            tipo      = int(grp["tipo"].iloc[0])
            tasa      = float(grp["tasa_custom"].iloc[0])
            start_ds  = grp["ds"].min()

            future_dates = pd.date_range(
                last_ds + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            for fds in future_dates:
                años  = (fds - start_ds).days / 365.25
                if tipo == 1:
                    floor = v_initial * (1 + 0.054) ** años
                elif tipo == 2:
                    floor = v_initial
                else:
                    floor = v_initial * (1 + tasa) ** años

                rows.append({
                    "unique_id":   uid,
                    "ds":          fds,
                    "años":        round(años, 4),
                    "floor_value": round(floor, 6),
                    "month":       float(fds.month),
                    "quarter":     float(fds.quarter),
                })

        return pd.DataFrame(rows)

    # ── Scale features ────────────────────────────────────────────────────────
    def fit_scalers(self, train_df: pd.DataFrame) -> Dict:
        \"\"\"
        Fit min-max or standard scalers on train_df for stat + hist features.
        NeuralForecast handles y scaling internally (scaler_type='robust').
        Returns a dict of {col: (min, max)} for inverse transform if needed.
        \"\"\"
        scalers = {}
        scale_cols = self.cfg.hist_exog_list + self.cfg.futr_exog_list
        for col in scale_cols:
            if col in train_df.columns:
                c_min, c_max = train_df[col].min(), train_df[col].max()
                scalers[col] = (c_min, c_max)
        return scalers

    def apply_scalers(
        self, df: pd.DataFrame, scalers: Dict, eps: float = 1e-8
    ) -> pd.DataFrame:
        df = df.copy()
        for col, (c_min, c_max) in scalers.items():
            if col in df.columns:
                df[col] = (df[col] - c_min) / (c_max - c_min + eps)
        return df
"""))

cells.append(new_code_cell("""\
preparer = NixtlaDataPreparer(CFG)
preparer.validate(feat_df)

train_df, val_df, test_df = preparer.split(feat_df)

# Scalers fitted on train only
scalers  = preparer.fit_scalers(train_df)
train_sc = preparer.apply_scalers(train_df, scalers)
val_sc   = preparer.apply_scalers(val_df,   scalers)
test_sc  = preparer.apply_scalers(test_df,  scalers)

# Future exogenous for inference
futr_df  = preparer.build_futr_df(test_df, h=CFG.horizon)

print(f"\\nfutr_df shape: {futr_df.shape}")
futr_df.head(3)
"""))

# ─────────────────────────────────────────────
# CELL 7: Custom Loss & Model
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("## Part 5 · Custom Loss & Model Architecture"))

cells.append(new_markdown_cell("### 5.1 ExpertFloorLoss (refactored)"))

cells.append(new_code_cell("""\
class ExpertFloorLoss(nn.Module):
    \"\"\"
    Business-constrained loss that penalises predictions below the
    contract floor with exponentially increasing severity the longer
    violations persist in a rolling window.

    Args:
        base_loss   : Any nn.Module loss (default MSELoss).
        window_size : Rolling window for violation count (months).
        sigmoid_k   : Sharpness of soft floor indicator.
        exp_div     : Controls how fast penalty grows with violation count.
    \"\"\"

    def __init__(
        self,
        base_loss: nn.Module = nn.MSELoss(),
        window_size: int     = 12,
        sigmoid_k: float     = 20.0,
        exp_div: float       = 4.0,
    ):
        super().__init__()
        self.base_loss   = base_loss
        self.window_size = window_size
        self.sigmoid_k   = sigmoid_k
        self.exp_div     = exp_div

    # ── Contract floor ────────────────────────────────────────────────────────
    def calculate_contract_floor(self, x_features: torch.Tensor) -> torch.Tensor:
        \"\"\"
        x_features : [..., 4]
            [0] v_initial   – Initial contract value
            [1] tipo        – Contract type (1, 2, 3)
            [2] tasa_custom – Custom rate for Type 3
            [3] años        – Time elapsed in years

        Returns floor tensor of same leading shape as x_features[..., 0].
        \"\"\"
        v_initial   = x_features[..., 0]
        tipo        = x_features[..., 1]
        tasa_custom = x_features[..., 2]
        años        = x_features[..., 3]

        piso_t1 = v_initial * (1.0 + 0.054) ** años
        piso_t2 = v_initial
        piso_t3 = v_initial * (1.0 + tasa_custom) ** años

        return torch.where(
            tipo == 1, piso_t1,
            torch.where(tipo == 2, piso_t2, piso_t3),
        )

    # ── Isolated penalty (for NeuralForecast integration) ─────────────────────
    def compute_floor_penalty(
        self,
        y_pred: torch.Tensor,
        dynamic_floor: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"
        y_pred        : [B, H]
        dynamic_floor : [B, H]  (pre-computed with calculate_contract_floor)
        \"\"\"
        is_below = torch.sigmoid((dynamic_floor - y_pred) * self.sigmoid_k)

        violation_count = torch.cumsum(is_below, dim=-1)
        if violation_count.shape[-1] > self.window_size:
            shift = torch.cat([
                torch.zeros_like(violation_count[..., :self.window_size]),
                violation_count[..., :-self.window_size],
            ], dim=-1)
            violation_count = violation_count - shift

        persistence_factor = torch.exp(violation_count / self.exp_div)
        floor_gap          = torch.clamp(dynamic_floor - y_pred, min=0) ** 2
        return torch.mean(persistence_factor * floor_gap)

    # ── Full loss (for standalone use / testing) ───────────────────────────────
    def forward(
        self,
        y_pred: torch.Tensor,
        y_actual: torch.Tensor,
        x_features: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"
        y_pred, y_actual : [B, H]
        x_features       : [B, H, 4]  or  [B, 4] (static, will be broadcast)
        \"\"\"
        if x_features.dim() == 2:            # [B, 4] → [B, H, 4]
            x_features = x_features.unsqueeze(1).expand(
                -1, y_pred.shape[-1], -1
            )

        standard_loss = self.base_loss(y_pred, y_actual)
        dynamic_floor = self.calculate_contract_floor(x_features)
        penalty       = self.compute_floor_penalty(y_pred, dynamic_floor)
        return standard_loss + penalty
"""))

cells.append(new_markdown_cell("### 5.2 NixtlaFloorLoss — NeuralForecast-compatible wrapper"))

cells.append(new_code_cell("""\
class NixtlaFloorLoss(MAE):
    \"\"\"
    NeuralForecast-compatible loss that wraps ExpertFloorLoss.

    It inherits from MAE so NeuralForecast accepts it as a standard loss.
    Before each training step the CustomTFT calls `set_contract_context()`
    to inject the current batch's contract features.
    \"\"\"

    def __init__(
        self,
        floor_loss: ExpertFloorLoss,
        floor_weight: float = 0.5,
        **mae_kwargs,
    ):
        super().__init__(**mae_kwargs)
        self.floor_loss   = floor_loss
        self.floor_weight = floor_weight
        self._ctx: Optional[dict] = None   # set by CustomTFT per step
        self._lock = threading.Lock()      # thread-safe for DataParallel

    def set_contract_context(
        self,
        v_initial:   torch.Tensor,   # [B]
        tipo:        torch.Tensor,   # [B]
        tasa_custom: torch.Tensor,   # [B]
        años:        torch.Tensor,   # [B, H]
    ) -> None:
        with self._lock:
            self._ctx = {
                "v_initial":   v_initial,
                "tipo":        tipo,
                "tasa_custom": tasa_custom,
                "años":        años,
            }

    def clear_context(self) -> None:
        with self._lock:
            self._ctx = None

    def __call__(
        self,
        y:    torch.Tensor,
        y_hat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_loss = super().__call__(y=y, y_hat=y_hat, mask=mask)

        if self._ctx is None:
            return base_loss

        try:
            ctx = self._ctx
            B, H = y_hat.squeeze(-1).shape

            # Expand static to [B, H]
            v0   = ctx["v_initial"].unsqueeze(1).expand(B, H)
            tipo = ctx["tipo"].unsqueeze(1).expand(B, H)
            tc   = ctx["tasa_custom"].unsqueeze(1).expand(B, H)
            años = ctx["años"]  # [B, H]

            x_feat = torch.stack([v0, tipo, tc, años], dim=-1)  # [B, H, 4]
            floor  = self.floor_loss.calculate_contract_floor(x_feat)

            penalty = self.floor_loss.compute_floor_penalty(
                y_pred=y_hat.squeeze(-1),
                dynamic_floor=floor,
            )
            return base_loss + self.floor_weight * penalty

        except Exception as e:
            # Graceful degradation: log once then fall back to MAE
            if not getattr(self, "_penalty_warn_shown", False):
                print(f"[NixtlaFloorLoss] Penalty skipped this step: {e}")
                self._penalty_warn_shown = True
            return base_loss
"""))

cells.append(new_markdown_cell("### 5.3 CustomTFT — overrides `training_step` for feature injection"))

cells.append(new_code_cell("""\
class CustomTFT(TFT):
    \"\"\"
    Subclass of NeuralForecast's TFT that injects contract features into the
    NixtlaFloorLoss before each training step.

    The feature injection uses column-name lookup so it is robust to
    different orderings of stat_exog_list / futr_exog_list.
    \"\"\"

    CONTRACT_STAT_KEYS = ["v_initial", "tipo", "tasa_custom"]
    CONTRACT_FUTR_KEY  = "años"

    def __init__(self, floor_loss: Optional[NixtlaFloorLoss] = None, **kwargs):
        super().__init__(**kwargs)
        self._floor_loss  = floor_loss
        self._stat_cols   = list(kwargs.get("stat_exog_list", []))
        self._futr_cols   = list(kwargs.get("futr_exog_list", []))
        self._inject_warn = False

    # ── Feature extraction ────────────────────────────────────────────────────
    def _try_inject(self, windows: dict) -> None:
        \"\"\"Extract contract tensors from the training windows dict and
        push them to the loss context.\"\"\"
        if self._floor_loss is None:
            return

        ctx = {}

        # Static features: batch['static'] shape [B, n_stat]
        stat = windows.get("static")   # may be None
        stat_cols = windows.get("static_cols", self._stat_cols)
        if stat is not None:
            for key in self.CONTRACT_STAT_KEYS:
                if key in stat_cols:
                    idx = list(stat_cols).index(key)
                    ctx[key] = stat[:, idx].float()

        # Future exogenous: windows['futr_exog'] shape [B, H, n_futr]
        futr = windows.get("futr_exog")
        futr_cols = windows.get("futr_exog_cols", self._futr_cols)
        if futr is not None and self.CONTRACT_FUTR_KEY in futr_cols:
            idx = list(futr_cols).index(self.CONTRACT_FUTR_KEY)
            ctx["años"] = futr[:, :, idx].float()

        if len(ctx) == 4:   # all four features present
            self._floor_loss.set_contract_context(**ctx)
        elif not self._inject_warn:
            self._inject_warn = True
            print("[CustomTFT] Contract context incomplete – "
                  "falling back to base MAE loss.")

    # ── Override training step ────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        try:
            # NeuralForecast ≥ 1.7 exposes _create_windows or processes
            # via the forward pass; the batch dict already carries
            # 'static', 'futr_exog', etc. when using the DataFrame API.
            self._try_inject(batch)
        except Exception:
            pass  # never let injection errors break training

        loss = super().training_step(batch, batch_idx)

        # Clear context after each step (thread safety in DataParallel)
        if self._floor_loss is not None:
            self._floor_loss.clear_context()

        return loss
"""))

# ─────────────────────────────────────────────
# CELL 8: Training
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("## Part 6 · Training"))

cells.append(new_code_cell("""\
class ModelBuilder:
    \"\"\"
    Constructs the CustomTFT + NixtlaFloorLoss from ProjectConfig.
    Separates construction from execution for easy hyperparameter swapping.
    \"\"\"

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    def build_loss(self) -> NixtlaFloorLoss:
        expert_loss = ExpertFloorLoss(
            base_loss=nn.MSELoss(),
            window_size=self.cfg.window_size,
        )
        return NixtlaFloorLoss(
            floor_loss=expert_loss,
            floor_weight=self.cfg.floor_loss_weight,
        )

    def build_model(self, floor_loss: NixtlaFloorLoss) -> CustomTFT:
        return CustomTFT(
            # ── Floor loss ──────────────────────────────────────────
            floor_loss=floor_loss,
            # ── Core TFT params ─────────────────────────────────────
            h=self.cfg.horizon,
            input_size=self.cfg.input_size,
            hidden_size=self.cfg.hidden_size,
            n_head=self.cfg.n_head,
            dropout=self.cfg.dropout,
            attn_dropout=self.cfg.attn_dropout,
            # ── Training ─────────────────────────────────────────────
            loss=floor_loss,          # NeuralForecast loss parameter
            max_steps=self.cfg.max_steps,
            learning_rate=self.cfg.learning_rate,
            batch_size=self.cfg.batch_size,
            windows_batch_size=self.cfg.windows_batch_size,
            scaler_type=self.cfg.scaler_type,
            # ── Validation ───────────────────────────────────────────
            val_check_steps=self.cfg.val_check_steps,
            early_stop_patience_steps=self.cfg.early_stop_patience,
            # ── Features ─────────────────────────────────────────────
            stat_exog_list=self.cfg.stat_exog_list,
            futr_exog_list=self.cfg.futr_exog_list,
            hist_exog_list=self.cfg.hist_exog_list,
            # ── Hardware ─────────────────────────────────────────────
            accelerator=DEVICE,
            devices=max(N_GPUS, 1),
        )

    def build(self) -> Tuple["CustomTFT", "NixtlaFloorLoss"]:
        loss  = self.build_loss()
        model = self.build_model(loss)
        return model, loss
"""))

cells.append(new_code_cell("""\
# ── Build ─────────────────────────────────────────────────────────────────────
builder     = ModelBuilder(CFG)
tft_model, floor_loss = builder.build()

nf = NeuralForecast(
    models=[tft_model],
    freq="MS",                    # monthly frequency
)

print(f"Model built. Parameters: "
      f"{sum(p.numel() for p in tft_model.parameters()):,}")
print(f"Accelerator: {DEVICE}  | Devices: {max(N_GPUS,1)}")
"""))

cells.append(new_code_cell("""\
# ── Train ─────────────────────────────────────────────────────────────────────
# For 20 K+ series: increase windows_batch_size and batch_size,
# enable gradient checkpointing via trainer_kwargs if memory-constrained.
nf.fit(
    df=train_sc,
    val_size=CFG.val_months,
    # Pass future exogenous needed for the validation window
    # futr_df is used during val look-ahead
)

# Save model to SM_OUTPUT
nf.save(path=SM_OUTPUT, model_index=None, overwrite=True, save_dataset=True)
print(f"Model saved to {SM_OUTPUT}")
"""))

# ─────────────────────────────────────────────
# CELL 9: Forecasting with Floor Constraint
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("## Part 7 · Forecasting with Floor Constraints"))

cells.append(new_code_cell("""\
class FloorConstrainedForecaster:
    \"\"\"
    Wraps NeuralForecast.predict() and applies the contract floor as a
    hard post-processing constraint.

    Strategy:
        1. Predict h steps with the trained model.
        2. Compute dynamic_floor for each future timestep.
        3. Clip predictions: y_hat = max(y_hat, dynamic_floor).
        4. Track and report floor violations before/after clipping.
    \"\"\"

    def __init__(
        self,
        nf: NeuralForecast,
        contract_info: pd.DataFrame,
        cfg: ProjectConfig,
        scalers: dict,
    ):
        self.nf            = nf
        self.contract_info = contract_info
        self.cfg           = cfg
        self.scalers       = scalers

    # ── Dynamic floor for future dates ────────────────────────────────────────
    def _compute_future_floor(self, futr_df: pd.DataFrame) -> pd.Series:
        floor_vals = []
        for _, row in futr_df.iterrows():
            uid   = row["unique_id"]
            años  = row["años"]
            ci    = self.contract_info.loc[uid]
            tipo  = int(ci["tipo"])
            v0    = float(ci["v_initial"])
            tasa  = float(ci["tasa_custom"])

            if tipo == 1:
                f = v0 * (1.054 ** años)
            elif tipo == 2:
                f = v0
            else:
                f = v0 * ((1 + tasa) ** años)
            floor_vals.append(f)
        return pd.Series(floor_vals, index=futr_df.index, name="floor")

    # ── Inverse scale y ───────────────────────────────────────────────────────
    def _inverse_scale_y(self, df: pd.DataFrame, col: str = "CustomTFT") -> pd.Series:
        # NeuralForecast handles y scaling internally with scaler_type='robust'
        # If you disabled it, apply inverse here.
        return df[col]

    # ── Main forecast ─────────────────────────────────────────────────────────
    def forecast(self, futr_df: pd.DataFrame) -> pd.DataFrame:
        raw_preds = self.nf.predict(futr_df=futr_df)

        pred_col  = "CustomTFT"    # NeuralForecast names predictions by class
        if pred_col not in raw_preds.columns:
            pred_col = [c for c in raw_preds.columns
                        if "TFT" in c][0]

        raw_preds = raw_preds.reset_index()

        # Merge future floor
        floor_series = self._compute_future_floor(futr_df.reset_index(drop=True))
        raw_preds["floor"] = floor_series.values

        # Hard clip
        raw_preds["y_hat_constrained"] = np.maximum(
            raw_preds[pred_col].values,
            raw_preds["floor"].values,
        )

        # Violation stats
        n_violations_pre  = (raw_preds[pred_col] < raw_preds["floor"]).sum()
        n_violations_post = (raw_preds["y_hat_constrained"] < raw_preds["floor"]).sum()
        pct = 100 * n_violations_pre / len(raw_preds)

        print(f"Floor violations (pre-clip) : {n_violations_pre:,} ({pct:.1f}%)")
        print(f"Floor violations (post-clip): {n_violations_post}")

        return raw_preds

    # ── Visualise ─────────────────────────────────────────────────────────────
    def plot(
        self,
        hist_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        n_series: int = 5,
    ) -> None:
        pred_col = [c for c in pred_df.columns if "TFT" in c][0]
        uids     = pred_df["unique_id"].unique()[:n_series]

        fig, axes = plt.subplots(len(uids), 1, figsize=(14, 3.5 * len(uids)))
        if len(uids) == 1:
            axes = [axes]

        for ax, uid in zip(axes, uids):
            hist = hist_df[hist_df["unique_id"] == uid].sort_values("ds")
            pred = pred_df[pred_df["unique_id"] == uid].sort_values("ds")

            ax.plot(hist["ds"], hist["y"], label="History", lw=1.5)
            ax.plot(pred["ds"], pred[pred_col], "--",  label="Forecast (raw)", alpha=0.8)
            ax.plot(pred["ds"], pred["y_hat_constrained"],
                    lw=2, label="Forecast (floor-clipped)", color="green")
            ax.fill_between(pred["ds"],
                            pred["floor"], pred["y_hat_constrained"],
                            alpha=0.15, color="orange", label="Floor zone")
            ax.set_title(uid, fontsize=9)
            ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(f"{SM_RESULTS}/forecasts.png", dpi=100, bbox_inches="tight")
        plt.show()
"""))

cells.append(new_code_cell("""\
forecaster = FloorConstrainedForecaster(
    nf=nf,
    contract_info=contract_info,
    cfg=CFG,
    scalers=scalers,
)

preds_df = forecaster.forecast(futr_df)
forecaster.plot(test_sc, preds_df, n_series=5)
preds_df.head()
"""))

# ─────────────────────────────────────────────
# CELL 10: Evaluation
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("## Part 8 · Evaluation"))

cells.append(new_code_cell("""\
class ModelEvaluator:
    \"\"\"
    Computes standard forecasting metrics plus business-specific metrics:
        - MAE, RMSE, MAPE, SMAPE
        - Floor Violation Rate (%) before and after clipping
        - Max violation depth (max(floor - y_hat, 0))
    \"\"\"

    @staticmethod
    def _safe_mape(actual, predicted):
        mask = actual != 0
        return 100 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))

    @staticmethod
    def _smape(actual, predicted):
        denom = (np.abs(actual) + np.abs(predicted)) / 2 + 1e-8
        return 100 * np.mean(np.abs(actual - predicted) / denom)

    def evaluate(
        self,
        actuals_df: pd.DataFrame,
        preds_df:   pd.DataFrame,
        pred_col:   str = "y_hat_constrained",
    ) -> pd.DataFrame:
        pred_col_raw = [c for c in preds_df.columns if "TFT" in c][0]

        merged = actuals_df[["unique_id", "ds", "y"]].merge(
            preds_df[["unique_id", "ds", pred_col_raw, pred_col, "floor"]],
            on=["unique_id", "ds"],
            how="inner",
        )

        if merged.empty:
            print("No overlapping dates for evaluation.")
            return pd.DataFrame()

        act = merged["y"].values
        raw = merged[pred_col_raw].values
        con = merged[pred_col].values
        flr = merged["floor"].values

        results = {
            "MAE (raw)":       np.mean(np.abs(act - raw)),
            "MAE (clipped)":   np.mean(np.abs(act - con)),
            "RMSE (raw)":      np.sqrt(np.mean((act - raw) ** 2)),
            "RMSE (clipped)":  np.sqrt(np.mean((act - con) ** 2)),
            "MAPE (raw)":      self._safe_mape(act, raw),
            "MAPE (clipped)":  self._safe_mape(act, con),
            "SMAPE (clipped)": self._smape(act, con),
            "Floor viol. %":   100 * (raw < flr).mean(),
            "Max viol. depth": float(np.max(np.maximum(flr - raw, 0))),
            "n_series_eval":   merged["unique_id"].nunique(),
            "n_points_eval":   len(merged),
        }

        df_res = pd.DataFrame.from_dict(results, orient="index", columns=["value"])
        print("\\n── Evaluation Results ──")
        print(df_res.round(4).to_string())
        df_res.to_csv(f"{SM_RESULTS}/evaluation.csv")
        return df_res

    def per_series_report(
        self,
        actuals_df: pd.DataFrame,
        preds_df:   pd.DataFrame,
        contract_info: pd.DataFrame,
    ) -> pd.DataFrame:
        \"\"\"Per-series breakdown including contract type.\"\"\"
        pred_col_raw = [c for c in preds_df.columns if "TFT" in c][0]
        merged = actuals_df[["unique_id", "ds", "y"]].merge(
            preds_df[["unique_id", "ds", pred_col_raw, "floor"]],
            on=["unique_id", "ds"], how="inner",
        )

        rows = []
        for uid, g in merged.groupby("unique_id"):
            act, raw = g["y"].values, g[pred_col_raw].values
            flr = g["floor"].values
            ci  = contract_info.loc[uid] if uid in contract_info.index else {}
            rows.append({
                "unique_id":   uid,
                "tipo":        ci.get("tipo", "?"),
                "MAE":         np.mean(np.abs(act - raw)),
                "RMSE":        np.sqrt(np.mean((act - raw) ** 2)),
                "viol_pct":    100 * (raw < flr).mean(),
                "max_viol":    float(np.max(np.maximum(flr - raw, 0))),
            })

        df = pd.DataFrame(rows).sort_values("viol_pct", ascending=False)
        df.to_csv(f"{SM_RESULTS}/per_series_eval.csv", index=False)
        return df
"""))

cells.append(new_code_cell("""\
evaluator = ModelEvaluator()

# Use the test window as ground truth (last test_months rows per series)
test_actuals = (
    test_sc
    .sort_values(["unique_id","ds"])
    .groupby("unique_id")
    .tail(CFG.test_months)
)

metrics_df  = evaluator.evaluate(test_actuals, preds_df)
per_series  = evaluator.per_series_report(test_actuals, preds_df, contract_info)
per_series.head(10)
"""))

# ─────────────────────────────────────────────
# CELL 11: Hyperparameter Tuning
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Part 9 · Hyperparameter Tuning

Two strategies are provided:
1. **AutoTFT** — NeuralForecast's built-in Optuna wrapper (recommended for quick search).
2. **Manual Optuna loop** — gives full control over search space and pruning.
"""))

cells.append(new_code_cell("""\
@dataclass
class TuningConfig:
    n_trials:     int   = 20
    timeout_secs: int   = 3600          # 1 hour wall-clock budget
    metric:       str   = "val_loss"
    direction:    str   = "minimize"
    sampler:      str   = "TPE"         # TPE | Random | CmaEs

    # Search bounds
    hidden_size_choices: List[int]   = field(default_factory=lambda: [64, 128, 256])
    n_head_choices:      List[int]   = field(default_factory=lambda: [2, 4, 8])
    lr_low:   float = 1e-4
    lr_high:  float = 1e-2
    dropout_low:  float = 0.0
    dropout_high: float = 0.3
    floor_w_low:  float = 0.1
    floor_w_high: float = 2.0
    max_steps_choices: List[int] = field(default_factory=lambda: [500, 1000, 2000])

TUNE_CFG = TuningConfig()
"""))

cells.append(new_markdown_cell("### 9.1 Strategy A — AutoTFT (NeuralForecast built-in)"))

cells.append(new_code_cell("""\
def run_auto_tft(
    train_df: pd.DataFrame,
    tune_cfg: TuningConfig,
    cfg: ProjectConfig,
) -> NeuralForecast:
    \"\"\"
    NeuralForecast's AutoTFT uses Optuna internally.
    Best for a fast sweep of core TFT hyperparameters.
    \"\"\"
    from neuralforecast.auto import AutoTFT
    import ray

    ray.init(ignore_reinit_error=True, num_gpus=max(N_GPUS, 0))

    config = AutoTFT.get_default_config()
    config.update({
        "h":           cfg.horizon,
        "input_size":  cfg.input_size,
        "hidden_size": tune_cfg.hidden_size_choices,
        "n_head":      tune_cfg.n_head_choices,
        "learning_rate": {"grid_search": [tune_cfg.lr_low, 5e-4, tune_cfg.lr_high]},
        "dropout":       {"grid_search": [0.05, 0.1, 0.2]},
        "stat_exog_list": cfg.stat_exog_list,
        "futr_exog_list": cfg.futr_exog_list,
        "hist_exog_list": cfg.hist_exog_list,
        "scaler_type": cfg.scaler_type,
    })

    auto_tft = AutoTFT(
        h=cfg.horizon,
        config=config,
        num_samples=tune_cfg.n_trials,
        loss=MAE(),                 # standard loss for search speed
    )

    nf_auto = NeuralForecast(models=[auto_tft], freq="MS")
    nf_auto.fit(df=train_df, val_size=cfg.val_months)

    best_params = auto_tft.results.get_best_result().config
    print("\\nBest AutoTFT params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    nf_auto.save(path=f"{SM_OUTPUT}/auto_tft_best", overwrite=True)
    return nf_auto
"""))

cells.append(new_markdown_cell("### 9.2 Strategy B — Manual Optuna Loop"))

cells.append(new_code_cell("""\
def run_optuna_tuning(
    train_df: pd.DataFrame,
    tune_cfg: TuningConfig,
    cfg:      ProjectConfig,
) -> dict:
    \"\"\"
    Full control Optuna loop with floor_weight in the search space.
    Returns the best hyperparameter dict.
    \"\"\"
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler

    SAMPLERS = {
        "TPE":    TPESampler(seed=42),
        "Random": RandomSampler(seed=42),
        "CmaEs":  CmaEsSampler(seed=42),
    }
    sampler = SAMPLERS.get(tune_cfg.sampler, TPESampler(seed=42))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden_size": trial.suggest_categorical(
                "hidden_size", tune_cfg.hidden_size_choices),
            "n_head": trial.suggest_categorical(
                "n_head", tune_cfg.n_head_choices),
            "learning_rate": trial.suggest_float(
                "learning_rate", tune_cfg.lr_low, tune_cfg.lr_high, log=True),
            "dropout": trial.suggest_float(
                "dropout", tune_cfg.dropout_low, tune_cfg.dropout_high),
            "max_steps": trial.suggest_categorical(
                "max_steps", tune_cfg.max_steps_choices),
            "floor_weight": trial.suggest_float(
                "floor_weight", tune_cfg.floor_w_low, tune_cfg.floor_w_high),
        }

        # Build trial model
        trial_cfg       = ProjectConfig(**{**cfg.__dict__, **params,
                                           "floor_loss_weight": params.pop("floor_weight")})
        trial_builder   = ModelBuilder(trial_cfg)
        t_model, t_loss = trial_builder.build()

        trial_nf = NeuralForecast(models=[t_model], freq="MS")
        try:
            trial_nf.fit(df=train_df, val_size=cfg.val_months)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

        # Use internal val_loss from model log (last checkpoint)
        val_loss = getattr(t_model, "valid_loss", None)
        if val_loss is None:
            # Fall back to cross-validation on held-out slice
            cv_preds = trial_nf.cross_validation(
                df=train_df, n_windows=1, h=cfg.horizon, step_size=cfg.horizon,
            )
            pred_col = [c for c in cv_preds.columns if "TFT" in c][0]
            val_loss = float(np.mean(np.abs(cv_preds["y"] - cv_preds[pred_col])))

        return float(val_loss)

    study = optuna.create_study(
        direction=tune_cfg.direction,
        sampler=sampler,
        study_name="tft_expertfloor",
    )
    study.optimize(
        objective,
        n_trials=tune_cfg.n_trials,
        timeout=tune_cfg.timeout_secs,
        n_jobs=1,               # 1 per GPU; increase for CPU-only search
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\\nBest trial val_loss: {study.best_value:.6f}")
    print(f"Best params: {best}")

    # Persist study
    import pickle
    with open(f"{SM_RESULTS}/optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    return best


# ── Choose strategy ────────────────────────────────────────────────────────────
TUNE_STRATEGY = "optuna"   # "auto_tft" | "optuna"

if TUNE_STRATEGY == "auto_tft":
    nf_tuned    = run_auto_tft(train_sc, TUNE_CFG, CFG)
    best_params = {}          # stored inside nf_tuned
elif TUNE_STRATEGY == "optuna":
    best_params = run_optuna_tuning(train_sc, TUNE_CFG, CFG)
    # Rebuild final model with best params and retrain on full train+val
    final_cfg   = ProjectConfig(**{**CFG.__dict__,
                                   **{k: v for k, v in best_params.items()
                                      if k in CFG.__dict__}})
    final_builder        = ModelBuilder(final_cfg)
    final_model, f_loss  = final_builder.build()
    nf_tuned = NeuralForecast(models=[final_model], freq="MS")
    nf_tuned.fit(df=val_sc, val_size=CFG.val_months)   # train on train+val
    nf_tuned.save(path=f"{SM_OUTPUT}/tuned_best", overwrite=True)
"""))

# ─────────────────────────────────────────────
# CELL 12: SageMaker Deployment
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("## Part 10 · SageMaker Training Job\n\nRun this section **from your local machine** to launch a remote training job."))

cells.append(new_code_cell("""\
class SageMakerDeployer:
    \"\"\"
    Packages and launches a SageMaker PyTorch training job.

    Instance recommendations:
        ml.g4dn.xlarge   – 1× T4  (16 GB VRAM) — prototype / small datasets
        ml.g4dn.4xlarge  – 1× T4  (16 GB VRAM) — larger CPU RAM + bandwidth
        ml.g4dn.12xlarge – 4× T4  (64 GB VRAM) — multi-GPU for 20 K+ series
        ml.p3.2xlarge    – 1× V100 (16 GB)      — fast single-GPU training
        ml.p3.8xlarge    – 4× V100 (64 GB)      — distributed large-scale
        ml.p4d.24xlarge  – 8× A100 (640 GB)     – production-scale (expensive)

    Estimated cost (us-east-1, on-demand):
        g4dn.xlarge  : ~$0.53/hr
        g4dn.12xlarge: ~$3.91/hr
        p3.8xlarge   : ~$12.24/hr
    \"\"\"

    ENTRY_POINT  = "nixtla_tft_expertfloor_sagemaker.py"
    SOURCE_DIR   = "."

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    def _get_role_and_session(self):
        import sagemaker
        session = sagemaker.Session()
        try:
            role = sagemaker.get_execution_role()
        except Exception:
            role = f"arn:aws:iam::{self.cfg.s3_bucket.split('-')[0]}:role/SageMakerRole"
        return role, session

    def upload_data(self, local_csv: str) -> str:
        \"\"\"Upload the local CSV to S3 and return the S3 URI.\"\"\"
        import boto3
        s3 = boto3.client("s3")
        key = f"{self.cfg.s3_prefix}/data/timeseries.csv"
        s3.upload_file(local_csv, self.cfg.s3_bucket, key)
        uri = f"s3://{self.cfg.s3_bucket}/{key}"
        print(f"Data uploaded → {uri}")
        return uri

    def launch(
        self,
        data_s3_uri:   Optional[str] = None,
        instance_type: Optional[str] = None,
        spot:          bool           = False,
    ) -> "sagemaker.estimator.Estimator":
        import sagemaker
        from sagemaker.pytorch import PyTorch

        role, session = self._get_role_and_session()
        inst = instance_type or self.cfg.instance_type

        estimator = PyTorch(
            entry_point  = self.ENTRY_POINT,
            source_dir   = self.SOURCE_DIR,
            role         = role,
            instance_count = 1,
            instance_type  = inst,
            framework_version = self.cfg.framework_version,
            py_version    = self.cfg.py_version,
            hyperparameters = {
                "horizon":          self.cfg.horizon,
                "input_size":       self.cfg.input_size,
                "hidden_size":      self.cfg.hidden_size,
                "n_head":           self.cfg.n_head,
                "max_steps":        self.cfg.max_steps,
                "batch_size":       self.cfg.batch_size,
                "learning_rate":    self.cfg.learning_rate,
                "floor_loss_weight": self.cfg.floor_loss_weight,
                "add_synthetic":    int(self.cfg.add_synthetic),
                "n_synthetic_per_type": self.cfg.n_synthetic_per_type,
            },
            volume_size  = 100,              # GB EBS
            max_run      = 86_400,           # 24 h max
            use_spot_instances     = spot,
            max_wait               = 172_800 if spot else None,
            output_path = f"s3://{self.cfg.s3_bucket}/{self.cfg.s3_prefix}/output",
            metric_definitions=[
                {"Name": "train:loss",     "Regex": r"train_loss=([0-9.]+)"},
                {"Name": "valid:loss",     "Regex": r"valid_loss=([0-9.]+)"},
                {"Name": "floor:penalty",  "Regex": r"floor_penalty=([0-9.]+)"},
            ],
            sagemaker_session = session,
        )

        channels = {}
        if data_s3_uri:
            channels["train"] = data_s3_uri

        estimator.fit(inputs=channels if channels else None, wait=False)
        print(f"\\nJob launched: {estimator.latest_training_job.name}")
        print(f"Instance:     {inst}")
        print(f"Spot:         {spot}")
        print(f"Outputs  → s3://{self.cfg.s3_bucket}/{self.cfg.s3_prefix}/output")
        return estimator

    def download_model(self, estimator) -> str:
        \"\"\"Download trained model artifacts from S3.\"\"\"
        import tarfile, boto3, io
        uri  = estimator.model_data
        bucket, key = uri[5:].split("/", 1)
        s3   = boto3.client("s3")
        buf  = io.BytesIO()
        s3.download_fileobj(bucket, key, buf)
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(SM_OUTPUT)
        print(f"Model extracted to {SM_OUTPUT}")
        return SM_OUTPUT
"""))

cells.append(new_code_cell("""\
# ── Launch (run locally, not inside SageMaker) ────────────────────────────────
deployer = SageMakerDeployer(CFG)

# Uncomment to run:
# data_uri  = deployer.upload_data(str(SAMPLE_CSV))
# estimator = deployer.launch(
#     data_s3_uri   = data_uri,
#     instance_type = "ml.g4dn.xlarge",
#     spot          = True,              # ~70% cost saving
# )
print("SageMaker launch ready. Uncomment lines above when running locally.")
"""))

# ─────────────────────────────────────────────
# CELL 13: Appendix — Raw Data Requirements
# ─────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## Appendix · Raw Data Requirements

For production with **20,000+ time series**, the pipeline requires:

---

### A. Core Table — `timeseries_main`

| Column | Type | Description |
|--------|------|-------------|
| `date` | INT YYYYMM | Observation month |
| `ts_id` | VARCHAR | Unique contract/item ID |
| `actual_cost_paid` | FLOAT | Monthly spend (target variable) |
| `contract_value` | FLOAT (nullable) | Reference contract amount |

**Scale**: expect 20K × 72 months = **1.44M rows** minimum.

---

### B. Contract Metadata Table — `contracts_meta`

| Column | Type | Description |
|--------|------|-------------|
| `ts_id` | VARCHAR | FK to timeseries_main |
| `contract_start_date` | DATE | When contract began |
| `contract_type` | INT (1/2/3) | Type as per ExpertFloorLoss |
| `initial_value` | FLOAT | v₀ for floor calculation |
| `custom_rate` | FLOAT | tasa_custom (0.0 for Types 1 & 2) |
| `supplier_id` | VARCHAR | Optional grouping dimension |
| `category` | VARCHAR | Optional grouping dimension |

> **If this table doesn't exist**, the `ContractTypeClassifier` will
> infer types automatically from the time-series patterns.

---

### C. Quality Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Series length | 24 months | 48+ months |
| Missing values in `y` | < 5 % | 0 % |
| Series with `contract_value` | 0 % | 60 %+ |
| Date continuity | Monthly, no gaps | Monthly, no gaps |
| `ts_id` uniqueness | Required | Required |

---

### D. S3 Layout (SageMaker)

```
s3://<bucket>/tft-expertfloor/
├── data/
│   ├── timeseries.csv          ← upload via deployer.upload_data()
│   └── contracts_meta.csv      ← optional contract metadata
├── output/
│   └── <job-name>/
│       └── output/model.tar.gz ← trained model artifacts
└── results/
    ├── eda_series.png
    ├── forecasts.png
    ├── evaluation.csv
    └── per_series_eval.csv
```

---

### E. Hardware Sizing Guide (Training)

| Series count | Recommended instance | Est. training time |
|-------------|---------------------|-------------------|
| < 5 K | `ml.g4dn.xlarge` (1× T4) | 30–60 min |
| 5–20 K | `ml.g4dn.4xlarge` (1× T4) | 1–3 h |
| 20–100 K | `ml.g4dn.12xlarge` (4× T4) | 2–6 h |
| 100 K+ | `ml.p3.8xlarge` (4× V100) | 4–12 h |

---

### F. Checklist Before Training

- [ ] CSV uploaded to S3 with correct schema
- [ ] `contract_value` populated for ≥ 50 % of series
- [ ] All series have ≥ 24 monthly observations
- [ ] `ts_id` values are stable (no renames across time)
- [ ] `ProjectConfig.s3_bucket` and `s3_prefix` set correctly
- [ ] SageMaker execution role has `s3:GetObject` and `s3:PutObject`
- [ ] GPU instance quota verified in AWS console
"""))

# ─────────────────────────────────────────────
# Assemble & write notebook
# ─────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.10.0",
}

out_path = Path(__file__).parent / "nixtla_tft_expertfloor_sagemaker.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"✓ Notebook written → {out_path}")
