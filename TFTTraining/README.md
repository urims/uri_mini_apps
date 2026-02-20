# Temporal Fusion Transformer — Multi-Component Forecasting

Production-ready implementation of the **Temporal Fusion Transformer (TFT)** for
multi-component time series forecasting, with an optional **α-lag contract floor
constraint** that guarantees sell prices never fall below the effective contract value.

## Project Structure

```
tft_project/
├── tft_core/                    # Standard TFT building blocks
│   ├── __init__.py
│   ├── layers.py                # GRN, VSN, InterpretableMultiHeadAttention
│   ├── model.py                 # build_tft() factory
│   ├── losses.py                # quantile_loss, weighted_quantile_loss
│   └── data_preparation.py      # Tables → TFT tensors pipeline
│
├── tft_constrained/             # Constrained TFT extension
│   ├── __init__.py
│   ├── constraint_layers.py     # AlphaShiftLayer, SoftConstraintLayer
│   ├── model.py                 # build_constrained_tft() factory
│   └── losses.py                # constrained_quantile_loss
│
├── training/                    # 5 training strategies
│   ├── __init__.py
│   ├── train_cpu_72core.py      # Strategy 1: 72-core CPU server
│   ├── train_sagemaker.py       # Strategy 2: Distributed GPU (SageMaker)
│   ├── train_gpu_local.py       # Strategy 3: Local GPU
│   ├── train_cpu_local.py       # Strategy 4: Local CPU
│   └── api_retrain.py           # Strategy 5: FastAPI re-training service
│
├── scripts/
│   ├── run_training.py          # CLI entry point
│   └── generate_sample_data.py  # Generate synthetic multi-component data
│
├── tests/
│   └── test_model.py            # Smoke tests
│
├── requirements.txt             # Core dependencies
├── requirements-gpu.txt         # GPU-specific dependencies
├── requirements-api.txt         # API service dependencies
├── requirements-sagemaker.txt   # SageMaker dependencies
└── README.md
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Generate sample data
python scripts/generate_sample_data.py

# Train (auto-detects hardware)
python scripts/run_training.py --strategy auto

# Train with specific strategy
python scripts/run_training.py --strategy gpu_local --constrained --alpha 3
```

## Components

### Standard TFT (`tft_core/`)
Full TFT with Variable Selection Networks, LSTM encoder-decoder, Interpretable
Multi-Head Attention, and quantile output (P10, P50, P90).

### Constrained TFT (`tft_constrained/`)
Extends Standard TFT with α-lag contract floor enforcement:
- **AlphaShiftLayer**: `floor(t) = contract(t − α) × (1 + margin)`
- **SoftConstraintLayer**: LogSumExp (training) / hard max (inference)

### Training Strategies (`training/`)
| Strategy | Hardware | Time/Epoch | Total |
|---|---|---|---|
| `cpu_72core` | 72 vCPU, 148 GB | ~45 min | 8-15 hrs |
| `sagemaker` | 4× A10G GPU | ~30 sec | 30-50 min |
| `gpu_local` | RTX 3090 | ~2-5 min | 1-3 hrs |
| `cpu_local` | 8-core laptop | ~15-40 min | 3-12 hrs |
| `api_retrain` | Any (incremental) | N/A | ~5 min/update |
```
