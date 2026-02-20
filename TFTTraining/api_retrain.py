"""
Strategy 5: Incremental Re-training via FastAPI Service.

Best for: Production systems needing continuous model updates.
Key design:
    - POST /forecast  → get predictions for any component
    - POST /retrain   → fine-tune model with new data (background task)
    - GET  /health    → check model version and training status

Fine-tuning strategy:
    - Low learning rate (10-100x lower than initial training)
    - Optional encoder freezing (keep learned temporal patterns)
    - Automatic model versioning with timestamps
    - Background task (non-blocking)

Run: uvicorn training.api_retrain:app --host 0.0.0.0 --port 8000
"""

import os
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import tensorflow as tf

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI dependencies required. Install: pip install -r requirements-api.txt"
    )

from tft_core.losses import quantile_loss

logger = logging.getLogger("tft_api")
logging.basicConfig(level=logging.INFO)

# ═══════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════


class SalesRecord(BaseModel):
    """Single sales observation for a component."""
    component_id: str
    date: str
    sales: float
    price: float
    contract_value: float
    promotion: int = 0
    lead_time: int = 14


class ForecastRequest(BaseModel):
    """Request for a forecast."""
    component_id: str
    horizon: int = 6
    known_future_contract: Optional[List[float]] = None
    known_future_promo: Optional[List[int]] = None


class ForecastResponse(BaseModel):
    """Forecast response with quantiles."""
    component_id: str
    predictions: List[dict]
    model_version: str
    timestamp: str


class RetrainRequest(BaseModel):
    """Request to fine-tune the model with new data."""
    new_records: List[SalesRecord]
    fine_tune_epochs: int = 10
    learning_rate: float = 1e-4
    freeze_encoder: bool = False


class HealthResponse(BaseModel):
    """API health status."""
    status: str
    model_version: str
    is_training: bool
    gpu_available: bool


# ═══════════════════════════════════════
# Application State
# ═══════════════════════════════════════

MODEL_PATH = os.environ.get("TFT_MODEL_PATH", "models/tft_production.keras")
model: Optional[tf.keras.Model] = None
model_version: str = "v1.0"
is_training: bool = False


# ═══════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════

app = FastAPI(
    title="TFT Forecast Service",
    description="Multi-component sales forecasting with optional constraint enforcement.",
    version="1.0.0",
)


@app.on_event("startup")
async def load_model():
    """Load the production model at startup."""
    global model, model_version

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"quantile_loss": quantile_loss},
        )
        logger.info(f"Model loaded from {MODEL_PATH} (version: {model_version})")
    else:
        logger.warning(f"Model not found at {MODEL_PATH}. Train a model first.")


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest):
    """
    Generate a forecast for a component.

    Returns P10, P50, P90 quantile predictions for the requested horizon.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Prepare input tensors (placeholder — implement per your data pipeline)
    static, hist, future = _prepare_inference_input(req)

    preds = model.predict(
        [np.array([static]), np.array([hist]), np.array([future])],
        verbose=0,
    )
    # preds shape: (1, H, 3) → [P10, P50, P90]

    predictions = []
    for h in range(req.horizon):
        predictions.append({
            "step": h + 1,
            "p10": float(preds[0, h, 0]),
            "p50": float(preds[0, h, 1]),
            "p90": float(preds[0, h, 2]),
        })

    return ForecastResponse(
        component_id=req.component_id,
        predictions=predictions,
        model_version=model_version,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/retrain")
async def retrain(req: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger incremental fine-tuning with new sales records.

    Runs as a background task — returns immediately.
    Check /health for training status.
    """
    global is_training

    if is_training:
        raise HTTPException(status_code=409, detail="Training already in progress")
    if model is None:
        raise HTTPException(status_code=503, detail="No base model to fine-tune")

    is_training = True
    background_tasks.add_task(_run_fine_tuning, req)

    return {
        "status": "training_started",
        "epochs": req.fine_tune_epochs,
        "learning_rate": req.learning_rate,
        "freeze_encoder": req.freeze_encoder,
        "n_records": len(req.new_records),
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check API and model status."""
    return HealthResponse(
        status="healthy" if model is not None else "no_model",
        model_version=model_version,
        is_training=is_training,
        gpu_available=len(tf.config.list_physical_devices("GPU")) > 0,
    )


# ═══════════════════════════════════════
# Internal Functions
# ═══════════════════════════════════════


async def _run_fine_tuning(req: RetrainRequest):
    """Background task: fine-tune model on new data."""
    global model, model_version, is_training

    try:
        logger.info(f"Fine-tuning with {len(req.new_records)} new records")

        # Prepare new data tensors
        X_static, X_hist, X_future, Y_new = _prepare_training_data(req.new_records)

        new_ds = (
            tf.data.Dataset.from_tensor_slices((
                {"static": X_static, "historical": X_hist, "known_future": X_future},
                Y_new,
            ))
            .shuffle(1000)
            .batch(64)
            .prefetch(2)
        )

        # Optionally freeze encoder layers
        if req.freeze_encoder:
            for layer in model.layers:
                if "enc" in layer.name:
                    layer.trainable = False
            logger.info("Encoder layers frozen")

        # CRITICAL: Low LR for fine-tuning to avoid catastrophic forgetting
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=req.learning_rate,
                clipnorm=0.5,
            ),
            loss=quantile_loss,
        )

        history = model.fit(new_ds, epochs=req.fine_tune_epochs, verbose=0)

        # Save versioned checkpoint
        new_version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        versioned_path = f"models/tft_{new_version}.keras"
        model.save(versioned_path)
        model.save(MODEL_PATH)

        # Unfreeze all layers for next cycle
        for layer in model.layers:
            layer.trainable = True

        model_version = new_version
        final_loss = history.history["loss"][-1]
        logger.info(f"Fine-tuning complete → {new_version} (loss: {final_loss:.4f})")

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}", exc_info=True)
    finally:
        is_training = False


def _prepare_inference_input(req: ForecastRequest):
    """
    Prepare inference tensors from a forecast request.

    NOTE: This is a placeholder. In production, you would:
        1. Look up the component's latest T timesteps from your database.
        2. Apply the same encoding/normalization as training.
        3. Build static, hist, future arrays.
    """
    # Placeholder shapes — replace with actual data loading
    T, H = 24, req.horizon
    static = np.zeros(4, dtype=np.float32)
    hist = np.zeros((T, 5), dtype=np.float32)
    future = np.zeros((H, 5), dtype=np.float32)
    return static, hist, future


def _prepare_training_data(records: List[SalesRecord]):
    """
    Prepare training tensors from new sales records.

    NOTE: Placeholder — replace with your TFTDataPipeline logic.
    """
    n = len(records)
    X_static = np.zeros((n, 4), dtype=np.float32)
    X_hist = np.zeros((n, 24, 5), dtype=np.float32)
    X_future = np.zeros((n, 6, 5), dtype=np.float32)
    Y = np.zeros((n, 6, 1), dtype=np.float32)
    return X_static, X_hist, X_future, Y
