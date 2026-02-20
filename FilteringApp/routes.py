"""
API v1 Routes
=============
RESTful endpoints for programmatic access to the filtering platform.
Designed for consumption with pandas + requests.

Endpoints:
- GET  /api/v1/catalog         → list available datasets
- POST /api/v1/validate        → validate a parts file
- POST /api/v1/filter          → run filtering and download ZIP
- GET  /api/v1/prod-runs       → list available production runs
- GET  /api/v1/health          → health check
"""

from __future__ import annotations

import json
import logging
import uuid

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import get_settings
from app.core.exceptions import (
    CatalogError,
    DeliveryError,
    DFPError,
    FileNotFoundInRunError,
    FilterError,
    ValidationError,
)
from app.schemas import (
    CatalogResponse,
    DateFilter,
    DeliveryManifest,
    FilterRequest,
    PartsFileValidation,
)
from app.services.catalog_service import catalog_service
from app.services.delivery_service import delivery_service
from app.services.filter_service import filter_service
from app.services.validation_service import validation_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["API v1"])


# ── Health ───────────────────────────────────────────────────────────────────


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "active_prod_run": settings.ACTIVE_PROD_RUN,
    }


# ── Catalog ──────────────────────────────────────────────────────────────────


@router.get("/catalog", response_model=CatalogResponse)
async def get_catalog():
    """Return the full dataset catalog with metadata."""
    try:
        return catalog_service.get_catalog_response()
    except CatalogError as e:
        raise HTTPException(status_code=500, detail=e.message)


@router.get("/prod-runs")
async def list_prod_runs():
    """List all available production run directories."""
    return {"prod_runs": catalog_service.list_available_prod_runs()}


# ── Validate ─────────────────────────────────────────────────────────────────


@router.post("/validate", response_model=PartsFileValidation)
async def validate_parts_file(parts_file: UploadFile = File(...)):
    """
    Validate an uploaded parts file without running any filters.
    Returns validation results including detected key mode,
    row counts, and any errors or warnings.
    """
    try:
        file_bytes = await parts_file.read()
        _, result = validation_service.validate_and_load(
            file_bytes, parts_file.filename or "unknown"
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={
            "message": e.message,
            "detail": e.detail,
        })


# ── Filter ───────────────────────────────────────────────────────────────────


@router.post("/filter")
async def run_filter(
    parts_file: UploadFile = File(...),
    selected_files: str = Form(...),
    date_filters: str = Form(default="{}"),
):
    """
    Main filtering endpoint. Accepts a parts file and filter parameters,
    runs the filtering pipeline, and returns a ZIP archive.

    Parameters (multipart form):
    - parts_file: CSV or XLSX file with part keys
    - selected_files: JSON string of dataset ID list, e.g. '["CH013","model_input_forecasting"]'
    - date_filters: JSON string of date filters, e.g. '{"CH013": {"start": 202101, "end": 202512}}'

    Returns:
    - ZIP file as download (application/zip)
    """
    request_id = str(uuid.uuid4())[:12]

    # 1. Parse form inputs
    try:
        selected_ids: list[str] = json.loads(selected_files)
        raw_dates: dict = json.loads(date_filters)
        parsed_dates: dict[str, DateFilter] = {
            k: DateFilter(**v) for k, v in raw_dates.items()
        }
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid JSON in form parameters: {e}",
        )

    if not selected_ids:
        raise HTTPException(status_code=422, detail="No files selected.")

    # 2. Validate parts file
    try:
        file_bytes = await parts_file.read()
        parts_df, validation = validation_service.validate_and_load(
            file_bytes, parts_file.filename or "unknown"
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={
            "message": e.message,
            "detail": e.detail,
        })

    if not validation.is_valid:
        raise HTTPException(status_code=422, detail={
            "message": "Parts file validation failed.",
            "errors": validation.errors,
        })

    # 3. Process each selected dataset
    results: list[dict] = []

    for dataset_id in selected_ids:
        try:
            dataset = catalog_service.get_dataset(dataset_id)
            file_path = catalog_service.resolve_file_path(dataset_id)

            # Load the source CSV
            data = pd.read_csv(file_path, low_memory=False)
            source_rows = len(data)

            # Check key compatibility
            _check_key_compatibility(validation, dataset)

            # Apply filters
            date_filter = parsed_dates.get(dataset_id)
            filtered = filter_service.filter_dataset(
                data, parts_df, dataset, date_filter
            )

            results.append({
                "dataset": dataset,
                "df": filtered,
                "source_rows": source_rows,
                "date_filter": date_filter,
                "error": None,
            })

        except FileNotFoundInRunError as e:
            results.append({
                "dataset": catalog_service.get_dataset(dataset_id),
                "df": None,
                "source_rows": 0,
                "date_filter": parsed_dates.get(dataset_id),
                "error": f"File not found: {e.message}",
            })
        except CatalogError as e:
            raise HTTPException(status_code=404, detail=e.message)
        except Exception as e:
            logger.exception("Error filtering %s", dataset_id)
            results.append({
                "dataset": catalog_service.get_dataset(dataset_id),
                "df": None,
                "source_rows": 0,
                "date_filter": parsed_dates.get(dataset_id),
                "error": str(e),
            })

    # 4. Package results
    try:
        zip_path, manifest = delivery_service.create_package(
            request_id=request_id,
            parts_filename=parts_file.filename or "unknown",
            parts_key_mode=validation.detected_mode,
            total_parts=validation.unique_parts,
            results=results,
        )
    except DeliveryError as e:
        raise HTTPException(status_code=500, detail=e.message)

    # 5. Return ZIP
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"data_delivery_{request_id}.zip",
        headers={
            "X-Request-Id": request_id,
            "X-Files-Delivered": str(manifest.files_delivered),
            "X-Total-Rows": str(manifest.total_rows_delivered),
        },
    )


@router.post("/filter/preview")
async def preview_filter(
    parts_file: UploadFile = File(...),
    selected_files: str = Form(...),
    date_filters: str = Form(default="{}"),
):
    """
    Dry-run of the filter. Returns the manifest/summary without
    creating a ZIP. Useful for verifying before downloading large packages.
    """
    request_id = str(uuid.uuid4())[:12]

    try:
        selected_ids: list[str] = json.loads(selected_files)
        raw_dates: dict = json.loads(date_filters)
        parsed_dates: dict[str, DateFilter] = {
            k: DateFilter(**v) for k, v in raw_dates.items()
        }
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}")

    file_bytes = await parts_file.read()
    try:
        parts_df, validation = validation_service.validate_and_load(
            file_bytes, parts_file.filename or "unknown"
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"message": e.message, "detail": e.detail})

    if not validation.is_valid:
        raise HTTPException(status_code=422, detail={"message": "Validation failed", "errors": validation.errors})

    summaries = []
    for dataset_id in selected_ids:
        try:
            dataset = catalog_service.get_dataset(dataset_id)
            file_path = catalog_service.resolve_file_path(dataset_id)
            data = pd.read_csv(file_path, low_memory=False)
            source_rows = len(data)
            _check_key_compatibility(validation, dataset)
            date_filter = parsed_dates.get(dataset_id)
            filtered = filter_service.filter_dataset(data, parts_df, dataset, date_filter)

            summaries.append({
                "dataset_id": dataset_id,
                "display_name": dataset.display_name,
                "source_rows": source_rows,
                "filtered_rows": len(filtered),
                "match_rate_pct": round(len(filtered) / source_rows * 100, 2) if source_rows > 0 else 0,
                "date_filtered": date_filter is not None,
                "status": "ok" if len(filtered) > 0 else "empty",
            })
        except Exception as e:
            summaries.append({
                "dataset_id": dataset_id,
                "display_name": dataset_id,
                "source_rows": 0,
                "filtered_rows": 0,
                "match_rate_pct": 0,
                "date_filtered": False,
                "status": "error",
                "error": str(e),
            })

    return {"request_id": request_id, "summaries": summaries, "validation": validation.model_dump()}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _check_key_compatibility(validation: PartsFileValidation, dataset):
    """
    Ensure the uploaded key mode is compatible with the dataset.
    e.g. a part_orp dataset requires orp_code in the upload.
    """
    from app.schemas import KeyMode

    if dataset.key_mode == KeyMode.NONE:
        return  # No filtering needed

    if dataset.key_mode == KeyMode.PART_ORP and validation.detected_mode == KeyMode.ONLY_PART_NUMBER:
        raise FilterError(
            f"Dataset '{dataset.display_name}' requires both part_number and orp_code, "
            f"but the uploaded file only contains part_number."
        )
