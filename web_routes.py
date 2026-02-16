"""
Web UI Routes
=============
Server-rendered pages using Jinja2 templates for the self-service web app.
"""

from __future__ import annotations

import json
import logging
import uuid

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings
from app.core.exceptions import (
    CatalogError,
    DeliveryError,
    FileNotFoundInRunError,
    ValidationError,
)
from app.schemas import DateFilter
from app.services.catalog_service import catalog_service
from app.services.delivery_service import delivery_service
from app.services.filter_service import filter_service
from app.services.validation_service import validation_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Web UI"])

templates = Jinja2Templates(directory="frontend/templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page: upload parts file + shopping cart."""
    try:
        catalog = catalog_service.get_catalog_response()
    except Exception:
        catalog = None

    return templates.TemplateResponse("index.html", {
        "request": request,
        "catalog": catalog,
        "settings": get_settings(),
    })


@router.post("/upload-and-filter")
async def upload_and_filter(
    request: Request,
    parts_file: UploadFile = File(...),
    selected_files: str = Form(default="[]"),
    date_filters: str = Form(default="{}"),
):
    """
    Handle the form submission: validate, filter, package, return ZIP.
    On error, re-render the page with error messages.
    """
    request_id = str(uuid.uuid4())[:12]
    errors: list[str] = []

    # ── Parse form data ──────────────────────────────────────────────
    try:
        selected_ids: list[str] = json.loads(selected_files)
        raw_dates: dict = json.loads(date_filters)
        parsed_dates: dict[str, DateFilter] = {
            k: DateFilter(**v) for k, v in raw_dates.items()
        }
    except Exception as e:
        errors.append(f"Invalid form data: {e}")
        return _render_error(request, errors)

    if not selected_ids:
        errors.append("Please select at least one dataset from the catalog.")
        return _render_error(request, errors)

    # ── Validate parts file ──────────────────────────────────────────
    try:
        file_bytes = await parts_file.read()
        parts_df, validation = validation_service.validate_and_load(
            file_bytes, parts_file.filename or "unknown"
        )
    except ValidationError as e:
        errors.append(f"File validation error: {e.message}")
        if e.detail:
            errors.append(e.detail)
        return _render_error(request, errors)

    if not validation.is_valid:
        errors.extend(validation.errors)
        return _render_error(request, errors)

    # ── Run filtering ────────────────────────────────────────────────
    results: list[dict] = []
    for dataset_id in selected_ids:
        try:
            dataset = catalog_service.get_dataset(dataset_id)
            file_path = catalog_service.resolve_file_path(dataset_id)
            data = pd.read_csv(file_path, low_memory=False)
            source_rows = len(data)

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
                "error": f"File not found in production run: {e.message}",
            })
        except Exception as e:
            logger.exception("Error filtering %s", dataset_id)
            try:
                ds = catalog_service.get_dataset(dataset_id)
            except CatalogError:
                continue
            results.append({
                "dataset": ds,
                "df": None,
                "source_rows": 0,
                "date_filter": parsed_dates.get(dataset_id),
                "error": str(e),
            })

    # ── Package ──────────────────────────────────────────────────────
    try:
        zip_path, manifest = delivery_service.create_package(
            request_id=request_id,
            parts_filename=parts_file.filename or "unknown",
            parts_key_mode=validation.detected_mode,
            total_parts=validation.unique_parts,
            results=results,
        )
    except DeliveryError as e:
        errors.append(f"Packaging error: {e.message}")
        return _render_error(request, errors)

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"data_delivery_{request_id}.zip",
    )


def _render_error(request: Request, errors: list[str]):
    """Re-render the main page with error messages."""
    try:
        catalog = catalog_service.get_catalog_response()
    except Exception:
        catalog = None

    return templates.TemplateResponse("index.html", {
        "request": request,
        "catalog": catalog,
        "settings": get_settings(),
        "errors": errors,
    }, status_code=422)
