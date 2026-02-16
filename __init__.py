"""
Pydantic models for API request/response validation.

These schemas enforce strict input requirements:
- Part lists must have exactly the allowed column names.
- ORP codes can be string or int but are normalized to string.
- Date filters use integer YYYYMM format.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class KeyMode(str, Enum):
    """Defines what key columns are expected in the uploaded parts file."""
    PART_ORP = "part_orp"           # Requires part_number + orp_code
    ONLY_PART_NUMBER = "only_part_number"  # Requires only part_number
    NONE = "none"                   # No key filtering (e.g. dedup only)


class FilterStrategy(str, Enum):
    """Strategy used to apply the filter on a dataset."""
    MODEL_PART_ID = "model_part_id"
    PART_ORP_SEPARATED = "part_orp_separated"
    IMPACT = "impact"
    CH013 = "ch013"
    REMOVE_DUPLICATES = "remove_duplicates"


# ── Catalog Models ───────────────────────────────────────────────────────────


class DatasetInfo(BaseModel):
    """Schema for a single dataset entry in the catalog."""
    id: str
    display_name: str
    description: str
    category: str
    filename_pattern: str
    filter_columns: list[str] | None = None
    filter_strategy: FilterStrategy
    key_mode: KeyMode
    date_filterable: bool = False
    date_column: str | None = None
    date_format: str | None = None


class CatalogResponse(BaseModel):
    """Full catalog returned to the frontend or API consumer."""
    datasets: list[DatasetInfo]
    categories: list[str]
    active_prod_run: str


# ── Filter Request ───────────────────────────────────────────────────────────


class DateFilter(BaseModel):
    """Date range filter for a single dataset. Uses YYYYMM integer format."""
    start: int = Field(..., ge=190001, le=209912, description="Start date as YYYYMM integer")
    end: int = Field(..., ge=190001, le=209912, description="End date as YYYYMM integer")

    @field_validator("end")
    @classmethod
    def end_must_be_gte_start(cls, v: int, info) -> int:
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"end ({v}) must be >= start ({start})")
        return v


class FilterRequest(BaseModel):
    """
    API request body for a filtering job.

    selected_files: list of dataset IDs from the catalog
    date_filters: optional per-dataset date ranges
    """
    selected_files: list[str] = Field(..., min_length=1)
    date_filters: dict[str, DateFilter] = Field(default_factory=dict)


# ── Validation Result ────────────────────────────────────────────────────────


class PartsFileValidation(BaseModel):
    """Result of validating an uploaded parts file."""
    is_valid: bool
    detected_mode: KeyMode | None = None
    row_count: int = 0
    unique_parts: int = 0
    unique_orps: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ── Delivery Summary ─────────────────────────────────────────────────────────


class FileDeliverySummary(BaseModel):
    """Quality summary for a single filtered file in the delivery."""
    dataset_id: str
    display_name: str
    source_rows: int
    filtered_rows: int
    match_rate_pct: float
    date_filtered: bool = False
    date_range: str | None = None
    status: str = "ok"  # ok | warning | empty | error
    message: str | None = None


class DeliveryManifest(BaseModel):
    """Complete manifest included in every delivery ZIP."""
    request_id: str
    created_at: datetime
    active_prod_run: str
    parts_file_name: str
    parts_key_mode: KeyMode
    total_parts_uploaded: int
    files_requested: int
    files_delivered: int
    files_with_warnings: int
    files_empty: int
    files_errored: int
    total_rows_delivered: int
    zip_size_bytes: int = 0
    file_summaries: list[FileDeliverySummary]
