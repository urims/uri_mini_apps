"""
Delivery Service
================
Packages filtered DataFrames into a compressed ZIP archive with a
quality manifest. Handles temp file cleanup and size limits.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import DeliveryError
from app.schemas import (
    DatasetInfo,
    DateFilter,
    DeliveryManifest,
    FileDeliverySummary,
    KeyMode,
)

logger = logging.getLogger(__name__)


class DeliveryService:
    """Builds ZIP packages from filtering results with quality summaries."""

    def create_package(
        self,
        request_id: str,
        parts_filename: str,
        parts_key_mode: KeyMode,
        total_parts: int,
        results: list[dict],
    ) -> tuple[Path, DeliveryManifest]:
        """
        Create a ZIP archive containing all filtered CSV files plus a manifest.

        Parameters
        ----------
        request_id     : Unique identifier for this request.
        parts_filename : Original name of the uploaded parts file.
        parts_key_mode : Detected key mode of the parts file.
        total_parts    : Number of unique part keys uploaded.
        results        : List of dicts with keys:
                         - dataset: DatasetInfo
                         - df: pd.DataFrame (filtered)
                         - source_rows: int
                         - date_filter: DateFilter | None
                         - error: str | None

        Returns
        -------
        (zip_path, manifest)
        """
        settings = get_settings()
        temp_dir = Path(settings.TEMP_DIR) / request_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        zip_path = temp_dir / f"data_delivery_{request_id}.zip"
        summaries: list[FileDeliverySummary] = []
        total_rows = 0

        try:
            with zipfile.ZipFile(
                zip_path, "w", zipfile.ZIP_DEFLATED,
                compresslevel=settings.ZIP_COMPRESSION_LEVEL,
            ) as zf:
                for item in results:
                    dataset: DatasetInfo = item["dataset"]
                    df: pd.DataFrame | None = item.get("df")
                    source_rows: int = item.get("source_rows", 0)
                    date_filter: DateFilter | None = item.get("date_filter")
                    error: str | None = item.get("error")

                    summary = self._build_summary(
                        dataset, df, source_rows, date_filter, error
                    )
                    summaries.append(summary)

                    if df is not None and len(df) > 0:
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_bytes = csv_buffer.getvalue().encode("utf-8")

                        # Organize by category in the ZIP
                        archive_name = f"{dataset.category}/{dataset.id}.csv"
                        zf.writestr(archive_name, csv_bytes)
                        total_rows += len(df)

                # Build manifest
                manifest = DeliveryManifest(
                    request_id=request_id,
                    created_at=datetime.now(timezone.utc),
                    active_prod_run=settings.ACTIVE_PROD_RUN,
                    parts_file_name=parts_filename,
                    parts_key_mode=parts_key_mode,
                    total_parts_uploaded=total_parts,
                    files_requested=len(results),
                    files_delivered=sum(1 for s in summaries if s.status == "ok"),
                    files_with_warnings=sum(
                        1 for s in summaries if s.status == "warning"
                    ),
                    files_empty=sum(1 for s in summaries if s.status == "empty"),
                    files_errored=sum(1 for s in summaries if s.status == "error"),
                    total_rows_delivered=total_rows,
                    file_summaries=summaries,
                )

                # Add manifest to ZIP as JSON
                manifest_json = manifest.model_dump_json(indent=2)
                zf.writestr("_manifest.json", manifest_json)

                # Add human-readable summary
                zf.writestr("_summary.txt", self._format_text_summary(manifest))

            # Check size limit
            zip_size = zip_path.stat().st_size
            manifest.zip_size_bytes = zip_size
            max_bytes = settings.MAX_DELIVERY_SIZE_MB * 1024 * 1024

            if zip_size > max_bytes:
                raise DeliveryError(
                    f"Package size ({zip_size / 1024 / 1024:.1f} MB) exceeds "
                    f"the {settings.MAX_DELIVERY_SIZE_MB} MB limit."
                )

            logger.info(
                "Created delivery package: %s (%.1f MB, %d files, %d rows)",
                request_id,
                zip_size / 1024 / 1024,
                manifest.files_delivered,
                total_rows,
            )

            return zip_path, manifest

        except DeliveryError:
            raise
        except Exception as exc:
            raise DeliveryError(
                "Failed to create delivery package",
                detail=str(exc),
            )

    def cleanup(self, request_id: str) -> None:
        """Remove temporary files for a request."""
        settings = get_settings()
        temp_dir = Path(settings.TEMP_DIR) / request_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _build_summary(
        self,
        dataset: DatasetInfo,
        df: pd.DataFrame | None,
        source_rows: int,
        date_filter: DateFilter | None,
        error: str | None,
    ) -> FileDeliverySummary:
        """Build a quality summary for a single file."""
        if error:
            return FileDeliverySummary(
                dataset_id=dataset.id,
                display_name=dataset.display_name,
                source_rows=source_rows,
                filtered_rows=0,
                match_rate_pct=0.0,
                status="error",
                message=error,
            )

        filtered_rows = len(df) if df is not None else 0
        match_rate = (filtered_rows / source_rows * 100) if source_rows > 0 else 0.0

        status = "ok"
        message = None
        if filtered_rows == 0:
            status = "empty"
            message = "No matching rows found after filtering."
        elif match_rate < 1.0:
            status = "warning"
            message = f"Very low match rate ({match_rate:.2f}%). Verify keys."

        date_range_str = None
        if date_filter:
            date_range_str = f"{date_filter.start} - {date_filter.end}"

        return FileDeliverySummary(
            dataset_id=dataset.id,
            display_name=dataset.display_name,
            source_rows=source_rows,
            filtered_rows=filtered_rows,
            match_rate_pct=round(match_rate, 2),
            date_filtered=date_filter is not None,
            date_range=date_range_str,
            status=status,
            message=message,
        )

    def _format_text_summary(self, manifest: DeliveryManifest) -> str:
        """Generate a human-readable text summary."""
        lines = [
            "=" * 70,
            "DATA DELIVERY SUMMARY",
            "=" * 70,
            f"Request ID:       {manifest.request_id}",
            f"Created:          {manifest.created_at.isoformat()}",
            f"Prod Run:         {manifest.active_prod_run}",
            f"Parts File:       {manifest.parts_file_name}",
            f"Key Mode:         {manifest.parts_key_mode.value}",
            f"Parts Uploaded:   {manifest.total_parts_uploaded}",
            "",
            f"Files Requested:  {manifest.files_requested}",
            f"Files Delivered:  {manifest.files_delivered}",
            f"Files Empty:      {manifest.files_empty}",
            f"Files Errored:    {manifest.files_errored}",
            f"Total Rows:       {manifest.total_rows_delivered:,}",
            "",
            "-" * 70,
            f"{'Dataset':<45} {'Rows':>8} {'Match%':>8} {'Status':>8}",
            "-" * 70,
        ]

        for s in manifest.file_summaries:
            name = s.display_name[:44]
            lines.append(
                f"{name:<45} {s.filtered_rows:>8,} {s.match_rate_pct:>7.1f}% {s.status:>8}"
            )
            if s.date_filtered:
                lines.append(f"  └─ Date filter: {s.date_range}")
            if s.message:
                lines.append(f"  └─ {s.message}")

        lines.extend(["", "=" * 70])
        return "\n".join(lines)


# Module-level singleton
delivery_service = DeliveryService()
