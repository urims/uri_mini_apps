"""
Validation Service
==================
Validates uploaded part-list files against strict schema requirements.

Rules:
- Allowed column sets: {part_number, orp_code} or {part_number} only.
- part_number must be non-null string.
- orp_code can be string or int32, normalized to string with '.0' stripped.
- No fully-empty rows.
- Reports duplicate detection as warnings (not errors).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from app.core.exceptions import ValidationError
from app.schemas import KeyMode, PartsFileValidation

logger = logging.getLogger(__name__)

# The ONLY column name sets we accept
ALLOWED_SCHEMAS = {
    frozenset({"part_number", "orp_code"}): KeyMode.PART_ORP,
    frozenset({"part_number"}): KeyMode.ONLY_PART_NUMBER,
}


class ValidationService:
    """Validates and normalizes uploaded part-list files."""

    def validate_and_load(
        self, file_bytes: bytes, filename: str
    ) -> tuple[pd.DataFrame, PartsFileValidation]:
        """
        Validate the uploaded file and return cleaned DataFrame + report.

        Parameters
        ----------
        file_bytes : raw bytes of the uploaded file
        filename   : original filename (used to detect extension)

        Returns
        -------
        (df, validation_result)

        Raises
        ------
        ValidationError if the file is fundamentally invalid.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # ── 1. Read file ────────────────────────────────────────────────
        ext = Path(filename).suffix.lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(io.BytesIO(file_bytes), dtype=str)
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(
                    io.BytesIO(file_bytes), engine="openpyxl", dtype=str
                )
            else:
                raise ValidationError(
                    f"Unsupported file extension: '{ext}'",
                    detail="Allowed: .csv, .xlsx",
                )
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(
                "Failed to parse uploaded file",
                detail=str(exc),
            )

        # ── 2. Normalize column names ───────────────────────────────────
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # ── 3. Check column schema ──────────────────────────────────────
        col_set = frozenset(df.columns)
        detected_mode = ALLOWED_SCHEMAS.get(col_set)

        if detected_mode is None:
            raise ValidationError(
                "Invalid column names in uploaded file.",
                detail=(
                    f"Found columns: {list(df.columns)}. "
                    "Accepted schemas: ['part_number', 'orp_code'] or ['part_number'] only."
                ),
            )

        # ── 4. Drop fully empty rows ────────────────────────────────────
        before = len(df)
        df = df.dropna(how="all")
        dropped_empty = before - len(df)
        if dropped_empty > 0:
            warnings.append(f"Dropped {dropped_empty} completely empty row(s).")

        if len(df) == 0:
            errors.append("File contains no data rows after removing blanks.")
            return df, PartsFileValidation(
                is_valid=False,
                detected_mode=detected_mode,
                errors=errors,
                warnings=warnings,
            )

        # ── 5. Validate part_number ─────────────────────────────────────
        null_pn = df["part_number"].isna().sum()
        if null_pn > 0:
            errors.append(
                f"{null_pn} row(s) have null/empty part_number. "
                "All rows must have a valid part_number."
            )
        df["part_number"] = df["part_number"].astype(str).str.strip()

        # ── 6. Validate orp_code (if present) ───────────────────────────
        unique_orps = 0
        if detected_mode == KeyMode.PART_ORP:
            null_orp = df["orp_code"].isna().sum()
            if null_orp > 0:
                errors.append(
                    f"{null_orp} row(s) have null/empty orp_code. "
                    "All rows must have a valid orp_code when using part_orp mode."
                )
            # Normalize: strip whitespace, remove trailing '.0' from int-like
            df["orp_code"] = (
                df["orp_code"]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
            unique_orps = df["orp_code"].nunique()

        # ── 7. Duplicate detection ──────────────────────────────────────
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            warnings.append(
                f"Found {dup_count} duplicate row(s). Duplicates will be removed."
            )
            df = df.drop_duplicates()

        # ── 8. Build result ─────────────────────────────────────────────
        result = PartsFileValidation(
            is_valid=len(errors) == 0,
            detected_mode=detected_mode,
            row_count=len(df),
            unique_parts=df["part_number"].nunique(),
            unique_orps=unique_orps,
            errors=errors,
            warnings=warnings,
        )

        logger.info(
            "Validation complete: valid=%s, mode=%s, rows=%d, errors=%d",
            result.is_valid,
            detected_mode,
            len(df),
            len(errors),
        )

        return df, result


# Module-level singleton
validation_service = ValidationService()
