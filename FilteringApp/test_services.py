"""
Tests for the Data Filtering Platform core services.

Run with: pytest tests/ -v
"""

import io
import os
import tempfile

import pandas as pd
import pytest

# Ensure settings use test paths
os.environ["DATA_ROOT"] = tempfile.mkdtemp()
os.environ["ACTIVE_PROD_RUN"] = "test_run"

from app.schemas import DatasetInfo, DateFilter, FilterStrategy, KeyMode
from app.services.filter_service import (
    FilterService,
    apply_date_filter,
)
from app.services.validation_service import ValidationService


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION SERVICE TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestValidationService:
    """Tests for parts file validation."""

    svc = ValidationService()

    def _csv_bytes(self, content: str) -> bytes:
        return content.encode("utf-8")

    def test_valid_part_orp_csv(self):
        csv = "part_number,orp_code\nABC123,100\nDEF456,200\n"
        df, result = self.svc.validate_and_load(self._csv_bytes(csv), "test.csv")
        assert result.is_valid
        assert result.detected_mode == KeyMode.PART_ORP
        assert result.row_count == 2
        assert result.unique_parts == 2

    def test_valid_part_only_csv(self):
        csv = "part_number\nABC123\nDEF456\nGHI789\n"
        df, result = self.svc.validate_and_load(self._csv_bytes(csv), "test.csv")
        assert result.is_valid
        assert result.detected_mode == KeyMode.ONLY_PART_NUMBER
        assert result.row_count == 3

    def test_invalid_columns(self):
        csv = "bad_column,another\n1,2\n"
        with pytest.raises(Exception):
            self.svc.validate_and_load(self._csv_bytes(csv), "test.csv")

    def test_orp_float_cleanup(self):
        csv = "part_number,orp_code\nABC,100.0\nDEF,200.0\n"
        df, result = self.svc.validate_and_load(self._csv_bytes(csv), "test.csv")
        assert result.is_valid
        assert "100" in df["orp_code"].values
        assert "100.0" not in df["orp_code"].values

    def test_duplicate_warning(self):
        csv = "part_number,orp_code\nABC,100\nABC,100\nDEF,200\n"
        df, result = self.svc.validate_and_load(self._csv_bytes(csv), "test.csv")
        assert result.is_valid
        assert result.row_count == 2  # Deduplicated
        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_empty_file(self):
        csv = "part_number,orp_code\n"
        df, result = self.svc.validate_and_load(self._csv_bytes(csv), "test.csv")
        assert not result.is_valid

    def test_unsupported_extension(self):
        with pytest.raises(Exception):
            self.svc.validate_and_load(b"data", "test.json")


# ═══════════════════════════════════════════════════════════════════════
# FILTER SERVICE TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestFilterService:
    """Tests for the filtering engine."""

    svc = FilterService()

    def _make_dataset(self, **overrides) -> DatasetInfo:
        defaults = dict(
            id="test",
            display_name="Test",
            description="Test dataset",
            category="test",
            filename_pattern="test",
            filter_columns=["part_number", "orp_code"],
            filter_strategy=FilterStrategy.PART_ORP_SEPARATED,
            key_mode=KeyMode.PART_ORP,
            date_filterable=False,
        )
        defaults.update(overrides)
        return DatasetInfo(**defaults)

    def test_part_orp_separated_filter(self):
        data = pd.DataFrame({
            "part_number": ["A", "B", "C", "D"],
            "orp_code": ["1", "2", "3", "4"],
            "value": [10, 20, 30, 40],
        })
        parts = pd.DataFrame({
            "part_number": ["A", "C"],
            "orp_code": ["1", "3"],
        })
        ds = self._make_dataset()
        result = self.svc.filter_dataset(data, parts, ds)
        assert len(result) == 2
        assert set(result["value"]) == {10, 30}

    def test_model_part_id_filter(self):
        data = pd.DataFrame({
            "model_part_id": ["PLANT1 PARTX 100", "PLANT2 PARTY 200", "PLANT3 PARTZ 300"],
            "value": [1, 2, 3],
        })
        parts = pd.DataFrame({
            "part_number": ["PARTX", "PARTZ"],
            "orp_code": ["100", "300"],
        })
        ds = self._make_dataset(
            filter_columns=["model_part_id"],
            filter_strategy=FilterStrategy.MODEL_PART_ID,
        )
        result = self.svc.filter_dataset(data, parts, ds)
        assert len(result) == 2

    def test_impact_filter(self):
        data = pd.DataFrame({
            "part_number_opod": ["A", "B", "C"],
            "qty": [10, 20, 30],
        })
        parts = pd.DataFrame({"part_number": ["A", "C"]})
        ds = self._make_dataset(
            filter_columns=["part_number_opod"],
            filter_strategy=FilterStrategy.IMPACT,
            key_mode=KeyMode.ONLY_PART_NUMBER,
        )
        result = self.svc.filter_dataset(data, parts, ds)
        assert len(result) == 2

    def test_remove_duplicates(self):
        data = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        parts = pd.DataFrame({"part_number": ["x"]})
        ds = self._make_dataset(
            filter_columns=None,
            filter_strategy=FilterStrategy.REMOVE_DUPLICATES,
            key_mode=KeyMode.NONE,
        )
        result = self.svc.filter_dataset(data, parts, ds)
        assert len(result) == 2

    def test_date_filter(self):
        df = pd.DataFrame({
            "year_month": [202101, 202106, 202201, 202306],
            "val": [1, 2, 3, 4],
        })
        result = apply_date_filter(df, "year_month", DateFilter(start=202101, end=202112))
        assert len(result) == 2

    def test_date_filter_with_key_filter(self):
        data = pd.DataFrame({
            "part_number": ["A", "A", "B", "B"],
            "orp_code": ["1", "1", "2", "2"],
            "year_month": [202101, 202201, 202101, 202201],
            "value": [10, 20, 30, 40],
        })
        parts = pd.DataFrame({"part_number": ["A"], "orp_code": ["1"]})
        ds = self._make_dataset(
            date_filterable=True,
            date_column="year_month",
        )
        date_f = DateFilter(start=202101, end=202112)
        result = self.svc.filter_dataset(data, parts, ds, date_f)
        assert len(result) == 1
        assert result.iloc[0]["value"] == 10
