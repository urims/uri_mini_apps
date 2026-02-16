"""
Filter Service
==============
Core filtering engine. Applies key-based and date-based filters to datasets
using the strategy defined in the catalog.

Filter Strategies:
- model_part_id:      File has a composite 'model_part_id' column (plant part orp).
                      Split and match against uploaded keys.
- part_orp_separated: File has 'part_number' and 'orp_code' as separate columns.
- impact:             IMPACT files with various part_number column names.
- ch013:              CH013 format with 'part number' and 'orp_cd' columns.
- remove_duplicates:  No key filtering, just deduplication.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

from app.schemas import DatasetInfo, DateFilter, FilterStrategy, KeyMode

logger = logging.getLogger(__name__)


# ── Strategy Interface ───────────────────────────────────────────────────────


class BaseFilterStrategy(ABC):
    """Base class for all filtering strategies."""

    @abstractmethod
    def apply(
        self,
        data: pd.DataFrame,
        parts: pd.DataFrame,
        dataset: DatasetInfo,
    ) -> pd.DataFrame:
        """
        Filter `data` using the uploaded `parts` DataFrame.

        Parameters
        ----------
        data    : The dataset to filter.
        parts   : The uploaded part-list (already validated and normalized).
        dataset : Catalog metadata for this dataset.

        Returns
        -------
        Filtered DataFrame.
        """
        ...


# ── Concrete Strategies ─────────────────────────────────────────────────────


class ModelPartIdStrategy(BaseFilterStrategy):
    """
    Handles files where parts are identified by a composite 'model_part_id'
    column in the format: 'PLANT_CODE PART_NUMBER ORP_CODE'.
    """

    def apply(
        self,
        data: pd.DataFrame,
        parts: pd.DataFrame,
        dataset: DatasetInfo,
    ) -> pd.DataFrame:
        col = dataset.filter_columns[0]  # e.g. "model_part_id"
        if col not in data.columns:
            logger.warning("Column '%s' not found in dataset %s", col, dataset.id)
            return pd.DataFrame()

        # Split composite key into components
        splits = data[col].astype(str).str.split(" ", expand=True)
        if splits.shape[1] < 3:
            logger.warning("model_part_id split produced < 3 columns in %s", dataset.id)
            return pd.DataFrame()

        data = data.copy()
        data["_part_number"] = splits[1].str.strip()
        data["_orp_code"] = splits[2].str.strip()

        # Build match keys
        parts_keys = set(
            zip(parts["part_number"], parts["orp_code"])
        )
        data_keys = list(zip(data["_part_number"], data["_orp_code"]))
        mask = [k in parts_keys for k in data_keys]

        result = data.loc[mask].drop(columns=["_part_number", "_orp_code"])
        return result


class PartOrpSeparatedStrategy(BaseFilterStrategy):
    """
    Handles files where 'part_number' and 'orp_code' are in separate columns.
    """

    def apply(
        self,
        data: pd.DataFrame,
        parts: pd.DataFrame,
        dataset: DatasetInfo,
    ) -> pd.DataFrame:
        pn_col, orp_col = dataset.filter_columns[0], dataset.filter_columns[1]

        for col in [pn_col, orp_col]:
            if col not in data.columns:
                logger.warning("Column '%s' not found in dataset %s", col, dataset.id)
                return pd.DataFrame()

        data = data.copy()
        data[pn_col] = data[pn_col].astype(str).str.strip()
        data[orp_col] = data[orp_col].astype(str).str.strip().str.replace(
            r"\.0$", "", regex=True
        )

        parts_keys = set(zip(parts["part_number"], parts["orp_code"]))
        data_keys = list(zip(data[pn_col], data[orp_col]))
        mask = [k in parts_keys for k in data_keys]

        return data.loc[mask]


class ImpactStrategy(BaseFilterStrategy):
    """
    Handles IMPACT files that filter by part number only.
    Column names vary (part_number_opod, part_number_hisi, etc.).
    """

    def apply(
        self,
        data: pd.DataFrame,
        parts: pd.DataFrame,
        dataset: DatasetInfo,
    ) -> pd.DataFrame:
        col = dataset.filter_columns[0]
        if col not in data.columns:
            logger.warning("Column '%s' not found in dataset %s", col, dataset.id)
            return pd.DataFrame()

        data = data.copy()
        data[col] = data[col].astype(str).str.strip()
        part_set = set(parts["part_number"].unique())

        return data.loc[data[col].isin(part_set)]


class CH013Strategy(BaseFilterStrategy):
    """
    Handles CH013 files with non-standard column names ('part number', 'orp_cd').
    Matches against the uploaded part_number + orp_code keys.
    """

    def apply(
        self,
        data: pd.DataFrame,
        parts: pd.DataFrame,
        dataset: DatasetInfo,
    ) -> pd.DataFrame:
        pn_col = dataset.filter_columns[0]  # "part number"
        orp_col = dataset.filter_columns[1]  # "orp_cd"

        for col in [pn_col, orp_col]:
            if col not in data.columns:
                logger.warning("Column '%s' not found in dataset %s", col, dataset.id)
                return pd.DataFrame()

        data = data.copy()
        data[pn_col] = data[pn_col].astype(str).str.strip()
        data[orp_col] = data[orp_col].astype(str).str.strip().str.replace(
            r"\.0$", "", regex=True
        )

        parts_keys = set(zip(parts["part_number"], parts["orp_code"]))
        data_keys = list(zip(data[pn_col], data[orp_col]))
        mask = [k in parts_keys for k in data_keys]

        return data.loc[mask]


class RemoveDuplicatesStrategy(BaseFilterStrategy):
    """No key filtering; just removes duplicate rows."""

    def apply(
        self,
        data: pd.DataFrame,
        parts: pd.DataFrame,
        dataset: DatasetInfo,
    ) -> pd.DataFrame:
        return data.drop_duplicates()


# ── Strategy Registry ────────────────────────────────────────────────────────

STRATEGY_MAP: dict[FilterStrategy, type[BaseFilterStrategy]] = {
    FilterStrategy.MODEL_PART_ID: ModelPartIdStrategy,
    FilterStrategy.PART_ORP_SEPARATED: PartOrpSeparatedStrategy,
    FilterStrategy.IMPACT: ImpactStrategy,
    FilterStrategy.CH013: CH013Strategy,
    FilterStrategy.REMOVE_DUPLICATES: RemoveDuplicatesStrategy,
}


# ── Date Filtering ───────────────────────────────────────────────────────────


def apply_date_filter(
    df: pd.DataFrame,
    date_column: str,
    date_filter: DateFilter,
) -> pd.DataFrame:
    """
    Filter a DataFrame by date range using integer YYYYMM column.

    Parameters
    ----------
    df          : DataFrame to filter
    date_column : Name of the date column
    date_filter : DateFilter with start/end as YYYYMM integers

    Returns
    -------
    Filtered DataFrame.
    """
    if date_column not in df.columns:
        logger.warning("Date column '%s' not found, skipping date filter", date_column)
        return df

    df = df.copy()
    df[date_column] = pd.to_numeric(df[date_column], errors="coerce")
    mask = (df[date_column] >= date_filter.start) & (df[date_column] <= date_filter.end)
    return df.loc[mask]


# ── Main Service ─────────────────────────────────────────────────────────────


class FilterService:
    """
    Orchestrates filtering: selects the right strategy, applies key filter,
    then optionally applies date filter.
    """

    def filter_dataset(
        self,
        data: pd.DataFrame,
        parts: pd.DataFrame,
        dataset: DatasetInfo,
        date_filter: DateFilter | None = None,
    ) -> pd.DataFrame:
        """
        Apply key-based filtering and optional date filtering to a dataset.

        Parameters
        ----------
        data        : Raw DataFrame loaded from the CSV.
        parts       : Validated parts DataFrame from user upload.
        dataset     : Catalog metadata.
        date_filter : Optional date range.

        Returns
        -------
        Filtered DataFrame.
        """
        # 1. Key-based filtering
        strategy_cls = STRATEGY_MAP.get(dataset.filter_strategy)
        if strategy_cls is None:
            raise ValueError(f"Unknown filter strategy: {dataset.filter_strategy}")

        strategy = strategy_cls()
        source_rows = len(data)

        filtered = strategy.apply(data, parts, dataset)
        logger.info(
            "Key filter on %s: %d → %d rows",
            dataset.id, source_rows, len(filtered),
        )

        # 2. Date filtering (if requested and supported)
        if date_filter and dataset.date_filterable and dataset.date_column:
            before_date = len(filtered)
            filtered = apply_date_filter(filtered, dataset.date_column, date_filter)
            logger.info(
                "Date filter on %s: %d → %d rows",
                dataset.id, before_date, len(filtered),
            )

        return filtered


# Module-level singleton
filter_service = FilterService()
