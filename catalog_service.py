"""
Catalog Service
===============
Manages the registry of available datasets. Reads the YAML catalog file
and resolves actual file paths within the active production run directory.

This is the primary extension point: to add a new module from the
data_proposal pipeline, just add an entry to file_catalog.yaml.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import yaml

from app.core.config import get_settings
from app.core.exceptions import CatalogError, FileNotFoundInRunError
from app.schemas import CatalogResponse, DatasetInfo

logger = logging.getLogger(__name__)


class CatalogService:
    """Loads the dataset catalog and resolves file paths in the prod run."""

    def __init__(self) -> None:
        self._datasets: list[DatasetInfo] = []
        self._loaded = False

    def load(self) -> None:
        """Parse the YAML catalog into DatasetInfo objects."""
        settings = get_settings()
        catalog_path = Path(settings.CATALOG_PATH)

        if not catalog_path.exists():
            raise CatalogError(f"Catalog file not found: {catalog_path}")

        with open(catalog_path, "r") as f:
            raw = yaml.safe_load(f)

        self._datasets = [DatasetInfo(**entry) for entry in raw.get("datasets", [])]
        self._loaded = True
        logger.info("Loaded %d datasets from catalog", len(self._datasets))

    @property
    def datasets(self) -> list[DatasetInfo]:
        """Return all registered datasets."""
        if not self._loaded:
            self.load()
        return self._datasets

    @property
    def categories(self) -> list[str]:
        """Return unique category names, preserving catalog order."""
        seen: set[str] = set()
        result: list[str] = []
        for ds in self.datasets:
            if ds.category not in seen:
                seen.add(ds.category)
                result.append(ds.category)
        return result

    def get_dataset(self, dataset_id: str) -> DatasetInfo:
        """Fetch a single dataset by its ID."""
        for ds in self.datasets:
            if ds.id == dataset_id:
                return ds
        raise CatalogError(f"Dataset not found in catalog: {dataset_id}")

    def get_catalog_response(self) -> CatalogResponse:
        """Build the full catalog response for the API / frontend."""
        settings = get_settings()
        return CatalogResponse(
            datasets=self.datasets,
            categories=self.categories,
            active_prod_run=settings.ACTIVE_PROD_RUN,
        )

    def resolve_file_path(self, dataset_id: str) -> Path:
        """
        Walk the active prod run directory and find the CSV matching
        the dataset's filename_pattern.

        Returns the full path to the matched file.
        Raises FileNotFoundInRunError if no match is found.
        """
        settings = get_settings()
        ds = self.get_dataset(dataset_id)
        prod_run_dir = Path(settings.DATA_ROOT) / settings.ACTIVE_PROD_RUN

        if not prod_run_dir.exists():
            raise FileNotFoundInRunError(
                f"Production run directory not found: {prod_run_dir}"
            )

        pattern = re.compile(re.escape(ds.filename_pattern), re.IGNORECASE)

        for root, _dirs, files in os.walk(prod_run_dir):
            for fname in files:
                if fname.lower().endswith(".csv") and pattern.search(fname):
                    return Path(root) / fname

        raise FileNotFoundInRunError(
            f"No CSV file matching pattern '{ds.filename_pattern}' "
            f"found in {prod_run_dir}",
            detail=f"Dataset: {dataset_id}",
        )

    def list_available_prod_runs(self) -> list[str]:
        """List all prod_run_* directories found under DATA_ROOT."""
        settings = get_settings()
        data_root = Path(settings.DATA_ROOT)
        if not data_root.exists():
            return []
        return sorted(
            d.name
            for d in data_root.iterdir()
            if d.is_dir() and d.name.startswith("prod_run")
        )


# Module-level singleton
catalog_service = CatalogService()
