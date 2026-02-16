"""
Application configuration loaded from environment variables.

All paths and settings are centralized here. Override via .env file or
environment variables in Docker.
"""

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with sensible defaults for development."""

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = "Data Filtering Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Paths ────────────────────────────────────────────────────────────
    # Root directory where prod_run_{date} folders are stored
    DATA_ROOT: str = "/data"

    # Active production run folder name (e.g. "prod_run_2026_01")
    ACTIVE_PROD_RUN: str = "prod_run_latest"

    # Temporary directory for uploads and generated ZIP packages
    TEMP_DIR: str = "/tmp/dfp"

    # Path to the file catalog YAML
    CATALOG_PATH: str = str(
        Path(__file__).resolve().parent.parent / "file_catalog.yaml"
    )

    # ── Upload limits ────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50  # Max part-list upload in MB
    ALLOWED_EXTENSIONS: list[str] = [".csv", ".xlsx"]

    # ── Delivery ─────────────────────────────────────────────────────────
    ZIP_COMPRESSION_LEVEL: int = 6  # 1-9, higher = smaller but slower
    MAX_DELIVERY_SIZE_MB: int = 1024  # 1 GB cap

    # ── Server ───────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 2

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton for app settings."""
    return Settings()
