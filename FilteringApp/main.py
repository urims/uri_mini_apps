"""
Data Filtering Platform - FastAPI Application
==============================================
Main entry point. Registers API and Web UI routers, configures
middleware, static files, and error handlers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.api.web_routes import router as web_router
from app.core.config import get_settings
from app.core.exceptions import DFPError
from app.services.catalog_service import catalog_service

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events."""
    settings = get_settings()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("Data root: %s", settings.DATA_ROOT)
    logger.info("Active prod run: %s", settings.ACTIVE_PROD_RUN)

    # Pre-load catalog
    catalog_service.load()
    logger.info("Catalog loaded: %d datasets", len(catalog_service.datasets))

    yield

    logger.info("Shutting down.")


# ── App ──────────────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Self-service platform for filtering and delivering ML production "
        "run data slices. Supports both a web UI and a REST API."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# Static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Routers
app.include_router(api_router)
app.include_router(web_router)


# ── Error Handlers ───────────────────────────────────────────────────────────


@app.exception_handler(DFPError)
async def dfp_error_handler(request: Request, exc: DFPError):
    """Handle all domain-specific errors uniformly."""
    return JSONResponse(
        status_code=400,
        content={
            "error": type(exc).__name__,
            "message": exc.message,
            "detail": exc.detail,
        },
    )
