# ═══════════════════════════════════════════════════════════════════════
# Data Filtering Platform - Dockerfile
# Multi-stage build for smaller production image
# ═══════════════════════════════════════════════════════════════════════

FROM python:3.12-slim AS base

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser

WORKDIR /app

# ── Dependencies ─────────────────────────────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────
COPY backend/ .
COPY frontend/ frontend/

# Create temp directory
RUN mkdir -p /tmp/dfp && chown -R appuser:appuser /tmp/dfp

# Ensure data directory exists (will be mounted)
RUN mkdir -p /data && chown -R appuser:appuser /data

# Own the app directory
RUN chown -R appuser:appuser /app

USER appuser

# ── Health check ─────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# ── Entrypoint ───────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
