#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Data Filtering Platform - Deployment Script for Rocky Linux
# ═══════════════════════════════════════════════════════════════════════
# Prerequisites: Docker and Docker Compose installed on Rocky Linux.
#
# Usage:
#   chmod +x scripts/deploy.sh
#   ./scripts/deploy.sh
#
# What this script does:
#   1. Verifies Docker is installed and running
#   2. Creates .env from template if not present
#   3. Creates data directory structure
#   4. Builds and starts the application
#   5. Runs health check
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

cd "$PROJECT_DIR"

# ── 1. Check Docker ─────────────────────────────────────────────────
log_info "Checking Docker installation..."

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed. Install it with:"
    echo "  sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo"
    echo "  sudo dnf install docker-ce docker-ce-cli containerd.io docker-compose-plugin"
    echo "  sudo systemctl enable --now docker"
    echo "  sudo usermod -aG docker \$USER"
    exit 1
fi

if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running. Start it with:"
    echo "  sudo systemctl start docker"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    log_error "Docker Compose plugin not found. Install with:"
    echo "  sudo dnf install docker-compose-plugin"
    exit 1
fi

log_info "Docker $(docker --version | awk '{print $3}') ✓"

# ── 2. Setup .env ────────────────────────────────────────────────────
if [ ! -f .env ]; then
    log_warn ".env file not found. Creating from template..."
    cp .env.example .env
    log_info "Created .env file. Please review and adjust settings:"
    echo "  vim .env"
    echo ""
    log_warn "Key settings to verify:"
    echo "  DATA_ROOT       = Path to your external disk with prod_run folders"
    echo "  ACTIVE_PROD_RUN = Name of the current production run folder"
    echo "  APP_PORT         = Port to expose (default: 8080)"
    echo ""
    read -p "Press Enter to continue with defaults, or Ctrl+C to edit .env first..."
fi

source .env 2>/dev/null || true

# ── 3. Create data directory ─────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/data}"
ACTIVE_PROD_RUN="${ACTIVE_PROD_RUN:-prod_run_latest}"

log_info "Data root: $DATA_ROOT"
log_info "Active prod run: $ACTIVE_PROD_RUN"

if [ ! -d "$DATA_ROOT" ]; then
    log_warn "Data root directory '$DATA_ROOT' does not exist. Creating..."
    sudo mkdir -p "$DATA_ROOT"
    sudo chown "$USER:$USER" "$DATA_ROOT"
fi

PROD_RUN_PATH="$DATA_ROOT/$ACTIVE_PROD_RUN"
if [ ! -d "$PROD_RUN_PATH" ]; then
    log_warn "Production run directory '$PROD_RUN_PATH' does not exist."
    log_warn "Creating it. Copy your CSV files into this directory."
    mkdir -p "$PROD_RUN_PATH"
fi

# ── 4. Build and start ──────────────────────────────────────────────
log_info "Building Docker image..."
docker compose build

log_info "Starting application..."
docker compose up -d

# ── 5. Health check ──────────────────────────────────────────────────
APP_PORT="${APP_PORT:-8080}"
log_info "Waiting for application to start..."

for i in {1..30}; do
    if curl -sf "http://localhost:$APP_PORT/api/v1/health" > /dev/null 2>&1; then
        echo ""
        log_info "Application is healthy! ✓"
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  Web UI:   http://$(hostname -I | awk '{print $1}'):$APP_PORT"
        echo "  API Docs: http://$(hostname -I | awk '{print $1}'):$APP_PORT/api/docs"
        echo "  Health:   http://$(hostname -I | awk '{print $1}'):$APP_PORT/api/v1/health"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        log_info "Data directory: $PROD_RUN_PATH"
        log_info "Place your CSV data files in this directory."
        echo ""
        log_info "Useful commands:"
        echo "  docker compose logs -f       # View live logs"
        echo "  docker compose restart       # Restart after config change"
        echo "  docker compose down          # Stop the application"
        echo "  docker compose up -d --build # Rebuild after code change"
        exit 0
    fi
    printf "."
    sleep 2
done

echo ""
log_error "Application failed to start within 60 seconds."
log_error "Check logs with: docker compose logs"
exit 1
