# Data Filtering Platform

A self-service web application and REST API for filtering and delivering slices of ML production run data. Built to replace manual email-driven data request workflows with an automated, auditable, and scalable platform.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     Frontend (Jinja2)                     │
│  Upload Parts → Shopping Cart → Date Filters → Download  │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼───────────────────────────────────┐
│                  FastAPI Backend                          │
│  ┌─────────┐  ┌───────────┐  ┌────────────────────────┐ │
│  │ Web UI  │  │  REST API  │  │  Background Workers    │ │
│  │ Routes  │  │  /api/v1   │  │  (filtering + zipping) │ │
│  └────┬────┘  └─────┬─────┘  └───────────┬────────────┘ │
│       └─────────────┼────────────────────┘              │
│              ┌──────▼──────┐                             │
│              │   Services  │                             │
│              │  - Catalog  │                             │
│              │  - Filter   │                             │
│              │  - Delivery │                             │
│              │  - Validate │                             │
│              └──────┬──────┘                             │
│              ┌──────▼──────┐                             │
│              │  File Store │                             │
│              │ /data/prod_ │                             │
│              │  run_{date} │                             │
│              └─────────────┘                             │
└──────────────────────────────────────────────────────────┘
```

## Design Decisions

### Modular Service Layer
The backend is organized around a **service layer pattern**:
- **CatalogService** – Reads the file catalog and exposes available datasets with metadata. New modules from `data_proposal.py` pipeline can register as new catalog entries.
- **ValidationService** – Strict input validation for uploaded part lists (schema, types, integrity).
- **FilterService** – Core filtering engine extracted from `FilterClasses.py`, refactored with strategy pattern for extensibility.
- **DeliveryService** – Packages filtered results into compressed ZIP archives with quality summaries.

### Extensibility for Future Modules
The catalog is defined in `file_catalog.yaml`. Adding a new filterable dataset requires only adding an entry to this YAML file — no code changes needed. This directly supports the goal of integrating more modules from `data_proposal.py` over time.

### API-First Design
Every operation available in the web UI is also available via the REST API (`/api/v1/`). This enables:
- Programmatic access via `pandas` + `requests`
- Future integrations with email triggers, scheduling, etc.

## Quick Start

### Prerequisites
- Rocky Linux VM with Docker and Docker Compose installed
- Data mounted at `/data` or configured via `.env`

### Deploy
```bash
git clone <repo-url> && cd data-filtering-platform
cp .env.example .env        # Edit paths as needed
docker compose up -d --build
```
The app will be available at `http://<vm-ip>:8080`.

### Development
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## API Usage (pandas example)

```python
import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile

# Upload a parts list and request filtered files
with open("parts.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8080/api/v1/filter",
        files={"parts_file": ("parts.csv", f, "text/csv")},
        data={
            "selected_files": '["model_input_forecasting","CH013"]',
            "date_filters": '{"CH013": {"start": 202101, "end": 202512}}'
        }
    )

# Response is a ZIP file
z = ZipFile(BytesIO(response.content))
for name in z.namelist():
    if name.endswith(".csv"):
        df = pd.read_csv(z.open(name))
        print(f"{name}: {len(df)} rows")
```

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for full API documentation.
