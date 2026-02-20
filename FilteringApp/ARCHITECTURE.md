# Architecture & Design Document

## 1. System Overview

The Data Filtering Platform replaces a manual, email-driven workflow for delivering filtered slices of ML production run data. It provides both a web UI for self-service users and a REST API for programmatic access.

```
User Request (Web UI or API)
    │
    ├── 1. Upload part list (.csv / .xlsx)
    ├── 2. Select datasets (shopping cart)
    ├── 3. Configure date filters (optional)
    │
    ▼
┌────────────────────────────────┐
│        FastAPI Backend         │
│                                │
│  ValidationService             │  ← Strict schema enforcement
│       │                        │
│  CatalogService                │  ← YAML-driven dataset registry
│       │                        │
│  FilterService                 │  ← Strategy pattern per file type
│       │                        │
│  DeliveryService               │  ← ZIP packaging + quality manifest
│                                │
└────────────────────────────────┘
    │
    ▼
ZIP download with filtered CSVs + manifest
```

## 2. Key Design Decisions

### 2.1 YAML-Driven Catalog (Extensibility)

The single most important design decision is the **YAML file catalog** (`file_catalog.yaml`). Every filterable dataset is declared here with its metadata: column names, filter strategy, key mode, date support, etc.

**Why this matters for future modules:** When new output files are added from the `data_proposal.py` pipeline (e.g., a new "actuals_and_forecasts" module), you only need to add a YAML entry — zero code changes. This is a direct response to requirement #10.

### 2.2 Strategy Pattern for Filtering

The original `FilterClasses.py` had filtering logic embedded in a class hierarchy with method names tied to file types. The new design uses the **Strategy Pattern**:

```
BaseFilterStrategy (ABC)
    ├── ModelPartIdStrategy       # Files with composite 'model_part_id'
    ├── PartOrpSeparatedStrategy  # Files with separate part_number + orp_code columns
    ├── ImpactStrategy            # IMPACT files with various part_number_* columns
    ├── CH013Strategy             # CH013 with non-standard column names
    └── RemoveDuplicatesStrategy  # Dedup-only files (e.g., PPTS_VENDOR_QUOTE)
```

Each dataset in the YAML catalog specifies which strategy to use. Adding a new strategy for a new file format requires only:
1. Implementing a new `BaseFilterStrategy` subclass
2. Registering it in `STRATEGY_MAP`
3. Referencing it in `file_catalog.yaml`

### 2.3 API-First Architecture

Every operation the web UI performs goes through the same API endpoints. This means:
- The REST API is a first-class citizen, not an afterthought.
- You can use `pandas` + `requests` to consume the API directly (see API_REFERENCE.md).
- Future integrations (email automation, scheduled jobs, CI/CD pipelines) can use the same API.

### 2.4 Quality Manifest

Every delivery ZIP includes a `_manifest.json` with per-file metrics:
- Source row count vs. filtered row count
- Match rate percentage
- Date filter applied (yes/no + range)
- Status flags: ok, warning, empty, error

This directly addresses requirement #5 (quality measurements and delivery summary).

### 2.5 Docker-First Deployment

Single `docker compose up` deploys the entire stack. The data directory is mounted read-only for safety. Resource limits are configured for the 4-CPU / 16GB RAM VM.

## 3. Directory Structure

```
data-filtering-platform/
├── README.md                       # Quick start guide
├── Dockerfile                      # Container build
├── docker-compose.yaml             # Orchestration
├── .env.example                    # Configuration template
│
├── backend/
│   ├── requirements.txt
│   ├── file_catalog.yaml           # ★ Dataset registry
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── core/
│   │   │   ├── config.py           # Settings from env vars
│   │   │   └── exceptions.py       # Domain exception hierarchy
│   │   ├── schemas/
│   │   │   └── __init__.py         # Pydantic models
│   │   ├── services/
│   │   │   ├── catalog_service.py  # YAML catalog → DatasetInfo
│   │   │   ├── validation_service.py # Strict input validation
│   │   │   ├── filter_service.py   # Strategy-based filtering
│   │   │   └── delivery_service.py # ZIP packaging + manifest
│   │   └── api/
│   │       ├── routes.py           # REST API endpoints
│   │       └── web_routes.py       # Server-rendered UI
│   └── tests/
│       └── test_services.py
│
├── frontend/
│   ├── templates/
│   │   ├── base.html               # Layout template
│   │   └── index.html              # Main page with shopping cart
│   └── static/
│       ├── css/styles.css
│       └── js/app.js               # Client-side logic
│
├── scripts/
│   └── deploy.sh                   # Rocky Linux deployment script
│
└── docs/
    ├── API_REFERENCE.md
    └── ARCHITECTURE.md             # This file
```

## 4. Data Flow

```
1. User uploads parts.csv
       │
2. ValidationService
   - Parse CSV/XLSX
   - Check columns ∈ {[part_number, orp_code], [part_number]}
   - Normalize types (strip .0 from orp_code)
   - Remove duplicates
   - Return PartsFileValidation
       │
3. User selects datasets (shopping cart)
   - CatalogService provides DatasetInfo list from YAML
   - UI disables incompatible datasets (e.g., part_orp datasets
     when file only has part_number)
       │
4. For each selected dataset:
   a. CatalogService.resolve_file_path() → find CSV in prod_run dir
   b. FilterService.filter_dataset()
      - Select strategy from STRATEGY_MAP[dataset.filter_strategy]
      - Apply key-based filter
      - Apply date filter if requested and supported
   c. Collect (DataFrame, source_rows, date_filter, error)
       │
5. DeliveryService.create_package()
   - Write filtered CSVs into ZIP organized by category
   - Build FileDeliverySummary for each file
   - Build DeliveryManifest with totals
   - Include _manifest.json and _summary.txt in ZIP
   - Verify size < MAX_DELIVERY_SIZE_MB
       │
6. Return ZIP to user
```

## 5. Adding a New Module (Extensibility Guide)

When a new output type is added to the ML pipeline:

**Step 1:** Add entry to `file_catalog.yaml`:
```yaml
  - id: "new_module_output"
    display_name: "New Module - Output"
    description: "Description of what this file contains."
    category: "new_category"  # Or use existing category
    filename_pattern: "new_module_output"
    filter_columns: ["part_number", "orp_code"]
    filter_strategy: "part_orp_separated"  # Use existing strategy
    key_mode: "part_orp"
    date_filterable: true
    date_column: "year_month"
    date_format: "yyyymm"
```

**Step 2:** If the file has a new format not covered by existing strategies, add a new strategy in `filter_service.py`:
```python
class NewFormatStrategy(BaseFilterStrategy):
    def apply(self, data, parts, dataset):
        # Your filtering logic here
        ...

# Register in STRATEGY_MAP
STRATEGY_MAP[FilterStrategy.NEW_FORMAT] = NewFormatStrategy
```

**Step 3:** Restart the container:
```bash
docker compose restart
```

No frontend changes needed — the YAML catalog drives the UI automatically.

## 6. Future Roadmap

| Phase | Feature | Effort |
|-------|---------|--------|
| 2 | Database for request history and audit trail | Medium |
| 2 | Integrate remaining `data_proposal.py` modules (uncompress, merge, actuals) | Low per module |
| 3 | Email-triggered requests (watch inbox → auto-process) | Medium |
| 3 | User authentication | Medium |
| 4 | Scheduled data freshness checks | Low |
| 4 | Web-based prod run management (upload, switch active run) | Medium |
