# API Reference

Base URL: `http://<your-vm-ip>:8080/api/v1`

Interactive docs: `http://<your-vm-ip>:8080/api/docs` (Swagger UI)

---

## Endpoints

### `GET /health`

Health check.

```bash
curl http://localhost:8080/api/v1/health
```

Response:
```json
{"status": "healthy", "version": "1.0.0", "active_prod_run": "prod_run_2026_02"}
```

---

### `GET /catalog`

Returns all available datasets with metadata.

```bash
curl http://localhost:8080/api/v1/catalog
```

Response: `CatalogResponse` with `datasets[]`, `categories[]`, and `active_prod_run`.

---

### `GET /prod-runs`

Lists available production run directories.

```bash
curl http://localhost:8080/api/v1/prod-runs
```

---

### `POST /validate`

Validate a parts file without running filters. Use this to check the file before submitting.

```bash
curl -X POST http://localhost:8080/api/v1/validate \
  -F "parts_file=@my_parts.csv"
```

Response:
```json
{
  "is_valid": true,
  "detected_mode": "part_orp",
  "row_count": 150,
  "unique_parts": 120,
  "unique_orps": 45,
  "errors": [],
  "warnings": ["Found 30 duplicate row(s). Duplicates will be removed."]
}
```

---

### `POST /filter`

**Main endpoint.** Upload a parts file, select datasets, and receive a ZIP.

**Parameters (multipart form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `parts_file` | File | Yes | CSV or XLSX with part keys |
| `selected_files` | string (JSON) | Yes | JSON array of dataset IDs |
| `date_filters` | string (JSON) | No | JSON object of date ranges per dataset |

**Example with curl:**
```bash
curl -X POST http://localhost:8080/api/v1/filter \
  -F "parts_file=@parts.csv" \
  -F 'selected_files=["model_input_forecasting","CH013"]' \
  -F 'date_filters={"CH013": {"start": 202101, "end": 202512}}' \
  --output delivery.zip
```

**Example with Python + pandas:**
```python
import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile

# Prepare request
with open("parts.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8080/api/v1/filter",
        files={"parts_file": ("parts.csv", f, "text/csv")},
        data={
            "selected_files": '["model_input_forecasting", "CH013"]',
            "date_filters": '{"CH013": {"start": 202101, "end": 202512}}'
        }
    )

response.raise_for_status()

# Extract and read CSVs
z = ZipFile(BytesIO(response.content))
for name in z.namelist():
    if name.endswith(".csv"):
        df = pd.read_csv(z.open(name))
        print(f"{name}: {len(df)} rows, columns: {list(df.columns)}")

# Read the manifest
import json
manifest = json.loads(z.read("_manifest.json"))
print(f"Delivered {manifest['files_delivered']} files, {manifest['total_rows_delivered']} total rows")
```

**Response headers:**
- `X-Request-Id`: Unique request identifier
- `X-Files-Delivered`: Number of files included
- `X-Total-Rows`: Total rows across all files

---

### `POST /filter/preview`

Dry-run the filter pipeline. Returns a summary without creating the ZIP. Same parameters as `/filter`.

```bash
curl -X POST http://localhost:8080/api/v1/filter/preview \
  -F "parts_file=@parts.csv" \
  -F 'selected_files=["model_input_forecasting"]'
```

---

## ZIP Package Contents

Every delivery ZIP contains:

```
data_delivery_{request_id}.zip
├── _manifest.json          # Machine-readable quality report
├── _summary.txt            # Human-readable summary
├── forecasting/
│   ├── forecasting-forecasts.csv
│   └── ...
├── model_input/
│   └── model_input_forecasting.csv
├── impact/
│   └── IMPACT_ACTIVE_PA.csv
└── other/
    └── CH013.csv
```

---

## Input File Requirements

Your parts file must follow one of these schemas exactly:

**Schema A – Part + ORP:**
```csv
part_number,orp_code
9340M39P01,513
4096706-373P03,511
```

**Schema B – Part only:**
```csv
part_number
9340M39P01
4096706-373P03
```

Rules:
- Column names must be exactly `part_number` and optionally `orp_code` (case-insensitive, spaces normalized).
- `part_number`: non-null string.
- `orp_code`: string or integer. Trailing `.0` is stripped automatically.
- Duplicate rows are removed with a warning.
- Empty rows are dropped.

---

## Error Responses

All errors return JSON:
```json
{
  "error": "ValidationError",
  "message": "Invalid column names in uploaded file.",
  "detail": "Found columns: ['bad_col']. Accepted: ['part_number', 'orp_code'] or ['part_number'] only."
}
```

HTTP status codes: `422` for validation errors, `404` for missing datasets, `500` for server errors.
