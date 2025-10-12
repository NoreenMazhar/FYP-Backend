## FYP-Backend

### Overview

This project provides a MySQL database (via Docker) and a FastAPI backend with user registration and login that returns a JWT.

### Prerequisites

- Docker and Docker Engine running
- Python 3.10+ installed

### 1) Build and run MySQL with Docker

From the repository root (so `SQL/*.sql` are in build context):

```powershell
docker build -t fyp-mysql .
```

```powershell
docker run --name fyp-mysql -e MYSQL_ROOT_PASSWORD=rootpass -e MYSQL_DATABASE=FYP-DB -e MYSQL_USER=FYP-USER -e MYSQL_PASSWORD=FYP-PASS -p 3306:3306 -v fyp_mysql_data:/var/lib/mysql fyp-mysql
```

Notes:

- All `.sql` files in `SQL/` run automatically on first startup (empty data dir); they run in alphabetical order.
- If you need to re-run init scripts, remove the volume or use a new one: `docker volume rm fyp_mysql_data`.

### 2) Create virtual environment and install dependencies

From the repository root in Windows PowerShell:

```powershell
cd Backend
python -m venv venv
venv/Scripts/Activate
pip install -r requirements.txt
```

### 3) Configure environment variables

Set env vars so the backend can connect to the database and issue JWTs.

```powershell
$env:DB_HOST = "127.0.0.1"
$env:DB_PORT = "3306"
$env:DB_USER = "FYP-USER"
$env:DB_PASSWORD = "FYP-PASS"
$env:DB_NAME = "FYP-DB"

$env:JWT_SECRET = "key"
$env:JWT_EXPIRES_IN = "3600"   # seconds (1 hour)
$env:JWT_PASS = "key"  # used as password hashing salt
```

Tips:

- `DB_*` variables are used by the backend. They also fall back to `MYSQL_*` if set in your container environment.
- Change `JWT_SECRET` and `JWT_PASS` to secure values for non-development use.

### 4) Run the FastAPI backend

Inserting the excels into the database

_Make sure to give proper path in the main_

```powershell
python DataInsertion.py
```

Start the API (ensure the MySQL container is running and accepting connections):

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On startup, the backend ensures a `users` table exists.

### 5) Test the endpoints

- Register:

```powershell
curl -X POST http://localhost:8000/auth/register ^
  -H "Content-Type: application/json" ^
  -d '{"email":"a@b.com","password":"secret","full_name":"Alice"}'
```

- Login:

```powershell
curl -X POST http://localhost:8000/auth/login ^
  -H "Content-Type: application/json" ^
  -d '{"email":"a@b.com","password":"secret"}'
```

A successful response returns an `access_token` (JWT) and basic user info.

### 6) Query endpoint (LLM-assisted SQL)

Route: `POST /query`

- Purpose: Natural language questions are translated to SQL and executed against the database.
- Table scope behavior:
  - If the question is about devices (e.g., mentions "device", "camera", "firmware", "telemetry", etc.), the agent queries only device tables like `devices`, `device_models`, `device_streams`, `device_health`, and related.
  - Otherwise, it queries only the `data_raw` table and uses MySQL JSON functions for fields inside `row_data`.
- Output is a structured Markdown answer with sections: Overview, Key Findings, SQL Used, Observations, In-Depth Analysis (when warranted), Next Steps.

Requirements:

- Environment variable `GOOGLE_API_KEY` must be set (Gemini). If not set, LLM features are disabled.

Request body:

```json
{
  "query": "Which camera models have the most offline events in the last 7 days?"
}
```

Response body (example):

````json
{
  "question": "Which camera models have the most offline events in the last 7 days?",
  "executed_sql": "SELECT ...",
  "result": "### Overview\n- ... structured answer ...\n### Key Findings\n- ...\n### SQL Used\n```sql\nSELECT ...\n```\n### Observations\n- ...\n### In-Depth Analysis (when requested or warranted)\n- ...\n### Next Steps\n- ...",
  "used_device_scope": true
}
````

Notes:

- `used_device_scope` indicates whether the device schema was used. If `false`, the query was answered from `data_raw`.
- `executed_sql` may be `null` if the agent didn’t need to run a final SQL statement (rare) or if it couldn’t be extracted.

Example call (PowerShell):

```powershell
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d '{"query":"List top 5 devices by error events this month"}'
```

### 7) Anomaly Detection (Optimized)

The system includes optimized anomaly detection with database caching for improved performance:

#### Detection Routes

- `POST /anomalies/detect` - Run anomaly detection and store results in database
- `GET /anomalies` - Get all anomalies from database (fast)
- `GET /anomalies/summary` - Get anomaly statistics from database (fast)
- `GET /anomalies/active` - Get only active anomalies from database (fast)

#### Performance Benefits

- **Detection**: Run once and cache results (5-30 seconds)
- **Retrieval**: Fast database queries (< 100ms)
- **Concurrent requests**: Can handle many simultaneous requests
- **Resource efficiency**: Low CPU/memory for retrieval requests

#### Usage Example

```powershell
# Run detection and store results
curl -X POST http://localhost:8000/anomalies/detect

# Get all anomalies (fast)
curl -X GET http://localhost:8000/anomalies

# Get only active anomalies (fast)
curl -X GET http://localhost:8000/anomalies/active
```

For detailed information about the anomaly detection optimization, see [ANOMALY_OPTIMIZATION_README.md](ANOMALY_OPTIMIZATION_README.md).
