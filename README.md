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
python3 -m venv venv
source venv/bin/Activate
pip install -r requirements.txt
```

### 3) Configure environment variables

Set env vars so the backend can connect to the database and issue JWTs.

```powershell
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_USER = "FYP-USER"
DB_PASSWORD = "FYP-PASS"
DB_NAME = "FYP-DB"

JWT_SECRET = "key"
JWT_EXPIRES_IN = "3600"   # seconds (1 hour)
JWT_PASS = "key"  # used as password hashing salt
GOOGLE_API_KEY= <insert your API key here>
```

Tips:

- `DB_*` variables are used by the backend. They also fall back to `MYSQL_*` if set in your container environment.
- Change `JWT_SECRET` and `JWT_PASS` to secure values for non-development use.

### 4) Run the FastAPI backend

Inserting the excels into the database

_Make sure to give proper path in the main_

```powershell
python3 DataInsertion.py
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
  -d '{"email":"a@b.com","password":"secret","display_name":"Alice"}'
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

### Automatic Schema Discovery

The SQL agent now automatically discovers database tables and their relationships instead of using hardcoded table lists. This makes the system more flexible and adaptable to schema changes.

#### Key Features

- **Automatic Table Discovery**: Inspects the database at runtime to find all tables, columns, and constraints
- **Relationship Detection**: Automatically discovers foreign key relationships between tables
- **Smart Table Grouping**: Groups tables by domain (devices, data, reports, chats, etc.)
- **Intelligent Table Selection**: Automatically selects relevant tables based on the query context

#### Schema API Endpoints

```powershell
# Get full schema information
curl http://localhost:8000/schema

# Get schema summary only
curl http://localhost:8000/schema?summary_only=true
```

#### How It Works

1. **Schema Discovery**: On first query, the system inspects the database to discover all tables and relationships
2. **Table Selection**: For each query, the agent intelligently selects relevant tables based on keywords and relationships
3. **Query Generation**: The LLM generates SQL using only the selected tables with full schema context
4. **Relationship Awareness**: Foreign key relationships are included in the prompt to help with joins

#### Testing Schema Discovery

Run the test script to see the schema discovery in action:

```powershell
cd Backend
python test_schema_discovery.py
```

This will display:

- All discovered tables and their relationships
- Table groupings by domain
- Intelligent table selection examples
- Detailed information about key tables

For detailed information about the schema discovery feature, see [SCHEMA_DISCOVERY.md](SCHEMA_DISCOVERY.md).

## API Routes Reference

### Authentication Routes

| Route            | Method | Description         | Request Body                                                                                               | Response                                                               |
| ---------------- | ------ | ------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `/auth/register` | POST   | Register a new user | `{"email": "user@example.com", "password": "password", "display_name": "User Name", "user_type": "admin"}` | `{"access_token": "jwt_token", "token_type": "bearer", "user": {...}}` |
| `/auth/login`    | POST   | Login user          | `{"email": "user@example.com", "password": "password"}`                                                    | `{"access_token": "jwt_token", "token_type": "bearer", "user": {...}}` |

### User Management Routes

| Route           | Method | Description                          | Request Body                                          | Response                                                         |
| --------------- | ------ | ------------------------------------ | ----------------------------------------------------- | ---------------------------------------------------------------- |
| `/users/status` | PUT    | Update user status (active/inactive) | `{"email": "user@example.com", "status": true}`       | `{"message": "User status updated successfully", "user": {...}}` |
| `/users/type`   | PUT    | Update user type                     | `{"email": "user@example.com", "user_type": "admin"}` | `{"message": "User type updated successfully", "user": {...}}`   |

### Device Management Routes

| Route                            | Method | Description                           | Request Body                                                                                                       | Response                                                                                                   |
| -------------------------------- | ------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `/devices`                       | GET    | Get all devices list                  | None                                                                                                               | `{"devices": [...], "total_count": 5}`                                                                     |
| `/devices`                       | POST   | Add new device                        | `{"device_uid": "A1", "name": "Device-A1", "location": "North Gate", "device_type": "camera", "status": "active"}` | `{"message": "Device added successfully", "device": {...}}`                                                |
| `/devices/{device_id}`           | PUT    | Update device information             | `{"name": "New Name", "status": "active"}`                                                                         | `{"message": "Device updated successfully", "device": {...}}`                                              |
| `/devices/{device_id}`           | DELETE | Delete device                         | None                                                                                                               | `{"message": "Device deleted successfully", "deleted_device": {...}}`                                      |
| `/devices/{device_id}/details`   | GET    | Get detailed device info with metrics | None                                                                                                               | `{"id": 1, "device_uid": "A1", "name": "Device-A1", "status": "Online", "uptime": 99.9, "metrics": {...}}` |
| `/devices/{device_id}/metrics`   | GET    | Get device performance metrics        | None                                                                                                               | `{"device_id": 1, "device_name": "Device-A1", "metrics": {...}, "last_updated": "..."}`                    |
| `/devices/{device_id}/telemetry` | POST   | Add device telemetry data             | Query params: `metric_name`, `metric_value`, `metric_units`                                                        | `{"message": "Telemetry data added successfully", ...}`                                                    |

### Data Analysis Routes

| Route                 | Method | Description                | Request Body                                                                                               | Response                                                                                                   |
| --------------------- | ------ | -------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `/query`              | POST   | Natural language SQL query | `{"query": "Show me detections by hour"}`                                                                  | `{"question": "...", "executed_sql": "...", "result": "...", "used_device_scope": true}`                   |
| `/plots`              | GET    | Get 2D plot data           | Query params: `start_date`, `end_date`, `device`, `vehicle_type`                                           | `[{"Data": {"X": [...], "Y": [...]}, "X-axis-label": "...", "Y-axis-label": "...", "Description": "..."}]` |
| `/text-to-plots`      | POST   | Convert text to plot data  | `{"text_description": "Show me detections by hour", "start_date": "2024-01-01", "end_date": "2024-01-31"}` | `[{"Data": {"X": [...], "Y": [...]}, "Plot-type": "bar", ...}]`                                            |
| `/vehicle-detections` | GET    | Get vehicle detection data | Query params: `start_date`, `end_date`                                                                     | `{"detections": [...], "total_count": 100}`                                                                |

### Anomaly Detection Routes

| Route                | Method | Description               | Request Body | Response                                                                                            |
| -------------------- | ------ | ------------------------- | ------------ | --------------------------------------------------------------------------------------------------- |
| `/anomalies/detect`  | POST   | Run anomaly detection     | None         | `{"message": "Anomaly detection completed", "anomalies_stored": 15, "detection_time": "..."}`       |
| `/anomalies`         | GET    | Get all anomalies         | None         | `{"anomalies": [...], "active_count": 5, "total_count": 15, "detection_time": "..."}`               |
| `/anomalies/summary` | GET    | Get anomaly summary       | None         | `{"active_anomalies": 5, "resolved_anomalies": 10, "total_anomalies": 15, "last_detection": "..."}` |
| `/anomalies/active`  | GET    | Get only active anomalies | None         | `{"active_anomalies": [...], "active_count": 5, "detection_time": "..."}`                           |

### Schema Discovery Routes

| Route     | Method | Description         | Request Body                           | Response                                                             |
| --------- | ------ | ------------------- | -------------------------------------- | -------------------------------------------------------------------- |
| `/schema` | GET    | Get database schema | Query param: `summary_only` (optional) | `{"schema": {...}, "message": "Full schema retrieved successfully"}` |

### Data Models

#### User Types

- `admin` - Full system access
- `Analyst` - Data analysis capabilities
- `View` - Read-only access
- `Security` - Security monitoring access

#### Device Status

- `inactive` - Device is offline
- `active` - Device is online and operational
- `maintenance` - Device is under maintenance
- `decommissioned` - Device is no longer in use

#### Device Metrics

- `detections` - Number of vehicle detections
- `errors` - Number of error events
- `cpu_usage` - CPU utilization percentage
- `memory_usage` - Memory utilization percentage
- `storage_usage` - Storage utilization percentage
- `network_latency` - Network latency in milliseconds
- `temperature` - Device temperature in Celsius

### Example Usage

#### Register a new user

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"password123","display_name":"Admin User","user_type":"admin"}'
```

#### Get all devices

```bash
curl -X GET http://localhost:8000/devices
```

#### Get detailed device information

```bash
curl -X GET http://localhost:8000/devices/1/details
```

#### Add device telemetry

```bash
curl -X POST "http://localhost:8000/devices/1/telemetry?metric_name=cpu_usage&metric_value=45.5&metric_units=%"
```

#### Update user status

```bash
curl -X PUT http://localhost:8000/users/status \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","status":false}'
```

#### Run anomaly detection

```bash
curl -X POST http://localhost:8000/anomalies/detect
```

#### Query with natural language

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Show me the top 5 devices with highest error rates"}'
```
