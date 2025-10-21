## FYP-Backend

### Overview

This project provides a MySQL database (via Docker) and a FastAPI backend with user registration and login that returns a JWT.

### Prerequisites

- Docker and Docker Engine running
- Python 3.10+ installed
- Docker Model Engine (for AI model)

### 1) Setup AI Model with Docker

First, set up the Qwen2.5 AI model for SQL generation:

```powershell
docker model pull ai/qwen2.5:3B-Q4_K_M
docker model run -d --name qwen2.5-container -p 8080:8080 ai/qwen2.5:3B-Q4_K_M
```

### 2) Build and run MySQL with Docker

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

### 3) Create virtual environment and install dependencies

From the repository root in Windows PowerShell:

```powershell
cd Backend
python3 -m venv venv
source venv/bin/Activate
pip install -r requirements.txt
```

### 4) Configure environment variables

Set env vars so the backend can connect to the database and AI model:

```powershell
# Database Configuration
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_USER = "FYP-USER"
DB_PASSWORD = "FYP-PASS"
DB_NAME = "FYP-DB"

# JWT Configuration
JWT_SECRET = "key"
JWT_EXPIRES_IN = "3600"   # seconds (1 hour)
JWT_PASS = "key"  # used as password hashing salt

# AI Model Configuration
LOCAL_MODEL_URL = "http://localhost:8080"

# Google API (for additional features)
GOOGLE_API_KEY= <insert your API key here>
```

Tips:

- `DB_*` variables are used by the backend. They also fall back to `MYSQL_*` if set in your container environment.
- Change `JWT_SECRET` and `JWT_PASS` to secure values for non-development use.

### 5) Run the FastAPI backend

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

## Docker Model Management

### Starting the AI Model

```powershell
docker start qwen2.5-container
```

### Stopping the AI Model

```powershell
docker stop qwen2.5-container
```

### Checking Model Status

```powershell
docker logs qwen2.5-container
```

### Testing Model Integration

```powershell
cd Backend/docker
python test_local_model.py
```

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
