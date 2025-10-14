## FYP-Backend

### Overview

This project provides a complete Docker-based setup with MySQL database, FastAPI backend, and local AI models (Ollama) for SQL generation and answer summarization. No external API keys required - everything runs locally on CPU.

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM (for running local AI models)
- 10GB+ free disk space (for AI models)

### Quick Start (Recommended)

Run database and AI models in Docker, backend locally:

```powershell
# 1. Start only MySQL and Ollama in Docker
docker compose up -d db ollama

# 2. Pull the required AI models
docker exec -it ollama ollama pull qwen2.5-coder:7b-q4_K_M
docker exec -it ollama ollama pull phi3.5:mini-instruct-q4_K_M

# 3. Set up Python environment and run backend locally
cd Backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt

# 4. Set environment variables
set DB_HOST=localhost
set DB_PORT=3306
set DB_USER=FYP-USER
set DB_PASSWORD=FYP-PASS
set DB_NAME=FYP-DB
set OLLAMA_BASE_URL=http://localhost:11434/v1
set OLLAMA_SQL_MODEL=qwen2.5-coder:7b-q4_K_M
set OLLAMA_ANSWER_MODEL=phi3.5:mini-instruct-q4_K_M

# 5. Insert sample data (optional)
python DataInsertion.py

# 6. Run the backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Services

- **MySQL Database**: `localhost:3306` (FYP-DB) - Docker
- **Ollama AI Models**: `http://localhost:11434` (OpenAI-compatible API) - Docker
- **FastAPI Backend**: `http://localhost:8000` - Local Python

### Alternative: Full Docker Setup

If you prefer everything in Docker:

```powershell
# Start all services including backend
docker compose up -d

# Pull models
docker exec -it ollama ollama pull qwen2.5-coder:7b-q4_K_M
docker exec -it ollama ollama pull phi3.5:mini-instruct-q4_K_M

# Insert sample data
docker exec -it backend python DataInsertion.py
```

**Note**: The backend will be available at `http://localhost:8000` but you won't have hot-reload for development.

### Testing the Setup

#### 1) Test Authentication

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

#### 2) Test AI-Powered Query Endpoint

Route: `POST /query`

- **Purpose**: Natural language questions are translated to SQL using local AI models and executed against the database.
- **AI Models Used**:
  - **SQL Generation**: `qwen2.5-coder:7b-q4_K_M` (code-focused model for SQL)
  - **Answer Summarization**: `phi3.5:mini-instruct-q4_K_M` (lightweight model for responses)
- **No External APIs**: Everything runs locally on CPU

Request body:

```json
{
  "query": "How many vehicles were detected by Device-A1 yesterday?"
}
```

Response body (example):

```json
{
  "question": "How many vehicles were detected by Device-A1 yesterday?",
  "executed_sql": "SELECT COUNT(*) FROM data_raw WHERE device_name = 'Device-A1' AND DATE(local_timestamp) = CURDATE() - INTERVAL 1 DAY",
  "result": {
    "Overview": "Device-A1 detected 15 vehicles yesterday",
    "Key Findings": "- Total detections: 15\n- Device: Device-A1\n- Date: Yesterday",
    "SQL Used": "SELECT COUNT(*) FROM data_raw WHERE device_name = 'Device-A1' AND DATE(local_timestamp) = CURDATE() - INTERVAL 1 DAY",
    "Observations": "This shows the daily traffic volume for Device-A1",
    "Possible Questions": [
      "What was the busiest hour?",
      "Which vehicle types were most common?",
      "How does this compare to other devices?"
    ]
  },
  "used_device_scope": false
}
```

Example calls (PowerShell):

```powershell
# Vehicle detection query
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d '{"query":"Show me all trucks detected in the last hour"}'

# Device performance query
curl -X POST http://localhost:8000/query ^
  -H "Content-Type: application/json" ^
  -d '{"query":"Which devices have the highest OCR confidence scores?"}'
```

#### 3) Check AI Model Status

```powershell
# List available models in Ollama
curl http://localhost:11434/api/tags

# Test Ollama directly
curl http://localhost:11434/api/chat -d '{"model":"phi3.5:mini-instruct-q4_K_M","messages":[{"role":"user","content":"Hello"}]}'
```

### Anomaly Detection (Optimized)

The system includes optimized anomaly detection with database caching for improved performance:

#### Detection Routes

- `POST /anomalies/detect` - Run anomaly detection and store results in database
- `GET /anomalies` - Get all anomalies from database (fast)
- `GET /anomalies/summary` - Get anomaly statistics from database (fast)
- `GET /anomalies/active` - Get only active anomalies from database (fast)

### Automatic Schema Discovery

The SQL agent automatically discovers database tables and their relationships instead of using hardcoded table lists. This makes the system more flexible and adaptable to schema changes.

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
3. **Query Generation**: The local AI model generates SQL using only the selected tables with full schema context
4. **Relationship Awareness**: Foreign key relationships are included in the prompt to help with joins

### Stopping the Services

```powershell
# Stop Docker services (MySQL + Ollama)
docker compose down

# Stop and remove volumes (WARNING: deletes all data)
docker compose down -v

# View logs
docker compose logs ollama
docker compose logs db

# Stop local backend: Ctrl+C in the terminal running uvicorn
```

### Development Workflow

1. **Start services**: `docker compose up -d db ollama`
2. **Pull models**: Run the ollama pull commands once
3. **Develop backend**: Run `uvicorn main:app --reload` locally for hot-reload
4. **Test changes**: Backend automatically restarts when you save files
5. **Stop services**: `docker compose down` when done
