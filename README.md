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
python -m venv .venv
. .venv/Scripts/Activate.ps1
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

### Troubleshooting

- If connection fails, confirm the Docker container is running: `docker ps`
- Check logs: `docker logs fyp-mysql`
- If SQL init did not run, remove the volume and re-run the container as noted above.
