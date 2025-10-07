import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from db import Database
from auth import hash_password, verify_password, create_jwt
from dotenv import load_dotenv
import logging
import json
from sql_agent import (
	configure_gemini_from_env,
	run_data_raw_agent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting FYP Backend")


load_dotenv()

app = FastAPI(title="FYP Backend")


class RegisterRequest(BaseModel):
	email: EmailStr
	password: str
	full_name: str | None = None


class LoginRequest(BaseModel):
	email: EmailStr
	password: str


class QueryRequest(BaseModel):
	query: str


def get_db() -> Database:
	return Database.get_instance()


def ensure_users_table(db: Database) -> None:
	db.execute(
		"""
		CREATE TABLE IF NOT EXISTS users (
			id BIGINT AUTO_INCREMENT PRIMARY KEY,
			email VARCHAR(255) NOT NULL UNIQUE,
			password_hash VARCHAR(255) NOT NULL,
			full_name VARCHAR(255) NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
		"""
	)


@app.on_event("startup")
def on_startup():
	db = Database.get_instance()
	ensure_users_table(db)
	configure_gemini_from_env()



@app.post("/auth/register")
def register(payload: RegisterRequest, db: Database = Depends(get_db)):
	# check if user exists
	logger.info(f"Registering user {payload.email}")
	existing = db.execute("SELECT id FROM users WHERE email=%s", (payload.email,))
	if existing:
		logger.info(f"User {payload.email} already exists")
		raise HTTPException(status_code=409, detail="User already exists")

	password_hash = hash_password(payload.password)
	logger.info(f"Hashing password for user {payload.email}")
	db.execute(
		"INSERT INTO users (email, password_hash, full_name) VALUES (%s, %s, %s)",
		(payload.email, password_hash, payload.full_name),
	)
	user = db.execute("SELECT id, email, full_name FROM users WHERE email=%s", (payload.email,))
	user = user[0]
	token = create_jwt({"sub": str(user["id"]), "email": user["email"]}, expires_in_seconds=int(os.getenv("JWT_EXPIRES_IN", "3600")))
	return {"access_token": token, "token_type": "bearer", "user": user}


@app.post("/auth/login")
def login(payload: LoginRequest, db: Database = Depends(get_db)):
	user = db.execute("SELECT id, email, full_name, password_hash FROM users WHERE email=%s", (payload.email,))
	if not user:
		raise HTTPException(status_code=401, detail="Invalid credentials")
	user = user[0]
	if not verify_password(payload.password, user["password_hash"]):
		raise HTTPException(status_code=401, detail="Invalid credentials")
	token = create_jwt({"sub": str(user["id"]), "email": user["email"]}, expires_in_seconds=int(os.getenv("JWT_EXPIRES_IN", "3600")))
	return {"access_token": token, "token_type": "bearer", "user": {"id": user["id"], "email": user["email"], "full_name": user["full_name"]}}



@app.post("/query")
def query_route(payload: QueryRequest, db: Database = Depends(get_db)):
	try:
		return run_data_raw_agent(db, payload.query)
	except ValueError as ve:
		raise HTTPException(status_code=400, detail=str(ve))
	except Exception as exc:
		logger.exception("/query failed")
		raise HTTPException(status_code=500, detail="Failed to process query") from exc

