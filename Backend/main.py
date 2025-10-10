import os
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, EmailStr
from db import Database
from auth import hash_password, verify_password, create_jwt
from dotenv import load_dotenv
import logging
import json
from datetime import datetime, date
from typing import List, Optional
from sql_agent import (
	configure_gemini_from_env,
	run_data_raw_agent,
)
from Anomaly_Detection import detect_anomalies, get_anomaly_summary

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

class VehicleDetection(BaseModel):
	timestamp: datetime
	device: str
	direction: str
	vehicle_type: str
	type_score: float
	license_plate: str
	ocr_score: float

class VehicleDetectionsResponse(BaseModel):
	detections: List[VehicleDetection]
	total_count: int


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

@app.get("/vehicle-detections", response_model=VehicleDetectionsResponse)
def get_vehicle_detections(
		start_date: date = Query(..., description="Start date for filtering detections (YYYY-MM-DD)"),
		end_date: date = Query(..., description="End date for filtering detections (YYYY-MM-DD)"),
		db: Database = Depends(get_db)
	):
	"""
	Get vehicle detections within a date range.
	Returns recent vehicle detections with timestamp, device, direction, vehicle type, 
	type score, license plate, and OCR score.
	"""
	try:
		# Convert dates to datetime for comparison
		start_datetime = datetime.combine(start_date, datetime.min.time())
		end_datetime = datetime.combine(end_date, datetime.max.time())
		
		# Query to extract vehicle detection data from data_raw table
		# Try multiple possible field names for flexibility
		query = """
		SELECT 
			COALESCE(
				JSON_EXTRACT(row_data, '$.timestamp'),
				JSON_EXTRACT(row_data, '$.Timestamp'),
				JSON_EXTRACT(row_data, '$.time'),
				JSON_EXTRACT(row_data, '$.Time'),
				JSON_EXTRACT(row_data, '$.detection_time'),
				JSON_EXTRACT(row_data, '$."Detection Time"')
			) as timestamp,
			COALESCE(
				JSON_EXTRACT(row_data, '$.device'),
				JSON_EXTRACT(row_data, '$.Device'),
				JSON_EXTRACT(row_data, '$.device_id'),
				JSON_EXTRACT(row_data, '$."Device ID"'),
				JSON_EXTRACT(row_data, '$.camera'),
				JSON_EXTRACT(row_data, '$.Camera')
			) as device,
			COALESCE(
				JSON_EXTRACT(row_data, '$.direction'),
				JSON_EXTRACT(row_data, '$.Direction'),
				JSON_EXTRACT(row_data, '$.movement'),
				JSON_EXTRACT(row_data, '$.Movement')
			) as direction,
			COALESCE(
				JSON_EXTRACT(row_data, '$.vehicle_type'),
				JSON_EXTRACT(row_data, '$."Vehicle Type"'),
				JSON_EXTRACT(row_data, '$.type'),
				JSON_EXTRACT(row_data, '$.Type'),
				JSON_EXTRACT(row_data, '$.class'),
				JSON_EXTRACT(row_data, '$.Class')
			) as vehicle_type,
			COALESCE(
				JSON_EXTRACT(row_data, '$.type_score'),
				JSON_EXTRACT(row_data, '$."Type Score"'),
				JSON_EXTRACT(row_data, '$.confidence'),
				JSON_EXTRACT(row_data, '$.Confidence'),
				JSON_EXTRACT(row_data, '$.detection_confidence')
			) as type_score,
			COALESCE(
				JSON_EXTRACT(row_data, '$.license_plate'),
				JSON_EXTRACT(row_data, '$."License Plate"'),
				JSON_EXTRACT(row_data, '$.plate'),
				JSON_EXTRACT(row_data, '$.Plate'),
				JSON_EXTRACT(row_data, '$.license'),
				JSON_EXTRACT(row_data, '$.License')
			) as license_plate,
			COALESCE(
				JSON_EXTRACT(row_data, '$.ocr_score'),
				JSON_EXTRACT(row_data, '$."OCR Score"'),
				JSON_EXTRACT(row_data, '$.plate_confidence'),
				JSON_EXTRACT(row_data, '$."Plate Confidence"'),
				JSON_EXTRACT(row_data, '$.text_confidence')
			) as ocr_score,
			imported_at,
			row_data
		FROM data_raw 
		WHERE imported_at BETWEEN %s AND %s
		AND (
			JSON_EXTRACT(row_data, '$.timestamp') IS NOT NULL OR
			JSON_EXTRACT(row_data, '$.Timestamp') IS NOT NULL OR
			JSON_EXTRACT(row_data, '$.time') IS NOT NULL OR
			JSON_EXTRACT(row_data, '$.Time') IS NOT NULL OR
			JSON_EXTRACT(row_data, '$.detection_time') IS NOT NULL OR
			JSON_EXTRACT(row_data, '$."Detection Time"') IS NOT NULL
		)
		ORDER BY imported_at DESC
		LIMIT 100
		"""
		
		results = db.execute(query, (start_datetime, end_datetime))
		
		if not results:
			return VehicleDetectionsResponse(detections=[], total_count=0)
		
		detections = []
		for row in results:
			try:
				# Parse timestamp
				timestamp_str = row['timestamp']
				if timestamp_str:
					# Remove quotes if present
					timestamp_str = timestamp_str.strip('"')
					try:
						timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
					except ValueError:
						# Try parsing with different formats
						try:
							timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
						except ValueError:
							timestamp = row['imported_at']
				else:
					timestamp = row['imported_at']
				
				# Helper function to safely extract string values
				def safe_string(value, default="Unknown"):
					if not value:
						return default
					return str(value).strip('"')
				
				# Helper function to safely extract numeric values
				def safe_float(value, default=0.0):
					if not value:
						return default
					try:
						return float(str(value).strip('"'))
					except (ValueError, TypeError):
						return default
				
				detection = VehicleDetection(
					timestamp=timestamp,
					device=safe_string(row['device'], "Unknown Device"),
					direction=safe_string(row['direction'], "Unknown"),
					vehicle_type=safe_string(row['vehicle_type'], "Unknown"),
					type_score=safe_float(row['type_score'], 0.0),
					license_plate=safe_string(row['license_plate'], "N/A"),
					ocr_score=safe_float(row['ocr_score'], 0.0)
				)
				detections.append(detection)
			except Exception as e:
				logger.warning(f"Skipping invalid row: {e}")
				continue
		
		return VehicleDetectionsResponse(
			detections=detections,
			total_count=len(detections)
		)
		
	except Exception as exc:
		logger.exception("Failed to get vehicle detections")
		raise HTTPException(status_code=500, detail="Failed to retrieve vehicle detections") from exc

@app.post("/query")
def query_route(payload: QueryRequest, db: Database = Depends(get_db)):
	try:
		return run_data_raw_agent(payload.query)
	except ValueError as ve:
		raise HTTPException(status_code=400, detail=str(ve))
	except Exception as exc:
		logger.exception("/query failed")
		raise HTTPException(status_code=500, detail="Failed to process query") from exc


@app.get("/anomalies")
async def get_anomalies():
	"""
	Get all detected anomalies in real-time.
	Returns comprehensive anomaly detection results including active and resolved anomalies.
	"""
	try:
		logger.info("Fetching anomaly detection results...")
		results = await detect_anomalies()
		return results
	except Exception as exc:
		logger.exception("Failed to get anomalies")
		raise HTTPException(status_code=500, detail="Failed to retrieve anomalies") from exc


@app.get("/anomalies/summary")
async def get_anomalies_summary():
	"""
	Get a summary of current anomaly status.
	Returns active count, resolved count, and last detection time.
	"""
	try:
		logger.info("Fetching anomaly summary...")
		summary = await get_anomaly_summary()
		return summary
	except Exception as exc:
		logger.exception("Failed to get anomaly summary")
		raise HTTPException(status_code=500, detail="Failed to retrieve anomaly summary") from exc


@app.get("/anomalies/active")
async def get_active_anomalies():
	"""
	Get only active anomalies (excluding resolved ones).
	Useful for real-time monitoring dashboards.
	"""
	try:
		logger.info("Fetching active anomalies...")
		results = await detect_anomalies()
		active_anomalies = [a for a in results['anomalies'] if a.get('status') == 'active']
		
		return {
			"active_anomalies": active_anomalies,
			"active_count": len(active_anomalies),
			"detection_time": results['detection_time']
		}
	except Exception as exc:
		logger.exception("Failed to get active anomalies")
		raise HTTPException(status_code=500, detail="Failed to retrieve active anomalies") from exc

