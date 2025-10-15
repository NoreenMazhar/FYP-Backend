import os
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, EmailStr, Field, model_validator
from db import Database
from auth import hash_password, verify_password, create_jwt
from dotenv import load_dotenv
import logging
import json
from datetime import datetime, date, timedelta
from typing import List, Optional
from sql_agent import (
	configure_gemini_from_env,
	run_data_raw_agent,
	discover_database_schema,
	get_schema_summary,
)
from Anomaly_Detection import detect_anomalies, get_anomaly_summary
from plots import get_2d_plots_via_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting FYP Backend")


load_dotenv()

app = FastAPI(title="FYP Backend")


class RegisterRequest(BaseModel):
	email: EmailStr
	password: str
	display_name: str | None = None
	full_name: str | None = None  # For backward compatibility
	user_type: str = Field(default="user", pattern="^(Streamer|admin|user)$")
	
	@model_validator(mode='after')
	def validate_name_fields(self):
		# Handle backward compatibility: if full_name is provided but display_name is not, use full_name
		if self.full_name and not self.display_name:
			self.display_name = self.full_name
		# Ensure display_name is not None (required field)
		if not self.display_name:
			raise ValueError("Either display_name or full_name must be provided")
		return self

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
			email VARCHAR(255) UNIQUE NOT NULL,
			display_name VARCHAR(120) NOT NULL,
			user_type VARCHAR(16) NOT NULL CHECK (user_type IN ('Streamer','admin','user')),
			password_hash TEXT NOT NULL,
			is_active BOOLEAN NOT NULL DEFAULT TRUE,
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
		"""
	)

def ensure_anomalies_table(db: Database) -> None:
	db.execute(
		"""
		CREATE TABLE IF NOT EXISTS anomalies (
			id BIGINT AUTO_INCREMENT PRIMARY KEY,
			anomaly_type VARCHAR(100) NOT NULL,
			description TEXT NOT NULL,
			status VARCHAR(20) NOT NULL DEFAULT 'active',
			severity VARCHAR(20) NOT NULL DEFAULT 'medium',
			device_id VARCHAR(100),
			icon VARCHAR(50),
			details JSON NOT NULL DEFAULT ('{}'),
			detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			resolved_at TIMESTAMP NULL,
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
			
			INDEX idx_anomaly_type (anomaly_type),
			INDEX idx_status (status),
			INDEX idx_device_id (device_id),
			INDEX idx_detected_at (detected_at),
			INDEX idx_severity (severity)
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
		"""
	)


@app.on_event("startup")
def on_startup():
	db = Database.get_instance()
	ensure_users_table(db)
	ensure_anomalies_table(db)
	configure_gemini_from_env()
	# Ensure visualizations table exists (aligns with SQL/basic.sql)
	db.execute(
		"""
		CREATE TABLE IF NOT EXISTS visualizations (
			id BIGINT AUTO_INCREMENT PRIMARY KEY,
			title VARCHAR(255) NOT NULL,
			viz_type VARCHAR(64) NOT NULL,
			config JSON NOT NULL,
			created_by BIGINT NOT NULL,
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
		"""
	)


@app.get("/plots/2d")
def get_2d_plots(
		start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
		end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
		device: Optional[str] = Query(None, description="Filter by device name"),
		vehicle_type: Optional[str] = Query(None, description="Filter by vehicle type"),
		db: Database = Depends(get_db),
		created_by: Optional[int] = Query(None, description="If provided, save visualizations under this user id")
	):
	"""
	Return multiple 2D-ready datasets using the SQL Agent for query generation.
	"""
	try:
		return get_2d_plots_via_agent(start_date, end_date, device, vehicle_type, db, created_by)
	except Exception as exc:
		logger.exception("Failed to generate 2D plots via agent")
		raise HTTPException(status_code=500, detail="Failed to generate 2D plots") from exc

@app.post("/auth/register")
def register(payload: RegisterRequest, db: Database = Depends(get_db)):
	try:
		# check if user exists
		logger.info(f"Registering user {payload.email}")
		existing = db.execute("SELECT id FROM users WHERE email=%s", (payload.email,))
		if existing:
			logger.info(f"User {payload.email} already exists")
			raise HTTPException(status_code=409, detail="User already exists")

		password_hash = hash_password(payload.password)
		logger.info(f"Hashing password for user {payload.email}")
		db.execute(
			"INSERT INTO users (email, password_hash, display_name, user_type) VALUES (%s, %s, %s, %s)",
			(payload.email, password_hash, payload.display_name, payload.user_type),
		)
		user = db.execute("SELECT id, email, display_name, user_type FROM users WHERE email=%s", (payload.email,))
		user = user[0]
		token = create_jwt({"sub": str(user["id"]), "email": user["email"]}, expires_in_seconds=int(os.getenv("JWT_EXPIRES_IN", "3600")))
		return {"access_token": token, "token_type": "bearer", "user": user}
	except ValueError as ve:
		raise HTTPException(status_code=422, detail=str(ve))
	except Exception as e:
		logger.exception(f"Registration failed for {payload.email}")
		raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login")
def login(payload: LoginRequest, db: Database = Depends(get_db)):
	user = db.execute("SELECT id, email, display_name, user_type, password_hash FROM users WHERE email=%s", (payload.email,))
	if not user:
		raise HTTPException(status_code=401, detail="Invalid credentials")
	user = user[0]
	if not verify_password(payload.password, user["password_hash"]):
		raise HTTPException(status_code=401, detail="Invalid credentials")
	token = create_jwt({"sub": str(user["id"]), "email": user["email"]}, expires_in_seconds=int(os.getenv("JWT_EXPIRES_IN", "3600")))
	return {"access_token": token, "token_type": "bearer", "user": {"id": user["id"], "email": user["email"], "display_name": user["display_name"], "user_type": user["user_type"]}}

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
		
		# Query to extract vehicle detection data from simplified data_raw table
		query = """
		SELECT 
			id,
			local_timestamp as timestamp,
			device_name as device,
			direction,
			vehicle_type,
			CAST(SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', 1) AS DECIMAL(10,9)) as type_score,
			SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) as license_plate,
			ocr_score
		FROM data_raw 
		WHERE DATE(local_timestamp) BETWEEN %s AND %s
		ORDER BY local_timestamp DESC
		LIMIT 100
		"""
		
		# Convert dates to string format for comparison with local_timestamp
		start_str = start_datetime.strftime('%Y-%m-%d')
		end_str = end_datetime.strftime('%Y-%m-%d')
		
		results = db.execute(query, (start_str, end_str))
		
		if not results:
			return VehicleDetectionsResponse(detections=[], total_count=0)
		
		detections = []
		for row in results:
			try:
				# Parse timestamp from local_timestamp field
				timestamp_str = row['timestamp']
				if timestamp_str:
					# Remove quotes if present
					timestamp_str = str(timestamp_str).strip('"')
					try:
						# Parse the standard format: YYYY-MM-DD HH:MM:SS
						timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
					except ValueError:
						try:
							# Try parsing with different formats
							timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
						except ValueError:
							try:
								# Try parsing the truncated format
								timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H')
							except ValueError:
								# Use current time as fallback
								timestamp = datetime.now()
				else:
					timestamp = datetime.now()
				
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
		result = run_data_raw_agent(payload.query)
		
		# Check if query was rejected for not being database-related
		if result.get("error") == "Question not database-related":
			# Return 400 Bad Request with clear message
			raise HTTPException(
				status_code=400, 
				detail={
					"message": "The query you asked is not related to the database, so I can't answer it.",
					"hint": "Please ask questions related to vehicle detections, license plates, devices, timestamps, or traffic data.",
					"result": result
				}
			)
		
		return result
	except HTTPException:
		# Re-raise HTTP exceptions
		raise
	except ValueError as ve:
		raise HTTPException(status_code=400, detail=str(ve))
	except Exception as exc:
		logger.exception("/query failed")
		raise HTTPException(status_code=500, detail="Failed to process query") from exc


@app.get("/schema")
def get_database_schema(summary_only: bool = Query(False, description="Return only a summary instead of full schema")):
	"""
	Get the discovered database schema including tables, columns, and relationships.
	This endpoint shows what tables and relationships the SQL agent has discovered.
	"""
	try:
		if summary_only:
			# Return just a summary of the schema
			summary = get_schema_summary()
			return {
				"summary": summary,
				"message": "Schema summary retrieved successfully. The SQL agent uses this information to intelligently select relevant tables for queries."
			}
		else:
			# Return full schema information
			schema = discover_database_schema()
			return {
				"schema": schema,
				"message": "Full schema retrieved successfully. The SQL agent uses this information to understand table relationships and select relevant tables for queries."
			}
	except Exception as exc:
		logger.exception("/schema failed")
		raise HTTPException(status_code=500, detail="Failed to retrieve schema information") from exc


@app.get("/anomalies")
async def get_anomalies(db: Database = Depends(get_db)):
	"""
	Get all detected anomalies from the database.
	Returns comprehensive anomaly detection results including active and resolved anomalies.
	"""
	try:
		logger.info("Fetching anomalies from database...")
		
		# Get all anomalies from database
		anomalies = db.execute("""
			SELECT 
				id, anomaly_type, description, status, severity, device_id, 
				icon, details, detected_at, resolved_at, created_at, updated_at
			FROM anomalies 
			ORDER BY detected_at DESC
		""")
		
		if not anomalies:
			return {
				"anomalies": [],
				"active_count": 0,
				"total_count": 0,
				"detection_time": datetime.now().isoformat()
			}
		
		# Convert database results to the expected format
		formatted_anomalies = []
		active_count = 0
		
		for anomaly in anomalies:
			# Parse details JSON
			details = {}
			if anomaly['details']:
				try:
					details = json.loads(anomaly['details']) if isinstance(anomaly['details'], str) else anomaly['details']
				except:
					details = {}
			
			formatted_anomaly = {
				"type": anomaly['anomaly_type'],
				"description": anomaly['description'],
				"status": anomaly['status'],
				"severity": anomaly['severity'],
				"device_id": anomaly['device_id'],
				"icon": anomaly['icon'],
				"details": details,
				"timestamp": anomaly['detected_at'].isoformat() if anomaly['detected_at'] else None
			}
			
			formatted_anomalies.append(formatted_anomaly)
			if anomaly['status'] == 'active':
				active_count += 1
		
		return {
			"anomalies": formatted_anomalies,
			"active_count": active_count,
			"total_count": len(formatted_anomalies),
			"detection_time": datetime.now().isoformat()
		}
		
	except Exception as exc:
		logger.exception("Failed to get anomalies")
		raise HTTPException(status_code=500, detail="Failed to retrieve anomalies") from exc


@app.get("/anomalies/summary")
async def get_anomalies_summary(db: Database = Depends(get_db)):
	"""
	Get a summary of current anomaly status from the database.
	Returns active count, resolved count, and last detection time.
	"""
	try:
		logger.info("Fetching anomaly summary from database...")
		
		# Get counts from database
		stats = db.execute("""
			SELECT 
				COUNT(*) as total_count,
				SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_count,
				SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_count,
				MAX(detected_at) as last_detection
			FROM anomalies
		""")
		
		if stats and len(stats) > 0:
			stat = stats[0]
			return {
				"active_anomalies": stat['active_count'] or 0,
				"resolved_anomalies": stat['resolved_count'] or 0,
				"total_anomalies": stat['total_count'] or 0,
				"last_detection": stat['last_detection'].isoformat() if stat['last_detection'] else datetime.now().isoformat()
			}
		else:
			return {
				"active_anomalies": 0,
				"resolved_anomalies": 0,
				"total_anomalies": 0,
				"last_detection": datetime.now().isoformat()
			}
		
	except Exception as exc:
		logger.exception("Failed to get anomaly summary")
		raise HTTPException(status_code=500, detail="Failed to retrieve anomaly summary") from exc


@app.get("/anomalies/active")
async def get_active_anomalies(db: Database = Depends(get_db)):
	"""
	Get only active anomalies (excluding resolved ones) from the database.
	Useful for real-time monitoring dashboards.
	"""
	try:
		logger.info("Fetching active anomalies from database...")
		
		# Get only active anomalies from database
		anomalies = db.execute("""
			SELECT 
				id, anomaly_type, description, status, severity, device_id, 
				icon, details, detected_at, resolved_at, created_at, updated_at
			FROM anomalies 
			WHERE status = 'active'
			ORDER BY detected_at DESC
		""")
		
		if not anomalies:
			return {
				"active_anomalies": [],
				"active_count": 0,
				"detection_time": datetime.now().isoformat()
			}
		
		# Convert database results to the expected format
		formatted_anomalies = []
		
		for anomaly in anomalies:
			# Parse details JSON
			details = {}
			if anomaly['details']:
				try:
					details = json.loads(anomaly['details']) if isinstance(anomaly['details'], str) else anomaly['details']
				except:
					details = {}
			
			formatted_anomaly = {
				"type": anomaly['anomaly_type'],
				"description": anomaly['description'],
				"status": anomaly['status'],
				"severity": anomaly['severity'],
				"device_id": anomaly['device_id'],
				"icon": anomaly['icon'],
				"details": details,
				"timestamp": anomaly['detected_at'].isoformat() if anomaly['detected_at'] else None
			}
			
			formatted_anomalies.append(formatted_anomaly)
		
		return {
			"active_anomalies": formatted_anomalies,
			"active_count": len(formatted_anomalies),
			"detection_time": datetime.now().isoformat()
		}
		
	except Exception as exc:
		logger.exception("Failed to get active anomalies")
		raise HTTPException(status_code=500, detail="Failed to retrieve active anomalies") from exc


@app.post("/anomalies/detect")
async def run_anomaly_detection(db: Database = Depends(get_db)):
	"""
	Run anomaly detection and store results in the database.
	This route performs the actual detection and caches results for fast retrieval.
	"""
	try:
		logger.info("Running anomaly detection and storing results...")
		
		# Run the anomaly detection
		results = await detect_anomalies()
		
		# Clear existing anomalies to avoid duplicates
		db.execute("DELETE FROM anomalies")
		
		# Store each anomaly in the database
		anomalies_stored = 0
		for anomaly in results.get('anomalies', []):
			try:
				db.execute(
					"""
					INSERT INTO anomalies 
					(anomaly_type, description, status, severity, device_id, icon, details, detected_at)
					VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
					""",
					(
						anomaly.get('type', 'Unknown'),
						anomaly.get('description', ''),
						anomaly.get('status', 'active'),
						anomaly.get('severity', 'medium'),
						anomaly.get('device_id'),
						anomaly.get('icon'),
						json.dumps(anomaly.get('details', {})),
						datetime.now()
					)
				)
				anomalies_stored += 1
			except Exception as e:
				logger.warning(f"Failed to store anomaly: {e}")
				continue
		
		logger.info(f"Stored {anomalies_stored} anomalies in database")
		
		return {
			"message": "Anomaly detection completed and results stored",
			"anomalies_stored": anomalies_stored,
			"detection_time": results.get('detection_time'),
			"total_detected": len(results.get('anomalies', []))
		}
		
	except Exception as exc:
		logger.exception("Failed to run anomaly detection")
		raise HTTPException(status_code=500, detail="Failed to run anomaly detection") from exc

