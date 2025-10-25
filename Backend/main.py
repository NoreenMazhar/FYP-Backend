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
	configure_ollama_from_env,
	run_data_raw_agent,
	discover_database_schema,
	get_schema_summary,
)
from Anomaly_Detection import detect_anomalies, get_anomaly_summary
from plots import get_2d_plots_via_agent, convert_text_to_plots
from report_generator import generate_comprehensive_report

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
	user_type: str = Field(default="View", pattern="^(admin|Analyst|View|Security)$")
	
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

class TextToPlotRequest(BaseModel):
	text_description: str
	start_date: Optional[date] = None
	end_date: Optional[date] = None
	device: Optional[str] = None
	vehicle_type: Optional[str] = None

class ReportRequest(BaseModel):
	start_date: date
	end_date: date
	title: Optional[str] = None
	description: Optional[str] = None

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

class UpdateUserStatusRequest(BaseModel):
	email: EmailStr
	status: bool  # True for active, False for inactive

class UpdateUserTypeRequest(BaseModel):
	email: EmailStr
	user_type: str = Field(pattern="^(admin|Analyst|View|Security)$")

class DeviceResponse(BaseModel):
	id: int
	device_uid: str
	name: str
	location: str | None = None
	status: str
	uptime: float | None = None
	last_sync: str | None = None
	device_type: str
	created_at: datetime
	updated_at: datetime

class DeviceListResponse(BaseModel):
	devices: List[DeviceResponse]
	total_count: int

class AddDeviceRequest(BaseModel):
	model_config = {"protected_namespaces": ()}
	
	device_uid: str
	name: str
	location: str | None = None
	device_type: str = Field(default="camera")
	model_id: int | None = None
	status: str = Field(default="inactive", pattern="^(inactive|active|maintenance|decommissioned)$")

class UpdateDeviceRequest(BaseModel):
	name: str | None = None
	location: str | None = None
	status: str | None = Field(None, pattern="^(inactive|active|maintenance|decommissioned)$")

class DeviceMetrics(BaseModel):
	detections: int
	errors: int
	cpu_usage: float
	memory_usage: float
	storage_usage: float

class DetailedDeviceResponse(BaseModel):
	id: int
	device_uid: str
	name: str
	location: str | None = None
	status: str
	uptime: float
	last_sync: str
	device_type: str
	metrics: DeviceMetrics
	created_at: datetime
	updated_at: datetime


def get_db() -> Database:
	return Database.get_instance()


def normalize_user_type(user_type: str) -> str:
	"""
	Normalize user_type to match the exact case expected by the database constraint.
	The database expects: 'admin', 'Analyst', 'View', 'Security'
	"""
	user_type_lower = user_type.lower()
	
	if user_type_lower == "admin":
		return "admin"
	elif user_type_lower == "analyst":
		return "Analyst"
	elif user_type_lower == "view":
		return "View"
	elif user_type_lower == "security":
		return "Security"
	else:
		# If it doesn't match any expected values, return the original
		# This will be caught by Pydantic validation
		return user_type


def ensure_users_table(db: Database) -> None:
	db.execute(
		"""
		CREATE TABLE IF NOT EXISTS users (
			id BIGINT AUTO_INCREMENT PRIMARY KEY,
			email VARCHAR(255) UNIQUE NOT NULL,
			display_name VARCHAR(120) NOT NULL,
			user_type VARCHAR(16) NOT NULL CHECK (user_type IN ('admin','Analyst','View','Security')),
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

def ensure_device_tables(db: Database) -> None:
	# Create device_models table
	db.execute(
		"""
		CREATE TABLE IF NOT EXISTS device_models (
			id BIGINT AUTO_INCREMENT PRIMARY KEY,
			vendor VARCHAR(120) NOT NULL,
			model_name VARCHAR(120) NOT NULL,
			device_type VARCHAR(32) NOT NULL,
			capabilities JSON NOT NULL DEFAULT ('{}'),
			datasheet_url VARCHAR(512),
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
		"""
	)
	
	# Create devices table
	db.execute(
		"""
		CREATE TABLE IF NOT EXISTS devices (
			id BIGINT AUTO_INCREMENT PRIMARY KEY,
			device_uid VARCHAR(128) UNIQUE NOT NULL,
			device_type VARCHAR(32) NOT NULL,
			model_id BIGINT NULL,
			name VARCHAR(255) NOT NULL,
			location_id BIGINT NULL,
			status VARCHAR(32) NOT NULL DEFAULT 'inactive',
			timezone VARCHAR(64),
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
			FOREIGN KEY (model_id) REFERENCES device_models(id) ON DELETE SET NULL
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
		"""
	)
	
	# Create device_health table for status tracking
	db.execute(
		"""
		CREATE TABLE IF NOT EXISTS device_health (
			id BIGINT AUTO_INCREMENT PRIMARY KEY,
			device_id BIGINT NOT NULL,
			health_status VARCHAR(32) NOT NULL,
			details JSON NOT NULL DEFAULT ('{}'),
			checked_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
		) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
		"""
	)


@app.on_event("startup")
def on_startup():
	db = Database.get_instance()
	ensure_users_table(db)
	ensure_anomalies_table(db)
	ensure_device_tables(db)
	configure_ollama_from_env()
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


@app.get("/plots")
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
	Returns a list of JSON objects with the following structure:
	[
		{
			"Data": {
				"X": [values],
				"Y": [values]
			},
			"X-axis-label": "string",
			"Y-axis-label": "string", 
			"Description": "brief explanation of plot"
		}
	]
	
	Supports line, bar, donut, and pie chart types.
	If start_date and end_date are not provided, queries the entire database.
	"""
	try:
		return get_2d_plots_via_agent(start_date, end_date, device, vehicle_type, db, created_by)
	except Exception as exc:
		logger.exception("Failed to generate 2D plots via agent")
		raise HTTPException(status_code=500, detail="Failed to generate 2D plots") from exc

@app.post("/text-to-plots")
def convert_text_to_plots_route(
		payload: TextToPlotRequest,
		db: Database = Depends(get_db),
		created_by: Optional[int] = Query(None, description="If provided, save visualizations under this user id")
	):
	"""
	Convert a text description to plot data using the SQL agent.
	
	This endpoint takes a natural language description of what kind of plot/analysis 
	the user wants and generates appropriate plot data by using the SQL agent to 
	create and execute relevant database queries.
	
	Returns a list of JSON objects with the following structure:
	[
		{
			"Data": {
				"X": [values],
				"Y": [values]
			},
			"Plot-type": "bar|line|pie|donut",
			"X-axis-label": "string",
			"Y-axis-label": "string", 
			"Description": "brief explanation of plot"
		}
	]
	
	Example text descriptions:
	- "Show me detections by hour of day"
	- "Average OCR score by device"
	- "Vehicle type distribution"
	- "Detections over time"
	- "Peak hours analysis"
	"""
	try:
		plots = convert_text_to_plots(
			text_description=payload.text_description,
			start_date=payload.start_date,
			end_date=payload.end_date,
			device=payload.device,
			vehicle_type=payload.vehicle_type
		)
		
		# Optionally persist each plot as a visualization
		if created_by is not None:
			conn = Database.get_instance()
			for plot in plots:
				try:
					viz_type = "chart"  # Default type
					title = f"Text-to-Plot: {payload.text_description[:50]}..."
					config = {
						"x": plot.get("Data", {}).get("X", []),
						"y": plot.get("Data", {}).get("Y", []),
						"description": plot.get("Description", ""),
						"x_axis_label": plot.get("X-axis-label", ""),
						"y_axis_label": plot.get("Y-axis-label", ""),
						"plot_type": plot.get("Plot-type", "bar"),
						"filters": {
							"start_date": payload.start_date.isoformat() if payload.start_date else None,
							"end_date": payload.end_date.isoformat() if payload.end_date else None,
							"device": payload.device,
							"vehicle_type": payload.vehicle_type,
						},
						"text_description": payload.text_description
					}
					conn.execute(
						"INSERT INTO visualizations (title, viz_type, config, created_by) VALUES (%s, %s, %s, %s)",
						(
							title,
							viz_type,
							json.dumps(config),
							int(created_by),
						),
					)
				except Exception as e:
					logger.warning(f"Failed to persist text-to-plot visualization: {e}")
		
		return plots
		
	except Exception as exc:
		logger.exception("Failed to convert text to plots")
		raise HTTPException(status_code=500, detail="Failed to convert text to plots") from exc

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
		
		# Normalize user_type to match database constraint case requirements
		normalized_user_type = normalize_user_type(payload.user_type)
		logger.info(f"Normalized user_type from '{payload.user_type}' to '{normalized_user_type}'")
		
		db.execute(
			"INSERT INTO users (email, password_hash, display_name, user_type) VALUES (%s, %s, %s, %s)",
			(payload.email, password_hash, payload.display_name, normalized_user_type),
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


@app.put("/users/status")
def update_user_status(payload: UpdateUserStatusRequest, db: Database = Depends(get_db)):
	"""
	Update user status (active/inactive) by email.
	"""
	try:
		logger.info(f"Updating status for user {payload.email} to {'active' if payload.status else 'inactive'}")
		
		# Check if user exists
		user = db.execute("SELECT id, email, display_name, user_type, is_active FROM users WHERE email=%s", (payload.email,))
		if not user:
			raise HTTPException(status_code=404, detail="User not found")
		
		user = user[0]
		
		# Update user status
		db.execute(
			"UPDATE users SET is_active=%s, updated_at=CURRENT_TIMESTAMP WHERE email=%s",
			(payload.status, payload.email)
		)
		
		logger.info(f"Successfully updated status for user {payload.email}")
		
		return {
			"message": f"User status updated successfully",
			"user": {
				"id": user["id"],
				"email": user["email"],
				"display_name": user["display_name"],
				"user_type": user["user_type"],
				"is_active": payload.status
			}
		}
		
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Failed to update user status for {payload.email}")
		raise HTTPException(status_code=500, detail="Failed to update user status") from exc


@app.put("/users/type")
def update_user_type(payload: UpdateUserTypeRequest, db: Database = Depends(get_db)):
	"""
	Update user type (admin/Analyst/View/Security) by email.
	"""
	try:
		logger.info(f"Updating user type for {payload.email} to {payload.user_type}")
		
		# Check if user exists
		user = db.execute("SELECT id, email, display_name, user_type, is_active FROM users WHERE email=%s", (payload.email,))
		if not user:
			raise HTTPException(status_code=404, detail="User not found")
		
		user = user[0]
		
		# Normalize user_type to match database constraint case requirements
		normalized_user_type = normalize_user_type(payload.user_type)
		logger.info(f"Normalized user_type from '{payload.user_type}' to '{normalized_user_type}'")
		
		# Update user type
		db.execute(
			"UPDATE users SET user_type=%s, updated_at=CURRENT_TIMESTAMP WHERE email=%s",
			(normalized_user_type, payload.email)
		)
		
		logger.info(f"Successfully updated user type for {payload.email}")
		
		return {
			"message": f"User type updated successfully",
			"user": {
				"id": user["id"],
				"email": user["email"],
				"display_name": user["display_name"],
				"user_type": payload.user_type,
				"is_active": user["is_active"]
			}
		}
		
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Failed to update user type for {payload.email}")
		raise HTTPException(status_code=500, detail="Failed to update user type") from exc


@app.get("/devices", response_model=DeviceListResponse)
def get_devices(db: Database = Depends(get_db)):
	"""
	Get all devices for the frontend dashboard.
	Returns device list with status, uptime, and last sync information.
	"""
	try:
		logger.info("Fetching devices for dashboard...")
		
		# Query devices with health information
		query = """
		SELECT 
			d.id,
			d.device_uid,
			d.name,
			d.device_type,
			d.status,
			d.created_at,
			d.updated_at,
			COALESCE(dh.health_status, 'offline') as health_status,
			COALESCE(dh.checked_at, d.updated_at) as last_checked,
			COALESCE(dh.details, '{}') as health_details
		FROM devices d
		LEFT JOIN device_health dh ON d.id = dh.device_id 
			AND dh.id = (
				SELECT MAX(id) FROM device_health 
				WHERE device_id = d.id
			)
		ORDER BY d.name
		"""
		
		results = db.execute(query)
		
		if not results:
			return DeviceListResponse(devices=[], total_count=0)
		
		devices = []
		for row in results:
			# Calculate uptime (simplified - in real implementation, this would be more complex)
			uptime = 99.5  # Default uptime, in real implementation calculate from health data
			if row['health_status'] == 'offline':
				uptime = 0.0
			elif row['health_status'] == 'warning':
				uptime = 85.0
			
			# Calculate last sync time
			last_checked = row['last_checked']
			if last_checked:
				time_diff = datetime.now() - last_checked
				if time_diff.total_seconds() < 60:
					last_sync = "Just now"
				elif time_diff.total_seconds() < 3600:
					last_sync = f"{int(time_diff.total_seconds() / 60)} min ago"
				elif time_diff.total_seconds() < 86400:
					last_sync = f"{int(time_diff.total_seconds() / 3600)} hours ago"
				else:
					last_sync = f"{int(time_diff.total_seconds() / 86400)} days ago"
			else:
				last_sync = "Never"
			
			# Map status to display format
			status_display = "Online" if row['status'] == 'active' and row['health_status'] == 'ok' else "Offline"
			
			device = DeviceResponse(
				id=row['id'],
				device_uid=row['device_uid'],
				name=row['name'],
				location=None,  # Will be populated when location system is implemented
				status=status_display,
				uptime=uptime,
				last_sync=last_sync,
				device_type=row['device_type'],
				created_at=row['created_at'],
				updated_at=row['updated_at']
			)
			devices.append(device)
		
		return DeviceListResponse(devices=devices, total_count=len(devices))
		
	except Exception as exc:
		logger.exception("Failed to get devices")
		raise HTTPException(status_code=500, detail="Failed to retrieve devices") from exc


@app.post("/devices")
def add_device(payload: AddDeviceRequest, db: Database = Depends(get_db)):
	"""
	Add a new device to the system.
	"""
	try:
		logger.info(f"Adding new device: {payload.name} ({payload.device_uid})")
		
		# Check if device already exists
		existing = db.execute("SELECT id FROM devices WHERE device_uid=%s", (payload.device_uid,))
		if existing:
			raise HTTPException(status_code=409, detail="Device with this UID already exists")
		
		# Insert new device
		db.execute(
			"""
			INSERT INTO devices (device_uid, device_type, model_id, name, status)
			VALUES (%s, %s, %s, %s, %s)
			""",
			(payload.device_uid, payload.device_type, payload.model_id, payload.name, payload.status)
		)
		
		# Get the created device
		device = db.execute("SELECT * FROM devices WHERE device_uid=%s", (payload.device_uid,))
		device = device[0]
		
		logger.info(f"Successfully added device: {device['name']}")
		
		return {
			"message": "Device added successfully",
			"device": {
				"id": device['id'],
				"device_uid": device['device_uid'],
				"name": device['name'],
				"device_type": device['device_type'],
				"status": device['status'],
				"created_at": device['created_at']
			}
		}
		
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Failed to add device {payload.name}")
		raise HTTPException(status_code=500, detail="Failed to add device") from exc


@app.put("/devices/{device_id}")
def update_device(device_id: int, payload: UpdateDeviceRequest, db: Database = Depends(get_db)):
	"""
	Update device information by device ID.
	"""
	try:
		logger.info(f"Updating device {device_id}")
		
		# Check if device exists
		device = db.execute("SELECT * FROM devices WHERE id=%s", (device_id,))
		if not device:
			raise HTTPException(status_code=404, detail="Device not found")
		
		device = device[0]
		
		# Build update query dynamically based on provided fields
		update_fields = []
		update_values = []
		
		if payload.name is not None:
			update_fields.append("name = %s")
			update_values.append(payload.name)
		
		if payload.status is not None:
			update_fields.append("status = %s")
			update_values.append(payload.status)
		
		if not update_fields:
			raise HTTPException(status_code=400, detail="No fields to update")
		
		# Add updated_at
		update_fields.append("updated_at = CURRENT_TIMESTAMP")
		update_values.append(device_id)
		
		# Execute update
		query = f"UPDATE devices SET {', '.join(update_fields)} WHERE id = %s"
		db.execute(query, tuple(update_values))
		
		# Get updated device
		updated_device = db.execute("SELECT * FROM devices WHERE id=%s", (device_id,))
		updated_device = updated_device[0]
		
		logger.info(f"Successfully updated device {device_id}")
		
		return {
			"message": "Device updated successfully",
			"device": {
				"id": updated_device['id'],
				"device_uid": updated_device['device_uid'],
				"name": updated_device['name'],
				"device_type": updated_device['device_type'],
				"status": updated_device['status'],
				"updated_at": updated_device['updated_at']
			}
		}
		
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Failed to update device {device_id}")
		raise HTTPException(status_code=500, detail="Failed to update device") from exc


@app.delete("/devices/{device_id}")
def delete_device(device_id: int, db: Database = Depends(get_db)):
	"""
	Delete a device by device ID.
	"""
	try:
		logger.info(f"Deleting device {device_id}")
		
		# Check if device exists
		device = db.execute("SELECT * FROM devices WHERE id=%s", (device_id,))
		if not device:
			raise HTTPException(status_code=404, detail="Device not found")
		
		device = device[0]
		
		# Delete device (cascade will handle related records)
		db.execute("DELETE FROM devices WHERE id=%s", (device_id,))
		
		logger.info(f"Successfully deleted device {device_id}: {device['name']}")
		
		return {
			"message": "Device deleted successfully",
			"deleted_device": {
				"id": device['id'],
				"device_uid": device['device_uid'],
				"name": device['name']
			}
		}
		
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Failed to delete device {device_id}")
		raise HTTPException(status_code=500, detail="Failed to delete device") from exc


@app.get("/devices/{device_id}/details", response_model=DetailedDeviceResponse)
def get_device_details(device_id: int, db: Database = Depends(get_db)):
	"""
	Get detailed device information including metrics for the device monitoring dashboard.
	Returns comprehensive device data with performance metrics.
	"""
	try:
		logger.info(f"Fetching detailed information for device {device_id}")
		
		# Get device basic information
		device_query = """
		SELECT 
			d.id,
			d.device_uid,
			d.name,
			d.device_type,
			d.status,
			d.created_at,
			d.updated_at,
			COALESCE(dh.health_status, 'offline') as health_status,
			COALESCE(dh.checked_at, d.updated_at) as last_checked,
			COALESCE(dh.details, '{}') as health_details
		FROM devices d
		LEFT JOIN device_health dh ON d.id = dh.device_id 
			AND dh.id = (
				SELECT MAX(id) FROM device_health 
				WHERE device_id = d.id
			)
		WHERE d.id = %s
		"""
		
		device_result = db.execute(device_query, (device_id,))
		if not device_result:
			raise HTTPException(status_code=404, detail="Device not found")
		
		device = device_result[0]
		
		# Get device metrics from telemetry
		metrics_query = """
		SELECT 
			metric_name,
			metric_value,
			recorded_at
		FROM device_telemetry 
		WHERE device_id = %s 
		AND metric_name IN ('detections', 'errors', 'cpu_usage', 'memory_usage', 'storage_usage')
		AND recorded_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
		ORDER BY recorded_at DESC
		"""
		
		metrics_result = db.execute(metrics_query, (device_id,))
		
		# Initialize default metrics
		metrics = {
			'detections': 0,
			'errors': 0,
			'cpu_usage': 0.0,
			'memory_usage': 0.0,
			'storage_usage': 0.0
		}
		
		# Process telemetry data to get latest metrics
		for metric in metrics_result:
			metric_name = metric['metric_name']
			metric_value = metric['metric_value']
			if metric_name in metrics:
				metrics[metric_name] = float(metric_value) if metric_value is not None else 0.0
		
		# If no recent telemetry data, generate sample data for demo
		if not metrics_result:
			# Generate sample metrics based on device status
			base_detections = 1000 + (device_id * 100)
			base_errors = 1 + (device_id % 3)
			
			metrics = {
				'detections': base_detections,
				'errors': base_errors,
				'cpu_usage': 45.0 + (device_id * 5),
				'memory_usage': 60.0 + (device_id * 2),
				'storage_usage': 35.0 + (device_id * 3)
			}
		
		# Calculate uptime
		uptime = 99.5
		if device['health_status'] == 'offline':
			uptime = 0.0
		elif device['health_status'] == 'warning':
			uptime = 85.0
		elif device['status'] == 'active':
			uptime = 99.0 + (device_id * 0.1)  # Slight variation for demo
		
		# Calculate last sync time
		last_checked = device['last_checked']
		if last_checked:
			time_diff = datetime.now() - last_checked
			if time_diff.total_seconds() < 60:
				last_sync = "Just now"
			elif time_diff.total_seconds() < 3600:
				last_sync = f"{int(time_diff.total_seconds() / 60)} min ago"
			elif time_diff.total_seconds() < 86400:
				last_sync = f"{int(time_diff.total_seconds() / 3600)} hours ago"
			else:
				last_sync = f"{int(time_diff.total_seconds() / 86400)} days ago"
		else:
			last_sync = "Never"
		
		# Map status to display format
		status_display = "Online" if device['status'] == 'active' and device['health_status'] == 'ok' else "Offline"
		
		device_metrics = DeviceMetrics(
			detections=int(metrics['detections']),
			errors=int(metrics['errors']),
			cpu_usage=round(metrics['cpu_usage'], 1),
			memory_usage=round(metrics['memory_usage'], 1),
			storage_usage=round(metrics['storage_usage'], 1)
		)
		
		detailed_device = DetailedDeviceResponse(
			id=device['id'],
			device_uid=device['device_uid'],
			name=device['name'],
			location=None,  # Will be populated when location system is implemented
			status=status_display,
			uptime=round(uptime, 1),
			last_sync=last_sync,
			device_type=device['device_type'],
			metrics=device_metrics,
			created_at=device['created_at'],
			updated_at=device['updated_at']
		)
		
		return detailed_device
		
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Failed to get device details for {device_id}")
		raise HTTPException(status_code=500, detail="Failed to retrieve device details") from exc


@app.get("/devices/{device_id}/metrics")
def get_device_metrics(device_id: int, db: Database = Depends(get_db)):
	"""
	Get device performance metrics for monitoring dashboards.
	Returns CPU, memory, storage usage and other performance indicators.
	"""
	try:
		logger.info(f"Fetching metrics for device {device_id}")
		
		# Check if device exists
		device = db.execute("SELECT id, name FROM devices WHERE id=%s", (device_id,))
		if not device:
			raise HTTPException(status_code=404, detail="Device not found")
		
		device = device[0]
		
		# Get recent telemetry data
		metrics_query = """
		SELECT 
			metric_name,
			metric_value,
			metric_units,
			recorded_at
		FROM device_telemetry 
		WHERE device_id = %s 
		AND recorded_at >= DATE_SUB(NOW(), INTERVAL 24 HOURS)
		ORDER BY recorded_at DESC
		"""
		
		metrics_result = db.execute(metrics_query, (device_id,))
		
		# Group metrics by name and get latest values
		latest_metrics = {}
		for metric in metrics_result:
			metric_name = metric['metric_name']
			if metric_name not in latest_metrics:
				latest_metrics[metric_name] = {
					'value': metric['metric_value'],
					'units': metric['metric_units'],
					'timestamp': metric['recorded_at']
				}
		
		# If no telemetry data, generate sample data
		if not latest_metrics:
			latest_metrics = {
				'detections': {'value': 1000 + (device_id * 100), 'units': 'count', 'timestamp': datetime.now()},
				'errors': {'value': 1 + (device_id % 3), 'units': 'count', 'timestamp': datetime.now()},
				'cpu_usage': {'value': 45.0 + (device_id * 5), 'units': '%', 'timestamp': datetime.now()},
				'memory_usage': {'value': 60.0 + (device_id * 2), 'units': '%', 'timestamp': datetime.now()},
				'storage_usage': {'value': 35.0 + (device_id * 3), 'units': '%', 'timestamp': datetime.now()},
				'network_latency': {'value': 12.5 + (device_id * 2), 'units': 'ms', 'timestamp': datetime.now()},
				'temperature': {'value': 45.0 + (device_id * 2), 'units': '°C', 'timestamp': datetime.now()}
			}
		
		return {
			"device_id": device_id,
			"device_name": device['name'],
			"metrics": latest_metrics,
			"last_updated": datetime.now().isoformat()
		}
		
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Failed to get device metrics for {device_id}")
		raise HTTPException(status_code=500, detail="Failed to retrieve device metrics") from exc


@app.post("/devices/{device_id}/telemetry")
def add_device_telemetry(
	device_id: int, 
	metric_name: str = Query(..., description="Metric name (e.g., cpu_usage, memory_usage)"),
	metric_value: float = Query(..., description="Metric value"),
	metric_units: str = Query("", description="Metric units (e.g., %, MB, ms)"),
	db: Database = Depends(get_db)
):
	"""
	Add telemetry data for a device.
	This endpoint can be used to push real-time metrics from devices.
	"""
	try:
		logger.info(f"Adding telemetry for device {device_id}: {metric_name} = {metric_value}")
		
		# Check if device exists
		device = db.execute("SELECT id FROM devices WHERE id=%s", (device_id,))
		if not device:
			raise HTTPException(status_code=404, detail="Device not found")
		
		# Insert telemetry data
		db.execute(
			"""
			INSERT INTO device_telemetry (device_id, metric_name, metric_value, metric_units, recorded_at)
			VALUES (%s, %s, %s, %s, %s)
			""",
			(device_id, metric_name, metric_value, metric_units, datetime.now())
		)
		
		logger.info(f"Successfully added telemetry for device {device_id}")
		
		return {
			"message": "Telemetry data added successfully",
			"device_id": device_id,
			"metric_name": metric_name,
			"metric_value": metric_value,
			"metric_units": metric_units,
			"recorded_at": datetime.now().isoformat()
		}
		
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Failed to add telemetry for device {device_id}")
		raise HTTPException(status_code=500, detail="Failed to add telemetry data") from exc


@app.post("/reports/generate")
def generate_report(
		payload: ReportRequest,
		db: Database = Depends(get_db),
		created_by: Optional[int] = Query(None, description="User ID creating the report")
	):
	"""
	Generate a comprehensive traffic monitoring report with visualizations and summary.
	The report includes 3-4 key sections with visualizations and an executive summary.
	"""
	try:
		logger.info(f"Generating report for period {payload.start_date} to {payload.end_date}")
		
		# Call the report generator module
		report_data = generate_comprehensive_report(
			start_date=payload.start_date,
			end_date=payload.end_date,
			title=payload.title,
			description=payload.description,
			created_by=created_by,
			db=db
		)
		
		return report_data
		
	except Exception as exc:
		logger.exception("Failed to generate report")
		raise HTTPException(status_code=500, detail="Failed to generate report") from exc


@app.get("/reports")
def get_reports(
		db: Database = Depends(get_db),
		limit: int = Query(10, description="Number of reports to return"),
		offset: int = Query(0, description="Number of reports to skip")
	):
	"""
	Get a list of generated reports with basic information.
	"""
	try:
		logger.info("Fetching reports list...")
		
		# Get reports with basic info
		reports = db.execute("""
			SELECT 
				r.id,
				r.title,
				r.description,
				r.status,
				r.created_at,
				r.updated_at,
				u.display_name as created_by_name
			FROM reports r
			LEFT JOIN users u ON r.created_by = u.id
			ORDER BY r.created_at DESC
			LIMIT %s OFFSET %s
		""", (limit, offset))
		
		# Get total count
		count_result = db.execute("SELECT COUNT(*) as total FROM reports")
		total_count = count_result[0]['total'] if count_result else 0
		
		return {
			"reports": reports,
			"total_count": total_count,
			"limit": limit,
			"offset": offset
		}
		
	except Exception as exc:
		logger.exception("Failed to get reports")
		raise HTTPException(status_code=500, detail="Failed to retrieve reports") from exc


@app.get("/reports/{report_id}")
def get_report_details(
		report_id: int,
		db: Database = Depends(get_db)
	):
	"""
	Get detailed information for a specific report including all sections and visualizations.
	"""
	try:
		logger.info(f"Fetching report details for {report_id}")
		
		# Get report basic info
		report = db.execute("""
			SELECT 
				r.id,
				r.title,
				r.description,
				r.status,
				r.created_at,
				r.updated_at,
				u.display_name as created_by_name
			FROM reports r
			LEFT JOIN users u ON r.created_by = u.id
			WHERE r.id = %s
		""", (report_id,))
		
		if not report:
			raise HTTPException(status_code=404, detail="Report not found")
		
		report = report[0]
		
		# Get report visualizations and metadata
		visualizations = db.execute("""
			SELECT 
				v.id,
				v.title,
				v.viz_type,
				v.config,
				v.created_at
			FROM report_visualizations rv
			JOIN visualizations v ON rv.visualization_id = v.id
			WHERE rv.report_id = %s
			ORDER BY rv.position
		""", (report_id,))
		
		# Extract report data from visualization config
		report_data = None
		if visualizations:
			config = visualizations[0].get('config', {})
			if isinstance(config, str):
				import json
				config = json.loads(config)
			report_data = config
		
		return {
			"report": report,
			"report_data": report_data,
			"visualizations": visualizations
		}
		
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Failed to get report details for {report_id}")
		raise HTTPException(status_code=500, detail="Failed to retrieve report details") from exc


@app.delete("/reports/{report_id}")
def delete_report(
		report_id: int,
		db: Database = Depends(get_db)
	):
	"""
	Delete a report and its associated visualizations.
	"""
	try:
		logger.info(f"Deleting report {report_id}")
		
		# Check if report exists
		report = db.execute("SELECT id, title FROM reports WHERE id = %s", (report_id,))
		if not report:
			raise HTTPException(status_code=404, detail="Report not found")
		
		report = report[0]
		
		# Delete report (cascade will handle related records)
		db.execute("DELETE FROM reports WHERE id = %s", (report_id,))
		
		logger.info(f"Successfully deleted report {report_id}: {report['title']}")
		
		return {
			"message": "Report deleted successfully",
			"deleted_report": {
				"id": report['id'],
				"title": report['title']
			}
		}
		
	except HTTPException:
		raise
	except Exception as exc:
		logger.exception(f"Failed to delete report {report_id}")
		raise HTTPException(status_code=500, detail="Failed to delete report") from exc

