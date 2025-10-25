import logging
import json
from datetime import date, timedelta
from typing import Optional

from db import Database
from sql_agent import run_data_raw_agent


logger = logging.getLogger(__name__)


def _execute_sql(db: Database, sql: str):
	"""Execute raw SQL using the project's MySQL connection and return rows.
	This expects a read-only SELECT produced by the SQL agent.
	"""
	if not sql or not sql.strip().lower().startswith("select"):
		raise ValueError("Only SELECT statements are allowed")
	return db.execute(sql)


def _build_filter_clause(start_date: Optional[date], end_date: Optional[date], device: Optional[str], vehicle_type: Optional[str]) -> str:
	"""Build WHERE clause for filtering data. If dates are not provided, no date filter is applied."""
	conds: list[str] = []
	
	# Only add date filter if both dates are provided
	if start_date and end_date:
		conds.append(f"DATE(local_timestamp) BETWEEN '{start_date:%Y-%m-%d}' AND '{end_date:%Y-%m-%d}'")
	
	if device:
		conds.append(f"device_name = '{device}'")
	if vehicle_type:
		conds.append(f"vehicle_type = '{vehicle_type}'")
	
	return " AND ".join(conds) if conds else "1=1"


def _ask_agent_for_xy(question: str) -> tuple[str, list[dict]]:
	"""Use SQL agent to produce SQL with two columns aliased as x and y, then execute it.
	Returns (executed_sql, rows).
	"""
	from sql_agent import generate_sql_for_question
	
	# First try direct SQL generation (faster and more reliable for simple queries)
	executed_sql = generate_sql_for_question(question)
	
	# If direct generation failed, try the full agent
	if not executed_sql:
		result = run_data_raw_agent(question)
		executed_sql = result.get("executed_sql")
		
		# If no SQL was returned, try to extract it from the result text
		if not executed_sql:
			output_text = result.get("result", {})
			if isinstance(output_text, dict):
				# Look for SQL in the result structure
				for key, value in output_text.items():
					if isinstance(value, str) and "SELECT" in value.upper():
						# Try to extract SQL from markdown code blocks
						if "```sql" in value:
							start = value.find("```sql") + 6
							end = value.find("```", start)
							if end > start:
								executed_sql = value[start:end].strip()
								break
						elif value.strip().upper().startswith("SELECT"):
							executed_sql = value.strip()
							break
	
	if not executed_sql:
		raise ValueError("SQL agent did not return an executable SQL query")
	
	# Execute with our connection for reliable row shape
	db = Database.get_instance()
	rows = _execute_sql(db, executed_sql) or []
	return executed_sql, rows


def _rows_to_xy(rows: list[dict]) -> tuple[list, list]:
	"""Extract X and Y from rows with keys 'x' and 'y'."""
	x_vals = []
	y_vals = []
	for r in rows:
		# MySQL python returns dict rows due to our db wrapper
		x_vals.append(str(r.get('x')) if r.get('x') is not None else "")
		y = r.get('y')
		try:
			y_vals.append(float(y) if y is not None else 0.0)
		except Exception:
			y_vals.append(0.0)
	return x_vals, y_vals


def _sanitize_x_labels(x_vals: list) -> list:
	"""Normalize X labels: replace blanks/None/NaN-like with 'Unknown'."""
	cleaned: list[str] = []
	for x in x_vals:
		label = "" if x is None else str(x).strip()
		if label == "" or label.lower() in {"none", "null", "nan"}:
			cleaned.append("Unknown")
		else:
			cleaned.append(label)
	return cleaned


def generate_2d_plots(
			start_date: Optional[date],
			end_date: Optional[date],
			device: Optional[str],
			vehicle_type: Optional[str],
	):
	"""
	Generate multiple 2D plot datasets by asking the SQL agent to produce SQL.
	Returns a list of plot objects in the specified JSON format.
	"""
	where_clause = _build_filter_clause(start_date, end_date, device, vehicle_type)
	plots: list[dict] = []

	# 1) Detections per day (Line chart)
	q1 = (
		"You are a SQL expert. Write a single MySQL SELECT over table data_raw that "
		"returns two columns aliased as x and y. x must be the date (YYYY-MM-DD) "
		"from DATE(local_timestamp) and y must be COUNT(*). "
		f"Filter with: {where_clause}. Group by DATE(local_timestamp). "
		"Order by DATE(local_timestamp). Do not include any text, only the SQL."
	)
	try:
		_, rows = _ask_agent_for_xy(q1)
		x_vals, y_vals = _rows_to_xy(rows)
		x_vals = _sanitize_x_labels(x_vals)
		plots.append({
			"Data": {
				"X": x_vals,
				"Y": y_vals
			},
			"Plot-type": "line",
			"X-axis-label": "Date",
			"Y-axis-label": "Number of Detections",
			"Description": "Total detections per day within the selected range"
		})
	except Exception as e:
		logger.warning(f"Detections per day plot failed: {e}")
		# Add fallback plot with sample data
		plots.append({
			"Data": {
				"X": ["No Data"],
				"Y": [0]
			},
			"Plot-type": "line",
			"X-axis-label": "Date",
			"Y-axis-label": "Number of Detections",
			"Description": "No data available for detections per day"
		})

	# 2) Detections by vehicle type (Bar chart)
	q2 = (
		"Write a MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		"x must be vehicle_type and y must be COUNT(*). "
		f"Filter with: {where_clause}. Group by vehicle_type. Order by COUNT(*) DESC. "
		"Only output SQL."
	)
	try:
		_, rows = _ask_agent_for_xy(q2)
		x_vals, y_vals = _rows_to_xy(rows)
		x_vals = _sanitize_x_labels(x_vals)
		plots.append({
			"Data": {
				"X": x_vals,
				"Y": y_vals
			},
			"Plot-type": "bar",
			"X-axis-label": "Vehicle Type",
			"Y-axis-label": "Number of Detections",
			"Description": "Distribution of detections across vehicle types"
		})
	except Exception as e:
		logger.warning(f"Vehicle type plot failed: {e}")
		# Add fallback plot
		plots.append({
			"Data": {
				"X": ["No Data"],
				"Y": [0]
			},
			"Plot-type": "bar",
			"X-axis-label": "Vehicle Type",
			"Y-axis-label": "Number of Detections",
			"Description": "No data available for vehicle type distribution"
		})

	# 3) Detections by direction (Pie chart)
	q3 = (
		"Write a MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		"x must be direction and y must be COUNT(*). "
		f"Filter with: {where_clause}. Group by direction. Order by COUNT(*) DESC. "
		"Only output SQL."
	)
	try:
		_, rows = _ask_agent_for_xy(q3)
		x_vals, y_vals = _rows_to_xy(rows)
		x_vals = _sanitize_x_labels(x_vals)
		plots.append({
			"Data": {
				"X": x_vals,
				"Y": y_vals
			},
			"Plot-type": "pie",
			"X-axis-label": "Direction",
			"Y-axis-label": "Number of Detections",
			"Description": "Inbound vs Outbound detections"
		})
	except Exception as e:
		logger.warning(f"Direction plot failed: {e}")
		# Add fallback plot
		plots.append({
			"Data": {
				"X": ["No Data"],
				"Y": [0]
			},
			"Plot-type": "pie",
			"X-axis-label": "Direction",
			"Y-axis-label": "Number of Detections",
			"Description": "No data available for direction analysis"
		})

	# 4) Average OCR score per day (Line chart)
	q4 = (
		"Write a MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		"x must be the date (YYYY-MM-DD) from DATE(local_timestamp) and y must be AVG(ocr_score). "
		f"Filter with: {where_clause}. Group by DATE(local_timestamp). Order by DATE(local_timestamp). "
		"Only output SQL."
	)
	try:
		_, rows = _ask_agent_for_xy(q4)
		x_vals, y_vals = _rows_to_xy(rows)
		x_vals = _sanitize_x_labels(x_vals)
		plots.append({
			"Data": {
				"X": x_vals,
				"Y": y_vals
			},
			"Plot-type": "line",
			"X-axis-label": "Date",
			"Y-axis-label": "Average OCR Score",
			"Description": "Mean OCR confidence per day within the selected range"
		})
	except Exception as e:
		logger.warning(f"Average OCR per day plot failed: {e}")
		# Add fallback plot
		plots.append({
			"Data": {
				"X": ["No Data"],
				"Y": [0]
			},
			"Plot-type": "line",
			"X-axis-label": "Date",
			"Y-axis-label": "Average OCR Score",
			"Description": "No data available for OCR score analysis"
		})

	# 5) Detections by device (Donut chart)
	q5 = (
		"Write a MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		"x must be device_name and y must be COUNT(*). "
		f"Filter with: {where_clause}. Group by device_name. Order by COUNT(*) DESC. "
		"Only output SQL."
	)
	try:
		_, rows = _ask_agent_for_xy(q5)
		x_vals, y_vals = _rows_to_xy(rows)
		x_vals = _sanitize_x_labels(x_vals)
		plots.append({
			"Data": {
				"X": x_vals,
				"Y": y_vals
			},
			"Plot-type": "donut",
			"X-axis-label": "Device",
			"Y-axis-label": "Number of Detections",
			"Description": "Distribution of detections across different devices"
		})
	except Exception as e:
		logger.warning(f"Device plot failed: {e}")
		# Add fallback plot
		plots.append({
			"Data": {
				"X": ["No Data"],
				"Y": [0]
			},
			"Plot-type": "donut",
			"X-axis-label": "Device",
			"Y-axis-label": "Number of Detections",
			"Description": "No data available for device analysis"
		})

	return plots


def convert_text_to_plots(
		text_description: str,
		start_date: Optional[date] = None,
		end_date: Optional[date] = None,
		device: Optional[str] = None,
		vehicle_type: Optional[str] = None,
):
	"""
	Convert a text description to plot data using the SQL agent.
	The text description should describe what kind of plot/analysis the user wants.
	Returns a list of plot objects in the specified JSON format.
	"""
	where_clause = _build_filter_clause(start_date, end_date, device, vehicle_type)
	plots: list[dict] = []

	# Create a comprehensive prompt for the SQL agent
	prompt = f"""
	You are a SQL expert. Based on the user's request: "{text_description}"
	
	Write a MySQL SELECT query over the data_raw table that returns two columns aliased as 'x' and 'y'.
	
	Requirements:
	- The query should be relevant to the user's request
	- Use the WHERE clause: {where_clause}
	- The 'x' column should contain categorical or time-based data
	- The 'y' column should contain numerical data (counts, averages, sums, etc.)
	- Group by the 'x' column if appropriate
	- Order the results logically
	- Only output the SQL query, no other text
	
	Available columns in data_raw:
	- local_timestamp (datetime)
	- device_name (string)
	- direction (string) 
	- vehicle_type (string)
	- ocr_score (float)
	- vehicle_types_lp_ocr (string)
	
	Examples of good queries:
	- For "show me detections by hour": SELECT HOUR(local_timestamp) as x, COUNT(*) as y FROM data_raw WHERE {where_clause} GROUP BY HOUR(local_timestamp) ORDER BY HOUR(local_timestamp)
	- For "average ocr score by device": SELECT device_name as x, AVG(ocr_score) as y FROM data_raw WHERE {where_clause} GROUP BY device_name ORDER BY AVG(ocr_score) DESC
	"""

	try:
		_, rows = _ask_agent_for_xy(prompt)
		if not rows:
			# Fallback to a basic query if the agent doesn't return results
			fallback_prompt = f"""
			Write a MySQL SELECT over data_raw that returns two columns aliased as x and y.
			x must be vehicle_type and y must be COUNT(*).
			Filter with: {where_clause}. Group by vehicle_type. Order by COUNT(*) DESC.
			Only output SQL.
			"""
			_, rows = _ask_agent_for_xy(fallback_prompt)
		
		if rows:
			x_vals, y_vals = _rows_to_xy(rows)
			x_vals = _sanitize_x_labels(x_vals)
			
			# Determine plot type based on the data
			plot_type = "bar"  # Default
			if len(x_vals) > 10:  # If many data points, use line chart
				plot_type = "line"
			elif len(x_vals) <= 5:  # If few categories, use pie chart
				plot_type = "pie"
			
			# Generate appropriate labels
			x_label = "Category"
			y_label = "Count"
			
			# Try to infer better labels from the data
			if any("hour" in str(x).lower() or "time" in str(x).lower() for x in x_vals):
				x_label = "Time"
			elif any("date" in str(x).lower() for x in x_vals):
				x_label = "Date"
			elif any("device" in str(x).lower() for x in x_vals):
				x_label = "Device"
			elif any("type" in str(x).lower() for x in x_vals):
				x_label = "Vehicle Type"
			elif any("direction" in str(x).lower() for x in x_vals):
				x_label = "Direction"
			
			if any("avg" in str(y).lower() or "average" in str(y).lower() for y in y_vals):
				y_label = "Average Value"
			elif any("sum" in str(y).lower() for y in y_vals):
				y_label = "Total Value"
			elif any("count" in str(y).lower() for y in y_vals):
				y_label = "Count"
			
			plots.append({
				"Data": {
					"X": x_vals,
					"Y": y_vals
				},
				"Plot-type": plot_type,
				"X-axis-label": x_label,
				"Y-axis-label": y_label,
				"Description": f"Analysis based on: {text_description}"
			})
		else:
			# If no data, return an empty plot with a message
			plots.append({
				"Data": {
					"X": ["No Data"],
					"Y": [0]
				},
				"Plot-type": "bar",
				"X-axis-label": "Status",
				"Y-axis-label": "Count",
				"Description": f"No data found for: {text_description}"
			})
			
	except Exception as e:
		logger.warning(f"Text to plot conversion failed: {e}")
		# Return an error plot
		plots.append({
			"Data": {
				"X": ["Error"],
				"Y": [0]
			},
			"Plot-type": "bar",
			"X-axis-label": "Status",
			"Y-axis-label": "Count",
			"Description": f"Error processing request: {text_description}"
		})

	return plots


def get_2d_plots_via_agent(
		start_date: Optional[date],
		end_date: Optional[date],
		device: Optional[str],
		vehicle_type: Optional[str],
		db: Database,
		created_by: Optional[int] = None,
	):
	# Generate plots via agent
	plots = generate_2d_plots(start_date, end_date, device, vehicle_type)

	# Optionally persist each plot as a visualization
	if created_by is not None:
		conn = Database.get_instance()
		for plot in plots:
			try:
				viz_type = "chart"  # Default type
				title = "Visualization"
				config = {
					"x": plot.get("Data", {}).get("X", []),
					"y": plot.get("Data", {}).get("Y", []),
					"description": plot.get("Description", ""),
					"x_axis_label": plot.get("X-axis-label", ""),
					"y_axis_label": plot.get("Y-axis-label", ""),
					"filters": {
						"start_date": start_date.isoformat() if start_date else None,
						"end_date": end_date.isoformat() if end_date else None,
						"device": device,
						"vehicle_type": vehicle_type,
					},
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
				logger.warning(f"Failed to persist visualization: {e}")

	return plots
