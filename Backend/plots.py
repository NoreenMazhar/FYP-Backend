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
	result = run_data_raw_agent(question)
	executed_sql = result.get("executed_sql")
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
