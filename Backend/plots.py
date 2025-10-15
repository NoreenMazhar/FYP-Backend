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


def _build_filter_clause(start_date: date, end_date: date, device: Optional[str], vehicle_type: Optional[str]) -> str:
	conds: list[str] = []
	# Dates are treated on DATE(local_timestamp)
	conds.append(f"DATE(local_timestamp) BETWEEN '{start_date:%Y-%m-%d}' AND '{end_date:%Y-%m-%d}'")
	if device:
		conds.append(f"device_name = '{device}'")
	if vehicle_type:
		conds.append(f"vehicle_type = '{vehicle_type}'")
	return " AND ".join(conds)


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


def generate_2d_plots(
		start_date: Optional[date],
		end_date: Optional[date],
		device: Optional[str],
		vehicle_type: Optional[str],
):
	"""
	Generate multiple 2D plot datasets by asking the SQL agent to produce SQL.
	Returns a list of plot objects with keys: Type of Plot, X, Y, Title, Description.
	"""
	# Default window: last 30 days if not provided
	if end_date is None:
		end_date = date.today()
	if start_date is None:
		start_date = end_date - timedelta(days=29)

	where_clause = _build_filter_clause(start_date, end_date, device, vehicle_type)
	plots: list[dict] = []

	# 1) Detections per day
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
		plots.append({
			"Type of Plot": "Line",
			"X": x_vals,
			"Y": y_vals,
			"Title": "Detections per Day",
			"Description": "Total detections per day within the selected range"
		})
	except Exception as e:
		logger.warning(f"Detections per day plot failed: {e}")

	# 2) Detections by vehicle type
	q2 = (
		"Write a MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		"x must be vehicle_type and y must be COUNT(*). "
		f"Filter with: {where_clause}. Group by vehicle_type. Order by COUNT(*) DESC. "
		"Only output SQL."
	)
	try:
		_, rows = _ask_agent_for_xy(q2)
		x_vals, y_vals = _rows_to_xy(rows)
		plots.append({
			"Type of Plot": "Bar",
			"X": x_vals,
			"Y": y_vals,
			"Title": "Detections by Vehicle Type",
			"Description": "Distribution of detections across vehicle types"
		})
	except Exception as e:
		logger.warning(f"Vehicle type plot failed: {e}")

	# 3) Detections by direction
	q3 = (
		"Write a MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		"x must be direction and y must be COUNT(*). "
		f"Filter with: {where_clause}. Group by direction. Order by COUNT(*) DESC. "
		"Only output SQL."
	)
	try:
		_, rows = _ask_agent_for_xy(q3)
		x_vals, y_vals = _rows_to_xy(rows)
		plots.append({
			"Type of Plot": "Bar",
			"X": x_vals,
			"Y": y_vals,
			"Title": "Detections by Direction",
			"Description": "Inbound vs Outbound detections"
		})
	except Exception as e:
		logger.warning(f"Direction plot failed: {e}")

	# 4) Average OCR score per day
	q4 = (
		"Write a MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		"x must be the date (YYYY-MM-DD) from DATE(local_timestamp) and y must be AVG(ocr_score). "
		f"Filter with: {where_clause}. Group by DATE(local_timestamp). Order by DATE(local_timestamp). "
		"Only output SQL."
	)
	try:
		_, rows = _ask_agent_for_xy(q4)
		x_vals, y_vals = _rows_to_xy(rows)
		plots.append({
			"Type of Plot": "Line",
			"X": x_vals,
			"Y": y_vals,
			"Title": "Average OCR Score per Day",
			"Description": "Mean OCR confidence per day within the selected range"
		})
	except Exception as e:
		logger.warning(f"Average OCR per day plot failed: {e}")

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
				viz_type = str(plot.get("Type of Plot") or "chart").lower()
				title = str(plot.get("Title") or "Visualization")
				config = {
					"type": plot.get("Type of Plot"),
					"x": plot.get("X", []),
					"y": plot.get("Y", []),
					"description": plot.get("Description", ""),
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

import logging
from datetime import date, timedelta
from typing import Optional

from db import Database
from sql_agent import generate_sql_for_question


logger = logging.getLogger(__name__)


def _build_where_clause_literal(start_date: date, end_date: date, device: Optional[str], vehicle_type: Optional[str]) -> str:
	parts: list[str] = [
		f"DATE(local_timestamp) BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'"
	]
	if device:
		parts.append(f"device_name = '{device.replace("'", "''")}'")
	if vehicle_type:
		parts.append(f"vehicle_type = '{vehicle_type.replace("'", "''")}'")
	return " AND ".join(parts)


def _exec_sql(db: Database, sql: str) -> list[dict]:
	try:
		return db.execute(sql) or []
	except Exception as e:
		logger.warning(f"SQL execution failed: {e}. SQL: {sql}")
		return []


def get_2d_plots_via_agent(
	start_date: Optional[date],
	end_date: Optional[date],
	device: Optional[str],
	vehicle_type: Optional[str],
	db: Database,
):
	if end_date is None:
		end_date = date.today()
	if start_date is None:
		start_date = end_date - timedelta(days=29)

	where_literal = _build_where_clause_literal(start_date, end_date, device, vehicle_type)
	plots: list[dict] = []

	# 1) Detections per day
	q1 = (
		"Write a single MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		f"x = DATE(local_timestamp), y = COUNT(*). Apply WHERE: {where_literal}. Group by x. Order by x."
	)
	sql1 = generate_sql_for_question(q1)
	if sql1:
		rows = _exec_sql(db, sql1)
		x_vals = [str(r.get('x') or r.get('day') or r.get('DATE(local_timestamp)') or list(r.values())[0]) for r in rows]
		y_vals = [int((r.get('y') if r.get('y') is not None else list(r.values())[-1]) or 0) for r in rows]
		plots.append({
			"Type of Plot": "Line",
			"X": x_vals,
			"Y": y_vals,
			"Title": "Detections per Day",
			"Description": "Total detections per day within the selected range"
		})

	# 2) Detections by vehicle type
	q2 = (
		"Write a single MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		f"x = vehicle_type, y = COUNT(*). Apply WHERE: {where_literal}. Group by x. Order by y DESC."
	)
	sql2 = generate_sql_for_question(q2)
	if sql2:
		rows = _exec_sql(db, sql2)
		x_vals = [str(r.get('x') or r.get('vehicle_type') or list(r.values())[0]) for r in rows]
		y_vals = [int((r.get('y') if r.get('y') is not None else list(r.values())[-1]) or 0) for r in rows]
		plots.append({
			"Type of Plot": "Bar",
			"X": x_vals,
			"Y": y_vals,
			"Title": "Detections by Vehicle Type",
			"Description": "Distribution of detections across vehicle types"
		})

	# 3) Detections by direction
	q3 = (
		"Write a single MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		f"x = direction, y = COUNT(*). Apply WHERE: {where_literal}. Group by x. Order by y DESC."
	)
	sql3 = generate_sql_for_question(q3)
	if sql3:
		rows = _exec_sql(db, sql3)
		x_vals = [str(r.get('x') or r.get('direction') or list(r.values())[0]) for r in rows]
		y_vals = [int((r.get('y') if r.get('y') is not None else list(r.values())[-1]) or 0) for r in rows]
		plots.append({
			"Type of Plot": "Bar",
			"X": x_vals,
			"Y": y_vals,
			"Title": "Detections by Direction",
			"Description": "Inbound vs Outbound detections"
		})

	# 4) Average OCR score per day
	q4 = (
		"Write a single MySQL SELECT over data_raw that returns two columns aliased as x and y. "
		f"x = DATE(local_timestamp), y = AVG(ocr_score). Apply WHERE: {where_literal}. Group by x. Order by x."
	)
	sql4 = generate_sql_for_question(q4)
	if sql4:
		rows = _exec_sql(db, sql4)
		x_vals = [str(r.get('x') or r.get('day') or list(r.values())[0]) for r in rows]
		y_vals = [float((r.get('y') if r.get('y') is not None else list(r.values())[-1]) or 0.0) for r in rows]
		plots.append({
			"Type of Plot": "Line",
			"X": x_vals,
			"Y": y_vals,
			"Title": "Average OCR Score per Day",
			"Description": "Mean OCR confidence per day within the selected range"
		})

	return plots


