import os
import json
import logging
from typing import List

import google.generativeai as genai


logger = logging.getLogger(__name__)


def configure_gemini_from_env() -> None:
	api_key = os.getenv("GOOGLE_API_KEY")
	if not api_key:
		logger.warning("GOOGLE_API_KEY is not set; LLM features will be disabled.")
		return
	genai.configure(api_key=api_key)


def get_data_raw_schema_text() -> str:
	"""Return a fixed schema description for the data_raw table only."""
	return (
		"Table: data_raw\n"
		"Columns:\n"
		"- id BIGINT (PK)\n"
		"- batch_id CHAR(36)\n"
		"- source_file VARCHAR(512)\n"
		"- row_index INT (1-based index from original sheet)\n"
		"- row_data JSON (entire row as JSON object; use JSON_EXTRACT to access fields)\n"
		"- imported_at TIMESTAMP\n"
		"- imported_by BIGINT (nullable)\n\n"
		"Notes: Use MySQL JSON functions, e.g., JSON_EXTRACT(row_data, '$.ColumnName')."
	)


def _extract_sql_from_text(text: str) -> str:
	start = text.find("```")
	if start != -1:
		end = text.find("```", start + 3)
		if end != -1:
			code = text[start + 3:end].strip()
			first_nl = code.find("\n")
			if first_nl != -1 and code[:first_nl].strip().lower() in {"sql", "mysql"}:
				code = code[first_nl + 1:]
			return code.strip().rstrip(";")
	return text.strip().rstrip(";")


def _is_safe_select(sql: str) -> bool:
	upper = sql.strip().upper()
	if upper.startswith("WITH "):
		return True
	return upper.startswith("SELECT ")


def _ensure_limit(sql: str, default_limit: int = 100) -> str:
	upper = sql.upper()
	if " LIMIT " in upper:
		return sql
	return f"{sql} LIMIT {default_limit}"


def _assert_only_data_raw(sql: str) -> None:
	lower = sql.lower()
	if "data_raw" not in lower:
		raise ValueError("Query must reference only the data_raw table")



def generate_sql_from_question(question: str, schema_text: str | None = None) -> str:
	model = genai.GenerativeModel("gemini-2.5-pro")
	if not schema_text:
		schema_text = get_data_raw_schema_text()
	prompt = (
		"You are a helpful MySQL SQL generator for a single table named data_raw. "
		"Write a single safe, read-only SQL query (SELECT only) using only the data_raw table. "
		"Use MySQL syntax. Prefer JSON_EXTRACT(row_data, '$.Field') to access JSON fields. "
		"Return only the SQL in a markdown code block. Never modify data.\n\n"
		f"Schema:\n{schema_text}\n\n"
		f"Question: {question}\n"
	)
	resp = model.generate_content(prompt)
	text = resp.text or ""
	sql = _extract_sql_from_text(text)
	if not _is_safe_select(sql):
		raise ValueError("Generated SQL is not a SELECT query")
	_assert_only_data_raw(sql)
	return _ensure_limit(sql)


def summarize_results(question: str, sql: str, rows: list[dict]) -> str:
	model = genai.GenerativeModel("gemini-1.5-flash")
	sample = rows[:20]
	data_preview = json.dumps(sample, ensure_ascii=False)
	prompt = (
		"You are a helpful analyst. Answer the user's question using ONLY the provided rows. "
		"Cite key numbers explicitly. If rows are empty, say there's no data answering the question.\n\n"
		f"Question: {question}\n"
		f"SQL: {sql}\n"
		f"First rows (JSON): {data_preview}\n"
	)
	resp = model.generate_content(prompt)
	return (resp.text or "").strip()


def run_data_raw_agent(db, question: str) -> dict:
	"""End-to-end: build SELECT for data_raw, execute, and summarize."""
	schema_text = get_data_raw_schema_text()
	sql = generate_sql_from_question(question, schema_text)
	rows = db.execute(sql) or []
	summary = summarize_results(question, sql, rows)
	return {"sql": sql, "rows": rows, "summary": summary}


