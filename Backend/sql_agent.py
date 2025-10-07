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


def get_db_schema_text(db) -> str:
	rows = db.execute(
		"""
		SELECT TABLE_NAME as table_name, COLUMN_NAME as column_name, DATA_TYPE as data_type
		FROM information_schema.COLUMNS
		WHERE TABLE_SCHEMA = DATABASE()
		ORDER BY TABLE_NAME, ORDINAL_POSITION
		"""
	)
	if not rows:
		return "(no schema)"
	schema: dict[str, List[tuple[str, str]]] = {}
	for r in rows:
		schema.setdefault(r["table_name"], []).append((r["column_name"], r["data_type"]))
	parts: List[str] = []
	for table, cols in schema.items():
		cols_s = ", ".join(f"{c} {t}" for c, t in cols)
		parts.append(f"{table}({cols_s})")
	return "\n".join(parts)


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


def generate_sql_from_question(question: str, schema_text: str) -> str:
	model = genai.GenerativeModel("gemini-1.5-pro")
	prompt = (
		"You are a helpful MySQL SQL generator. Given the database schema and a user question, "
		"write a single safe, read-only SQL query (SELECT only). Use MySQL syntax. "
		"Return only the SQL in a markdown code block. Never modify data.\n\n"
		f"Schema:\n{schema_text}\n\n"
		f"Question: {question}\n"
	)
	resp = model.generate_content(prompt)
	text = resp.text or ""
	sql = _extract_sql_from_text(text)
	if not _is_safe_select(sql):
		raise ValueError("Generated SQL is not a SELECT query")
	return _ensure_limit(sql)


def summarize_results(question: str, sql: str, rows: list[dict]) -> str:
	model = genai.GenerativeModel("gemini-1.5-flash")
	sample = rows[:20]
	data_preview = json.dumps(sample, ensure_ascii=False)
	prompt = (
		"You are a helpful analyst. Summarize the SQL results succinctly for a business user. "
		"Include key numbers, trends, and caveats. If rows are empty, say there's no data.\n\n"
		f"Question: {question}\n"
		f"SQL: {sql}\n"
		f"First rows (JSON): {data_preview}\n"
	)
	resp = model.generate_content(prompt)
	return (resp.text or "").strip()


