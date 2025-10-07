import os
import json
import logging
from typing import Any

import google.generativeai as genai
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI


logger = logging.getLogger(__name__)


def configure_gemini_from_env() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY is not set; LLM features will be disabled.")
        return
    genai.configure(api_key=api_key)


def _get_sqlalchemy_url_from_env() -> str:
    user = os.getenv("DB_USER", os.getenv("MYSQL_USER", "FYP-USER"))
    password = os.getenv("DB_PASSWORD", os.getenv("MYSQL_PASSWORD", "FYP-PASS"))
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    database = os.getenv("DB_NAME", os.getenv("MYSQL_DATABASE", "FYP-DB"))
    return f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"


_LC_DB: SQLDatabase | None = None


def get_lc_sql_db() -> SQLDatabase:
    global _LC_DB
    if _LC_DB is None:
        url = _get_sqlalchemy_url_from_env()
        _LC_DB = SQLDatabase.from_uri(url, include_tables=["data_raw"], sample_rows_in_table_info=2)
    return _LC_DB


def get_data_raw_schema_text() -> str:
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


def _is_safe_select(sql: str) -> bool:
    upper = sql.strip().upper()
    if upper.startswith("WITH "):
        return True
    return upper.startswith("SELECT ")


def _assert_only_data_raw(sql: str) -> None:
    lower = sql.lower()
    if "data_raw" not in lower:
        raise ValueError("Query must reference only the data_raw table")


def run_data_raw_agent(question: str) -> dict:
    """End-to-end: use LangChain SQL Agent (Gemini) to generate, run, and summarize SQL on data_raw."""
    configure_gemini_from_env()
    lc_db = get_lc_sql_db()

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    agent_executor = create_sql_agent(
        llm=llm,
        db=lc_db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    result = agent_executor.invoke({"input": question})
    output_text = result.get("output", "").strip()

    # Extract executed SQL for debugging
    executed_sql = None
    for step in result.get("intermediate_steps", []):
        if isinstance(step, tuple) and "SELECT" in step[0].upper():
            executed_sql = step[0]
            break

    return {
        "question": question,
        "executed_sql": executed_sql,
        "result": output_text,
    }
