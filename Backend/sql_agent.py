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


# Caches for different table scopes
_LC_DBS: dict[str, SQLDatabase] = {}

# Table scopes
DATA_RAW_TABLES = ["data_raw"]
DEVICES_TABLES = [
    "device_models",
    "devices",
    "device_firmware",
    "device_config_profiles",
    "device_config_applied",
    "device_network_interfaces",
    "device_streams",
    "device_mounts",
    "device_health",
    "device_telemetry",
    "device_events",
    "device_permissions",
    "device_groups",
    "device_group_members",
    "device_maintenance",
    "device_credentials",
]


def get_lc_sql_db(include_tables: list[str]) -> SQLDatabase:
    """Get or create a cached SQLDatabase for the given include_tables scope."""
    global _LC_DBS
    key = ",".join(sorted(include_tables)) or "__all__"
    if key not in _LC_DBS:
        url = _get_sqlalchemy_url_from_env()
        try:
            _LC_DBS[key] = SQLDatabase.from_uri(
                url,
                include_tables=include_tables,
                sample_rows_in_table_info=2,
            )
        except Exception:
            # Fallback to no include filter if specific tables are not present
            _LC_DBS[key] = SQLDatabase.from_uri(url, sample_rows_in_table_info=2)
    return _LC_DBS[key]


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


def _is_device_query(question: str) -> bool:
    """Heuristic to detect device-related queries."""
    if not question:
        return False
    q = question.lower()
    keywords = [
        "device",
        "devices",
        "camera",
        "cameras",
        "firmware",
        "stream",
        "streams",
        "rtsp",
        "gateway",
        "nvr",
        "telemetry",
        "health",
        "maintenance",
        "mount",
        "network interface",
        "mac address",
        "ip address",
        "event",
        "events",
        "model",
        "models",
        "group",
        "groups",
        "credential",
        "credentials",
    ]
    return any(k in q for k in keywords)


def _build_prompt(question: str, use_device_scope: bool) -> str:
    """Compose guidance plus the user question to steer the agent."""
    if use_device_scope:
        tables_text = ", ".join(DEVICES_TABLES)
        scope_instruction = (
            "You must answer by querying ONLY the device tables: "
            f"{tables_text}. Do not query unrelated tables."
        )
        domain_context = (
            "Schema gist: devices and device_models define inventory; device_streams/firmware/config* "
            "cover media and configuration; device_health/telemetry/events record status and metrics; "
            "device_groups and permissions handle organization and access."
        )
    else:
        tables_text = ", ".join(DATA_RAW_TABLES)
        scope_instruction = (
            "You must answer by querying ONLY the data lake table: "
            f"{tables_text}. Access fields via JSON_EXTRACT on row_data."
        )
        domain_context = (
            "Schema gist: data_raw holds imported spreadsheet rows as JSON. Use MySQL JSON functions."
        )

    formatting = (
        "Format your final answer in clear Markdown with these sections:\n"
        "### Overview\n"
        "- Briefly state the direct answer in plain language.\n"
        "### Key Findings\n"
        "- Bullet important numbers, trends, or comparisons.\n"
        "### SQL Used\n"
        "```sql\n<your final SQL here>\n```\n"
        "### Observations\n"
        "- Short insights or anomalies worth noting.\n"
        "### Next Steps\n"
        "- 2-3 concise suggestions for follow-up analysis or checks."
    )

    guardrails = (
        "Constraints: Use only read-only SELECTs; avoid DDL/DML. If multiple queries are needed, "
        "show the final, most important SQL in the SQL Used section. Keep the language simple."
    )

    return (
        f"{scope_instruction}\n{domain_context}\n{formatting}\n{guardrails}\n\n"
        f"User question: {question}"
    )


def run_data_raw_agent(question: str) -> dict:
    """End-to-end: use LangChain SQL Agent (Gemini) to generate, run, and summarize SQL.

    If the question is device-related, switch to device tables; otherwise use data_raw.
    """
    configure_gemini_from_env()

    use_device_scope = _is_device_query(question)
    include_tables = DEVICES_TABLES if use_device_scope else DATA_RAW_TABLES
    lc_db = get_lc_sql_db(include_tables)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    agent_executor = create_sql_agent(
        llm=llm,
        db=lc_db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    enhanced_prompt = _build_prompt(question, use_device_scope)

    result = agent_executor.invoke({"input": enhanced_prompt})
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
        "used_device_scope": use_device_scope,
    }
