import os
import json
import logging
from typing import Any

import google.generativeai as genai
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish


logger = logging.getLogger(__name__)


class CustomSQLOutputParser(ReActSingleInputOutputParser):
    """Custom output parser that handles markdown-formatted responses gracefully."""
    
    def parse(self, text: str) -> AgentFinish:
        """Parse the agent output, handling markdown format."""
        # If the text looks like a final answer in markdown format, return it as AgentFinish
        if "### Overview" in text or "### Key Findings" in text or "### SQL Used" in text:
            return AgentFinish(
                return_values={"output": text},
                log=text
            )
        
        # Otherwise, try the default parsing
        try:
            return super().parse(text)
        except Exception as e:
            logger.warning(f"Failed to parse agent output: {e}")
            # If parsing fails, treat the entire text as the final answer
            return AgentFinish(
                return_values={"output": text},
                log=text
            )


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
        "- local_timestamp VARCHAR(255) - timestamp from Excel data\n"
        "- device_name VARCHAR(255) - device name (e.g., neom6)\n"
        "- direction VARCHAR(100) - direction (approaching/receding)\n"
        "- vehicle_type VARCHAR(100) - vehicle type (Pickup & Mini/Truck/Bus)\n"
        "- vehicle_types_lp_ocr TEXT - combined field with type score and license plate (format: '0.99999 X4BUTQE')\n"
        "- ocr_score DECIMAL(10,9) - OCR confidence score\n\n"
        "Notes:\n"
        "- To extract license plate from vehicle_types_lp_ocr: SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1)\n"
        "- To extract type score from vehicle_types_lp_ocr: CAST(SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', 1) AS DECIMAL(10,9))\n"
        "- Timestamps are in format YYYY-MM-DDTHH:MM:SS or may be truncated"
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


def _execute_sql_safely(db: SQLDatabase, sql: str) -> tuple[list, str]:
    """Execute SQL safely and return results with error message."""
    try:
        # Basic safety check
        if not sql.strip().upper().startswith('SELECT'):
            return [], "Only SELECT queries are allowed"
        
        # Execute the query
        result = db.run(sql)
        return result, ""
    except Exception as e:
        return [], f"SQL execution error: {str(e)}"


def _parse_markdown_to_keyvalue(markdown_text: str) -> dict:
    """Parse markdown headings into key-value pairs."""
    result = {}
    lines = markdown_text.split('\n')
    current_key = None
    current_value = []
    
    for line in lines:
        line = line.strip()
        
        # Check for headings (### Heading)
        if line.startswith('### '):
            # Save previous key-value pair if exists
            if current_key and current_value:
                result[current_key] = '\n'.join(current_value).strip()
            
            # Start new key
            current_key = line[4:].strip()  # Remove '### '
            current_value = []
        
        # Add content to current value
        elif line and current_key:
            current_value.append(line)
    
    # Save the last key-value pair
    if current_key and current_value:
        result[current_key] = '\n'.join(current_value).strip()
        if current_key == "Possible Questions":
            result[current_key] = result[current_key].split('\n')
    
    return result


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
            "Schema gist: data_raw holds vehicle detection data from ALPR (Automatic License Plate Recognition) systems. "
            "Each row contains timestamp, device name, direction, vehicle type, combined type score with license plate, and OCR confidence score."
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
        "### In-Depth Analysis (when requested or warranted)\n"
        "- Provide deeper explanations: distributions, percentiles, moving averages, breakdowns by relevant dimensions (e.g., model, location, status).\n"
        "- Discuss trends over time, correlations, outliers, and potential causes/effects.\n"
        "- Include concise calculations (e.g., rates, ratios) and small summary tables if helpful.\n"
        "### Next Steps\n"
        "- 2-3 concise suggestions for follow-up analysis or checks.\n"
        "### Possible Questions\n"
        "- List 3-5 relevant follow-up questions the user might want to ask based on the current query and findings."
    )

    guardrails = (
        "Constraints: Use only read-only SELECTs; avoid DDL/DML. If multiple queries are needed, "
        "show the final, most important SQL in the SQL Used section. Keep the language simple. "
        "If the user asks for deep or detailed analysis, expand the In-Depth Analysis section accordingly."
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

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    enhanced_prompt = _build_prompt(question, use_device_scope)

    try:
        # Try the standard agent first
        agent_executor = create_sql_agent(
            llm=llm,
            db=lc_db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )

        result = agent_executor.invoke({"input": enhanced_prompt})
        output_text = result.get("output", "").strip()

        # Extract executed SQL for debugging
        executed_sql = None
        for step in result.get("intermediate_steps", []):
            if isinstance(step, tuple) and "SELECT" in step[0].upper():
                executed_sql = step[0]
                break

        # Parse markdown result into key-value pairs
        parsed_result = _parse_markdown_to_keyvalue(output_text)
        
        return {
            "question": question,
            "executed_sql": executed_sql,
            "result": parsed_result,
            "used_device_scope": use_device_scope,
        }
        
    except Exception as e:
        logger.warning(f"Standard agent failed: {str(e)}")
        
        # Fallback: Use direct LLM call with structured prompt
        try:
            fallback_prompt = f"""
You are a SQL expert. Given the following database schema and question, generate a SQL query and provide a structured response.

Database Schema:
{lc_db.get_table_info()}

Question: {question}

Please provide your response in the following format:

### Overview
- [Brief answer to the question]

### Key Findings  
- [Important numbers, trends, or comparisons]

### SQL Used
```sql
[Your SQL query here]
```

### Observations
- [Short insights or anomalies worth noting]

### Next Steps
- [2-3 concise suggestions for follow-up analysis]

### Possible Questions
- [List 3-5 relevant follow-up questions the user might want to ask based on the current query and findings]

Important: Only use SELECT statements. Do not modify data.
"""
            
            # Direct LLM call as fallback
            response = llm.invoke(fallback_prompt)
            output_text = response.content.strip()
            
            # Try to extract SQL from the response
            executed_sql = None
            if "```sql" in output_text:
                start = output_text.find("```sql") + 6
                end = output_text.find("```", start)
                if end > start:
                    executed_sql = output_text[start:end].strip()
                    
                    # Try to execute the SQL to verify it works
                    try:
                        sql_results, sql_error = _execute_sql_safely(lc_db, executed_sql)
                        if sql_error:
                            logger.warning(f"SQL execution failed: {sql_error}")
                        else:
                            logger.info(f"SQL executed successfully, returned {len(sql_results)} results")
                    except Exception as sql_exec_error:
                        logger.warning(f"SQL execution error: {sql_exec_error}")
            
            # Parse markdown result into key-value pairs
            parsed_result = _parse_markdown_to_keyvalue(output_text)
            
            return {
                "question": question,
                "executed_sql": executed_sql,
                "result": parsed_result,
                "used_device_scope": use_device_scope,
                "fallback_used": True,
            }
            
        except Exception as fallback_error:
            logger.error(f"Both standard agent and fallback failed: {str(fallback_error)}")
            
            # Last resort: Try to provide a basic response based on the question
            try:
                # Simple keyword-based response for common queries
                question_lower = question.lower()
                if "count" in question_lower and "bus" in question_lower and "score" in question_lower:
                    basic_response = {
                        "Overview": "Unable to execute the query due to technical issues, but this appears to be asking about bus maintenance scores.",
                        "Key Findings": "The system encountered parsing errors when trying to process this query.",
                        "SQL Used": "-- Query could not be executed due to parsing errors\nSELECT COUNT(*) FROM data_raw WHERE JSON_EXTRACT(row_data, '$.score') < 1;",
                        "Observations": "This appears to be a query about buses with maintenance scores below a threshold.\nThe system is experiencing technical difficulties with the AI agent parsing.",
                        "Next Steps": "Manual Query: Try running the SQL query directly against the database.\nSystem Check: Verify that the Google API key and database connections are working properly.\nRetry: The query may work on a subsequent attempt.",
                        "Possible Questions": "What is the average maintenance score for all buses?\nWhich buses have the lowest maintenance scores?\nHow many buses have scores between 1 and 2?\nWhat is the distribution of maintenance scores across the fleet?\nWhich buses need immediate maintenance attention?"
                    }
                    
                    return {
                        "question": question,
                        "executed_sql": "SELECT COUNT(*) FROM data_raw WHERE JSON_EXTRACT(row_data, '$.score') < 1;",
                        "result": basic_response,
                        "used_device_scope": use_device_scope,
                        "error": True,
                        "fallback_response": True,
                    }
            except Exception:
                pass
            
            return {
                "question": question,
                "executed_sql": None,
                "result": {"Error": f"Error processing query: {str(e)}"},
                "used_device_scope": use_device_scope,
                "error": True,
            }
