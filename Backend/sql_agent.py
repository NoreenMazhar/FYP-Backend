import os
import json
import logging
from typing import Any
from sqlalchemy import create_engine, inspect, MetaData, text
from collections import defaultdict

import google.generativeai as genai
from openai import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish


logger = logging.getLogger(__name__)


class CustomSQLOutputParser(ReActSingleInputOutputParser):
    """Custom output parser that handles markdown-formatted responses gracefully."""
    
    def parse(self, text: str) -> AgentFinish:
        """Parse the agent output, handling markdown format."""
        # If the text looks like a final answer in markdown format, return it as AgentFinish
        if any(marker in text for marker in ["### Overview", "### Key Findings", "### SQL Used", "### Observations"]):
            return AgentFinish(
                return_values={"output": text},
                log=text
            )
        
        # Check if it's a direct answer without ReAct format
        if not any(marker in text for marker in ["Action:", "Observation:", "Thought:"]):
            # This looks like a direct answer, treat it as final
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
    # No-op (Gemini removed for SQL). Keeping for compatibility.
    return


def _get_ollama_client() -> OpenAI:
    """Create an OpenAI-compatible client pointed at Ollama (CPU-only) for summarization.

    Controlled via env:
    - OLLAMA_BASE_URL: default "http://ollama:11434/v1" (docker), fallback to "http://localhost:11434/v1" if not set
    - OLLAMA_API_KEY: not required by Ollama; use any placeholder
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
    api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    try:
        return OpenAI(base_url=base_url, api_key=api_key)
    except Exception:
        # Fallback to localhost for local dev if service name not resolvable
        return OpenAI(base_url="http://localhost:11434/v1", api_key=api_key)


def _summarize_with_ollama(question: str, executed_sql: str, rows: list) -> dict:
    """Use a lightweight CPU model via Ollama to turn SQL results into a user-facing answer.

    Returns a parsed dict with keys matching the UI expectation.
    """
    if rows is None:
        rows = []

    # Build a compact preview (first N rows) to keep CPU latency low
    max_preview_rows = int(os.getenv("SUMMARY_PREVIEW_ROWS", "30"))
    preview_rows = rows[:max_preview_rows]

    # Ensure JSON-serializable preview
    try:
        preview_json = json.dumps(preview_rows, ensure_ascii=False, default=str)
    except Exception:
        preview_json = json.dumps([], ensure_ascii=False)

    client = _get_ollama_client()
    model = os.getenv("OLLAMA_ANSWER_MODEL", "phi3.5:mini-instruct-q4_K_M")

    system_prompt = (
        "You are a precise data analyst. Answer ONLY using the provided SQL results. "
        "If data is insufficient, say so. Keep the answer concise and structured."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Executed SQL:\n{executed_sql or 'N/A'}\n\n"
        f"Row count: {len(rows)}\n"
        f"Preview (JSON, first {len(preview_rows)} rows):\n{preview_json}\n\n"
        "Return Markdown with sections: \n"
        "### Overview\n- Direct answer in plain language.\n"
        "### Key Findings\n- Bulleted numeric highlights.\n"
        "### SQL Used\n```sql\n<the SQL used>\n```\n"
        "### Observations\n- Short insights or caveats.\n"
        "### Possible Questions\n- 3-5 follow-ups.\n"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=int(os.getenv("SUMMARY_MAX_TOKENS", "400")),
        )
        text = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Ollama summarization failed: {e}")
        # Minimal fallback
        text = (
            "### Overview\n- Generated summary unavailable. Showing raw results preview.\n\n"
            "### Key Findings\n- Rows returned: " + str(len(rows)) + "\n\n"
            "### SQL Used\n```sql\n" + (executed_sql or "-- unknown") + "\n```\n\n"
            "### Observations\n- Consider narrowing the query to fewer rows.\n\n"
            "### Possible Questions\n- What timeframe matters?\n- Which devices?\n- What thresholds?\n"
        )

    return _parse_markdown_to_keyvalue(text)


def _generate_sql_with_ollama(question: str, lc_db: SQLDatabase) -> str | None:
    """Use a local code-focused model to generate safe SELECT-only SQL for MySQL.

    Model/env:
    - OLLAMA_SQL_MODEL: default "qwen2.5-coder:7b-q4_K_M"
    """
    client = _get_ollama_client()
    model = os.getenv("OLLAMA_SQL_MODEL", "qwen2.5-coder:7b-q4_K_M")

    # Provide compact table info to keep CPU latency reasonable
    try:
        table_info = lc_db.get_table_info()
    except Exception:
        table_info = ""

    system_prompt = (
        "You write safe, executable MySQL SQL.\n"
        "Rules: Only SELECT; no DDL/DML; use valid columns; add LIMIT 100 unless specified;\n"
        "Return ONLY SQL, no prose."
    )
    user_prompt = (
        f"Schema:\n{table_info}\n\n"
        f"Question:\n{question}\n\n"
        "Return only one SQL statement (no Markdown)."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        sql_text = resp.choices[0].message.content.strip()
        if not sql_text:
            return None
        # Basic guardrails
        if not _is_safe_select(sql_text):
            return None
        return sql_text
    except Exception as e:
        logger.warning(f"Ollama SQL generation failed: {e}")
        return None


def _get_sqlalchemy_url_from_env() -> str:
    user = os.getenv("DB_USER", os.getenv("MYSQL_USER", "FYP-USER"))
    password = os.getenv("DB_PASSWORD", os.getenv("MYSQL_PASSWORD", "FYP-PASS"))
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    database = os.getenv("DB_NAME", os.getenv("MYSQL_DATABASE", "FYP-DB"))
    return f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"


# Caches for different table scopes
_LC_DBS: dict[str, SQLDatabase] = {}

# Cache for discovered schema
_DISCOVERED_SCHEMA: dict = None


def discover_database_schema(connection_url: str = None) -> dict:
    """
    Automatically discover all tables and their relationships from the database.
    Returns a dictionary with table names, their columns, and foreign key relationships.
    """
    global _DISCOVERED_SCHEMA
    
    if _DISCOVERED_SCHEMA is not None:
        return _DISCOVERED_SCHEMA
    
    if connection_url is None:
        connection_url = _get_sqlalchemy_url_from_env()
    
    try:
        # Create engine and inspector
        engine = create_engine(connection_url)
        inspector = inspect(engine)
        
        schema_info = {
            "tables": {},
            "relationships": [],
            "table_groups": {}
        }
        
        # Get all table names
        table_names = inspector.get_table_names()
        logger.info(f"Discovered {len(table_names)} tables: {table_names}")
        
        # Get detailed information for each table
        for table_name in table_names:
            columns = inspector.get_columns(table_name)
            primary_keys = inspector.get_pk_constraint(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            indexes = inspector.get_indexes(table_name)
            
            schema_info["tables"][table_name] = {
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col["nullable"],
                        "default": col.get("default")
                    }
                    for col in columns
                ],
                "primary_keys": primary_keys.get("constrained_columns", []),
                "foreign_keys": [
                    {
                        "columns": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"]
                    }
                    for fk in foreign_keys
                ],
                "indexes": [idx["name"] for idx in indexes]
            }
            
            # Build relationship list
            for fk in foreign_keys:
                schema_info["relationships"].append({
                    "from_table": table_name,
                    "from_columns": fk["constrained_columns"],
                    "to_table": fk["referred_table"],
                    "to_columns": fk["referred_columns"]
                })
        
        # Group tables by their relationships
        schema_info["table_groups"] = _group_tables_by_domain(schema_info)
        
        _DISCOVERED_SCHEMA = schema_info
        logger.info(f"Schema discovery complete. Found {len(table_names)} tables and {len(schema_info['relationships'])} relationships")
        
        engine.dispose()
        return schema_info
        
    except Exception as e:
        logger.error(f"Error discovering database schema: {e}")
        # Return a minimal schema with common tables as fallback
        return {
            "tables": {},
            "relationships": [],
            "table_groups": {
                "data": ["data_raw"],
                "devices": ["devices", "device_models"],
                "users": ["users"]
            }
        }


def _group_tables_by_domain(schema_info: dict) -> dict:
    """
    Group tables by domain based on naming patterns and relationships.
    Returns a dictionary mapping domain names to lists of table names.
    """
    groups = defaultdict(list)
    
    for table_name in schema_info["tables"].keys():
        # Categorize by table name prefix
        if table_name.startswith("device_"):
            groups["devices"].append(table_name)
        elif table_name.startswith("report_"):
            groups["reports"].append(table_name)
        elif table_name.startswith("chat_"):
            groups["chats"].append(table_name)
        elif table_name == "data_raw":
            groups["data"].append(table_name)
        elif table_name == "users":
            groups["users"].append(table_name)
        elif table_name == "visualizations":
            groups["visualizations"].append(table_name)
        elif table_name == "reports":
            groups["reports"].append(table_name)
        elif table_name == "chats":
            groups["chats"].append(table_name)
        else:
            groups["other"].append(table_name)
    
    # Also add base tables that have many relationships
    for table_name in schema_info["tables"].keys():
        related_count = sum(
            1 for rel in schema_info["relationships"]
            if rel["to_table"] == table_name
        )
        if related_count >= 3 and table_name not in groups["core"]:
            groups["core"].append(table_name)
    
    return dict(groups)


def get_tables_for_query(question: str, schema_info: dict = None) -> list[str]:
    """
    Determine which tables are relevant for a given question.
    Uses schema information and query keywords to intelligently select tables.
    """
    if schema_info is None:
        schema_info = discover_database_schema()
    
    question_lower = question.lower()
    relevant_tables = set()
    
    # Check if question mentions specific table names or domain keywords
    for table_name in schema_info["tables"].keys():
        # Direct table name mention
        if table_name in question_lower:
            relevant_tables.add(table_name)
            # Add related tables through foreign keys
            for rel in schema_info["relationships"]:
                if rel["from_table"] == table_name:
                    relevant_tables.add(rel["to_table"])
                elif rel["to_table"] == table_name:
                    relevant_tables.add(rel["from_table"])
    
    # If no specific tables found, use domain-based selection
    if not relevant_tables:
        groups = schema_info.get("table_groups", {})
        
        # Device-related keywords
        if _is_device_query(question):
            relevant_tables.update(groups.get("devices", []))
            relevant_tables.update(groups.get("users", []))  # Often needed for permissions
        # Data analysis keywords
        elif any(kw in question_lower for kw in ["vehicle", "license", "plate", "detection", "alpr", "traffic"]):
            relevant_tables.update(groups.get("data", []))
        # Report-related keywords
        elif any(kw in question_lower for kw in ["report", "visualization"]):
            relevant_tables.update(groups.get("reports", []))
            relevant_tables.update(groups.get("visualizations", []))
        # Chat-related keywords
        elif any(kw in question_lower for kw in ["chat", "conversation"]):
            relevant_tables.update(groups.get("chats", []))
        # Default: include data_raw and devices as they're most common
        else:
            relevant_tables.update(groups.get("data", []))
    
    # Always include core tables if they exist
    if "core" in schema_info.get("table_groups", {}):
        relevant_tables.update(schema_info["table_groups"]["core"])
    
    result = list(relevant_tables) if relevant_tables else list(schema_info["tables"].keys())
    logger.info(f"Selected {len(result)} tables for query: {result}")
    return result


def get_schema_summary(schema_info: dict = None) -> dict:
    """
    Get a human-readable summary of the discovered database schema.
    Useful for debugging and understanding the database structure.
    """
    if schema_info is None:
        schema_info = discover_database_schema()
    
    summary = {
        "total_tables": len(schema_info.get("tables", {})),
        "total_relationships": len(schema_info.get("relationships", [])),
        "table_groups": {},
        "tables": {}
    }
    
    # Summarize table groups
    for group_name, tables in schema_info.get("table_groups", {}).items():
        summary["table_groups"][group_name] = {
            "count": len(tables),
            "tables": tables
        }
    
    # Summarize each table
    for table_name, table_info in schema_info.get("tables", {}).items():
        summary["tables"][table_name] = {
            "columns": len(table_info.get("columns", [])),
            "primary_keys": table_info.get("primary_keys", []),
            "foreign_keys_count": len(table_info.get("foreign_keys", [])),
            "indexes_count": len(table_info.get("indexes", []))
        }
    
    return summary


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
        "- local_timestamp VARCHAR(255) - timestamp from Excel data (format: YYYY-MM-DD HH:MM:SS)\n"
        "- device_name VARCHAR(255) - device name (e.g., Device-A1, Device-B3, Device-C2)\n"
        "- direction VARCHAR(100) - direction (Inbound/Outbound)\n"
        "- vehicle_type VARCHAR(100) - vehicle type (Car/Truck/Bus/Motorcycle)\n"
        "- vehicle_types_lp_ocr TEXT - combined field with type score and license plate (format: '0.95 ABC-1234')\n"
        "- ocr_score DECIMAL(10,9) - OCR confidence score (0.0-1.0) or 0-100 (percentage)   \n\n"
        "Notes:\n"
        "- To extract license plate from vehicle_types_lp_ocr: SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1)\n"
        "- To extract type score from vehicle_types_lp_ocr: CAST(SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', 1) AS DECIMAL(10,9))\n"
        "- Timestamps are in format YYYY-MM-DD HH:MM:SS\n"
        "- Device names follow pattern: Device-{Letter}{Number}\n"
        "- Vehicle types are: Car, Truck, Bus, Motorcycle\n"
        "- Directions are: Inbound, Outbound"
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


def _extract_sql_from_text(text: str) -> str:
    """Extract SQL query from text, looking for common patterns."""
    if not text:
        return None
    
    # Look for SQL in code blocks
    if "```sql" in text:
        start = text.find("```sql") + 6
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    
    # Look for SQL in action format
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "Action Input:" in line and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if "SELECT" in next_line.upper():
                return next_line
    
    # Look for standalone SELECT statements
    for line in lines:
        line = line.strip()
        if line.upper().startswith("SELECT ") and not line.upper().startswith("SELECT * FROM"):
            return line
    
    return None


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


def _format_schema_relationships(schema_info: dict, relevant_tables: list[str]) -> str:
    """Format schema relationships in a human-readable way for the LLM."""
    if not schema_info or not relevant_tables:
        return ""
    
    relationships_text = []
    for rel in schema_info.get("relationships", []):
        if rel["from_table"] in relevant_tables and rel["to_table"] in relevant_tables:
            from_cols = ", ".join(rel["from_columns"])
            to_cols = ", ".join(rel["to_columns"])
            relationships_text.append(
                f"- {rel['from_table']}.{from_cols} → {rel['to_table']}.{to_cols}"
            )
    
    if relationships_text:
        return "\n\nTable Relationships:\n" + "\n".join(relationships_text)
    return ""


def _build_prompt(question: str, use_device_scope: bool, schema_info: dict = None, relevant_tables: list[str] = None) -> str:
    """Compose guidance plus the user question to steer the agent with dynamic schema information."""
    if schema_info is None:
        schema_info = discover_database_schema()
    
    if relevant_tables is None:
        relevant_tables = get_tables_for_query(question, schema_info)
    
    tables_text = ", ".join(relevant_tables)
    scope_instruction = (
        f"You must answer by querying the following tables: {tables_text}. "
        "These tables have been automatically selected as relevant to the question."
    )
    
    # Build domain context based on table groups
    groups = schema_info.get("table_groups", {})
    domain_parts = []
    
    if any(t.startswith("device_") or t == "devices" for t in relevant_tables):
        domain_parts.append(
            "Device tables: Track device inventory, models, firmware, configuration, streams, health, "
            "telemetry, events, permissions, groups, maintenance, and credentials."
        )
    
    if "data_raw" in relevant_tables:
        domain_parts.append(
            "Data table: Contains vehicle detection data from ALPR (Automatic License Plate Recognition) systems. "
            "Fields include timestamp, device name, direction, vehicle type, license plate with OCR score. "
            "The vehicle_types_lp_ocr field contains both type confidence score and license plate (format: '0.95 ABC-1234'). "
            "Device names follow pattern Device-{Letter}{Number}. Vehicle types are Car, Truck, Bus, Motorcycle. "
            "Directions are Inbound/Outbound. OCR scores range from 0.0 to 1.0."
        )
    
    if any(t in ["reports", "report_members", "report_visualizations"] for t in relevant_tables):
        domain_parts.append(
            "Report tables: Handle report creation, collaboration (multiple authors), and linking to visualizations."
        )
    
    if any(t in ["chats", "chat_visualizations"] for t in relevant_tables):
        domain_parts.append(
            "Chat tables: Manage user chat sessions and associated visualizations."
        )
    
    if "visualizations" in relevant_tables:
        domain_parts.append(
            "Visualizations: Reusable charts/graphs that can be included in both chats and reports."
        )
    
    domain_context = " ".join(domain_parts) if domain_parts else "Query the selected database tables."
    
    # Add relationship information
    relationship_context = _format_schema_relationships(schema_info, relevant_tables)

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
        "### Possible Questions\n"
        "- List 3-5 relevant follow-up questions the user might want to ask based on the current query and findings.\n"
        "\n"
        "IMPORTANT DATA STRUCTURE NOTES:\n"
        "- vehicle_types_lp_ocr format: '0.95 ABC-1234' (type_score + space + license_plate)\n"
        "- Use SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', 1) for type score\n"
        "- Use SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) for license plate\n"
        "- Device names: Device-A1, Device-B3, Device-C2, etc.\n"
        "- Vehicle types: Car, Truck, Bus, Motorcycle\n"
        "- Directions: Inbound, Outbound\n"
        "- OCR scores: 0.0-1.0 (decimal) or 0-100 (percentage)\n"
        "- Type scores: 0.0-1.0 (decimal from vehicle_types_lp_ocr)"
    )

    guardrails = (
        "Constraints: Use only read-only SELECTs; avoid DDL/DML. If multiple queries are needed, "
        "show the final, most important SQL in the SQL Used section. Keep the language simple. "
        "If the user asks for deep or detailed analysis, expand the In-Depth Analysis section accordingly.\n\n"
        "IMPORTANT: Follow the ReAct format. Use 'Action:' to specify what you're doing, 'Observation:' to show results, "
        "and 'Final Answer:' to provide your complete response in the markdown format specified above."
    )

    return (
        f"{scope_instruction}\n{domain_context}{relationship_context}\n{formatting}\n{guardrails}\n\n"
        f"User question: {question}"
    )


def _is_database_related_query(question: str) -> tuple[bool, str]:
    """
    Check if the question is related to the database content.
    Returns (is_related, reason) tuple.
    Uses both keyword matching and LLM-based validation.
    """
    question_lower = question.lower()
    
    # Database-related keywords for ALPR/vehicle detection data
    database_keywords = [
        # Vehicle-related
        'vehicle', 'car', 'truck', 'bus', 'pickup', 'mini',
        # ALPR-related
        'license', 'plate', 'ocr', 'detection', 'detected',
        # Direction-related
        'approaching', 'receding', 'direction',
        # Device-related
        'device', 'camera', 'sensor', 'neom',
        # Time-related queries
        'timestamp', 'time', 'date', 'when', 'hour', 'day', 'week', 'month',
        # Counting/statistics
        'how many', 'count', 'total', 'number of', 'average', 'sum',
        # Data queries
        'show', 'list', 'find', 'get', 'fetch', 'retrieve', 'search',
        'what', 'which', 'where',
        # Analysis
        'analyze', 'report', 'statistics', 'stats', 'summary',
        # Specific to our data
        'alpr', 'traffic', 'score', 'confidence'
    ]
    
    # First pass: Quick keyword check
    has_keyword = any(keyword in question_lower for keyword in database_keywords)
    
    # If no keywords found, use LLM for more sophisticated check
    if not has_keyword:
        try:
            # Gemini disabled; skip LLM validation on CPU-only
            raise RuntimeError("skip-llm-validation")
            
            validation_prompt = f"""You are a database query validator. The database contains vehicle detection data from ALPR (Automatic License Plate Recognition) systems with the following information:
- Vehicle detections with timestamps (format: YYYY-MM-DD HH:MM:SS)
- Device names (cameras/sensors) following pattern Device-{{Letter}}{{Number}}
- Vehicle types: Car, Truck, Bus, Motorcycle
- License plate numbers (extracted from vehicle_types_lp_ocr field)
- Direction of travel: Inbound, Outbound
- OCR confidence scores (0.0-1.0) or 0-100 (percentage)
- Type confidence scores (0.0-1.0)

Question: "{question}"

Is this question asking for information that could be found in this vehicle detection database?
Answer with ONLY "YES" or "NO" followed by a brief reason.

Examples:
- "How many trucks were detected?" -> YES - Asks about vehicle count
- "What is the weather today?" -> NO - Not related to vehicle detection data
- "Show me license plates from yesterday" -> YES - Asks about license plate data
- "What is Python?" -> NO - General programming question
- "Tell me a joke" -> NO - Not a data query
- "Which devices detected the most vehicles?" -> YES - Asks about device performance
- "What are the average OCR scores by vehicle type?" -> YES - Asks about data analysis

Your answer:"""
            
            response = llm.invoke(validation_prompt)
            response_text = response.content.strip().upper()
            
            if response_text.startswith("YES"):
                return True, ""
            else:
                reason = "The query you asked is not related to the database, so I can't answer it."
                return False, reason
                
        except Exception as e:
            # If LLM check fails, be conservative and reject
            logger.warning(f"LLM validation failed: {e}")
            reason = "The query you asked is not related to the database, so I can't answer it."
            return False, reason
    
    # Has relevant keywords, allow the query
    return True, ""


def run_data_raw_agent(question: str) -> dict:
    """End-to-end: use LangChain SQL Agent (Gemini) to generate, run, and summarize SQL.

    Uses automatic schema discovery to determine relevant tables dynamically.
    First checks if the question is database-related before processing.
    """
    configure_gemini_from_env()
    
    # Check if question is database-related
    is_related, rejection_reason = _is_database_related_query(question)
    if not is_related:
        return {
            "question": question,
            "executed_sql": None,
            "result": {
                "Overview": rejection_reason,
                "Key Findings": "This question cannot be answered using the database.",
                "SQL Used": "N/A - Question not related to database content",
                "Observations": "The question appears to be about topics outside the scope of the vehicle detection database.",
                "Next Steps": "Please ask questions related to vehicle detections, license plates, devices, timestamps, or traffic data."
            },
            "used_device_scope": False,
            "error": "Question not database-related"
        }

    # Discover database schema and determine relevant tables
    schema_info = discover_database_schema()
    include_tables = get_tables_for_query(question, schema_info)
    use_device_scope = _is_device_query(question)
    
    
    lc_db = get_lc_sql_db(include_tables)

    # Gemini removed; we'll use local model for SQL generation

    enhanced_prompt = _build_prompt(question, use_device_scope, schema_info, include_tables)

    try:
        # Try the standard agent first (Gemini to plan SQL)
        executed_sql = _generate_sql_with_ollama(question, lc_db)

        # Execute SQL ourselves to obtain rows for summarization
        sql_results = []
        if executed_sql:
            try:
                logger.info(f"Executing SQL: {executed_sql}")
                sql_results, sql_error = _execute_sql_safely(lc_db, executed_sql)
                if sql_error:
                    logger.warning(f"SQL execution failed: {sql_error}")
            except Exception as e:
                logger.warning(f"SQL execution error: {e}")

        # Summarize via Ollama model
        parsed_result = _summarize_with_ollama(question, executed_sql, sql_results)

        return {
            "question": question,
            "executed_sql": executed_sql,
            "result": parsed_result,
            "used_device_scope": use_device_scope,
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Standard agent failed: {error_msg}")
        
        # Check if it's a parsing error specifically
        if "Could not parse LLM output" in error_msg or "output parsing error" in error_msg.lower():
            logger.info("Detected parsing error, attempting to extract response from error message...")
            # Try to extract the actual response from the error message
            if "### Overview" in error_msg:
                # Extract the response from the error message
                start_marker = "Could not parse LLM output: `"
                end_marker = "`"
                start_idx = error_msg.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = error_msg.find(end_marker, start_idx)
                    if end_idx != -1:
                        extracted_response = error_msg[start_idx:end_idx]
                        logger.info("Successfully extracted response from error message")
                        parsed_result = _parse_markdown_to_keyvalue(extracted_response)
                        return {
                            "question": question,
                            "executed_sql": None,
                            "result": parsed_result,
                            "used_device_scope": use_device_scope,
                            "extracted_from_error": True,
                        }
        
        # Fallback: regenerate SQL with local model and summarize
        try:
            executed_sql = _generate_sql_with_ollama(question, lc_db)
            sql_results = []
            if executed_sql:
                try:
                    sql_results, sql_error = _execute_sql_safely(lc_db, executed_sql)
                    if sql_error:
                        logger.warning(f"SQL execution failed: {sql_error}")
                except Exception as sql_exec_error:
                    logger.warning(f"SQL execution error: {sql_exec_error}")

            parsed_result = _summarize_with_ollama(question, executed_sql, sql_results)

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
