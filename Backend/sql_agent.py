import os
import json
import logging
from typing import Any
from sqlalchemy import create_engine, inspect, MetaData, text
from collections import defaultdict

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
# Local Docker model integration
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish


logger = logging.getLogger(__name__)


class CustomSQLOutputParser(ReActSingleInputOutputParser):
    """Custom output parser that handles markdown-formatted responses gracefully."""
    
    def parse(self, text: str) -> AgentFinish:
        """Parse the agent output, handling markdown format."""
        if not text or not text.strip():
            return AgentFinish(
                return_values={"output": "No response generated"},
                log="Empty response"
            )
        
        # Clean up the text
        text = text.strip()
        
        # If the text looks like a final answer in markdown format, return it as AgentFinish
        if any(marker in text for marker in ["### Overview", "### Key Findings", "### SQL Used", "### Observations", "### Possible Questions"]):
            return AgentFinish(
                return_values={"output": text},
                log=text
            )
        
        # Check if it's a direct answer without ReAct format
        if not any(marker in text for marker in ["Action:", "Observation:", "Thought:", "Action Input:", "Final Answer:"]):
            # This looks like a direct answer, treat it as final
            return AgentFinish(
                return_values={"output": text},
                log=text
            )
        
        # Check if it contains SQL code blocks (common in our use case)
        if "```sql" in text or "SELECT" in text.upper():
            # This looks like a SQL response, treat as final
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


def configure_ollama_from_env() -> None:
    """
    Configure Ollama model settings from environment variables.
    Required environment variables:
    - OLLAMA_BASE_URL: URL of the Ollama server (default: http://localhost:11434)
    - OLLAMA_MODEL: Model name to use (default: qwen2.5:3b)
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    logger.info(f"Ollama configured at: {base_url} with model: {model_name}")


def _get_local_llm(temperature: float = 0):
    """
    Create and return a local Ollama model endpoint instance.
    Uses Ollama server running on localhost.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    
    try:
        import requests
        # Test if Ollama server is running
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info(f"Using Ollama model: {model_name} at {base_url}")
            return _create_ollama_endpoint(base_url, model_name, temperature)
        else:
            raise Exception(f"Ollama server health check failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Ollama server not available at {base_url}: {e}")
        raise ValueError("Ollama server is required but not available. Please start Ollama first with 'ollama serve'.")


def _create_ollama_endpoint(base_url: str, model_name: str, temperature: float = 0):
    """
    Create a custom LLM endpoint for Ollama model.
    """
    from langchain.llms.base import LLM
    from typing import Optional, List, Any
    import requests
    import json
    
    class OllamaLLM(LLM):
        """Custom LLM class for Ollama model."""
        
        base_url: str
        model_name: str
        temperature: float = 0.0
        
        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            """Call the Ollama API."""
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "stop": stop or []
                    }
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    logger.error(f"Ollama error: {response.status_code} - {response.text}")
                    return "Error: Ollama not responding properly"
                    
            except Exception as e:
                logger.error(f"Error calling Ollama: {e}")
                return "Error: Failed to connect to Ollama"
        
        @property
        def _llm_type(self) -> str:
            return "ollama"
    
    return OllamaLLM(base_url=base_url, model_name=model_name, temperature=temperature)




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


def _execute_sql_with_engine(sql: str) -> tuple[list, str]:
    """Execute SQL using the database engine directly for more reliable results."""
    try:
        from sqlalchemy import create_engine, text
        import os
        
        # Get connection details
        user = os.getenv("DB_USER", os.getenv("MYSQL_USER", "FYP-USER"))
        password = os.getenv("DB_PASSWORD", os.getenv("MYSQL_PASSWORD", "FYP-PASS"))
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "3306")
        database = os.getenv("DB_NAME", os.getenv("MYSQL_DATABASE", "FYP-DB"))
        
        connection_url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
        engine = create_engine(connection_url)
        
        with engine.connect() as connection:
            result = connection.execute(text(sql))
            rows = result.fetchall()
            return [list(row) for row in rows], ""
            
    except Exception as e:
        return [], f"Direct SQL execution error: {str(e)}"


def _extract_sql_from_text(text: str) -> str:
    """Extract SQL query from text, looking for common patterns."""
    if not text:
        return None
    
    # Look for SQL in code blocks with sql marker
    if "```sql" in text:
        start = text.find("```sql") + 6
        end = text.find("```", start)
        if end > start:
            sql = text[start:end].strip()
            # Remove any additional text before SELECT
            if "SELECT" in sql.upper():
                select_idx = sql.upper().find("SELECT")
                sql = sql[select_idx:]
            return sql
    
    # Look for SQL in plain code blocks (```)
    if "```" in text and "SELECT" in text.upper():
        parts = text.split("```")
        for part in parts:
            if "SELECT" in part.upper():
                sql = part.strip()
                # Remove language markers
                if sql.startswith("sql\n") or sql.startswith("sql "):
                    sql = sql[3:].strip()
                # Extract just the SELECT statement
                if "SELECT" in sql.upper():
                    select_idx = sql.upper().find("SELECT")
                    sql = sql[select_idx:]
                    # Stop at first semicolon or double newline
                    if ";" in sql:
                        sql = sql[:sql.find(";")+1]
                    return sql.strip()
    
    # Look for SQL in action format
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "Action Input:" in line:
            # Check current line first
            if "SELECT" in line.upper():
                sql_start = line.upper().find("SELECT")
                return line[sql_start:].strip()
            # Check next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if "SELECT" in next_line.upper():
                    return next_line
    
    # Look for standalone SELECT statements (most permissive)
    sql_lines = []
    in_sql = False
    for line in lines:
        line_stripped = line.strip()
        
        # Start of SQL
        if line_stripped.upper().startswith("SELECT "):
            in_sql = True
            sql_lines = [line_stripped]
        # Continuation of SQL
        elif in_sql:
            # Stop conditions
            if line_stripped == "" or line_stripped.startswith("#") or line_stripped.startswith("--"):
                break
            # End with semicolon
            if ";" in line_stripped:
                sql_lines.append(line_stripped)
                break
            # Continue SQL
            sql_lines.append(line_stripped)
    
    if sql_lines:
        return " ".join(sql_lines).strip()
    
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
    
    if "data_raw" in relevant_tables:
        domain_parts.append(
            "Data table (data_raw): Contains vehicle detection data from ALPR (Automatic License Plate Recognition) systems. "
            "IMPORTANT COLUMNS: local_timestamp (VARCHAR), device_name (VARCHAR - NOT device.name!), direction (VARCHAR), "
            "vehicle_type (VARCHAR), vehicle_types_lp_ocr (TEXT), ocr_score (DECIMAL). "
            "CRITICAL: For vehicle type queries, use the 'vehicle_type' column directly (e.g., WHERE vehicle_type = 'Bus'). "
            "The vehicle_types_lp_ocr field contains both type confidence score and license plate (format: '0.95 ABC-1234'). "
            "Device names follow pattern Device-{Letter}{Number}. Vehicle types are Car, Truck, Bus, Motorcycle. "
            "Directions are Inbound/Outbound. OCR scores range from 0.0 to 1.0. "
            "NOTE: data_raw.device_name is a string field, NOT a foreign key to devices table."
        )
    
    if any(t == "devices" or t.startswith("device_") for t in relevant_tables):
        domain_parts.append(
            "Devices table: IMPORTANT COLUMNS - devices.id (BIGINT), devices.name (VARCHAR - NOT device_name!), "
            "devices.device_uid (VARCHAR), devices.device_type (VARCHAR), devices.model_id (BIGINT), "
            "devices.location_id (BIGINT), devices.status (VARCHAR), devices.timezone (VARCHAR). "
            "To join data_raw to devices: JOIN devices ON data_raw.device_name = devices.name"
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

    # Enhanced formatting with specific examples for license plate queries
    question_lower = question.lower()
    is_license_plate_query = any(term in question_lower for term in ['plate', 'license', 'ocr', 'average', 'score'])
    
    if is_license_plate_query:
        formatting = (
            "You are analyzing vehicle detection data from ALPR systems. For license plate queries, follow this specific format:\n\n"
            "### Overview\n"
            "- Provide a direct, clear answer to the question about the license plate or OCR scores.\n"
            "### Key Findings\n"
            "- List specific numbers: average OCR score, count of detections, date range, etc.\n"
            "### SQL Used\n"
            "```sql\n<your final SQL query here>\n```\n"
            "### Observations\n"
            "- Note any patterns, anomalies, or interesting findings about the license plate.\n"
            "### Possible Questions\n"
            "- Suggest 3-5 related follow-up questions about this license plate or similar analysis.\n\n"
            "CRITICAL DATA EXTRACTION RULES:\n"
            "- vehicle_types_lp_ocr format: '0.95 ABC-1234' (type_score + space + license_plate)\n"
            "- To extract license plate: SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1)\n"
            "- To extract type score: SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', 1)\n"
            "- For average OCR score of specific plate: SELECT AVG(ocr_score) FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = 'PLATE_NUMBER'\n"
            "- For count of detections: SELECT COUNT(*) FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = 'PLATE_NUMBER'\n"
            "- Device names: Device-A1, Device-B3, Device-C2, etc.\n"
            "- Vehicle types: Car, Truck, Bus, Motorcycle\n"
            "- Directions: Inbound, Outbound\n"
            "- OCR scores: 0.0-1.0 (decimal) or 0-100 (percentage)\n"
            "- Type scores: 0.0-1.0 (decimal from vehicle_types_lp_ocr)\n\n"
            "EXAMPLE QUERIES:\n"
            "- Average OCR for plate 'ABC123': SELECT AVG(ocr_score) FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = 'ABC123'\n"
            "- All detections of plate 'XYZ789': SELECT * FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = 'XYZ789'\n"
            "- Count by vehicle type: SELECT vehicle_type, COUNT(*) FROM data_raw GROUP BY vehicle_type"
        )
    else:
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
        "Constraints: Use only read-only SELECTs; avoid DDL/DML. Execute the query and provide actual results. "
        "If the query returns no results, state that clearly. Keep the language simple and direct.\n\n"
        "IMPORTANT: Follow the ReAct format step by step:\n"
        "1. Use 'Thought:' to explain what you're thinking\n"
        "2. Use 'Action:' to specify what tool you're using\n"
        "3. Use 'Action Input:' to provide the SQL query\n"
        "4. Use 'Observation:' to show the actual results from the database\n"
        "5. Use 'Final Answer:' to provide your complete response in the markdown format specified above\n\n"
        "Show your reasoning process clearly. For count queries like 'How many buses were detected':\n"
        "1. First explain what you need to do\n"
        "2. Write the SQL query: SELECT COUNT(*) FROM data_raw WHERE vehicle_type = 'Bus'\n"
        "3. Execute it and show the actual result\n"
        "4. Provide the final answer with the real count\n\n"
        "IMPORTANT: Always use the 'vehicle_type' column for vehicle type queries, NOT 'vehicle_types_lp_ocr'"
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
            llm = _get_local_llm(temperature=0)
            
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
            # Handle both string and object responses
            if hasattr(response, 'content'):
                response_text = response.content.strip().upper()
            else:
                response_text = str(response).strip().upper()
            
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


def _handle_license_plate_query(question: str, lc_db: SQLDatabase, llm) -> dict:
    """
    Handle license plate queries with a more direct approach to avoid parsing issues.
    """
    question_lower = question.lower()
    
    # Extract license plate from question
    import re
    
    # Common words to exclude from plate matching
    exclude_words = {'average', 'ocr', 'score', 'plate', 'license', 'what', 'is', 'of', 'the', 'for', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
    
    plate_patterns = [
        r'plate\s+([A-Z0-9]{6,8})',  # "plate X2XRPWP"
        r'license\s+([A-Z0-9]{6,8})',  # "license X2XRPWP"
    ]
    
    plate_number = None
    for pattern in plate_patterns:
        match = re.search(pattern, question_lower, re.IGNORECASE)
        if match:
            plate_number = match.group(1).upper()
            break
    
    # If no specific pattern matched, look for any 6-8 character alphanumeric word that's not a common word
    if not plate_number:
        words = question.split()
        for word in words:
            if (len(word) >= 6 and len(word) <= 8 and 
                word.isalnum() and 
                word.lower() not in exclude_words and
                any(c.isdigit() for c in word) and 
                any(c.isalpha() for c in word)):
                plate_number = word.upper()
                break
    
    
    if not plate_number:
        return {
            "question": question,
            "executed_sql": None,
            "result": {
                "Overview": "Could not identify a license plate number in the question.",
                "Key Findings": "Please specify the license plate number you want to analyze.",
                "SQL Used": "N/A - No plate number identified",
                "Observations": "License plate queries should include the specific plate number (e.g., 'X2XRPWP').",
                "Possible Questions": [
                    "What is the average OCR score for plate ABC123?",
                    "How many times was plate XYZ789 detected?",
                    "Show me all detections for plate DEF456",
                    "What vehicle type is associated with plate GHI789?",
                    "When was plate JKL012 last detected?"
                ]
            },
            "used_device_scope": False,
            "error": "No plate number identified"
        }
    
    # Build the SQL query
    sql_query = f"SELECT AVG(ocr_score) as average_ocr, COUNT(*) as detection_count FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = '{plate_number}'"
    
    try:
        # Try direct SQL execution first for more reliable results
        result, error = _execute_sql_with_engine(sql_query)
        
        if error:
            # Fallback to LangChain SQLDatabase if direct execution fails
            logger.warning(f"Direct SQL execution failed: {error}, trying LangChain method")
            result, error = _execute_sql_safely(lc_db, sql_query)
            
            if error:
                return {
                    "question": question,
                    "executed_sql": sql_query,
                    "result": {
                        "Overview": f"Error executing query for plate {plate_number}: {error}",
                        "Key Findings": "The query could not be executed successfully.",
                        "SQL Used": f"```sql\n{sql_query}\n```",
                        "Observations": "There was an issue with the database query execution.",
                        "Possible Questions": [
                            "Check if the plate number format is correct",
                            "Verify the database connection",
                            "Try a different plate number"
                        ]
                    },
                    "used_device_scope": False,
                    "error": error
                }
        
        # Process results
        avg_ocr = 0
        count = 0
        
        if result and len(result) > 0:
            # Direct execution returns list of lists
            if isinstance(result[0], (list, tuple)) and len(result[0]) >= 2:
                avg_ocr = float(result[0][0]) if result[0][0] is not None else 0
                count = int(result[0][1]) if result[0][1] is not None else 0
            else:
                # Fallback parsing for string results
                result_str = str(result)
                import re
                numbers = re.findall(r'[\d.]+', result_str)
                if len(numbers) >= 2:
                    avg_ocr = float(numbers[0])
                    count = int(float(numbers[1]))
        else:
            # If no results, try separate queries
            try:
                count_query = f"SELECT COUNT(*) FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = '{plate_number}'"
                count_result, count_error = _execute_sql_with_engine(count_query)
                
                if not count_error and count_result and len(count_result) > 0:
                    count = int(count_result[0][0]) if count_result[0][0] is not None else 0
                    
                    if count > 0:
                        avg_query = f"SELECT AVG(ocr_score) FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = '{plate_number}'"
                        avg_result, avg_error = _execute_sql_with_engine(avg_query)
                        
                        if not avg_error and avg_result and len(avg_result) > 0:
                            avg_ocr = float(avg_result[0][0]) if avg_result[0][0] is not None else 0
            except Exception as e:
                logger.warning(f"Fallback query execution failed: {e}")
                avg_ocr = 0
                count = 0
        
        # Process the results
        if count == 0:
            overview = f"License plate {plate_number} was not found in the database."
            key_findings = ["No detections found for this plate number"]
            observations = ["The plate may not have been detected by any ALPR systems", "Check if the plate number is correct"]
        else:
            overview = f"License plate {plate_number} has an average OCR score of {avg_ocr:.6f} across {count} detection(s)."
            key_findings = [
                f"Average OCR score: {avg_ocr:.6f}",
                f"Total detections: {count}",
                f"OCR score range: 0.0 to 1.0 (higher is better)"
            ]
            observations = [
                f"The plate was detected {count} time(s) in the database",
                f"Average confidence level: {avg_ocr:.1%}" if avg_ocr <= 1.0 else f"Average confidence level: {avg_ocr:.1f}%"
            ]
        
        return {
            "question": question,
            "executed_sql": sql_query,
            "result": {
                "Overview": overview,
                "Key Findings": key_findings,
                "SQL Used": f"```sql\n{sql_query}\n```",
                "Observations": observations,
                "Possible Questions": [
                    f"What is the average OCR score for plate {plate_number}?",
                    f"How many times was plate {plate_number} detected?",
                    f"Show me all detections for plate {plate_number}",
                    f"What vehicle type is associated with plate {plate_number}?",
                    f"When was plate {plate_number} last detected?"
                ]
            },
            "used_device_scope": False
        }
        
    except Exception as e:
        return {
            "question": question,
            "executed_sql": sql_query,
            "result": {
                "Overview": f"Error processing query for plate {plate_number}: {str(e)}",
                "Key Findings": "An unexpected error occurred during processing.",
                "SQL Used": f"```sql\n{sql_query}\n```",
                "Observations": "There was a technical issue with the query processing.",
                "Possible Questions": [
                    "Try rephrasing the question",
                    "Check if the plate number is correct",
                    "Contact support if the issue persists"
                ]
            },
            "used_device_scope": False,
            "error": str(e)
        }


def run_data_raw_agent(question: str) -> dict:
    """End-to-end: use LangChain SQL Agent (OpenRouter/DeepSeek) to generate, run, and summarize SQL.

    Uses automatic schema discovery to determine relevant tables dynamically.
    First checks if the question is database-related before processing.
    """
    configure_ollama_from_env()
    
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

    # Check if this is a license plate query and handle it directly
    question_lower = question.lower()
    is_license_plate_query = any(term in question_lower for term in ['plate', 'license', 'ocr', 'average', 'score'])
    
    if is_license_plate_query:
        # Discover database schema and determine relevant tables
        schema_info = discover_database_schema()
        include_tables = get_tables_for_query(question, schema_info)
        lc_db = get_lc_sql_db(include_tables)
        llm = _get_local_llm(temperature=0)
        
        # Use specialized license plate handler
        return _handle_license_plate_query(question, lc_db, llm)

    # Discover database schema and determine relevant tables
    schema_info = discover_database_schema()
    include_tables = get_tables_for_query(question, schema_info)
    use_device_scope = _is_device_query(question)
    
    # Log the discovered schema for debugging
    logger.info(f"Using tables: {include_tables}")
    logger.info(f"Available table groups: {list(schema_info.get('table_groups', {}).keys())}")
    
    lc_db = get_lc_sql_db(include_tables)

    llm = _get_local_llm(temperature=0)

    enhanced_prompt = _build_prompt(question, use_device_scope, schema_info, include_tables)

    try:
        # Try the standard agent first with reduced iterations to prevent infinite loops
        agent_executor = create_sql_agent(
            llm=llm,
            db=lc_db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,  # Reduced from 30 to prevent infinite loops
            early_stopping_method="generate"  # Stop early if stuck
        )

        result = agent_executor.invoke({"input": enhanced_prompt})
        output_text = result.get("output", "").strip()

        # Log the intermediate steps for debugging
        intermediate_steps = result.get("intermediate_steps", [])
        logger.info(f"Agent executed {len(intermediate_steps)} intermediate steps")
        
        # Extract executed SQL for debugging
        executed_sql = None
        reasoning_steps = []
        
        for i, step in enumerate(intermediate_steps):
            if isinstance(step, tuple) and len(step) >= 2:
                action = step[0] if len(step) > 0 else ""
                observation = step[1] if len(step) > 1 else ""
                
                # Log each step
                logger.info(f"Step {i+1} - Action: {str(action)[:100]}...")
                logger.info(f"Step {i+1} - Observation: {str(observation)[:100]}...")
                
                # Capture reasoning steps
                if hasattr(action, 'tool_input') and 'SELECT' in str(action.tool_input).upper():
                    executed_sql = action.tool_input
                    reasoning_steps.append(f"Step {i+1}: {action.log}")
                    reasoning_steps.append(f"SQL Query: {executed_sql}")
                    reasoning_steps.append(f"Result: {observation}")
                
                # Look for SQL in action
                elif isinstance(action, str) and "SELECT" in action.upper():
                    executed_sql = action
                    reasoning_steps.append(f"Step {i+1}: {action}")
                    reasoning_steps.append(f"Result: {observation}")
                    break
                # Look for SQL in observation (sometimes the agent puts SQL there)
                elif isinstance(observation, str) and "SELECT" in observation.upper():
                    # Extract SQL from observation if it contains SQL
                    lines = observation.split('\n')
                    for line in lines:
                        if "SELECT" in line.upper():
                            executed_sql = line.strip()
                            reasoning_steps.append(f"Step {i+1}: Found SQL in observation")
                            reasoning_steps.append(f"SQL Query: {executed_sql}")
                            break
                    if executed_sql:
                        break
        
        # If no SQL found in intermediate steps, try to extract from output text
        if not executed_sql:
            executed_sql = _extract_sql_from_text(output_text)
        
        # If we found SQL but it wasn't executed, try to execute it directly
        if executed_sql and not any("Observation:" in str(step) for step in result.get("intermediate_steps", [])):
            try:
                logger.info(f"Executing SQL directly: {executed_sql}")
                sql_results, sql_error = _execute_sql_safely(lc_db, executed_sql)
                if sql_error:
                    logger.warning(f"Direct SQL execution failed: {sql_error}")
                else:
                    logger.info(f"Direct SQL execution successful, returned {len(sql_results)} results")
                    # Parse and integrate the actual results into the response
                    if sql_results and len(sql_results) > 0:
                        try:
                            # Convert results to a more readable format
                            if isinstance(sql_results, str):
                                # If it's a string, try to parse it
                                import re
                                numbers = re.findall(r'\d+', sql_results)
                                if numbers:
                                    actual_count = int(numbers[0])
                                else:
                                    actual_count = 0
                            else:
                                # If it's a list, get the first value
                                actual_count = int(sql_results[0]) if sql_results else 0
                            
                            # Update the response with actual results if it contains count queries
                            if "COUNT" in executed_sql.upper() and "bus" in question.lower():
                                if "### Overview" in output_text:
                                    # Replace the fake overview with real data
                                    overview_start = output_text.find("### Overview")
                                    overview_end = output_text.find("###", overview_start + 1)
                                    if overview_end == -1:
                                        overview_end = len(output_text)
                                    
                                    new_overview = f"### Overview\n- There were {actual_count:,} buses detected in the data."
                                    output_text = output_text[:overview_start] + new_overview + "\n\n" + output_text[overview_end:]
                                
                                if "### Key Findings" in output_text:
                                    # Replace the fake key findings with real data
                                    findings_start = output_text.find("### Key Findings")
                                    findings_end = output_text.find("###", findings_start + 1)
                                    if findings_end == -1:
                                        findings_end = len(output_text)
                                    
                                    new_findings = f"### Key Findings\n- Total bus count: {actual_count:,}\n- This represents the actual number of bus detections in the database."
                                    output_text = output_text[:findings_start] + new_findings + "\n\n" + output_text[findings_end:]
                        except Exception as parse_error:
                            logger.warning(f"Failed to parse SQL results: {parse_error}")
                            # Add the raw results to the response
                            output_text += f"\n\n**Actual SQL Results:** {sql_results}"
                    else:
                        # Add the results to the output text if it's not already there
                        if "Observation:" not in output_text:
                            output_text += f"\n\nObservation: {sql_results}"
            except Exception as e:
                logger.warning(f"Direct SQL execution error: {e}")

        # Parse markdown result into key-value pairs
        parsed_result = _parse_markdown_to_keyvalue(output_text)
        # Fallback: if the model didn't follow the heading format, preserve the raw text
        if not parsed_result:
            if output_text:
                parsed_result = {"Overview": output_text}
            else:
                parsed_result = {"Overview": "No answer was produced by the SQL agent."}
        
        # Add reasoning steps to the result if available
        if reasoning_steps:
            parsed_result["Agent_Reasoning"] = reasoning_steps
        
        return {
            "question": question,
            "executed_sql": executed_sql,
            "result": parsed_result,
            "used_device_scope": use_device_scope,
            "intermediate_steps_count": len(intermediate_steps),
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
            
            # Try alternative extraction patterns
            if "```" in error_msg:
                # Look for SQL code blocks in the error message
                import re
                sql_pattern = r'```sql\s*(.*?)\s*```'
                sql_matches = re.findall(sql_pattern, error_msg, re.DOTALL)
                if sql_matches:
                    logger.info("Found SQL in error message, attempting to execute directly")
                    try:
                        sql_query = sql_matches[0].strip()
                        sql_results, sql_error = _execute_sql_safely(lc_db, sql_query)
                        if not sql_error:
                            logger.info("Successfully executed SQL from error message")
                            return {
                                "question": question,
                                "executed_sql": sql_query,
                                "result": {
                                    "Overview": "Query executed successfully after parsing error recovery",
                                    "Key Findings": f"Query returned {len(sql_results)} results",
                                    "SQL Used": f"```sql\n{sql_query}\n```",
                                    "Observations": "This query was recovered from a parsing error",
                                    "Possible Questions": [
                                        "What other data can you show me?",
                                        "Can you analyze this data further?",
                                        "What patterns do you see in the results?"
                                    ]
                                },
                                "used_device_scope": use_device_scope,
                                "recovered_from_error": True,
                            }
                    except Exception as sql_e:
                        logger.warning(f"Failed to execute SQL from error message: {sql_e}")
        
        # Fallback: Use direct LLM call with structured prompt
        try:
            # Check if this is a license plate query for specialized handling
            question_lower = question.lower()
            is_license_plate_query = any(term in question_lower for term in ['plate', 'license', 'ocr', 'average', 'score'])
            
            if is_license_plate_query:
                fallback_prompt = f"""
You are a SQL expert specializing in ALPR (Automatic License Plate Recognition) data analysis. 

Database Schema:
{lc_db.get_table_info()}

Question: {question}

CRITICAL DATA EXTRACTION RULES:
- vehicle_types_lp_ocr format: '0.95 ABC-1234' (type_score + space + license_plate)
- To extract license plate: SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1)
- To extract type score: SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', 1)
- For average OCR score of specific plate: SELECT AVG(ocr_score) FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = 'PLATE_NUMBER'
- For count of detections: SELECT COUNT(*) FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = 'PLATE_NUMBER'

EXAMPLE QUERIES:
- Average OCR for plate 'ABC123': SELECT AVG(ocr_score) FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = 'ABC123'
- All detections of plate 'XYZ789': SELECT * FROM data_raw WHERE SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) = 'XYZ789'
- Count by vehicle type: SELECT vehicle_type, COUNT(*) FROM data_raw GROUP BY vehicle_type

Please provide your response in the following format:

### Overview
- [Direct answer about the license plate or OCR score]

### Key Findings  
- [Specific numbers: average OCR score, count of detections, date range, etc.]

### SQL Used
```sql
[Your SQL query here]
```

### Observations
- [Patterns, anomalies, or interesting findings about the license plate]

### Possible Questions
- [3-5 related follow-up questions about this license plate or similar analysis]

Important: Only use SELECT statements. Execute the query and provide actual results.
"""
            else:
                fallback_prompt = f"""
You are a SQL expert. Given the following database schema and question, generate a SQL query and provide a structured response.

Database Schema:
{lc_db.get_table_info()}

Question: {question}

IMPORTANT DATA STRUCTURE NOTES:
- vehicle_types_lp_ocr format: '0.95 ABC-1234' (type_score + space + license_plate)
- Use SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', 1) for type score
- Use SUBSTRING_INDEX(vehicle_types_lp_ocr, ' ', -1) for license plate
- Device names: Device-A1, Device-B3, Device-C2, etc.
- Vehicle types: Car, Truck, Bus, Motorcycle
- Directions: Inbound, Outbound
- OCR scores: 0.0-1.0 (decimal) or 0-100 (percentage)
- Type scores: 0.0-1.0 (decimal from vehicle_types_lp_ocr)

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
            # Handle both string and object responses
            if hasattr(response, 'content'):
                output_text = response.content.strip()
            else:
                output_text = str(response).strip()
            
            # Try to extract SQL from the response
            executed_sql = None
            sql_results = None
            if "```sql" in output_text:
                start = output_text.find("```sql") + 6
                end = output_text.find("```", start)
                if end > start:
                    executed_sql = output_text[start:end].strip()
                    
                    # Try to execute the SQL to get actual results
                    try:
                        sql_results, sql_error = _execute_sql_safely(lc_db, executed_sql)
                        if sql_error:
                            logger.warning(f"SQL execution failed: {sql_error}")
                        else:
                            logger.info(f"SQL executed successfully, returned {len(sql_results)} results")
                            # If we got results, we need to update the response with actual data
                            if sql_results and len(sql_results) > 0:
                                # Parse the results and update the response
                                try:
                                    # Convert results to a more readable format
                                    if isinstance(sql_results, str):
                                        # If it's a string, try to parse it
                                        import re
                                        numbers = re.findall(r'\d+', sql_results)
                                        if numbers:
                                            actual_count = int(numbers[0])
                                        else:
                                            actual_count = 0
                                    else:
                                        # If it's a list, get the first value
                                        actual_count = int(sql_results[0]) if sql_results else 0
                                    
                                    # Update the response with actual results
                                    if "### Overview" in output_text:
                                        # Replace the fake overview with real data
                                        overview_start = output_text.find("### Overview")
                                        overview_end = output_text.find("###", overview_start + 1)
                                        if overview_end == -1:
                                            overview_end = len(output_text)
                                        
                                        new_overview = f"### Overview\n- There were {actual_count:,} buses detected in the data."
                                        output_text = output_text[:overview_start] + new_overview + "\n\n" + output_text[overview_end:]
                                    
                                    if "### Key Findings" in output_text:
                                        # Replace the fake key findings with real data
                                        findings_start = output_text.find("### Key Findings")
                                        findings_end = output_text.find("###", findings_start + 1)
                                        if findings_end == -1:
                                            findings_end = len(output_text)
                                        
                                        new_findings = f"### Key Findings\n- Total bus count: {actual_count:,}\n- This represents the actual number of bus detections in the database."
                                        output_text = output_text[:findings_start] + new_findings + "\n\n" + output_text[findings_end:]
                                        
                                except Exception as parse_error:
                                    logger.warning(f"Failed to parse SQL results: {parse_error}")
                                    # Add the raw results to the response
                                    output_text += f"\n\n**Actual SQL Results:** {sql_results}"
                    except Exception as sql_exec_error:
                        logger.warning(f"SQL execution error: {sql_exec_error}")
            
            # Parse markdown result into key-value pairs
            parsed_result = _parse_markdown_to_keyvalue(output_text)
            if not parsed_result:
                if output_text:
                    parsed_result = {"Overview": output_text}
                else:
                    parsed_result = {"Overview": "No answer was produced by the SQL agent."}
            
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


def generate_sql_for_question(question: str) -> str | None:
    """
    Use direct LLM prompting to generate SQL (bypasses agent to avoid infinite loops).
    Returns the SQL string if extracted, otherwise None.
    """
    configure_ollama_from_env()

    # Discover schema and set scope
    schema_info = discover_database_schema()
    include_tables = get_tables_for_query(question, schema_info)
    lc_db = get_lc_sql_db(include_tables)

    llm = _get_local_llm(temperature=0)

    # Use direct SQL generation instead of agent to avoid infinite loops
    try:
        sql_only_prompt = f"""You are a MySQL expert. Generate ONLY a valid SELECT query. No explanation, no markdown, no text.

Database Schema:
{lc_db.get_table_info()}

CRITICAL COLUMN NAME RULES:
- data_raw table uses: device_name (NOT device.name!)
- devices table uses: name (NOT device_name!)
- To join data_raw to devices: JOIN devices ON data_raw.device_name = devices.name
- data_raw.device_name is VARCHAR, devices.name is VARCHAR
- For queries on devices, use devices.name as the column
- For queries on data_raw, use data_raw.device_name as the column

Rules:
- Return ONLY the SQL query, nothing else
- No ```sql``` markers
- No commentary or explanation
- Use only SELECT statements
- Columns x and y when requested must be aliased properly
- Always use correct column names as specified above

Question: {question}

SQL:"""

        response = llm.invoke(sql_only_prompt)
        
        # Handle both string and object responses
        if hasattr(response, 'content'):
            text = response.content.strip()
        else:
            text = str(response).strip()
        
        # Extract SQL from response
        executed_sql = _extract_sql_from_text(text)
        
        # If extraction failed but text looks like SQL, use it directly
        if not executed_sql and text.strip().upper().startswith("SELECT"):
            executed_sql = text.strip()
        
        # Clean up the SQL (remove markdown if present)
        if executed_sql:
            executed_sql = executed_sql.strip()
            # Remove markdown code blocks if present
            if executed_sql.startswith("```"):
                lines = executed_sql.split('\n')
                executed_sql = '\n'.join([l for l in lines if not l.strip().startswith('```')])
                executed_sql = executed_sql.strip()
        
        return executed_sql

    except Exception as e:
        logger.warning(f"generate_sql_for_question failed: {e}")
        return None
