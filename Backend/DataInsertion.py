import os
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from db import Database


def ensure_data_raw_table(db: Database) -> None:
    """Create the data_raw table if it does not exist (MySQL compatible)."""
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS data_raw (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            batch_id CHAR(36) NOT NULL,
            source_file VARCHAR(512) NOT NULL,
            row_index INT NOT NULL,
            row_data JSON NOT NULL,
            imported_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            imported_by BIGINT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
    )


def list_excel_files(directory_path: str) -> List[str]:
    if not os.path.isdir(directory_path):
        return []
    return [
        os.path.join(directory_path, name)
        for name in os.listdir(directory_path)
        if name.lower().endswith(".xlsx") and os.path.isfile(os.path.join(directory_path, name))
    ]


def convert_cell_value(value: Any) -> Any:
    """Convert pandas/numpy values to JSON-serializable Python primitives."""
    if pd.isna(value):
        return None
    # pandas timestamps to ISO strings
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    # numpy scalars -> python scalars
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def row_to_clean_dict(series: pd.Series) -> Dict[str, Any]:
    return {str(col): convert_cell_value(series[col]) for col in series.index}


def build_rows_for_insert(
    dataframe: pd.DataFrame,
    source_file: str,
    batch_id: str,
) -> Iterable[Tuple[str, str, int, str]]:
    """
    Yield tuples of (batch_id, source_file, row_index, row_data_json) for executemany.
    row_index is 1-based to match typical spreadsheet indexing.
    """
    # Ensure columns are strings to preserve keys
    dataframe.columns = [str(c) for c in dataframe.columns]
    for idx, row in dataframe.iterrows():
        row_dict = row_to_clean_dict(row)
        row_json = pd.io.json.dumps(row_dict, ensure_ascii=False)
        yield (batch_id, source_file, int(idx) + 1, row_json)


def insert_rows(db: Database, rows: Iterable[Tuple[str, str, int, str]], chunk_size: int = 2000) -> int:
    """Insert rows into data_raw in chunks; returns total inserted count."""
    sql = (
        "INSERT INTO data_raw (batch_id, source_file, row_index, row_data) "
        "VALUES (%s, %s, %s, CAST(%s AS JSON))"
    )
    total = 0
    conn = db.get_connection()
    try:
        cursor = conn.cursor()
        buffer: List[Tuple[str, str, int, str]] = []
        for row in rows:
            buffer.append(row)
            if len(buffer) >= chunk_size:
                cursor.executemany(sql, buffer)
                conn.commit()
                total += len(buffer)
                buffer.clear()
        if buffer:
            cursor.executemany(sql, buffer)
            conn.commit()
            total += len(buffer)
        cursor.close()
    finally:
        conn.close()
    return total


def import_excel_file(db: Database, filepath: str, batch_id: str) -> int:
    """Import all sheets from one Excel file; returns number of inserted rows."""
    # Read all sheets as dict of DataFrames
    excel = pd.read_excel(filepath, sheet_name=None, dtype=object)
    inserted = 0
    for sheet_name, df in excel.items():
        if df.empty:
            continue
        # Normalize empties to NA, drop rows with any NA, drop duplicates, and reset index
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        df = df.dropna(how="any")
        if df.empty:
            continue
        df = df.drop_duplicates()
        if df.empty:
            continue
        df = df.reset_index(drop=True)
        rows_iter = build_rows_for_insert(df, f"{os.path.basename(filepath)}#{sheet_name}", batch_id)
        inserted += insert_rows(db, rows_iter)
    return inserted


def import_directory(directory_path: str) -> None:
    db = Database.get_instance()
    ensure_data_raw_table(db)

    files = list_excel_files(directory_path)
    if not files:
        print(f"No .xlsx files found in: {directory_path}")
        return

    batch_id = str(uuid.uuid4())
    print(f"Starting import. Batch ID: {batch_id}")

    total_inserted = 0
    for file_path in files:
        print(f"Importing file: {file_path}")
        try:
            inserted = import_excel_file(db, file_path, batch_id)
            total_inserted += inserted
            print(f"Inserted {inserted} rows from {os.path.basename(file_path)}")
        except Exception as exc:
            print(f"Failed to import {file_path}: {exc}")

    print(f"Import complete. Total rows inserted: {total_inserted}. Batch ID: {batch_id}")


if __name__ == "__main__":
    import_directory("C:\\Users\\Hp\\Desktop\\FYP-Backend\\Data")

