import os
import uuid
from datetime import datetime
from typing import Any, List, Tuple

import pandas as pd

from db import Database


def ensure_data_raw_table(db: Database) -> None:
    """Create the data_raw table if it does not exist (MySQL compatible)."""
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS data_raw (
            id                    BIGINT AUTO_INCREMENT PRIMARY KEY,
            local_timestamp       VARCHAR(255) NOT NULL,
            device_name           VARCHAR(255) NOT NULL,
            direction             VARCHAR(100) NOT NULL,
            vehicle_type          VARCHAR(100) NOT NULL,
            vehicle_types_lp_ocr  TEXT NOT NULL,
            ocr_score             DECIMAL(10,9) NOT NULL,
            
            INDEX idx_local_timestamp (local_timestamp),
            INDEX idx_device_name (device_name),
            INDEX idx_direction (direction),
            INDEX idx_vehicle_type (vehicle_type),
            INDEX idx_ocr_score (ocr_score)
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
    """Convert pandas/numpy values to Python primitives."""
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


def build_rows_for_insert(dataframe: pd.DataFrame) -> List[Tuple[str, str, str, str, str, float]]:
    """
    Build rows for insertion from DataFrame.
    Returns list of tuples: (local_timestamp, device_name, direction, vehicle_type, vehicle_types_lp_ocr, ocr_score)
    """
    rows = []
    
    # Expected column names from the Excel file
    expected_columns = {
        'localTimesta': 'local_timestamp',
        'deviceName': 'device_name',
        'direction': 'direction',
        'vehicleType': 'vehicle_type',
        'vehicleTypes lpOcr': 'vehicle_types_lp_ocr',
        'ocrScore': 'ocr_score'
    }
    
    # Check if all required columns exist
    missing_columns = [col for col in expected_columns.keys() if col not in dataframe.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        return rows
    
    for idx, row in dataframe.iterrows():
        try:
            local_timestamp = convert_cell_value(row['localTimesta'])
            device_name = convert_cell_value(row['deviceName'])
            direction = convert_cell_value(row['direction'])
            vehicle_type = convert_cell_value(row['vehicleType'])
            vehicle_types_lp_ocr = convert_cell_value(row['vehicleTypes lpOcr'])
            ocr_score = convert_cell_value(row['ocrScore'])
            
            # Skip rows with missing required data
            if not all([local_timestamp, device_name, direction, vehicle_type, vehicle_types_lp_ocr, ocr_score]):
                continue
            
            # Convert to strings (except ocr_score which is float)
            local_timestamp = str(local_timestamp)
            device_name = str(device_name)
            direction = str(direction)
            vehicle_type = str(vehicle_type)
            vehicle_types_lp_ocr = str(vehicle_types_lp_ocr)
            
            # Convert ocr_score to float
            try:
                ocr_score = float(ocr_score)
            except (ValueError, TypeError):
                ocr_score = 0.0
            
            rows.append((local_timestamp, device_name, direction, vehicle_type, vehicle_types_lp_ocr, ocr_score))
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    return rows


def insert_rows(db: Database, rows: List[Tuple[str, str, str, str, str, float]], chunk_size: int = 2000) -> int:
    """Insert rows into data_raw in chunks; returns total inserted count."""
    sql = """
        INSERT INTO data_raw 
        (local_timestamp, device_name, direction, vehicle_type, vehicle_types_lp_ocr, ocr_score) 
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    total = 0
    conn = db.get_connection()
    try:
        cursor = conn.cursor()
        buffer = []
        for row in rows:
            buffer.append(row)
            if len(buffer) >= chunk_size:
                cursor.executemany(sql, buffer)
                conn.commit()
                total += len(buffer)
                print(f"Inserted {total} rows so far...")
                buffer.clear()
        if buffer:
            cursor.executemany(sql, buffer)
            conn.commit()
            total += len(buffer)
        cursor.close()
    finally:
        conn.close()
    return total


def import_excel_file(db: Database, filepath: str) -> int:
    """Import all sheets from one Excel file; returns number of inserted rows."""
    print(f"Reading Excel file: {filepath}")
    
    # Read all sheets as dict of DataFrames
    excel = pd.read_excel(filepath, sheet_name=None, dtype=object)
    inserted = 0
    
    for sheet_name, df in excel.items():
        print(f"Processing sheet: {sheet_name}")
        
        if df.empty:
            print(f"Sheet {sheet_name} is empty, skipping...")
            continue
        
        # Clean data: replace empty strings with NA, drop rows with any NA
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        df = df.dropna(how="any")
        
        if df.empty:
            print(f"Sheet {sheet_name} has no valid data after cleaning, skipping...")
            continue
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        if df.empty:
            print(f"Sheet {sheet_name} has no data after removing duplicates, skipping...")
            continue
        
        df = df.reset_index(drop=True)
        
        print(f"Sheet {sheet_name} has {len(df)} rows to process")
        
        # Build rows for insertion
        rows = build_rows_for_insert(df)
        
        if rows:
            print(f"Inserting {len(rows)} rows from sheet {sheet_name}...")
            inserted_count = insert_rows(db, rows)
            inserted += inserted_count
            print(f"Successfully inserted {inserted_count} rows from sheet {sheet_name}")
        else:
            print(f"No valid rows to insert from sheet {sheet_name}")
    
    return inserted


def import_directory(directory_path: str) -> None:
    """Import all Excel files from a directory."""
    db = Database.get_instance()
    ensure_data_raw_table(db)

    files = list_excel_files(directory_path)
    if not files:
        print(f"No .xlsx files found in: {directory_path}")
        return

    print(f"Found {len(files)} Excel file(s) to import")
    print(f"Files: {[os.path.basename(f) for f in files]}")

    total_inserted = 0
    for file_path in files:
        print(f"\n{'='*60}")
        print(f"Importing file: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        try:
            inserted = import_excel_file(db, file_path)
            total_inserted += inserted
            print(f"✅ Inserted {inserted} rows from {os.path.basename(file_path)}")
        except Exception as exc:
            print(f"❌ Failed to import {file_path}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Import complete. Total rows inserted: {total_inserted}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import_directory("C:\\Users\\Hp\\Desktop\\FYP-Backend\\Data")
