# Dummy Data Setup for FYP Backend

This directory contains SQL scripts and Python utilities to set up comprehensive test data for the FYP Backend vehicle detection system.

## Files Overview

### SQL Scripts

- `basic.sql` - Core database tables (users, data_raw, visualizations, etc.)
- `devices.sql` - Device management tables (devices, models, streams, health, etc.)
- `dummy_data.sql` - PostgreSQL-compatible dummy data
- `dummy_data_mysql.sql` - MySQL-compatible dummy data

### Python Scripts

- `load_dummy_data.py` - Automated script to load all SQL files into the database
- `test_api_comprehensive.py` - Comprehensive test suite for the vehicle detections API
- `test_vehicle_detections.py` - Simple test script for the API endpoint

## Quick Setup

### 1. Start MySQL Database

Make sure MySQL is running and accessible with the credentials configured in your environment variables.

### 2. Load Database Schema

```bash
cd Backend
python load_dummy_data.py
```

### 3. Start the API Server

```bash
cd Backend
uvicorn main:app --reload
```

### 4. Test the API

```bash
cd Backend
python test_api_comprehensive.py
```

## Dummy Data Contents

The dummy data includes:

### Device Models (5 models)

- Hikvision DS-2CD2143G0-I
- Dahua IPC-HFW4431R-Z
- Axis M3045-V
- Bosch FLEXIDOME IP 4000i
- Samsung SNV-6013M

### Devices (8 cameras)

- Device-A1, Device-B3, Device-C2, Device-A2
- Device-B1, Device-C3, Device-A3, Device-B2
- Various statuses: active, maintenance
- RTSP stream configurations

### Vehicle Detections (22 records)

- Recent detections (last 2 hours)
- Historical data spanning 30 days
- Various vehicle types: Car, Truck, Motorcycle, Bus
- Realistic license plates and confidence scores
- Different directions: Inbound, Outbound

### Supporting Data

- Device health records with CPU, memory, temperature metrics
- Device events (license events, errors, warnings)
- Device telemetry data
- Device permissions and groups

## API Testing

### Vehicle Detections Endpoint

```
GET /vehicle-detections?start_date=2025-01-01&end_date=2025-01-31
```

**Response Format:**

```json
{
  "detections": [
    {
      "timestamp": "2025-01-15T14:32:15",
      "device": "Device-A1",
      "direction": "Inbound",
      "vehicle_type": "Car",
      "type_score": 95.0,
      "license_plate": "ABC-1234",
      "ocr_score": 92.0
    }
  ],
  "total_count": 1
}
```

### Test Scenarios

The comprehensive test suite covers:

- Recent data (today)
- Historical data (last 7 days, last 30 days)
- Specific date ranges
- Error handling (invalid dates, missing parameters)
- Edge cases (future dates, empty results)

## Manual Testing

You can also test the API manually using:

- **Interactive API Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Troubleshooting

### Database Connection Issues

- Ensure MySQL is running
- Check environment variables: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- Verify database exists and user has proper permissions

### API Server Issues

- Check if port 8000 is available
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate virtual environment if using one

### Data Loading Issues

- Check SQL file syntax for your database version
- Ensure all required tables exist before loading data
- Check database logs for specific error messages

## Data Customization

To modify the dummy data:

1. Edit `dummy_data_mysql.sql` for MySQL
2. Edit `dummy_data.sql` for PostgreSQL
3. Re-run `load_dummy_data.py`

The vehicle detection data uses realistic timestamps and follows the exact format expected by the frontend interface.
