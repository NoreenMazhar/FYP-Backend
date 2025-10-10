#!/usr/bin/env python3
"""
Test script for the vehicle detections API endpoint.
This script tests the /vehicle-detections endpoint with sample data.
"""

import requests
import json
from datetime import datetime, date, timedelta

# Configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/vehicle-detections"

def test_vehicle_detections():
    """Test the vehicle detections endpoint."""
    
    # Test with date range (last 30 days)
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    print(f"Testing vehicle detections API...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Endpoint: {API_ENDPOINT}")
    
    try:
        # Make the API request
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
        response = requests.get(API_ENDPOINT, params=params)
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total detections: {data.get('total_count', 0)}")
            
            if data.get('detections'):
                print(f"\nFirst few detections:")
                for i, detection in enumerate(data['detections'][:3]):
                    print(f"  {i+1}. {detection['timestamp']} - {detection['device']} - {detection['vehicle_type']} - {detection['license_plate']}")
            else:
                print("No detections found in the specified date range.")
                
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the FastAPI server is running on localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

def test_api_docs():
    """Test if the API docs are accessible."""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print(f"\nAPI documentation available at: {BASE_URL}/docs")
        else:
            print(f"API docs not accessible: {response.status_code}")
    except Exception as e:
        print(f"Could not access API docs: {e}")

if __name__ == "__main__":
    print("=== Vehicle Detections API Test ===")
    test_api_docs()
    test_vehicle_detections()
    print("\n=== Test Complete ===")
