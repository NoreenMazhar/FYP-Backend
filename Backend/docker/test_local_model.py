#!/usr/bin/env python3
"""
Test script for local Docker model integration
"""

import os
import sys
import requests
import json

def test_local_model():
    """Test the local Docker model."""
    print("🧪 Testing Local Docker Model Integration")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Simple generation
    print("\n2. Testing text generation...")
    try:
        payload = {
            "prompt": "What is SQL?",
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = requests.post(
            "http://localhost:8080/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Text generation successful")
            print(f"Response: {result.get('text', 'No response')[:200]}...")
        else:
            print(f"❌ Text generation failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Text generation error: {e}")
        return False
    
    # Test 3: SQL generation
    print("\n3. Testing SQL generation...")
    try:
        payload = {
            "prompt": "Generate a SQL query to select all users from a users table",
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        response = requests.post(
            "http://localhost:8080/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SQL generation successful")
            print(f"SQL Query: {result.get('text', 'No response')[:300]}...")
        else:
            print(f"❌ SQL generation failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ SQL generation error: {e}")
        return False
    
    print("\n🎉 All tests passed! Local model is working correctly.")
    return True

def test_sql_agent_integration():
    """Test the SQL agent with local model."""
    print("\n🔧 Testing SQL Agent Integration")
    print("=" * 50)
    
    try:
        # Import the SQL agent
        from sql_agent import get_sql_agent
        
        print("1. Testing SQL agent initialization...")
        agent = get_sql_agent()
        if agent:
            print("✅ SQL agent initialized successfully")
        else:
            print("❌ SQL agent initialization failed")
            return False
        
        print("\n2. Testing simple SQL query...")
        # Test a simple query
        result = agent.run("Show me all tables in the database")
        print(f"✅ SQL agent response: {str(result)[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ SQL agent test error: {e}")
        return False

if __name__ == "__main__":
    print("Starting Local Model Tests...")
    
    # Test local model
    model_ok = test_local_model()
    
    if model_ok:
        # Test SQL agent integration
        agent_ok = test_sql_agent_integration()
        
        if agent_ok:
            print("\n🎉 All tests passed! Your local Docker model is ready to use.")
        else:
            print("\n⚠️  Local model works but SQL agent integration has issues.")
    else:
        print("\n❌ Local model is not working. Please check Docker setup.")
        print("Run: python setup_docker_model.py")
