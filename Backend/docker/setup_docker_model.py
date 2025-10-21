#!/usr/bin/env python3
"""
Docker Model Setup Script for Qwen2.5
This script helps set up and run the Qwen2.5 model in Docker.
"""

import subprocess
import time
import requests
import os
import sys

def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_docker_running():
    """Check if Docker is running."""
    try:
        subprocess.run("docker --version", shell=True, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def check_model_available():
    """Check if the model is available."""
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🚀 Setting up Qwen2.5 Docker Model")
    print("=" * 50)
    
    # Check if Docker is running
    if not check_docker_running():
        print("❌ Docker is not running or not installed.")
        print("Please install Docker and start it, then run this script again.")
        return False
    
    # Stop existing container if running
    print("🛑 Stopping existing container if running...")
    subprocess.run("docker stop qwen2.5-container", shell=True, capture_output=True)
    subprocess.run("docker rm qwen2.5-container", shell=True, capture_output=True)
    
    # Pull the model
    if not run_command("docker model pull ai/qwen2.5:3B-Q4_K_M", "Pulling Qwen2.5 model"):
        return False
    
    # Run the container
    if not run_command(
        "docker model run -d --name qwen2.5-container -p 8080:8080 ai/qwen2.5:3B-Q4_K_M",
        "Starting Qwen2.5 container"
    ):
        return False
    
    # Wait for model to be ready
    print("⏳ Waiting for model to be ready...")
    for i in range(30):  # Wait up to 30 seconds
        if check_model_available():
            print("✅ Model is ready!")
            break
        time.sleep(1)
        print(f"   Waiting... ({i+1}/30)")
    else:
        print("❌ Model failed to start within 30 seconds")
        print("Check container logs: docker logs qwen2.5-container")
        return False
    
    # Test the model
    print("🧪 Testing model...")
    try:
        response = requests.post(
            "http://localhost:8080/generate",
            json={
                "prompt": "Hello, how are you?",
                "temperature": 0.7,
                "max_tokens": 50
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model response: {result.get('text', 'No response')[:100]}...")
        else:
            print(f"❌ Model test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("Your local Qwen2.5 model is now running at http://localhost:8080")
    print("\nTo use it in your application:")
    print("1. Set LOCAL_MODEL_URL=http://localhost:8080 in your .env file")
    print("2. The SQL agent will automatically use the local model")
    print("\nTo stop the model: docker stop qwen2.5-container")
    print("To start it again: docker start qwen2.5-container")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
