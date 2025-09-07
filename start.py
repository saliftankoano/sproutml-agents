#!/usr/bin/env python3
"""
Railway startup script for SproutML API
This ensures proper port handling and environment variable parsing
"""
import os
import subprocess
import sys

def main():
    print("Starting SproutML API...")
    
    # Get port from environment, default to 8000
    port = os.getenv("PORT", "8000")
    
    # Validate port is numeric
    try:
        port_int = int(port)
        if port_int < 1 or port_int > 65535:
            raise ValueError("Port out of range")
    except ValueError as e:
        print(f"Invalid PORT value '{port}': {e}")
        print("Using default port 8000")
        port = "8000"
    
    print(f"PORT: {port}")
    print(f"OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    
    # Build uvicorn command
    cmd = [
        "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", port,
        "--workers", "1",
        "--log-level", "info"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    # Execute uvicorn
    try:
        os.execvp("uvicorn", cmd)
    except Exception as e:
        print(f"Failed to start uvicorn: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
