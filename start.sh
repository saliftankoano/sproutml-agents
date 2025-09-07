#!/bin/bash

# Railway startup script for SproutML API
echo "Starting SproutML API..."
echo "PORT: ${PORT:-8000}"
echo "Environment: $(env | grep -E '^(PORT|OPENAI_API_KEY)' || echo 'No relevant env vars found')"

# Start the application
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info
