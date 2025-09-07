#!/bin/bash

# Railway startup script for SproutML API
echo "Starting SproutML API..."

# Set default port if PORT is not set
if [ -z "$PORT" ]; then
    export PORT=8000
fi

echo "PORT: $PORT"
echo "Environment: $(env | grep -E '^(PORT|OPENAI_API_KEY)' || echo 'No relevant env vars found')"

# Start the application
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info
