# Use Python 3.11 slim image with fallback
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies with retry logic
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    || (sleep 5 && apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*)

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies with retry logic
RUN pip install --no-cache-dir --upgrade pip || (sleep 5 && pip install --no-cache-dir --upgrade pip)
RUN pip install --no-cache-dir -r requirements.txt || (sleep 5 && pip install --no-cache-dir -r requirements.txt)

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application directly
CMD ["python3", "main.py"]
