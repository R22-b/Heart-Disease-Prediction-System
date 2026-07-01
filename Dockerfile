FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt --no-cache-dir

# Copy the rest of the application
COPY . .

EXPOSE 5000

# Healthcheck (calls the /health endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Start the app with Gunicorn — Render sets $PORT automatically
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000}
