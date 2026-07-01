FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies needed for numerical libraries and curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libatlas-base-dev \
        libopenblas-dev \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Upgrade pip and install core binary packages first to avoid building from source
RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy==1.26.3 pandas==2.1.4 scikit-learn==1.4.1 flask==3.0.0 gunicorn==21.2.0 --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir || true

# Copy the rest of the application
COPY . .

EXPOSE 5000

# Healthcheck (calls the /health endpoint we added)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Start the app with Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
