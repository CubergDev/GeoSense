# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13-slim
FROM python:${PYTHON_VERSION} as base

# Install OS deps (curl for healthcheck), create non-root user
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /app

# Install Python deps first for better caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY start.py ./start.py
COPY populate_db_from_csv.py ./populate_db_from_csv.py
COPY frontend ./frontend
COPY models ./models
COPY data ./data
COPY trips.csv ./trips.csv

# Create required directories and set permissions
RUN mkdir -p /app/data/cache \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Default command: run DB init (idempotent) then start API with multiple workers
CMD ["sh", "-c", "python -m app.core.init_db || true; uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-2}"]

HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://localhost:8000/health || exit 1
