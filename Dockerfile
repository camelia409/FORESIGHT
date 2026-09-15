# ==============================================================================
# PROJECT FORESIGHT — Production Dockerfile
# Multi-stage image supporting API serving, Streamlit dashboard, and Pipeline DAG
# ==============================================================================

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and artifacts
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY configs/ ./configs/
COPY data/ ./data/
COPY models/ ./models/
COPY artifacts/ ./artifacts/
COPY reports/ ./reports/

# Create non-root user for security
RUN useradd -m -u 1000 foresight && \
    chown -R foresight:foresight /app

USER foresight

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8501

# Default command starts the production FastAPI service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
