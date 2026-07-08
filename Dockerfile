# syntax=docker/dockerfile:1

# =============================================================================
# Finance-DeepSeek — Multi-stage Docker Build
# Stage 1: Base Python 3.11 runtime with CUDA 12.1 support
# Stage 2: Builder (compile heavy deps with build tools)
# Stage 3: Production runtime (slim, no build tools)
# =============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# ------------------------------------------------------------------------------
# Stage 2: Builder — install heavy Python dependencies
# ------------------------------------------------------------------------------
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    gcc \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install \
    torch==2.4.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install -r /app/requirements.txt

# ------------------------------------------------------------------------------
# Stage 3: Production runtime
# ------------------------------------------------------------------------------
FROM base AS production

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser

# Project structure
COPY --chown=appuser:appuser . /app

# Ensure models directory exists (will be mounted at runtime for large files)
RUN mkdir -p /app/models/base/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B && \
    chown -R appuser:appuser /app/models

# Create data / output directories
RUN mkdir -p /app/data /app/outputs /app/logs && \
    chown -R appuser:appuser /app/data /app/outputs /app/logs

USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: start FastAPI server via uvicorn
# Override with docker run --entrypoint for training / data-gen / indexing
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
