# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# System deps for scientific Python wheels + healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv for fast Python installs
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifests first so docker cache hits on code-only changes
COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/

# Install runtime dependencies into system Python (smaller image, no venv overhead)
RUN uv pip install --system --no-cache .

# Default port (Fly.io overrides via PORT env)
ENV PORT=8765 \
    SARGVISION_N_DRONES=24 \
    SARGVISION_SCENARIO=border_strike \
    PYTHONUNBUFFERED=1

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

CMD ["sh", "-c", "exec python -m uvicorn sargvision_swarm.server.bridge:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
