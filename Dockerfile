# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

# Set up non-root user
RUN groupadd --gid 1000 flux && \
    useradd --uid 1000 --gid flux --shell /bin/bash --create-home flux

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# App setup
WORKDIR /app
COPY --chown=flux:flux requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r requirements-server.txt

COPY --chown=flux:flux router/ ./router/
COPY --chown=flux:flux pyproject.toml README.md LICENSE ./

# Switch to non-root user
USER flux

EXPOSE 8000

# Healthcheck against the running proxy's /health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

# Default command: run the OpenAI-compatible HTTP proxy.
# Binds 0.0.0.0 inside the container — the container boundary is the network
# boundary; set FLUX_SERVER_TOKEN to require auth before exposing the port.
CMD ["uvicorn", "router.server:app", "--host", "0.0.0.0", "--port", "8000"]
