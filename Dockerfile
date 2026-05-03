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
COPY --chown=flux:flux requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=flux:flux router/ ./router/
COPY --chown=flux:flux pyproject.toml README.md LICENSE ./

# Switch to non-root user
USER flux

# Healthcheck (basic Python import test)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import router; print('ok')" || exit 1

# Default command
CMD ["python", "-m", "router"]
