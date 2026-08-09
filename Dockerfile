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

# Install the package itself so the `flux` console script exists on PATH.
# --no-deps: the runtime requirements were installed above and pinned there.
RUN pip install --no-cache-dir --no-deps .

# Where the usage database lives. Mount a volume here (see docker-compose.yml)
# or the container's spend history dies with the container.
ENV FLUX_DATA_DIR=/data
RUN mkdir -p /data && chown flux:flux /data
VOLUME ["/data"]

# The container boundary is the network boundary, so the proxy binds all
# interfaces. Set through the environment rather than a uvicorn --host flag:
# the dashboard's mount guard reads the CONFIGURED bind address, and passing
# the host on the command line left it believing it was on loopback while
# actually serving every tenant's spend to the network.
#
# Authentication is mandatory because this is a non-loopback bind. Startup
# fails closed without FLUX_SERVER_TOKEN/FLUX_SERVER_TOKENS. A deployment whose
# network boundary supplies equivalent authentication may explicitly accept
# the risk with FLUX_ALLOW_UNAUTHENTICATED_REMOTE=1.
ENV FLUX_SERVER_HOST=0.0.0.0

# Switch to non-root user
USER flux

EXPOSE 8000

# Healthcheck against the running proxy's /health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

# Default command: run the OpenAI-compatible HTTP proxy, with the data
# directory created and the usage database pointed inside it. Bind address and
# data dir come from the ENV above; an authentication token is required.
CMD ["flux", "serve"]
