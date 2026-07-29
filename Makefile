.PHONY: install install-dev install-server test test-fast lint security-check format clean run serve evals docker-build help

help:
	@echo "Flux — Available commands:"
	@echo "  install         Install runtime dependencies"
	@echo "  install-dev     Install dev dependencies"
	@echo "  install-server  Install HTTP proxy server dependencies (fastapi/uvicorn)"
	@echo "  test            Run full test suite"
	@echo "  test-fast       Run tests, stop on first failure"
	@echo "  lint            Run ruff linter"
	@echo "  security-check  Run bandit + safety + pip-audit"
	@echo "  format          Auto-format code with ruff"
	@echo "  clean           Remove build artifacts and caches"
	@echo "  run             Run the demo"
	@echo "  serve           Run the OpenAI-compatible HTTP proxy (router/server.py)"
	@echo "  evals           Run the cost-vs-quality eval harness (mock, offline)"
	@echo "  docker-build    Build Docker image"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

install-server:
	pip install -r requirements.txt -r requirements-server.txt

test:
	pytest router/tests/ -v --timeout=30

test-fast:
	pytest router/tests/ -v --timeout=10 -x

lint:
	ruff check router/

format:
	ruff check --fix router/
	ruff format router/

security-check:
	bandit -r router/ -ll
	safety check
	pip-audit

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m router.demo

serve:
	uvicorn router.server:app --host $${FLUX_SERVER_HOST:-127.0.0.1} --port $${FLUX_SERVER_PORT:-8000}

evals:
	python -m router.evals

docker-build:
	docker build -t flux-router:latest .
