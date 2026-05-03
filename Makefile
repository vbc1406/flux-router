.PHONY: install install-dev test test-fast lint security-check format clean run docker-build help

help:
	@echo "Flux — Available commands:"
	@echo "  install         Install runtime dependencies"
	@echo "  install-dev     Install dev dependencies"
	@echo "  test            Run full test suite"
	@echo "  test-fast       Run tests, stop on first failure"
	@echo "  lint            Run ruff linter"
	@echo "  security-check  Run bandit + safety + pip-audit"
	@echo "  format          Auto-format code with ruff"
	@echo "  clean           Remove build artifacts and caches"
	@echo "  run             Run the demo"
	@echo "  docker-build    Build Docker image"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

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

docker-build:
	docker build -t flux-router:latest .
