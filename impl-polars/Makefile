.PHONY: help install install-dev test test-unit test-integration coverage lint format type-check clean docs run-notebook

help:
	@echo "Available commands:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make test          Run all tests"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make coverage      Run tests with coverage report"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black and isort"
	@echo "  make type-check    Run type checking with mypy"
	@echo "  make clean         Clean up generated files"
	@echo "  make docs          Build documentation"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,docs]"

test:
	pytest

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

coverage:
	pytest --cov=rsai --cov-report=html --cov-report=term

lint:
	flake8 rsai tests
	black --check rsai tests
	isort --check-only rsai tests

format:
	black rsai tests scripts
	isort rsai tests scripts

type-check:
	mypy rsai

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build dist *.egg-info
	rm -rf htmlcov .coverage coverage.xml
	rm -rf docs/_build

docs:
	cd docs && sphinx-build -b html . _build/html

run-notebook:
	jupyter lab