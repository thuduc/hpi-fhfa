.PHONY: install test lint format clean run-pipeline

install:
	pip install -e ".[dev]"

test:
	pytest

test-coverage:
	pytest --cov-report=html
	open htmlcov/index.html

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	ruff check --fix src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info
	rm -rf htmlcov .coverage .pytest_cache

run-pipeline:
	python scripts/run_pipeline.py