#!/bin/bash

# Script to run RSAI tests

echo "Running RSAI Tests"
echo "=================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install -e .

# Generate sample data
echo "Generating sample data..."
python rsai/tests/generate_sample_data.py

# Run unit tests
echo -e "\nRunning unit tests..."
pytest rsai/tests/test_data_models.py -v
pytest rsai/tests/test_data_ingestion.py -v
pytest rsai/tests/test_supertract.py -v
pytest rsai/tests/test_bmn_regression.py -v
pytest rsai/tests/test_weights.py -v
pytest rsai/tests/test_export.py -v

# Run integration tests
echo -e "\nRunning integration tests..."
pytest rsai/tests/test_integration.py -v -m "not slow"

# Run all tests with coverage
echo -e "\nRunning all tests with coverage..."
pytest --cov=rsai.src --cov-report=html --cov-report=term

echo -e "\nTest run complete!"
echo "Coverage report generated in htmlcov/index.html"