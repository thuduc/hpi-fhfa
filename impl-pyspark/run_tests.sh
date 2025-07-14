#!/bin/bash

# Script to run PySpark tests locally

echo "Setting up environment for PySpark tests..."

# Set Java home - try to find Java 11 or 17
export JAVA_HOME=$(/usr/libexec/java_home -v 11 2>/dev/null || /usr/libexec/java_home -v 17 2>/dev/null || /usr/libexec/java_home -v 1.8 2>/dev/null || /usr/libexec/java_home)

if [ -z "$JAVA_HOME" ]; then
    echo "Error: Java not found. Please install Java 8, 11, or 17."
    exit 1
fi

echo "Using Java at: $JAVA_HOME"

# Set up Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Set Spark configuration for local testing
export SPARK_LOCAL_IP="127.0.0.1"
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p logs
mkdir -p .pytest_cache

echo "Running PySpark tests..."

# Run tests with appropriate settings
python -m pytest tests/ \
    -v \
    --tb=short \
    --disable-warnings \
    -p no:warnings \
    --log-cli-level=ERROR \
    2>&1 | tee test_output.log

# Check test results
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed. Check test_output.log for details."
fi