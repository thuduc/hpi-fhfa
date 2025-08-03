# Multi-stage build for HPI-FHFA PySpark application
FROM apache/spark:3.5.0-python3 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM apache/spark:3.5.0-python3

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/app
ENV SPARK_HOME=/opt/spark
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Create app user
RUN useradd -m -u 1000 -s /bin/bash sparkuser

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.*/site-packages /usr/local/lib/python3.10/site-packages

# Copy application code
WORKDIR /app
COPY --chown=sparkuser:sparkuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints /app/output && \
    chown -R sparkuser:sparkuser /app

# Switch to non-root user
USER sparkuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pyspark; print('Spark is healthy')" || exit 1

# Default command (can be overridden)
CMD ["spark-submit", \
     "--master", "local[*]", \
     "--conf", "spark.driver.memory=4g", \
     "--conf", "spark.executor.memory=4g", \
     "scripts/run_pipeline.py"]