# RSAI PySpark Implementation

**Repeat Sales Automated Index** - A distributed computing implementation using Apache Spark for real estate price index calculation.

[![Tests](https://img.shields.io/badge/tests-59%2F59%20passing-brightgreen)](./FINAL_TEST_REPORT.md)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![PySpark](https://img.shields.io/badge/pyspark-4.0.0+-orange.svg)](https://spark.apache.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Model](#running-the-model)
  - [Example Usage](#example-usage)
- [Testing](#testing)
  - [Environment Setup](#environment-setup)
  - [Running Unit Tests](#running-unit-tests)
  - [Running Integration Tests](#running-integration-tests)
  - [Running All Tests](#running-all-tests)
- [Data Requirements](#data-requirements)
- [Output Formats](#output-formats)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

The RSAI (Repeat Sales Automated Index) PySpark implementation provides a scalable, distributed computing solution for calculating real estate price indices using the Bailey-Muth-Nourse (BMN) repeat sales methodology. This implementation leverages Apache Spark's distributed computing capabilities to process large-scale real estate transaction datasets efficiently.

### Key Capabilities

- **Distributed Processing**: Utilizes Apache Spark for handling large datasets across multiple cores/nodes
- **BMN Regression**: Implements the industry-standard Bailey-Muth-Nourse repeat sales regression method
- **Geographic Aggregation**: Supports multiple geographic levels (tract, county, CBSA, state, national)
- **Multiple Weighting Schemes**: Equal, value-based, Case-Shiller, and BMN weighting options
- **Quality Validation**: Comprehensive data quality checks and outlier detection
- **Flexible Output**: Multiple export formats including Parquet, CSV, JSON, and visualizations

## Features

### Core Functionality
- ✅ **Data Ingestion**: Parallel loading of transaction and property data
- ✅ **Repeat Sales Identification**: Efficient pairing of property transactions
- ✅ **BMN Regression**: MLlib-based linear regression with time dummy variables
- ✅ **Index Calculation**: Price index computation from regression coefficients
- ✅ **Geographic Aggregation**: Hierarchical aggregation across geography levels
- ✅ **Quality Assurance**: Data validation and outlier filtering

### Advanced Features
- ✅ **Supertract Generation**: K-means clustering for custom geographic units
- ✅ **Multiple Weighting Schemes**: Flexible observation weighting options
- ✅ **Visualization**: Time series plots, heatmaps, and geographic comparisons
- ✅ **Export Integration**: Tableau-ready data formatting
- ✅ **Methodology Documentation**: Automated methodology report generation

### Technical Features
- ✅ **Distributed Computing**: Spark-native DataFrame operations
- ✅ **Memory Optimization**: Efficient resource management and caching
- ✅ **Error Handling**: Robust exception handling and logging
- ✅ **Configuration Management**: Flexible JSON-based configuration system

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Configuration  │    │  Spark Cluster  │
│                 │    │                 │    │                 │
│ • Transactions  │───▶│ • Parameters    │───▶│ • Driver        │
│ • Properties    │    │ • Geography     │    │ • Executors     │
│ • Geography     │    │ • Spark Config  │    │ • Storage       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Data Ingestion  │    │ BMN Regression  │    │ Index Aggregate │
│                 │    │                 │    │                 │
│ • Loading       │───▶│ • Time Periods  │───▶│ • Hierarchical  │
│ • Validation    │    │ • MLlib Regress │    │ • Weighting     │
│ • Filtering     │    │ • Coefficients  │    │ • Aggregation   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Repeat Sales    │    │ Index Values    │    │ Export & Report │
│                 │    │                 │    │                 │
│ • Pair Matching │───▶│ • Price Indices │───▶│ • Multiple      │
│ • Price Ratios  │    │ • Time Series   │    │   Formats       │
│ • Holding Period│    │ • Statistics    │    │ • Visualizations│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Ingestion**: Load transaction and property data from Parquet/CSV files
2. **Pairing**: Identify repeat sales using window functions
3. **Validation**: Apply quality filters and outlier detection
4. **Regression**: Fit BMN models for each geographic area using MLlib
5. **Aggregation**: Create hierarchical indices across geographic levels
6. **Export**: Generate reports, visualizations, and data exports

## Installation

### Prerequisites

- **Python**: 3.12 or higher
- **Java**: 17 or higher (required for Spark 4.0.0+)
- **Memory**: Minimum 8GB RAM recommended for development
- **Storage**: Sufficient disk space for input data and outputs

### System Setup

1. **Install Java 17+**
   ```bash
   # macOS (using Homebrew)
   brew install openjdk@17
   
   # Ubuntu/Debian
   sudo apt-get install openjdk-17-jdk
   
   # Set JAVA_HOME
   export JAVA_HOME=/path/to/java17
   ```

2. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd impl-pyspark
   ```

3. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables

Set the following environment variables for optimal performance:

```bash
# Java Configuration
export JAVA_HOME=/path/to/java17
export PYSPARK_SUBMIT_ARGS='--driver-memory 4g --executor-memory 4g pyspark-shell'

# Python Path
export PYTHONPATH="${PYTHONPATH}:/path/to/impl-pyspark"

# Spark Configuration (optional)
export SPARK_HOME=/path/to/spark  # If using standalone Spark installation
```

## Configuration

The model uses JSON configuration files to specify parameters, data sources, and Spark settings.

### Configuration File Structure

```json
{
  "min_price": 10000,
  "max_price": 10000000,
  "max_holding_period_years": 20,
  "min_pairs_threshold": 30,
  "outlier_std_threshold": 3.0,
  "frequency": "monthly",
  "base_period": null,
  "weighting_scheme": "equal",
  "geography_levels": ["tract", "county", "cbsa", "state"],
  "clustering_method": "kmeans",
  "n_clusters": 500,
  "spark_app_name": "RSAI Model",
  "spark_master": "local[*]",
  "spark_executor_memory": "4g",
  "spark_driver_memory": "4g",
  "input_files": {
    "transactions": "data/transactions.parquet",
    "properties": "data/properties.parquet"
  },
  "output_dir": "output"
}
```

### Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `frequency` | Time period frequency | `"monthly"` | `"daily"`, `"monthly"`, `"quarterly"` |
| `weighting_scheme` | Observation weighting method | `"equal"` | `"equal"`, `"value"`, `"case_shiller"`, `"bmn"` |
| `geography_levels` | Geographic levels to process | `["tract", "county"]` | Any combination of levels |
| `min_pairs_threshold` | Minimum repeat sales pairs | `30` | Positive integer |
| `outlier_std_threshold` | Outlier detection sensitivity | `3.0` | Positive float |

## Usage

### Running the Model

#### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run with configuration file
python -m rsai.src.main config.json
```

#### Advanced Usage

```bash
# Run with custom output format
python -m rsai.src.main config.json --output-format text

# Run with specific Java configuration
JAVA_HOME=/usr/lib/jvm/java-17 python -m rsai.src.main config.json
```

#### Programmatic Usage

```python
from rsai.src.main import RSAIPipeline
from rsai.src.data.models import RSAIConfig

# Load configuration
config = RSAIConfig.from_file("config.json")

# Create and run pipeline
pipeline = RSAIPipeline("config.json")
results = pipeline.run()

# Check results
if results["status"] == "success":
    print(f"Pipeline completed successfully!")
    print(f"Processed {results['total_repeat_sales']} repeat sales")
    print(f"Created indices for {len(results['geography_levels_processed'])} geography levels")
else:
    print(f"Pipeline failed: {results['error']}")

# Clean up
pipeline.stop()
```

### Example Usage

#### Complete Workflow Example

```python
import json
from pathlib import Path
from rsai.src.main import RSAIPipeline

# 1. Create configuration
config = {
    "min_price": 50000,
    "max_price": 5000000,
    "frequency": "monthly",
    "weighting_scheme": "value",
    "geography_levels": ["county", "state"],
    "input_files": {
        "transactions": "data/my_transactions.parquet",
        "properties": "data/my_properties.parquet"
    },
    "output_dir": "my_output",
    "spark_master": "local[4]",
    "spark_driver_memory": "8g"
}

# Save configuration
config_path = Path("my_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# 2. Run pipeline
pipeline = RSAIPipeline(config_path)
results = pipeline.run()

# 3. Process results
output_files = results["output_files"]
print(f"Index file: {output_files['indices']}")
print(f"Report file: {output_files['summary_report']}")
print(f"Plots directory: {output_files['plots']}")

# 4. Clean up
pipeline.stop()
```

## Testing

The project includes comprehensive unit and integration tests covering all functionality.

### Environment Setup

Before running tests, ensure your environment is properly configured:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Set Java path (critical for Spark tests)
export JAVA_HOME=/usr/lib/jvm/java-17  # Adjust path as needed

# 3. Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 4. Verify Java version
java -version  # Should show Java 17+
```

### Running Unit Tests

Unit tests validate individual components without requiring Spark integration:

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run specific test modules
python -m pytest tests/unit/test_data_models.py -v
python -m pytest tests/unit/test_bmn_regression.py -v
python -m pytest tests/unit/test_aggregation.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=rsai --cov-report=html
```

### Running Integration Tests

Integration tests validate end-to-end pipeline functionality:

```bash
# Run all integration tests
python -m pytest tests/integration/ -v

# Run specific integration tests
python -m pytest tests/integration/test_pipeline.py::TestRSAIPipeline::test_pipeline_end_to_end -v

# Run with detailed output
python -m pytest tests/integration/ -v -s
```

### Running All Tests

#### Method 1: Using the Test Runner Script

The project includes a test runner that handles Java configuration automatically:

```bash
# Run all tests with proper configuration
python run_tests.py
```

#### Method 2: Using pytest with Manual Configuration

```bash
# Set environment and run all tests
export JAVA_HOME=/usr/lib/jvm/java-17
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m pytest tests/ -v
```

#### Method 3: Using the Ordered Test Runner

For maximum reliability, use the ordered test runner:

```bash
# Run tests in dependency order
python run_tests_ordered.py
```

### Test Categories

| Test Category | Location | Count | Purpose |
|---------------|----------|-------|---------|
| **Data Models** | `tests/unit/test_data_models.py` | 11 tests | Schema validation, enums, data structures |
| **Data Ingestion** | `tests/unit/test_data_ingestion.py` | 9 tests | Data loading, filtering, validation |
| **Weight Calculation** | `tests/unit/test_weights.py` | 11 tests | All weighting schemes and calculations |
| **BMN Regression** | `tests/unit/test_bmn_regression.py` | 9 tests | Regression fitting and index calculation |
| **Index Aggregation** | `tests/unit/test_aggregation.py` | 10 tests | Geographic aggregation and hierarchies |
| **Spark Setup** | `tests/test_spark_setup.py` | 3 tests | Spark session and configuration |
| **Pipeline Integration** | `tests/integration/test_pipeline.py` | 6 tests | End-to-end pipeline validation |

### Test Configuration

Tests use the following configuration files:

- `pytest.ini`: pytest configuration and markers
- `tests/conftest.py`: Shared fixtures and Spark session management
- `tests/fixtures/`: Sample data for testing

### Environment Variables for Testing

```bash
# Required for all tests
export JAVA_HOME=/path/to/java17
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Optional test configurations
export SPARK_LOCAL_IP=127.0.0.1
export PYSPARK_SUBMIT_ARGS='--driver-memory 2g --executor-memory 2g pyspark-shell'
```

### Troubleshooting Tests

#### Common Issues and Solutions

1. **Java Version Errors**
   ```bash
   # Error: UnsupportedClassVersionError
   # Solution: Ensure Java 17+ is installed and JAVA_HOME is set
   export JAVA_HOME=/usr/lib/jvm/java-17
   java -version
   ```

2. **Spark Session Conflicts**
   ```bash
   # Error: Java gateway process exited
   # Solution: Use the provided test runner or set proper Java configuration
   python run_tests.py
   ```

3. **Memory Issues**
   ```bash
   # Error: OutOfMemoryError
   # Solution: Increase memory allocation
   export PYSPARK_SUBMIT_ARGS='--driver-memory 4g --executor-memory 4g pyspark-shell'
   ```

4. **Module Import Errors**
   ```bash
   # Error: ModuleNotFoundError
   # Solution: Set Python path correctly
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

## Data Requirements

### Input Data Format

#### Transaction Data Schema

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `transaction_id` | String | Unique transaction identifier | ✅ |
| `property_id` | String | Unique property identifier | ✅ |
| `sale_date` | Date | Transaction date | ✅ |
| `sale_price` | Double | Sale price in dollars | ✅ |
| `transaction_type` | String | Type of transaction | ✅ |

#### Property Data Schema

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `property_id` | String | Unique property identifier | ✅ |
| `property_type` | String | Type of property | ✅ |
| `year_built` | Integer | Year property was built | ❌ |
| `square_feet` | Double | Property square footage | ❌ |
| `latitude` | Double | Property latitude | ❌ |
| `longitude` | Double | Property longitude | ❌ |
| `tract` | String | Census tract identifier | ✅ |
| `county` | String | County identifier | ✅ |
| `cbsa` | String | CBSA identifier | ❌ |
| `state` | String | State identifier | ✅ |
| `address` | String | Property address | ❌ |

### Supported File Formats

- **Parquet** (recommended): Optimal for large datasets
- **CSV**: Supported with proper schema specification
- **JSON**: Supported for smaller datasets

### Data Quality Requirements

- Minimum 30 repeat sales pairs per geographic area
- Valid date ranges and positive prices
- Consistent geographic identifiers
- Arms-length transactions only

## Output Formats

The model generates multiple output formats for different use cases:

### Index Files
- **Parquet**: `output/indices/index_YYYYMMDD_HHMMSS.parquet`
- **CSV**: Option for tabular data consumers
- **JSON**: For API integration

### Reports
- **HTML Report**: `output/reports/summary_report_YYYYMMDD_HHMMSS.html`
- **Regression Results**: `output/reports/regression_results_YYYYMMDD_HHMMSS.json`
- **Methodology**: `output/reports/methodology.md`

### Visualizations
- **Time Series Plot**: `output/plots/index_time_series.png`
- **Growth Heatmap**: `output/plots/growth_heatmap.png`
- **Geographic Comparison**: `output/plots/geographic_comparison.png`

### Business Intelligence
- **Tableau Data**: `output/data/tableau_data_YYYYMMDD_HHMMSS.csv`

## Performance Tuning

### Spark Configuration

```json
{
  "spark_master": "local[*]",
  "spark_driver_memory": "8g",
  "spark_executor_memory": "4g",
  "spark_config": {
    "spark.sql.shuffle.partitions": "200",
    "spark.default.parallelism": "100",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.executor.cores": "4",
    "spark.executor.instances": "2"
  }
}
```

### Hardware Recommendations

| Dataset Size | Memory | Cores | Storage |
|--------------|---------|-------|---------|
| < 1M transactions | 8GB | 4 cores | 50GB |
| 1M - 10M transactions | 16GB | 8 cores | 100GB |
| 10M - 100M transactions | 32GB | 16 cores | 500GB |
| > 100M transactions | 64GB+ | 32+ cores | 1TB+ |

### Optimization Tips

1. **Partitioning**: Adjust `spark.sql.shuffle.partitions` based on data size
2. **Caching**: Enable DataFrame caching for iterative operations
3. **File Format**: Use Parquet for optimal compression and query performance
4. **Cluster Configuration**: Use standalone or YARN cluster for large datasets

## Troubleshooting

### Common Issues

1. **Java Version Compatibility**
   ```
   Error: UnsupportedClassVersionError
   Solution: Install Java 17+ and set JAVA_HOME
   ```

2. **Memory Errors**
   ```
   Error: OutOfMemoryError
   Solution: Increase driver/executor memory in configuration
   ```

3. **Spark Session Issues**
   ```
   Error: Java gateway process exited
   Solution: Check Java installation and JAVA_HOME setting
   ```

4. **Performance Issues**
   ```
   Issue: Slow execution
   Solution: Optimize partition count and resource allocation
   ```

### Getting Help

- Check the [test report](./FINAL_TEST_REPORT.md) for detailed validation results
- Review log files in the `logs/` directory
- Enable DEBUG logging for detailed troubleshooting

### Log Configuration

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements.txt`
4. Run tests to ensure everything works: `python run_tests.py`
5. Make your changes
6. Add/update tests as needed
7. Run the full test suite: `python run_tests.py`
8. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Add comprehensive docstrings
- Maintain test coverage above 90%

### Testing Requirements

All contributions must:
- Include appropriate unit tests
- Pass all existing tests
- Maintain or improve code coverage
- Include integration tests for new features

---

**For detailed test results and validation reports, see [FINAL_TEST_REPORT.md](./FINAL_TEST_REPORT.md)**