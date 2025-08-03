"""API documentation generator for HPI-FHFA."""

import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import pkgutil
import structlog

logger = structlog.get_logger()


def generate_api_documentation(output_path: Path) -> None:
    """Generate comprehensive API documentation.
    
    Args:
        output_path: Directory to write documentation files
    """
    output_path.mkdir(exist_ok=True)
    
    # Main API documentation
    api_doc = _generate_main_api_doc()
    (output_path / "API.md").write_text(api_doc)
    
    # Module documentation
    modules_doc = _generate_modules_doc()
    (output_path / "MODULES.md").write_text(modules_doc)
    
    # Configuration guide
    config_doc = _generate_config_doc()
    (output_path / "CONFIGURATION.md").write_text(config_doc)
    
    # Usage examples
    examples_doc = _generate_examples_doc()
    (output_path / "EXAMPLES.md").write_text(examples_doc)
    
    logger.info(f"API documentation generated in {output_path}")


def _generate_main_api_doc() -> str:
    """Generate main API documentation."""
    return """# HPI-FHFA API Documentation

## Overview

The HPI-FHFA library provides a comprehensive implementation of the Federal Housing Finance Agency's House Price Index methodology using Python and Polars for high-performance data processing.

## Quick Start

```python
from hpi_fhfa.processing.pipeline import HPIPipeline
from hpi_fhfa.config.settings import HPIConfig
from pathlib import Path

# Configure the pipeline
config = HPIConfig(
    transaction_data_path=Path("transactions.parquet"),
    geographic_data_path=Path("geographic.parquet"),
    output_path=Path("output/"),
    start_year=2020,
    end_year=2023,
    weight_schemes=["sample", "value", "unit"]
)

# Run the pipeline
pipeline = HPIPipeline(config)
results = pipeline.run()

# Access results
print(f"Generated {len(results.tract_indices)} tract indices")
print(f"Generated indices for {len(results.city_indices)} weight schemes")
```

## Core Components

### 1. HPIPipeline

The main pipeline class that orchestrates the entire HPI calculation process.

**Key Methods:**
- `run()`: Execute the complete pipeline
- `run_partial()`: Execute specific pipeline steps

### 2. HPIConfig

Configuration class that defines all pipeline parameters.

**Key Parameters:**
- `transaction_data_path`: Path to transaction data file
- `geographic_data_path`: Path to geographic/census data file
- `output_path`: Directory for output files
- `start_year`, `end_year`: Year range for index calculation
- `weight_schemes`: List of weighting schemes to calculate
- `n_jobs`: Number of parallel processing jobs

### 3. Data Processing Components

#### DataLoader
Handles loading of transaction and geographic data from various formats:
- Parquet (recommended)
- CSV
- Arrow/Feather

#### DataValidator
Validates data quality and schema compliance:
- Schema validation
- Data quality checks
- Duplicate detection

#### TransactionFilter
Applies FHFA filtering rules:
- Same-period transaction removal
- CAGR outlier filtering (30% threshold)
- Cumulative appreciation filtering (0.25x-10x range)

### 4. Core Algorithms

#### BMNRegression
Implements Bailey-Muth-Nourse regression:
- Sparse matrix construction for time dummy variables
- Least squares estimation with regularization
- Index value calculation

#### SupertractBuilder
Implements dynamic supertract aggregation:
- Distance-based tract clustering
- Minimum half-pairs threshold enforcement (40)
- Centroid calculation and updates

#### WeightingFactory
Provides six weighting schemes:
- `sample`: Equal weighting
- `value`: Housing value-weighted
- `unit`: Housing unit-weighted  
- `upb`: Unpaid principal balance-weighted
- `college`: College share-weighted
- `nonwhite`: Non-white share-weighted

### 5. Index Construction

#### TractLevelIndex
Constructs tract-level indices:
- Balanced panel creation
- Base year normalization
- Missing value handling

#### CityLevelIndex
Constructs city/CBSA-level indices:
- Weighted aggregation across supertracts
- Multiple weighting scheme support
- Year-over-year appreciation calculation

## Data Schema Requirements

### Transaction Data
Required columns:
- `property_id`: Unique property identifier
- `transaction_date`: Transaction date
- `transaction_price`: Sale price
- `census_tract`: Census tract ID
- `cbsa_code`: CBSA code
- `distance_to_cbd`: Distance to central business district

### Geographic Data
Required columns:
- `tract_id`: Census tract ID
- `cbsa_code`: CBSA code
- `centroid_lat`, `centroid_lon`: Tract centroid coordinates
- `housing_units`: Number of housing units
- `housing_value`: Total housing value
- `college_share`: Share of college-educated residents
- `nonwhite_share`: Share of non-white residents

## Output Format

### Tract-Level Indices
- `tract_id`: Census tract identifier
- `year`: Index year
- `index_value`: Index value (base=100)
- `appreciation_rate`: Year-over-year appreciation percentage
- `supertract_id`: Associated supertract identifier

### City-Level Indices  
- `cbsa_code`: CBSA identifier
- `year`: Index year
- `index_value`: Index value (base=100)
- `appreciation_rate`: Year-over-year appreciation percentage
- `weight_scheme`: Weighting scheme used
- `n_supertracts`: Number of supertracts aggregated
- `total_weight`: Sum of weights used

## Performance Considerations

### Memory Usage
- Use `use_lazy_evaluation=True` for large datasets
- Set appropriate `chunk_size` for memory-constrained environments
- Consider `n_jobs=1` for memory-intensive operations

### Processing Speed
- Enable parallel processing with `n_jobs > 1`
- Use Parquet format for optimal I/O performance
- Enable checkpointing for long-running processes

### Scaling Guidelines
- < 1M transactions: Standard configuration
- 1M-10M transactions: Enable lazy evaluation, increase chunk size
- > 10M transactions: Use distributed processing (future feature)

## Error Handling

The library provides comprehensive error handling:
- `ConfigurationError`: Invalid configuration parameters
- `DataValidationError`: Data quality or schema issues
- `ProcessingError`: Pipeline execution failures

## Logging

Structured logging is provided via `structlog`:
```python
import structlog
logger = structlog.get_logger()
```

Log levels:
- `DEBUG`: Detailed processing information
- `INFO`: General progress and results
- `WARNING`: Non-fatal issues
- `ERROR`: Processing failures

## Validation and Testing

The library includes comprehensive validation tools:
```python
from hpi_fhfa.validation import HPIValidator

validator = HPIValidator(tolerance=0.001)
results = validator.validate_all(tract_indices, city_indices)
print(validator.get_summary_report())
```

## Performance Benchmarking

Built-in benchmarking capabilities:
```python
from hpi_fhfa.validation import benchmark_pipeline

result = benchmark_pipeline(config)
print(f"Processed {result.n_transactions:,} transactions in {result.duration_seconds:.1f}s")
```
"""


def _generate_modules_doc() -> str:
    """Generate modules documentation."""
    return """# HPI-FHFA Modules Documentation

## Module Structure

```
hpi_fhfa/
├── config/                 # Configuration management
│   ├── settings.py         # Main configuration class
│   └── constants.py        # System constants
├── data/                   # Data handling
│   ├── loader.py          # Data loading utilities
│   ├── schemas.py         # Data schema definitions
│   ├── validators.py      # Data validation
│   └── filters.py         # Transaction filtering
├── models/                 # Core algorithms
│   ├── bmn_regression.py  # Bailey-Muth-Nourse regression
│   ├── supertract.py      # Supertract construction
│   └── weighting.py       # Weighting schemes
├── processing/            # Data processing pipeline
│   ├── pipeline.py        # Main pipeline orchestration
│   ├── repeat_sales.py    # Repeat sales identification
│   └── half_pairs.py      # Half-pairs calculation
├── indices/               # Index construction
│   ├── tract_level.py     # Tract-level indices
│   └── city_level.py      # City-level indices
├── utils/                 # Utilities
│   ├── exceptions.py      # Custom exceptions
│   ├── logging.py         # Logging configuration
│   └── performance.py     # Performance optimization
├── validation/            # Validation and testing
│   ├── validators.py      # Result validation
│   └── benchmarks.py      # Performance benchmarking
└── docs/                  # Documentation
    ├── api_docs.py        # API documentation generator
    └── examples.py        # Usage examples
```

## Core Modules

### config.settings

**HPIConfig Class**
- Main configuration class for pipeline parameters
- Automatic validation of paths and parameters
- Support for serialization/deserialization

**Key Configuration Groups:**
- Data paths and formats
- Processing parameters
- Performance tuning
- Validation settings

### data.loader

**DataLoader Class**
- Multi-format data loading (Parquet, CSV, Arrow)
- Lazy vs eager evaluation support
- Schema casting and validation
- Memory-efficient chunked processing

**Supported Formats:**
- Parquet (recommended for performance)
- CSV with automatic type inference
- Arrow/Feather for cross-language compatibility

### data.validators

**DataValidator Class**  
- Schema compliance validation
- Data quality assessment
- Duplicate detection and handling
- Statistical validation

**Validation Types:**
- Schema validation (column types, names)
- Data quality (missing values, outliers)
- Business rule validation (date ranges, price reasonableness)

### models.bmn_regression

**BMNRegression Class**
- Bailey-Muth-Nourse regression implementation
- Sparse matrix optimization for large datasets
- Regularization and numerical stability features
- Index value extraction and confidence intervals

**Key Features:**
- Efficient sparse matrix construction
- Robust numerical methods
- Memory-optimized for large datasets

### models.supertract

**SupertractBuilder Class**
- Dynamic supertract construction algorithm
- Distance-based clustering with geographic constraints
- Minimum half-pairs threshold enforcement
- Centroid calculation and updates

**Algorithm Features:**
- Haversine distance calculation
- Iterative merging with neighbor finding
- Geographic constraint enforcement

### processing.pipeline

**HPIPipeline Class**
- Main pipeline orchestration
- Checkpointing and resumption support
- Parallel processing coordination
- Comprehensive error handling

**Pipeline Steps:**
1. Data loading and validation
2. Repeat sales identification
3. Transaction filtering
4. Half-pairs calculation
5. Supertract construction
6. BMN regression execution
7. Index construction

### validation.validators

**HPIValidator Class**
- Comprehensive result validation
- Reference implementation comparison
- Statistical property validation
- Cross-consistency checks

**Validation Categories:**
- Data integrity checks
- Numerical accuracy validation
- Business logic compliance
- Performance requirement verification

## Extension Points

### Custom Weighting Schemes

Implement the `WeightingScheme` interface:
```python
from hpi_fhfa.models.weighting import WeightingScheme

class CustomWeights(WeightingScheme):
    def calculate_weights(self, data, period, geographic_df, transaction_df):
        # Implement custom weighting logic
        return weights_df
```

### Custom Filters

Extend the `TransactionFilter` class:
```python
from hpi_fhfa.data.filters import TransactionFilter

class EnhancedFilter(TransactionFilter):
    def apply_custom_filter(self, df):
        # Implement additional filtering logic
        return filtered_df
```

### Custom Validation

Implement custom validation rules:
```python
from hpi_fhfa.validation.validators import ValidationResult

def custom_validation(indices):
    # Implement custom validation logic
    return ValidationResult(...)
```

## Integration Guidelines

### Database Integration
- Use DataLoader with custom data sources
- Implement streaming for large datasets
- Consider partitioning strategies

### Web API Integration
- Wrap HPIPipeline in REST API
- Implement asynchronous processing
- Add progress monitoring endpoints

### Monitoring Integration
- Use structured logging output
- Implement custom metrics collection
- Add alerting for validation failures
"""


def _generate_config_doc() -> str:
    """Generate configuration documentation."""
    return """# HPI-FHFA Configuration Guide

## Overview

The HPI-FHFA library uses the `HPIConfig` class to manage all configuration parameters. This guide covers all available options and their recommended values.

## Basic Configuration

### Required Parameters

```python
from hpi_fhfa.config.settings import HPIConfig
from pathlib import Path

config = HPIConfig(
    # Data file paths (required)
    transaction_data_path=Path("data/transactions.parquet"),
    geographic_data_path=Path("data/geographic.parquet"),
    output_path=Path("output/"),
    
    # Time range (required)
    start_year=2020,
    end_year=2023
)
```

### Optional Parameters

```python
config = HPIConfig(
    # ... required parameters ...
    
    # Processing options
    weight_schemes=["sample", "value", "unit"],  # Default: all 6 schemes
    n_jobs=4,                                    # Default: 1 (sequential)
    chunk_size=100000,                          # Default: 50000
    
    # Data validation
    validate_data=True,                         # Default: True
    strict_validation=False,                    # Default: False
    
    # Performance tuning
    use_lazy_evaluation=True,                   # Default: True
    checkpoint_frequency=10,                    # Default: 10 periods
)
```

## Parameter Reference

### Data Parameters

#### `transaction_data_path: Path`
Path to transaction data file.

**Supported formats:** Parquet (recommended), CSV, Arrow/Feather
**File size:** No hard limit, but consider memory constraints
**Required columns:** property_id, transaction_date, transaction_price, census_tract, cbsa_code, distance_to_cbd

#### `geographic_data_path: Path`  
Path to geographic/census data file.

**Supported formats:** Parquet (recommended), CSV, Arrow/Feather
**Typical size:** 50MB-500MB depending on geographic coverage
**Required columns:** tract_id, cbsa_code, centroid_lat, centroid_lon, housing_units, housing_value, college_share, nonwhite_share

#### `output_path: Path`
Directory for output files.

**Created automatically** if doesn't exist
**Required space:** 10-100MB per million transactions
**Output files:**
- `tract_level_indices.parquet`
- `city_level_indices_{scheme}.parquet` (one per weight scheme)
- `metadata.json`
- `checkpoints/` (if checkpointing enabled)

### Time Range Parameters

#### `start_year: int`, `end_year: int`
Year range for index calculation.

**Valid range:** 1975-2030 (system enforced)
**Minimum span:** 2 years
**Recommended span:** 5-10 years for meaningful trends
**Performance impact:** Linear in number of years

### Processing Parameters

#### `weight_schemes: List[str]`
List of weighting schemes to calculate.

**Available schemes:**
- `"sample"`: Equal weighting (fastest)
- `"value"`: Housing value-weighted
- `"unit"`: Housing unit-weighted
- `"upb"`: Unpaid principal balance-weighted
- `"college"`: College share-weighted
- `"nonwhite"`: Non-white share-weighted

**Default:** All 6 schemes
**Performance impact:** Linear in number of schemes
**Recommended:** Start with `["sample"]` for testing

#### `n_jobs: int`
Number of parallel processing jobs.

**Range:** 1 to number of CPU cores
**Default:** 1 (sequential processing)
**Memory impact:** Each job uses additional memory
**Recommended:** 2-4 for most systems, 1 for memory-constrained environments

#### `chunk_size: int`
Processing chunk size for large datasets.

**Range:** 1,000 to 1,000,000
**Default:** 50,000
**Memory impact:** Larger chunks use more memory but may be faster
**I/O impact:** Smaller chunks have more I/O overhead
**Recommended:** 100,000 for most datasets

### Validation Parameters

#### `validate_data: bool`
Enable data validation during loading.

**Default:** True
**Performance impact:** 5-10% processing overhead
**Recommended:** True for production, False for testing with known-good data

#### `strict_validation: bool`
Enable strict validation mode.

**Default:** False
**Impact:** Fails on warnings that would normally be logged
**Use case:** High-quality production environments

### Performance Parameters

#### `use_lazy_evaluation: bool`
Enable Polars lazy evaluation.

**Default:** True
**Memory benefit:** Significantly reduced memory usage for large datasets
**Performance:** May be faster for complex queries
**Compatibility:** Some operations require eager evaluation

#### `checkpoint_frequency: int`
Frequency of checkpoint saves (in periods).

**Range:** 0 (disabled) to 100
**Default:** 10
**Disk usage:** Each checkpoint uses 10-100MB
**Recovery benefit:** Resume processing from last checkpoint
**Recommended:** 5-10 for long-running processes, 0 for short runs

## Environment-Specific Configurations

### Development Environment

```python
config = HPIConfig(
    transaction_data_path=Path("test_data/small_transactions.parquet"),
    geographic_data_path=Path("test_data/test_geographic.parquet"),
    output_path=Path("dev_output/"),
    start_year=2020,
    end_year=2022,
    weight_schemes=["sample"],          # Single scheme for speed
    n_jobs=1,                          # Avoid parallel overhead
    validate_data=False,               # Skip validation for speed
    checkpoint_frequency=0,            # Disable checkpointing
    use_lazy_evaluation=False          # Simpler debugging
)
```

### Production Environment

```python
config = HPIConfig(
    transaction_data_path=Path("/data/production/transactions.parquet"),
    geographic_data_path=Path("/data/production/geographic.parquet"),
    output_path=Path("/output/hpi/"),
    start_year=1989,
    end_year=2023,
    weight_schemes=["sample", "value", "unit"],  # Common schemes
    n_jobs=8,                                   # Utilize available cores
    validate_data=True,                         # Full validation
    strict_validation=True,                     # Fail on warnings
    checkpoint_frequency=5,                     # Frequent checkpoints
    use_lazy_evaluation=True,                   # Memory optimization
    chunk_size=200000                          # Large chunks for efficiency
)
```

### Memory-Constrained Environment

```python
config = HPIConfig(
    # ... paths and years ...
    weight_schemes=["sample"],          # Single scheme only
    n_jobs=1,                          # No parallel processing
    chunk_size=25000,                  # Smaller chunks
    use_lazy_evaluation=True,          # Essential for memory savings
    checkpoint_frequency=2,            # Frequent saves
    validate_data=True,                # Catch issues early
)
```

### High-Performance Environment

```python
config = HPIConfig(
    # ... paths and years ...
    weight_schemes=["sample", "value", "unit", "upb", "college", "nonwhite"],
    n_jobs=16,                         # Max parallelization
    chunk_size=500000,                 # Large chunks
    use_lazy_evaluation=True,          # Still beneficial
    checkpoint_frequency=20,           # Less frequent saves
    validate_data=True,
    strict_validation=False
)
```

## Configuration Validation

The system automatically validates configuration:

### Path Validation
- Files must exist and be readable
- Output directory created if needed
- File formats validated

### Parameter Validation  
- Year ranges checked for reasonableness
- Numeric parameters validated for sensible ranges
- Weight schemes validated against available implementations

### Resource Validation
- Available memory estimated
- CPU core count checked
- Disk space requirements calculated

## Best Practices

### 1. Start Small
Begin with a subset of data and single weight scheme to verify configuration.

### 2. Monitor Memory Usage
Use system monitoring to tune `chunk_size` and `n_jobs` parameters.

### 3. Profile Performance
Use built-in benchmarking to optimize configuration:
```python
from hpi_fhfa.validation import benchmark_pipeline
result = benchmark_pipeline(config)
```

### 4. Use Checkpointing
Enable checkpointing for runs longer than 30 minutes.

### 5. Validate Early
Keep validation enabled during development to catch data issues early.

### 6. Environment Variables
Consider using environment variables for deployment-specific settings:
```python
import os
config = HPIConfig(
    n_jobs=int(os.getenv('HPI_N_JOBS', '4')),
    chunk_size=int(os.getenv('HPI_CHUNK_SIZE', '100000'))
)
```
"""


def _generate_examples_doc() -> str:
    """Generate examples documentation."""
    return """# HPI-FHFA Usage Examples

## Basic Usage

### Simple Pipeline Execution

```python
from hpi_fhfa.processing.pipeline import HPIPipeline
from hpi_fhfa.config.settings import HPIConfig
from pathlib import Path

# Basic configuration
config = HPIConfig(
    transaction_data_path=Path("data/transactions.parquet"),
    geographic_data_path=Path("data/geographic.parquet"),
    output_path=Path("output/"),
    start_year=2020,
    end_year=2023,
    weight_schemes=["sample", "value"]
)

# Run pipeline
pipeline = HPIPipeline(config)
results = pipeline.run()

# Check results
print(f"Generated {len(results.tract_indices)} tract indices")
print(f"Generated {len(results.city_indices)} city index sets")
print(f"Processing time: {results.metadata['processing_time']:.1f} seconds")
```

### Working with Results

```python
# Access tract-level indices
tract_df = results.tract_indices
print("Tract indices summary:")
print(tract_df.describe())

# Access city-level indices by weight scheme
for scheme, city_df in results.city_indices.items():
    print(f"\\n{scheme.title()} weighted city indices:")
    print(f"  CBSAs covered: {city_df['cbsa_code'].n_unique()}")
    print(f"  Years covered: {city_df['year'].n_unique()}")
    
    # Calculate average appreciation by year
    annual_appreciation = (
        city_df
        .filter(pl.col("appreciation_rate").is_not_null())
        .group_by("year")
        .agg(pl.col("appreciation_rate").mean().alias("avg_appreciation"))
        .sort("year")
    )
    print(f"  Average annual appreciation:")
    for row in annual_appreciation.iter_rows(named=True):
        print(f"    {row['year']}: {row['avg_appreciation']:.2f}%")
```

## Advanced Usage

### Custom Configuration

```python
# High-performance configuration
config = HPIConfig(
    transaction_data_path=Path("large_dataset.parquet"),
    geographic_data_path=Path("geographic.parquet"),
    output_path=Path("output/"),
    start_year=2015,
    end_year=2023,
    weight_schemes=["sample", "value", "unit"],
    n_jobs=8,                    # Parallel processing
    chunk_size=200000,           # Large chunks for efficiency
    use_lazy_evaluation=True,    # Memory optimization
    checkpoint_frequency=5,      # Regular checkpoints
    validate_data=True
)

# Run with progress monitoring
import time
start_time = time.time()

pipeline = HPIPipeline(config)
results = pipeline.run()

duration = time.time() - start_time
print(f"Processed {results.metadata['n_transactions']:,} transactions in {duration:.1f}s")
print(f"Throughput: {results.metadata['n_transactions']/duration:.0f} transactions/sec")
```

### Error Handling

```python
from hpi_fhfa.utils.exceptions import ProcessingError, DataValidationError, ConfigurationError

try:
    config = HPIConfig(
        transaction_data_path=Path("data.parquet"),
        geographic_data_path=Path("geo.parquet"),
        output_path=Path("output/"),
        start_year=2020,
        end_year=2021
    )
    
    pipeline = HPIPipeline(config)
    results = pipeline.run()
    
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle configuration issues
    
except DataValidationError as e:
    print(f"Data validation error: {e}")
    # Handle data quality issues
    
except ProcessingError as e:
    print(f"Processing error: {e}")
    # Handle pipeline execution issues
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected issues
```

## Data Preparation Examples

### Loading and Exploring Data

```python
import polars as pl
from hpi_fhfa.data.loader import DataLoader
from hpi_fhfa.data.validators import DataValidator

# Load data using the built-in loader
config = HPIConfig(...)  # Your configuration
loader = DataLoader(config)

# Load and explore transaction data
transactions = loader.load_transactions()
print("Transaction data summary:")
print(f"  Rows: {len(transactions):,}")
print(f"  Columns: {transactions.width}")
print(f"  Date range: {transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}")
print(f"  Price range: ${transactions['transaction_price'].min():,.0f} to ${transactions['transaction_price'].max():,.0f}")

# Validate data quality
validator = DataValidator(config)
validation_results = validator.validate_transactions(transactions)
print(f"Validation completed with {len(validation_results)} issues identified")
```

### Custom Data Filtering

```python
from hpi_fhfa.data.filters import TransactionFilter

# Apply standard filters
filter = TransactionFilter()
filtered_data = filter.apply_filters(repeat_sales_data)

# Get filtering summary
summary = filter.get_filter_summary()
print("Filter summary:")
for row in summary.iter_rows(named=True):
    print(f"  {row['filter']}: removed {row['removed']} transactions ({row['pct']})")

# Apply custom filtering
custom_filtered = (
    filtered_data
    .filter(pl.col("transaction_price") >= 50000)  # Minimum price
    .filter(pl.col("transaction_price") <= 2000000)  # Maximum price
    .filter(pl.col("distance_to_cbd") <= 50)  # Within 50 miles of CBD
)

print(f"Custom filtering: {len(filtered_data)} → {len(custom_filtered)} transactions")
```

## Validation and Testing Examples

### Result Validation

```python
from hpi_fhfa.validation import HPIValidator

# Create validator with 0.1% tolerance
validator = HPIValidator(tolerance=0.001)

# Validate results (without reference data)
validation_results = validator.validate_all(
    tract_indices=results.tract_indices,
    city_indices=results.city_indices
)

# Print summary
print(validator.get_summary_report())

# Check specific validation results
failed_tests = [r for r in validation_results if not r.passed]
if failed_tests:
    print(f"\\n{len(failed_tests)} validation tests failed:")
    for test in failed_tests:
        print(f"  - {test.test_name}: {test.message}")
```

### Performance Benchmarking

```python
from hpi_fhfa.validation import PerformanceBenchmark

# Create benchmark suite
benchmark = PerformanceBenchmark()

# Benchmark different configurations
configs = [
    ("Sequential", HPIConfig(..., n_jobs=1)),
    ("Parallel-4", HPIConfig(..., n_jobs=4)),
    ("Large-chunks", HPIConfig(..., chunk_size=200000)),
]

for name, config in configs:
    result = benchmark.benchmark_pipeline(config, name=name)
    print(f"{name}: {result.duration_seconds:.1f}s, {result.peak_memory_mb:.0f}MB")

# Generate comprehensive report
print("\\n" + benchmark.get_summary_report())
```

## Integration Examples

### Batch Processing Multiple Datasets

```python
from pathlib import Path
import pandas as pd

def process_multiple_datasets(data_directory: Path):
    \"\"\"Process multiple transaction datasets.\"\"\"
    
    results_summary = []
    
    for data_file in data_directory.glob("transactions_*.parquet"):
        # Extract year or region from filename
        dataset_name = data_file.stem
        
        print(f"Processing {dataset_name}...")
        
        try:
            config = HPIConfig(
                transaction_data_path=data_file,
                geographic_data_path=data_directory / "geographic.parquet",
                output_path=data_directory / "output" / dataset_name,
                start_year=2020,
                end_year=2023,
                weight_schemes=["sample", "value"],
                validate_data=True
            )
            
            pipeline = HPIPipeline(config)
            results = pipeline.run()
            
            # Collect summary statistics
            results_summary.append({
                "dataset": dataset_name,
                "n_transactions": results.metadata["n_transactions"],
                "n_tracts": len(results.tract_indices["tract_id"].unique()),
                "n_cbsas": len(results.city_indices["sample"]["cbsa_code"].unique()),
                "processing_time": results.metadata["processing_time"],
                "status": "success"
            })
            
        except Exception as e:
            print(f"  Failed: {e}")
            results_summary.append({
                "dataset": dataset_name,
                "status": "failed",
                "error": str(e)
            })
    
    # Create summary report
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(data_directory / "processing_summary.csv", index=False)
    
    return summary_df

# Usage
summary = process_multiple_datasets(Path("data/regional_datasets/"))
print(summary)
```

### Web API Integration

```python
from flask import Flask, request, jsonify
import tempfile
import json

app = Flask(__name__)

@app.route('/calculate_hpi', methods=['POST'])
def calculate_hpi():
    \"\"\"Web API endpoint for HPI calculation.\"\"\"
    
    try:
        # Parse request
        data = request.get_json()
        
        # Validate required parameters
        required_params = ['transaction_data_url', 'geographic_data_url', 'start_year', 'end_year']
        for param in required_params:
            if param not in data:
                return jsonify({"error": f"Missing required parameter: {param}"}), 400
        
        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Download data files (implement your download logic)
            txn_path = download_data(data['transaction_data_url'], temp_path / "transactions.parquet")
            geo_path = download_data(data['geographic_data_url'], temp_path / "geographic.parquet")
            
            # Configure pipeline
            config = HPIConfig(
                transaction_data_path=txn_path,
                geographic_data_path=geo_path,
                output_path=temp_path / "output",
                start_year=data['start_year'],
                end_year=data['end_year'],
                weight_schemes=data.get('weight_schemes', ['sample']),
                n_jobs=data.get('n_jobs', 2),
                validate_data=data.get('validate_data', True)
            )
            
            # Run pipeline
            pipeline = HPIPipeline(config)
            results = pipeline.run()
            
            # Prepare response
            response = {
                "status": "success",
                "metadata": results.metadata,
                "summary": {
                    "n_tract_indices": len(results.tract_indices),
                    "n_city_indices": {scheme: len(df) for scheme, df in results.city_indices.items()},
                    "years_covered": sorted(results.tract_indices["year"].unique().to_list())
                }
            }
            
            # Optionally include sample results
            if data.get('include_sample_data', False):
                response["sample_tract_indices"] = results.tract_indices.head(10).to_dicts()
                response["sample_city_indices"] = {
                    scheme: df.head(5).to_dicts() 
                    for scheme, df in results.city_indices.items()
                }
            
            return jsonify(response)
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def download_data(url: str, local_path: Path) -> Path:
    \"\"\"Download data file from URL to local path.\"\"\"
    # Implement your download logic here
    # This is a placeholder
    import requests
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)
    return local_path

if __name__ == '__main__':
    app.run(debug=True)
```

### Monitoring and Alerting

```python
import logging
from datetime import datetime
from hpi_fhfa.validation import HPIValidator

# Configure logging for monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hpi_processing.log'),
        logging.StreamHandler()
    ]
)

def run_monitored_pipeline(config: HPIConfig, alert_thresholds: dict = None):
    \"\"\"Run pipeline with comprehensive monitoring and alerting.\"\"\"
    
    alert_thresholds = alert_thresholds or {
        'max_processing_time': 3600,  # 1 hour
        'max_memory_mb': 8192,        # 8GB
        'min_tract_coverage': 0.95,   # 95% of expected tracts
        'max_validation_failures': 0  # No validation failures allowed
    }
    
    start_time = datetime.now()
    
    try:
        # Run pipeline
        pipeline = HPIPipeline(config)
        results = pipeline.run()
        
        # Validate results
        validator = HPIValidator(tolerance=0.001)
        validation_results = validator.validate_all(
            results.tract_indices, 
            results.city_indices
        )
        
        # Check alerting thresholds
        alerts = []
        
        # Processing time check
        processing_time = results.metadata['processing_time']
        if processing_time > alert_thresholds['max_processing_time']:
            alerts.append(f"Processing time exceeded threshold: {processing_time:.0f}s > {alert_thresholds['max_processing_time']}s")
        
        # Validation failures check
        failed_validations = [r for r in validation_results if not r.passed]
        if len(failed_validations) > alert_thresholds['max_validation_failures']:
            alerts.append(f"Validation failures: {len(failed_validations)} > {alert_thresholds['max_validation_failures']}")
        
        # Coverage check (example: expected number of tracts)
        expected_tracts = 1000  # Set based on your data
        actual_tracts = len(results.tract_indices["tract_id"].unique())
        coverage = actual_tracts / expected_tracts
        if coverage < alert_thresholds['min_tract_coverage']:
            alerts.append(f"Tract coverage below threshold: {coverage:.2%} < {alert_thresholds['min_tract_coverage']:.2%}")
        
        # Log results
        if alerts:
            logging.warning(f"Pipeline completed with {len(alerts)} alerts:")
            for alert in alerts:
                logging.warning(f"  ALERT: {alert}")
        else:
            logging.info("Pipeline completed successfully with no alerts")
        
        # Log summary statistics
        logging.info(f"Processing summary:")
        logging.info(f"  Duration: {processing_time:.1f}s")
        logging.info(f"  Transactions: {results.metadata['n_transactions']:,}")
        logging.info(f"  Tract indices: {len(results.tract_indices):,}")
        logging.info(f"  City indices: {sum(len(df) for df in results.city_indices.values()):,}")
        
        return results, alerts
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        # Send alert (implement your alerting system)
        send_alert(f"HPI Pipeline Failed: {e}")
        raise

def send_alert(message: str):
    \"\"\"Send alert via your preferred alerting system.\"\"\"
    # Implement email, Slack, PagerDuty, etc. integration
    logging.critical(f"ALERT: {message}")

# Usage
config = HPIConfig(...)  # Your configuration
results, alerts = run_monitored_pipeline(config)
```

These examples demonstrate the flexibility and power of the HPI-FHFA library for various use cases, from simple academic research to production-scale financial applications.
"""