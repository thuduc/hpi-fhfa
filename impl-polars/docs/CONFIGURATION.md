# HPI-FHFA Configuration Guide

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
