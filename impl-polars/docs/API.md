# HPI-FHFA API Documentation

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
