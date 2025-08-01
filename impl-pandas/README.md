# HPI-FHFA Implementation - Python/Pandas

Implementation of the Federal Housing Finance Agency (FHFA) Repeat-Sales Aggregation Index (RSAI) method for constructing house price indices, based on Contat & Larson (2022).

## Overview

This package implements the FHFA RSAI methodology which addresses limitations of traditional city-level house price indices by:
- Creating balanced panel of Census tract-level indices
- Handling low transaction counts through dynamic aggregation
- Supporting flexible weighting schemes for different use cases
- Avoiding sampling bias and composition effects

## Installation

```bash
# Install minimal requirements (recommended)
pip install -r requirements-minimal.txt

# Install all requirements from original plan
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install from source (after installing requirements)
pip install -e .
```

## Quick Start

```python
import pandas as pd
from hpi_fhfa.data import load_transactions, filter_transactions
from hpi_fhfa.models import construct_repeat_sales_pairs, BMNRegressor
from hpi_fhfa.config import Settings

# Load configuration
settings = Settings(
    min_half_pairs=40,
    max_cagr=0.30,
    base_year=1989
)

# Load and validate transaction data
transactions = load_transactions('path/to/transactions.parquet')

# Construct repeat sales pairs
pairs = construct_repeat_sales_pairs(transactions)

# Apply filters
filtered_pairs = filter_transactions(pairs)

# Fit BMN regression
regressor = BMNRegressor(use_sparse=True)
results = regressor.fit(filtered_pairs)

# Get appreciation rates
appreciation = results.get_appreciation(period_t=10, period_t_1=9)
```

## Features

### Phase 1: Core Data Structures (Completed)
- ✅ Configuration management with validation
- ✅ Pandera schemas for data validation
- ✅ Data loaders supporting multiple formats
- ✅ Comprehensive transaction filtering

### Phase 2: Mathematical Components (Completed)
- ✅ Bailey-Muth-Nourse (BMN) regression with sparse matrix support
- ✅ Price relative calculations
- ✅ Repeat sales pair construction
- ✅ Half-pairs calculation for tract aggregation

### Upcoming Phases
- Phase 3: Geographic Processing (Weeks 5-6)
- Phase 4: Index Construction (Weeks 7-8)
- Phase 5: Optimization & Validation (Weeks 9-10)
- Phase 6: Testing & Coverage (Weeks 11-12)

## Testing

Run tests with coverage:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=hpi_fhfa --cov-report=html

# Run specific test module
pytest tests/unit/test_bmn_regression.py

# Run with verbose output
pytest -v
```

## Project Structure

```
impl-pandas/
├── hpi_fhfa/          # Main package
│   ├── config/        # Configuration and constants
│   ├── data/          # Data schemas, loaders, filters
│   ├── models/        # BMN regression, price relatives
│   ├── geography/     # Geographic processing (upcoming)
│   └── aggregation/   # Index aggregation (upcoming)
├── tests/             # Test suite
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests (upcoming)
├── requirements.txt   # Package dependencies
└── setup.py          # Package setup
```

## Data Requirements

### Transaction Data Schema
- `property_id`: Unique property identifier
- `transaction_date`: Date of transaction
- `transaction_price`: Sale price (positive)
- `census_tract`: 11-digit 2010 Census tract ID
- `cbsa_code`: 5-digit Core-Based Statistical Area code
- `distance_to_cbd`: Distance to central business district

### Census Tract Data Schema
- `census_tract`: 11-digit tract identifier
- `cbsa_code`: CBSA code
- `centroid_lat/lon`: Geographic coordinates
- `housing_units`: Number of housing units (optional)
- `aggregate_value`: Total housing value (optional)
- `college_share`: Share of college-educated population (optional)
- `nonwhite_share`: Share of non-white population (optional)

## Configuration

Key parameters can be configured via the Settings class:

```python
from hpi_fhfa.config import Settings

settings = Settings(
    # Supertract parameters
    min_half_pairs=40,  # Minimum half-pairs per tract/year
    
    # Filtering thresholds
    max_cagr=0.30,  # Maximum compound annual growth rate
    
    # Index parameters
    base_year=1989,  # Base year for normalization
    start_year=1989,  # First year of index
    end_year=2021,   # Last year of index
    
    # Performance settings
    chunk_size=100000,  # Rows per chunk for large files
    n_jobs=-1,  # Number of parallel jobs
    use_sparse_matrices=True,  # Use sparse matrices
    use_numba=True  # Use Numba acceleration
)

# Save/load configuration
settings.to_json('config.json')
settings = Settings.from_json('config.json')
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

Contat, J. and Larson, W. (2022). "The Federal Housing Finance Agency Repeat-Sales Aggregation Index." FHFA Staff Working Paper.