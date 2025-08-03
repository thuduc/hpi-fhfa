# HPI-FHFA Modules Documentation

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
