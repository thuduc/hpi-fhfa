# HPI-FHFA Implementation Plan - Python/Pandas Technical Stack

## 1. Project Overview

This implementation plan details the development of the Federal Housing Finance Agency (FHFA) Repeat-Sales Aggregation Index (RSAI) method using Python and Pandas as the core technical stack. The system will handle large-scale housing transaction data to produce balanced panel house price indices at both tract and city levels.

### 1.1 Key Technical Requirements
- Process 63.3 million repeat-sales pairs
- Handle 63,122 census tracts across 581 CBSAs
- Generate annual indices from 1989-2021
- Achieve 80%+ test coverage for core model code

### 1.2 Technology Stack
- **Core**: Python 3.9+, Pandas 1.5+, NumPy 1.23+
- **ML/Statistics**: scikit-learn, statsmodels, scipy
- **Geographic**: GeoPandas, Shapely, pyproj
- **Performance**: Numba, Dask (for large-scale processing)
- **Testing**: pytest, pytest-cov, hypothesis
- **Validation**: pandera (data validation schemas)

## 2. Architecture Design

### 2.1 Module Structure
```
hpi_fhfa/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── constants.py         # MIN_HALF_PAIRS, thresholds, etc.
│   └── settings.py          # Configuration management
├── data/
│   ├── __init__.py
│   ├── schemas.py           # Pandera data validation schemas
│   ├── loaders.py           # Data loading utilities
│   └── filters.py           # Transaction filtering logic
├── models/
│   ├── __init__.py
│   ├── bmn_regression.py    # Bailey-Muth-Nourse implementation
│   ├── price_relatives.py   # Price relative calculations
│   └── repeat_sales.py      # Repeat-sales pair construction
├── geography/
│   ├── __init__.py
│   ├── census_tract.py      # Census tract handling
│   ├── supertract.py        # Supertract algorithm
│   └── distance.py          # Geographic distance calculations
├── aggregation/
│   ├── __init__.py
│   ├── weights.py           # Weight calculation methods
│   ├── index_builder.py     # Index construction logic
│   └── city_level.py        # City-level aggregation
├── utils/
│   ├── __init__.py
│   ├── performance.py       # Performance optimization utilities
│   └── validation.py        # Data validation helpers
└── tests/
    ├── __init__.py
    ├── unit/                # Unit tests for each module
    ├── integration/         # End-to-end integration tests
    └── fixtures/            # Test data and fixtures
```

### 2.2 Core Classes

```python
# Key data structures
class TransactionData:
    """Container for property transaction data with validation"""
    
class CensusTract:
    """Geographic unit with boundaries and properties"""
    
class Supertract:
    """Dynamic aggregation of census tracts"""
    
class RepeatSalesPair:
    """Validated repeat-sales transaction pair"""
    
class HPIIndex:
    """House price index with metadata and methods"""

# Processing pipelines
class BMNRegressor:
    """Bailey-Muth-Nourse regression implementation"""
    
class SupertractBuilder:
    """Algorithm for dynamic tract aggregation"""
    
class IndexAggregator:
    """Weighted index aggregation framework"""
```

## 3. Data Processing Pipeline

### 3.1 Data Ingestion and Validation

```python
# schemas.py - Pandera schemas for data validation
transaction_schema = pa.DataFrameSchema({
    "property_id": pa.Column(str, nullable=False),
    "transaction_date": pa.Column(pd.Timestamp, nullable=False),
    "transaction_price": pa.Column(float, pa.Check.greater_than(0)),
    "census_tract": pa.Column(str, regex=r'^\d{11}$'),
    "cbsa_code": pa.Column(str, nullable=False),
    "distance_to_cbd": pa.Column(float, pa.Check.greater_than_equal(0))
})

# Data loading with validation
def load_transactions(filepath: str) -> pd.DataFrame:
    df = pd.read_parquet(filepath)
    return transaction_schema.validate(df)
```

### 3.2 Repeat-Sales Pair Construction

```python
def construct_repeat_sales_pairs(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    1. Group transactions by property_id
    2. Create all valid pairs
    3. Calculate price relatives
    4. Apply filters (growth rate, time period)
    """
    
def calculate_price_relatives(pair: RepeatSalesPair) -> float:
    """p_itτ = log(price_t) - log(price_τ)"""
    
def apply_filters(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Remove pairs with:
    - Same 12-month period
    - CAGR > |30%|
    - Cumulative appreciation > 10x or < 0.25x
    """
```

### 3.3 Geographic Processing

```python
class SupertractAlgorithm:
    def __init__(self, min_half_pairs: int = 40):
        self.min_half_pairs = min_half_pairs
    
    def build_supertracts(self, 
                         tracts: List[CensusTract], 
                         transactions: pd.DataFrame,
                         period: int) -> List[Supertract]:
        """
        1. Calculate half-pairs per tract
        2. Merge low-volume tracts with nearest neighbors
        3. Return optimized supertract configuration
        """
        
def calculate_centroid_distances(tract1: CensusTract, 
                                tract2: CensusTract) -> float:
    """Calculate distance between tract centroids"""
```

### 3.4 BMN Regression Implementation

```python
class BMNRegressor:
    def fit(self, repeat_sales_pairs: pd.DataFrame) -> BMNResults:
        """
        Implement Bailey-Muth-Nourse regression:
        p_itτ = D'_tτ * δ_tτ + ε_itτ
        
        Using sparse matrix for efficiency
        """
        
    def predict(self, period_t: int, period_t_1: int) -> float:
        """Calculate p̂_pooled(t,t-1) = δ̂_t - δ̂_t-1"""
```

### 3.5 Weight Calculation Framework

```python
class WeightCalculator:
    """Factory for different weighting schemes"""
    
    @staticmethod
    def calculate_weights(weight_type: str, 
                         data: pd.DataFrame) -> pd.Series:
        """
        Support weight types:
        - sample: Share of half-pairs
        - value: Share of aggregate housing value
        - unit: Share of housing units
        - upb: Share of unpaid principal balance
        - college: Share of college-educated population
        - nonwhite: Share of non-white population
        """
```

### 3.6 Index Construction

```python
class CityLevelIndexBuilder:
    def build_annual_index(self, 
                          transactions: pd.DataFrame,
                          weight_type: str,
                          start_year: int = 1989) -> HPIIndex:
        """
        Algorithm:
        1. Initialize P_a(t=0) = 1
        2. For each period:
           - Construct supertracts
           - Calculate BMN indices
           - Apply weights
           - Aggregate to city level
        """
```

## 4. Performance Optimization Strategy

### 4.1 Memory Optimization
- Use categorical dtype for repeated string values (property_id, census_tract)
- Implement chunked processing for large datasets
- Utilize sparse matrices for BMN regression design matrix

### 4.2 Computational Optimization
- Numba JIT compilation for distance calculations
- Vectorized operations for price relative calculations
- Parallel processing for independent supertract regressions
- Caching for repeated geographic calculations

### 4.3 Scalability Approach
```python
# Use Dask for large-scale processing
import dask.dataframe as dd

def process_large_dataset(filepath: str) -> dd.DataFrame:
    """Process data that doesn't fit in memory"""
    ddf = dd.read_parquet(filepath, blocksize="128MB")
    # Implement distributed processing logic
```

## 5. Testing Strategy

### 5.1 Unit Testing Structure

```python
# test_price_relatives.py
class TestPriceRelatives:
    def test_basic_calculation(self):
        """Test log price difference calculation"""
        
    def test_filter_same_period(self):
        """Test filtering of same 12-month period"""
        
    def test_growth_rate_filter(self):
        """Test CAGR > |30%| filter"""
        
    @hypothesis.given(
        price1=st.floats(min_value=1000, max_value=1e7),
        price2=st.floats(min_value=1000, max_value=1e7)
    )
    def test_price_relative_properties(self, price1, price2):
        """Property-based testing for price relatives"""

# test_bmn_regression.py
class TestBMNRegression:
    def test_design_matrix_construction(self):
        """Test dummy variable matrix creation"""
        
    def test_regression_coefficients(self):
        """Test coefficient estimation accuracy"""
        
    def test_sparse_matrix_efficiency(self):
        """Test memory usage with sparse matrices"""

# test_supertract.py
class TestSupertractAlgorithm:
    def test_minimum_threshold_enforcement(self):
        """Test MIN_HALF_PAIRS threshold"""
        
    def test_nearest_neighbor_merging(self):
        """Test geographic merging logic"""
        
    def test_iterative_aggregation(self):
        """Test multiple merge iterations"""
```

### 5.2 Integration Testing

```python
# test_end_to_end.py
class TestEndToEndPipeline:
    def test_full_index_construction(self, sample_data):
        """Test complete pipeline from raw data to indices"""
        
    def test_multi_cbsa_processing(self):
        """Test processing multiple CBSAs"""
        
    def test_weight_scheme_consistency(self):
        """Verify all weight schemes produce valid indices"""
```

### 5.3 Performance Testing

```python
# test_performance.py
class TestPerformance:
    @pytest.mark.benchmark
    def test_large_dataset_processing(self, benchmark):
        """Benchmark processing 1M transactions"""
        
    def test_memory_usage(self):
        """Monitor memory consumption during processing"""
        
    def test_regression_scalability(self):
        """Test BMN regression with varying data sizes"""
```

### 5.4 Validation Testing

```python
# test_validation.py
class TestMathematicalValidation:
    def test_index_continuity(self):
        """Verify index continuity over time"""
        
    def test_weight_normalization(self):
        """Ensure weights sum to 1"""
        
    def test_appreciation_bounds(self):
        """Verify reasonable appreciation rates"""
```

### 5.5 Test Coverage Requirements

```yaml
# .coveragerc
[run]
source = hpi_fhfa
omit = 
    */tests/*
    */config/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    raise NotImplementedError
    if __name__ == "__main__":

[html]
directory = coverage_html_report

# Target: 80%+ coverage for core model code
# Critical modules requiring 90%+ coverage:
# - models/bmn_regression.py
# - models/price_relatives.py
# - geography/supertract.py
# - aggregation/index_builder.py
```

## 6. Data Validation Framework

```python
# Comprehensive validation using Pandera
class DataValidator:
    @staticmethod
    def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
        """Validate transaction data schema and constraints"""
        
    @staticmethod
    def validate_geographic_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate census tract and CBSA codes"""
        
    @staticmethod
    def validate_index_output(index: HPIIndex) -> bool:
        """Validate final index properties"""
```

## 7. Error Handling and Logging

```python
import logging
from typing import Optional

# Custom exceptions
class InsufficientDataError(Exception):
    """Raised when tract has insufficient half-pairs"""

class RegressionConvergenceError(Exception):
    """Raised when BMN regression fails to converge"""

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Error handling patterns
def safe_regression(data: pd.DataFrame) -> Optional[BMNResults]:
    try:
        return BMNRegressor().fit(data)
    except RegressionConvergenceError:
        logger.warning(f"Regression failed for {len(data)} pairs")
        return None
```

## 8. Implementation Timeline

### Phase 1: Core Data Structures (Week 1-2)
- Implement transaction data schemas
- Create geographic data structures
- Build repeat-sales pair construction
- Unit tests for data modules

### Phase 2: Mathematical Components (Week 3-4)
- Implement BMN regression
- Create price relative calculations
- Build filtering logic
- Unit tests for mathematical modules

### Phase 3: Geographic Processing (Week 5-6)
- Implement supertract algorithm
- Create distance calculations
- Build geographic aggregation
- Integration tests for geographic pipeline

### Phase 4: Index Construction (Week 7-8)
- Implement weight calculations
- Build index aggregation logic
- Create city-level indices
- End-to-end integration tests

### Phase 5: Optimization & Validation (Week 9-10)
- Performance optimization
- Large-scale data testing
- Validation against known results
- Documentation and code cleanup

### Phase 6: Testing & Coverage (Week 11-12)
- Achieve 80%+ test coverage
- Performance benchmarking
- Edge case testing
- Final validation

## 9. Deliverables

1. **Source Code**: Complete Python package with all modules
2. **Tests**: Comprehensive test suite with 80%+ coverage
3. **Documentation**: API documentation and usage guides
4. **Performance Report**: Benchmarks and scalability analysis
5. **Validation Report**: Comparison with expected results
6. **Example Notebooks**: Jupyter notebooks demonstrating usage

## 10. Risk Mitigation

### Technical Risks
- **Memory constraints**: Mitigated by chunked processing and Dask
- **Computational complexity**: Mitigated by sparse matrices and parallelization
- **Geographic accuracy**: Mitigated by robust distance calculations

### Data Risks
- **Missing data**: Handled by validation and error reporting
- **Data quality**: Addressed by comprehensive filtering
- **Edge cases**: Covered by extensive testing

## 11. Success Criteria

1. Process full dataset (63.3M pairs) within reasonable time
2. Achieve 80%+ test coverage for core modules
3. Produce indices matching FHFA methodology
4. Handle all edge cases gracefully
5. Provide clear documentation and examples