# House Price Index (HPI) - FHFA Method Implementation Plan
## Python & Polars Technical Stack

## 1. Executive Summary

This implementation plan outlines the development of the FHFA Repeat-Sales Aggregation Index (RSAI) method using Python with Polars as the primary data processing framework. The system will process millions of real estate transactions to create tract-level and city-level house price indices with comprehensive testing coverage.

### Key Technologies
- **Python 3.11+**: Core programming language
- **Polars**: High-performance DataFrame library for data processing
- **NumPy/SciPy**: Numerical computations and BMN regression
- **scikit-learn**: Sparse matrix operations and linear algebra
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **hypothesis**: Property-based testing
- **Dask/Ray** (optional): Distributed computing for large-scale processing

## 2. Project Structure

```
hpi-fhfa/
├── src/
│   ├── hpi_fhfa/
│   │   ├── __init__.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── settings.py         # Configuration management
│   │   │   └── constants.py        # System constants
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py           # Data ingestion with Polars
│   │   │   ├── validators.py       # Data validation rules
│   │   │   ├── filters.py          # Transaction filters
│   │   │   └── schemas.py          # Polars schema definitions
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── bmn_regression.py   # BMN regression implementation
│   │   │   ├── supertract.py       # Supertract algorithm
│   │   │   ├── aggregation.py      # Index aggregation logic
│   │   │   └── weighting.py        # Weighting scheme implementations
│   │   ├── processing/
│   │   │   ├── __init__.py
│   │   │   ├── repeat_sales.py     # Repeat-sales pair identification
│   │   │   ├── half_pairs.py       # Half-pairs calculation
│   │   │   ├── geographic.py       # Geographic processing
│   │   │   └── pipeline.py         # Main processing pipeline
│   │   ├── indices/
│   │   │   ├── __init__.py
│   │   │   ├── tract_level.py      # Tract-level index construction
│   │   │   ├── city_level.py       # City-level index construction
│   │   │   └── builders.py         # Index builder classes
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── performance.py      # Performance monitoring
│   │       ├── logging.py           # Logging configuration
│   │       └── exceptions.py        # Custom exceptions
├── tests/
│   ├── unit/
│   │   ├── test_bmn_regression.py
│   │   ├── test_supertract.py
│   │   ├── test_aggregation.py
│   │   ├── test_weighting.py
│   │   ├── test_repeat_sales.py
│   │   ├── test_filters.py
│   │   └── test_validators.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   ├── test_index_construction.py
│   │   └── test_end_to_end.py
│   ├── fixtures/
│   │   ├── sample_transactions.py
│   │   ├── sample_geography.py
│   │   └── expected_outputs.py
│   └── conftest.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_bmn_validation.ipynb
│   └── 03_index_comparison.ipynb
├── scripts/
│   ├── run_pipeline.py
│   ├── generate_indices.py
│   └── validate_results.py
├── pyproject.toml
├── Makefile
├── .pre-commit-config.yaml
└── README.md
```

## 3. Data Models (Polars Schemas)

### 3.1 Transaction Schema
```python
import polars as pl

TRANSACTION_SCHEMA = {
    "property_id": pl.String,
    "transaction_date": pl.Date,
    "transaction_price": pl.Float64,
    "census_tract": pl.String,  # 2010 Census tract ID
    "cbsa_code": pl.String,
    "distance_to_cbd": pl.Float64,
}

# Derived fields
REPEAT_SALES_SCHEMA = TRANSACTION_SCHEMA | {
    "prev_transaction_date": pl.Date,
    "prev_transaction_price": pl.Float64,
    "log_price_diff": pl.Float64,  # p_itτ
    "time_diff_years": pl.Float64,
    "cagr": pl.Float64,  # Compound annual growth rate
}
```

### 3.2 Geographic Schema
```python
GEOGRAPHIC_SCHEMA = {
    "tract_id": pl.String,
    "cbsa_code": pl.String,
    "centroid_lat": pl.Float64,
    "centroid_lon": pl.Float64,
    "housing_units": pl.Int64,
    "housing_value": pl.Float64,
    "college_share": pl.Float64,
    "nonwhite_share": pl.Float64,
}

SUPERTRACT_SCHEMA = {
    "supertract_id": pl.String,
    "period": pl.Int32,
    "component_tracts": pl.List(pl.String),
    "total_half_pairs": pl.Int32,
}
```

## 4. Core Algorithm Implementations

### 4.1 BMN Regression Module
```python
# bmn_regression.py
class BMNRegression:
    """Bailey-Muth-Nourse repeat-sales regression implementation"""
    
    def __init__(self, time_periods: list[int]):
        self.time_periods = time_periods
        self.n_periods = len(time_periods)
        
    def create_dummy_matrix(self, repeat_sales_df: pl.DataFrame) -> sparse.csr_matrix:
        """Create sparse dummy variable matrix D_tτ"""
        
    def fit(self, repeat_sales_df: pl.DataFrame) -> np.ndarray:
        """Estimate δ coefficients using OLS"""
        
    def calculate_appreciation(self, delta_t: float, delta_t_1: float) -> float:
        """Calculate p̂_n = δ̂_n,t - δ̂_n,t-1"""
```

### 4.2 Supertract Algorithm
```python
# supertract.py
class SupertractBuilder:
    """Dynamic aggregation of census tracts"""
    
    MIN_HALF_PAIRS = 40
    
    def __init__(self, geographic_df: pl.DataFrame):
        self.geographic_df = geographic_df
        self.distance_matrix = self._calculate_distances()
        
    def build_supertracts(self, 
                         half_pairs_df: pl.DataFrame, 
                         period: int) -> pl.DataFrame:
        """Build supertracts for given period ensuring minimum half-pairs"""
        
    def _merge_nearest_tracts(self, 
                             tract_id: str, 
                             available_tracts: set[str]) -> str:
        """Find and merge with nearest tract by centroid distance"""
```

### 4.3 Weighting Schemes
```python
# weighting.py
class WeightingScheme:
    """Base class for different weighting schemes"""
    
    @abstractmethod
    def calculate_weights(self, supertract_df: pl.DataFrame, period: int) -> pl.DataFrame:
        pass

class SampleWeights(WeightingScheme):
    """w_sample: Share of half-pairs"""
    
class ValueWeights(WeightingScheme):
    """w_value: Share of aggregate housing value (Laspeyres)"""
    
class UnitWeights(WeightingScheme):
    """w_unit: Share of housing units"""
    
# Additional weight types: UPB, College, NonWhite
```

## 5. Processing Pipeline

### 5.1 Main Pipeline Architecture
```python
# pipeline.py
class HPIPipeline:
    """Main processing pipeline for HPI calculation"""
    
    def __init__(self, config: HPIConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.validator = DataValidator(config)
        
    def run(self, start_year: int, end_year: int) -> HPIResults:
        """Execute full pipeline from data loading to index generation"""
        
        # Step 1: Load and validate data
        transactions = self.data_loader.load_transactions()
        transactions = self.validator.validate(transactions)
        
        # Step 2: Identify repeat sales
        repeat_sales = self.identify_repeat_sales(transactions)
        
        # Step 3: Apply filters
        filtered_sales = self.apply_filters(repeat_sales)
        
        # Step 4: Calculate half-pairs
        half_pairs = self.calculate_half_pairs(filtered_sales)
        
        # Step 5: Build supertracts for each period
        supertracts = self.build_all_supertracts(half_pairs)
        
        # Step 6: Run BMN regressions
        regression_results = self.run_bmn_regressions(supertracts, filtered_sales)
        
        # Step 7: Calculate indices
        tract_indices = self.calculate_tract_indices(regression_results)
        city_indices = self.calculate_city_indices(regression_results, self.config.weight_schemes)
        
        return HPIResults(tract_indices, city_indices)
```

### 5.2 Polars-Optimized Operations
```python
def identify_repeat_sales(transactions: pl.DataFrame) -> pl.DataFrame:
    """Identify repeat sales using Polars window functions"""
    return (
        transactions
        .sort(["property_id", "transaction_date"])
        .with_columns([
            pl.col("transaction_date").shift(1).over("property_id").alias("prev_date"),
            pl.col("transaction_price").shift(1).over("property_id").alias("prev_price"),
        ])
        .filter(pl.col("prev_date").is_not_null())
        .with_columns([
            (pl.col("transaction_price").log() - pl.col("prev_price").log()).alias("log_price_diff"),
            ((pl.col("transaction_date") - pl.col("prev_date")).dt.days() / 365.25).alias("years_diff"),
        ])
    )
```

## 6. Testing Strategy

### 6.1 Unit Test Coverage Goals
- **Core Model Code**: ≥80% coverage
- **Data Processing**: ≥75% coverage
- **Utilities**: ≥70% coverage

### 6.2 Test Categories

#### Unit Tests
```python
# test_bmn_regression.py
class TestBMNRegression:
    def test_dummy_matrix_creation(self, sample_repeat_sales):
        """Test sparse dummy matrix generation"""
        
    def test_regression_coefficients(self, known_input_output):
        """Test regression produces expected coefficients"""
        
    def test_edge_cases(self):
        """Test handling of edge cases (single observation, etc.)"""
        
    @hypothesis.given(repeat_sales_strategy())
    def test_regression_properties(self, repeat_sales_data):
        """Property-based testing for regression stability"""
```

#### Integration Tests
```python
# test_pipeline.py
class TestPipeline:
    def test_full_pipeline_small_dataset(self, small_dataset):
        """Test complete pipeline with manageable dataset"""
        
    def test_pipeline_resumability(self, checkpoint_data):
        """Test pipeline can resume from checkpoints"""
        
    def test_output_consistency(self, reference_outputs):
        """Verify outputs match expected format and values"""
```

### 6.3 Test Data Generation
```python
# fixtures/sample_transactions.py
@pytest.fixture
def synthetic_transactions():
    """Generate synthetic transaction data with known properties"""
    return pl.DataFrame({
        "property_id": [f"P{i:06d}" for i in range(10000)],
        "transaction_date": generate_random_dates(10000, "2010-01-01", "2021-12-31"),
        "transaction_price": generate_lognormal_prices(10000, mean=250000, std=100000),
        # ... other fields
    })
```

### 6.4 Coverage Configuration
```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "--cov=src/hpi_fhfa --cov-report=html --cov-report=term-missing"
testpaths = ["tests"]

[tool.coverage.run]
source = ["src/hpi_fhfa"]
omit = ["*/tests/*", "*/config/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

## 7. Performance Optimization

### 7.1 Polars Optimizations
- Use lazy evaluation for complex queries
- Leverage columnar operations
- Minimize data copying with in-place operations
- Use appropriate data types (categoricals for tract IDs)

### 7.2 Computational Optimizations
```python
# Parallel BMN regression for multiple supertracts
def parallel_bmn_regression(supertract_groups: list[pl.DataFrame], n_jobs: int = -1):
    """Run BMN regressions in parallel using joblib"""
    from joblib import Parallel, delayed
    
    return Parallel(n_jobs=n_jobs)(
        delayed(run_single_bmn)(group) for group in supertract_groups
    )
```

### 7.3 Memory Management
- Process data in chunks for large datasets
- Use sparse matrices for dummy variables
- Implement data checkpointing for long-running processes

## 8. Development Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up project structure and dependencies
- [ ] Implement data loading and validation
- [ ] Create basic Polars schemas
- [ ] Set up testing framework
- [ ] Implement transaction filtering

### Phase 2: Core Algorithms (Weeks 3-5)
- [ ] Implement BMN regression with sparse matrices
- [ ] Develop supertract algorithm
- [ ] Create weighting scheme implementations
- [ ] Build repeat-sales identification logic
- [ ] Achieve 50% test coverage

### Phase 3: Pipeline Integration (Weeks 6-7)
- [ ] Build main processing pipeline
- [ ] Implement tract-level index calculation
- [ ] Develop city-level aggregation
- [ ] Add checkpointing and resumability
- [ ] Achieve 70% test coverage

### Phase 4: Optimization & Testing (Weeks 8-9)
- [ ] Profile and optimize performance bottlenecks
- [ ] Implement parallel processing
- [ ] Add comprehensive integration tests
- [ ] Achieve 80%+ test coverage for core models
- [ ] Stress test with large datasets

### Phase 5: Validation & Documentation (Week 10)
- [ ] Validate against reference implementations
- [ ] Complete API documentation
- [ ] Create usage examples and notebooks
- [ ] Performance benchmarking report
- [ ] Final testing and bug fixes

## 9. Key Libraries and Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "polars>=0.20.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",  # For compatibility with existing tools
    "pyarrow>=14.0.0",  # For efficient I/O
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "hypothesis>=6.90.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "joblib>=1.3.0",
    "tqdm>=4.66.0",
    "structlog>=23.2.0",
    "click>=8.1.0",  # For CLI
]

[project.optional-dependencies]
distributed = [
    "dask[complete]>=2023.10.0",
    "ray>=2.8.0",
]
viz = [
    "matplotlib>=3.7.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
]
```

## 10. Success Metrics

### 10.1 Functional Metrics
- Process 63.3M repeat-sales pairs successfully
- Generate indices for all 63,122 census tracts
- Support all 581 CBSAs
- Produce balanced panel 1989-2021

### 10.2 Performance Metrics
- Process 1 year of data in < 10 minutes
- Full historical run in < 8 hours
- Memory usage < 32GB for standard run
- Support distributed processing for larger datasets

### 10.3 Quality Metrics
- Test coverage ≥ 80% for core model code
- All regression tests passing
- Index values within 0.1% of reference implementation
- No data quality violations in output

## 11. Risk Mitigation

### 11.1 Technical Risks
- **Memory constraints**: Implement streaming/chunked processing
- **Computation time**: Add distributed computing support
- **Numerical stability**: Use robust linear algebra libraries
- **Data quality**: Comprehensive validation and error handling

### 11.2 Implementation Risks
- **Scope creep**: Strict adherence to PRD requirements
- **Testing complexity**: Incremental test development
- **Performance regression**: Continuous benchmarking
- **Integration issues**: Well-defined interfaces and contracts

## 12. Deliverables

1. **Production-ready Python package** with Polars-based implementation
2. **Comprehensive test suite** with 80%+ coverage for core models
3. **Performance benchmarks** and optimization report
4. **API documentation** and usage examples
5. **Validation report** comparing to reference implementations
6. **Deployment guide** for different environments
7. **Jupyter notebooks** for exploration and validation

## 13. Conclusion

This implementation plan provides a robust framework for building the FHFA HPI system using Python and Polars. The emphasis on testing, performance, and maintainability ensures the system will be reliable and scalable for production use. The phased approach allows for incremental development and validation while maintaining focus on the core requirements.