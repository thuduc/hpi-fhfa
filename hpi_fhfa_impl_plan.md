# Implementation Plan for Repeat-Sales Aggregation Index (RSAI) Model

Based on the product requirements document, here's a detailed implementation plan for the Repeat-Sales Aggregation Index (RSAI) Model:

## Implementation Plan

### 1. **Project Structure**
```
rsai/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py          # Data models/schemas
│   │   ├── ingestion.py       # Data loading and filtering
│   │   └── validation.py      # Data validation utilities
│   ├── geography/
│   │   ├── __init__.py
│   │   ├── supertract.py      # Supertract generation algorithm
│   │   └── distance.py        # Geographic distance calculations
│   ├── index/
│   │   ├── __init__.py
│   │   ├── bmn_regression.py  # BMN regression implementation
│   │   ├── aggregation.py     # City-level aggregation
│   │   └── weights.py         # Weighting scheme calculations
│   ├── output/
│   │   ├── __init__.py
│   │   └── export.py          # Output generation and chaining
│   └── main.py                # Main entry point
├── tests/
├── data/                      # Sample data directory
├── requirements.txt
├── setup.py
└── README.md
```

### 2. **Technical Implementation Details**

**Phase 1: Data Infrastructure**
- Define Pydantic/dataclass models for transactions, geographic data, and weights
- Implement data validation and filtering logic
- Create repeat-sales pair identification algorithm
- Build price relatives calculation

**Phase 2: Geographic Processing**
- Implement centroid distance calculations using haversine formula
- Create dynamic supertract generation algorithm with 40 half-pair threshold
- Build nearest neighbor merging logic
- Ensure year-specific geography handling

**Phase 3: Statistical Analysis**
- Implement BMN regression using statsmodels OLS
- Extract annual appreciation rates from regression coefficients
- Handle edge cases for small sample sizes

**Phase 4: Aggregation Engine**
- Implement all 6 weighting schemes (Sample, Value, Unit, UPB, College, Non-White)
- Create flexible weighting framework for custom schemes
- Build city-level aggregation logic

**Phase 5: Output & Integration**
- Implement index chaining algorithm
- Create CSV/Parquet export functionality
- Build end-to-end pipeline orchestration

### 3. **Key Implementation Considerations**

- **Performance**: Use vectorized operations with pandas/numpy for large datasets
- **Memory**: Implement chunking for processing years of transaction data
- **Extensibility**: Design weight calculation as pluggable interface
- **Validation**: Comprehensive data quality checks at each step
- **Testing**: Unit tests for each component, integration tests with synthetic data

The implementation follows the paper's methodology exactly, ensuring robust city-level house price indices through granular submarket analysis.

## Todo List

1. Set up project structure and dependencies
2. Create data models and schema definitions
3. Implement data ingestion and preparation module (Feature 1)
4. Implement supertract generation algorithm (Feature 2)
5. Implement BMN regression and submarket index calculation (Feature 3)
6. Implement city-level aggregation with weighting schemes (Feature 4)
7. Implement output generation and index chaining (Feature 5)
8. Create unit tests for all modules
9. Create integration tests with sample data
10. Write documentation and usage examples