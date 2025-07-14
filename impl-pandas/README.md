# Repeat-Sales Aggregation Index (RSAI) Model

This is a Python implementation of the Repeat-Sales Aggregation Index (RSAI) methodology, as described in the FHFA Working Paper 21-01: "A Flexible Method of House Price Index Construction using Repeat-Sales Aggregates".

## Overview

The RSAI model produces robust, city-level house price indices by:
1. Estimating granular price changes in small, localized submarkets (Census tracts)
2. Dynamically creating "supertracts" to ensure sufficient observations
3. Aggregating submarket indices using various weighting schemes

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd impl-pandas

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

```bash
python -m rsai.src.main \
    path/to/transactions.csv \
    path/to/geographic.csv \
    path/to/output.csv \
    --weighting-file path/to/weights.csv \
    --start-year 1989 \
    --end-year 2021 \
    --weighting-schemes sample value unit
```

### Python API

```python
from rsai.src.main import RSAIPipeline

# Initialize pipeline
pipeline = RSAIPipeline(
    min_half_pairs=40,
    base_index_value=100.0,
    base_year=1989
)

# Run pipeline
index_df = pipeline.run_pipeline(
    transaction_file='data/transactions.csv',
    geographic_file='data/geographic.csv',
    output_file='output/indices.csv',
    start_year=1989,
    end_year=2021,
    weighting_file='data/weights.csv',
    weighting_schemes=['sample', 'value', 'unit']
)
```

## Data Requirements

### Transaction Data (Required)
CSV file with columns:
- `property_id`: Unique property identifier
- `transaction_date`: Date of sale
- `transaction_price`: Sale price in USD
- `census_tract_2010`: Census tract ID
- `cbsa_id`: Core-Based Statistical Area ID

### Geographic Data (Required)
CSV file with columns:
- `census_tract_2010`: Census tract ID
- `centroid_lat`: Latitude of tract centroid
- `centroid_lon`: Longitude of tract centroid
- `cbsa_id`: CBSA ID

### Weighting Data (Optional)
CSV file with columns:
- `census_tract_2010`: Census tract ID
- `year`: Year
- `total_housing_units`: Count of single-family units
- `total_housing_value`: Aggregate housing value
- `total_upb`: Aggregate unpaid principal balance
- `college_population`: College-educated population
- `non_white_population`: Non-white population

## Weighting Schemes

The model supports six weighting schemes:

1. **Sample**: Based on half-pairs count
2. **Value**: Laspeyres index using housing values
3. **Unit**: Based on housing unit counts
4. **UPB**: Using unpaid principal balance
5. **College**: College-educated population share
6. **Non-White**: Non-white population share

## Output

The model generates:
- House price indices for each CBSA and weighting scheme
- Summary statistics
- Optional wide-format CSV with all weighting schemes

## Project Structure

```
rsai/
├── src/
│   ├── data/          # Data ingestion and validation
│   ├── geography/     # Supertract generation
│   ├── index/         # BMN regression and aggregation
│   ├── output/        # Index chaining and export
│   └── main.py        # Main pipeline
├── tests/             # Unit tests
└── data/              # Sample data
```

## References

Bailey, M. J., Muth, R. F., & Nourse, H. O. (1963). A regression method for real estate price index construction. Journal of the American Statistical Association, 58(304), 933-942.