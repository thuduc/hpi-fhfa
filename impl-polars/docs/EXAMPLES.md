# HPI-FHFA Usage Examples

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
    print(f"\n{scheme.title()} weighted city indices:")
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
    print(f"\n{len(failed_tests)} validation tests failed:")
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
print("\n" + benchmark.get_summary_report())
```

## Integration Examples

### Batch Processing Multiple Datasets

```python
from pathlib import Path
import pandas as pd

def process_multiple_datasets(data_directory: Path):
    """Process multiple transaction datasets."""
    
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
    """Web API endpoint for HPI calculation."""
    
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
    """Download data file from URL to local path."""
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
    """Run pipeline with comprehensive monitoring and alerting."""
    
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
    """Send alert via your preferred alerting system."""
    # Implement email, Slack, PagerDuty, etc. integration
    logging.critical(f"ALERT: {message}")

# Usage
config = HPIConfig(...)  # Your configuration
results, alerts = run_monitored_pipeline(config)
```

These examples demonstrate the flexibility and power of the HPI-FHFA library for various use cases, from simple academic research to production-scale financial applications.
