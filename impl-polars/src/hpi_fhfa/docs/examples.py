"""Usage examples and demonstration code for HPI-FHFA."""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import date, timedelta
import tempfile
from typing import Tuple

from ..processing.pipeline import HPIPipeline
from ..config.settings import HPIConfig
from ..validation import HPIValidator, benchmark_pipeline


def create_sample_data(
    n_transactions: int = 5000,
    n_properties: int = 1000,
    n_tracts: int = 20,
    n_cbsas: int = 2,
    start_date: date = date(2020, 1, 1),
    end_date: date = date(2023, 12, 31)
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Create sample transaction and geographic data for demonstration.
    
    Args:
        n_transactions: Number of transactions to generate
        n_properties: Number of unique properties
        n_tracts: Number of census tracts
        n_cbsas: Number of CBSAs
        start_date: Start date for transactions
        end_date: End date for transactions
        
    Returns:
        Tuple of (transactions_df, geographic_df)
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate property IDs with repeat sales pattern
    property_weights = np.random.exponential(2, n_properties)
    property_weights = property_weights / property_weights.sum()
    property_ids = np.random.choice(
        [f"P{i:06d}" for i in range(n_properties)],
        size=n_transactions,
        p=property_weights
    )
    
    # Generate dates
    date_range = (end_date - start_date).days
    dates = [
        start_date + timedelta(days=int(d)) 
        for d in np.random.randint(0, date_range, n_transactions)
    ]
    
    # Generate prices with appreciation trend
    base_price = 300000
    years_from_start = [(d - start_date).days / 365.25 for d in dates]
    annual_appreciation = 0.04  # 4% annual appreciation
    price_trend = [base_price * (1 + annual_appreciation) ** year for year in years_from_start]
    price_noise = np.random.lognormal(0, 0.2, n_transactions)  # 20% noise
    prices = [max(50000, trend * noise) for trend, noise in zip(price_trend, price_noise)]
    
    # Generate geographic assignments
    tract_ids = [f"06037{i:06d}" for i in range(n_tracts)]
    cbsa_codes = [f"CBSA{i+1:02d}" for i in range(n_cbsas)]
    
    transaction_tracts = np.random.choice(tract_ids, n_transactions)
    transaction_cbsas = np.random.choice(cbsa_codes, n_transactions)
    
    # Create transaction DataFrame
    transactions = pl.DataFrame({
        "property_id": property_ids,
        "transaction_date": dates,
        "transaction_price": prices,
        "census_tract": transaction_tracts,
        "cbsa_code": transaction_cbsas,
        "distance_to_cbd": np.random.uniform(1, 40, n_transactions)
    })
    
    # Create geographic DataFrame
    geographic = pl.DataFrame({
        "tract_id": tract_ids,
        "cbsa_code": np.random.choice(cbsa_codes, n_tracts),
        "centroid_lat": np.random.uniform(33.0, 35.0, n_tracts),
        "centroid_lon": np.random.uniform(-119.0, -117.0, n_tracts),
        "housing_units": np.random.randint(500, 3000, n_tracts),
        "housing_value": np.random.uniform(500_000_000, 3_000_000_000, n_tracts),
        "college_share": np.random.beta(3, 2, n_tracts),  # Skewed toward higher education
        "nonwhite_share": np.random.beta(2, 3, n_tracts)  # Diverse distribution
    })
    
    return transactions, geographic


def basic_example():
    """Basic usage example of the HPI pipeline."""
    print("=" * 50)
    print("HPI-FHFA Basic Usage Example")
    print("=" * 50)
    
    # Create sample data
    print("Generating sample data...")
    transactions, geographic = create_sample_data(
        n_transactions=2000,
        n_properties=500,
        n_tracts=10,
        n_cbsas=2
    )
    
    print(f"Generated {len(transactions):,} transactions")
    print(f"Generated {len(geographic)} census tracts")
    print(f"Date range: {transactions['transaction_date'].min()} to {transactions['transaction_date'].max()}")
    print(f"Price range: ${transactions['transaction_price'].min():,.0f} to ${transactions['transaction_price'].max():,.0f}")
    
    # Save to temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        txn_path = temp_path / "transactions.parquet"
        geo_path = temp_path / "geographic.parquet"
        output_path = temp_path / "output"
        
        transactions.write_parquet(txn_path)
        geographic.write_parquet(geo_path)
        
        print(f"\\nSaved data to: {temp_path}")
        
        # Configure pipeline
        config = HPIConfig(
            transaction_data_path=txn_path,
            geographic_data_path=geo_path,
            output_path=output_path,
            start_year=2021,
            end_year=2023,
            weight_schemes=["sample", "value"],
            n_jobs=1,
            validate_data=True,
            use_lazy_evaluation=False
        )
        
        print("\\nPipeline configuration:")
        print(f"  Years: {config.start_year}-{config.end_year}")
        print(f"  Weight schemes: {config.weight_schemes}")
        print(f"  Validation: {config.validate_data}")
        
        # Run pipeline
        print("\\nRunning HPI pipeline...")
        pipeline = HPIPipeline(config)
        results = pipeline.run()
        
        # Display results
        print("\\nResults:")
        print(f"  Processing time: {results.metadata['processing_time']:.2f} seconds")
        print(f"  Transactions processed: {results.metadata['n_transactions']:,}")
        print(f"  Repeat sales found: {results.metadata['n_repeat_sales']:,}")
        print(f"  Filtered sales: {results.metadata['n_filtered_sales']:,}")
        
        print(f"\\nTract-level indices: {len(results.tract_indices)} records")
        if not results.tract_indices.is_empty():
            print("  Sample tract indices:")
            sample = results.tract_indices.head(5)
            for row in sample.iter_rows(named=True):
                print(f"    Tract {row['tract_id']}, {row['year']}: {row['index_value']:.2f}")
        
        print(f"\\nCity-level indices:")
        for scheme, city_df in results.city_indices.items():
            print(f"  {scheme}: {len(city_df)} records")
            if not city_df.is_empty():
                avg_appreciation = city_df.filter(
                    pl.col("appreciation_rate").is_not_null()
                )["appreciation_rate"].mean()
                print(f"    Average appreciation: {avg_appreciation:.2f}%")


def validation_example():
    """Example of validation and quality checking."""
    print("\\n" + "=" * 50)
    print("HPI-FHFA Validation Example")
    print("=" * 50)
    
    # Create sample data
    transactions, geographic = create_sample_data(n_transactions=1000, n_tracts=5)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save data
        transactions.write_parquet(temp_path / "transactions.parquet")
        geographic.write_parquet(temp_path / "geographic.parquet")
        
        # Configure and run pipeline
        config = HPIConfig(
            transaction_data_path=temp_path / "transactions.parquet",
            geographic_data_path=temp_path / "geographic.parquet",
            output_path=temp_path / "output",
            start_year=2021,
            end_year=2023,
            weight_schemes=["sample"],
            validate_data=True,
            use_lazy_evaluation=False
        )
        
        pipeline = HPIPipeline(config)
        results = pipeline.run()
        
        # Validate results
        print("\\nValidating results...")
        validator = HPIValidator(tolerance=0.001)  # 0.1% tolerance
        validation_results = validator.validate_all(
            results.tract_indices,
            results.city_indices
        )
        
        # Display validation report
        print("\\nValidation Report:")
        print(validator.get_summary_report())
        
        # Show specific validation details
        print("\\nDetailed Validation Results:")
        passed_tests = [r for r in validation_results if r.passed]
        failed_tests = [r for r in validation_results if not r.passed]
        
        print(f"✓ Passed: {len(passed_tests)} tests")
        for test in passed_tests[:3]:  # Show first 3
            print(f"  • {test.test_name}: {test.message}")
        
        if failed_tests:
            print(f"✗ Failed: {len(failed_tests)} tests")
            for test in failed_tests:
                print(f"  • {test.test_name}: {test.message}")
        else:
            print("✓ All validation tests passed!")


def performance_example():
    """Example of performance benchmarking."""
    print("\\n" + "=" * 50)
    print("HPI-FHFA Performance Benchmarking Example")
    print("=" * 50)
    
    # Test different configurations
    test_configs = [
        ("Small Dataset", {"n_transactions": 1000, "n_jobs": 1}),
        ("Medium Dataset", {"n_transactions": 3000, "n_jobs": 1}),
        ("Parallel Processing", {"n_transactions": 3000, "n_jobs": 2}),
    ]
    
    for config_name, params in test_configs:
        print(f"\\nBenchmarking: {config_name}")
        print("-" * 30)
        
        # Generate test data
        transactions, geographic = create_sample_data(
            n_transactions=params["n_transactions"],
            n_properties=params["n_transactions"] // 3,
            n_tracts=10
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save data
            transactions.write_parquet(temp_path / "transactions.parquet")
            geographic.write_parquet(temp_path / "geographic.parquet")
            
            # Configure pipeline
            config = HPIConfig(
                transaction_data_path=temp_path / "transactions.parquet",
                geographic_data_path=temp_path / "geographic.parquet",
                output_path=temp_path / "output",
                start_year=2021,
                end_year=2023,
                weight_schemes=["sample"],
                n_jobs=params["n_jobs"],
                validate_data=False,  # Skip validation for speed
                use_lazy_evaluation=False
            )
            
            # Run benchmark
            try:
                result = benchmark_pipeline(config, name=config_name)
                
                print(f"Duration: {result.duration_seconds:.2f} seconds")
                print(f"Peak Memory: {result.peak_memory_mb:.1f} MB")
                print(f"Throughput: {result.throughput_transactions_per_sec:.0f} transactions/sec")
                print(f"Transactions: {result.n_transactions:,}")
                print(f"Tracts: {result.n_tracts}")
                
            except Exception as e:
                print(f"Benchmark failed: {e}")


def advanced_usage_example():
    """Example of advanced usage patterns."""
    print("\\n" + "=" * 50)
    print("HPI-FHFA Advanced Usage Example")
    print("=" * 50)
    
    # Create larger, more realistic dataset
    transactions, geographic = create_sample_data(
        n_transactions=10000,
        n_properties=3000,
        n_tracts=25,
        n_cbsas=3,
        start_date=date(2018, 1, 1),
        end_date=date(2023, 12, 31)
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save data
        transactions.write_parquet(temp_path / "transactions.parquet")
        geographic.write_parquet(temp_path / "geographic.parquet")
        
        # Advanced configuration
        config = HPIConfig(
            transaction_data_path=temp_path / "transactions.parquet",
            geographic_data_path=temp_path / "geographic.parquet",
            output_path=temp_path / "output",
            start_year=2019,
            end_year=2023,
            weight_schemes=["sample", "value", "unit", "college"],  # Multiple schemes
            n_jobs=2,                    # Parallel processing
            chunk_size=5000,             # Moderate chunk size
            use_lazy_evaluation=True,    # Memory optimization
            checkpoint_frequency=2,      # Frequent checkpoints
            validate_data=True,
            strict_validation=False
        )
        
        print("Advanced configuration:")
        print(f"  Data size: {len(transactions):,} transactions, {len(geographic)} tracts")
        print(f"  Years: {config.start_year}-{config.end_year}")
        print(f"  Weight schemes: {len(config.weight_schemes)}")
        print(f"  Parallel jobs: {config.n_jobs}")
        print(f"  Lazy evaluation: {config.use_lazy_evaluation}")
        print(f"  Checkpointing: Every {config.checkpoint_frequency} periods")
        
        # Run pipeline with monitoring
        print("\\nRunning advanced pipeline...")
        pipeline = HPIPipeline(config)
        results = pipeline.run()
        
        # Analyze results in detail
        print("\\nDetailed Results Analysis:")
        print(f"Processing time: {results.metadata['processing_time']:.2f}s")
        print(f"Throughput: {results.metadata['n_transactions']/results.metadata['processing_time']:.0f} txn/s")
        
        # Tract-level analysis
        tract_df = results.tract_indices
        if not tract_df.is_empty():
            # Calculate statistics by year
            yearly_stats = (
                tract_df
                .filter(pl.col("appreciation_rate").is_not_null())
                .group_by("year")
                .agg([
                    pl.col("appreciation_rate").mean().alias("avg_appreciation"),
                    pl.col("appreciation_rate").std().alias("std_appreciation"),
                    pl.col("appreciation_rate").median().alias("median_appreciation"),
                    pl.len().alias("n_tracts")
                ])
                .sort("year")
            )
            
            print("\\nTract-level appreciation by year:")
            for row in yearly_stats.iter_rows(named=True):
                print(f"  {row['year']}: {row['avg_appreciation']:.2f}% ± {row['std_appreciation']:.2f}% "
                      f"(median: {row['median_appreciation']:.2f}%, n={row['n_tracts']})")
        
        # City-level comparison across weight schemes
        print("\\nCity-level indices by weight scheme:")
        for scheme, city_df in results.city_indices.items():
            if not city_df.is_empty():
                # Calculate average index values
                avg_index = city_df["index_value"].mean()
                final_year_appreciation = (
                    city_df
                    .filter(pl.col("year") == config.end_year)
                    .filter(pl.col("appreciation_rate").is_not_null())
                    ["appreciation_rate"].mean()
                )
                
                print(f"  {scheme.title()}: avg index = {avg_index:.2f}, "
                      f"final year appreciation = {final_year_appreciation:.2f}%")
        
        # Check output files
        output_files = list((temp_path / "output").glob("*.parquet"))
        print(f"\\nOutput files generated: {len(output_files)}")
        for file in output_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name}: {size_mb:.2f} MB")
        
        # Validate final results
        validator = HPIValidator(tolerance=0.002)  # 0.2% tolerance for larger dataset
        validation_results = validator.validate_all(
            results.tract_indices,
            results.city_indices
        )
        
        passed = sum(1 for r in validation_results if r.passed)
        total = len(validation_results)
        print(f"\\nFinal validation: {passed}/{total} tests passed ({passed/total*100:.1f}%)")


def create_usage_examples(output_path: Path) -> None:
    """Create comprehensive usage examples and save to files.
    
    Args:
        output_path: Directory to save example files
    """
    output_path.mkdir(exist_ok=True)
    
    # Create example scripts
    examples = [
        ("basic_usage.py", _create_basic_script()),
        ("validation_example.py", _create_validation_script()),
        ("performance_benchmark.py", _create_benchmark_script()),
        ("advanced_pipeline.py", _create_advanced_script()),
        ("batch_processing.py", _create_batch_script())
    ]
    
    for filename, content in examples:
        (output_path / filename).write_text(content)
    
    print(f"Created {len(examples)} example scripts in {output_path}")


def _create_basic_script() -> str:
    """Create basic usage script."""
    return '''#!/usr/bin/env python3
"""
Basic HPI-FHFA usage example.

This script demonstrates the most common use case:
1. Load transaction and geographic data
2. Configure the pipeline
3. Run HPI calculation
4. Display results
"""

from hpi_fhfa.processing.pipeline import HPIPipeline
from hpi_fhfa.config.settings import HPIConfig
from pathlib import Path

def main():
    # Configure the pipeline
    config = HPIConfig(
        transaction_data_path=Path("data/transactions.parquet"),
        geographic_data_path=Path("data/geographic.parquet"),
        output_path=Path("output/"),
        start_year=2020,
        end_year=2023,
        weight_schemes=["sample", "value"],  # Two common schemes
        validate_data=True
    )
    
    # Run the pipeline
    print("Starting HPI calculation...")
    pipeline = HPIPipeline(config)
    results = pipeline.run()
    
    # Display results
    print(f"\\nCompleted in {results.metadata['processing_time']:.1f} seconds")
    print(f"Processed {results.metadata['n_transactions']:,} transactions")
    print(f"Generated {len(results.tract_indices):,} tract indices")
    
    for scheme, city_df in results.city_indices.items():
        print(f"Generated {len(city_df):,} {scheme} city indices")
    
    # Show sample results
    if not results.tract_indices.is_empty():
        print("\\nSample tract indices:")
        sample = results.tract_indices.head(5)
        for row in sample.iter_rows(named=True):
            print(f"  Tract {row['tract_id']}, {row['year']}: {row['index_value']:.2f}")

if __name__ == "__main__":
    main()
'''


def _create_validation_script() -> str:
    """Create validation example script."""
    return '''#!/usr/bin/env python3
"""
HPI validation example.

This script demonstrates how to validate HPI calculation results
and compare against reference implementations.
"""

from hpi_fhfa.processing.pipeline import HPIPipeline
from hpi_fhfa.config.settings import HPIConfig
from hpi_fhfa.validation import HPIValidator
from pathlib import Path

def main():
    # Configure pipeline
    config = HPIConfig(
        transaction_data_path=Path("data/transactions.parquet"),
        geographic_data_path=Path("data/geographic.parquet"),
        output_path=Path("output/"),
        start_year=2020,
        end_year=2023,
        weight_schemes=["sample"]
    )
    
    # Run pipeline
    pipeline = HPIPipeline(config)
    results = pipeline.run()
    
    # Validate results
    print("Validating HPI results...")
    validator = HPIValidator(tolerance=0.001)  # 0.1% tolerance
    
    validation_results = validator.validate_all(
        results.tract_indices,
        results.city_indices
    )
    
    # Generate and display report
    print("\\n" + validator.get_summary_report())
    
    # Check for specific issues
    failed_tests = [r for r in validation_results if not r.passed]
    if failed_tests:
        print("\\nFailed validation tests:")
        for test in failed_tests:
            print(f"  - {test.test_name}: {test.message}")
            if test.details:
                for key, value in test.details.items():
                    print(f"    {key}: {value}")
    else:
        print("\\n✓ All validation tests passed!")

if __name__ == "__main__":
    main()
'''


def _create_benchmark_script() -> str:
    """Create benchmark example script."""
    return '''#!/usr/bin/env python3
"""
HPI performance benchmarking example.

This script demonstrates how to benchmark different pipeline configurations
to optimize performance for your specific use case.
"""

from hpi_fhfa.config.settings import HPIConfig
from hpi_fhfa.validation import PerformanceBenchmark
from pathlib import Path

def main():
    # Define configurations to test
    configs = {
        "Sequential": HPIConfig(
            transaction_data_path=Path("data/transactions.parquet"),
            geographic_data_path=Path("data/geographic.parquet"),
            output_path=Path("output/sequential/"),
            start_year=2020,
            end_year=2022,
            weight_schemes=["sample"],
            n_jobs=1,
            validate_data=False
        ),
        "Parallel": HPIConfig(
            transaction_data_path=Path("data/transactions.parquet"),
            geographic_data_path=Path("data/geographic.parquet"),
            output_path=Path("output/parallel/"),
            start_year=2020,
            end_year=2022,
            weight_schemes=["sample"],
            n_jobs=4,
            validate_data=False
        ),
        "Lazy Evaluation": HPIConfig(
            transaction_data_path=Path("data/transactions.parquet"),
            geographic_data_path=Path("data/geographic.parquet"),
            output_path=Path("output/lazy/"),
            start_year=2020,
            end_year=2022,
            weight_schemes=["sample"],
            n_jobs=2,
            use_lazy_evaluation=True,
            validate_data=False
        )
    }
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    
    print("Running performance benchmarks...")
    print("=" * 50)
    
    for name, config in configs.items():
        print(f"\\nBenchmarking: {name}")
        print("-" * 30)
        
        try:
            result = benchmark.benchmark_pipeline(config, name=name.lower())
            
            print(f"Duration: {result.duration_seconds:.2f}s")
            print(f"Peak Memory: {result.peak_memory_mb:.1f} MB")
            print(f"Throughput: {result.throughput_transactions_per_sec:.0f} txn/s")
            print(f"Efficiency: {result.throughput_transactions_per_sec/result.peak_memory_mb:.2f} txn/s/MB")
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
    
    # Generate summary report
    print("\\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(benchmark.get_summary_report())

if __name__ == "__main__":
    main()
'''


def _create_advanced_script() -> str:
    """Create advanced usage script."""
    return '''#!/usr/bin/env python3
"""
Advanced HPI pipeline usage example.

This script demonstrates advanced features:
- Multiple weight schemes
- Checkpointing and resumption
- Custom validation
- Result analysis
"""

import polars as pl
from hpi_fhfa.processing.pipeline import HPIPipeline
from hpi_fhfa.config.settings import HPIConfig
from hpi_fhfa.validation import HPIValidator
from pathlib import Path

def main():
    # Advanced configuration
    config = HPIConfig(
        transaction_data_path=Path("data/large_transactions.parquet"),
        geographic_data_path=Path("data/geographic.parquet"),
        output_path=Path("output/advanced/"),
        start_year=2015,
        end_year=2023,
        weight_schemes=["sample", "value", "unit", "college", "nonwhite"],
        n_jobs=6,
        chunk_size=100000,
        use_lazy_evaluation=True,
        checkpoint_frequency=3,
        validate_data=True,
        strict_validation=False
    )
    
    print("Advanced HPI Pipeline Configuration:")
    print(f"  Years: {config.start_year}-{config.end_year}")
    print(f"  Weight schemes: {len(config.weight_schemes)}")
    print(f"  Parallel jobs: {config.n_jobs}")
    print(f"  Chunk size: {config.chunk_size:,}")
    print(f"  Checkpointing: Every {config.checkpoint_frequency} periods")
    
    # Run pipeline
    print("\\nRunning advanced pipeline...")
    pipeline = HPIPipeline(config)
    results = pipeline.run()
    
    # Analyze results
    print("\\nResults Analysis:")
    print(f"Processing time: {results.metadata['processing_time']:.1f}s")
    print(f"Throughput: {results.metadata['n_transactions']/results.metadata['processing_time']:.0f} txn/s")
    
    # Tract-level analysis
    analyze_tract_results(results.tract_indices)
    
    # City-level comparison
    compare_weight_schemes(results.city_indices)
    
    # Advanced validation
    perform_advanced_validation(results)

def analyze_tract_results(tract_df: pl.DataFrame):
    """Analyze tract-level results."""
    print("\\nTract-Level Analysis:")
    
    if tract_df.is_empty():
        print("  No tract indices generated")
        return
    
    # Summary statistics
    n_tracts = tract_df["tract_id"].n_unique()
    n_years = tract_df["year"].n_unique()
    
    print(f"  Tracts covered: {n_tracts:,}")
    print(f"  Years covered: {n_years}")
    
    # Appreciation statistics
    appreciation_stats = tract_df.filter(
        pl.col("appreciation_rate").is_not_null()
    )["appreciation_rate"]
    
    if len(appreciation_stats) > 0:
        print(f"  Average appreciation: {appreciation_stats.mean():.2f}%")
        print(f"  Median appreciation: {appreciation_stats.median():.2f}%")
        print(f"  Std deviation: {appreciation_stats.std():.2f}%")
        print(f"  Range: {appreciation_stats.min():.2f}% to {appreciation_stats.max():.2f}%")

def compare_weight_schemes(city_indices: dict):
    """Compare results across different weight schemes."""
    print("\\nWeight Scheme Comparison:")
    
    for scheme, city_df in city_indices.items():
        if city_df.is_empty():
            print(f"  {scheme}: No data")
            continue
        
        # Calculate average index and appreciation
        avg_index = city_df["index_value"].mean()
        
        latest_year_data = city_df.filter(
            pl.col("year") == city_df["year"].max()
        )
        
        if not latest_year_data.is_empty():
            latest_appreciation = latest_year_data["appreciation_rate"].mean()
            print(f"  {scheme}: avg index = {avg_index:.2f}, "
                  f"latest appreciation = {latest_appreciation:.2f}%")

def perform_advanced_validation(results):
    """Perform advanced validation checks."""
    print("\\nAdvanced Validation:")
    
    validator = HPIValidator(tolerance=0.002)  # 0.2% tolerance
    validation_results = validator.validate_all(
        results.tract_indices,
        results.city_indices
    )
    
    # Categorize results
    critical_failures = []
    warnings = []
    
    for result in validation_results:
        if not result.passed:
            if "missing" in result.test_name or "positive" in result.test_name:
                critical_failures.append(result)
            else:
                warnings.append(result)
    
    print(f"  Critical failures: {len(critical_failures)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Passed tests: {len(validation_results) - len(critical_failures) - len(warnings)}")
    
    if critical_failures:
        print("\\n  Critical Issues:")
        for failure in critical_failures:
            print(f"    ✗ {failure.test_name}: {failure.message}")
    
    if warnings:
        print("\\n  Warnings:")
        for warning in warnings[:3]:  # Show first 3
            print(f"    ⚠ {warning.test_name}: {warning.message}")

if __name__ == "__main__":
    main()
'''


def _create_batch_script() -> str:
    """Create batch processing script."""
    return '''#!/usr/bin/env python3
"""
Batch processing example for multiple datasets.

This script demonstrates how to process multiple datasets
in batch with consistent configuration and reporting.
"""

import pandas as pd
from pathlib import Path
from hpi_fhfa.processing.pipeline import HPIPipeline
from hpi_fhfa.config.settings import HPIConfig
from hpi_fhfa.validation import HPIValidator

def process_batch_datasets(data_directory: Path, output_base: Path):
    """Process multiple datasets in batch."""
    
    # Find all transaction files
    transaction_files = list(data_directory.glob("transactions_*.parquet"))
    geographic_file = data_directory / "geographic.parquet"
    
    if not geographic_file.exists():
        raise FileNotFoundError(f"Geographic file not found: {geographic_file}")
    
    print(f"Found {len(transaction_files)} datasets to process")
    
    results_summary = []
    
    for txn_file in transaction_files:
        dataset_name = txn_file.stem
        print(f"\\nProcessing {dataset_name}...")
        
        try:
            # Configure pipeline
            config = HPIConfig(
                transaction_data_path=txn_file,
                geographic_data_path=geographic_file,
                output_path=output_base / dataset_name,
                start_year=2020,
                end_year=2023,
                weight_schemes=["sample", "value"],
                n_jobs=4,
                validate_data=True
            )
            
            # Run pipeline
            pipeline = HPIPipeline(config)
            results = pipeline.run()
            
            # Validate results
            validator = HPIValidator(tolerance=0.001)
            validation_results = validator.validate_all(
                results.tract_indices,
                results.city_indices
            )
            
            # Collect summary
            passed_validations = sum(1 for r in validation_results if r.passed)
            total_validations = len(validation_results)
            
            summary = {
                "dataset": dataset_name,
                "status": "success",
                "n_transactions": results.metadata["n_transactions"],
                "n_tract_indices": len(results.tract_indices),
                "n_city_indices": sum(len(df) for df in results.city_indices.values()),
                "processing_time": results.metadata["processing_time"],
                "validation_pass_rate": passed_validations / total_validations if total_validations > 0 else 0,
                "avg_tract_appreciation": None,
                "avg_city_appreciation": None
            }
            
            # Calculate average appreciations
            if not results.tract_indices.is_empty():
                tract_appreciation = results.tract_indices.filter(
                    pl.col("appreciation_rate").is_not_null()
                )["appreciation_rate"]
                if len(tract_appreciation) > 0:
                    summary["avg_tract_appreciation"] = tract_appreciation.mean()
            
            if results.city_indices:
                city_appreciations = []
                for city_df in results.city_indices.values():
                    if not city_df.is_empty():
                        city_appreciation = city_df.filter(
                            pl.col("appreciation_rate").is_not_null()
                        )["appreciation_rate"]
                        if len(city_appreciation) > 0:
                            city_appreciations.extend(city_appreciation.to_list())
                
                if city_appreciations:
                    summary["avg_city_appreciation"] = sum(city_appreciations) / len(city_appreciations)
            
            results_summary.append(summary)
            
            print(f"  ✓ Success: {results.metadata['n_transactions']:,} transactions, "
                  f"{results.metadata['processing_time']:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results_summary.append({
                "dataset": dataset_name,
                "status": "failed",
                "error": str(e),
                "n_transactions": None,
                "processing_time": None
            })
    
    # Create summary report
    summary_df = pd.DataFrame(results_summary)
    
    # Save summary
    summary_file = output_base / "batch_processing_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary statistics
    print("\\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)
    
    successful = summary_df[summary_df["status"] == "success"]
    failed = summary_df[summary_df["status"] == "failed"]
    
    print(f"Total datasets: {len(summary_df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if len(successful) > 0:
        total_transactions = successful["n_transactions"].sum()
        total_time = successful["processing_time"].sum()
        avg_throughput = total_transactions / total_time if total_time > 0 else 0
        
        print(f"\\nSuccessful Processing:")
        print(f"  Total transactions: {total_transactions:,}")
        print(f"  Total processing time: {total_time:.1f}s")
        print(f"  Average throughput: {avg_throughput:.0f} txn/s")
        
        avg_tract_app = successful["avg_tract_appreciation"].mean()
        avg_city_app = successful["avg_city_appreciation"].mean()
        
        if not pd.isna(avg_tract_app):
            print(f"  Average tract appreciation: {avg_tract_app:.2f}%")
        if not pd.isna(avg_city_app):
            print(f"  Average city appreciation: {avg_city_app:.2f}%")
    
    if len(failed) > 0:
        print(f"\\nFailed Datasets:")
        for _, row in failed.iterrows():
            print(f"  {row['dataset']}: {row['error']}")
    
    print(f"\\nSummary saved to: {summary_file}")
    return summary_df

def main():
    data_dir = Path("data/batch_datasets/")
    output_dir = Path("output/batch_results/")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please create the directory and add transaction files named 'transactions_*.parquet'")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = process_batch_datasets(data_dir, output_dir)

if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    # Run all examples when script is executed directly
    basic_example()
    validation_example()
    performance_example()
    advanced_usage_example()