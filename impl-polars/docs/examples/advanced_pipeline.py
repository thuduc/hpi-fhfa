#!/usr/bin/env python3
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
    print("\nRunning advanced pipeline...")
    pipeline = HPIPipeline(config)
    results = pipeline.run()
    
    # Analyze results
    print("\nResults Analysis:")
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
    print("\nTract-Level Analysis:")
    
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
    print("\nWeight Scheme Comparison:")
    
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
    print("\nAdvanced Validation:")
    
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
        print("\n  Critical Issues:")
        for failure in critical_failures:
            print(f"    ✗ {failure.test_name}: {failure.message}")
    
    if warnings:
        print("\n  Warnings:")
        for warning in warnings[:3]:  # Show first 3
            print(f"    ⚠ {warning.test_name}: {warning.message}")

if __name__ == "__main__":
    main()
