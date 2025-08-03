#!/usr/bin/env python3
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
    print(f"\nCompleted in {results.metadata['processing_time']:.1f} seconds")
    print(f"Processed {results.metadata['n_transactions']:,} transactions")
    print(f"Generated {len(results.tract_indices):,} tract indices")
    
    for scheme, city_df in results.city_indices.items():
        print(f"Generated {len(city_df):,} {scheme} city indices")
    
    # Show sample results
    if not results.tract_indices.is_empty():
        print("\nSample tract indices:")
        sample = results.tract_indices.head(5)
        for row in sample.iter_rows(named=True):
            print(f"  Tract {row['tract_id']}, {row['year']}: {row['index_value']:.2f}")

if __name__ == "__main__":
    main()
