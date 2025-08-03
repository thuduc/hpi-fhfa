#!/usr/bin/env python3
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
    print("\n" + validator.get_summary_report())
    
    # Check for specific issues
    failed_tests = [r for r in validation_results if not r.passed]
    if failed_tests:
        print("\nFailed validation tests:")
        for test in failed_tests:
            print(f"  - {test.test_name}: {test.message}")
            if test.details:
                for key, value in test.details.items():
                    print(f"    {key}: {value}")
    else:
        print("\n✓ All validation tests passed!")

if __name__ == "__main__":
    main()
