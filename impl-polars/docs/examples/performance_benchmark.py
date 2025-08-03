#!/usr/bin/env python3
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
        print(f"\nBenchmarking: {name}")
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
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(benchmark.get_summary_report())

if __name__ == "__main__":
    main()
