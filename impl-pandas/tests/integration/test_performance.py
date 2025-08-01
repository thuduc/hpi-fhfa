"""Performance benchmark tests."""

import pytest
import pandas as pd
import numpy as np
import time

from hpi_fhfa.benchmarks import PerformanceBenchmark
from hpi_fhfa.models.repeat_sales import construct_repeat_sales_pairs
from tests.integration.test_end_to_end import TestEndToEndPipeline


class TestPerformance:
    """Test performance benchmarks."""
    
    def setup_method(self):
        """Set up test data generator."""
        self.data_generator = TestEndToEndPipeline()
    
    def test_repeat_sales_performance(self):
        """Benchmark repeat sales construction performance."""
        # Generate different sized datasets
        sizes = [100, 500, 1000]
        
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=3)
        
        for n_props in sizes:
            transactions = self.data_generator.generate_synthetic_transactions(
                n_properties=n_props,
                n_tracts=10
            )
            
            result = benchmark.benchmark_repeat_sales_construction(transactions)
            
            # Basic performance assertions
            assert result.n_records == len(transactions)
            assert result.execution_time > 0
            assert result.throughput > 0
            
            # Log performance
            print(f"\nRepeat sales construction ({n_props} properties):")
            print(f"  - Records: {result.n_records}")
            print(f"  - Time: {result.execution_time:.3f}s")
            print(f"  - Throughput: {result.throughput:.0f} records/s")
            print(f"  - Pairs created: {result.metadata.get('n_pairs_created', 0)}")
    
    def test_bmn_regression_performance(self):
        """Benchmark BMN regression performance."""
        # Generate test data
        transactions = self.data_generator.generate_synthetic_transactions(
            n_properties=500,
            n_tracts=5
        )
        
        # Create repeat sales pairs
        pairs = construct_repeat_sales_pairs(transactions)
        
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=3)
        
        # Test standard implementation
        result_standard = benchmark.benchmark_bmn_regression(pairs, use_numba=False)
        
        assert result_standard.execution_time > 0
        assert result_standard.n_records == len(pairs)
        
        print(f"\nBMN Regression (standard):")
        print(f"  - Pairs: {result_standard.n_records}")
        print(f"  - Time: {result_standard.execution_time:.3f}s")
        print(f"  - Throughput: {result_standard.throughput:.0f} pairs/s")
    
    def test_scaling_behavior(self):
        """Test how performance scales with data size."""
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=2)
        
        # Generate base dataset
        base_transactions = self.data_generator.generate_synthetic_transactions(
            n_properties=200,
            n_tracts=5
        )
        
        # Test scaling
        scale_factors = [0.5, 1.0, 2.0]
        results_df = benchmark.run_scaling_benchmark(
            base_transactions,
            scale_factors=scale_factors
        )
        
        assert len(results_df) == len(scale_factors)
        
        # Performance should scale roughly linearly
        # (allowing for some overhead)
        for i in range(1, len(results_df)):
            scale_ratio = results_df.iloc[i]['scale_factor'] / results_df.iloc[0]['scale_factor']
            time_ratio = results_df.iloc[i]['execution_time'] / results_df.iloc[0]['execution_time']
            
            # Allow up to 50% overhead
            assert time_ratio <= scale_ratio * 1.5
        
        print("\nScaling behavior:")
        print(results_df[['scale_factor', 'n_records', 'execution_time', 'throughput']])
    
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with larger dataset (marked as slow)."""
        # Generate large dataset
        transactions = self.data_generator.generate_synthetic_transactions(
            n_properties=5000,
            n_tracts=50
        )
        
        start_time = time.time()
        
        # Run complete pipeline
        pairs = construct_repeat_sales_pairs(transactions)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nLarge dataset performance:")
        print(f"  - Transactions: {len(transactions)}")
        print(f"  - Pairs created: {len(pairs)}")
        print(f"  - Total time: {total_time:.2f}s")
        print(f"  - Throughput: {len(transactions)/total_time:.0f} transactions/s")
        
        # Should complete in reasonable time
        assert total_time < 30.0  # 30 seconds for 5000 properties