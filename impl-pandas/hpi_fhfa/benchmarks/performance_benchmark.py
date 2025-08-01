"""Performance benchmarking for HPI calculations."""

import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import psutil
import logging

from ..aggregation import CityLevelIndexBuilder, WeightType
from ..models.repeat_sales import construct_repeat_sales_pairs
from ..models.bmn_regression import BMNRegressor
from ..optimization.bmn_optimized import OptimizedBMNRegressor
from ..optimization.numba_functions import is_numba_available

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Results from performance benchmarking."""
    
    operation: str
    n_records: int
    execution_time: float
    memory_used_mb: float
    throughput: float  # records per second
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'n_records': self.n_records,
            'execution_time': self.execution_time,
            'memory_used_mb': self.memory_used_mb,
            'throughput': self.throughput,
            **self.metadata
        }


class PerformanceBenchmark:
    """Benchmark performance of HPI calculations."""
    
    def __init__(self, warmup_runs: int = 2, benchmark_runs: int = 5):
        """Initialize benchmark.
        
        Args:
            warmup_runs: Number of warmup runs before timing
            benchmark_runs: Number of timed runs
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.process = psutil.Process()
    
    def _measure_execution(self, 
                         func: Callable,
                         *args,
                         **kwargs) -> Tuple[float, float, any]:
        """Measure execution time and memory usage.
        
        Returns:
            Tuple of (execution_time, memory_used_mb, result)
        """
        # Get initial memory
        self.process.memory_info()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Time execution
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Get final memory
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        memory_used = final_memory - initial_memory
        
        return execution_time, memory_used, result
    
    def benchmark_repeat_sales_construction(self,
                                          transactions: pd.DataFrame) -> BenchmarkResults:
        """Benchmark repeat sales pair construction.
        
        Args:
            transactions: Transaction data
            
        Returns:
            BenchmarkResults
        """
        logger.info("Benchmarking repeat sales construction")
        
        # Warmup
        for _ in range(self.warmup_runs):
            construct_repeat_sales_pairs(transactions)
        
        # Benchmark
        times = []
        memory_usage = []
        
        for _ in range(self.benchmark_runs):
            exec_time, mem_used, pairs = self._measure_execution(
                construct_repeat_sales_pairs,
                transactions
            )
            times.append(exec_time)
            memory_usage.append(mem_used)
        
        avg_time = np.mean(times)
        avg_memory = np.mean(memory_usage)
        n_pairs = len(pairs) if pairs is not None else 0
        
        return BenchmarkResults(
            operation="repeat_sales_construction",
            n_records=len(transactions),
            execution_time=avg_time,
            memory_used_mb=avg_memory,
            throughput=len(transactions) / avg_time,
            metadata={
                'n_pairs_created': n_pairs,
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        )
    
    def benchmark_bmn_regression(self,
                               repeat_sales_pairs: pd.DataFrame,
                               use_numba: bool = True) -> BenchmarkResults:
        """Benchmark BMN regression.
        
        Args:
            repeat_sales_pairs: Repeat sales pairs
            use_numba: Whether to use Numba optimization
            
        Returns:
            BenchmarkResults
        """
        operation = "bmn_regression_numba" if use_numba else "bmn_regression_standard"
        logger.info(f"Benchmarking {operation}")
        
        # Choose regressor
        if use_numba and is_numba_available():
            regressor = OptimizedBMNRegressor(use_numba=True)
        else:
            regressor = BMNRegressor()
        
        # Warmup
        for _ in range(self.warmup_runs):
            regressor.fit(repeat_sales_pairs)
        
        # Benchmark
        times = []
        memory_usage = []
        
        for _ in range(self.benchmark_runs):
            exec_time, mem_used, results = self._measure_execution(
                regressor.fit,
                repeat_sales_pairs
            )
            times.append(exec_time)
            memory_usage.append(mem_used)
        
        avg_time = np.mean(times)
        avg_memory = np.mean(memory_usage)
        
        return BenchmarkResults(
            operation=operation,
            n_records=len(repeat_sales_pairs),
            execution_time=avg_time,
            memory_used_mb=avg_memory,
            throughput=len(repeat_sales_pairs) / avg_time,
            metadata={
                'n_periods': results.n_parameters if results else 0,
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'numba_available': is_numba_available()
            }
        )
    
    def benchmark_index_construction(self,
                                   transactions: pd.DataFrame,
                                   census_tracts: List,
                                   weight_type: WeightType = WeightType.SAMPLE) -> BenchmarkResults:
        """Benchmark complete index construction.
        
        Args:
            transactions: Transaction data
            census_tracts: List of census tracts
            weight_type: Type of weighting
            
        Returns:
            BenchmarkResults
        """
        logger.info(f"Benchmarking index construction with {weight_type} weights")
        
        builder = CityLevelIndexBuilder()
        
        # Warmup
        for _ in range(self.warmup_runs):
            builder.build_annual_index(
                transactions,
                census_tracts,
                weight_type,
                start_year=2015,
                end_year=2021
            )
        
        # Benchmark
        times = []
        memory_usage = []
        
        for _ in range(self.benchmark_runs):
            exec_time, mem_used, index = self._measure_execution(
                builder.build_annual_index,
                transactions,
                census_tracts,
                weight_type,
                start_year=2015,
                end_year=2021
            )
            times.append(exec_time)
            memory_usage.append(mem_used)
        
        avg_time = np.mean(times)
        avg_memory = np.mean(memory_usage)
        
        return BenchmarkResults(
            operation=f"index_construction_{weight_type}",
            n_records=len(transactions),
            execution_time=avg_time,
            memory_used_mb=avg_memory,
            throughput=len(transactions) / avg_time,
            metadata={
                'n_tracts': len(census_tracts),
                'weight_type': str(weight_type),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        )
    
    def compare_implementations(self,
                              repeat_sales_pairs: pd.DataFrame) -> pd.DataFrame:
        """Compare standard vs optimized implementations.
        
        Args:
            repeat_sales_pairs: Repeat sales pairs
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        # Benchmark standard implementation
        standard_result = self.benchmark_bmn_regression(
            repeat_sales_pairs, use_numba=False
        )
        results.append(standard_result.to_dict())
        
        # Benchmark Numba implementation if available
        if is_numba_available():
            numba_result = self.benchmark_bmn_regression(
                repeat_sales_pairs, use_numba=True
            )
            results.append(numba_result.to_dict())
            
            # Calculate speedup
            speedup = standard_result.execution_time / numba_result.execution_time
            logger.info(f"Numba speedup: {speedup:.2f}x")
        
        return pd.DataFrame(results)
    
    def run_scaling_benchmark(self,
                            base_transactions: pd.DataFrame,
                            scale_factors: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]) -> pd.DataFrame:
        """Benchmark performance at different data scales.
        
        Args:
            base_transactions: Base transaction dataset
            scale_factors: Multipliers for dataset size
            
        Returns:
            DataFrame with scaling results
        """
        results = []
        
        for scale in scale_factors:
            # Scale the data
            if scale < 1.0:
                n_samples = int(len(base_transactions) * scale)
                scaled_data = base_transactions.sample(n=n_samples, random_state=42)
            else:
                # Replicate data
                scaled_data = pd.concat([base_transactions] * int(scale), ignore_index=True)
                # Modify property IDs to make them unique
                scaled_data['property_id'] = [
                    f"{pid}_{i}" for i, pid in enumerate(scaled_data['property_id'])
                ]
            
            logger.info(f"Benchmarking with scale factor {scale} ({len(scaled_data)} records)")
            
            # Benchmark repeat sales construction
            result = self.benchmark_repeat_sales_construction(scaled_data)
            result_dict = result.to_dict()
            result_dict['scale_factor'] = scale
            results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def generate_performance_report(self,
                                  benchmark_results: List[BenchmarkResults]) -> pd.DataFrame:
        """Generate summary performance report.
        
        Args:
            benchmark_results: List of benchmark results
            
        Returns:
            DataFrame with performance summary
        """
        data = [r.to_dict() for r in benchmark_results]
        df = pd.DataFrame(data)
        
        # Add derived metrics
        df['records_per_second'] = df['throughput']
        df['ms_per_record'] = 1000 / df['throughput']
        df['efficiency'] = df['throughput'] / df['memory_used_mb']  # Records per second per MB
        
        return df