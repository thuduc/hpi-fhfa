"""Performance benchmarking utilities for HPI pipeline."""

import time
import psutil
import polars as pl
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import structlog

from ..processing.pipeline import HPIPipeline, HPIResults
from ..config.settings import HPIConfig

logger = structlog.get_logger()


@dataclass
class BenchmarkResult:
    """Result of performance benchmark."""
    name: str
    duration_seconds: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_delta_mb: float
    n_transactions: int
    n_tracts: int
    n_cbsas: int
    throughput_transactions_per_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return (
            f"Benchmark: {self.name}\n"
            f"Duration: {self.duration_seconds:.2f}s\n"
            f"Peak Memory: {self.peak_memory_mb:.1f}MB\n"
            f"Throughput: {self.throughput_transactions_per_sec:.0f} transactions/sec\n"
            f"Data: {self.n_transactions:,} transactions, {self.n_tracts:,} tracts, {self.n_cbsas} CBSAs"
        )


class PerformanceBenchmark:
    """Performance benchmarking for HPI pipeline."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_memory = self._get_memory_usage()
        
    def benchmark_pipeline(
        self,
        config: HPIConfig,
        name: str = "pipeline_benchmark"
    ) -> BenchmarkResult:
        """Benchmark full pipeline execution.
        
        Args:
            config: Pipeline configuration
            name: Benchmark name
            
        Returns:
            Benchmark result
        """
        logger.info(f"Starting benchmark: {name}")
        
        # Record initial state
        start_time = time.time()
        start_memory = self._get_memory_usage()
        peak_memory = start_memory
        
        # Memory monitoring
        memory_samples = []
        
        try:
            # Create and run pipeline
            pipeline = HPIPipeline(config)
            
            # Monitor memory during execution
            def memory_monitor():
                nonlocal peak_memory
                current_memory = self._get_memory_usage()
                memory_samples.append(current_memory)
                peak_memory = max(peak_memory, current_memory)
            
            # Run pipeline with periodic memory monitoring
            results = pipeline.run()
            
            # Final measurements
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Get data statistics
            n_transactions = results.metadata.get("n_transactions", 0)
            n_tracts = len(results.tract_indices["tract_id"].unique()) if not results.tract_indices.is_empty() else 0
            n_cbsas = len(set().union(*[
                df["cbsa_code"].unique().to_list() 
                for df in results.city_indices.values() 
                if not df.is_empty()
            ])) if results.city_indices else 0
            
            throughput = n_transactions / duration if duration > 0 else 0
            
            # Create result
            benchmark_result = BenchmarkResult(
                name=name,
                duration_seconds=duration,
                peak_memory_mb=peak_memory / (1024 * 1024),
                final_memory_mb=end_memory / (1024 * 1024),
                memory_delta_mb=memory_delta / (1024 * 1024),
                n_transactions=n_transactions,
                n_tracts=n_tracts,
                n_cbsas=n_cbsas,
                throughput_transactions_per_sec=throughput,
                metadata={
                    "config": {
                        "start_year": config.start_year,
                        "end_year": config.end_year,
                        "n_jobs": config.n_jobs,
                        "weight_schemes": config.weight_schemes,
                        "chunk_size": config.chunk_size,
                        "use_lazy_evaluation": config.use_lazy_evaluation
                    },
                    "results_metadata": results.metadata,
                    "memory_samples": memory_samples[:10]  # Keep first 10 samples
                }
            )
            
            self.results.append(benchmark_result)
            
            logger.info(
                f"Benchmark completed: {name}",
                duration=f"{duration:.2f}s",
                peak_memory_mb=f"{peak_memory / (1024 * 1024):.1f}",
                throughput=f"{throughput:.0f} txn/s"
            )
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Benchmark failed: {name}", error=str(e))
            raise
    
    def benchmark_data_sizes(
        self,
        base_config: HPIConfig,
        data_sizes: List[int],
        data_generator_func: callable
    ) -> List[BenchmarkResult]:
        """Benchmark pipeline with different data sizes.
        
        Args:
            base_config: Base configuration to modify
            data_sizes: List of transaction counts to test
            data_generator_func: Function to generate test data
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for size in data_sizes:
            logger.info(f"Benchmarking with {size:,} transactions")
            
            # Generate test data
            transaction_data, geographic_data = data_generator_func(size)
            
            # Create temporary config
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                txn_path = temp_path / f"transactions_{size}.parquet"
                geo_path = temp_path / f"geographic_{size}.parquet"
                
                transaction_data.write_parquet(txn_path)
                geographic_data.write_parquet(geo_path)
                
                # Update config
                test_config = HPIConfig(
                    transaction_data_path=txn_path,
                    geographic_data_path=geo_path,
                    output_path=temp_path / "output",
                    start_year=base_config.start_year,
                    end_year=base_config.end_year,
                    n_jobs=base_config.n_jobs,
                    weight_schemes=base_config.weight_schemes[:1],  # Use only one scheme for speed
                    use_lazy_evaluation=base_config.use_lazy_evaluation,
                    validate_data=False  # Skip validation for speed
                )
                
                # Run benchmark
                result = self.benchmark_pipeline(
                    test_config, 
                    name=f"scaling_{size}_transactions"
                )
                results.append(result)
        
        return results
    
    def generate_scaling_report(self, results: List[BenchmarkResult]) -> str:
        """Generate scaling analysis report."""
        if not results:
            return "No benchmark results available."
        
        # Sort by transaction count
        sorted_results = sorted(results, key=lambda r: r.n_transactions)
        
        report = [
            "HPI PIPELINE SCALING ANALYSIS",
            "=" * 50,
            f"Benchmarks run: {len(results)}",
            "",
            "PERFORMANCE BY DATA SIZE:",
            "-" * 40,
            f"{'Transactions':<12} {'Duration':<10} {'Memory':<10} {'Throughput':<12} {'Efficiency':<10}",
            f"{'Count':<12} {'(sec)':<10} {'(MB)':<10} {'(txn/sec)':<12} {'(txn/MB)':<10}",
            "-" * 64
        ]
        
        for result in sorted_results:
            efficiency = result.throughput_transactions_per_sec / result.peak_memory_mb if result.peak_memory_mb > 0 else 0
            report.append(
                f"{result.n_transactions:<12,} "
                f"{result.duration_seconds:<10.1f} "
                f"{result.peak_memory_mb:<10.0f} "
                f"{result.throughput_transactions_per_sec:<12.0f} "
                f"{efficiency:<10.2f}"
            )
        
        # Calculate scaling factors
        if len(sorted_results) >= 2:
            baseline = sorted_results[0]
            largest = sorted_results[-1]
            
            size_ratio = largest.n_transactions / baseline.n_transactions
            time_ratio = largest.duration_seconds / baseline.duration_seconds
            memory_ratio = largest.peak_memory_mb / baseline.peak_memory_mb
            
            report.extend([
                "",
                "SCALING ANALYSIS:",
                "-" * 20,
                f"Data size increase: {size_ratio:.1f}x ({baseline.n_transactions:,} → {largest.n_transactions:,})",
                f"Time increase: {time_ratio:.1f}x ({baseline.duration_seconds:.1f}s → {largest.duration_seconds:.1f}s)",
                f"Memory increase: {memory_ratio:.1f}x ({baseline.peak_memory_mb:.0f}MB → {largest.peak_memory_mb:.0f}MB)",
                f"Time complexity: O(n^{np.log(time_ratio)/np.log(size_ratio):.2f})",
                f"Memory complexity: O(n^{np.log(memory_ratio)/np.log(size_ratio):.2f})"
            ])
        
        return "\n".join(report)
    
    def _get_memory_usage(self) -> int:
        """Get current process memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0
    
    def get_summary_report(self) -> str:
        """Generate summary report of all benchmarks."""
        if not self.results:
            return "No benchmark results available."
        
        report = [
            "HPI PERFORMANCE BENCHMARK SUMMARY",
            "=" * 50,
            f"Total benchmarks: {len(self.results)}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "BENCHMARK RESULTS:",
            "-" * 30
        ]
        
        for result in self.results:
            report.extend([
                f"Name: {result.name}",
                f"  Duration: {result.duration_seconds:.2f}s",
                f"  Peak Memory: {result.peak_memory_mb:.1f}MB",
                f"  Throughput: {result.throughput_transactions_per_sec:.0f} transactions/sec",
                f"  Data: {result.n_transactions:,} transactions, {result.n_tracts:,} tracts",
                ""
            ])
        
        # Overall statistics
        avg_duration = np.mean([r.duration_seconds for r in self.results])
        avg_memory = np.mean([r.peak_memory_mb for r in self.results])
        avg_throughput = np.mean([r.throughput_transactions_per_sec for r in self.results])
        
        report.extend([
            "OVERALL STATISTICS:",
            "-" * 20,
            f"Average Duration: {avg_duration:.2f}s",
            f"Average Peak Memory: {avg_memory:.1f}MB",
            f"Average Throughput: {avg_throughput:.0f} transactions/sec"
        ])
        
        return "\n".join(report)


def benchmark_pipeline(config: HPIConfig, name: str = "benchmark") -> BenchmarkResult:
    """Convenience function to benchmark a pipeline configuration."""
    benchmark = PerformanceBenchmark()
    return benchmark.benchmark_pipeline(config, name)