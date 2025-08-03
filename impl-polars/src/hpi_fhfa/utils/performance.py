"""Performance monitoring and optimization utilities."""

import time
import functools
import psutil
import polars as pl
from typing import Callable, Dict, Any, Optional, List
from contextlib import contextmanager
import structlog

logger = structlog.get_logger()


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        
    @contextmanager
    def measure(self, operation: str):
        """Context manager to measure operation performance.
        
        Usage:
            with monitor.measure("data_loading"):
                # code to measure
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Store metrics
            if operation not in self.metrics:
                self.metrics[operation] = []
            
            self.metrics[operation].append({
                "duration": duration,
                "memory_delta_mb": memory_delta / 1024 / 1024,
                "timestamp": time.time()
            })
            
            logger.info(
                f"Performance: {operation}",
                duration_s=f"{duration:.2f}",
                memory_delta_mb=f"{memory_delta / 1024 / 1024:.1f}"
            )
    
    def _get_memory_usage(self) -> int:
        """Get current process memory usage in bytes."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all measured operations."""
        summary = {}
        
        for operation, measurements in self.metrics.items():
            durations = [m["duration"] for m in measurements]
            memory_deltas = [m["memory_delta_mb"] for m in measurements]
            
            summary[operation] = {
                "count": len(measurements),
                "total_duration": sum(durations),
                "avg_duration": sum(durations) / len(durations),
                "max_duration": max(durations),
                "avg_memory_mb": sum(memory_deltas) / len(memory_deltas),
                "max_memory_mb": max(memory_deltas)
            }
        
        return summary
    
    def log_summary(self):
        """Log performance summary."""
        summary = self.get_summary()
        
        logger.info("Performance Summary")
        for operation, stats in summary.items():
            logger.info(
                f"  {operation}:",
                count=stats["count"],
                total_s=f"{stats['total_duration']:.1f}",
                avg_s=f"{stats['avg_duration']:.2f}",
                max_memory_mb=f"{stats['max_memory_mb']:.1f}"
            )


def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper


class OptimizedOperations:
    """Optimized implementations of common operations."""
    
    @staticmethod
    def parallel_group_apply(
        df: pl.DataFrame,
        group_col: str,
        func: Callable[[pl.DataFrame], pl.DataFrame],
        n_jobs: int = -1
    ) -> pl.DataFrame:
        """Apply function to groups in parallel.
        
        Args:
            df: DataFrame to process
            group_col: Column to group by
            func: Function to apply to each group
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Combined results DataFrame
        """
        from joblib import Parallel, delayed
        
        if n_jobs == -1:
            n_jobs = psutil.cpu_count()
        
        # Get unique groups
        groups = df[group_col].unique().to_list()
        
        # Define function to process single group
        def process_group(group_value):
            group_df = df.filter(pl.col(group_col) == group_value)
            return func(group_df)
        
        # Process in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_group)(group) for group in groups
        )
        
        # Combine results
        return pl.concat(results)
    
    @staticmethod
    def chunked_processing(
        df: pl.DataFrame,
        func: Callable[[pl.DataFrame], pl.DataFrame],
        chunk_size: int = 100_000
    ) -> pl.DataFrame:
        """Process large DataFrame in chunks to manage memory.
        
        Args:
            df: DataFrame to process
            func: Function to apply to each chunk
            chunk_size: Size of each chunk
            
        Returns:
            Combined results
        """
        n_chunks = (len(df) + chunk_size - 1) // chunk_size
        results = []
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            
            chunk = df.slice(start_idx, end_idx - start_idx)
            result = func(chunk)
            results.append(result)
            
            # Log progress
            if i % 10 == 0:
                logger.debug(f"Processed chunk {i+1}/{n_chunks}")
        
        return pl.concat(results)
    
    @staticmethod
    def optimize_repeat_sales_join(
        transactions: pl.DataFrame,
        batch_size: int = 50_000
    ) -> pl.DataFrame:
        """Optimized repeat sales identification using batched self-join.
        
        For large datasets, self-joins can be memory intensive.
        This processes properties in batches.
        """
        # Get unique properties
        unique_properties = transactions["property_id"].unique().to_list()
        n_batches = (len(unique_properties) + batch_size - 1) // batch_size
        
        results = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(unique_properties))
            batch_properties = unique_properties[start_idx:end_idx]
            
            # Filter transactions for this batch
            batch_transactions = transactions.filter(
                pl.col("property_id").is_in(batch_properties)
            )
            
            # Identify repeat sales for this batch
            batch_repeat_sales = (
                batch_transactions
                .sort(["property_id", "transaction_date"])
                .with_columns([
                    pl.col("transaction_date").shift(1).over("property_id").alias("prev_transaction_date"),
                    pl.col("transaction_price").shift(1).over("property_id").alias("prev_transaction_price"),
                ])
                .filter(pl.col("prev_transaction_date").is_not_null())
            )
            
            results.append(batch_repeat_sales)
            
            if i % 10 == 0:
                logger.debug(f"Processed property batch {i+1}/{n_batches}")
        
        return pl.concat(results)
    
    @staticmethod
    def optimize_bmn_regression(
        supertracts: Dict[str, set],
        filtered_sales: pl.DataFrame,
        periods: List[int],
        n_jobs: int = -1
    ) -> Dict[str, Dict[int, float]]:
        """Run BMN regressions in parallel."""
        from joblib import Parallel, delayed
        from ..models.bmn_regression import BMNRegression
        
        if n_jobs == -1:
            n_jobs = psutil.cpu_count()
        
        def run_single_regression(supertract_id: str, component_tracts: set):
            try:
                # Filter sales for this supertract
                supertract_sales = filtered_sales.filter(
                    pl.col("census_tract").is_in(list(component_tracts))
                )
                
                if len(supertract_sales) >= len(periods):
                    bmn = BMNRegression(periods)
                    bmn.fit(supertract_sales)
                    return supertract_id, bmn.get_index_values()
                return supertract_id, None
            except Exception as e:
                logger.warning(f"Regression failed for {supertract_id}: {e}")
                return supertract_id, None
        
        # Run regressions in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_regression)(sid, tracts)
            for sid, tracts in supertracts.items()
        )
        
        # Convert to dictionary, filtering out failures
        return {
            sid: values
            for sid, values in results
            if values is not None
        }


class MemoryOptimizer:
    """Utilities for memory optimization."""
    
    @staticmethod
    def optimize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
        """Optimize DataFrame data types to reduce memory usage."""
        optimized = df
        
        # Optimize integer columns
        for col in df.columns:
            if df[col].dtype in [pl.Int64, pl.Int32]:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if col_min >= -128 and col_max <= 127:
                    optimized = optimized.with_columns(pl.col(col).cast(pl.Int8))
                elif col_min >= -32768 and col_max <= 32767:
                    optimized = optimized.with_columns(pl.col(col).cast(pl.Int16))
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    optimized = optimized.with_columns(pl.col(col).cast(pl.Int32))
        
        # Convert string columns with low cardinality to categorical
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                n_unique = df[col].n_unique()
                n_total = len(df)
                
                # If less than 50% unique values, use categorical
                if n_unique < n_total * 0.5:
                    optimized = optimized.with_columns(pl.col(col).cast(pl.Categorical))
        
        # Log memory savings
        original_size = df.estimated_size()
        optimized_size = optimized.estimated_size()
        savings_pct = (1 - optimized_size / original_size) * 100
        
        logger.info(
            "Memory optimization complete",
            original_mb=f"{original_size / 1024 / 1024:.1f}",
            optimized_mb=f"{optimized_size / 1024 / 1024:.1f}",
            savings_pct=f"{savings_pct:.1f}%"
        )
        
        return optimized