"""Validate HPI results against known benchmarks."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

from ..aggregation.index_builder import HPIIndex

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from benchmark validation."""
    
    is_valid: bool
    correlation: float
    rmse: float
    max_deviation: float
    tracking_error: float
    period_deviations: Dict[int, float]
    message: str
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert validation results to DataFrame."""
        return pd.DataFrame({
            'metric': ['is_valid', 'correlation', 'rmse', 'max_deviation', 'tracking_error'],
            'value': [self.is_valid, self.correlation, self.rmse, 
                     self.max_deviation, self.tracking_error]
        })


class BenchmarkValidator:
    """Validate HPI indices against known benchmarks."""
    
    def __init__(self,
                 correlation_threshold: float = 0.95,
                 rmse_threshold: float = 0.05,
                 max_deviation_threshold: float = 0.10):
        """Initialize validator.
        
        Args:
            correlation_threshold: Minimum acceptable correlation
            rmse_threshold: Maximum acceptable RMSE
            max_deviation_threshold: Maximum acceptable single-period deviation
        """
        self.correlation_threshold = correlation_threshold
        self.rmse_threshold = rmse_threshold
        self.max_deviation_threshold = max_deviation_threshold
    
    def validate_against_benchmark(self,
                                 calculated_index: HPIIndex,
                                 benchmark_data: pd.DataFrame,
                                 value_column: str = 'index_value',
                                 period_column: str = 'period') -> ValidationResult:
        """Validate calculated index against benchmark data.
        
        Args:
            calculated_index: Calculated HPI index
            benchmark_data: DataFrame with benchmark values
            value_column: Column name for index values in benchmark
            period_column: Column name for periods in benchmark
            
        Returns:
            ValidationResult with detailed comparison metrics
        """
        logger.info(f"Validating index {calculated_index.entity_id} against benchmark")
        
        # Align data
        calc_df = calculated_index.to_dataframe()
        
        # Merge on period
        merged = pd.merge(
            calc_df[['period', 'index_value']].rename(columns={'index_value': 'calculated'}),
            benchmark_data[[period_column, value_column]].rename(
                columns={period_column: 'period', value_column: 'benchmark'}
            ),
            on='period',
            how='inner'
        )
        
        if len(merged) == 0:
            return ValidationResult(
                is_valid=False,
                correlation=0.0,
                rmse=np.inf,
                max_deviation=np.inf,
                tracking_error=np.inf,
                period_deviations={},
                message="No overlapping periods between calculated and benchmark"
            )
        
        # Calculate metrics
        correlation = merged['calculated'].corr(merged['benchmark'])
        
        # RMSE
        rmse = np.sqrt(np.mean((merged['calculated'] - merged['benchmark'])**2))
        
        # Max deviation
        deviations = abs(merged['calculated'] - merged['benchmark']) / merged['benchmark']
        max_deviation = deviations.max()
        
        # Tracking error (standard deviation of differences)
        tracking_error = np.std(merged['calculated'] - merged['benchmark'])
        
        # Period-by-period deviations
        period_deviations = dict(zip(merged['period'], deviations))
        
        # Determine if valid
        is_valid = (
            correlation >= self.correlation_threshold and
            rmse <= self.rmse_threshold and
            max_deviation <= self.max_deviation_threshold
        )
        
        # Create message
        if is_valid:
            message = "Index validation passed all criteria"
        else:
            failures = []
            if correlation < self.correlation_threshold:
                failures.append(f"correlation {correlation:.3f} < {self.correlation_threshold}")
            if rmse > self.rmse_threshold:
                failures.append(f"RMSE {rmse:.3f} > {self.rmse_threshold}")
            if max_deviation > self.max_deviation_threshold:
                failures.append(f"max deviation {max_deviation:.3f} > {self.max_deviation_threshold}")
            message = f"Validation failed: {', '.join(failures)}"
        
        result = ValidationResult(
            is_valid=is_valid,
            correlation=correlation,
            rmse=rmse,
            max_deviation=max_deviation,
            tracking_error=tracking_error,
            period_deviations=period_deviations,
            message=message
        )
        
        logger.info(f"Validation result: {message}")
        return result
    
    def validate_multiple_indices(self,
                                indices: Dict[str, HPIIndex],
                                benchmark_data: Dict[str, pd.DataFrame]) -> Dict[str, ValidationResult]:
        """Validate multiple indices against benchmarks.
        
        Args:
            indices: Dictionary of entity_id to HPIIndex
            benchmark_data: Dictionary of entity_id to benchmark DataFrame
            
        Returns:
            Dictionary of entity_id to ValidationResult
        """
        results = {}
        
        for entity_id, index in indices.items():
            if entity_id in benchmark_data:
                result = self.validate_against_benchmark(
                    index, benchmark_data[entity_id]
                )
                results[entity_id] = result
            else:
                logger.warning(f"No benchmark data for entity {entity_id}")
        
        return results
    
    def generate_validation_report(self,
                                 validation_results: Dict[str, ValidationResult]) -> pd.DataFrame:
        """Generate summary report of validation results.
        
        Args:
            validation_results: Dictionary of validation results
            
        Returns:
            DataFrame with summary statistics
        """
        report_data = []
        
        for entity_id, result in validation_results.items():
            report_data.append({
                'entity_id': entity_id,
                'is_valid': result.is_valid,
                'correlation': result.correlation,
                'rmse': result.rmse,
                'max_deviation': result.max_deviation,
                'tracking_error': result.tracking_error,
                'message': result.message
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Add summary statistics
        summary = pd.DataFrame({
            'metric': ['Total Entities', 'Valid Entities', 'Invalid Entities',
                      'Avg Correlation', 'Avg RMSE', 'Avg Max Deviation'],
            'value': [
                len(report_df),
                report_df['is_valid'].sum(),
                (~report_df['is_valid']).sum(),
                report_df['correlation'].mean(),
                report_df['rmse'].mean(),
                report_df['max_deviation'].mean()
            ]
        })
        
        return report_df, summary
    
    def cross_validate_indices(self,
                             indices: List[HPIIndex],
                             max_deviation: float = 0.05) -> Tuple[bool, pd.DataFrame]:
        """Cross-validate indices against each other.
        
        Useful when indices should be similar (e.g., neighboring areas).
        
        Args:
            indices: List of indices to cross-validate
            max_deviation: Maximum acceptable deviation between indices
            
        Returns:
            Tuple of (all_valid, comparison_matrix)
        """
        n = len(indices)
        comparison_matrix = pd.DataFrame(
            np.nan,
            index=[idx.entity_id for idx in indices],
            columns=[idx.entity_id for idx in indices]
        )
        
        all_valid = True
        
        for i in range(n):
            for j in range(i+1, n):
                idx1, idx2 = indices[i], indices[j]
                
                # Get common periods
                periods1 = set(idx1.index_values.keys())
                periods2 = set(idx2.index_values.keys())
                common_periods = periods1 & periods2
                
                if len(common_periods) > 0:
                    # Calculate average deviation
                    deviations = []
                    for period in common_periods:
                        val1 = idx1.index_values[period]
                        val2 = idx2.index_values[period]
                        dev = abs(val1 - val2) / max(val1, val2)
                        deviations.append(dev)
                    
                    avg_deviation = np.mean(deviations)
                    comparison_matrix.loc[idx1.entity_id, idx2.entity_id] = avg_deviation
                    comparison_matrix.loc[idx2.entity_id, idx1.entity_id] = avg_deviation
                    
                    if avg_deviation > max_deviation:
                        all_valid = False
        
        # Fill diagonal with zeros
        np.fill_diagonal(comparison_matrix.values, 0.0)
        
        return all_valid, comparison_matrix