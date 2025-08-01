"""Optimized BMN regression using Numba acceleration."""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional

from ..models.bmn_regression import BMNRegressor, BMNResults
from .numba_functions import fast_design_matrix, fast_log_diff, is_numba_available
import logging

logger = logging.getLogger(__name__)


class OptimizedBMNRegressor(BMNRegressor):
    """BMN Regressor with Numba acceleration for performance-critical operations."""
    
    def __init__(self, *args, use_numba: bool = True, **kwargs):
        """Initialize optimized BMN regressor.
        
        Args:
            use_numba: Whether to use Numba acceleration (if available)
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.use_numba = use_numba and is_numba_available()
        
        if self.use_numba:
            logger.info("Numba acceleration enabled for BMN regression")
        else:
            logger.info("Numba acceleration not available, using standard implementation")
    
    def fit(self,
            repeat_sales_df: pd.DataFrame,
            price_relative_col: str = 'price_relative',
            period1_col: str = 'sale1_period',
            period2_col: str = 'sale2_period',
            normalize_first: bool = True) -> BMNResults:
        """Fit BMN regression with optional Numba acceleration.
        
        Uses Numba-accelerated functions for:
        - Design matrix construction
        - Log price difference calculation (if needed)
        
        Args:
            repeat_sales_df: DataFrame with repeat sales pairs
            price_relative_col: Column containing log price differences
            period1_col: Column with first sale period index
            period2_col: Column with second sale period index
            normalize_first: Whether to normalize first period coefficient to 0
            
        Returns:
            BMNResults: Regression results
        """
        # Check if we need to calculate price relatives
        if price_relative_col not in repeat_sales_df.columns:
            if self.use_numba and 'sale1_price' in repeat_sales_df.columns:
                logger.debug("Using Numba-accelerated log difference calculation")
                prices1 = repeat_sales_df['sale1_price'].values
                prices2 = repeat_sales_df['sale2_price'].values
                repeat_sales_df = repeat_sales_df.copy()
                repeat_sales_df[price_relative_col] = fast_log_diff(prices1, prices2)
        
        # Extract data
        y = repeat_sales_df[price_relative_col].values
        period1 = repeat_sales_df[period1_col].values
        period2 = repeat_sales_df[period2_col].values
        
        # Get unique periods and create mapping to sequential indices
        all_periods = np.unique(np.concatenate([period1, period2]))
        n_periods = len(all_periods)
        n_obs = len(y)
        
        # Create mapping from original periods to sequential indices
        period_map = {period: idx for idx, period in enumerate(all_periods)}
        
        # Map periods to sequential indices
        period1_mapped = np.array([period_map[p] for p in period1])
        period2_mapped = np.array([period_map[p] for p in period2])
        
        logger.info(f"Fitting BMN regression: {n_obs:,} observations, {n_periods} periods")
        
        # Create design matrix
        if self.use_numba and self.use_sparse:
            logger.debug("Using Numba-accelerated design matrix construction")
            X = self._create_design_matrix_numba(
                period1_mapped, period2_mapped, n_periods, 
                normalize_first=normalize_first
            )
        else:
            # Fall back to parent implementation
            X = self._create_design_matrix(
                period1_mapped, period2_mapped, n_periods,
                use_sparse=self.use_sparse,
                normalize_first=normalize_first
            )
        
        # Estimate coefficients (using parent class methods)
        if self.use_sparse:
            coefficients = self._fit_sparse(X, y)
        else:
            coefficients = self._fit_dense(X, y)
        
        # Add back normalized coefficient if needed
        if normalize_first:
            coefficients = np.concatenate([[0.0], coefficients])
        
        # Calculate additional statistics
        results = BMNResults(
            coefficients=coefficients,
            n_observations=n_obs,
            n_parameters=len(coefficients)
        )
        
        # Add period labels if available
        if 'sale1_date' in repeat_sales_df.columns:
            unique_dates = pd.concat([
                repeat_sales_df['sale1_date'],
                repeat_sales_df['sale2_date']
            ]).unique()
            results.time_periods = pd.Series(unique_dates).sort_values().reset_index(drop=True)
        
        # Calculate standard errors and R-squared if requested
        if self.calculate_std_errors:
            self._calculate_statistics(X, y, coefficients, results, normalize_first)
        
        return results
    
    def _create_design_matrix_numba(self,
                                   period1: np.ndarray,
                                   period2: np.ndarray,
                                   n_periods: int,
                                   normalize_first: bool = True) -> sparse.csr_matrix:
        """Create design matrix using Numba acceleration.
        
        Args:
            period1: First period indices
            period2: Second period indices
            n_periods: Total number of periods
            normalize_first: Whether to drop first period
            
        Returns:
            Sparse CSR matrix
        """
        # Get sparse matrix components using Numba
        row_indices, col_indices, values = fast_design_matrix(
            period1, period2, n_periods, normalize_first
        )
        
        # Determine matrix shape
        n_obs = len(period1)
        if normalize_first:
            n_cols = n_periods - 1
        else:
            n_cols = n_periods
        
        # Create sparse matrix
        X = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_obs, n_cols)
        )
        
        return X