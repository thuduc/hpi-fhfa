"""Bailey-Muth-Nourse (BMN) regression implementation for repeat sales indices."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import logging

from ..config import constants

logger = logging.getLogger(__name__)


@dataclass
class BMNResults:
    """Results from BMN regression estimation."""
    
    coefficients: np.ndarray  # δ coefficients for each time period
    std_errors: Optional[np.ndarray] = None  # Standard errors of coefficients
    residuals: Optional[np.ndarray] = None  # Regression residuals
    r_squared: Optional[float] = None  # R-squared statistic
    n_observations: int = 0  # Number of observations used
    n_parameters: int = 0  # Number of parameters estimated
    time_periods: Optional[pd.Index] = None  # Time period labels
    convergence_info: Optional[Dict[str, Any]] = None  # Solver convergence info
    
    def get_appreciation(self, period_t: int, period_t_1: int) -> float:
        """
        Calculate appreciation between two periods.
        
        p̂(t,t-1) = δ̂_t - δ̂_t-1
        """
        return self.coefficients[period_t] - self.coefficients[period_t_1]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for easier analysis."""
        df = pd.DataFrame({
            'period': self.time_periods if self.time_periods is not None else range(len(self.coefficients)),
            'coefficient': self.coefficients
        })
        
        if self.std_errors is not None:
            df['std_error'] = self.std_errors
            df['t_stat'] = df['coefficient'] / df['std_error']
            
        return df


class BMNRegressor:
    """
    Bailey-Muth-Nourse repeat sales regression estimator.
    
    Implements the regression:
    p_itτ = D'_tτ * δ_tτ + ε_itτ
    
    Where:
    - p_itτ: Log price difference for property i between times t and τ
    - D_tτ: Dummy variable matrix for time periods
    - δ_tτ: Coefficient vector (appreciation rates)
    - ε_itτ: Error term
    """
    
    def __init__(
        self,
        use_sparse: bool = True,
        calculate_std_errors: bool = True,
        tolerance: float = None,
        max_iterations: int = None
    ):
        """
        Initialize BMN regressor.
        
        Parameters
        ----------
        use_sparse : bool, default True
            Whether to use sparse matrices for efficiency
        calculate_std_errors : bool, default True
            Whether to calculate standard errors
        tolerance : float, optional
            Convergence tolerance for iterative solver
        max_iterations : int, optional
            Maximum iterations for iterative solver
        """
        self.use_sparse = use_sparse
        self.calculate_std_errors = calculate_std_errors
        self.tolerance = tolerance or constants.BMN_CONVERGENCE_TOLERANCE
        self.max_iterations = max_iterations or constants.BMN_MAX_ITERATIONS
        
    def fit(
        self,
        repeat_sales_df: pd.DataFrame,
        price_relative_col: str = 'price_relative',
        period1_col: str = 'sale1_period',
        period2_col: str = 'sale2_period',
        normalize_first: bool = True
    ) -> BMNResults:
        """
        Fit BMN regression to repeat sales data.
        
        Parameters
        ----------
        repeat_sales_df : pd.DataFrame
            DataFrame with repeat sales pairs
        price_relative_col : str
            Column containing log price differences
        period1_col : str
            Column with first sale period index
        period2_col : str
            Column with second sale period index
        normalize_first : bool, default True
            Whether to normalize first period coefficient to 0
            
        Returns
        -------
        BMNResults
            Regression results
        """
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
        
        # Create design matrix with mapped periods
        X = self._create_design_matrix(
            period1_mapped, period2_mapped, n_periods, 
            use_sparse=self.use_sparse, 
            normalize_first=normalize_first
        )
        
        # Estimate coefficients
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
    
    def _create_design_matrix(
        self,
        period1: np.ndarray,
        period2: np.ndarray,
        n_periods: int,
        use_sparse: bool = True,
        normalize_first: bool = True
    ) -> np.ndarray:
        """
        Create dummy variable design matrix for BMN regression.
        
        Each row has:
        - -1 in column corresponding to first sale period
        - +1 in column corresponding to second sale period
        - 0 elsewhere
        """
        n_obs = len(period1)
        
        if normalize_first:
            # Drop first period for identification
            n_cols = n_periods - 1
            period1_adj = period1 - 1  # Shift indices
            period2_adj = period2 - 1
        else:
            n_cols = n_periods
            period1_adj = period1
            period2_adj = period2
        
        if use_sparse:
            # Create sparse matrix efficiently
            # Build lists for COO format
            row_list = []
            col_list = []
            data_list = []
            
            for i in range(n_obs):
                # Add -1 for period1 if valid
                if not normalize_first or period1_adj[i] >= 0:
                    row_list.append(i)
                    col_list.append(period1_adj[i])
                    data_list.append(-1.0)
                
                # Add +1 for period2 if valid
                if not normalize_first or period2_adj[i] >= 0:
                    row_list.append(i)
                    col_list.append(period2_adj[i])
                    data_list.append(1.0)
            
            X = sparse.csr_matrix(
                (data_list, (row_list, col_list)),
                shape=(n_obs, n_cols)
            )
        else:
            # Create dense matrix
            X = np.zeros((n_obs, n_cols))
            for i in range(n_obs):
                if normalize_first and period1_adj[i] >= 0:
                    X[i, period1_adj[i]] = -1
                elif not normalize_first:
                    X[i, period1_adj[i]] = -1
                    
                if normalize_first and period2_adj[i] >= 0:
                    X[i, period2_adj[i]] = 1
                elif not normalize_first:
                    X[i, period2_adj[i]] = 1
        
        return X
    
    def _fit_sparse(self, X: sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
        """Fit regression using sparse matrix operations."""
        # Use sparse least squares solver
        # Normal equations: X'X β = X'y
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Add small ridge penalty for numerical stability
        ridge_penalty = 1e-8
        XtX = XtX + ridge_penalty * sparse.eye(XtX.shape[0])
        
        # Solve using sparse solver
        coefficients = sp_linalg.spsolve(XtX, Xty)
        
        return coefficients
    
    def _fit_dense(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit regression using dense matrix operations."""
        # Handle edge case with insufficient data
        if X.shape[0] == 0 or X.shape[1] == 0:
            return np.array([])
            
        # Standard OLS: (X'X)^-1 X'y
        XtX = X.T @ X
        Xty = X.T @ y
        
        # Add small ridge penalty for numerical stability
        ridge_penalty = 1e-8
        if XtX.ndim == 2:
            XtX = XtX + ridge_penalty * np.eye(XtX.shape[0])
        elif XtX.ndim == 0:
            # Scalar case
            XtX = XtX + ridge_penalty
        else:
            # 1D case - convert to 2D
            XtX = np.atleast_2d(XtX)
            XtX = XtX + ridge_penalty * np.eye(XtX.shape[0])
        
        # Solve using standard linear algebra
        try:
            coefficients = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback to least squares for singular matrices
            coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        
        return coefficients
    
    def _calculate_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coefficients: np.ndarray,
        results: BMNResults,
        normalize_first: bool
    ) -> None:
        """Calculate standard errors, residuals, and R-squared."""
        # Adjust coefficients for prediction if normalized
        if normalize_first:
            coef_for_pred = coefficients[1:]  # Remove added zero
        else:
            coef_for_pred = coefficients
        
        # Calculate predictions and residuals
        if self.use_sparse and sparse.issparse(X):
            y_pred = X @ coef_for_pred
        else:
            y_pred = X @ coef_for_pred
            
        residuals = y - y_pred
        results.residuals = residuals
        
        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        results.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Standard errors (assuming homoskedasticity)
        n = len(y)
        k = len(coef_for_pred)
        
        if n > k:
            # Residual standard error
            sigma_squared = ss_res / (n - k)
            
            # Variance-covariance matrix
            if self.use_sparse and sparse.issparse(X):
                XtX = X.T @ X
                # Add ridge penalty for stability
                XtX = XtX + 1e-8 * sparse.eye(XtX.shape[0])
                XtX_inv = sparse.linalg.inv(XtX.tocsc())
                var_coef = sigma_squared * XtX_inv.diagonal()
            else:
                XtX = X.T @ X
                XtX = XtX + 1e-8 * np.eye(XtX.shape[0])
                XtX_inv = np.linalg.inv(XtX)
                var_coef = sigma_squared * np.diagonal(XtX_inv)
            
            # Standard errors
            std_errors = np.sqrt(var_coef)
            
            # Add back zero for first period if normalized
            if normalize_first:
                std_errors = np.concatenate([[0.0], std_errors])
                
            results.std_errors = std_errors
        
        logger.info(f"BMN regression complete: R² = {results.r_squared:.4f}")


def calculate_index_from_coefficients(
    coefficients: np.ndarray,
    base_period: int = 0,
    base_value: float = 100.0
) -> np.ndarray:
    """
    Calculate price index from BMN coefficients.
    
    Parameters
    ----------
    coefficients : np.ndarray
        BMN regression coefficients (δ values)
    base_period : int, default 0
        Period to use as base (index = base_value)
    base_value : float, default 100.0
        Value for base period
        
    Returns
    -------
    np.ndarray
        Price index values
    """
    # Calculate cumulative appreciation from base period
    relative_coef = coefficients - coefficients[base_period]
    
    # Convert to index levels
    index_values = base_value * np.exp(relative_coef)
    
    return index_values