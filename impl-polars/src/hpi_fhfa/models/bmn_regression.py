"""Bailey-Muth-Nourse (BMN) repeat-sales regression implementation."""

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict, List, Optional
import structlog

from ..utils.exceptions import InsufficientDataError, ProcessingError

logger = structlog.get_logger()


class BMNRegression:
    """Bailey-Muth-Nourse repeat-sales regression implementation.
    
    The BMN regression estimates price appreciation rates by regressing
    log price differences on time dummy variables:
    
    p_itτ = D'_tτ * δ_tτ + ε_itτ
    
    Where:
    - p_itτ: Log price difference for property i between times t and τ
    - D_tτ: Dummy variable matrix for time periods
    - δ_tτ: Coefficient vector (appreciation rates)
    - ε_itτ: Error term
    """
    
    def __init__(self, time_periods: List[int]):
        """Initialize BMN regression.
        
        Args:
            time_periods: List of time periods (e.g., years)
        """
        self.time_periods = sorted(time_periods)
        self.n_periods = len(time_periods)
        self.period_to_idx = {period: idx for idx, period in enumerate(self.time_periods)}
        self.coefficients = None
        self.residuals = None
        
    def create_dummy_matrix(
        self, 
        repeat_sales_df: pl.DataFrame
    ) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Create sparse dummy variable matrix D_tτ.
        
        For each repeat sale from period τ to period t:
        - Set D[i, τ] = -1
        - Set D[i, t] = +1
        
        Args:
            repeat_sales_df: DataFrame with repeat sales data
                Must contain: transaction_date, prev_transaction_date
                
        Returns:
            Tuple of (dummy_matrix, log_price_diffs)
        """
        n_obs = len(repeat_sales_df)
        
        # Extract periods from dates
        df = repeat_sales_df.with_columns([
            pl.col("transaction_date").dt.year().alias("sale_period"),
            pl.col("prev_transaction_date").dt.year().alias("prev_period")
        ])
        
        # Get arrays for matrix construction
        sale_periods = df["sale_period"].to_numpy()
        prev_periods = df["prev_period"].to_numpy()
        log_price_diffs = df["log_price_diff"].to_numpy()
        
        # Build sparse matrix
        row_indices = []
        col_indices = []
        data = []
        
        for i, (sale_period, prev_period) in enumerate(zip(sale_periods, prev_periods)):
            if sale_period in self.period_to_idx and prev_period in self.period_to_idx:
                # Previous period gets -1
                row_indices.append(i)
                col_indices.append(self.period_to_idx[prev_period])
                data.append(-1.0)
                
                # Current period gets +1
                row_indices.append(i)
                col_indices.append(self.period_to_idx[sale_period])
                data.append(1.0)
        
        # Create sparse matrix
        dummy_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_obs, self.n_periods)
        )
        
        logger.debug(
            "Created dummy matrix",
            shape=dummy_matrix.shape,
            nnz=dummy_matrix.nnz,
            sparsity=f"{dummy_matrix.nnz / (n_obs * self.n_periods) * 100:.2f}%"
        )
        
        return dummy_matrix, log_price_diffs
    
    def fit(
        self, 
        repeat_sales_df: pl.DataFrame,
        method: str = "ols",
        normalize_first_period: bool = True
    ) -> np.ndarray:
        """Estimate δ coefficients using regression.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales data
            method: Regression method ('ols' or 'lsqr')
            normalize_first_period: If True, set first period coefficient to 0
            
        Returns:
            Array of estimated coefficients (δ)
            
        Raises:
            InsufficientDataError: If not enough data for regression
            ProcessingError: If regression fails
        """
        logger.info(
            "Fitting BMN regression",
            n_observations=len(repeat_sales_df),
            n_periods=self.n_periods,
            method=method
        )
        
        if len(repeat_sales_df) < self.n_periods:
            raise InsufficientDataError(
                f"Need at least {self.n_periods} observations, got {len(repeat_sales_df)}"
            )
        
        try:
            # Create dummy matrix and dependent variable
            X, y = self.create_dummy_matrix(repeat_sales_df)
            
            if normalize_first_period:
                # Drop first column to avoid multicollinearity
                # This sets the first period as the base (δ_0 = 0)
                X = X[:, 1:]
                
            # Fit regression
            if method == "ols":
                # Use sklearn for small problems
                if X.shape[0] < 10000:
                    model = LinearRegression(fit_intercept=False)
                    model.fit(X, y)
                    coeffs = model.coef_
                else:
                    # Use scipy.sparse.linalg for larger problems
                    coeffs, istop, itn, r1norm = lsqr(X, y)[:4]
                    if istop > 2:  # Check convergence
                        logger.warning(f"LSQR convergence issue: istop={istop}")
            elif method == "lsqr":
                # Always use LSQR
                coeffs, istop, itn, r1norm = lsqr(X, y)[:4]
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Add back the zero for first period if normalized
            if normalize_first_period:
                coeffs = np.concatenate([[0.0], coeffs])
            
            self.coefficients = coeffs
            
            # Calculate residuals
            y_pred = X @ coeffs[1:] if normalize_first_period else X @ coeffs
            self.residuals = y - y_pred
            
            logger.info(
                "BMN regression complete",
                rmse=np.sqrt(np.mean(self.residuals**2)),
                r_squared=1 - np.var(self.residuals) / np.var(y)
            )
            
            return self.coefficients
            
        except Exception as e:
            logger.error("BMN regression failed", error=str(e))
            raise ProcessingError(f"BMN regression failed: {e}")
    
    def calculate_appreciation(
        self, 
        period_t: int, 
        period_t_1: int
    ) -> float:
        """Calculate appreciation rate between two periods.
        
        Appreciation = δ_t - δ_{t-1}
        
        Args:
            period_t: Current period
            period_t_1: Previous period
            
        Returns:
            Appreciation rate (log difference)
        """
        if self.coefficients is None:
            raise ValueError("Must fit regression before calculating appreciation")
        
        if period_t not in self.period_to_idx:
            raise ValueError(f"Period {period_t} not in fitted periods")
        if period_t_1 not in self.period_to_idx:
            raise ValueError(f"Period {period_t_1} not in fitted periods")
        
        delta_t = self.coefficients[self.period_to_idx[period_t]]
        delta_t_1 = self.coefficients[self.period_to_idx[period_t_1]]
        
        return delta_t - delta_t_1
    
    def get_index_values(self, base_period: Optional[int] = None) -> Dict[int, float]:
        """Convert coefficients to index values.
        
        Args:
            base_period: Period to use as base (100). If None, uses first period.
            
        Returns:
            Dictionary mapping periods to index values
        """
        if self.coefficients is None:
            raise ValueError("Must fit regression before getting index values")
        
        if base_period is None:
            base_period = self.time_periods[0]
        
        base_idx = self.period_to_idx[base_period]
        base_delta = self.coefficients[base_idx]
        
        # Calculate index values relative to base
        index_values = {}
        for period, idx in self.period_to_idx.items():
            # Index = 100 * exp(δ_t - δ_base)
            index_values[period] = 100 * np.exp(self.coefficients[idx] - base_delta)
        
        return index_values
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Get regression diagnostics.
        
        Returns:
            Dictionary with diagnostic statistics
        """
        if self.residuals is None:
            return {}
        
        return {
            "n_observations": len(self.residuals),
            "n_periods": self.n_periods,
            "rmse": np.sqrt(np.mean(self.residuals**2)),
            "mae": np.mean(np.abs(self.residuals)),
            "residual_std": np.std(self.residuals),
            "min_residual": np.min(self.residuals),
            "max_residual": np.max(self.residuals)
        }