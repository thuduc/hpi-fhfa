"""
Bailey-Muth-Nourse (BMN) regression implementation using Polars data.

This module implements the BMN repeat sales regression method for calculating
price indices from repeat sales pairs.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import date, datetime
from collections import defaultdict

import polars as pl
import numpy as np
import statsmodels.api as sm
from scipy import sparse
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.regression.linear_model import WLS

from rsai.src.data.models import (
    BMNRegressionResult,
    IndexValue,
    GeographyLevel,
    WeightingScheme
)

logger = logging.getLogger(__name__)


class BMNRegression:
    """
    Implements Bailey-Muth-Nourse repeat sales regression.
    
    The BMN method estimates price indices by regressing log price ratios
    on time dummy variables, providing index values for each time period.
    """
    
    def __init__(
        self,
        base_period: Optional[date] = None,
        frequency: str = "monthly",
        min_pairs_per_period: int = 10,
        robust_se: bool = True,
        weighted: bool = True
    ):
        """
        Initialize BMN regression.
        
        Args:
            base_period: Base period for index (default: first period)
            frequency: Time frequency ('daily', 'monthly', 'quarterly')
            min_pairs_per_period: Minimum pairs required per period
            robust_se: Use heteroscedasticity-robust standard errors
            weighted: Use weighted least squares
        """
        self.base_period = base_period
        self.frequency = frequency
        self.min_pairs_per_period = min_pairs_per_period
        self.robust_se = robust_se
        self.weighted = weighted
        self.results: Dict[str, BMNRegressionResult] = {}
        
    def fit(
        self,
        repeat_sales_df: pl.DataFrame,
        geography_level: GeographyLevel,
        geography_id: str,
        weights_df: Optional[pl.DataFrame] = None
    ) -> BMNRegressionResult:
        """
        Fit BMN regression model to repeat sales data.
        
        Args:
            repeat_sales_df: Polars DataFrame with repeat sales pairs
            geography_level: Geographic level for index
            geography_id: Geographic identifier
            weights_df: Optional DataFrame with observation weights
            
        Returns:
            BMNRegressionResult with regression output and index values
        """
        logger.info(f"Fitting BMN regression for {geography_level.value} {geography_id}")
        
        # Check if we have data
        if len(repeat_sales_df) == 0:
            raise ValueError("No repeat sales data provided")
        
        # Prepare time periods
        periods_df = self._create_time_periods(repeat_sales_df)
        
        # Create design matrix
        X, y, weights, period_mapping = self._create_design_matrix(
            repeat_sales_df,
            periods_df,
            weights_df
        )
        
        # Check for sufficient data
        if X.shape[0] < self.min_pairs_per_period:
            raise ValueError(f"Insufficient data: {X.shape[0]} pairs < {self.min_pairs_per_period} minimum")
            
        # Fit regression model
        if self.weighted and weights is not None:
            model = WLS(y, X, weights=weights)
            results = model.fit(cov_type='HC3' if self.robust_se else 'nonrobust')
        else:
            model = sm.OLS(y, X)
            results = model.fit(cov_type='HC3' if self.robust_se else 'nonrobust')
            
        # Extract coefficients and statistics
        coefficients = {}
        standard_errors = {}
        t_statistics = {}
        p_values = {}
        
        for i, period in enumerate(period_mapping):
            if i < len(results.params):  # Skip if base period
                coefficients[period] = results.params[i]
                standard_errors[period] = results.bse[i]
                t_statistics[period] = results.tvalues[i]
                p_values[period] = results.pvalues[i]
                
        # Calculate index values
        index_values = self._calculate_index_values(
            coefficients,
            standard_errors,
            periods_df,
            repeat_sales_df,
            geography_level,
            geography_id
        )
        
        # Create result object
        result = BMNRegressionResult(
            geography_level=geography_level,
            geography_id=geography_id,
            start_period=periods_df["period"].min(),
            end_period=periods_df["period"].max(),
            num_periods=len(periods_df),
            num_observations=X.shape[0],
            r_squared=results.rsquared,
            adj_r_squared=results.rsquared_adj,
            coefficients=coefficients,
            standard_errors=standard_errors,
            t_statistics=t_statistics,
            p_values=p_values,
            index_values=index_values
        )
        
        self.results[f"{geography_level.value}_{geography_id}"] = result
        return result
        
    def fit_multiple_geographies(
        self,
        repeat_sales_df: pl.DataFrame,
        geography_col: str,
        geography_level: GeographyLevel,
        weights_df: Optional[pl.DataFrame] = None,
        min_pairs: int = 30
    ) -> Dict[str, BMNRegressionResult]:
        """
        Fit BMN regression for multiple geographic areas.
        
        Args:
            repeat_sales_df: Polars DataFrame with repeat sales pairs
            geography_col: Column name for geographic identifier
            geography_level: Geographic level
            weights_df: Optional DataFrame with weights
            min_pairs: Minimum pairs required per geography
            
        Returns:
            Dictionary mapping geography ID to regression results
        """
        logger.info(f"Fitting BMN regression for multiple {geography_level.value} areas")
        
        results = {}
        
        # Get unique geographies with sufficient data
        geo_counts = repeat_sales_df.group_by(geography_col).agg(
            pl.count().alias("num_pairs")
        ).filter(pl.col("num_pairs") >= min_pairs)
        
        logger.info(f"Processing {len(geo_counts)} geographic areas")
        
        for geo_id in geo_counts[geography_col]:
            try:
                # Filter data for this geography
                geo_data = repeat_sales_df.filter(
                    pl.col(geography_col) == geo_id
                )
                
                # Filter weights if provided
                geo_weights = None
                if weights_df is not None:
                    geo_weights = weights_df.filter(
                        pl.col(geography_col) == geo_id
                    )
                    
                # Fit model
                result = self.fit(
                    geo_data,
                    geography_level,
                    str(geo_id),
                    geo_weights
                )
                
                results[str(geo_id)] = result
                
            except Exception as e:
                logger.error(f"Failed to fit model for {geo_id}: {str(e)}")
                continue
                
        logger.info(f"Successfully fitted {len(results)} models")
        return results
        
    def _create_time_periods(
        self,
        repeat_sales_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Create time period mapping based on frequency.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            
        Returns:
            DataFrame with unique time periods
        """
        # Extract all unique dates
        all_dates = pl.concat([
            repeat_sales_df.select("sale1_date"),
            repeat_sales_df.select("sale2_date").rename({"sale2_date": "sale1_date"})
        ])["sale1_date"].unique().sort()
        
        # Convert to periods based on frequency
        if self.frequency == "monthly":
            periods = all_dates.dt.truncate("1mo")
        elif self.frequency == "quarterly":
            periods = all_dates.dt.truncate("1q")
        elif self.frequency == "daily":
            periods = all_dates
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")
            
        # Create period DataFrame
        periods_df = pl.DataFrame({
            "period": periods.unique().sort(),
        }).with_row_count("period_index")
        
        # Set base period if not specified
        if self.base_period is None:
            self.base_period = periods_df["period"].min()
            
        return periods_df
        
    def _create_design_matrix(
        self,
        repeat_sales_df: pl.DataFrame,
        periods_df: pl.DataFrame,
        weights_df: Optional[pl.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Create design matrix for BMN regression.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            periods_df: DataFrame with time periods
            weights_df: Optional weights DataFrame
            
        Returns:
            Tuple of (X matrix, y vector, weights, period mapping)
        """
        # Convert dates to periods
        if self.frequency == "monthly":
            df = repeat_sales_df.with_columns([
                pl.col("sale1_date").dt.truncate("1mo").alias("period1"),
                pl.col("sale2_date").dt.truncate("1mo").alias("period2")
            ])
        elif self.frequency == "quarterly":
            df = repeat_sales_df.with_columns([
                pl.col("sale1_date").dt.truncate("1q").alias("period1"),
                pl.col("sale2_date").dt.truncate("1q").alias("period2")
            ])
        else:
            df = repeat_sales_df.with_columns([
                pl.col("sale1_date").alias("period1"),
                pl.col("sale2_date").alias("period2")
            ])
            
        # Join with period indices
        df = df.join(
            periods_df.rename({"period": "period1", "period_index": "period1_idx"}),
            on="period1"
        ).join(
            periods_df.rename({"period": "period2", "period_index": "period2_idx"}),
            on="period2"
        )
        
        # Get base period index
        base_period_df = periods_df.filter(
            pl.col("period") == self.base_period
        )
        if len(base_period_df) == 0:
            # No periods found
            return np.array([]), np.array([]), None, []
        base_idx = base_period_df["period_index"][0]
        
        # Number of periods (excluding base)
        n_periods = len(periods_df) - 1
        n_obs = len(df)
        
        # Create sparse design matrix
        # Each row has -1 for sale1 period and +1 for sale2 period
        row_indices = []
        col_indices = []
        data = []
        
        for i, row in enumerate(df.iter_rows(named=True)):
            p1_idx = row["period1_idx"]
            p2_idx = row["period2_idx"]
            
            # Adjust for base period
            if p1_idx != base_idx:
                col_idx = p1_idx if p1_idx < base_idx else p1_idx - 1
                row_indices.append(i)
                col_indices.append(col_idx)
                data.append(-1)
                
            if p2_idx != base_idx:
                col_idx = p2_idx if p2_idx < base_idx else p2_idx - 1
                row_indices.append(i)
                col_indices.append(col_idx)
                data.append(1)
                
        # Create sparse matrix
        X_sparse = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_obs, n_periods)
        )
        
        # Convert to dense for statsmodels
        X = X_sparse.toarray()
        
        # Extract y values (log price ratios)
        y = df["log_price_ratio"].to_numpy()
        
        # Extract weights if provided
        weights = None
        if weights_df is not None:
            # Merge weights with repeat sales
            df_with_weights = df.join(
                weights_df.select(["pair_id", "weight"]),
                on="pair_id",
                how="left"
            )
            weights = df_with_weights["weight"].fill_null(1.0).to_numpy()
            
        # Create period mapping (excluding base period)
        period_mapping = []
        sorted_periods = periods_df.sort("period")["period"].to_list()
        for period in sorted_periods:
            if period != self.base_period:
                # Handle both date objects and strings
                if hasattr(period, 'strftime'):
                    period_mapping.append(period.strftime("%Y-%m-%d"))
                else:
                    period_mapping.append(str(period))
                
        return X, y, weights, period_mapping
        
    def _calculate_index_values(
        self,
        coefficients: Dict[str, float],
        standard_errors: Dict[str, float],
        periods_df: pl.DataFrame,
        repeat_sales_df: pl.DataFrame,
        geography_level: GeographyLevel,
        geography_id: str
    ) -> List[IndexValue]:
        """
        Calculate index values from regression coefficients.
        
        Args:
            coefficients: Period coefficients
            standard_errors: Standard errors
            periods_df: DataFrame with periods
            repeat_sales_df: Original repeat sales data
            geography_level: Geographic level
            geography_id: Geographic identifier
            
        Returns:
            List of IndexValue objects
        """
        index_values = []
        
        # Calculate statistics by period
        period_stats = repeat_sales_df.with_columns([
            pl.when(self.frequency == "monthly")
            .then(pl.col("sale2_date").dt.truncate("1mo"))
            .when(self.frequency == "quarterly")
            .then(pl.col("sale2_date").dt.truncate("1q"))
            .otherwise(pl.col("sale2_date"))
            .alias("period")
        ]).group_by("period").agg([
            pl.count().alias("num_pairs"),
            pl.col("property_id").n_unique().alias("num_properties"),
            pl.col("sale2_price").median().alias("median_price")
        ])
        
        # Create index values for each period
        sorted_periods = periods_df.sort("period")["period"].to_list()
        for period in sorted_periods:
            # Handle both date objects and strings
            if hasattr(period, 'strftime'):
                period_str = period.strftime("%Y-%m-%d")
            else:
                period_str = str(period)
            
            # Get statistics for this period
            stats = period_stats.filter(pl.col("period") == period)
            num_pairs = stats["num_pairs"][0] if len(stats) > 0 else 0
            num_properties = stats["num_properties"][0] if len(stats) > 0 else 0
            median_price = stats["median_price"][0] if len(stats) > 0 else None
            
            # Calculate index value
            if period == self.base_period:
                index_value = 100.0
                se = 0.0
            elif period_str in coefficients:
                # Index = 100 * exp(coefficient)
                index_value = 100.0 * np.exp(coefficients[period_str])
                se = standard_errors.get(period_str, 0.0)
            else:
                continue
                
            # Calculate confidence intervals
            if se > 0:
                # 95% confidence interval
                z_score = 1.96
                lower = 100.0 * np.exp(coefficients[period_str] - z_score * se)
                upper = 100.0 * np.exp(coefficients[period_str] + z_score * se)
            else:
                lower = index_value
                upper = index_value
                
            index_values.append(IndexValue(
                geography_level=geography_level,
                geography_id=geography_id,
                period=period,
                index_value=index_value,
                num_pairs=num_pairs,
                num_properties=num_properties,
                median_price=median_price,
                standard_error=se * index_value / 100.0,  # Convert to index scale
                confidence_lower=lower,
                confidence_upper=upper
            ))
            
        return index_values
        
    def calculate_returns(
        self,
        index_values: List[IndexValue],
        return_type: str = "log"
    ) -> pl.DataFrame:
        """
        Calculate returns from index values.
        
        Args:
            index_values: List of IndexValue objects
            return_type: Type of return ('log' or 'simple')
            
        Returns:
            Polars DataFrame with returns
        """
        # Convert to DataFrame
        index_df = pl.DataFrame([
            {
                "period": iv.period,
                "index_value": iv.index_value,
                "geography_id": iv.geography_id
            }
            for iv in index_values
        ]).sort("period")
        
        # Calculate returns
        if return_type == "log":
            index_df = index_df.with_columns([
                (pl.col("index_value").log() - pl.col("index_value").shift(1).log())
                .alias("return")
            ])
        else:
            index_df = index_df.with_columns([
                ((pl.col("index_value") / pl.col("index_value").shift(1)) - 1)
                .alias("return")
            ])
            
        # Add annualized returns
        if self.frequency == "monthly":
            periods_per_year = 12
        elif self.frequency == "quarterly":
            periods_per_year = 4
        else:
            periods_per_year = 365
            
        index_df = index_df.with_columns([
            (pl.col("return") * periods_per_year).alias("annualized_return")
        ])
        
        return index_df
        
    def calculate_volatility(
        self,
        index_values: List[IndexValue],
        window: int = 12
    ) -> pl.DataFrame:
        """
        Calculate rolling volatility of index returns.
        
        Args:
            index_values: List of IndexValue objects
            window: Rolling window size
            
        Returns:
            Polars DataFrame with volatility measures
        """
        # Calculate returns
        returns_df = self.calculate_returns(index_values, "log")
        
        # Calculate rolling volatility
        volatility_df = returns_df.with_columns([
            pl.col("return").rolling_std(window).alias("volatility"),
            pl.col("return").rolling_mean(window).alias("mean_return")
        ])
        
        # Annualize volatility
        if self.frequency == "monthly":
            annualization_factor = np.sqrt(12)
        elif self.frequency == "quarterly":
            annualization_factor = np.sqrt(4)
        else:
            annualization_factor = np.sqrt(365)
            
        volatility_df = volatility_df.with_columns([
            (pl.col("volatility") * annualization_factor).alias("annualized_volatility")
        ])
        
        return volatility_df