"""
Unit tests for BMN regression implementation.

Tests the Bailey-Muth-Nourse regression method for calculating
price indices from repeat sales pairs.
"""

import pytest
from datetime import date, timedelta
import polars as pl
import numpy as np
from unittest.mock import Mock, patch

from rsai.src.index.bmn_regression import BMNRegression
from rsai.src.data.models import (
    GeographyLevel,
    BMNRegressionResult,
    IndexValue
)


class TestBMNRegression:
    """Test BMNRegression class."""
    
    def test_initialization(self):
        """Test BMNRegression initialization."""
        bmn = BMNRegression(
            base_period=date(2020, 1, 1),
            frequency="monthly",
            min_pairs_per_period=10,
            robust_se=True,
            weighted=True
        )
        assert bmn.base_period == date(2020, 1, 1)
        assert bmn.frequency == "monthly"
        assert bmn.min_pairs_per_period == 10
        assert bmn.robust_se is True
        assert bmn.weighted is True
        assert len(bmn.results) == 0
    
    def test_create_time_periods_monthly(self, sample_repeat_sales_df):
        """Test time period creation for monthly frequency."""
        bmn = BMNRegression(frequency="monthly")
        
        periods_df = bmn._create_time_periods(sample_repeat_sales_df)
        
        assert isinstance(periods_df, pl.DataFrame)
        assert "period" in periods_df.columns
        assert "period_index" in periods_df.columns
        
        # Check that periods are monthly
        periods = periods_df["period"].to_list()
        for i in range(1, len(periods)):
            # All periods should be first of month
            assert periods[i].day == 1
    
    def test_create_time_periods_quarterly(self, sample_repeat_sales_df):
        """Test time period creation for quarterly frequency."""
        bmn = BMNRegression(frequency="quarterly")
        
        periods_df = bmn._create_time_periods(sample_repeat_sales_df)
        
        assert isinstance(periods_df, pl.DataFrame)
        # Periods should be quarterly
        periods = periods_df["period"].to_list()
        for period in periods:
            assert period.month in [1, 4, 7, 10]
            assert period.day == 1
    
    def test_create_design_matrix(self, sample_repeat_sales_df):
        """Test design matrix creation."""
        bmn = BMNRegression(frequency="monthly")
        
        # Create time periods
        periods_df = bmn._create_time_periods(sample_repeat_sales_df)
        
        # Create design matrix
        X, y, weights, period_mapping = bmn._create_design_matrix(
            sample_repeat_sales_df,
            periods_df
        )
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == len(sample_repeat_sales_df)  # Rows = observations
        assert X.shape[1] == len(periods_df) - 1  # Cols = periods - 1 (base period)
        assert len(y) == len(sample_repeat_sales_df)
        assert len(period_mapping) == X.shape[1]
        
        # Check that y contains log price ratios
        expected_y = sample_repeat_sales_df["log_price_ratio"].to_numpy()
        np.testing.assert_array_almost_equal(y, expected_y)
    
    def test_create_design_matrix_with_weights(self, sample_repeat_sales_df):
        """Test design matrix creation with weights."""
        bmn = BMNRegression(frequency="monthly", weighted=True)
        
        # Create time periods
        periods_df = bmn._create_time_periods(sample_repeat_sales_df)
        
        # Create weights DataFrame
        weights_df = sample_repeat_sales_df.select([
            pl.col("pair_id"),
            pl.lit(1.0).alias("weight")  # Equal weights for simplicity
        ])
        
        X, y, weights, period_mapping = bmn._create_design_matrix(
            sample_repeat_sales_df,
            periods_df,
            weights_df
        )
        
        assert weights is not None
        assert len(weights) == len(sample_repeat_sales_df)
        assert all(w == 1.0 for w in weights)
    
    def test_calculate_index_values(self, sample_repeat_sales_df):
        """Test index value calculation from coefficients."""
        bmn = BMNRegression(
            base_period=date(2020, 1, 1),
            frequency="monthly"
        )
        
        # Create mock coefficients
        periods_df = bmn._create_time_periods(sample_repeat_sales_df)
        coefficients = {}
        standard_errors = {}
        
        for i, period in enumerate(periods_df["period"][1:]):  # Skip base period
            period_str = period.strftime("%Y-%m-%d")
            # Simulate 5% annual growth
            monthly_growth = 0.05 / 12
            coefficients[period_str] = np.log(1 + monthly_growth * (i + 1))
            standard_errors[period_str] = 0.01
        
        index_values = bmn._calculate_index_values(
            coefficients,
            standard_errors,
            periods_df,
            sample_repeat_sales_df,
            GeographyLevel.COUNTY,
            "06037"
        )
        
        assert isinstance(index_values, list)
        assert len(index_values) == len(periods_df)
        
        # Check base period
        base_value = next(iv for iv in index_values if iv.period == bmn.base_period)
        assert base_value.index_value == 100.0
        assert base_value.standard_error == 0.0
        
        # Check other periods
        for iv in index_values[1:]:
            assert iv.index_value > 100.0  # Should show growth
            assert iv.standard_error > 0
            assert iv.confidence_lower < iv.index_value < iv.confidence_upper
    
    def test_fit_basic(self, sample_repeat_sales_df):
        """Test basic model fitting."""
        bmn = BMNRegression(
            frequency="monthly",
            min_pairs_per_period=5
        )
        
        # Mock the regression results
        with patch('statsmodels.api.OLS') as mock_ols:
            mock_results = Mock()
            mock_results.params = np.random.randn(10)
            mock_results.bse = np.abs(np.random.randn(10)) * 0.01
            mock_results.tvalues = mock_results.params / mock_results.bse
            mock_results.pvalues = np.abs(np.random.randn(10)) * 0.1
            mock_results.rsquared = 0.85
            mock_results.rsquared_adj = 0.84
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_results
            mock_ols.return_value = mock_model
            
            result = bmn.fit(
                sample_repeat_sales_df,
                GeographyLevel.COUNTY,
                "06037"
            )
        
        assert isinstance(result, BMNRegressionResult)
        assert result.geography_level == GeographyLevel.COUNTY
        assert result.geography_id == "06037"
        assert result.r_squared == 0.85
        assert len(result.index_values) > 0
    
    def test_fit_insufficient_data(self, sample_repeat_sales_df):
        """Test error handling with insufficient data."""
        bmn = BMNRegression(min_pairs_per_period=1000)  # Very high threshold
        
        with pytest.raises(ValueError, match="Insufficient data"):
            bmn.fit(
                sample_repeat_sales_df.head(5),  # Too few observations
                GeographyLevel.COUNTY,
                "06037"
            )
    
    def test_fit_multiple_geographies(self, sample_repeat_sales_df):
        """Test fitting models for multiple geographic areas."""
        # Add another county to the data
        df_county1 = sample_repeat_sales_df.with_columns([
            pl.lit("06037").alias("county_fips")
        ])
        df_county2 = sample_repeat_sales_df.with_columns([
            pl.lit("06059").alias("county_fips"),
            pl.col("pair_id").str.replace("P", "Q").alias("pair_id")  # Different IDs
        ])
        
        combined_df = pl.concat([df_county1, df_county2])
        
        bmn = BMNRegression(min_pairs_per_period=5)
        
        # Mock regression for speed
        with patch.object(bmn, 'fit') as mock_fit:
            mock_fit.return_value = Mock(spec=BMNRegressionResult)
            
            results = bmn.fit_multiple_geographies(
                combined_df,
                "county_fips",
                GeographyLevel.COUNTY,
                min_pairs=10
            )
        
        # Should attempt to fit for counties with enough data
        assert mock_fit.call_count >= 1
    
    def test_calculate_returns_log(self, sample_index_values):
        """Test log return calculation."""
        bmn = BMNRegression()
        
        returns_df = bmn.calculate_returns(sample_index_values, return_type="log")
        
        assert isinstance(returns_df, pl.DataFrame)
        assert "return" in returns_df.columns
        assert "annualized_return" in returns_df.columns
        
        # Check that first return is null (no previous period)
        assert returns_df["return"][0] is None
        
        # Check return calculation
        for i in range(1, min(5, len(returns_df))):
            expected_return = np.log(
                sample_index_values[i].index_value / 
                sample_index_values[i-1].index_value
            )
            assert returns_df["return"][i] == pytest.approx(expected_return, rel=1e-6)
    
    def test_calculate_returns_simple(self, sample_index_values):
        """Test simple return calculation."""
        bmn = BMNRegression()
        
        returns_df = bmn.calculate_returns(sample_index_values, return_type="simple")
        
        assert isinstance(returns_df, pl.DataFrame)
        assert "return" in returns_df.columns
        
        # Check return calculation
        for i in range(1, min(5, len(returns_df))):
            expected_return = (
                sample_index_values[i].index_value / 
                sample_index_values[i-1].index_value - 1
            )
            assert returns_df["return"][i] == pytest.approx(expected_return, rel=1e-6)
    
    def test_calculate_volatility(self, sample_index_values):
        """Test volatility calculation."""
        bmn = BMNRegression(frequency="monthly")
        
        volatility_df = bmn.calculate_volatility(sample_index_values, window=12)
        
        assert isinstance(volatility_df, pl.DataFrame)
        assert "volatility" in volatility_df.columns
        assert "mean_return" in volatility_df.columns
        assert "annualized_volatility" in volatility_df.columns
        
        # Check annualization factor
        # Monthly data should be annualized by sqrt(12)
        non_null_vols = volatility_df.filter(
            pl.col("volatility").is_not_null()
        )
        if len(non_null_vols) > 0:
            ratio = (
                non_null_vols["annualized_volatility"][0] / 
                non_null_vols["volatility"][0]
            )
            assert ratio == pytest.approx(np.sqrt(12), rel=1e-6)
    
    def test_weighted_regression(self, sample_repeat_sales_df):
        """Test weighted least squares regression."""
        bmn = BMNRegression(weighted=True)
        
        # Create weights
        weights_df = sample_repeat_sales_df.select([
            pl.col("pair_id"),
            pl.lit(1.0).alias("weight")
        ])
        
        # Mock WLS
        with patch('rsai.src.index.bmn_regression.WLS') as mock_wls:
            mock_results = Mock()
            mock_results.params = np.random.randn(10)
            mock_results.bse = np.abs(np.random.randn(10)) * 0.01
            mock_results.tvalues = mock_results.params / mock_results.bse
            mock_results.pvalues = np.abs(np.random.randn(10)) * 0.1
            mock_results.rsquared = 0.85
            mock_results.rsquared_adj = 0.84
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_results
            mock_wls.return_value = mock_model
            
            result = bmn.fit(
                sample_repeat_sales_df,
                GeographyLevel.COUNTY,
                "06037",
                weights_df
            )
        
        # Should have called WLS
        assert mock_wls.called
    
    def test_robust_standard_errors(self, sample_repeat_sales_df):
        """Test robust standard error calculation."""
        bmn = BMNRegression(robust_se=True)
        
        with patch('statsmodels.api.OLS') as mock_ols:
            mock_results = Mock()
            mock_results.params = np.random.randn(10)
            mock_results.bse = np.abs(np.random.randn(10)) * 0.01
            mock_results.tvalues = mock_results.params / mock_results.bse
            mock_results.pvalues = np.abs(np.random.randn(10)) * 0.1
            mock_results.rsquared = 0.85
            mock_results.rsquared_adj = 0.84
            
            mock_model = Mock()
            mock_model.fit.return_value = mock_results
            mock_ols.return_value = mock_model
            
            result = bmn.fit(
                sample_repeat_sales_df,
                GeographyLevel.COUNTY,
                "06037"
            )
            
            # Should have called fit with HC3 covariance
            mock_model.fit.assert_called_with(cov_type='HC3')
    
    def test_period_mapping_consistency(self, sample_repeat_sales_df):
        """Test that period mapping is consistent with design matrix."""
        bmn = BMNRegression(frequency="monthly")
        
        periods_df = bmn._create_time_periods(sample_repeat_sales_df)
        X, y, weights, period_mapping = bmn._create_design_matrix(
            sample_repeat_sales_df,
            periods_df
        )
        
        # Period mapping should have one entry per column in X
        assert len(period_mapping) == X.shape[1]
        
        # All periods except base should be in mapping
        base_period = periods_df["period"].min()
        non_base_periods = periods_df.filter(
            pl.col("period") != base_period
        )["period"]
        
        assert len(period_mapping) == len(non_base_periods)