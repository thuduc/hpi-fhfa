"""Unit tests for BMN regression."""

import pytest
import polars as pl
import numpy as np
from datetime import date
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.hpi_fhfa.models.bmn_regression import BMNRegression
from src.hpi_fhfa.utils.exceptions import InsufficientDataError, ProcessingError


class TestBMNRegression:
    """Test BMN regression implementation."""
    
    def test_initialization(self, time_periods):
        """Test BMN regression initialization."""
        bmn = BMNRegression(time_periods)
        
        assert bmn.time_periods == sorted(time_periods)
        assert bmn.n_periods == len(time_periods)
        assert len(bmn.period_to_idx) == len(time_periods)
        assert bmn.coefficients is None
        assert bmn.residuals is None
    
    def test_dummy_matrix_creation(self):
        """Test sparse dummy matrix creation."""
        # Create simple controlled data
        df = pl.DataFrame({
            "property_id": ["P1", "P2", "P3", "P4"],
            "transaction_date": [date(2016, 1, 1), date(2017, 1, 1), date(2018, 1, 1), date(2017, 6, 1)],
            "prev_transaction_date": [date(2015, 1, 1), date(2015, 1, 1), date(2016, 1, 1), date(2016, 1, 1)],
            "log_price_diff": [0.1, 0.2, 0.15, 0.12]
        })
        
        bmn = BMNRegression([2015, 2016, 2017, 2018])
        X, y = bmn.create_dummy_matrix(df)
        
        # Check matrix dimensions
        assert X.shape[0] == 4  # 4 observations
        assert X.shape[1] == 4  # 4 time periods
        
        # Check y matches log price differences
        np.testing.assert_array_almost_equal(y, [0.1, 0.2, 0.15, 0.12])
        
        # Check matrix has correct number of non-zero entries
        assert X.nnz == 8  # 4 rows * 2 entries per row
        
        # Verify specific entries
        # Row 0: 2015 (-1) to 2016 (+1)
        assert X[0, 0] == -1.0
        assert X[0, 1] == 1.0
        
        # Row 1: 2015 (-1) to 2017 (+1)
        assert X[1, 0] == -1.0
        assert X[1, 2] == 1.0
    
    def test_fit_simple_case(self):
        """Test fitting with a simple, known case."""
        # Create more observations to meet minimum requirement
        df = pl.DataFrame({
            "property_id": ["P1", "P2", "P3", "P4", "P5"],
            "transaction_date": [
                date(2016, 1, 1),
                date(2017, 1, 1),
                date(2016, 6, 1),
                date(2017, 6, 1),
                date(2017, 3, 1)
            ],
            "prev_transaction_date": [
                date(2015, 1, 1),
                date(2015, 6, 1),
                date(2014, 1, 1),
                date(2015, 1, 1),
                date(2016, 1, 1)
            ],
            "transaction_price": [
                110000.0,
                121000.0,
                133100.0,
                146410.0,
                121000.0
            ],
            "prev_transaction_price": [
                100000.0,
                100000.0,
                110000.0,
                110000.0,
                100000.0
            ],
            "log_price_diff": [
                np.log(1.1),
                np.log(1.21),
                np.log(1.21),
                np.log(1.331),
                np.log(1.21)
            ],
            "census_tract": ["06037000100"] * 5,
            "cbsa_code": ["31080"] * 5,
            "distance_to_cbd": [5.0] * 5
        })
        
        bmn = BMNRegression([2014, 2015, 2016, 2017])
        coeffs = bmn.fit(df, normalize_first_period=True)
        
        # First period should be 0
        assert coeffs[0] == 0.0
        
        # With normalized first period, coefficients should generally increase
        assert len(coeffs) == 4  # One for each period
        
        # Later periods should have positive coefficients (appreciation)
        assert coeffs[3] > 0  # 2017 should be positive
    
    def test_fit_with_ols(self, sample_repeat_sales, time_periods):
        """Test fitting with OLS method."""
        bmn = BMNRegression(time_periods)
        
        # Filter to test periods
        df = sample_repeat_sales.with_columns([
            pl.col("transaction_date").dt.year().alias("sale_year"),
            pl.col("prev_transaction_date").dt.year().alias("prev_year")
        ]).filter(
            (pl.col("sale_year").is_in(time_periods)) &
            (pl.col("prev_year").is_in(time_periods))
        )
        
        coeffs = bmn.fit(df, method="ols")
        
        assert len(coeffs) == len(time_periods)
        assert coeffs[0] == 0.0  # First period normalized
        assert bmn.residuals is not None
        assert len(bmn.residuals) == len(df)
    
    def test_fit_insufficient_data(self, time_periods):
        """Test fitting with insufficient data."""
        # Create DataFrame with fewer observations than periods
        df = pl.DataFrame({
            "property_id": ["P1"],
            "transaction_date": [date(2016, 1, 1)],
            "prev_transaction_date": [date(2015, 1, 1)],
            "transaction_price": [110000],
            "prev_transaction_price": [100000],
            "log_price_diff": [np.log(1.1)]
        })
        
        bmn = BMNRegression(time_periods)
        
        with pytest.raises(InsufficientDataError):
            bmn.fit(df)
    
    def test_calculate_appreciation(self, sample_repeat_sales, time_periods):
        """Test appreciation calculation."""
        bmn = BMNRegression(time_periods)
        
        # Filter and fit
        df = sample_repeat_sales.with_columns([
            pl.col("transaction_date").dt.year().alias("sale_year"),
            pl.col("prev_transaction_date").dt.year().alias("prev_year")
        ]).filter(
            (pl.col("sale_year").is_in(time_periods)) &
            (pl.col("prev_year").is_in(time_periods))
        )
        
        bmn.fit(df)
        
        # Calculate appreciation between consecutive periods
        appreciation = bmn.calculate_appreciation(2016, 2015)
        
        # Should be positive for normal market
        assert isinstance(appreciation, float)
        
        # Test error cases
        with pytest.raises(ValueError, match="not in fitted periods"):
            bmn.calculate_appreciation(2025, 2024)
    
    def test_get_index_values(self, sample_repeat_sales, time_periods):
        """Test index value calculation."""
        bmn = BMNRegression(time_periods)
        
        # Filter and fit
        df = sample_repeat_sales.with_columns([
            pl.col("transaction_date").dt.year().alias("sale_year"),
            pl.col("prev_transaction_date").dt.year().alias("prev_year")
        ]).filter(
            (pl.col("sale_year").is_in(time_periods)) &
            (pl.col("prev_year").is_in(time_periods))
        )
        
        bmn.fit(df)
        
        # Get index values
        index_values = bmn.get_index_values()
        
        assert len(index_values) == len(time_periods)
        assert index_values[2015] == 100.0  # Base period
        
        # Test with different base period
        index_values_2016 = bmn.get_index_values(base_period=2016)
        assert index_values_2016[2016] == 100.0
    
    def test_get_diagnostics(self, sample_repeat_sales, time_periods):
        """Test regression diagnostics."""
        bmn = BMNRegression(time_periods)
        
        # Before fitting
        diagnostics = bmn.get_diagnostics()
        assert diagnostics == {}
        
        # After fitting
        df = sample_repeat_sales.with_columns([
            pl.col("transaction_date").dt.year().alias("sale_year"),
            pl.col("prev_transaction_date").dt.year().alias("prev_year")
        ]).filter(
            (pl.col("sale_year").is_in(time_periods)) &
            (pl.col("prev_year").is_in(time_periods))
        )
        
        bmn.fit(df)
        diagnostics = bmn.get_diagnostics()
        
        assert "n_observations" in diagnostics
        assert "n_periods" in diagnostics
        assert "rmse" in diagnostics
        assert "mae" in diagnostics
        assert diagnostics["n_observations"] == len(df)
        assert diagnostics["n_periods"] == len(time_periods)