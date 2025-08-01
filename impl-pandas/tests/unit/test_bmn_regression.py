"""Unit tests for BMN regression implementation."""

import pytest
import numpy as np
import pandas as pd
from scipy import sparse

from hpi_fhfa.models.bmn_regression import (
    BMNRegressor, 
    BMNResults,
    calculate_index_from_coefficients
)


class TestBMNResults:
    """Test BMNResults dataclass functionality."""
    
    def test_bmn_results_creation(self):
        coefficients = np.array([0.0, 0.05, 0.12, 0.18])
        results = BMNResults(
            coefficients=coefficients,
            n_observations=100,
            n_parameters=4
        )
        
        assert len(results.coefficients) == 4
        assert results.n_observations == 100
        assert results.n_parameters == 4
        
    def test_get_appreciation(self):
        coefficients = np.array([0.0, 0.05, 0.12, 0.18])
        results = BMNResults(coefficients=coefficients)
        
        # Appreciation from period 0 to 1
        assert results.get_appreciation(1, 0) == 0.05
        
        # Appreciation from period 1 to 3
        assert results.get_appreciation(3, 1) == 0.13
        
        # Negative appreciation
        assert results.get_appreciation(0, 2) == -0.12
        
    def test_to_dataframe(self):
        coefficients = np.array([0.0, 0.05, 0.12])
        std_errors = np.array([0.0, 0.01, 0.015])
        time_periods = pd.to_datetime(['2019-01-01', '2020-01-01', '2021-01-01'])
        
        results = BMNResults(
            coefficients=coefficients,
            std_errors=std_errors,
            time_periods=time_periods
        )
        
        df = results.to_dataframe()
        
        assert len(df) == 3
        assert 'coefficient' in df.columns
        assert 'std_error' in df.columns
        assert 't_stat' in df.columns
        assert df['t_stat'].iloc[1] == 5.0  # 0.05 / 0.01
        

class TestBMNRegressor:
    """Test BMN regression estimator."""
    
    @pytest.fixture
    def simple_repeat_sales_data(self):
        """Create simple repeat sales data for testing."""
        # 4 time periods, 10 properties
        data = {
            'property_id': ['P1', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'],
            'sale1_period': [0, 1, 0, 1, 1, 2, 0, 1, 2, 0],
            'sale2_period': [1, 2, 2, 3, 2, 3, 3, 3, 3, 1],
            'price_relative': [0.05, 0.06, 0.11, 0.08, 0.06, 0.05, 0.18, 0.08, 0.05, 0.05]
        }
        return pd.DataFrame(data)
    
    def test_basic_fit(self, simple_repeat_sales_data):
        regressor = BMNRegressor(use_sparse=False)
        results = regressor.fit(simple_repeat_sales_data)
        
        assert results.n_observations == 10
        assert results.n_parameters == 4  # 4 periods
        assert len(results.coefficients) == 4
        
        # First coefficient should be 0 (normalized)
        assert results.coefficients[0] == 0.0
        
        # Coefficients should be increasing (positive appreciation)
        assert all(np.diff(results.coefficients) >= 0)
        
    def test_sparse_fit(self, simple_repeat_sales_data):
        regressor = BMNRegressor(use_sparse=True)
        results = regressor.fit(simple_repeat_sales_data)
        
        assert results.n_observations == 10
        assert len(results.coefficients) == 4
        
        # Results should be similar to dense version
        dense_regressor = BMNRegressor(use_sparse=False)
        dense_results = dense_regressor.fit(simple_repeat_sales_data)
        
        np.testing.assert_allclose(
            results.coefficients, 
            dense_results.coefficients,
            rtol=1e-6
        )
        
    def test_design_matrix_construction(self):
        regressor = BMNRegressor()
        
        # Simple case: 3 periods, 2 observations
        period1 = np.array([0, 1])
        period2 = np.array([1, 2])
        
        # Dense matrix
        X_dense = regressor._create_design_matrix(
            period1, period2, n_periods=3, 
            use_sparse=False, normalize_first=True
        )
        
        # Expected shape: 2 observations, 2 parameters (period 0 dropped)
        assert X_dense.shape == (2, 2)
        
        # First row: sale from period 0 to 1 (only period 1 gets +1)
        np.testing.assert_array_equal(X_dense[0], [1, 0])
        
        # Second row: sale from period 1 to 2
        np.testing.assert_array_equal(X_dense[1], [-1, 1])
        
    def test_design_matrix_sparse(self):
        regressor = BMNRegressor()
        
        period1 = np.array([0, 1, 0, 2])
        period2 = np.array([2, 3, 1, 3])
        
        X_sparse = regressor._create_design_matrix(
            period1, period2, n_periods=4,
            use_sparse=True, normalize_first=True
        )
        
        assert sparse.issparse(X_sparse)
        assert X_sparse.shape == (4, 3)  # 4 obs, 3 params (first dropped)
        
        # Convert to dense for testing
        X_dense = X_sparse.toarray()
        
        # Verify structure
        expected = np.array([
            [0, 1, 0],   # period 0 to 2
            [-1, 0, 1],  # period 1 to 3
            [1, 0, 0],   # period 0 to 1
            [0, -1, 1]   # period 2 to 3
        ])
        
        np.testing.assert_array_equal(X_dense, expected)
        
    def test_no_normalization(self, simple_repeat_sales_data):
        regressor = BMNRegressor()
        results = regressor.fit(
            simple_repeat_sales_data,
            normalize_first=False
        )
        
        # Should have coefficients for all periods
        assert len(results.coefficients) == 4
        
        # Coefficients not normalized to 0
        assert results.coefficients[0] != 0.0
        
    def test_with_dates(self):
        # Create data with actual dates
        dates = pd.date_range('2019-01-01', periods=4, freq='YE')
        
        data = pd.DataFrame({
            'sale1_date': [dates[0], dates[1], dates[0]],
            'sale2_date': [dates[1], dates[2], dates[3]],
            'sale1_period': [0, 1, 0],
            'sale2_period': [1, 2, 3],
            'price_relative': [0.05, 0.06, 0.17]
        })
        
        regressor = BMNRegressor()
        results = regressor.fit(data)
        
        assert results.time_periods is not None
        assert len(results.time_periods) == 4
        
    def test_statistics_calculation(self, simple_repeat_sales_data):
        regressor = BMNRegressor(calculate_std_errors=True)
        results = regressor.fit(simple_repeat_sales_data)
        
        assert results.std_errors is not None
        assert len(results.std_errors) == len(results.coefficients)
        assert results.r_squared is not None
        assert 0 <= results.r_squared <= 1
        assert results.residuals is not None
        assert len(results.residuals) == results.n_observations
        
    def test_perfect_fit(self):
        # Create data that should fit perfectly
        # If p_01 = 0.1 and p_12 = 0.1, then p_02 = 0.2
        data = pd.DataFrame({
            'sale1_period': [0, 1, 0],
            'sale2_period': [1, 2, 2],
            'price_relative': [0.1, 0.1, 0.2]
        })
        
        regressor = BMNRegressor(calculate_std_errors=True)
        results = regressor.fit(data)
        
        # Should have very high R-squared
        assert results.r_squared > 0.99
        
        # Coefficients should be approximately [0, 0.1, 0.2]
        expected = np.array([0.0, 0.1, 0.2])
        np.testing.assert_allclose(results.coefficients, expected, atol=1e-6)


class TestIndexCalculation:
    """Test index calculation from coefficients."""
    
    def test_basic_index_calculation(self):
        # Log coefficients representing 5% appreciation each period
        coefficients = np.array([0.0, 0.05, 0.10, 0.15])
        
        index = calculate_index_from_coefficients(coefficients)
        
        # Base period should be 100
        assert index[0] == 100.0
        
        # Each period should be ~5% higher
        expected = 100 * np.exp(coefficients)
        np.testing.assert_allclose(index, expected)
        
    def test_different_base_period(self):
        coefficients = np.array([0.0, 0.05, 0.10, 0.15])
        
        # Set period 2 as base
        index = calculate_index_from_coefficients(
            coefficients, 
            base_period=2,
            base_value=100.0
        )
        
        assert index[2] == 100.0
        
        # Period 0 should be lower than 100
        assert index[0] < 100.0
        
        # Period 3 should be higher than 100
        assert index[3] > 100.0
        
    def test_different_base_value(self):
        coefficients = np.array([0.0, 0.05, 0.10])
        
        index = calculate_index_from_coefficients(
            coefficients,
            base_value=1000.0
        )
        
        assert index[0] == 1000.0
        assert index[1] > 1000.0
        

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_observation(self):
        data = pd.DataFrame({
            'sale1_period': [0],
            'sale2_period': [1],
            'price_relative': [0.05]
        })
        
        regressor = BMNRegressor()
        results = regressor.fit(data)
        
        assert results.n_observations == 1
        assert len(results.coefficients) == 2
        
    def test_missing_periods(self):
        # Period 2 is missing from the data
        data = pd.DataFrame({
            'sale1_period': [0, 1, 3],
            'sale2_period': [1, 3, 4],
            'price_relative': [0.05, 0.15, 0.06]
        })
        
        regressor = BMNRegressor()
        results = regressor.fit(data)
        
        # Should estimate coefficients for periods present in data (0, 1, 3, 4)
        assert len(results.coefficients) == 4
        
    def test_large_sparse_problem(self):
        # Simulate large dataset
        np.random.seed(42)
        n_obs = 10000
        n_periods = 100
        
        period1 = np.random.randint(0, n_periods-1, n_obs)
        period2 = period1 + np.random.randint(1, 5, n_obs)
        period2 = np.clip(period2, 0, n_periods-1)
        
        price_relative = np.random.normal(0.05, 0.02, n_obs)
        
        data = pd.DataFrame({
            'sale1_period': period1,
            'sale2_period': period2,
            'price_relative': price_relative
        })
        
        # Remove any same-period observations
        data = data[data['sale1_period'] != data['sale2_period']]
        
        regressor = BMNRegressor(use_sparse=True, calculate_std_errors=False)
        results = regressor.fit(data)
        
        assert len(results.coefficients) == n_periods
        assert results.n_observations == len(data)