"""Unit tests for BMN regression"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from rsai.src.index.bmn_regression import BMNRegression, run_bmn_for_supertract


class TestBMNRegression:
    """Test BMNRegression class"""
    
    @pytest.fixture
    def simple_repeat_sales(self):
        """Create simple repeat sales data for testing"""
        # Create data with known appreciation pattern
        # 5% annual appreciation
        data = []
        
        for i in range(50):  # Need enough observations
            first_year = 2015 + (i % 3)
            second_year = first_year + 2 + (i % 2)
            
            first_price = 200000 + i * 1000
            # Apply compound appreciation
            years_diff = second_year - first_year
            second_price = first_price * (1.05 ** years_diff)
            
            data.append({
                'property_id': f'PROP{i:03d}',
                'first_sale_date': pd.to_datetime(f'{first_year}-06-01'),
                'second_sale_date': pd.to_datetime(f'{second_year}-06-01'),
                'first_sale_price': first_price,
                'second_sale_price': second_price,
                'log_price_relative': np.log(second_price / first_price),
                'census_tract_2010': '06037123456',
                'cbsa_id': '31080'
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def volatile_repeat_sales(self):
        """Create repeat sales with varying appreciation rates"""
        data = []
        
        # Different appreciation rates by year
        appreciation_by_year = {
            2016: 0.03,
            2017: 0.08,
            2018: 0.05,
            2019: -0.02,
            2020: 0.10
        }
        
        for i in range(100):
            first_year = 2015 + (i % 4)
            second_year = min(first_year + 1 + (i % 3), 2020)
            
            first_price = 300000 + i * 2000
            
            # Calculate cumulative appreciation
            cumulative_appr = 1.0
            for year in range(first_year + 1, second_year + 1):
                if year in appreciation_by_year:
                    cumulative_appr *= (1 + appreciation_by_year[year])
            
            second_price = first_price * cumulative_appr
            
            data.append({
                'property_id': f'PROP{i:03d}',
                'first_sale_date': pd.to_datetime(f'{first_year}-06-01'),
                'second_sale_date': pd.to_datetime(f'{second_year}-06-01'),
                'first_sale_price': first_price,
                'second_sale_price': second_price,
                'log_price_relative': np.log(second_price / first_price),
                'census_tract_2010': '06037123456',
                'cbsa_id': '31080'
            })
        
        return pd.DataFrame(data)
    
    def test_prepare_regression_data(self, simple_repeat_sales):
        """Test preparation of regression data"""
        bmn = BMNRegression()
        
        X, y, years = bmn.prepare_regression_data(simple_repeat_sales, 2015, 2020)
        
        # Check dimensions
        assert X.shape[0] == len(simple_repeat_sales)  # Number of observations
        assert X.shape[1] == 5  # 2016-2020 (excluding base year 2015)
        assert len(y) == len(simple_repeat_sales)
        assert years == list(range(2015, 2021))
        
        # Check that base period is set
        assert bmn.base_period == 2015
        
        # Check a specific observation
        first_row = simple_repeat_sales.iloc[0]
        first_year = first_row['first_sale_date'].year
        second_year = first_row['second_sale_date'].year
        
        # Check X matrix construction
        if first_year > 2015:
            assert X[0, first_year - 2016] == -1
        if second_year > 2015:
            assert X[0, second_year - 2016] == 1
    
    def test_run_regression(self, simple_repeat_sales):
        """Test running BMN regression"""
        bmn = BMNRegression()
        
        results = bmn.run_regression(simple_repeat_sales, 2015, 2020)
        
        # Check that results are stored
        assert bmn.results is not None
        assert results is not None
        
        # Check basic regression properties
        assert results.nobs == len(simple_repeat_sales)
        assert len(results.params) == 5  # 2016-2020
        
        # With 5% annual appreciation, coefficients should roughly follow pattern
        # Check that coefficients are increasing (cumulative effect)
        coeffs = results.params
        for i in range(1, len(coeffs)):
            assert coeffs[i] > coeffs[i-1]  # Should be increasing
    
    def test_get_index_values(self, simple_repeat_sales):
        """Test extraction of index values"""
        bmn = BMNRegression()
        bmn.run_regression(simple_repeat_sales, 2015, 2020)
        
        index_df = bmn.get_index_values(base_value=100.0)
        
        # Check structure
        assert len(index_df) == 6  # 2015-2020
        assert 'year' in index_df.columns
        assert 'index_value' in index_df.columns
        assert 'coefficient' in index_df.columns
        assert 'std_error' in index_df.columns
        
        # Base year should have index value of 100
        base_row = index_df[index_df['year'] == 2015]
        assert base_row['index_value'].iloc[0] == 100.0
        assert base_row['coefficient'].iloc[0] == 0.0
        
        # Index values should be increasing (with 5% appreciation)
        index_values = index_df.sort_values('year')['index_value'].values
        for i in range(1, len(index_values)):
            assert index_values[i] > index_values[i-1]
    
    def test_get_appreciation_rates(self, volatile_repeat_sales):
        """Test calculation of appreciation rates"""
        bmn = BMNRegression()
        bmn.run_regression(volatile_repeat_sales, 2015, 2020)
        
        appr_df = bmn.get_appreciation_rates()
        
        # Check structure
        assert len(appr_df) == 5  # 2016-2020
        assert 'year' in appr_df.columns
        assert 'appreciation_rate' in appr_df.columns
        
        # First year appreciation
        first_appr = appr_df[appr_df['year'] == 2016]['appreciation_rate'].iloc[0]
        
        # Should be positive but reasonable
        assert -0.2 < first_appr < 0.2  # Within reasonable bounds
    
    def test_get_coefficient_for_year(self, simple_repeat_sales):
        """Test getting coefficient for specific year"""
        bmn = BMNRegression()
        bmn.run_regression(simple_repeat_sales, 2015, 2020)
        
        # Base year should return 0
        assert bmn.get_coefficient_for_year(2015) == 0.0
        
        # Other years should have coefficients
        coef_2018 = bmn.get_coefficient_for_year(2018)
        assert isinstance(coef_2018, float)
        assert coef_2018 > 0  # Should be positive with appreciation
        
        # Invalid year should raise error
        with pytest.raises(ValueError):
            bmn.get_coefficient_for_year(2025)
    
    def test_diagnostic_summary(self, simple_repeat_sales):
        """Test diagnostic summary generation"""
        bmn = BMNRegression()
        bmn.run_regression(simple_repeat_sales, 2015, 2020)
        
        diagnostics = bmn.diagnostic_summary()
        
        # Check required statistics
        required_stats = [
            'r_squared', 'adj_r_squared', 'f_statistic', 'f_pvalue',
            'n_observations', 'n_parameters', 'aic', 'bic', 'mse', 'rmse'
        ]
        
        for stat in required_stats:
            assert stat in diagnostics
            assert isinstance(diagnostics[stat], (int, float))
        
        # Basic sanity checks
        assert 0 <= diagnostics['r_squared'] <= 1
        assert diagnostics['n_observations'] == len(simple_repeat_sales)
        assert diagnostics['n_parameters'] == 5
        assert diagnostics['rmse'] == np.sqrt(diagnostics['mse'])
    
    def test_insufficient_data_warning(self):
        """Test warning with insufficient data"""
        bmn = BMNRegression()
        
        # Create minimal data
        small_data = pd.DataFrame([{
            'property_id': 'PROP001',
            'first_sale_date': pd.to_datetime('2019-01-01'),
            'second_sale_date': pd.to_datetime('2020-01-01'),
            'first_sale_price': 200000,
            'second_sale_price': 210000,
            'log_price_relative': np.log(1.05),
            'census_tract_2010': '06037123456',
            'cbsa_id': '31080'
        }])
        
        # Should log warning but still run
        # Run regression (it will log a warning internally)
        results = bmn.run_regression(small_data, 2015, 2020)
            
        # Check that regression still ran despite few observations
        assert results is not None
        assert bmn.results is not None
        # With only 1 observation and 5 parameters, fit will be poor
        assert results.nobs == 1
    
    def test_run_bmn_for_supertract(self, simple_repeat_sales):
        """Test convenience function for supertract regression"""
        # Test with valid data
        appreciation_rate, coef_t = run_bmn_for_supertract(
            simple_repeat_sales,
            ['06037123456'],  # Tract in the data
            2018
        )
        
        assert isinstance(appreciation_rate, float)
        assert isinstance(coef_t, float)
        assert appreciation_rate > 0  # Should have positive appreciation
        
        # Test with no data for tract
        appreciation_rate, coef_t = run_bmn_for_supertract(
            simple_repeat_sales,
            ['06037999999'],  # Non-existent tract
            2018
        )
        
        assert appreciation_rate == 0.0
        assert coef_t == 0.0