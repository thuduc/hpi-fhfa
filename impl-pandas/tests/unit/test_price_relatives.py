"""Unit tests for price relative calculations."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from hpi_fhfa.models.price_relatives import (
    calculate_price_relative,
    calculate_price_relatives,
    calculate_half_pairs,
    calculate_appreciation_rate,
    calculate_cagr,
    add_time_variables,
    summarize_price_relatives
)


class TestPriceRelativeCalculation:
    """Test basic price relative calculations."""
    
    def test_calculate_price_relative_log(self):
        # Test log transformation (default)
        result = calculate_price_relative(100000, 110000)
        expected = np.log(110000) - np.log(100000)
        assert np.isclose(result, expected)
        
        # Test with exact doubling
        result = calculate_price_relative(100000, 200000)
        assert np.isclose(result, np.log(2))
        
    def test_calculate_price_relative_ratio(self):
        # Test without log transformation
        result = calculate_price_relative(100000, 110000, log_transform=False)
        assert result == 1.1
        
        result = calculate_price_relative(200000, 100000, log_transform=False)
        assert result == 0.5
        
    def test_invalid_prices(self):
        # Zero price
        with pytest.raises(ValueError, match="Prices must be positive"):
            calculate_price_relative(0, 100000)
            
        with pytest.raises(ValueError, match="Prices must be positive"):
            calculate_price_relative(100000, 0)
            
        # Negative price
        with pytest.raises(ValueError, match="Prices must be positive"):
            calculate_price_relative(-100000, 200000)
            

class TestBatchPriceRelatives:
    """Test batch price relative calculations."""
    
    def test_calculate_price_relatives_dataframe(self):
        df = pd.DataFrame({
            'sale1_price': [100000, 200000, 150000],
            'sale2_price': [110000, 250000, 180000]
        })
        
        result = calculate_price_relatives(df)
        
        assert 'price_relative' in result.columns
        assert len(result) == 3
        
        # Verify calculations
        expected = [
            np.log(110000/100000),
            np.log(250000/200000),
            np.log(180000/150000)
        ]
        np.testing.assert_allclose(result['price_relative'], expected)
        
    def test_custom_column_names(self):
        df = pd.DataFrame({
            'first_price': [100000, 200000],
            'second_price': [120000, 240000]
        })
        
        result = calculate_price_relatives(
            df,
            price1_col='first_price',
            price2_col='second_price',
            output_col='log_return'
        )
        
        assert 'log_return' in result.columns
        assert 'price_relative' not in result.columns
        
    def test_invalid_prices_in_dataframe(self):
        df = pd.DataFrame({
            'sale1_price': [100000, 0, 150000],
            'sale2_price': [110000, 250000, 180000]
        })
        
        with pytest.raises(ValueError, match="All prices must be positive"):
            calculate_price_relatives(df)


class TestHalfPairsCalculation:
    """Test half-pairs counting logic."""
    
    @pytest.fixture
    def transaction_data(self):
        """Create sample transaction data."""
        return pd.DataFrame({
            'property_id': ['P1', 'P1', 'P1', 'P2', 'P2', 'P3', 'P3', 'P4'],
            'transaction_date': pd.to_datetime([
                '2018-01-01', '2019-01-01', '2020-01-01',  # P1: 3 transactions
                '2018-06-01', '2020-06-01',                # P2: 2 transactions
                '2019-03-01', '2021-03-01',                # P3: 2 transactions
                '2020-12-01'                               # P4: 1 transaction (no repeat)
            ]),
            'census_tract': ['12345678901'] * 8,
            'transaction_price': [100000, 110000, 120000, 200000, 240000, 150000, 180000, 300000]
        })
    
    def test_half_pairs_basic(self, transaction_data):
        result = calculate_half_pairs(transaction_data, by_period=False)
        
        # All transactions are in same tract
        assert len(result) == 1
        assert result['census_tract'].iloc[0] == '12345678901'
        
        # Expected half-pairs:
        # P1: 1 + 2 + 1 = 4
        # P2: 1 + 1 = 2
        # P3: 1 + 1 = 2
        # P4: 0 (single transaction)
        # Total: 8
        assert result['half_pairs_count'].iloc[0] == 8
        
    def test_half_pairs_by_period(self, transaction_data):
        result = calculate_half_pairs(transaction_data, by_period=True)
        
        # Should have entries for each tract-year combination
        expected_years = [2018, 2019, 2020, 2021]
        assert sorted(result['year'].unique()) == expected_years
        
        # Verify half-pairs by year
        year_counts = dict(zip(result['year'], result['half_pairs_count']))
        
        # 2018: P1(1) + P2(1) = 2
        assert year_counts[2018] == 2
        
        # 2019: P1(2) + P3(1) = 3
        assert year_counts[2019] == 3
        
        # 2020: P1(1) + P2(1) = 2
        assert year_counts[2020] == 2
        
        # 2021: P3(1) = 1
        assert year_counts[2021] == 1
        
    def test_half_pairs_multiple_tracts(self):
        data = pd.DataFrame({
            'property_id': ['P1', 'P1', 'P2', 'P2', 'P3', 'P3'],
            'transaction_date': pd.to_datetime([
                '2019-01-01', '2020-01-01',
                '2019-06-01', '2020-06-01',
                '2019-03-01', '2020-03-01'
            ]),
            'census_tract': ['12345678901', '12345678901', '12345678902', '12345678902', '12345678901', '12345678901'],
            'transaction_price': [100000, 110000, 200000, 220000, 150000, 165000]
        })
        
        result = calculate_half_pairs(data, by_period=False)
        
        assert len(result) == 2
        
        # Tract 1: P1(2) + P3(2) = 4
        tract1_count = result[result['census_tract'] == '12345678901']['half_pairs_count'].iloc[0]
        assert tract1_count == 4
        
        # Tract 2: P2(2) = 2
        tract2_count = result[result['census_tract'] == '12345678902']['half_pairs_count'].iloc[0]
        assert tract2_count == 2


class TestAppreciationCalculations:
    """Test appreciation rate and CAGR calculations."""
    
    def test_appreciation_rate(self):
        # 10% appreciation over 2 years
        price_relative = np.log(1.1)
        time_diff = 2.0
        
        rate = calculate_appreciation_rate(price_relative, time_diff)
        assert np.isclose(rate, np.log(1.1) / 2)
        
    def test_appreciation_rate_invalid(self):
        with pytest.raises(ValueError, match="Time difference must be positive"):
            calculate_appreciation_rate(0.1, 0)
            
        with pytest.raises(ValueError, match="Time difference must be positive"):
            calculate_appreciation_rate(0.1, -1)
            
    def test_cagr_calculation(self):
        # 21% total appreciation over 2 years = 10% CAGR
        cagr = calculate_cagr(100000, 121000, 2.0)
        assert np.isclose(cagr, 0.1)
        
        # 100% appreciation over 5 years
        cagr = calculate_cagr(100000, 200000, 5.0)
        expected = np.power(2, 1/5) - 1
        assert np.isclose(cagr, expected)
        
        # Depreciation
        cagr = calculate_cagr(200000, 100000, 2.0)
        assert cagr < 0
        
    def test_cagr_invalid_inputs(self):
        with pytest.raises(ValueError, match="Prices must be positive"):
            calculate_cagr(0, 100000, 1)
            
        with pytest.raises(ValueError, match="Prices must be positive"):
            calculate_cagr(100000, -100000, 1)
            
        with pytest.raises(ValueError, match="Time difference must be positive"):
            calculate_cagr(100000, 120000, 0)


class TestTimeVariables:
    """Test time variable creation."""
    
    def test_add_time_variables(self):
        df = pd.DataFrame({
            'sale1_date': pd.to_datetime(['2018-03-15', '2019-06-20']),
            'sale2_date': pd.to_datetime(['2020-09-10', '2021-12-25'])
        })
        
        result = add_time_variables(df)
        
        # Check all expected columns exist
        expected_cols = [
            'time_diff_days', 'time_diff_years',
            'sale1_year', 'sale2_year',
            'sale1_quarter', 'sale2_quarter',
            'sale1_period', 'sale2_period'
        ]
        for col in expected_cols:
            assert col in result.columns
            
        # Verify calculations
        assert result['sale1_year'].tolist() == [2018, 2019]
        assert result['sale2_year'].tolist() == [2020, 2021]
        assert result['sale1_quarter'].tolist() == [1, 2]
        assert result['sale2_quarter'].tolist() == [3, 4]
        
        # Time differences
        expected_days = [
            (pd.Timestamp('2020-09-10') - pd.Timestamp('2018-03-15')).days,
            (pd.Timestamp('2021-12-25') - pd.Timestamp('2019-06-20')).days
        ]
        assert result['time_diff_days'].tolist() == expected_days
        
        # Period indices should be sequential
        assert result['sale1_period'].iloc[0] < result['sale2_period'].iloc[0]
        assert result['sale1_period'].iloc[1] < result['sale2_period'].iloc[1]
        
    def test_period_mapping(self):
        # Test that period mapping is consistent
        df = pd.DataFrame({
            'sale1_date': pd.to_datetime(['2018-Q1', '2018-Q1', '2018-Q2']),
            'sale2_date': pd.to_datetime(['2018-Q2', '2018-Q3', '2018-Q3'])
        })
        
        result = add_time_variables(df)
        
        # Same quarters should have same period index
        assert result['sale1_period'].iloc[0] == result['sale1_period'].iloc[1]
        assert result['sale2_period'].iloc[1] == result['sale2_period'].iloc[2]


class TestSummaryStatistics:
    """Test price relative summary statistics."""
    
    def test_global_summary(self):
        df = pd.DataFrame({
            'price_relative': np.random.normal(0.05, 0.02, 1000)
        })
        
        summary = summarize_price_relatives(df)
        
        # Check the summary is a DataFrame with one row
        assert len(summary) == 1  # Should have 1 row
        assert 'count' in summary.columns
        assert 'mean' in summary.columns
        assert 'std' in summary.columns
        assert 'min' in summary.columns
        assert 'max' in summary.columns
        assert 'q25' in summary.columns
        assert 'median' in summary.columns
        assert 'q75' in summary.columns
        
        # Access values from DataFrame
        assert summary['count'].iloc[0] == 1000
        assert abs(summary['mean'].iloc[0] - 0.05) < 0.01
        assert abs(summary['std'].iloc[0] - 0.02) < 0.01
        
    def test_grouped_summary(self):
        df = pd.DataFrame({
            'census_tract': ['12345678901'] * 500 + ['12345678902'] * 500,
            'price_relative': np.concatenate([
                np.random.normal(0.03, 0.01, 500),
                np.random.normal(0.07, 0.02, 500)
            ])
        })
        
        summary = summarize_price_relatives(df, group_col='census_tract')
        
        assert len(summary) == 2
        assert 'census_tract' in summary.columns
        assert 'count' in summary.columns
        assert 'mean' in summary.columns
        
        # Verify counts
        assert summary['count'].tolist() == [500, 500]
        
        # Verify means are different
        tract1_mean = summary[summary['census_tract'] == '12345678901']['mean'].iloc[0]
        tract2_mean = summary[summary['census_tract'] == '12345678902']['mean'].iloc[0]
        assert abs(tract1_mean - 0.03) < 0.01
        assert abs(tract2_mean - 0.07) < 0.01