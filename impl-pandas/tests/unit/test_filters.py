"""Unit tests for transaction filtering functions."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from hpi_fhfa.data.filters import (
    filter_transactions,
    apply_same_period_filter,
    apply_cagr_filter,
    apply_cumulative_filter,
    filter_outliers_by_zscore,
    validate_filtered_data
)


class TestFilterTransactions:
    """Test comprehensive transaction filtering."""
    
    @pytest.fixture
    def sample_pairs(self):
        """Create sample repeat sales pairs for testing."""
        return pd.DataFrame({
            'property_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'sale1_date': pd.to_datetime([
                '2018-01-15', '2019-06-20', '2017-03-10', '2018-12-01', '2019-05-15'
            ]),
            'sale1_price': [200000, 300000, 250000, 400000, 150000],
            'sale2_date': pd.to_datetime([
                '2020-01-15', '2020-06-20', '2019-03-10', '2018-12-31', '2021-05-15'
            ]),
            'sale2_price': [240000, 360000, 600000, 480000, 450000],
            'census_tract': ['12345678901'] * 5,
            'cbsa_code': ['10420'] * 5,
            'distance_to_cbd': [5.2, 3.8, 7.1, 2.5, 4.3]
        })
    
    def test_filter_all_default(self, sample_pairs):
        # Add required calculated fields
        sample_pairs['time_diff_years'] = (
            (sample_pairs['sale2_date'] - sample_pairs['sale1_date']).dt.days / 365.25
        )
        price_ratio = sample_pairs['sale2_price'] / sample_pairs['sale1_price']
        sample_pairs['cagr'] = np.power(price_ratio, 1 / sample_pairs['time_diff_years']) - 1
        
        filtered = filter_transactions(sample_pairs)
        
        # P003 should be filtered out (140% appreciation > 30% CAGR)
        # P004 should be filtered out (same year)
        # P005 should be filtered out (200% appreciation > 30% CAGR)
        assert len(filtered) == 2
        assert set(filtered['property_id']) == {'P001', 'P002'}
        
    def test_filter_custom_thresholds(self, sample_pairs):
        # Add required calculated fields
        sample_pairs['time_diff_years'] = (
            (sample_pairs['sale2_date'] - sample_pairs['sale1_date']).dt.days / 365.25
        )
        price_ratio = sample_pairs['sale2_price'] / sample_pairs['sale1_price']
        sample_pairs['cagr'] = np.power(price_ratio, 1 / sample_pairs['time_diff_years']) - 1
        
        # More lenient CAGR filter
        filtered = filter_transactions(sample_pairs, max_cagr=0.60)
        
        # P004 still filtered (same year), P003 passes now (55% CAGR)
        assert len(filtered) > 2
        
    def test_filter_no_same_period(self, sample_pairs):
        # Add required calculated fields
        sample_pairs['time_diff_years'] = (
            (sample_pairs['sale2_date'] - sample_pairs['sale1_date']).dt.days / 365.25
        )
        price_ratio = sample_pairs['sale2_price'] / sample_pairs['sale1_price']
        sample_pairs['cagr'] = np.power(price_ratio, 1 / sample_pairs['time_diff_years']) - 1
        
        filtered = filter_transactions(sample_pairs, filter_same_period=False)
        
        # P004 still filtered due to extreme CAGR (30 days = 820% CAGR)
        # Only P001 and P002 pass all filters
        assert len(filtered) == 2
        assert set(filtered['property_id']) == {'P001', 'P002'}


class TestSamePeriodFilter:
    """Test same 12-month period filtering."""
    
    def test_filter_same_year(self):
        df = pd.DataFrame({
            'sale1_date': pd.to_datetime(['2020-01-15', '2020-06-20', '2019-12-31']),
            'sale2_date': pd.to_datetime(['2020-12-30', '2021-01-05', '2020-01-01'])
        })
        
        filtered = apply_same_period_filter(df)
        
        # First pair: same year (2020), should be filtered
        # Second pair: different years (2020/2021), should pass
        # Third pair: different years (2019/2020), should pass
        assert len(filtered) == 2
        assert filtered.index.tolist() == [1, 2]
        
    def test_all_same_year(self):
        df = pd.DataFrame({
            'sale1_date': pd.to_datetime(['2020-01-15', '2020-06-20', '2020-11-30']),
            'sale2_date': pd.to_datetime(['2020-03-15', '2020-09-20', '2020-12-30'])
        })
        
        filtered = apply_same_period_filter(df)
        assert len(filtered) == 0
        
    def test_all_different_years(self):
        df = pd.DataFrame({
            'sale1_date': pd.to_datetime(['2018-01-15', '2019-06-20', '2020-11-30']),
            'sale2_date': pd.to_datetime(['2019-03-15', '2020-09-20', '2021-12-30'])
        })
        
        filtered = apply_same_period_filter(df)
        assert len(filtered) == 3


class TestCAGRFilter:
    """Test compound annual growth rate filtering."""
    
    def test_calculate_cagr(self):
        df = pd.DataFrame({
            'sale1_date': pd.to_datetime(['2018-01-01', '2019-01-01']),
            'sale2_date': pd.to_datetime(['2020-01-01', '2021-01-01']),
            'sale1_price': [100000, 200000],
            'sale2_price': [121000, 242000]  # 10% annual growth
        })
        
        filtered = apply_cagr_filter(df)
        
        # Both should pass with 10% CAGR (< 30% threshold)
        assert len(filtered) == 2
        assert 'cagr' in filtered.columns
        assert np.allclose(filtered['cagr'].values, [0.1, 0.1], rtol=1e-3)
        
    def test_filter_high_cagr(self):
        df = pd.DataFrame({
            'sale1_date': pd.to_datetime(['2019-01-01', '2020-01-01']),
            'sale2_date': pd.to_datetime(['2020-01-01', '2021-01-01']),
            'sale1_price': [100000, 200000],
            'sale2_price': [150000, 300000]  # 50% growth in 1 year
        })
        
        filtered = apply_cagr_filter(df)
        
        # First should be filtered (50% > 30%)
        # Second should be filtered (50% > 30%)
        assert len(filtered) == 0
        
    def test_filter_negative_cagr(self):
        df = pd.DataFrame({
            'sale1_date': pd.to_datetime(['2019-01-01', '2020-01-01']),
            'sale2_date': pd.to_datetime(['2020-01-01', '2021-01-01']),
            'sale1_price': [200000, 100000],
            'sale2_price': [100000, 50000]  # -50% growth
        })
        
        filtered = apply_cagr_filter(df)
        
        # Both should be filtered (|-50%| > 30%)
        assert len(filtered) == 0
        
    def test_custom_cagr_thresholds(self):
        df = pd.DataFrame({
            'sale1_date': pd.to_datetime(['2019-01-01', '2020-01-01']),
            'sale2_date': pd.to_datetime(['2020-01-01', '2021-01-01']),
            'sale1_price': [100000, 200000],
            'sale2_price': [140000, 260000]  # 40% and 30% growth
        })
        
        # Default filter
        filtered_default = apply_cagr_filter(df)
        assert len(filtered_default) == 1  # Only 30% passes
        
        # Custom thresholds
        filtered_custom = apply_cagr_filter(df, min_cagr=-0.50, max_cagr=0.50)
        assert len(filtered_custom) == 2  # Both within ±50%


class TestCumulativeFilter:
    """Test cumulative appreciation filtering."""
    
    def test_filter_high_appreciation(self):
        df = pd.DataFrame({
            'sale1_price': [100000, 200000, 300000],
            'sale2_price': [1100000, 400000, 600000]  # 11x, 2x, 2x
        })
        
        filtered = apply_cumulative_filter(df)
        
        # First should be filtered (11x > 10x)
        assert len(filtered) == 2
        assert filtered.index.tolist() == [1, 2]
        
    def test_filter_low_appreciation(self):
        df = pd.DataFrame({
            'sale1_price': [100000, 200000, 300000],
            'sale2_price': [20000, 100000, 150000]  # 0.2x, 0.5x, 0.5x
        })
        
        filtered = apply_cumulative_filter(df)
        
        # First should be filtered (0.2x < 0.25x)
        assert len(filtered) == 2
        assert filtered.index.tolist() == [1, 2]
        
    def test_custom_thresholds(self):
        df = pd.DataFrame({
            'sale1_price': [100000, 200000],
            'sale2_price': [300000, 1000000]  # 3x, 5x
        })
        
        # Default thresholds (0.25x - 10x)
        filtered_default = apply_cumulative_filter(df)
        assert len(filtered_default) == 2
        
        # Stricter thresholds
        filtered_strict = apply_cumulative_filter(df, min_ratio=0.5, max_ratio=4.0)
        assert len(filtered_strict) == 1  # Only 3x passes


class TestOutlierFiltering:
    """Test z-score based outlier filtering."""
    
    def test_global_zscore_filter(self):
        # Create data with outlier
        np.random.seed(42)
        values = np.random.normal(0, 1, 100)
        values[50] = 10  # Clear outlier
        
        df = pd.DataFrame({'value': values})
        
        filtered = filter_outliers_by_zscore(df, 'value', threshold=3.0)
        
        assert len(filtered) < 100
        assert 50 not in filtered.index  # Outlier removed
        
    def test_grouped_zscore_filter(self):
        # Create data with group-specific outliers
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(10, 1, 50)
        
        group1[25] = 10  # Outlier in group 1
        group2[25] = 0   # Outlier in group 2
        
        df = pd.DataFrame({
            'group': ['A'] * 50 + ['B'] * 50,
            'value': np.concatenate([group1, group2])
        })
        
        filtered = filter_outliers_by_zscore(df, 'value', threshold=3.0, by_group='group')
        
        assert len(filtered) < 100
        # Both group-specific outliers should be removed


class TestDataValidation:
    """Test filtered data validation."""
    
    def test_valid_data(self):
        df = pd.DataFrame({
            'property_id': ['P001', 'P002'],
            'sale1_date': pd.to_datetime(['2019-01-01', '2020-01-01']),
            'sale2_date': pd.to_datetime(['2020-01-01', '2021-01-01']),
            'sale1_price': [100000, 200000],
            'sale2_price': [120000, 240000],
            'census_tract': ['12345678901', '12345678902']
        })
        
        is_valid, message = validate_filtered_data(df)
        assert is_valid
        assert message == "Data validation passed"
        
    def test_empty_data(self):
        df = pd.DataFrame()
        
        is_valid, message = validate_filtered_data(df)
        assert not is_valid
        assert "No transactions remain" in message
        
    def test_missing_columns(self):
        df = pd.DataFrame({
            'property_id': ['P001', 'P002'],
            'sale1_date': pd.to_datetime(['2019-01-01', '2020-01-01']),
            # Missing other required columns
        })
        
        is_valid, message = validate_filtered_data(df)
        assert not is_valid
        assert "Missing required columns" in message
        
    def test_invalid_date_order(self):
        df = pd.DataFrame({
            'property_id': ['P001', 'P002'],
            'sale1_date': pd.to_datetime(['2020-01-01', '2020-01-01']),
            'sale2_date': pd.to_datetime(['2019-01-01', '2021-01-01']),  # First is invalid
            'sale1_price': [100000, 200000],
            'sale2_price': [120000, 240000],
            'census_tract': ['12345678901', '12345678902']
        })
        
        is_valid, message = validate_filtered_data(df)
        assert not is_valid
        assert "sale2_date must be after sale1_date" in message
        
    def test_invalid_prices(self):
        df = pd.DataFrame({
            'property_id': ['P001', 'P002'],
            'sale1_date': pd.to_datetime(['2019-01-01', '2020-01-01']),
            'sale2_date': pd.to_datetime(['2020-01-01', '2021-01-01']),
            'sale1_price': [0, 200000],  # Invalid price
            'sale2_price': [120000, 240000],
            'census_tract': ['12345678901', '12345678902']
        })
        
        is_valid, message = validate_filtered_data(df)
        assert not is_valid
        assert "All prices must be positive" in message