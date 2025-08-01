"""Unit tests for repeat sales pair construction."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from hpi_fhfa.models.repeat_sales import (
    RepeatSalesPair,
    construct_repeat_sales_pairs,
    create_time_dummies,
    split_by_tract,
    calculate_pair_statistics,
    validate_repeat_sales_pairs
)


class TestRepeatSalesPair:
    """Test RepeatSalesPair dataclass."""
    
    def test_repeat_sales_pair_creation(self):
        pair = RepeatSalesPair(
            property_id='P001',
            sale1_date=pd.Timestamp('2019-01-01'),
            sale1_price=200000,
            sale2_date=pd.Timestamp('2021-01-01'),
            sale2_price=240000,
            census_tract='12345678901',
            cbsa_code='10420',
            distance_to_cbd=5.2
        )
        
        assert pair.property_id == 'P001'
        assert pair.sale1_price == 200000
        assert pair.sale2_price == 240000
        
    def test_price_relative_calculation(self):
        pair = RepeatSalesPair(
            property_id='P001',
            sale1_date=pd.Timestamp('2019-01-01'),
            sale1_price=100000,
            sale2_date=pd.Timestamp('2021-01-01'),
            sale2_price=121000,
            census_tract='12345678901',
            cbsa_code='10420',
            distance_to_cbd=5.2
        )
        
        # 21% appreciation
        expected = np.log(121000 / 100000)
        assert np.isclose(pair.price_relative, expected)
        
    def test_time_diff_calculation(self):
        pair = RepeatSalesPair(
            property_id='P001',
            sale1_date=pd.Timestamp('2019-01-01'),
            sale1_price=200000,
            sale2_date=pd.Timestamp('2021-01-01'),
            sale2_price=240000,
            census_tract='12345678901',
            cbsa_code='10420',
            distance_to_cbd=5.2
        )
        
        # Exactly 2 years
        assert np.isclose(pair.time_diff_years, 2.0, rtol=0.01)
        
    def test_cagr_calculation(self):
        pair = RepeatSalesPair(
            property_id='P001',
            sale1_date=pd.Timestamp('2019-01-01'),
            sale1_price=100000,
            sale2_date=pd.Timestamp('2021-01-01'),
            sale2_price=121000,
            census_tract='12345678901',
            cbsa_code='10420',
            distance_to_cbd=5.2
        )
        
        # 10% CAGR (21% over 2 years)
        assert np.isclose(pair.cagr, 0.1, rtol=0.001)
        
    def test_to_dict(self):
        pair = RepeatSalesPair(
            property_id='P001',
            sale1_date=pd.Timestamp('2019-01-01'),
            sale1_price=200000,
            sale2_date=pd.Timestamp('2021-01-01'),
            sale2_price=240000,
            census_tract='12345678901',
            cbsa_code='10420',
            distance_to_cbd=5.2
        )
        
        d = pair.to_dict()
        
        assert d['property_id'] == 'P001'
        assert d['sale1_price'] == 200000
        assert d['sale2_price'] == 240000
        assert 'price_relative' in d
        assert 'time_diff_years' in d
        assert 'cagr' in d


class TestConstructRepeatSalesPairs:
    """Test repeat sales pair construction from transactions."""
    
    @pytest.fixture
    def transaction_data(self):
        """Create sample transaction data."""
        return pd.DataFrame({
            'property_id': ['P1', 'P1', 'P1', 'P2', 'P2', 'P3', 'P4'],
            'transaction_date': pd.to_datetime([
                '2018-01-01', '2019-01-01', '2020-01-01',  # P1: 3 transactions
                '2018-06-01', '2020-06-01',                # P2: 2 transactions
                '2019-03-01',                              # P3: 1 transaction
                '2020-12-01'                               # P4: 1 transaction
            ]),
            'transaction_price': [100000, 110000, 121000, 200000, 240000, 150000, 300000],
            'census_tract': ['12345678901'] * 7,
            'cbsa_code': ['10420'] * 7,
            'distance_to_cbd': [5.2] * 7
        })
    
    def test_basic_pair_construction(self, transaction_data):
        pairs = construct_repeat_sales_pairs(transaction_data, apply_filters=False)
        
        # P1: 2 pairs (1->2, 2->3)
        # P2: 1 pair (1->2)
        # P3, P4: 0 pairs (single transactions)
        # Total: 3 pairs
        assert len(pairs) == 3
        
        # Check property IDs
        assert sorted(pairs['property_id'].unique()) == ['P1', 'P2']
        
        # Check P1 pairs
        p1_pairs = pairs[pairs['property_id'] == 'P1']
        assert len(p1_pairs) == 2
        assert p1_pairs['sale1_price'].tolist() == [100000, 110000]
        assert p1_pairs['sale2_price'].tolist() == [110000, 121000]
        
    def test_calculated_fields(self, transaction_data):
        pairs = construct_repeat_sales_pairs(transaction_data, apply_filters=False)
        
        # Check all calculated fields exist
        required_fields = [
            'price_relative', 'time_diff_years', 'cagr',
            'sale1_year', 'sale2_year', 'sale1_quarter', 'sale2_quarter',
            'sale1_period', 'sale2_period'
        ]
        for field in required_fields:
            assert field in pairs.columns
            
        # Verify price relatives
        assert all(pairs['price_relative'] > 0)
        
        # Verify time differences
        assert all(pairs['time_diff_years'] > 0)
        
        # Verify CAGRs are reasonable
        assert all(pairs['cagr'] > -0.5)
        assert all(pairs['cagr'] < 1.0)
        
    def test_with_filters(self, transaction_data):
        # Add a property with extreme appreciation
        extreme_data = pd.DataFrame({
            'property_id': ['P5'],
            'transaction_date': pd.to_datetime(['2019-01-01']),
            'transaction_price': [100000],
            'census_tract': ['12345678901'],
            'cbsa_code': ['10420'],
            'distance_to_cbd': [5.2]
        })
        
        # Second transaction with 500% appreciation in 1 year
        extreme_data2 = pd.DataFrame({
            'property_id': ['P5'],
            'transaction_date': pd.to_datetime(['2020-01-01']),
            'transaction_price': [600000],
            'census_tract': ['12345678901'],
            'cbsa_code': ['10420'],
            'distance_to_cbd': [5.2]
        })
        
        all_data = pd.concat([transaction_data, extreme_data, extreme_data2])
        
        pairs_unfiltered = construct_repeat_sales_pairs(all_data, apply_filters=False)
        pairs_filtered = construct_repeat_sales_pairs(all_data, apply_filters=True)
        
        # P5 should be filtered out
        assert 'P5' in pairs_unfiltered['property_id'].values
        assert 'P5' not in pairs_filtered['property_id'].values
        
    def test_custom_column_names(self):
        data = pd.DataFrame({
            'prop_id': ['P1', 'P1'],
            'date': pd.to_datetime(['2019-01-01', '2020-01-01']),
            'price': [100000, 110000],
            'tract': ['12345678901', '12345678901'],
            'cbsa': ['10420', '10420'],
            'cbd_dist': [5.2, 5.2]
        })
        
        pairs = construct_repeat_sales_pairs(
            data,
            property_col='prop_id',
            date_col='date',
            price_col='price',
            tract_col='tract',
            cbsa_col='cbsa',
            cbd_dist_col='cbd_dist',
            apply_filters=False
        )
        
        assert len(pairs) == 1
        assert pairs['property_id'].iloc[0] == 'P1'


class TestTimeDummies:
    """Test time dummy variable creation."""
    
    def test_create_time_dummies_dense(self):
        pairs = pd.DataFrame({
            'sale1_period': [0, 1, 0, 2],
            'sale2_period': [1, 2, 2, 3]
        })
        
        dummies = create_time_dummies(pairs, sparse=False)
        
        # Should have columns for each period
        assert len(dummies.columns) == 4
        assert all(col.startswith('period_') for col in dummies.columns)
        
        # Check first row (period 0 -> 1)
        assert dummies.iloc[0]['period_0'] == -1
        assert dummies.iloc[0]['period_1'] == 1
        assert dummies.iloc[0]['period_2'] == 0
        assert dummies.iloc[0]['period_3'] == 0
        
        # Check second row (period 1 -> 2)
        assert dummies.iloc[1]['period_1'] == -1
        assert dummies.iloc[1]['period_2'] == 1
        
    def test_create_time_dummies_sparse(self):
        pairs = pd.DataFrame({
            'sale1_period': [0, 1, 0, 2],
            'sale2_period': [1, 2, 2, 3]
        })
        
        dummies = create_time_dummies(pairs, sparse=True)
        
        # Should be sparse
        assert all(isinstance(dummies[col].array, pd.arrays.SparseArray) for col in dummies.columns)
        
        # Values should match dense version
        dense_dummies = create_time_dummies(pairs, sparse=False)
        pd.testing.assert_frame_equal(
            dummies.astype('int8'),
            dense_dummies
        )


class TestSplitByTract:
    """Test splitting pairs by census tract."""
    
    def test_split_by_tract(self):
        pairs = pd.DataFrame({
            'property_id': ['P1', 'P2', 'P3', 'P4'],
            'census_tract': ['12345678901', '12345678901', '12345678902', '12345678902'],
            'sale1_price': [100000, 200000, 150000, 250000],
            'sale2_price': [110000, 220000, 165000, 275000]
        })
        
        tract_groups = split_by_tract(pairs)
        
        assert len(tract_groups) == 2
        assert '12345678901' in tract_groups
        assert '12345678902' in tract_groups
        
        # Check tract 1
        tract1 = tract_groups['12345678901']
        assert len(tract1) == 2
        assert set(tract1['property_id']) == {'P1', 'P2'}
        
        # Check tract 2
        tract2 = tract_groups['12345678902']
        assert len(tract2) == 2
        assert set(tract2['property_id']) == {'P3', 'P4'}


class TestPairStatistics:
    """Test calculation of pair statistics."""
    
    def test_calculate_pair_statistics(self):
        pairs = pd.DataFrame({
            'property_id': ['P1', 'P1', 'P2', 'P3'],
            'sale1_date': pd.to_datetime(['2018-01-01', '2019-01-01', '2018-06-01', '2019-03-01']),
            'sale2_date': pd.to_datetime(['2019-01-01', '2020-01-01', '2020-06-01', '2021-03-01']),
            'time_diff_years': [1.0, 1.0, 2.0, 2.0],
            'price_relative': [0.095, 0.091, 0.182, 0.223],
            'cagr': [0.095, 0.091, 0.091, 0.112],
            'census_tract': ['12345678901', '12345678901', '12345678902', '12345678901'],
            'sale1_year': [2018, 2019, 2018, 2019]
        })
        
        stats = calculate_pair_statistics(pairs)
        
        assert stats.loc['n_pairs', 'value'] == 4
        assert stats.loc['n_properties', 'value'] == 3
        assert stats.loc['avg_time_between_sales', 'value'] == 1.5
        assert stats.loc['n_tracts', 'value'] == 2
        
        # Check year distribution
        assert 'pairs_by_year' in stats.index
        

class TestValidation:
    """Test repeat sales pair validation."""
    
    def test_valid_pairs(self):
        pairs = pd.DataFrame({
            'property_id': ['P1', 'P2'],
            'sale1_date': pd.to_datetime(['2019-01-01', '2020-01-01']),
            'sale2_date': pd.to_datetime(['2020-01-01', '2021-01-01']),
            'sale1_price': [100000, 200000],
            'sale2_price': [110000, 220000],
            'census_tract': ['12345678901', '12345678902'],
            'price_relative': [0.095, 0.095],
            'time_diff_years': [1.0, 1.0]
        })
        
        is_valid, errors = validate_repeat_sales_pairs(pairs)
        assert is_valid
        assert len(errors) == 0
        
    def test_missing_columns(self):
        pairs = pd.DataFrame({
            'property_id': ['P1', 'P2'],
            'sale1_date': pd.to_datetime(['2019-01-01', '2020-01-01'])
            # Missing other required columns
        })
        
        is_valid, errors = validate_repeat_sales_pairs(pairs)
        assert not is_valid
        assert any('Missing required columns' in e for e in errors)
        
    def test_invalid_date_order(self):
        pairs = pd.DataFrame({
            'property_id': ['P1', 'P2'],
            'sale1_date': pd.to_datetime(['2020-01-01', '2020-01-01']),
            'sale2_date': pd.to_datetime(['2019-01-01', '2021-01-01']),
            'sale1_price': [100000, 200000],
            'sale2_price': [110000, 220000],
            'census_tract': ['12345678901', '12345678902']
        })
        
        is_valid, errors = validate_repeat_sales_pairs(pairs)
        assert not is_valid
        assert any('sale1_date >= sale2_date' in e for e in errors)
        
    def test_invalid_prices(self):
        pairs = pd.DataFrame({
            'property_id': ['P1', 'P2'],
            'sale1_date': pd.to_datetime(['2019-01-01', '2020-01-01']),
            'sale2_date': pd.to_datetime(['2020-01-01', '2021-01-01']),
            'sale1_price': [0, 200000],  # Zero price
            'sale2_price': [110000, -220000],  # Negative price
            'census_tract': ['12345678901', '12345678902']
        })
        
        is_valid, errors = validate_repeat_sales_pairs(pairs)
        assert not is_valid
        assert any('sale1_price values are not positive' in e for e in errors)
        assert any('sale2_price values are not positive' in e for e in errors)