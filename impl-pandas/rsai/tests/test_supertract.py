"""Unit tests for supertract generation"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from rsai.src.geography.supertract import SupertractGenerator


class TestSupertractGenerator:
    """Test SupertractGenerator class"""
    
    @pytest.fixture
    def geographic_data(self):
        """Create sample geographic data for testing"""
        # Create a grid of census tracts
        tracts = []
        for i in range(3):  # 3 CBSAs
            for j in range(10):  # 10 tracts per CBSA
                tracts.append({
                    'census_tract_2010': f'0603700{i}{j:02d}',
                    'centroid_lat': 34.0 + i * 0.1 + j * 0.01,
                    'centroid_lon': -118.0 + i * 0.1 + j * 0.01,
                    'cbsa_id': f'CBSA00{i}'
                })
        
        return pd.DataFrame(tracts)
    
    @pytest.fixture
    def repeat_sales_data(self):
        """Create repeat sales data with varying transaction counts"""
        data = []
        
        # CBSA000: Mix of high and low transaction tracts
        high_trans_tracts = ['0603700000', '0603700001', '0603700002']
        low_trans_tracts = ['0603700003', '0603700004', '0603700005']
        
        # High transaction tracts (50+ half-pairs each)
        for tract in high_trans_tracts:
            for i in range(30):  # 30 repeat sales = 60 half-pairs
                data.append({
                    'property_id': f'{tract}_PROP{i:03d}',
                    'first_sale_date': pd.to_datetime(f'2019-{(i%12)+1:02d}-01'),
                    'second_sale_date': pd.to_datetime(f'2020-{(i%12)+1:02d}-01'),
                    'first_sale_price': 200000 + i * 1000,
                    'second_sale_price': 220000 + i * 1000,
                    'census_tract_2010': tract,
                    'cbsa_id': 'CBSA000',
                    'log_price_relative': np.log(1.1)
                })
        
        # Low transaction tracts (10-20 half-pairs each)
        for tract in low_trans_tracts:
            for i in range(8):  # 8 repeat sales = 16 half-pairs
                data.append({
                    'property_id': f'{tract}_PROP{i:03d}',
                    'first_sale_date': pd.to_datetime(f'2019-{(i%12)+1:02d}-01'),
                    'second_sale_date': pd.to_datetime(f'2020-{(i%12)+1:02d}-01'),
                    'first_sale_price': 300000 + i * 2000,
                    'second_sale_price': 330000 + i * 2000,
                    'census_tract_2010': tract,
                    'cbsa_id': 'CBSA000',
                    'log_price_relative': np.log(1.1)
                })
        
        return pd.DataFrame(data)
    
    def test_calculate_half_pairs_single_tract(self, repeat_sales_data):
        """Test half-pairs calculation for a single tract"""
        generator = SupertractGenerator(pd.DataFrame(), min_half_pairs=40)
        
        # Test for a high-transaction tract
        half_pairs_2020 = generator.calculate_half_pairs(
            repeat_sales_data, 2020, '0603700000'
        )
        
        # Should have half-pairs from transactions where either sale is in 2020
        tract_sales = repeat_sales_data[repeat_sales_data['census_tract_2010'] == '0603700000']
        expected = (
            (tract_sales['first_sale_date'].dt.year == 2020).sum() +
            (tract_sales['second_sale_date'].dt.year == 2020).sum()
        )
        
        assert half_pairs_2020 == expected
    
    def test_calculate_half_pairs_multi_tract(self, repeat_sales_data):
        """Test half-pairs calculation for multiple tracts"""
        generator = SupertractGenerator(pd.DataFrame(), min_half_pairs=40)
        
        tracts = ['0603700000', '0603700001']
        half_pairs = generator.calculate_half_pairs_multi(
            repeat_sales_data, 2020, tracts
        )
        
        # Should be sum of half-pairs for both tracts
        individual_sum = sum(
            generator.calculate_half_pairs(repeat_sales_data, 2020, tract)
            for tract in tracts
        )
        
        assert half_pairs == individual_sum
    
    def test_generate_supertracts_for_year(self, geographic_data, repeat_sales_data):
        """Test supertract generation for a specific year"""
        generator = SupertractGenerator(geographic_data, min_half_pairs=40)
        
        supertracts = generator.generate_supertracts_for_year(
            repeat_sales_data, 2020, 'CBSA000'
        )
        
        # Check that all tracts are in supertracts
        all_tracts = set()
        for st_id, components in supertracts.items():
            all_tracts.update(components)
        
        # Should have all CBSA000 tracts covered
        expected_tracts = set([f'0603700{0}{i:02d}' for i in range(10)])
        assert expected_tracts.issubset(all_tracts)
        
        # Low transaction tracts should be merged
        low_trans_tracts = ['0603700003', '0603700004', '0603700005']
        merged_supertracts = []
        
        for st_id, components in supertracts.items():
            if any(tract in components for tract in low_trans_tracts):
                merged_supertracts.append(components)
        
        # Should have merged tracts
        assert any(len(st) > 1 for st in merged_supertracts)
    
    def test_threshold_enforcement(self, geographic_data):
        """Test that threshold is enforced for both current and previous year"""
        # Create data where tract meets threshold in 2020 but not 2019
        data = []
        
        # Tract with transactions only in 2020
        for i in range(25):  # 50 half-pairs in 2020, 0 in 2019
            data.append({
                'property_id': f'PROP{i:03d}',
                'first_sale_date': pd.to_datetime('2020-01-01'),
                'second_sale_date': pd.to_datetime('2020-06-01'),
                'first_sale_price': 200000,
                'second_sale_price': 210000,
                'census_tract_2010': '0603700100',
                'cbsa_id': 'CBSA001',
                'log_price_relative': np.log(1.05)
            })
        
        repeat_sales_df = pd.DataFrame(data)
        
        # Add this tract to geographic data
        geo_subset = geographic_data[geographic_data['cbsa_id'] == 'CBSA001'].copy()
        new_tract = pd.DataFrame([{
            'census_tract_2010': '0603700100',
            'centroid_lat': 34.1,
            'centroid_lon': -118.1,
            'cbsa_id': 'CBSA001'
        }])
        geo_subset = pd.concat([geo_subset, new_tract], ignore_index=True)
        
        generator = SupertractGenerator(geo_subset, min_half_pairs=40)
        
        supertracts = generator.generate_supertracts_for_year(
            repeat_sales_df, 2020, 'CBSA001'
        )
        
        # Tract should be merged despite having 50 half-pairs in 2020
        # because it has 0 in 2019
        tract_supertract = None
        for st_id, components in supertracts.items():
            if '0603700100' in components:
                tract_supertract = components
                break
        
        assert tract_supertract is not None
        assert len(tract_supertract) > 1  # Should be merged
    
    def test_nearest_neighbor_merging(self, geographic_data):
        """Test that tracts merge with nearest neighbors"""
        # Create specific pattern of transactions
        data = []
        
        # Three tracts in a line with low transactions
        tracts = ['0603700200', '0603700201', '0603700202']  # These should be adjacent
        
        for tract in tracts:
            for i in range(10):  # 20 half-pairs each (below threshold)
                data.append({
                    'property_id': f'{tract}_PROP{i:03d}',
                    'first_sale_date': pd.to_datetime('2019-06-01'),
                    'second_sale_date': pd.to_datetime('2020-06-01'),
                    'first_sale_price': 200000,
                    'second_sale_price': 210000,
                    'census_tract_2010': tract,
                    'cbsa_id': 'CBSA002',
                    'log_price_relative': np.log(1.05)
                })
        
        repeat_sales_df = pd.DataFrame(data)
        
        # Use subset of geographic data
        geo_subset = geographic_data[geographic_data['cbsa_id'] == 'CBSA002']
        
        generator = SupertractGenerator(geo_subset, min_half_pairs=40)
        
        supertracts = generator.generate_supertracts_for_year(
            repeat_sales_df, 2020, 'CBSA002'
        )
        
        # Find which tracts were merged together
        for st_id, components in supertracts.items():
            if '0603700201' in components:  # Middle tract
                # Should be merged with at least one neighbor
                assert len(components) >= 2
                # Should contain adjacent tract(s)
                assert '0603700200' in components or '0603700202' in components
    
    def test_generate_all_supertracts(self, geographic_data, repeat_sales_data):
        """Test generation of supertracts for multiple years"""
        generator = SupertractGenerator(geographic_data, min_half_pairs=40)
        
        all_supertracts = generator.generate_all_supertracts(
            repeat_sales_data, 2019, 2020
        )
        
        # Should have results for 2 years
        assert len(all_supertracts['year'].unique()) == 2
        
        # Check structure
        assert 'supertract_id' in all_supertracts.columns
        assert 'component_tracts' in all_supertracts.columns
        assert 'half_pairs_count' in all_supertracts.columns
        
        # Each supertract should have unique ID
        assert len(all_supertracts['supertract_id'].unique()) == len(all_supertracts)
        
        # Half-pairs counts should be calculated
        assert all(all_supertracts['half_pairs_count'] >= 0)