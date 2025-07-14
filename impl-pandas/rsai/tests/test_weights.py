"""Unit tests for weighting schemes"""

import pytest
import pandas as pd
import numpy as np

from rsai.src.index.weights import (
    WeightCalculator, SampleWeighting, ValueWeighting,
    UnitWeighting, UPBWeighting, CollegeWeighting, NonWhiteWeighting
)


class TestWeightingSchemes:
    """Test individual weighting schemes"""
    
    @pytest.fixture
    def supertract_data(self):
        """Create sample supertract data"""
        data = []
        
        for year in [2019, 2020]:
            for i in range(5):
                data.append({
                    'supertract_id': f'31080_{year}_ST{i:04d}',
                    'year': year,
                    'cbsa_id': '31080',
                    'component_tracts': [f'0603712345{i}', f'0603712345{i+5}'],
                    'half_pairs_count': 40 + i * 10  # 40, 50, 60, 70, 80
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def weighting_data(self):
        """Create sample weighting data"""
        data = []
        
        # Create data for 10 tracts over 2 years
        for year in [2019, 2020]:
            for i in range(10):
                data.append({
                    'census_tract_2010': f'0603712345{i}',
                    'year': year,
                    'total_housing_units': 1000 + i * 100,
                    'total_housing_value': 500000000 + i * 50000000,
                    'total_upb': 400000000 + i * 40000000,
                    'college_population': 2000 + i * 200,
                    'non_white_population': 1500 + i * 150
                })
        
        # Add 2010 data for demographic weights
        for i in range(10):
            data.append({
                'census_tract_2010': f'0603712345{i}',
                'year': 2010,
                'total_housing_units': None,
                'total_housing_value': None,
                'total_upb': None,
                'college_population': 2000 + i * 200,
                'non_white_population': 1500 + i * 150
            })
        
        return pd.DataFrame(data)
    
    def test_sample_weighting(self, supertract_data):
        """Test sample-based weighting"""
        weighting = SampleWeighting()
        
        weights = weighting.calculate_weights(supertract_data, 2020)
        
        # Check weights sum to 1
        assert abs(weights.sum() - 1.0) < 0.0001
        
        # Check weights are proportional to half-pairs
        year_data = supertract_data[supertract_data['year'] == 2020]
        expected_weights = year_data.set_index('supertract_id')['half_pairs_count'] / year_data['half_pairs_count'].sum()
        
        for st_id in weights.index:
            assert abs(weights[st_id] - expected_weights[st_id]) < 0.0001
    
    def test_value_weighting(self, supertract_data, weighting_data):
        """Test value-based (Laspeyres) weighting"""
        weighting = ValueWeighting()
        
        weights = weighting.calculate_weights(
            supertract_data, 2020, weighting_data=weighting_data
        )
        
        # Check weights sum to 1
        assert abs(weights.sum() - 1.0) < 0.0001
        
        # Verify it uses previous year (2019) values
        # Supertract with higher-indexed tracts should have higher weight
        weights_list = weights.sort_index().values
        assert weights_list[-1] > weights_list[0]  # Last should be larger
    
    def test_unit_weighting(self, supertract_data, weighting_data):
        """Test unit-based weighting"""
        weighting = UnitWeighting()
        
        weights = weighting.calculate_weights(
            supertract_data, 2020, weighting_data=weighting_data
        )
        
        # Check weights sum to 1
        assert abs(weights.sum() - 1.0) < 0.0001
        
        # All weights should be positive
        assert all(weights > 0)
    
    def test_upb_weighting(self, supertract_data, weighting_data):
        """Test UPB weighting"""
        weighting = UPBWeighting()
        
        weights = weighting.calculate_weights(
            supertract_data, 2020, weighting_data=weighting_data
        )
        
        # Check weights sum to 1
        assert abs(weights.sum() - 1.0) < 0.0001
        
        # All weights should be positive
        assert all(weights > 0)
    
    def test_college_weighting(self, supertract_data, weighting_data):
        """Test college population weighting"""
        weighting = CollegeWeighting()
        
        weights = weighting.calculate_weights(
            supertract_data, 2020, weighting_data=weighting_data
        )
        
        # Check weights sum to 1
        assert abs(weights.sum() - 1.0) < 0.0001
        
        # Should use 2010 demographic data
        # All weights should be positive
        assert all(weights > 0)
    
    def test_non_white_weighting(self, supertract_data, weighting_data):
        """Test non-white population weighting"""
        weighting = NonWhiteWeighting()
        
        weights = weighting.calculate_weights(
            supertract_data, 2020, weighting_data=weighting_data
        )
        
        # Check weights sum to 1
        assert abs(weights.sum() - 1.0) < 0.0001
        
        # All weights should be positive
        assert all(weights > 0)
    
    def test_missing_weighting_data_error(self, supertract_data):
        """Test error when weighting data is missing"""
        weighting = ValueWeighting()
        
        with pytest.raises(ValueError, match="weighting_data required"):
            weighting.calculate_weights(supertract_data, 2020)
    
    def test_zero_weights_handling(self):
        """Test handling when all weights are zero"""
        # Create data where all values are zero
        supertract_data = pd.DataFrame([{
            'supertract_id': f'31080_2020_ST{i:04d}',
            'year': 2020,
            'cbsa_id': '31080',
            'component_tracts': [f'tract{i}'],
            'half_pairs_count': 0
        } for i in range(3)])
        
        weighting = SampleWeighting()
        weights = weighting.calculate_weights(supertract_data, 2020)
        
        # Should return equal weights
        assert abs(weights.sum() - 1.0) < 0.0001
        assert all(abs(w - 1/3) < 0.0001 for w in weights)


class TestWeightCalculator:
    """Test WeightCalculator factory class"""
    
    @pytest.fixture
    def calculator(self):
        return WeightCalculator()
    
    @pytest.fixture
    def sample_data(self):
        """Create minimal sample data"""
        supertract_data = pd.DataFrame([{
            'supertract_id': '31080_2020_ST0001',
            'year': 2020,
            'cbsa_id': '31080',
            'component_tracts': ['tract1'],
            'half_pairs_count': 50
        }])
        
        weighting_data = pd.DataFrame([{
            'census_tract_2010': 'tract1',
            'year': 2020,
            'total_housing_units': 1000,
            'total_housing_value': 500000000,
            'total_upb': 400000000
        }, {
            'census_tract_2010': 'tract1',
            'year': 2010,
            'college_population': 2000,
            'non_white_population': 1500
        }])
        
        return supertract_data, weighting_data
    
    def test_calculate_weights_sample(self, calculator, sample_data):
        """Test calculating sample weights through factory"""
        supertract_data, _ = sample_data
        
        weights = calculator.calculate_weights('sample', supertract_data, 2020)
        
        assert len(weights) == 1
        assert weights.iloc[0] == 1.0  # Only one supertract
    
    def test_calculate_weights_all_schemes(self, calculator, sample_data):
        """Test calculating weights for all schemes"""
        supertract_data, weighting_data = sample_data
        
        schemes = ['sample', 'value', 'unit', 'upb', 'college', 'non_white']
        
        for scheme in schemes:
            weights = calculator.calculate_weights(
                scheme, supertract_data, 2020, weighting_data
            )
            assert len(weights) == 1
            assert weights.iloc[0] == 1.0
    
    def test_unknown_scheme_error(self, calculator, sample_data):
        """Test error for unknown weighting scheme"""
        supertract_data, _ = sample_data
        
        with pytest.raises(ValueError, match="Unknown weighting scheme"):
            calculator.calculate_weights('invalid_scheme', supertract_data, 2020)
    
    def test_calculate_all_weights(self, calculator, sample_data):
        """Test calculating all weights at once"""
        supertract_data, weighting_data = sample_data
        
        all_weights = calculator.calculate_all_weights(
            supertract_data, 2020, weighting_data
        )
        
        # Should have all schemes as columns
        expected_schemes = ['sample', 'value', 'unit', 'upb', 'college', 'non_white']
        assert all(scheme in all_weights.columns for scheme in expected_schemes)
        
        # All weights should be 1.0 (only one supertract)
        assert all(all_weights.iloc[0] == 1.0)
    
    def test_custom_scheme(self, calculator, sample_data):
        """Test adding custom weighting scheme"""
        supertract_data, _ = sample_data
        
        # Create custom weighting scheme
        class CustomWeighting(SampleWeighting):
            def calculate_weights(self, supertract_data, year, **kwargs):
                # Just return equal weights
                year_data = supertract_data[supertract_data['year'] == year]
                weights = pd.Series(1.0, index=year_data['supertract_id'])
                return self.normalize_weights(weights)
        
        # Add custom scheme
        calculator.add_custom_scheme('custom', CustomWeighting())
        
        # Use custom scheme
        weights = calculator.calculate_weights('custom', supertract_data, 2020)
        assert len(weights) == 1
        assert weights.iloc[0] == 1.0