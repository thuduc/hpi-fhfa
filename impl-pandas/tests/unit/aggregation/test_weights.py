"""Unit tests for weights module."""

import pytest
import pandas as pd
import numpy as np
from hpi_fhfa.aggregation.weights import WeightCalculator, WeightType
from hpi_fhfa.geography.census_tract import CensusTract
from hpi_fhfa.geography.supertract import Supertract


class TestWeightCalculator:
    """Test WeightCalculator functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.tracts = [
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.7128,
                centroid_lon=-74.0060,
                distance_to_cbd=5.5,
                population=1000,
                housing_units=400,
                college_share=0.3,
                nonwhite_share=0.2
            ),
            CensusTract(
                tract_code="12345678902",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678902",
                centroid_lat=40.7130,
                centroid_lon=-74.0062,
                distance_to_cbd=5.4,
                population=2000,
                housing_units=800,
                college_share=0.4,
                nonwhite_share=0.3
            ),
            CensusTract(
                tract_code="12345678903",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678903",
                centroid_lat=40.7135,
                centroid_lon=-74.0070,
                distance_to_cbd=5.3,
                population=1500,
                housing_units=600,
                college_share=0.5,
                nonwhite_share=0.1
            )
        ]
        
        self.supertracts = [
            Supertract(
                supertract_id="ST_001",
                component_tracts=[self.tracts[0]],
                period=2020,
                half_pairs_count=50
            ),
            Supertract(
                supertract_id="ST_002",
                component_tracts=[self.tracts[1], self.tracts[2]],
                period=2020,
                half_pairs_count=100
            )
        ]
        
        self.repeat_sales_data = pd.DataFrame({
            'property_id': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
            'census_tract': ['12345678901'] * 2 + ['12345678902'] * 3 + ['12345678903'],
            'period_1': [2019] * 6,
            'period_2': [2020] * 6,
            'price_relative': [0.05] * 6,
            'days_diff': [365] * 6
        })
    
    def test_calculate_weights_string_type(self):
        """Test weight calculation with string type."""
        weights = WeightCalculator.calculate_weights(
            'sample', self.tracts, self.repeat_sales_data
        )
        
        assert isinstance(weights, dict)
        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(1.0)
    
    def test_calculate_weights_enum_type(self):
        """Test weight calculation with enum type."""
        weights = WeightCalculator.calculate_weights(
            WeightType.SAMPLE, self.tracts, self.repeat_sales_data
        )
        
        assert isinstance(weights, dict)
        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(1.0)
    
    def test_calculate_weights_invalid_type(self):
        """Test error with invalid weight type."""
        with pytest.raises(ValueError, match="Unknown weight type"):
            WeightCalculator.calculate_weights('invalid', self.tracts)
    
    def test_sample_weights_tracts(self):
        """Test sample weights for census tracts."""
        weights = WeightCalculator._calculate_sample_weights(
            self.tracts, self.repeat_sales_data
        )
        
        # Tract 1: 2 pairs = 4 half-pairs
        # Tract 2: 3 pairs = 6 half-pairs
        # Tract 3: 1 pair = 2 half-pairs
        # Total: 12 half-pairs
        assert weights['12345678901'] == pytest.approx(4/12)
        assert weights['12345678902'] == pytest.approx(6/12)
        assert weights['12345678903'] == pytest.approx(2/12)
    
    def test_sample_weights_supertracts(self):
        """Test sample weights for supertracts."""
        weights = WeightCalculator._calculate_sample_weights(
            self.supertracts, self.repeat_sales_data
        )
        
        # ST_001: 50 half-pairs
        # ST_002: 100 half-pairs
        # Total: 150 half-pairs
        assert weights['ST_001'] == pytest.approx(50/150)
        assert weights['ST_002'] == pytest.approx(100/150)
    
    def test_sample_weights_no_data(self):
        """Test sample weights with no repeat sales data."""
        weights = WeightCalculator._calculate_sample_weights(
            self.tracts, None
        )
        
        # Should use equal weights
        assert all(w == pytest.approx(1/3) for w in weights.values())
    
    def test_value_weights(self):
        """Test value-based weights."""
        value_data = pd.DataFrame({
            'tract_code': ['12345678901', '12345678902', '12345678903'],
            'aggregate_value': [1000000, 2000000, 1500000]
        })
        
        weights = WeightCalculator._calculate_value_weights(
            self.tracts, value_data
        )
        
        # Total value: 4.5M
        assert weights['12345678901'] == pytest.approx(1/4.5)
        assert weights['12345678902'] == pytest.approx(2/4.5)
        assert weights['12345678903'] == pytest.approx(1.5/4.5)
    
    def test_value_weights_supertracts(self):
        """Test value weights for supertracts."""
        value_data = pd.DataFrame({
            'tract_code': ['12345678901', '12345678902', '12345678903'],
            'aggregate_value': [1000000, 2000000, 1500000]
        })
        
        weights = WeightCalculator._calculate_value_weights(
            self.supertracts, value_data
        )
        
        # ST_001: 1M
        # ST_002: 3.5M (2M + 1.5M)
        assert weights['ST_001'] == pytest.approx(1/4.5)
        assert weights['ST_002'] == pytest.approx(3.5/4.5)
    
    def test_value_weights_no_data(self):
        """Test value weights with no data."""
        weights = WeightCalculator._calculate_value_weights(
            self.tracts, None
        )
        
        # Should use equal weights
        assert all(w == pytest.approx(1/3) for w in weights.values())
    
    def test_unit_weights(self):
        """Test housing unit weights."""
        weights = WeightCalculator._calculate_unit_weights(self.tracts)
        
        # Total units: 400 + 800 + 600 = 1800
        assert weights['12345678901'] == pytest.approx(400/1800)
        assert weights['12345678902'] == pytest.approx(800/1800)
        assert weights['12345678903'] == pytest.approx(600/1800)
    
    def test_unit_weights_supertracts(self):
        """Test unit weights for supertracts."""
        weights = WeightCalculator._calculate_unit_weights(self.supertracts)
        
        # ST_001: 400 units
        # ST_002: 1400 units (800 + 600)
        assert weights['ST_001'] == pytest.approx(400/1800)
        assert weights['ST_002'] == pytest.approx(1400/1800)
    
    def test_unit_weights_missing_data(self):
        """Test unit weights with missing data."""
        # Create tracts without housing units
        tracts_no_units = [
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.7128,
                centroid_lon=-74.0060,
                distance_to_cbd=5.5
            )
        ]
        
        weights = WeightCalculator._calculate_unit_weights(tracts_no_units)
        
        # Should use equal weights
        assert weights['12345678901'] == 1.0
    
    def test_upb_weights(self):
        """Test UPB weights."""
        upb_data = pd.DataFrame({
            'tract_code': ['12345678901', '12345678902', '12345678903'],
            'upb': [5000000, 10000000, 7500000]
        })
        
        weights = WeightCalculator._calculate_upb_weights(
            self.tracts, upb_data
        )
        
        # Total UPB: 22.5M
        assert weights['12345678901'] == pytest.approx(5/22.5)
        assert weights['12345678902'] == pytest.approx(10/22.5)
        assert weights['12345678903'] == pytest.approx(7.5/22.5)
    
    def test_upb_weights_no_data(self):
        """Test UPB weights with no data."""
        weights = WeightCalculator._calculate_upb_weights(
            self.tracts, None
        )
        
        # Should use equal weights
        assert all(w == pytest.approx(1/3) for w in weights.values())
    
    def test_demographic_weights_college(self):
        """Test college-based demographic weights."""
        weights = WeightCalculator._calculate_demographic_weights(
            self.tracts, 'college'
        )
        
        # Population-weighted college shares
        # Tract 1: 1000 * 0.3 = 300
        # Tract 2: 2000 * 0.4 = 800
        # Tract 3: 1500 * 0.5 = 750
        # Total: 1850
        assert weights['12345678901'] == pytest.approx(300/1850)
        assert weights['12345678902'] == pytest.approx(800/1850)
        assert weights['12345678903'] == pytest.approx(750/1850)
    
    def test_demographic_weights_nonwhite(self):
        """Test nonwhite-based demographic weights."""
        weights = WeightCalculator._calculate_demographic_weights(
            self.tracts, 'nonwhite'
        )
        
        # Population-weighted nonwhite shares
        # Tract 1: 1000 * 0.2 = 200
        # Tract 2: 2000 * 0.3 = 600
        # Tract 3: 1500 * 0.1 = 150
        # Total: 950
        assert weights['12345678901'] == pytest.approx(200/950)
        assert weights['12345678902'] == pytest.approx(600/950)
        assert weights['12345678903'] == pytest.approx(150/950)
    
    def test_demographic_weights_supertracts(self):
        """Test demographic weights for supertracts."""
        weights = WeightCalculator._calculate_demographic_weights(
            self.supertracts, 'college'
        )
        
        # ST_001: 300 (from tract 1)
        # ST_002: 1550 (800 + 750 from tracts 2 & 3)
        assert weights['ST_001'] == pytest.approx(300/1850)
        assert weights['ST_002'] == pytest.approx(1550/1850)
    
    def test_demographic_weights_no_population(self):
        """Test demographic weights with no population data."""
        # Create tracts without population
        tracts_no_pop = [
            CensusTract(
                tract_code="12345678901",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number="678901",
                centroid_lat=40.7128,
                centroid_lon=-74.0060,
                distance_to_cbd=5.5,
                college_share=0.3
            )
        ]
        
        weights = WeightCalculator._calculate_demographic_weights(
            tracts_no_pop, 'college'
        )
        
        # Should use equal weights
        assert weights['12345678901'] == 1.0
    
    def test_validate_weights_valid(self):
        """Test weight validation with valid weights."""
        weights = {
            'tract1': 0.3,
            'tract2': 0.5,
            'tract3': 0.2
        }
        
        assert WeightCalculator.validate_weights(weights)
    
    def test_validate_weights_invalid(self):
        """Test weight validation with invalid weights."""
        weights = {
            'tract1': 0.3,
            'tract2': 0.5,
            'tract3': 0.3  # Sum > 1
        }
        
        assert not WeightCalculator.validate_weights(weights)
    
    def test_validate_weights_empty(self):
        """Test weight validation with empty weights."""
        assert not WeightCalculator.validate_weights({})
    
    def test_combine_weights_single(self):
        """Test combining single weight set."""
        weights = {'tract1': 0.5, 'tract2': 0.5}
        combined = WeightCalculator.combine_weights([weights])
        
        assert combined == weights
    
    def test_combine_weights_multiple(self):
        """Test combining multiple weight sets."""
        weights1 = {'tract1': 0.6, 'tract2': 0.4}
        weights2 = {'tract1': 0.4, 'tract2': 0.6}
        
        # Equal combination weights
        combined = WeightCalculator.combine_weights([weights1, weights2])
        
        assert combined['tract1'] == pytest.approx(0.5)
        assert combined['tract2'] == pytest.approx(0.5)
    
    def test_combine_weights_custom_weights(self):
        """Test combining with custom combination weights."""
        weights1 = {'tract1': 0.6, 'tract2': 0.4}
        weights2 = {'tract1': 0.4, 'tract2': 0.6}
        
        # 75% weight1, 25% weight2
        combined = WeightCalculator.combine_weights(
            [weights1, weights2],
            combination_weights=[0.75, 0.25]
        )
        
        assert combined['tract1'] == pytest.approx(0.6 * 0.75 + 0.4 * 0.25)
        assert combined['tract2'] == pytest.approx(0.4 * 0.75 + 0.6 * 0.25)
    
    def test_combine_weights_different_keys(self):
        """Test combining weights with different keys."""
        weights1 = {'tract1': 0.5, 'tract2': 0.5}
        weights2 = {'tract2': 0.6, 'tract3': 0.4}
        
        combined = WeightCalculator.combine_weights([weights1, weights2])
        
        # Should include all keys
        assert 'tract1' in combined
        assert 'tract2' in combined
        assert 'tract3' in combined
        
        # Should sum to 1
        assert sum(combined.values()) == pytest.approx(1.0)
    
    def test_combine_weights_invalid_combination_weights(self):
        """Test error with invalid combination weights."""
        weights1 = {'tract1': 0.5, 'tract2': 0.5}
        weights2 = {'tract1': 0.5, 'tract2': 0.5}
        
        # Wrong number of weights
        with pytest.raises(ValueError, match="must match number"):
            WeightCalculator.combine_weights(
                [weights1, weights2],
                combination_weights=[0.5]
            )
        
        # Weights don't sum to 1
        with pytest.raises(ValueError, match="must sum to 1.0"):
            WeightCalculator.combine_weights(
                [weights1, weights2],
                combination_weights=[0.6, 0.6]
            )
    
    def test_combine_weights_empty(self):
        """Test combining empty weight list."""
        combined = WeightCalculator.combine_weights([])
        assert combined == {}
    
    def test_all_weight_types(self):
        """Test all weight types work correctly."""
        # Additional data for value and UPB weights
        value_data = pd.DataFrame({
            'tract_code': ['12345678901', '12345678902', '12345678903'],
            'aggregate_value': [1000000, 2000000, 1500000]
        })
        
        upb_data = pd.DataFrame({
            'tract_code': ['12345678901', '12345678902', '12345678903'],
            'upb': [5000000, 10000000, 7500000]
        })
        
        # Test each weight type
        for weight_type in WeightType:
            if weight_type == WeightType.VALUE:
                additional_data = value_data
            elif weight_type == WeightType.UPB:
                additional_data = upb_data
            else:
                additional_data = None
            
            weights = WeightCalculator.calculate_weights(
                weight_type,
                self.tracts,
                self.repeat_sales_data,
                additional_data
            )
            
            assert isinstance(weights, dict)
            assert len(weights) == 3
            assert sum(weights.values()) == pytest.approx(1.0)
            assert all(w >= 0 for w in weights.values())