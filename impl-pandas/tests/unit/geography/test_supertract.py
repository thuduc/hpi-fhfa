"""Unit tests for supertract module."""

import pytest
import pandas as pd
import numpy as np
from hpi_fhfa.geography.census_tract import CensusTract
from hpi_fhfa.geography.supertract import Supertract, SupertractAlgorithm
from hpi_fhfa.config.constants import MIN_HALF_PAIRS


class TestSupertract:
    """Test Supertract class functionality."""
    
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
                college_share=0.3
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
                college_share=0.4
            )
        ]
    
    def test_valid_supertract_creation(self):
        """Test creating a valid supertract."""
        st = Supertract(
            supertract_id="ST_001",
            component_tracts=self.tracts,
            period=2020,
            half_pairs_count=50
        )
        
        assert st.supertract_id == "ST_001"
        assert len(st.component_tracts) == 2
        assert st.period == 2020
        assert st.half_pairs_count == 50
    
    def test_empty_tracts_error(self):
        """Test error when creating supertract with no tracts."""
        with pytest.raises(ValueError, match="must contain at least one tract"):
            Supertract(
                supertract_id="ST_001",
                component_tracts=[],
                period=2020,
                half_pairs_count=50
            )
    
    def test_multiple_cbsa_error(self):
        """Test error when tracts from multiple CBSAs."""
        mixed_tracts = self.tracts + [
            CensusTract(
                tract_code="98765432101",
                cbsa_code="98765",  # Different CBSA
                state_code="98",
                county_code="765",
                tract_number="432101",
                centroid_lat=41.0,
                centroid_lon=-75.0,
                distance_to_cbd=10.0
            )
        ]
        
        with pytest.raises(ValueError, match="multiple CBSAs"):
            Supertract(
                supertract_id="ST_001",
                component_tracts=mixed_tracts,
                period=2020,
                half_pairs_count=50
            )
    
    def test_centroid_calculation(self):
        """Test weighted centroid calculation."""
        st = Supertract(
            supertract_id="ST_001",
            component_tracts=self.tracts,
            period=2020,
            half_pairs_count=50
        )
        
        # With populations 1000 and 2000, weights are 1/3 and 2/3
        expected_lat = (40.7128 * 1/3) + (40.7130 * 2/3)
        expected_lon = (-74.0060 * 1/3) + (-74.0062 * 2/3)
        
        assert st.centroid_lat == pytest.approx(expected_lat)
        assert st.centroid_lon == pytest.approx(expected_lon)
    
    def test_tract_codes_property(self):
        """Test tract codes property."""
        st = Supertract(
            supertract_id="ST_001",
            component_tracts=self.tracts,
            period=2020,
            half_pairs_count=50
        )
        
        codes = st.tract_codes
        assert len(codes) == 2
        assert "12345678901" in codes
        assert "12345678902" in codes
    
    def test_cbsa_code_property(self):
        """Test CBSA code property."""
        st = Supertract(
            supertract_id="ST_001",
            component_tracts=self.tracts,
            period=2020,
            half_pairs_count=50
        )
        
        assert st.cbsa_code == "12345"
    
    def test_is_single_tract(self):
        """Test single tract detection."""
        st_single = Supertract(
            supertract_id="ST_001",
            component_tracts=[self.tracts[0]],
            period=2020,
            half_pairs_count=50
        )
        
        st_multi = Supertract(
            supertract_id="ST_002",
            component_tracts=self.tracts,
            period=2020,
            half_pairs_count=100
        )
        
        assert st_single.is_single_tract
        assert not st_multi.is_single_tract
    
    def test_get_aggregate_weight_share_based(self):
        """Test aggregate weight calculation for share-based weights."""
        st = Supertract(
            supertract_id="ST_001",
            component_tracts=self.tracts,
            period=2020,
            half_pairs_count=50
        )
        
        # College share: weighted average by population
        # (0.3 * 1000 + 0.4 * 2000) / 3000 = 1100/3000 ≈ 0.367
        college_weight = st.get_aggregate_weight('college')
        assert college_weight == pytest.approx(0.367, rel=0.01)
    
    def test_get_aggregate_weight_count_based(self):
        """Test aggregate weight calculation for count-based weights."""
        st = Supertract(
            supertract_id="ST_001",
            component_tracts=self.tracts,
            period=2020,
            half_pairs_count=50
        )
        
        # Population: sum of all tracts
        pop_weight = st.get_aggregate_weight('population')
        assert pop_weight == 3000
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        st = Supertract(
            supertract_id="ST_001",
            component_tracts=self.tracts,
            period=2020,
            half_pairs_count=50,
            metadata={'source': 'test'}
        )
        
        result = st.to_dict()
        assert result['supertract_id'] == "ST_001"
        assert result['period'] == 2020
        assert result['half_pairs_count'] == 50
        assert result['n_tracts'] == 2
        assert result['cbsa_code'] == "12345"
        assert result['source'] == 'test'
    
    def test_string_representations(self):
        """Test string and repr methods."""
        st = Supertract(
            supertract_id="ST_001",
            component_tracts=self.tracts,
            period=2020,
            half_pairs_count=50
        )
        
        assert str(st) == "Supertract(ST_001, tracts=2, half_pairs=50)"
        assert "id='ST_001'" in repr(st)
        assert "period=2020" in repr(st)


class TestSupertractAlgorithm:
    """Test SupertractAlgorithm functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.tracts = [
            CensusTract(
                tract_code=f"1234567890{i}",
                cbsa_code="12345",
                state_code="12",
                county_code="345",
                tract_number=f"67890{i}",
                centroid_lat=40.71 + i * 0.01,
                centroid_lon=-74.00 - i * 0.01,
                distance_to_cbd=5.0 + i * 0.1
            )
            for i in range(5)
        ]
        
        # Create sample repeat sales data
        self.repeat_sales_pairs = pd.DataFrame({
            'property_id': ['P1', 'P2', 'P3', 'P4', 'P5'] * 10,
            'census_tract': ['12345678900'] * 20 + ['12345678901'] * 20 + ['12345678902'] * 10,
            'period_1': [2019] * 25 + [2020] * 25,
            'period_2': [2020] * 25 + [2021] * 25,
            'price_relative': np.random.normal(0.05, 0.02, 50),
            'days_diff': [365] * 50
        })
    
    def test_algorithm_initialization(self):
        """Test algorithm initialization."""
        algo = SupertractAlgorithm(min_half_pairs=40, max_merge_distance=10.0)
        
        assert algo.min_half_pairs == 40
        assert algo.max_merge_distance == 10.0
        assert algo.prefer_adjacent
    
    def test_build_supertracts_all_sufficient(self):
        """Test building supertracts when all have sufficient data."""
        algo = SupertractAlgorithm(min_half_pairs=5)  # Low threshold
        
        supertracts = algo.build_supertracts(
            self.tracts[:3],
            self.repeat_sales_pairs,
            period=2020
        )
        
        # Each tract should be its own supertract
        assert len(supertracts) == 3
        assert all(st.is_single_tract for st in supertracts)
    
    def test_build_supertracts_merging_required(self):
        """Test building supertracts with merging."""
        # Create data with insufficient pairs for some tracts
        sparse_pairs = pd.DataFrame({
            'property_id': ['P1', 'P2'],
            'census_tract': ['12345678900', '12345678901'],
            'period_1': [2019, 2019],
            'period_2': [2020, 2020],
            'price_relative': [0.05, 0.04],
            'days_diff': [365, 365]
        })
        
        algo = SupertractAlgorithm(min_half_pairs=40)
        
        supertracts = algo.build_supertracts(
            self.tracts[:3],
            sparse_pairs,
            period=2020
        )
        
        # Should have fewer supertracts than original tracts
        assert len(supertracts) < 3
        # At least one should be merged
        assert any(not st.is_single_tract for st in supertracts)
    
    def test_build_supertracts_empty_tracts(self):
        """Test with empty tract list."""
        algo = SupertractAlgorithm()
        supertracts = algo.build_supertracts([], self.repeat_sales_pairs, 2020)
        assert supertracts == []
    
    def test_calculate_tract_half_pairs(self):
        """Test half-pairs calculation for tracts."""
        algo = SupertractAlgorithm()
        
        half_pairs = algo._calculate_tract_half_pairs(
            self.tracts[:3],
            self.repeat_sales_pairs,
            period=2020
        )
        
        assert isinstance(half_pairs, dict)
        assert len(half_pairs) == 3
        # Tract 0 has 20 pairs involving 2020
        assert half_pairs.get('12345678900', 0) > 0
    
    def test_find_merge_candidate(self):
        """Test finding merge candidates."""
        algo = SupertractAlgorithm(max_merge_distance=10.0)
        
        # Create supertracts with different half-pair counts
        st1 = Supertract(
            supertract_id="ST_001",
            component_tracts=[self.tracts[0]],
            period=2020,
            half_pairs_count=20  # Below threshold
        )
        
        st2 = Supertract(
            supertract_id="ST_002",
            component_tracts=[self.tracts[1]],
            period=2020,
            half_pairs_count=30  # Also below threshold
        )
        
        st3 = Supertract(
            supertract_id="ST_003",
            component_tracts=[self.tracts[4]],  # Far away
            period=2020,
            half_pairs_count=25
        )
        
        all_supertracts = [st1, st2, st3]
        
        candidate = algo._find_merge_candidate(st1, all_supertracts, self.tracts)
        
        # Should find st2 as best candidate (close and would meet threshold)
        assert candidate == st2
    
    def test_merge_supertracts(self):
        """Test merging two supertracts."""
        algo = SupertractAlgorithm()
        
        st1 = Supertract(
            supertract_id="ST_001",
            component_tracts=[self.tracts[0]],
            period=2020,
            half_pairs_count=20
        )
        
        st2 = Supertract(
            supertract_id="ST_002",
            component_tracts=[self.tracts[1]],
            period=2020,
            half_pairs_count=30
        )
        
        tract_half_pairs = {
            '12345678900': 20,
            '12345678901': 30
        }
        
        merged = algo._merge_supertracts(st1, st2, 2020, tract_half_pairs)
        
        assert len(merged.component_tracts) == 2
        assert merged.half_pairs_count == 50
        assert 'merged_from' in merged.metadata
        assert merged.metadata['n_merges'] == 1
    
    def test_build_supertracts_multi_period(self):
        """Test building supertracts for multiple periods."""
        algo = SupertractAlgorithm(min_half_pairs=5)
        
        periods = [2019, 2020, 2021]
        results = algo.build_supertracts_multi_period(
            self.tracts[:3],
            self.repeat_sales_pairs,
            periods
        )
        
        assert len(results) == 3
        assert all(period in results for period in periods)
        assert all(isinstance(supertracts, list) for supertracts in results.values())
    
    def test_merge_candidate_scoring(self):
        """Test merge candidate scoring logic."""
        algo = SupertractAlgorithm(min_half_pairs=40)
        
        # Target with 20 half-pairs
        target = Supertract(
            supertract_id="ST_TARGET",
            component_tracts=[self.tracts[0]],
            period=2020,
            half_pairs_count=20
        )
        
        # Candidate that would just meet threshold (better)
        st_optimal = Supertract(
            supertract_id="ST_OPTIMAL",
            component_tracts=[self.tracts[1]],
            period=2020,
            half_pairs_count=25  # 20 + 25 = 45, just above 40
        )
        
        # Candidate that would far exceed threshold (worse)
        st_excess = Supertract(
            supertract_id="ST_EXCESS",
            component_tracts=[self.tracts[2]],
            period=2020,
            half_pairs_count=80  # 20 + 80 = 100, way above 40
        )
        
        all_supertracts = [target, st_optimal, st_excess]
        
        candidate = algo._find_merge_candidate(target, all_supertracts, self.tracts)
        
        # Should prefer the optimal candidate
        assert candidate == st_optimal