"""Unit tests for weighting schemes."""

import pytest
import polars as pl
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.hpi_fhfa.models.weighting import (
    WeightingScheme, SampleWeights, ValueWeights, UnitWeights,
    CollegeWeights, NonWhiteWeights, WeightingFactory
)


class TestWeightingSchemes:
    """Test weighting scheme implementations."""
    
    @pytest.fixture
    def supertract_df(self):
        """Create sample supertract mapping data."""
        return pl.DataFrame({
            "tract_id": ["06037000100", "06037000200", "06037000300", "06037000400"],
            "supertract_id": ["S2020_001", "S2020_001", "S2020_002", "S2020_002"],
            "period": [2020, 2020, 2020, 2020],
            "n_component_tracts": [2, 2, 2, 2],
            "total_half_pairs": [100, 100, 50, 50]  # S2020_001: 200, S2020_002: 100
        })
    
    def test_normalize_weights(self):
        """Test weight normalization."""
        weights_df = pl.DataFrame({
            "supertract_id": ["S001", "S002", "S003"],
            "weight": [10.0, 20.0, 30.0]
        })
        
        scheme = SampleWeights()
        normalized = scheme.normalize_weights(weights_df)
        
        # Weights should sum to 1
        assert abs(normalized["weight"].sum() - 1.0) < 1e-10
        
        # Check individual weights
        assert abs(normalized["weight"][0] - 10/60) < 1e-10
        assert abs(normalized["weight"][1] - 20/60) < 1e-10
        assert abs(normalized["weight"][2] - 30/60) < 1e-10
    
    def test_normalize_weights_all_zero(self):
        """Test normalization when all weights are zero."""
        weights_df = pl.DataFrame({
            "supertract_id": ["S001", "S002"],
            "weight": [0.0, 0.0]
        })
        
        scheme = SampleWeights()
        normalized = scheme.normalize_weights(weights_df)
        
        # Should give equal weights
        assert abs(normalized["weight"][0] - 0.5) < 1e-10
        assert abs(normalized["weight"][1] - 0.5) < 1e-10
    
    def test_sample_weights(self, supertract_df):
        """Test sample-based weighting."""
        scheme = SampleWeights()
        weights = scheme.calculate_weights(supertract_df, period=2020)
        
        assert len(weights) == 2  # Two supertracts
        assert abs(weights["weight"].sum() - 1.0) < 1e-10
        
        # S2020_001 has 200 half-pairs, S2020_002 has 100
        # So weights should be 2/3 and 1/3
        w1 = weights.filter(pl.col("supertract_id") == "S2020_001")["weight"][0]
        w2 = weights.filter(pl.col("supertract_id") == "S2020_002")["weight"][0]
        
        assert abs(w1 - 2/3) < 1e-10
        assert abs(w2 - 1/3) < 1e-10
    
    def test_value_weights(self, supertract_df, sample_geographic_data):
        """Test value-based weighting."""
        scheme = ValueWeights()
        
        # Need to match tract IDs
        geographic_df = sample_geographic_data.head(4).with_columns(
            pl.Series("tract_id", ["06037000100", "06037000200", "06037000300", "06037000400"])
        )
        
        weights = scheme.calculate_weights(
            supertract_df, period=2020, geographic_df=geographic_df
        )
        
        assert len(weights) == 2
        assert abs(weights["weight"].sum() - 1.0) < 1e-10
        
        # Weights should be proportional to sum of housing values
    
    def test_value_weights_missing_geographic(self, supertract_df):
        """Test value weights without geographic data."""
        scheme = ValueWeights()
        
        with pytest.raises(ValueError, match="Geographic data required"):
            scheme.calculate_weights(supertract_df, period=2020)
    
    def test_unit_weights(self, supertract_df, sample_geographic_data):
        """Test unit-based weighting."""
        scheme = UnitWeights()
        
        geographic_df = sample_geographic_data.head(4).with_columns(
            pl.Series("tract_id", ["06037000100", "06037000200", "06037000300", "06037000400"])
        )
        
        weights = scheme.calculate_weights(
            supertract_df, period=2020, geographic_df=geographic_df
        )
        
        assert len(weights) == 2
        assert abs(weights["weight"].sum() - 1.0) < 1e-10
    
    def test_college_weights(self, supertract_df, sample_geographic_data):
        """Test college-based weighting."""
        scheme = CollegeWeights()
        
        # Set specific college shares for testing
        geographic_df = sample_geographic_data.head(4).with_columns([
            pl.Series("tract_id", ["06037000100", "06037000200", "06037000300", "06037000400"]),
            pl.Series("college_share", [0.5, 0.3, 0.2, 0.1]),  # Different shares
            pl.Series("housing_units", [1000, 1000, 1000, 1000])  # Same units
        ])
        
        weights = scheme.calculate_weights(
            supertract_df, period=2020, geographic_df=geographic_df
        )
        
        assert len(weights) == 2
        assert abs(weights["weight"].sum() - 1.0) < 1e-10
        
        # S2020_001 has tracts with 0.5 and 0.3 college share = 800 college pop
        # S2020_002 has tracts with 0.2 and 0.1 college share = 300 college pop
        # Weights should be 800/1100 and 300/1100
        w1 = weights.filter(pl.col("supertract_id") == "S2020_001")["weight"][0]
        w2 = weights.filter(pl.col("supertract_id") == "S2020_002")["weight"][0]
        
        assert abs(w1 - 800/1100) < 1e-3
        assert abs(w2 - 300/1100) < 1e-3
    
    def test_nonwhite_weights(self, supertract_df, sample_geographic_data):
        """Test non-white population weighting."""
        scheme = NonWhiteWeights()
        
        geographic_df = sample_geographic_data.head(4).with_columns([
            pl.Series("tract_id", ["06037000100", "06037000200", "06037000300", "06037000400"]),
            pl.Series("nonwhite_share", [0.6, 0.4, 0.3, 0.2]),
            pl.Series("housing_units", [1000, 1000, 1000, 1000])
        ])
        
        weights = scheme.calculate_weights(
            supertract_df, period=2020, geographic_df=geographic_df
        )
        
        assert len(weights) == 2
        assert abs(weights["weight"].sum() - 1.0) < 1e-10
    
    def test_weighting_factory_create(self):
        """Test factory creation of weighting schemes."""
        # Test all valid schemes
        for scheme_name in ["sample", "value", "unit", "upb", "college", "nonwhite"]:
            scheme = WeightingFactory.create(scheme_name)
            assert isinstance(scheme, WeightingScheme)
        
        # Test invalid scheme
        with pytest.raises(ValueError, match="Unknown weighting scheme"):
            WeightingFactory.create("invalid_scheme")
    
    def test_weighting_factory_calculate_all(self, supertract_df, sample_geographic_data):
        """Test calculating multiple weighting schemes."""
        geographic_df = sample_geographic_data.head(4).with_columns(
            pl.Series("tract_id", ["06037000100", "06037000200", "06037000300", "06037000400"])
        )
        
        # Skip UPB since it needs transaction data
        scheme_names = ["sample", "value", "unit", "college", "nonwhite"]
        
        weights_dict = WeightingFactory.calculate_all_weights(
            scheme_names,
            supertract_df,
            period=2020,
            geographic_df=geographic_df
        )
        
        assert len(weights_dict) == len(scheme_names)
        
        for scheme_name in scheme_names:
            assert scheme_name in weights_dict
            weights = weights_dict[scheme_name]
            assert isinstance(weights, pl.DataFrame)
            assert "supertract_id" in weights.columns
            assert "weight" in weights.columns
            assert abs(weights["weight"].sum() - 1.0) < 1e-10