"""
Unit tests for weighting schemes.

Tests various weighting schemes including equal weights, value weights,
Case-Shiller weights, and custom weighting functions.
"""

import pytest
from datetime import date, timedelta
import polars as pl
import numpy as np

from rsai.src.index.weights import WeightCalculator
from rsai.src.data.models import WeightingScheme


class TestWeightCalculator:
    """Test WeightCalculator class."""
    
    def test_initialization(self):
        """Test WeightCalculator initialization."""
        calc = WeightCalculator(WeightingScheme.EQUAL)
        assert calc.scheme == WeightingScheme.EQUAL
        assert calc.custom_weight_func is None
        assert len(calc.weight_functions) == 5
    
    def test_equal_weights(self, sample_repeat_sales_df):
        """Test equal weighting scheme."""
        calc = WeightCalculator(WeightingScheme.EQUAL)
        
        weighted_df = calc.calculate_weights(sample_repeat_sales_df)
        
        assert "weight" in weighted_df.columns
        assert "weight_type" in weighted_df.columns
        
        # All weights should be 1.0
        assert all(w == 1.0 for w in weighted_df["weight"])
        assert all(wt == "equal" for wt in weighted_df["weight_type"])
    
    def test_value_weights(self, sample_repeat_sales_df):
        """Test value-based weighting scheme."""
        calc = WeightCalculator(WeightingScheme.VALUE)
        
        weighted_df = calc.calculate_weights(sample_repeat_sales_df)
        
        assert "weight" in weighted_df.columns
        assert "weight_type" in weighted_df.columns
        
        # Weights should be proportional to average sale value
        avg_values = (
            (sample_repeat_sales_df["sale1_price"] + 
             sample_repeat_sales_df["sale2_price"]) / 2
        )
        median_value = avg_values.median()
        
        # Check that higher value properties have higher weights
        high_value_mask = avg_values > median_value * 1.5
        low_value_mask = avg_values < median_value * 0.5
        
        if high_value_mask.sum() > 0 and low_value_mask.sum() > 0:
            avg_high_weight = weighted_df.filter(high_value_mask)["weight"].mean()
            avg_low_weight = weighted_df.filter(low_value_mask)["weight"].mean()
            assert avg_high_weight > avg_low_weight
    
    def test_value_weights_capping(self):
        """Test that value weights are capped to avoid outliers."""
        # Create data with extreme values
        df = pl.DataFrame({
            "pair_id": ["P1", "P2", "P3", "P4"],
            "sale1_price": [100000, 200000, 300000, 10000000],  # Last is extreme
            "sale2_price": [120000, 250000, 350000, 12000000],
            "property_id": ["PROP1", "PROP2", "PROP3", "PROP4"],
            "sale1_date": [date(2020, 1, 1)] * 4,
            "sale2_date": [date(2021, 1, 1)] * 4,
            "price_ratio": [1.2, 1.25, 1.17, 1.2],
            "log_price_ratio": [np.log(1.2), np.log(1.25), np.log(1.17), np.log(1.2)],
            "holding_period_days": [365] * 4,
            "annualized_return": [0.2, 0.25, 0.17, 0.2],
            "sale1_transaction_id": ["T1", "T3", "T5", "T7"],
            "sale2_transaction_id": ["T2", "T4", "T6", "T8"],
            "is_valid": [True] * 4,
            "validation_flags": [[]] * 4
        })
        
        calc = WeightCalculator(WeightingScheme.VALUE)
        weighted_df = calc.calculate_weights(df)
        
        # Check that extreme value weight is capped
        weights = weighted_df["weight"]
        max_weight = weights.max()
        
        # The extreme value should not dominate
        assert max_weight < weights.mean() * 5
    
    def test_case_shiller_weights(self, sample_repeat_sales_df):
        """Test Case-Shiller weighting scheme."""
        calc = WeightCalculator(WeightingScheme.CASE_SHILLER)
        
        weighted_df = calc.calculate_weights(
            sample_repeat_sales_df,
            interval_correction=True,
            heteroscedasticity_correction=True
        )
        
        assert "weight" in weighted_df.columns
        assert "weight_type" in weighted_df.columns
        assert all(wt == "case_shiller" for wt in weighted_df["weight_type"])
        
        # Weights should have mean close to 1 (normalized)
        assert weighted_df["weight"].mean() == pytest.approx(1.0, rel=0.1)
        
        # Check interval correction - longer holding periods should have lower weights
        long_holding = sample_repeat_sales_df["holding_period_days"] > 365 * 2
        short_holding = sample_repeat_sales_df["holding_period_days"] < 365
        
        if long_holding.sum() > 0 and short_holding.sum() > 0:
            avg_long_weight = weighted_df.filter(long_holding)["weight"].mean()
            avg_short_weight = weighted_df.filter(short_holding)["weight"].mean()
            assert avg_long_weight < avg_short_weight
    
    def test_case_shiller_no_corrections(self, sample_repeat_sales_df):
        """Test Case-Shiller weights without corrections."""
        calc = WeightCalculator(WeightingScheme.CASE_SHILLER)
        
        weighted_df = calc.calculate_weights(
            sample_repeat_sales_df,
            interval_correction=False,
            heteroscedasticity_correction=False
        )
        
        # Without corrections, weights should be close to equal
        weight_std = weighted_df["weight"].std()
        assert weight_std < 0.1  # Low variation
    
    def test_bmn_weights(self, sample_repeat_sales_df):
        """Test BMN weighting scheme."""
        calc = WeightCalculator(WeightingScheme.BMN)
        
        weighted_df = calc.calculate_weights(sample_repeat_sales_df)
        
        assert "weight" in weighted_df.columns
        assert "weight_type" in weighted_df.columns
        assert all(wt == "bmn" for wt in weighted_df["weight_type"])
        
        # BMN weights should decrease with holding period
        # Create bins of holding periods
        short_mask = sample_repeat_sales_df["holding_period_days"] < 365
        long_mask = sample_repeat_sales_df["holding_period_days"] > 365 * 3
        
        if short_mask.sum() > 0 and long_mask.sum() > 0:
            avg_short_weight = weighted_df.filter(short_mask)["weight"].mean()
            avg_long_weight = weighted_df.filter(long_mask)["weight"].mean()
            assert avg_short_weight > avg_long_weight
    
    def test_bmn_weights_with_age(self):
        """Test BMN weights with property age correction."""
        # Create data with property age
        df = pl.DataFrame({
            "pair_id": ["P1", "P2", "P3"],
            "sale1_price": [200000, 250000, 300000],
            "sale2_price": [220000, 280000, 330000],
            "property_age": [5, 25, 50],  # Years
            "holding_period_days": [365, 730, 365],
            "property_id": ["PROP1", "PROP2", "PROP3"],
            "sale1_date": [date(2020, 1, 1)] * 3,
            "sale2_date": [date(2021, 1, 1), date(2022, 1, 1), date(2021, 1, 1)],
            "price_ratio": [1.1, 1.12, 1.1],
            "log_price_ratio": [np.log(1.1), np.log(1.12), np.log(1.1)],
            "annualized_return": [0.1, 0.06, 0.1],
            "sale1_transaction_id": ["T1", "T3", "T5"],
            "sale2_transaction_id": ["T2", "T4", "T6"],
            "is_valid": [True] * 3,
            "validation_flags": [[]] * 3
        })
        
        calc = WeightCalculator(WeightingScheme.BMN)
        weighted_df = calc.calculate_weights(df, age_correction=True)
        
        # Older properties should have lower weights
        weights = weighted_df["weight"].to_list()
        assert weights[0] > weights[2]  # 5 years old > 50 years old
    
    def test_custom_weights(self, sample_repeat_sales_df):
        """Test custom weighting function."""
        def custom_func(df, **kwargs):
            # Simple custom function: weight by holding period
            return df["holding_period_days"] / 365.0
        
        calc = WeightCalculator(WeightingScheme.CUSTOM)
        
        weighted_df = calc.calculate_weights(
            sample_repeat_sales_df,
            weight_func=custom_func
        )
        
        assert "weight" in weighted_df.columns
        assert "weight_type" in weighted_df.columns
        assert all(wt == "custom" for wt in weighted_df["weight_type"])
        
        # Check weights match custom function
        expected_weights = sample_repeat_sales_df["holding_period_days"] / 365.0
        actual_weights = weighted_df["weight"]
        
        for i in range(min(5, len(weighted_df))):
            assert actual_weights[i] == pytest.approx(expected_weights[i], rel=1e-6)
    
    def test_custom_weights_array_return(self, sample_repeat_sales_df):
        """Test custom weight function returning numpy array."""
        def custom_func(df, **kwargs):
            return np.ones(len(df)) * 2.0
        
        calc = WeightCalculator(WeightingScheme.CUSTOM)
        
        weighted_df = calc.calculate_weights(
            sample_repeat_sales_df,
            weight_func=custom_func
        )
        
        assert all(w == 2.0 for w in weighted_df["weight"])
    
    def test_custom_weights_no_function(self, sample_repeat_sales_df):
        """Test error when no custom function provided."""
        calc = WeightCalculator(WeightingScheme.CUSTOM)
        
        with pytest.raises(ValueError, match="No custom weight function"):
            calc.calculate_weights(sample_repeat_sales_df)
    
    def test_geographic_weights(self, sample_repeat_sales_df, sample_properties_df):
        """Test geographic distance-based weights."""
        # Add geographic data to repeat sales
        repeat_with_geo = sample_repeat_sales_df.join(
            sample_properties_df.select(["property_id", "latitude", "longitude"]),
            on="property_id",
            how="left"
        )
        
        calc = WeightCalculator()
        
        # Target location (center of LA)
        target_lat = 34.05
        target_lon = -118.25
        
        weighted_df = calc.geographic_weights(
            repeat_with_geo,
            target_lat=target_lat,
            target_lon=target_lon,
            decay_distance_km=10.0,
            min_weight=0.1
        )
        
        assert "weight" in weighted_df.columns
        assert "distance_km" in weighted_df.columns
        assert "weight_type" in weighted_df.columns
        
        # Weights should be between min_weight and 1
        assert all(0.1 <= w <= 1.0 for w in weighted_df["weight"])
        
        # Closer properties should have higher weights
        close_mask = weighted_df["distance_km"] < 5
        far_mask = weighted_df["distance_km"] > 20
        
        if close_mask.sum() > 0 and far_mask.sum() > 0:
            avg_close_weight = weighted_df.filter(close_mask)["weight"].mean()
            avg_far_weight = weighted_df.filter(far_mask)["weight"].mean()
            assert avg_close_weight > avg_far_weight
    
    def test_geographic_weights_missing_coords(self, sample_repeat_sales_df):
        """Test error handling for missing coordinates."""
        calc = WeightCalculator()
        
        with pytest.raises(ValueError, match="Geographic coordinates required"):
            calc.geographic_weights(
                sample_repeat_sales_df,
                target_lat=34.0,
                target_lon=-118.0
            )
    
    def test_temporal_weights(self, sample_repeat_sales_df):
        """Test temporal distance-based weights."""
        calc = WeightCalculator()
        
        reference_date = date(2021, 6, 1)
        
        weighted_df = calc.temporal_weights(
            sample_repeat_sales_df,
            reference_date=reference_date,
            decay_years=2.0,
            forward_weight=0.5
        )
        
        assert "weight" in weighted_df.columns
        assert "weight_type" in weighted_df.columns
        
        # All weights should be positive
        assert all(w > 0 for w in weighted_df["weight"])
        
        # Sales closer to reference date should have higher weights
        for row in weighted_df.head(10).iter_rows(named=True):
            days_diff = abs((row["sale2_date"] - reference_date).days)
            if days_diff < 180:  # Within 6 months
                assert row["weight"] > 0.7
            elif days_diff > 730:  # More than 2 years
                assert row["weight"] < 0.7  # Relaxed threshold
    
    def test_quality_adjusted_weights(self, sample_repeat_sales_df):
        """Test quality-based weights."""
        calc = WeightCalculator()
        
        # Test with automatic quality scoring
        weighted_df = calc.quality_adjusted_weights(
            sample_repeat_sales_df,
            min_quality=0.1
        )
        
        assert "weight" in weighted_df.columns
        assert "weight_type" in weighted_df.columns
        assert "quality_score" in weighted_df.columns
        
        # All weights should be between min and 1
        assert all(0.1 <= w <= 1.0 for w in weighted_df["weight"])
    
    def test_quality_adjusted_weights_external_scores(self, sample_repeat_sales_df):
        """Test quality weights with external quality scores."""
        # Create quality scores
        quality_scores = pl.DataFrame({
            "pair_id": sample_repeat_sales_df["pair_id"],
            "quality_score": np.random.uniform(0.5, 1.0, len(sample_repeat_sales_df))
        })
        
        calc = WeightCalculator()
        
        weighted_df = calc.quality_adjusted_weights(
            sample_repeat_sales_df,
            quality_scores=quality_scores,
            min_quality=0.3
        )
        
        # Weights should match quality scores (clipped to min)
        for i in range(min(5, len(weighted_df))):
            expected = max(0.3, quality_scores["quality_score"][i])
            assert weighted_df["weight"][i] == pytest.approx(expected, rel=1e-6)
    
    def test_combine_weights(self, sample_repeat_sales_df):
        """Test combining multiple weighting schemes."""
        calc = WeightCalculator()
        
        weight_schemes = {
            WeightingScheme.EQUAL: 0.5,
            WeightingScheme.VALUE: 0.3,
            WeightingScheme.BMN: 0.2
        }
        
        weighted_df = calc.combine_weights(
            sample_repeat_sales_df,
            weight_schemes
        )
        
        assert "weight" in weighted_df.columns
        assert "weight_type" in weighted_df.columns
        assert all(wt == "combined" for wt in weighted_df["weight_type"])
        
        # Combined weights should be positive
        assert all(w > 0 for w in weighted_df["weight"])
    
    def test_diagnose_weights(self, sample_repeat_sales_df):
        """Test weight diagnostics."""
        calc = WeightCalculator(WeightingScheme.VALUE)
        
        weighted_df = calc.calculate_weights(sample_repeat_sales_df)
        diagnostics = calc.diagnose_weights(weighted_df)
        
        assert isinstance(diagnostics, dict)
        assert "mean" in diagnostics
        assert "std" in diagnostics
        assert "min" in diagnostics
        assert "max" in diagnostics
        assert "cv" in diagnostics
        assert "percentiles" in diagnostics
        assert "zero_weights" in diagnostics
        assert "extreme_weights" in diagnostics
        assert "effective_sample_size" in diagnostics
        assert "warnings" in diagnostics
        
        # Check calculations
        weights = weighted_df["weight"]
        assert diagnostics["mean"] == pytest.approx(weights.mean(), rel=1e-6)
        assert diagnostics["std"] == pytest.approx(weights.std(), rel=1e-6)
        assert diagnostics["min"] == pytest.approx(weights.min(), rel=1e-6)
        assert diagnostics["max"] == pytest.approx(weights.max(), rel=1e-6)
    
    def test_diagnose_weights_warnings(self):
        """Test weight diagnostic warnings."""
        # Create data with problematic weights
        df = pl.DataFrame({
            "weight": [0.0, 0.0, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 100.0]  # Zero weights and extreme outlier
        })
        
        calc = WeightCalculator()
        diagnostics = calc.diagnose_weights(df)
        
        assert len(diagnostics["warnings"]) > 0
        assert any("zero weight" in w for w in diagnostics["warnings"])
        assert any("Extreme weight outliers" in w for w in diagnostics["warnings"])
    
    def test_unknown_weighting_scheme(self, sample_repeat_sales_df):
        """Test error handling for unknown weighting scheme."""
        calc = WeightCalculator()
        calc.scheme = "invalid_scheme"  # Force invalid scheme
        
        with pytest.raises(ValueError, match="Unknown weighting scheme"):
            calc.calculate_weights(sample_repeat_sales_df)