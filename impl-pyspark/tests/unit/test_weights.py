"""
Unit tests for weighting schemes module.
"""

import pytest
from datetime import date
import numpy as np

from pyspark.sql import functions as F

from rsai.src.index.weights import WeightCalculator
from rsai.src.data.models import WeightingScheme


class TestWeightCalculator:
    """Test WeightCalculator class."""
    
    def test_equal_weights(self, spark, sample_repeat_sales_df):
        """Test equal weighting scheme."""
        calc = WeightCalculator(spark, WeightingScheme.EQUAL)
        
        # Calculate weights
        weighted_df = calc.calculate_weights(sample_repeat_sales_df)
        
        # Check that weights were added
        assert "weight" in weighted_df.columns
        assert "weight_type" in weighted_df.columns
        
        # All weights should be 1.0
        weights = weighted_df.select("weight").collect()
        assert all(row["weight"] == 1.0 for row in weights)
        
        # Weight type should be "equal"
        types = weighted_df.select("weight_type").distinct().collect()
        assert len(types) == 1
        assert types[0]["weight_type"] == "equal"
        
    def test_value_weights(self, spark, sample_repeat_sales_df):
        """Test value-based weighting scheme."""
        calc = WeightCalculator(spark, WeightingScheme.VALUE)
        
        # Calculate weights
        weighted_df = calc.calculate_weights(sample_repeat_sales_df)
        
        # Weights should vary based on property value
        weights = weighted_df.select("weight", "sale1_price", "sale2_price").collect()
        weight_values = [row["weight"] for row in weights]
        
        # Should have different weights
        assert len(set(weight_values)) > 1
        
        # Higher value properties should have higher weights (generally)
        # Get average prices
        avg_prices = []
        for row in weights:
            avg_price = (row["sale1_price"] + row["sale2_price"]) / 2.0
            avg_prices.append((avg_price, row["weight"]))
            
        # Sort by price
        avg_prices.sort(key=lambda x: x[0])
        
        # Weights should be capped between 0.2 and 5.0
        assert all(0.2 <= w <= 5.0 for _, w in avg_prices)
        
    def test_case_shiller_weights(self, spark, sample_repeat_sales_df):
        """Test Case-Shiller weighting scheme."""
        calc = WeightCalculator(spark, WeightingScheme.CASE_SHILLER)
        
        # Calculate weights with both corrections
        weighted_df = calc.calculate_weights(
            sample_repeat_sales_df,
            interval_correction=True,
            heteroscedasticity_correction=True
        )
        
        # Check that weights were calculated
        assert weighted_df.filter(F.col("weight") > 0).count() == sample_repeat_sales_df.count()
        
        # Weights should be normalized to mean 1
        mean_weight = weighted_df.agg(F.mean("weight")).collect()[0][0]
        assert abs(mean_weight - 1.0) < 0.1
        
        # Test without corrections
        calc_no_corr = WeightCalculator(spark, WeightingScheme.CASE_SHILLER)
        weighted_no_corr = calc_no_corr.calculate_weights(
            sample_repeat_sales_df,
            interval_correction=False,
            heteroscedasticity_correction=False
        )
        
        # All weights should be 1.0 without corrections
        weights_no_corr = weighted_no_corr.select("weight").collect()
        assert all(abs(row["weight"] - 1.0) < 0.01 for row in weights_no_corr)
        
    def test_bmn_weights(self, spark, sample_repeat_sales_df):
        """Test BMN temporal weighting scheme."""
        calc = WeightCalculator(spark, WeightingScheme.BMN)
        
        # Calculate weights
        weighted_df = calc.calculate_weights(sample_repeat_sales_df)
        
        # Weights should vary based on holding period
        weights_data = weighted_df.select(
            "weight", "holding_period_days"
        ).collect()
        
        # Longer holding periods should have lower weights
        sorted_data = sorted(weights_data, key=lambda x: x["holding_period_days"])
        weights_sorted = [row["weight"] for row in sorted_data]
        
        # Check that weights generally decrease with holding period
        # (allowing for normalization effects)
        if len(weights_sorted) > 2:
            assert weights_sorted[0] >= weights_sorted[-1] * 0.8
            
    def test_custom_weights(self, spark, sample_repeat_sales_df):
        """Test custom weighting function."""
        calc = WeightCalculator(spark, WeightingScheme.CUSTOM)
        
        # Define custom weight function
        def custom_weight_func(col1, col2):
            return col1 * col2
            
        custom_udf = F.udf(custom_weight_func, "double")
        
        # Add columns for custom weighting
        df_with_cols = sample_repeat_sales_df.withColumn(
            "col1", F.lit(0.5)
        ).withColumn(
            "col2", F.lit(2.0)
        )
        
        # Calculate weights
        weighted_df = calc.calculate_weights(
            df_with_cols,
            weight_func=custom_udf,
            weight_cols=["col1", "col2"]
        )
        
        # All weights should be 1.0 (0.5 * 2.0)
        weights = weighted_df.select("weight").collect()
        assert all(abs(row["weight"] - 1.0) < 0.01 for row in weights)
        
    def test_geographic_weights(self, spark):
        """Test geographic distance-based weights."""
        calc = WeightCalculator(spark, WeightingScheme.EQUAL)
        
        # Create sample data with coordinates
        data = [
            ("RS001", "P001", 40.7128, -74.0060),  # NYC
            ("RS002", "P002", 40.7580, -73.9855),  # Times Square
            ("RS003", "P003", 34.0522, -118.2437), # LA
        ]
        
        df = spark.createDataFrame(
            data,
            schema=["pair_id", "property_id", "latitude", "longitude"]
        )
        
        # Calculate weights relative to NYC coordinates
        weighted_df = calc.geographic_weights(
            df,
            target_lat=40.7128,
            target_lon=-74.0060,
            decay_distance_km=10.0,
            min_weight=0.1
        )
        
        # Check weights
        weights = weighted_df.orderBy("pair_id").collect()
        
        # First property (at target) should have highest weight
        assert weights[0]["weight"] > 0.9
        
        # Times Square (nearby) should have high weight
        assert weights[1]["weight"] > 0.5
        
        # LA (far away) should have minimum weight
        assert abs(weights[2]["weight"] - 0.1) < 0.01
        
    def test_temporal_weights(self, spark):
        """Test temporal distance weights."""
        calc = WeightCalculator(spark, WeightingScheme.EQUAL)
        
        # Create sample data with different sale dates
        reference_date = date(2021, 1, 1)
        data = [
            ("RS001", "P001", date(2021, 1, 1)),   # At reference
            ("RS002", "P002", date(2020, 1, 1)),   # 1 year before
            ("RS003", "P003", date(2022, 1, 1)),   # 1 year after
            ("RS004", "P004", date(2019, 1, 1)),   # 2 years before
        ]
        
        df = spark.createDataFrame(
            data,
            schema=["pair_id", "property_id", "sale2_date"]
        )
        
        # Calculate temporal weights
        weighted_df = calc.temporal_weights(
            df,
            reference_date=reference_date,
            decay_years=1.0,
            forward_weight=0.8
        )
        
        # Check weights
        weights = {row["pair_id"]: row["weight"] 
                  for row in weighted_df.collect()}
        
        # At reference date should have highest weight
        assert weights["RS001"] > weights["RS002"]
        assert weights["RS001"] > weights["RS003"]
        
        # Future sales should be down-weighted
        assert weights["RS003"] < weights["RS002"] * 0.9
        
        # Distant past should have lowest weight
        assert weights["RS004"] < weights["RS002"]
        
    def test_quality_adjusted_weights(self, spark, sample_repeat_sales_df):
        """Test quality-based weights."""
        calc = WeightCalculator(spark, WeightingScheme.EQUAL)
        
        # Add quality indicators
        df_with_quality = sample_repeat_sales_df.withColumn(
            "property_type", F.lit("single_family")
        ).withColumn(
            "validation_flags", F.array()
        )
        
        # Calculate quality weights
        weighted_df = calc.quality_adjusted_weights(
            df_with_quality,
            min_quality=0.5
        )
        
        # All should have high quality (complete data, no flags)
        weights = weighted_df.select("weight").collect()
        assert all(row["weight"] >= 0.8 for row in weights)
        
    def test_combine_weights(self, spark, sample_repeat_sales_df):
        """Test combining multiple weighting schemes."""
        calc = WeightCalculator(spark, WeightingScheme.EQUAL)
        
        # Combine equal and value weights
        weight_schemes = {
            WeightingScheme.EQUAL: 0.5,
            WeightingScheme.VALUE: 0.5
        }
        
        combined_df = calc.combine_weights(
            sample_repeat_sales_df,
            weight_schemes
        )
        
        # Should have combined weights
        assert "weight" in combined_df.columns
        assert "weight_type" in combined_df.columns
        
        # Weight type should be "combined"
        types = combined_df.select("weight_type").distinct().collect()
        assert len(types) == 1
        assert types[0]["weight_type"] == "combined"
        
        # Weights should be between equal (1.0) and value weights
        weights = combined_df.select("weight").collect()
        weight_values = [row["weight"] for row in weights]
        
        # Should have some variation
        assert len(set(weight_values)) > 1
        
    def test_weight_calculator_initialization(self, spark):
        """Test WeightCalculator initialization with different schemes."""
        # Test all schemes
        for scheme in WeightingScheme:
            calc = WeightCalculator(spark, scheme)
            assert calc.scheme == scheme
            assert calc.spark == spark
            
    def test_invalid_custom_weights(self, spark, sample_repeat_sales_df):
        """Test error handling for custom weights without function."""
        calc = WeightCalculator(spark, WeightingScheme.CUSTOM)
        
        # Should raise error without weight function
        with pytest.raises(ValueError, match="No custom weight function"):
            calc.calculate_weights(sample_repeat_sales_df)