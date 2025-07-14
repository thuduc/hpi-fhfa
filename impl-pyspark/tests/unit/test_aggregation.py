"""
Unit tests for index aggregation module.
"""

import pytest
from datetime import date

from pyspark.sql import functions as F

from rsai.src.index.aggregation import IndexAggregator
from rsai.src.data.models import GeographyLevel, WeightingScheme


class TestIndexAggregator:
    """Test IndexAggregator class."""
    
    def test_validate_aggregation(self, spark):
        """Test aggregation path validation."""
        aggregator = IndexAggregator(spark)
        
        # Valid paths
        assert aggregator._validate_aggregation(
            GeographyLevel.TRACT, GeographyLevel.COUNTY
        )
        assert aggregator._validate_aggregation(
            GeographyLevel.COUNTY, GeographyLevel.CBSA
        )
        assert aggregator._validate_aggregation(
            GeographyLevel.CBSA, GeographyLevel.STATE
        )
        assert aggregator._validate_aggregation(
            GeographyLevel.STATE, GeographyLevel.NATIONAL
        )
        
        # Multi-level valid path
        assert aggregator._validate_aggregation(
            GeographyLevel.TRACT, GeographyLevel.STATE
        )
        
        # Invalid paths
        assert not aggregator._validate_aggregation(
            GeographyLevel.COUNTY, GeographyLevel.TRACT
        )
        assert not aggregator._validate_aggregation(
            GeographyLevel.STATE, GeographyLevel.CBSA
        )
        
    def test_add_geography_mapping(self, spark):
        """Test adding geography mapping to index data."""
        aggregator = IndexAggregator(spark)
        
        # Create sample index data
        index_data = [
            ("tract", "36061000100", date(2021, 1, 1), 100.0, 10, 8, 200000.0),
            ("tract", "36061000200", date(2021, 1, 1), 102.0, 12, 10, 210000.0),
            ("tract", "36061000300", date(2021, 1, 1), 98.0, 8, 6, 190000.0),
        ]
        
        index_df = spark.createDataFrame(
            index_data,
            schema=["geography_level", "geography_id", "period", "index_value",
                   "num_pairs", "num_properties", "median_price"]
        )
        
        # Create mapping
        mapping_data = [
            ("36061000100", "36061"),
            ("36061000200", "36061"),
            ("36061000300", "36061"),
        ]
        
        mapping_df = spark.createDataFrame(
            mapping_data,
            schema=["tract_id", "county_id"]
        )
        
        # Add mapping
        mapped_df = aggregator._add_geography_mapping(
            index_df,
            mapping_df,
            GeographyLevel.TRACT,
            GeographyLevel.COUNTY
        )
        
        # Check results
        assert "target_geography_id" in mapped_df.columns
        assert mapped_df.count() == 3
        
        # All should map to same county
        counties = mapped_df.select("target_geography_id").distinct().collect()
        assert len(counties) == 1
        assert counties[0]["target_geography_id"] == "36061"
        
    def test_apply_weights_equal(self, spark):
        """Test applying equal weights."""
        aggregator = IndexAggregator(spark)
        
        # Create sample data
        data = [
            ("tract1", "county1", 100.0, 10, 8, 200000.0),
            ("tract2", "county1", 102.0, 12, 10, 210000.0),
        ]
        
        df = spark.createDataFrame(
            data,
            schema=["geography_id", "target_geography_id", "index_value",
                   "num_pairs", "num_properties", "median_price"]
        )
        
        # Apply equal weights
        weighted_df = aggregator._apply_weights(
            df, WeightingScheme.EQUAL
        )
        
        # All weights should be 1.0
        weights = weighted_df.select("weight").collect()
        assert all(row["weight"] == 1.0 for row in weights)
        
    def test_apply_weights_value(self, spark):
        """Test applying value-based weights."""
        aggregator = IndexAggregator(spark)
        
        # Create sample data
        data = [
            ("tract1", "county1", 100.0, 10, 8, 200000.0),
            ("tract2", "county1", 102.0, 20, 15, 400000.0),
        ]
        
        df = spark.createDataFrame(
            data,
            schema=["geography_id", "target_geography_id", "index_value",
                   "num_pairs", "num_properties", "median_price"]
        )
        
        # Apply value weights
        weighted_df = aggregator._apply_weights(
            df, WeightingScheme.VALUE
        )
        
        # Higher value tract should have higher weight
        weights = weighted_df.orderBy("geography_id").collect()
        assert weights[1]["weight"] > weights[0]["weight"]
        
    def test_perform_aggregation(self, spark):
        """Test performing the actual aggregation."""
        aggregator = IndexAggregator(spark)
        
        # Create sample weighted data
        data = [
            ("tract1", "county1", date(2021, 1, 1), 100.0, 10, 8, 200000.0, 1.0),
            ("tract2", "county1", date(2021, 1, 1), 110.0, 20, 15, 220000.0, 2.0),
            ("tract3", "county1", date(2021, 1, 1), 105.0, 15, 12, 210000.0, 1.5),
        ]
        
        weighted_df = spark.createDataFrame(
            data,
            schema=["geography_id", "target_geography_id", "period", "index_value",
                   "num_pairs", "num_properties", "median_price", "weight"]
        )
        
        # Perform aggregation
        aggregated_df = aggregator._perform_aggregation(
            weighted_df,
            GeographyLevel.COUNTY
        )
        
        # Check results
        assert aggregated_df.count() == 1
        
        result = aggregated_df.first()
        assert result["geography_level"] == "county"
        assert result["geography_id"] == "county1"
        
        # Check weighted average: (100*1 + 110*2 + 105*1.5) / (1 + 2 + 1.5)
        expected_index = (100*1 + 110*2 + 105*1.5) / 4.5
        assert abs(result["index_value"] - expected_index) < 0.01
        
        # Check sums
        assert result["num_pairs"] == 45  # 10 + 20 + 15
        assert result["num_properties"] == 35  # 8 + 15 + 12
        assert result["num_submarkets"] == 3
        
    def test_aggregate_indices(self, spark):
        """Test full aggregation from one level to another."""
        aggregator = IndexAggregator(spark)
        
        # Create tract-level indices
        index_data = []
        for tract in range(3):
            for month in range(3):
                index_data.append((
                    "tract",
                    f"3606100{tract}100",
                    date(2021, month + 1, 1),
                    100.0 + tract + month,
                    10 + tract,
                    8 + tract,
                    200000.0 + tract * 10000
                ))
                
        index_df = spark.createDataFrame(
            index_data,
            schema=["geography_level", "geography_id", "period", "index_value",
                   "num_pairs", "num_properties", "median_price"]
        )
        
        # Create tract to county mapping
        mapping_data = [
            (f"3606100{i}100", "36061") for i in range(3)
        ]
        
        mapping_df = spark.createDataFrame(
            mapping_data,
            schema=["from_id", "to_id"]
        )
        
        # Aggregate to county level
        county_indices = aggregator.aggregate_indices(
            index_df,
            GeographyLevel.TRACT,
            GeographyLevel.COUNTY,
            mapping_df,
            WeightingScheme.EQUAL
        )
        
        # Should have 3 periods
        assert county_indices.count() == 3
        
        # All should be for county 36061
        counties = county_indices.select("geography_id").distinct().collect()
        assert len(counties) == 1
        assert counties[0]["geography_id"] == "36061"
        
        # Check that values are averages of tract values
        for month in range(3):
            period = date(2021, month + 1, 1)
            county_value = county_indices.filter(
                F.col("period") == period
            ).first()["index_value"]
            
            # Average of three tracts
            expected = sum(100.0 + tract + month for tract in range(3)) / 3
            assert abs(county_value - expected) < 0.01
            
    def test_chain_indices(self, spark):
        """Test chaining indices to reference period."""
        aggregator = IndexAggregator(spark)
        
        # Create sample indices
        data = [
            ("county", "36061", date(2021, 1, 1), 95.0, 100, 80, 200000.0),
            ("county", "36061", date(2021, 2, 1), 100.0, 120, 95, 210000.0),
            ("county", "36061", date(2021, 3, 1), 105.0, 130, 100, 220000.0),
        ]
        
        index_df = spark.createDataFrame(
            data,
            schema=["geography_level", "geography_id", "period", "index_value",
                   "num_pairs", "num_properties", "median_price"]
        )
        
        # Chain to February (100.0)
        reference_period = date(2021, 2, 1)
        chained_df = aggregator.chain_indices(
            index_df,
            reference_period=reference_period,
            reference_value=100.0
        )
        
        # Check results
        assert "chained_index" in chained_df.columns
        
        results = chained_df.orderBy("period").collect()
        
        # January: 95/100 * 100 = 95
        assert abs(results[0]["chained_index"] - 95.0) < 0.01
        
        # February: 100/100 * 100 = 100
        assert abs(results[1]["chained_index"] - 100.0) < 0.01
        
        # March: 105/100 * 100 = 105
        assert abs(results[2]["chained_index"] - 105.0) < 0.01
        
    def test_calculate_growth_rates(self, spark):
        """Test growth rate calculations."""
        aggregator = IndexAggregator(spark)
        
        # Create monthly indices
        data = []
        base_value = 100.0
        base_date = date(2020, 1, 1)
        for i in range(13):  # 13 months for YoY calculation
            month = ((i % 12) + 1)
            year = 2020 + (i // 12)
            current_date = base_date.replace(month=month, year=year)
            data.append((
                "county", "36061",
                current_date,
                base_value * (1.01 ** i),  # 1% monthly growth
                100, 80, 200000.0
            ))
            
        index_df = spark.createDataFrame(
            data,
            schema=["geography_level", "geography_id", "period", "index_value",
                   "num_pairs", "num_properties", "median_price"]
        )
        
        # Calculate growth rates
        growth_df = aggregator.calculate_growth_rates(
            index_df,
            periods=[1, 12]  # MoM and YoY
        )
        
        # Check columns
        assert "growth_1m" in growth_df.columns
        assert "growth_12m" in growth_df.columns
        
        # Check MoM growth (should be ~1%)
        mom_growth = growth_df.filter(
            F.col("growth_1m").isNotNull()
        ).select("growth_1m").collect()
        
        for row in mom_growth:
            assert abs(row["growth_1m"] - 1.0) < 0.1
            
        # Check YoY growth (should be ~12.68% for 1% monthly)
        yoy_growth = growth_df.filter(
            F.col("growth_12m").isNotNull()
        ).select("growth_12m").first()
        
        expected_yoy = ((1.01 ** 12) - 1) * 100
        assert abs(yoy_growth["growth_12m"] - expected_yoy) < 0.5
        
    def test_detect_level(self, spark):
        """Test geographic level detection."""
        aggregator = IndexAggregator(spark)
        
        # Test different ID formats
        assert aggregator._detect_level("36061000100") == GeographyLevel.TRACT
        assert aggregator._detect_level("36061") == GeographyLevel.COUNTY
        assert aggregator._detect_level("36") == GeographyLevel.STATE
        
        # Default for unknown
        assert aggregator._detect_level("unknown") == GeographyLevel.TRACT
        
    def test_create_hierarchical_indices(self, spark):
        """Test creating indices for multiple geographic levels."""
        aggregator = IndexAggregator(spark)
        
        # Create base tract indices
        base_indices = {}
        
        for tract_num in range(2):
            tract_id = f"3606100{tract_num}100"
            data = [
                ("tract", tract_id, date(2021, 1, 1), 100.0, 10, 8, 200000.0),
                ("tract", tract_id, date(2021, 2, 1), 102.0, 12, 10, 210000.0),
            ]
            
            df = spark.createDataFrame(
                data,
                schema=["geography_level", "geography_id", "period", "index_value",
                       "num_pairs", "num_properties", "median_price"]
            )
            base_indices[tract_id] = df
            
        # Create mappings
        tract_to_county = spark.createDataFrame(
            [(f"3606100{i}100", "36061") for i in range(2)],
            schema=["from_id", "to_id"]
        )
        
        county_to_state = spark.createDataFrame(
            [("36061", "36")],
            schema=["from_id", "to_id"]
        )
        
        geography_mappings = {
            "tract_to_county": tract_to_county,
            "county_to_state": county_to_state
        }
        
        # Create hierarchical indices
        target_levels = [
            GeographyLevel.TRACT,
            GeographyLevel.COUNTY,
            GeographyLevel.STATE
        ]
        
        results = aggregator.create_hierarchical_indices(
            base_indices,
            geography_mappings,
            target_levels,
            WeightingScheme.EQUAL
        )
        
        # Should have indices for all levels
        assert GeographyLevel.TRACT in results
        assert GeographyLevel.COUNTY in results
        assert GeographyLevel.STATE in results
        
        # Check counts
        assert results[GeographyLevel.TRACT].count() == 4  # 2 tracts * 2 periods
        assert results[GeographyLevel.COUNTY].count() == 2  # 1 county * 2 periods
        assert results[GeographyLevel.STATE].count() == 2  # 1 state * 2 periods