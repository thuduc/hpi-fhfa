"""
Unit tests for data ingestion module.
"""

import pytest
from datetime import date, timedelta
from pathlib import Path

from pyspark.sql import functions as F

from rsai.src.data.ingestion import DataIngestion
from rsai.src.data.models import RSAIConfig, WeightingScheme, GeographyLevel, get_transaction_schema


class TestDataIngestion:
    """Test DataIngestion class."""
    
    def test_identify_repeat_sales(self, spark, sample_config, sample_transactions_df):
        """Test repeat sales identification."""
        ingestion = DataIngestion(spark, sample_config)
        
        # Identify repeat sales
        repeat_sales_df = ingestion.identify_repeat_sales(sample_transactions_df)
        
        # Check results
        assert repeat_sales_df.count() == 3  # Should find 3 repeat sales
        
        # Check specific pairs
        pairs = repeat_sales_df.collect()
        property_ids = {row["property_id"] for row in pairs}
        assert "P001" in property_ids
        assert "P002" in property_ids
        assert "P003" in property_ids
        assert "P004" not in property_ids  # Single sale
        assert "P005" not in property_ids  # Non-arms_length
        
        # Check calculations
        p001_pair = repeat_sales_df.filter(F.col("property_id") == "P001").first()
        assert p001_pair["sale1_price"] == 200000.0
        assert p001_pair["sale2_price"] == 250000.0
        assert p001_pair["holding_period_days"] > 0
        assert p001_pair["log_price_ratio"] > 0  # Price increased
        
    def test_merge_geographic_data(
        self, spark, sample_config, sample_repeat_sales_df, sample_properties_df
    ):
        """Test merging geographic data."""
        ingestion = DataIngestion(spark, sample_config)
        
        # Merge geographic data
        merged_df = ingestion.merge_geographic_data(
            sample_repeat_sales_df,
            sample_properties_df
        )
        
        # Check that geographic fields were added
        assert "property_type" in merged_df.columns
        assert "year_built" in merged_df.columns
        assert "square_feet" in merged_df.columns
        assert "latitude" in merged_df.columns
        assert "longitude" in merged_df.columns
        assert "county" in merged_df.columns
        assert "cbsa" in merged_df.columns
        assert "state" in merged_df.columns
        
        # Check data
        first_row = merged_df.first()
        assert first_row["property_type"] is not None
        assert first_row["latitude"] is not None
        assert first_row["longitude"] is not None
        
    def test_filter_outliers(self, spark, sample_config, sample_repeat_sales_df):
        """Test outlier filtering."""
        ingestion = DataIngestion(spark, sample_config)
        
        # Add an outlier
        outlier_data = [
            ("RS004", "P004",
             "T007", date(2020, 1, 1), 100000.0,
             "T008", date(2020, 2, 1), 1000000.0,  # 10x increase
             31, 2.3026, 27.9911, "36061000400", [])
        ]
        schema = sample_repeat_sales_df.schema
        outlier_df = spark.createDataFrame(outlier_data, schema=schema)
        
        # Combine with sample data
        combined_df = sample_repeat_sales_df.union(outlier_df)
        assert combined_df.count() == 4
        
        # Filter outliers
        filtered_df = ingestion.filter_outliers(combined_df)
        
        # Should remove the outlier
        assert filtered_df.count() == 3
        assert filtered_df.filter(F.col("pair_id") == "RS004").count() == 0
        
    def test_validate_data(self, spark, sample_config, sample_repeat_sales_df):
        """Test data validation."""
        ingestion = DataIngestion(spark, sample_config)
        
        # Validate data
        metrics = ingestion.validate_data(sample_repeat_sales_df)
        
        # Check metrics
        assert metrics.total_records == 3
        assert metrics.valid_records == 3
        assert metrics.invalid_records == 0
        assert metrics.validity_score == 1.0
        assert metrics.completeness_score == 1.0
        
    def test_price_filtering(self, spark, sample_config):
        """Test price range filtering."""
        # Create config with restrictive price range
        config = RSAIConfig(
            min_price=200000,
            max_price=300000,
            max_holding_period_years=20,
            min_pairs_threshold=5,
            outlier_std_threshold=3.0
        )
        
        ingestion = DataIngestion(spark, config)
        
        # Create transactions with various prices
        data = [
            ("T001", "P001", date(2020, 1, 1), 150000.0, "arms_length"),  # Below min
            ("T002", "P001", date(2021, 1, 1), 250000.0, "arms_length"),  # In range
            ("T003", "P002", date(2020, 1, 1), 280000.0, "arms_length"),  # In range
            ("T004", "P002", date(2021, 1, 1), 350000.0, "arms_length"),  # Above max
        ]
        
        trans_df = spark.createDataFrame(
            data, 
            schema=["transaction_id", "property_id", "sale_date", "sale_price", "transaction_type"]
        )
        
        # Identify repeat sales (should filter by price)
        repeat_sales = ingestion.identify_repeat_sales(trans_df)
        
        # Should have no repeat sales (one sale in each pair is out of range)
        assert repeat_sales.count() == 0
        
    def test_holding_period_filtering(self, spark, sample_config):
        """Test holding period filtering."""
        # Create config with short max holding period
        config = RSAIConfig(
            min_price=10000,
            max_price=10000000,
            max_holding_period_years=2,  # Only 2 years max
            min_pairs_threshold=5,
            outlier_std_threshold=3.0
        )
        
        ingestion = DataIngestion(spark, config)
        
        # Create transactions with various holding periods
        data = [
            # Short holding period (1 year) - should be included
            ("T001", "P001", date(2020, 1, 1), 200000.0, "arms_length"),
            ("T002", "P001", date(2021, 1, 1), 220000.0, "arms_length"),
            
            # Long holding period (5 years) - should be excluded
            ("T003", "P002", date(2015, 1, 1), 300000.0, "arms_length"),
            ("T004", "P002", date(2020, 1, 1), 350000.0, "arms_length"),
        ]
        
        trans_df = spark.createDataFrame(
            data,
            schema=["transaction_id", "property_id", "sale_date", "sale_price", "transaction_type"]
        )
        
        # Identify repeat sales
        repeat_sales = ingestion.identify_repeat_sales(trans_df)
        
        # Should only have 1 repeat sale
        assert repeat_sales.count() == 1
        assert repeat_sales.first()["property_id"] == "P001"
        
    def test_transaction_type_filtering(self, spark, sample_config):
        """Test filtering by transaction type."""
        ingestion = DataIngestion(spark, sample_config)
        
        # Create transactions with different types
        data = [
            # Arms_length transactions
            ("T001", "P001", date(2020, 1, 1), 200000.0, "arms_length"),
            ("T002", "P001", date(2021, 1, 1), 220000.0, "arms_length"),
            
            # Non-arms_length transactions
            ("T003", "P002", date(2020, 1, 1), 100000.0, "family"),
            ("T004", "P002", date(2021, 1, 1), 110000.0, "family"),
            
            # Mixed (one arms_length, one not)
            ("T005", "P003", date(2020, 1, 1), 300000.0, "arms_length"),
            ("T006", "P003", date(2021, 1, 1), 50000.0, "foreclosure"),
        ]
        
        trans_df = spark.createDataFrame(
            data,
            schema=["transaction_id", "property_id", "sale_date", "sale_price", "transaction_type"]
        )
        
        # Identify repeat sales
        repeat_sales = ingestion.identify_repeat_sales(trans_df)
        
        # Should only include P001 (both arms_length)
        assert repeat_sales.count() == 1
        assert repeat_sales.first()["property_id"] == "P001"
        
    def test_empty_dataframes(self, spark, sample_config):
        """Test handling of empty DataFrames."""
        ingestion = DataIngestion(spark, sample_config)
        
        # Create empty DataFrame with proper schema
        schema = get_transaction_schema()
        empty_df = spark.createDataFrame([], schema=schema)
        
        # Should handle empty input gracefully
        repeat_sales = ingestion.identify_repeat_sales(empty_df)
        assert repeat_sales.count() == 0
        
        # Validate empty data
        metrics = ingestion.validate_data(repeat_sales)
        assert metrics.total_records == 0
        assert metrics.valid_records == 0
        
    def test_duplicate_transactions(self, spark, sample_config):
        """Test handling of duplicate transactions."""
        ingestion = DataIngestion(spark, sample_config)
        
        # Create transactions with duplicates
        data = [
            ("T001", "P001", date(2020, 1, 1), 200000.0, "arms_length"),
            ("T001", "P001", date(2020, 1, 1), 200000.0, "arms_length"),  # Duplicate
            ("T002", "P001", date(2021, 1, 1), 220000.0, "arms_length"),
        ]
        
        trans_df = spark.createDataFrame(
            data,
            schema=["transaction_id", "property_id", "sale_date", "sale_price", "transaction_type"]
        )
        
        # Should handle duplicates properly
        repeat_sales = ingestion.identify_repeat_sales(trans_df)
        
        # Should still identify the repeat sale
        assert repeat_sales.count() >= 1