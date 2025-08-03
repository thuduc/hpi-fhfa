"""Unit tests for DataProcessor class"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
from datetime import datetime, date
import numpy as np

from hpi_fhfa.etl.data_processor import DataProcessor
from hpi_fhfa.schemas.data_schemas import DataSchemas


class TestDataProcessor:
    """Test suite for DataProcessor functionality"""
    
    @pytest.fixture
    def data_processor(self, spark):
        """Create DataProcessor instance"""
        return DataProcessor(spark)
    
    @pytest.fixture
    def sample_transactions(self, spark):
        """Create sample transaction data"""
        data = [
            ("P1", date(2010, 1, 1), 100000.0, "12345", "19100", 5.0),
            ("P1", date(2015, 6, 1), 150000.0, "12345", "19100", 5.0),
            ("P2", date(2012, 3, 1), 200000.0, "12346", "19100", 3.0),
            ("P2", date(2018, 9, 1), 180000.0, "12346", "19100", 3.0),
            ("P3", date(2010, 5, 1), 300000.0, "12347", "19100", 7.0),
            ("P3", date(2011, 5, 1), 320000.0, "12347", "19100", 7.0),
            ("P3", date(2020, 5, 1), 450000.0, "12347", "19100", 7.0),
        ]
        schema = DataSchemas.TRANSACTION_SCHEMA
        return spark.createDataFrame(data, schema)
    
    def test_create_repeat_sales_pairs(self, data_processor, sample_transactions):
        """Test repeat-sales pair creation"""
        pairs = data_processor.create_repeat_sales_pairs(sample_transactions)
        
        # Test pair creation
        assert pairs.count() == 4  # P1: 1 pair, P2: 1 pair, P3: 2 pairs
        
        # Test schema
        assert set(pairs.columns) == {
            "property_id", "sale_date_1", "sale_price_1", 
            "sale_date_2", "sale_price_2", "census_tract", 
            "cbsa_code", "distance_to_cbd", "price_relative", 
            "time_diff_years", "cagr"
        }
        
        # Test price relative calculation for P1
        p1_pair = pairs.filter(pairs.property_id == "P1").first()
        expected_relative = np.log(150000) - np.log(100000)
        assert abs(p1_pair.price_relative - expected_relative) < 0.0001
        
        # Test time difference calculation
        expected_years = (date(2015, 6, 1) - date(2010, 1, 1)).days / 365.25
        assert abs(p1_pair.time_diff_years - expected_years) < 0.01
        
        # Test CAGR calculation
        expected_cagr = (150000/100000)**(1/expected_years) - 1
        assert abs(p1_pair.cagr - expected_cagr) < 0.001
    
    def test_apply_filters(self, data_processor, spark):
        """Test data quality filters"""
        # Create test data with various edge cases
        data = [
            # Normal case
            ("P1", date(2010, 1, 1), 100000.0, date(2015, 1, 1), 
             150000.0, "12345", "19100", 5.0, float(np.log(1.5)), 5.0, 0.084),
            
            # Same year filter (should be removed)
            ("P2", date(2010, 1, 1), 100000.0, date(2010, 11, 1), 
             110000.0, "12346", "19100", 3.0, float(np.log(1.1)), 0.83, 0.12),
            
            # Extreme CAGR (should be removed)
            ("P3", date(2010, 1, 1), 100000.0, date(2011, 1, 1), 
             200000.0, "12347", "19100", 5.0, float(np.log(2)), 1.0, 1.0),
            
            # Extreme cumulative appreciation (should be removed)
            ("P4", date(2010, 1, 1), 100000.0, date(2020, 1, 1), 
             1200000.0, "12348", "19100", 5.0, float(np.log(12)), 10.0, 0.28),
        ]
        
        schema = StructType([
            StructField("property_id", StringType(), False),
            StructField("sale_date_1", DateType(), False),
            StructField("sale_price_1", DoubleType(), False),
            StructField("sale_date_2", DateType(), False),
            StructField("sale_price_2", DoubleType(), False),
            StructField("census_tract", StringType(), False),
            StructField("cbsa_code", StringType(), False),
            StructField("distance_to_cbd", DoubleType(), False),
            StructField("price_relative", DoubleType(), False),
            StructField("time_diff_years", DoubleType(), False),
            StructField("cagr", DoubleType(), False)
        ])
        
        repeat_sales = spark.createDataFrame(data, schema)
        filtered = data_processor.apply_filters(repeat_sales)
        
        # Only P1 should pass all filters
        # P2: Same year (2010) - filtered out
        # P3: CAGR = 1.0 (100%) - filtered out
        # P4: Cumulative = 12x - filtered out
        assert filtered.count() == 1
        assert filtered.first().property_id == "P1"
    
    def test_calculate_half_pairs(self, data_processor, spark):
        """Test half-pairs calculation"""
        # Create repeat-sales data
        repeat_sales_data = [
            ("P1", date(2010, 1, 1), date(2015, 1, 1), 
             "12345", "19100"),
            ("P2", date(2010, 1, 1), date(2015, 1, 1), 
             "12345", "19100"),
            ("P3", date(2011, 1, 1), date(2016, 1, 1), 
             "12346", "19100"),
            ("P4", date(2015, 1, 1), date(2020, 1, 1), 
             "12345", "19100"),
        ]
        
        schema = ["property_id", "sale_date_1", "sale_date_2", 
                 "census_tract", "cbsa_code"]
        repeat_sales = spark.createDataFrame(repeat_sales_data, schema)
        
        half_pairs = data_processor.calculate_half_pairs(repeat_sales)
        
        # Check aggregation for tract 12345 in 2010
        tract_12345_2010 = half_pairs.filter(
            (half_pairs.census_tract == "12345") & 
            (half_pairs.year == 2010)
        ).first()
        
        assert tract_12345_2010.total_half_pairs == 2  # P1 and P2
        
        # Check aggregation for tract 12345 in 2015
        tract_12345_2015 = half_pairs.filter(
            (half_pairs.census_tract == "12345") & 
            (half_pairs.year == 2015)
        ).first()
        
        assert tract_12345_2015.total_half_pairs == 3  # P1, P2, and P4
        
        # Check total unique tract-year combinations
        assert half_pairs.count() == 5  # 12345: 2010,2015,2020 + 12346: 2011,2016
    
    def test_edge_cases(self, data_processor, spark):
        """Test edge cases and error handling"""
        # Test with empty DataFrame
        empty_df = spark.createDataFrame([], DataSchemas.TRANSACTION_SCHEMA)
        pairs = data_processor.create_repeat_sales_pairs(empty_df)
        assert pairs.count() == 0
        
        # Test with single transaction per property
        single_data = [
            ("P1", date(2010, 1, 1), 100000.0, "12345", "19100", 5.0),
            ("P2", date(2010, 1, 1), 200000.0, "12346", "19100", 3.0),
        ]
        single_df = spark.createDataFrame(single_data, DataSchemas.TRANSACTION_SCHEMA)
        pairs = data_processor.create_repeat_sales_pairs(single_df)
        assert pairs.count() == 0  # No repeat sales