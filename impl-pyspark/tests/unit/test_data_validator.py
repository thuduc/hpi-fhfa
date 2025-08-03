"""Unit tests for DataValidator class"""

import pytest
from pyspark.sql import SparkSession
from datetime import date
import numpy as np

from hpi_fhfa.etl.data_validator import DataValidator
from hpi_fhfa.schemas.data_schemas import DataSchemas


class TestDataValidator:
    """Test suite for DataValidator functionality"""
    
    @pytest.fixture
    def data_validator(self, spark):
        """Create DataValidator instance"""
        return DataValidator(spark)
    
    @pytest.fixture
    def valid_transactions(self, spark):
        """Create valid transaction data"""
        data = [
            ("P1", date(2010, 1, 1), 100000.0, "12345", "19100", 5.0),
            ("P2", date(2015, 6, 1), 250000.0, "12346", "19100", 3.0),
            ("P3", date(2018, 3, 1), 350000.0, "12347", "19100", 7.0),
        ]
        return spark.createDataFrame(data, DataSchemas.TRANSACTION_SCHEMA)
    
    @pytest.fixture
    def problematic_transactions(self, spark):
        """Create transaction data with issues"""
        from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType
        
        # Create a schema that allows nulls for testing
        nullable_schema = StructType([
            StructField("property_id", StringType(), True),  # Allow nulls
            StructField("transaction_date", DateType(), True),  # Allow nulls
            StructField("transaction_price", DoubleType(), True),  # Allow nulls
            StructField("census_tract", StringType(), True),
            StructField("cbsa_code", StringType(), True),
            StructField("distance_to_cbd", DoubleType(), True)
        ])
        
        data = [
            ("P1", date(2010, 1, 1), 100000.0, "12345", "19100", 5.0),
            ("P2", date(2015, 6, 1), -50000.0, "12346", "19100", 3.0),  # Negative price
            ("P3", None, 350000.0, "12347", "19100", 7.0),  # Null date
            (None, date(2018, 3, 1), 400000.0, "12348", "19100", 2.0),  # Null property_id
        ]
        return spark.createDataFrame(data, nullable_schema)
    
    def test_validate_transactions_valid(self, data_validator, valid_transactions):
        """Test validation of valid transaction data"""
        results = data_validator.validate_transactions(valid_transactions)
        
        assert results["has_required_columns"] == True
        assert results["no_nulls_property_id"] == True
        assert results["no_nulls_transaction_date"] == True
        assert results["no_nulls_transaction_price"] == True
        assert results["valid_price_range"] == True
    
    def test_validate_transactions_problematic(self, data_validator, problematic_transactions):
        """Test validation of problematic transaction data"""
        results = data_validator.validate_transactions(problematic_transactions)
        
        assert results["has_required_columns"] == True
        assert results["no_nulls_property_id"] == False  # Has null
        assert results["no_nulls_transaction_date"] == False  # Has null
        assert results["valid_price_range"] == False  # Has negative price
    
    def test_validate_repeat_sales(self, data_validator, spark):
        """Test validation of repeat-sales data"""
        # Create repeat-sales data with various scenarios
        data = [
            # Valid pair
            ("P1", date(2010, 1, 1), 100000.0, date(2015, 1, 1), 
             150000.0, "12345", "19100", float(np.log(1.5)), 5.0, 0.084),
            
            # Invalid date order
            ("P2", date(2015, 1, 1), 200000.0, date(2010, 1, 1), 
             250000.0, "12346", "19100", float(np.log(1.25)), -5.0, -0.045),
            
            # Extreme CAGR
            ("P3", date(2010, 1, 1), 100000.0, date(2011, 1, 1), 
             300000.0, "12347", "19100", float(np.log(3)), 1.0, 2.0),
        ]
        
        schema = ["property_id", "sale_date_1", "sale_price_1", 
                 "sale_date_2", "sale_price_2", "census_tract", 
                 "cbsa_code", "price_relative", "time_diff_years", "cagr"]
        
        repeat_sales = spark.createDataFrame(data, schema)
        results = data_validator.validate_repeat_sales(repeat_sales)
        
        assert results["valid_date_order"] == False  # P2 has invalid dates
        assert results["positive_prices"] == True
        assert results["reasonable_time_gaps"] == False  # P2 has negative time gap
    
    def test_validate_half_pairs(self, data_validator, spark):
        """Test validation of half-pairs data"""
        # Create half-pairs data with varying counts
        data = [
            ("12345", "19100", 2010, 100),  # Above threshold
            ("12346", "19100", 2010, 30),   # Below threshold
            ("12347", "19100", 2010, 5),    # Well below threshold
            ("12348", "19100", 2010, 45),   # Just above threshold
        ]
        
        schema = ["census_tract", "cbsa_code", "year", "total_half_pairs"]
        half_pairs = spark.createDataFrame(data, schema)
        
        results = data_validator.validate_half_pairs(half_pairs, min_threshold=40)
        
        assert results["tracts_below_threshold"] == 2  # 12346 and 12347
        assert results["total_tracts"] == 4
        assert results["percent_below_threshold"] == 50.0
        
        # Check distribution
        dist = results["distribution"]
        assert dist["min"] == 5
        assert dist["max"] == 100
        assert dist["median"] >= 30 and dist["median"] <= 45