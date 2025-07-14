"""
Unit tests for data models and schemas.
"""

import pytest
from datetime import date, datetime
from pydantic import ValidationError

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from rsai.src.data.models import (
    RSAIConfig,
    IndexValue,
    BMNRegressionResult,
    QualityMetrics,
    GeographyLevel,
    WeightingScheme,
    ClusteringMethod,
    get_transaction_schema,
    get_property_schema,
    get_repeat_sales_schema
)


class TestDataModels:
    """Test data model classes."""
    
    def test_rsai_config_validation(self):
        """Test RSAIConfig validation."""
        # Valid config
        config = RSAIConfig(
            min_price=10000,
            max_price=1000000,
            max_holding_period_years=20,
            min_pairs_threshold=30,
            outlier_std_threshold=3.0,
            frequency="monthly",
            weighting_scheme=WeightingScheme.EQUAL,
            geography_levels=[GeographyLevel.TRACT, GeographyLevel.COUNTY]
        )
        assert config.min_price == 10000
        assert config.frequency == "monthly"
        
        # Invalid price range
        with pytest.raises(ValidationError):
            RSAIConfig(
                min_price=1000000,
                max_price=10000,  # Less than min_price
                max_holding_period_years=20,
                min_pairs_threshold=30,
                outlier_std_threshold=3.0
            )
            
        # Invalid frequency
        with pytest.raises(ValidationError):
            RSAIConfig(
                min_price=10000,
                max_price=1000000,
                max_holding_period_years=20,
                min_pairs_threshold=30,
                outlier_std_threshold=3.0,
                frequency="weekly"  # Invalid
            )
            
    def test_index_value_model(self):
        """Test IndexValue model."""
        iv = IndexValue(
            geography_level=GeographyLevel.TRACT,
            geography_id="36061000100",
            period=date(2021, 1, 1),
            index_value=105.5,
            num_pairs=100,
            num_properties=80,
            median_price=250000.0,
            standard_error=2.5,
            confidence_lower=100.6,
            confidence_upper=110.4
        )
        
        assert iv.geography_level == GeographyLevel.TRACT
        assert iv.index_value == 105.5
        assert iv.num_pairs == 100
        
    def test_bmn_regression_result(self):
        """Test BMNRegressionResult model."""
        result = BMNRegressionResult(
            geography_level=GeographyLevel.COUNTY,
            geography_id="36061",
            start_period=date(2020, 1, 1),
            end_period=date(2022, 12, 31),
            num_periods=36,
            num_observations=500,
            r_squared=0.85,
            adj_r_squared=0.84,
            coefficients={"2020-01": 0.0, "2020-02": 0.05},
            standard_errors={"2020-01": 0.01, "2020-02": 0.012},
            t_statistics={"2020-01": 0.0, "2020-02": 4.17},
            p_values={"2020-01": 1.0, "2020-02": 0.0001},
            index_values=[]
        )
        
        assert result.r_squared == 0.85
        assert result.num_observations == 500
        assert len(result.coefficients) == 2
        
    def test_quality_metrics(self):
        """Test QualityMetrics model."""
        metrics = QualityMetrics(
            total_records=1000,
            valid_records=950,
            invalid_records=50,
            missing_fields={"price": 20, "date": 10},
            validation_errors={"price_range": 15, "date_format": 5},
            completeness_score=0.95,
            validity_score=0.95,
            overall_score=0.95
        )
        
        assert metrics.total_records == 1000
        assert metrics.completeness_score == 0.95
        assert metrics.missing_fields["price"] == 20


class TestSchemas:
    """Test PySpark schemas."""
    
    def test_transaction_schema(self, spark):
        """Test transaction schema."""
        schema = get_transaction_schema()
        
        # Check it's a valid schema
        assert isinstance(schema, StructType)
        
        # Check required fields
        field_names = [field.name for field in schema.fields]
        assert "transaction_id" in field_names
        assert "property_id" in field_names
        assert "sale_date" in field_names
        assert "sale_price" in field_names
        assert "transaction_type" in field_names
        
        # Create DataFrame with schema
        data = [
            ("T001", "P001", date(2021, 1, 1), 200000.0, "arms_length")
        ]
        df = spark.createDataFrame(data, schema=schema)
        assert df.count() == 1
        
    def test_property_schema(self, spark):
        """Test property schema."""
        schema = get_property_schema()
        
        # Check required fields
        field_names = [field.name for field in schema.fields]
        assert "property_id" in field_names
        assert "property_type" in field_names
        assert "year_built" in field_names
        assert "square_feet" in field_names
        assert "latitude" in field_names
        assert "longitude" in field_names
        assert "tract" in field_names
        
        # Create DataFrame with schema
        data = [
            ("P001", "single_family", 2000, 1500, 40.7128, -74.0060,
             "36061000100", "36061", "35620", "36", "123 Main St")
        ]
        df = spark.createDataFrame(data, schema=schema)
        assert df.count() == 1
        
    def test_repeat_sales_schema(self, spark):
        """Test repeat sales schema."""
        schema = get_repeat_sales_schema()
        
        # Check required fields
        field_names = [field.name for field in schema.fields]
        assert "pair_id" in field_names
        assert "property_id" in field_names
        assert "sale1_transaction_id" in field_names
        assert "sale1_date" in field_names
        assert "sale1_price" in field_names
        assert "sale2_transaction_id" in field_names
        assert "sale2_date" in field_names
        assert "sale2_price" in field_names
        assert "holding_period_days" in field_names
        assert "log_price_ratio" in field_names
        assert "annualized_return" in field_names
        
        # Create DataFrame with schema
        data = [
            ("RS001", "P001",
             "T001", date(2020, 1, 1), 200000.0,
             "T002", date(2021, 1, 1), 220000.0,
             365, 0.0953, 0.0953, "36061000100", [])
        ]
        df = spark.createDataFrame(data, schema=schema)
        assert df.count() == 1
        
    def test_schema_field_types(self):
        """Test specific field types in schemas."""
        # Transaction schema
        trans_schema = get_transaction_schema()
        trans_fields = {field.name: field.dataType for field in trans_schema.fields}
        assert isinstance(trans_fields["sale_price"], DoubleType)
        assert isinstance(trans_fields["transaction_id"], StringType)
        
        # Property schema
        prop_schema = get_property_schema()
        prop_fields = {field.name: field.dataType for field in prop_schema.fields}
        assert isinstance(prop_fields["latitude"], DoubleType)
        assert isinstance(prop_fields["longitude"], DoubleType)
        
        # Repeat sales schema
        rs_schema = get_repeat_sales_schema()
        rs_fields = {field.name: field.dataType for field in rs_schema.fields}
        assert isinstance(rs_fields["log_price_ratio"], DoubleType)
        assert isinstance(rs_fields["annualized_return"], DoubleType)


class TestEnums:
    """Test enum classes."""
    
    def test_geography_level_enum(self):
        """Test GeographyLevel enum."""
        assert GeographyLevel.TRACT.value == "tract"
        assert GeographyLevel.COUNTY.value == "county"
        assert GeographyLevel.CBSA.value == "cbsa"
        assert GeographyLevel.STATE.value == "state"
        assert GeographyLevel.NATIONAL.value == "national"
        
        # Test string conversion
        assert str(GeographyLevel.TRACT) == "GeographyLevel.TRACT"
        
    def test_weighting_scheme_enum(self):
        """Test WeightingScheme enum."""
        assert WeightingScheme.EQUAL.value == "equal"
        assert WeightingScheme.VALUE.value == "value"
        assert WeightingScheme.CASE_SHILLER.value == "case_shiller"
        assert WeightingScheme.BMN.value == "bmn"
        assert WeightingScheme.CUSTOM.value == "custom"
        
    def test_clustering_method_enum(self):
        """Test ClusteringMethod enum."""
        assert ClusteringMethod.KMEANS.value == "kmeans"
        assert ClusteringMethod.BISECTING_KMEANS.value == "bisecting_kmeans"
        assert ClusteringMethod.GAUSSIAN_MIXTURE.value == "gaussian_mixture"
        assert ClusteringMethod.HIERARCHICAL.value == "hierarchical"