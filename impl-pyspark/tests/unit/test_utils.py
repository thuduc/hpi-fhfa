"""Unit tests for utility functions"""

import pytest
from pyspark.sql import SparkSession
import math

from hpi_fhfa.utils.geo_utils import haversine_distance
from hpi_fhfa.utils.spark_utils import get_spark_config, create_spark_session


class TestGeoUtils:
    """Test geographic utility functions"""
    
    def test_haversine_distance(self):
        """Test haversine distance calculation"""
        # Test known distance: New York to Philadelphia (~130 km)
        nyc_lat, nyc_lon = 40.7128, -74.0060
        philly_lat, philly_lon = 39.9526, -75.1652
        
        distance = haversine_distance(nyc_lat, nyc_lon, philly_lat, philly_lon)
        
        # Should be approximately 130 km
        assert 125 < distance < 135
        
        # Test same location
        same_distance = haversine_distance(nyc_lat, nyc_lon, nyc_lat, nyc_lon)
        assert same_distance == 0
        
        # Test antipodal points (opposite sides of Earth)
        # Should be approximately half Earth's circumference (~20,000 km)
        antipodal_distance = haversine_distance(0, 0, 0, 180)
        assert 19000 < antipodal_distance < 21000


class TestSparkUtils:
    """Test Spark utility functions"""
    
    def test_get_spark_config_local(self):
        """Test local mode Spark configuration"""
        config = get_spark_config("local")
        
        assert config["spark.sql.adaptive.enabled"] == "true"
        assert config["spark.driver.memory"] == "4g"
        assert config["spark.sql.shuffle.partitions"] == "100"
        assert "spark.serializer" in config
    
    def test_get_spark_config_cluster(self):
        """Test cluster mode Spark configuration"""
        config = get_spark_config("cluster")
        
        assert config["spark.driver.memory"] == "8g"
        assert config["spark.executor.memory"] == "16g"
        assert config["spark.sql.shuffle.partitions"] == "2000"
        assert config["spark.sql.autoBroadcastJoinThreshold"] == "100MB"
    
    def test_create_spark_session(self):
        """Test Spark session creation"""
        # Stop any existing session first
        from pyspark.sql import SparkSession
        existing = SparkSession.getActiveSession()
        if existing:
            existing.stop()
        
        # Create a test session
        spark = create_spark_session(
            app_name="test-session",
            mode="local",
            additional_config={"spark.ui.enabled": "false"}
        )
        
        try:
            # Verify session is created
            assert spark is not None
            assert spark.sparkContext.appName == "test-session"
            
            # Verify configuration is applied
            assert spark.conf.get("spark.ui.enabled") == "false"
            assert spark.conf.get("spark.sql.adaptive.enabled") == "true"
            
        finally:
            spark.stop()
    
    def test_spark_config_with_custom_settings(self):
        """Test Spark configuration with custom settings"""
        custom_config = {
            "spark.custom.setting": "test_value",
            "spark.sql.shuffle.partitions": "50"
        }
        
        spark = create_spark_session(
            app_name="test-custom",
            mode="local",
            additional_config=custom_config
        )
        
        try:
            assert spark.conf.get("spark.custom.setting") == "test_value"
            assert spark.conf.get("spark.sql.shuffle.partitions") == "50"
        finally:
            spark.stop()