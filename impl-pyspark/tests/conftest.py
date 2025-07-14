"""
Pytest configuration and shared fixtures for RSAI tests.
"""

import pytest
from datetime import date, datetime
import tempfile
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from rsai.src.data.models import (
    RSAIConfig,
    GeographyLevel,
    WeightingScheme,
    get_transaction_schema,
    get_property_schema,
    get_repeat_sales_schema
)


def _create_spark_session(scope_name="session"):
    """Helper function to create a SparkSession."""
    import os
    import time
    import uuid
    
    # Generate unique app name to avoid conflicts
    app_name = f"RSAI_Tests_{scope_name}_{uuid.uuid4().hex[:8]}"
    
    # Set Java options for testing
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-memory 2g --executor-memory 2g pyspark-shell'
    
    # Stop any existing Spark sessions
    try:
        active_session = SparkSession.getActiveSession()
        if active_session is not None:
            active_session.stop()
    except:
        pass
    
    # Wait a moment for cleanup
    time.sleep(1)
    
    # Configure Spark session
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
        .config("spark.ui.enabled", "false") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.sql.warehouse.dir", f"/tmp/spark-warehouse-{uuid.uuid4().hex[:8]}") \
        .config("spark.sql.catalogImplementation", "in-memory") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryo.unsafe", "false") \
        .config("spark.sql.adaptive.localShuffleReader.enabled", "false") \
        .config("spark.driver.maxResultSize", "1g") \
        .getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("ERROR")
    
    return spark


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing (session scope for unit tests)."""
    spark = _create_spark_session("session")
    yield spark
    
    # Cleanup
    try:
        spark.stop()
    except:
        pass


@pytest.fixture(scope="function")
def spark_fresh():
    """Create a fresh SparkSession for each test (function scope for integration tests)."""
    spark = _create_spark_session("function")
    yield spark
    
    # Cleanup
    try:
        spark.stop()
    except:
        pass


@pytest.fixture
def sample_config():
    """Create a sample RSAIConfig for testing."""
    return RSAIConfig(
        min_price=10000,
        max_price=1000000,
        max_holding_period_years=10,
        min_pairs_threshold=5,
        outlier_std_threshold=3.0,
        frequency="monthly",
        base_period=None,
        weighting_scheme=WeightingScheme.EQUAL,
        geography_levels=[
            GeographyLevel.TRACT,
            GeographyLevel.COUNTY,
            GeographyLevel.STATE
        ],
        clustering_method="kmeans",
        n_clusters=10,
        min_cluster_size=5,
        max_cluster_size=50,
        spark_app_name="RSAI Test",
        spark_master="local[*]",
        spark_executor_memory="1g",
        spark_driver_memory="1g",
        input_files={
            "transactions": "test_transactions.parquet",
            "properties": "test_properties.parquet"
        },
        output_dir="test_output"
    )


@pytest.fixture
def sample_transactions_df(spark):
    """Create sample transaction data."""
    data = [
        # Property 1 - repeat sale
        ("T001", "P001", date(2020, 1, 15), 200000.0, "arms_length"),
        ("T002", "P001", date(2021, 6, 20), 250000.0, "arms_length"),
        
        # Property 2 - repeat sale
        ("T003", "P002", date(2019, 3, 10), 300000.0, "arms_length"),
        ("T004", "P002", date(2021, 9, 15), 350000.0, "arms_length"),
        
        # Property 3 - repeat sale
        ("T005", "P003", date(2020, 5, 1), 150000.0, "arms_length"),
        ("T006", "P003", date(2022, 2, 10), 180000.0, "arms_length"),
        
        # Property 4 - single sale
        ("T007", "P004", date(2021, 1, 1), 400000.0, "arms_length"),
        
        # Property 5 - non-arms_length
        ("T008", "P005", date(2020, 1, 1), 100000.0, "family"),
        ("T009", "P005", date(2021, 1, 1), 120000.0, "family"),
    ]
    
    schema = get_transaction_schema()
    return spark.createDataFrame(data, schema=schema)


@pytest.fixture
def sample_properties_df(spark):
    """Create sample property data."""
    data = [
        ("P001", "single_family", 2000, 1500, 40.7128, -74.0060, 
         "36061000100", "36061", "35620", "36", "123 Main St"),
        ("P002", "condo", 2010, 1200, 40.7580, -73.9855, 
         "36061000200", "36061", "35620", "36", "456 Park Ave"),
        ("P003", "single_family", 1995, 1800, 40.7489, -73.9680, 
         "36061000300", "36061", "35620", "36", "789 Broadway"),
        ("P004", "townhouse", 2015, 1600, 40.7614, -73.9776, 
         "36061000400", "36061", "35620", "36", "321 5th Ave"),
        ("P005", "single_family", 1990, 1400, 40.7282, -73.9942, 
         "36061000500", "36061", "35620", "36", "654 Houston St"),
    ]
    
    schema = get_property_schema()
    return spark.createDataFrame(data, schema=schema)


@pytest.fixture
def sample_repeat_sales_df(spark):
    """Create sample repeat sales data."""
    data = [
        ("RS001", "P001", 
         "T001", date(2020, 1, 15), 200000.0,
         "T002", date(2021, 6, 20), 250000.0,
         521, 0.2231, 0.1563, "36061000100", []),
        ("RS002", "P002",
         "T003", date(2019, 3, 10), 300000.0,
         "T004", date(2021, 9, 15), 350000.0,
         919, 0.1549, 0.0615, "36061000200", []),
        ("RS003", "P003",
         "T005", date(2020, 5, 1), 150000.0,
         "T006", date(2022, 2, 10), 180000.0,
         650, 0.1823, 0.1024, "36061000300", []),
    ]
    
    schema = get_repeat_sales_schema()
    return spark.createDataFrame(data, schema=schema)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_distance_matrix(spark):
    """Create a mock distance matrix for testing."""
    data = [
        ("36061000100", "36061000100", 0.0),
        ("36061000100", "36061000200", 2.5),
        ("36061000100", "36061000300", 3.1),
        ("36061000200", "36061000100", 2.5),
        ("36061000200", "36061000200", 0.0),
        ("36061000200", "36061000300", 1.8),
        ("36061000300", "36061000100", 3.1),
        ("36061000300", "36061000200", 1.8),
        ("36061000300", "36061000300", 0.0),
    ]
    
    return spark.createDataFrame(
        data, 
        schema=["id1", "id2", "distance_km"]
    )


@pytest.fixture
def mock_weights_df(spark):
    """Create mock weights for testing."""
    data = [
        ("RS001", 1.0),
        ("RS002", 1.2),
        ("RS003", 0.8),
    ]
    
    return spark.createDataFrame(
        data,
        schema=["pair_id", "weight"]
    )