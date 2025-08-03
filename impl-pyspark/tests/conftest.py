"""Pytest configuration and fixtures"""

import pytest
from pyspark.sql import SparkSession
import tempfile
import shutil
import os


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing"""
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("hpi-fhfa-tests") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    yield spark
    
    spark.stop()


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test outputs"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data_path():
    """Path to sample test data"""
    return os.path.join(os.path.dirname(__file__), "fixtures")