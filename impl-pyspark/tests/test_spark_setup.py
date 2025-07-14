"""
Test to verify Spark setup is working correctly.
"""

import pytest


def test_spark_session_creation(spark):
    """Test that Spark session can be created."""
    assert spark is not None
    assert spark.version is not None
    print(f"\nSpark version: {spark.version}")


def test_spark_dataframe_operations(spark):
    """Test basic DataFrame operations."""
    # Create a simple DataFrame
    data = [(1, "Alice"), (2, "Bob"), (3, "Charlie")]
    df = spark.createDataFrame(data, ["id", "name"])
    
    # Test count
    assert df.count() == 3
    
    # Test filter
    filtered = df.filter(df.id > 1)
    assert filtered.count() == 2
    
    # Test collect
    rows = df.collect()
    assert len(rows) == 3
    assert rows[0]["name"] == "Alice"


def test_spark_sql_operations(spark):
    """Test Spark SQL operations."""
    # Create a DataFrame
    data = [(1, 100), (2, 200), (3, 300)]
    df = spark.createDataFrame(data, ["id", "value"])
    
    # Register as temp view
    df.createOrReplaceTempView("test_table")
    
    # Run SQL query
    result = spark.sql("SELECT SUM(value) as total FROM test_table")
    total = result.collect()[0]["total"]
    
    assert total == 600