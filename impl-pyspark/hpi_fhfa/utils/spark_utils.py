"""Spark utility functions"""

from pyspark.sql import SparkSession
from typing import Dict, Optional


def get_spark_config(mode: str = "local") -> Dict[str, str]:
    """Get Spark configuration based on execution mode"""
    
    base_config = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryo.registrationRequired": "false",
    }
    
    if mode == "local":
        config = {
            **base_config,
            "spark.driver.memory": "4g",
            "spark.executor.memory": "4g",
            "spark.sql.shuffle.partitions": "100",
        }
    elif mode == "cluster":
        config = {
            **base_config,
            "spark.driver.memory": "8g",
            "spark.executor.memory": "16g",
            "spark.executor.memoryOverhead": "4g",
            "spark.default.parallelism": "400",
            "spark.sql.shuffle.partitions": "2000",
            "spark.sql.autoBroadcastJoinThreshold": "100MB",
        }
    else:
        config = base_config
    
    # Add Delta Lake support if available
    # Commented out for now as it causes issues in test environment
    # try:
    #     config.update({
    #         "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
    #         "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog"
    #     })
    # except:
    #     pass
    
    return config


def create_spark_session(
    app_name: str = "HPI-FHFA-Pipeline",
    mode: str = "local",
    additional_config: Optional[Dict[str, str]] = None
) -> SparkSession:
    """Create and configure a Spark session"""
    
    builder = SparkSession.builder.appName(app_name)
    
    # Apply base configuration
    config = get_spark_config(mode)
    for key, value in config.items():
        builder = builder.config(key, value)
    
    # Apply any additional configuration
    if additional_config:
        for key, value in additional_config.items():
            builder = builder.config(key, value)
    
    # Set master for local mode
    if mode == "local":
        builder = builder.master("local[*]")
    
    return builder.getOrCreate()