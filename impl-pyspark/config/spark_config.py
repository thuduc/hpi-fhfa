"""Spark configuration for HPI-FHFA pipeline"""

SPARK_CONFIG = {
    # Memory Management
    "spark.driver.memory": "8g",
    "spark.executor.memory": "16g",
    "spark.executor.memoryOverhead": "4g",
    
    # Parallelism
    "spark.default.parallelism": "400",
    "spark.sql.shuffle.partitions": "2000",
    
    # Adaptive Query Execution
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled": "true",
    
    # Broadcast Joins
    "spark.sql.autoBroadcastJoinThreshold": "100MB",
    
    # Serialization
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.kryo.registrationRequired": "false",
    
    # Delta Lake (optional)
    "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
    "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog"
}

# Optimal partitioning for different stages
PARTITIONING_STRATEGY = {
    "transactions": {
        "partition_by": ["cbsa_code", "year"],
        "num_partitions": 2000
    },
    "repeat_sales": {
        "partition_by": ["cbsa_code", "sale_year"],
        "num_partitions": 1000
    },
    "indices": {
        "partition_by": ["cbsa_code", "weight_type"],
        "num_partitions": 500
    }
}