"""Data schema definitions for HPI-FHFA PySpark implementation"""

from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, 
    DoubleType, ArrayType, IntegerType
)


class DataSchemas:
    """Central location for all data schemas used in the pipeline"""
    
    TRANSACTION_SCHEMA = StructType([
        StructField("property_id", StringType(), False),
        StructField("transaction_date", DateType(), False),
        StructField("transaction_price", DoubleType(), False),
        StructField("census_tract", StringType(), False),
        StructField("cbsa_code", StringType(), False),
        StructField("distance_to_cbd", DoubleType(), False)
    ])
    
    GEOGRAPHIC_SCHEMA = StructType([
        StructField("census_tract", StringType(), False),
        StructField("cbsa_code", StringType(), False),
        StructField("centroid_lat", DoubleType(), False),
        StructField("centroid_lon", DoubleType(), False),
        StructField("adjacent_tracts", ArrayType(StringType()), False)
    ])
    
    REPEAT_SALES_SCHEMA = StructType([
        StructField("property_id", StringType(), False),
        StructField("sale_date_1", DateType(), False),
        StructField("sale_price_1", DoubleType(), False),
        StructField("sale_date_2", DateType(), False),
        StructField("sale_price_2", DoubleType(), False),
        StructField("census_tract", StringType(), False),
        StructField("cbsa_code", StringType(), False),
        StructField("price_relative", DoubleType(), False),
        StructField("time_diff_years", DoubleType(), False),
        StructField("cagr", DoubleType(), False)
    ])
    
    WEIGHT_SCHEMA = StructType([
        StructField("census_tract", StringType(), False),
        StructField("cbsa_code", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("value_measure", DoubleType(), True),
        StructField("unit_measure", DoubleType(), True),
        StructField("upb_measure", DoubleType(), True),
        StructField("college_share", DoubleType(), True),
        StructField("nonwhite_share", DoubleType(), True)
    ])
    
    HALF_PAIRS_SCHEMA = StructType([
        StructField("census_tract", StringType(), False),
        StructField("cbsa_code", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("total_half_pairs", IntegerType(), False)
    ])
    
    SUPERTRACT_SCHEMA = StructType([
        StructField("supertract_id", StringType(), False),
        StructField("cbsa_code", StringType(), False),
        StructField("tract_list", ArrayType(StringType()), False),
        StructField("min_half_pairs", IntegerType(), False)
    ])
    
    INDEX_OUTPUT_SCHEMA = StructType([
        StructField("cbsa_code", StringType(), False),
        StructField("year", IntegerType(), False),
        StructField("weight_type", StringType(), False),
        StructField("appreciation_rate", DoubleType(), False),
        StructField("index_value", DoubleType(), False),
        StructField("yoy_change", DoubleType(), False)
    ])