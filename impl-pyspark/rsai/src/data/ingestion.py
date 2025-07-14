"""
Data ingestion and preparation using PySpark DataFrames.

This module handles loading property transaction data and preparing
it for repeat sales analysis using PySpark's distributed computing.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
from datetime import date, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

from rsai.src.data.models import (
    RSAIConfig,
    TransactionType,
    QualityMetrics,
    get_transaction_schema,
    get_property_schema,
    get_repeat_sales_schema
)

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data loading and preparation for RSAI model using PySpark."""
    
    def __init__(self, spark: SparkSession, config: RSAIConfig):
        """
        Initialize data ingestion.
        
        Args:
            spark: SparkSession instance
            config: RSAI configuration
        """
        self.spark = spark
        self.config = config
        self.transaction_schema = get_transaction_schema()
        self.property_schema = get_property_schema()
        
    def load_transactions(self, file_path: str) -> DataFrame:
        """
        Load transaction data from file.
        
        Args:
            file_path: Path to transaction data file
            
        Returns:
            PySpark DataFrame with transactions
        """
        logger.info(f"Loading transactions from {file_path}")
        
        # Check if file exists
        import os
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            # Return empty DataFrame with correct schema
            return self.spark.createDataFrame([], schema=self.transaction_schema)
        
        # Determine file format and load
        if file_path.endswith('.parquet'):
            df = self.spark.read.parquet(file_path)
        elif file_path.endswith('.csv'):
            df = self.spark.read.csv(
                file_path, 
                header=True, 
                schema=self.transaction_schema
            )
        else:
            # Try to infer format
            df = self.spark.read.load(file_path)
            
        # Apply date filters if specified
        if self.config.start_date is not None:
            df = df.filter(F.col("sale_date") >= self.config.start_date)
        if self.config.end_date is not None:
            df = df.filter(F.col("sale_date") <= self.config.end_date)
        
        # Apply price filters
        df = df.filter(
            (F.col("sale_price") >= self.config.min_price) &
            (F.col("sale_price") <= self.config.max_price)
        )
        
        # Filter for arms-length transactions if specified
        if "transaction_type" in df.columns:
            df = df.filter(
                F.col("transaction_type") == TransactionType.ARMS_LENGTH.value
            )
            
        logger.info(f"Loaded {df.count()} transactions")
        return df
        
    def load_properties(self, file_path: str) -> DataFrame:
        """
        Load property data from file.
        
        Args:
            file_path: Path to property data file
            
        Returns:
            PySpark DataFrame with properties
        """
        logger.info(f"Loading properties from {file_path}")
        
        # Check if file exists
        import os
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            # Return empty DataFrame with correct schema
            return self.spark.createDataFrame([], schema=self.property_schema)
        
        # Determine file format and load
        if file_path.endswith('.parquet'):
            df = self.spark.read.parquet(file_path)
        elif file_path.endswith('.csv'):
            df = self.spark.read.csv(
                file_path, 
                header=True, 
                schema=self.property_schema
            )
        else:
            df = self.spark.read.load(file_path)
            
        logger.info(f"Loaded {df.count()} properties")
        return df
        
    def identify_repeat_sales(
        self, 
        transactions_df: DataFrame,
        min_holding_days: int = 180
    ) -> DataFrame:
        """
        Identify repeat sales pairs from transaction data.
        
        Args:
            transactions_df: DataFrame with property transactions
            min_holding_days: Minimum days between sales
            
        Returns:
            DataFrame with repeat sales pairs
        """
        logger.info("Identifying repeat sales")
        
        # Window specification for property-based operations
        property_window = Window.partitionBy("property_id").orderBy("sale_date")
        
        # Add row numbers and lead columns
        df_with_next = transactions_df.withColumn(
            "row_num", F.row_number().over(property_window)
        ).withColumn(
            "next_transaction_id", F.lead("transaction_id").over(property_window)
        ).withColumn(
            "next_price", F.lead("sale_price").over(property_window)
        ).withColumn(
            "next_date", F.lead("sale_date").over(property_window)
        ).withColumn(
            "next_transaction_type", F.lead("transaction_type").over(property_window)
        )
        
        # Filter for properties with next sale and valid transaction types
        repeat_sales = df_with_next.filter(
            F.col("next_transaction_id").isNotNull() &
            (F.col("transaction_type") == TransactionType.ARMS_LENGTH.value) &
            (F.col("next_transaction_type") == TransactionType.ARMS_LENGTH.value) &
            (F.col("sale_price") >= self.config.min_price) &
            (F.col("sale_price") <= self.config.max_price) &
            (F.col("next_price") >= self.config.min_price) &
            (F.col("next_price") <= self.config.max_price)
        )
        
        # Calculate holding period and price metrics
        repeat_sales = repeat_sales.withColumn(
            "holding_period_days",
            F.datediff(F.col("next_date"), F.col("sale_date"))
        ).withColumn(
            "price_ratio",
            F.col("next_price") / F.col("sale_price")
        ).withColumn(
            "log_price_ratio",
            F.log(F.col("price_ratio"))
        )
        
        # Filter by minimum holding period and max years
        max_days = self.config.max_holding_period_years * 365
        repeat_sales = repeat_sales.filter(
            (F.col("holding_period_days") >= min_holding_days) &
            (F.col("holding_period_days") <= max_days)
        )
        
        # Calculate annualized return
        repeat_sales = repeat_sales.withColumn(
            "annualized_return",
            F.pow(F.col("price_ratio"), 365.0 / F.col("holding_period_days")) - 1
        )
        
        # Create pair ID
        repeat_sales = repeat_sales.withColumn(
            "pair_id",
            F.concat_ws("_", 
                F.col("property_id"), 
                F.col("transaction_id"), 
                F.col("next_transaction_id")
            )
        )
        
        # Select and rename columns to match schema
        repeat_sales = repeat_sales.select(
            F.col("pair_id"),
            F.col("property_id"),
            F.col("transaction_id").alias("sale1_transaction_id"),
            F.col("sale_price").alias("sale1_price"),
            F.col("sale_date").alias("sale1_date"),
            F.col("next_transaction_id").alias("sale2_transaction_id"),
            F.col("next_price").alias("sale2_price"),
            F.col("next_date").alias("sale2_date"),
            F.col("price_ratio"),
            F.col("log_price_ratio"),
            F.col("holding_period_days"),
            F.col("annualized_return"),
            F.lit(True).alias("is_valid"),
            F.array().alias("validation_flags")
        )
        
        logger.info(f"Identified {repeat_sales.count()} repeat sale pairs")
        return repeat_sales
        
    def merge_geographic_data(
        self,
        repeat_sales_df: DataFrame,
        properties_df: DataFrame
    ) -> DataFrame:
        """
        Merge geographic data from properties to repeat sales.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales pairs
            properties_df: DataFrame with property information
            
        Returns:
            DataFrame with geographic fields added
        """
        logger.info("Merging geographic data")
        
        # Select relevant property fields
        geo_fields = properties_df.select(
            "property_id",
            "tract",
            "county",
            "cbsa",
            "state",
            "property_type",
            "year_built",
            "square_feet",
            "latitude",
            "longitude"
        )
        
        # Join with repeat sales
        merged_df = repeat_sales_df.join(
            geo_fields,
            on="property_id",
            how="left"
        )
        
        return merged_df
        
    def filter_outliers(
        self,
        repeat_sales_df: DataFrame,
        method: str = "iqr"
    ) -> DataFrame:
        """
        Filter outlier transactions based on price ratios.
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            method: Outlier detection method ('iqr' or 'std')
            
        Returns:
            Filtered DataFrame
        """
        logger.info(f"Filtering outliers using {method} method")
        
        if method == "iqr":
            # Calculate IQR for log price ratio
            quantiles = repeat_sales_df.select(
                F.expr("percentile_approx(log_price_ratio, 0.25)").alias("q1"),
                F.expr("percentile_approx(log_price_ratio, 0.75)").alias("q3")
            ).collect()[0]
            
            q1, q3 = quantiles["q1"], quantiles["q3"]
            
            # Handle case where quantiles are None (insufficient data)
            if q1 is None or q3 is None:
                logger.warning("Insufficient data for IQR calculation, skipping outlier filtering")
                return repeat_sales_df
                
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            filtered_df = repeat_sales_df.filter(
                (F.col("log_price_ratio") >= lower_bound) &
                (F.col("log_price_ratio") <= upper_bound)
            )
            
        elif method == "std":
            # Calculate mean and std for log price ratio
            stats = repeat_sales_df.select(
                F.mean("log_price_ratio").alias("mean"),
                F.stddev("log_price_ratio").alias("std")
            ).collect()[0]
            
            mean, std = stats["mean"], stats["std"]
            
            # Handle case where mean or std are None (insufficient data)
            if mean is None or std is None:
                logger.warning("Insufficient data for std calculation, skipping outlier filtering")
                return repeat_sales_df
                
            threshold = self.config.outlier_std_threshold
            
            filtered_df = repeat_sales_df.filter(
                F.abs(F.col("log_price_ratio") - mean) <= threshold * std
            )
            
        else:
            raise ValueError(f"Unknown outlier method: {method}")
            
        logger.info(f"Retained {filtered_df.count()} pairs after outlier filtering")
        return filtered_df
        
    def validate_data(self, df: DataFrame) -> QualityMetrics:
        """
        Validate data quality and flag issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            QualityMetrics with validation results
        """
        logger.info("Validating data quality")
        
        total_count = df.count()
        
        # Calculate missing fields
        missing_fields = {}
        required_fields = [
            "property_id", "sale1_price", "sale2_price",
            "sale1_date", "sale2_date"
        ]
        
        for field in required_fields:
            if field in df.columns:
                missing_count = df.filter(F.col(field).isNull()).count()
                if missing_count > 0:
                    missing_fields[field] = missing_count
        
        # Calculate validation errors
        validation_errors = {}
        
        # Check for invalid price ratios using log_price_ratio
        if "log_price_ratio" in df.columns:
            invalid_ratio_count = df.filter(
                (F.col("log_price_ratio") > F.log(F.lit(10.0))) |  # More than 10x increase
                (F.col("log_price_ratio") < F.log(F.lit(0.1)))     # More than 90% decrease
            ).count()
            if invalid_ratio_count > 0:
                validation_errors["invalid_price_ratio"] = invalid_ratio_count
        
        # Count valid records (no missing required fields and valid price ratio)
        valid_df = df
        for field in required_fields:
            if field in df.columns:
                valid_df = valid_df.filter(F.col(field).isNotNull())
        
        if "log_price_ratio" in df.columns:
            valid_df = valid_df.filter(
                (F.col("log_price_ratio") <= F.log(F.lit(10.0))) &
                (F.col("log_price_ratio") >= F.log(F.lit(0.1)))
            )
        
        valid_count = valid_df.count()
        invalid_count = total_count - valid_count
        
        # Calculate scores
        completeness_score = 1.0 - (sum(missing_fields.values()) / (total_count * len(required_fields)) if total_count > 0 else 0)
        validity_score = valid_count / total_count if total_count > 0 else 0.0
        overall_score = (completeness_score + validity_score) / 2
        
        logger.info(f"Validation complete: {valid_count}/{total_count} valid records")
        
        return QualityMetrics(
            total_records=total_count,
            valid_records=valid_count,
            invalid_records=invalid_count,
            missing_fields=missing_fields,
            validation_errors=validation_errors,
            completeness_score=completeness_score,
            validity_score=validity_score,
            overall_score=overall_score
        )
        
    def prepare_analysis_data(
        self,
        transactions_path: str,
        properties_path: str
    ) -> Tuple[DataFrame, Dict[str, Any]]:
        """
        Complete data preparation pipeline.
        
        Args:
            transactions_path: Path to transactions file
            properties_path: Path to properties file
            
        Returns:
            Tuple of (prepared DataFrame, preparation metrics)
        """
        # Load data
        transactions_df = self.load_transactions(transactions_path)
        properties_df = self.load_properties(properties_path)
        
        # Cache for performance
        transactions_df.cache()
        properties_df.cache()
        
        # Identify repeat sales
        repeat_sales_df = self.identify_repeat_sales(transactions_df)
        
        # Merge geographic data
        repeat_sales_df = self.merge_geographic_data(repeat_sales_df, properties_df)
        
        # Filter outliers
        repeat_sales_df = self.filter_outliers(repeat_sales_df)
        
        # Validate data
        repeat_sales_df, validation_metrics = self.validate_data(repeat_sales_df)
        
        # Prepare final metrics
        metrics = {
            "num_transactions": transactions_df.count(),
            "num_properties": properties_df.count(),
            "num_repeat_pairs": repeat_sales_df.count(),
            "num_valid_pairs": repeat_sales_df.filter(F.col("is_valid")).count(),
            "validation": validation_metrics
        }
        
        # Unpersist cached data
        transactions_df.unpersist()
        properties_df.unpersist()
        
        return repeat_sales_df, metrics