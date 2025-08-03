"""Data processing pipeline for HPI-FHFA project"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import Dict, List, Tuple, Optional
import logging

from ..utils.logging_config import setup_logging


class DataProcessor:
    """Main data processing class for creating repeat-sales pairs and calculating metrics"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = setup_logging(self.__class__.__name__)
        
    def create_repeat_sales_pairs(self, transactions: DataFrame) -> DataFrame:
        """
        Create repeat-sales pairs using self-join with window functions
        
        Args:
            transactions: DataFrame with transaction data
            
        Returns:
            DataFrame with repeat-sales pairs
        """
        self.logger.info("Creating repeat-sales pairs from transactions")
        
        # Add transaction sequence numbers per property
        window_spec = Window.partitionBy("property_id").orderBy("transaction_date")
        
        transactions_numbered = transactions.withColumn(
            "transaction_seq", F.row_number().over(window_spec)
        )
        
        # Self-join to create pairs
        t1 = transactions_numbered.alias("t1")
        t2 = transactions_numbered.alias("t2")
        
        repeat_sales = t1.join(
            t2,
            (F.col("t1.property_id") == F.col("t2.property_id")) &
            (F.col("t1.transaction_seq") == F.col("t2.transaction_seq") - 1),
            "inner"
        ).select(
            F.col("t1.property_id").alias("property_id"),
            F.col("t1.transaction_date").alias("sale_date_1"),
            F.col("t1.transaction_price").alias("sale_price_1"),
            F.col("t2.transaction_date").alias("sale_date_2"),
            F.col("t2.transaction_price").alias("sale_price_2"),
            F.col("t1.census_tract").alias("census_tract"),
            F.col("t1.cbsa_code").alias("cbsa_code"),
            F.col("t1.distance_to_cbd").alias("distance_to_cbd")
        )
        
        # Calculate price relatives and time differences
        repeat_sales = repeat_sales.withColumn(
            "price_relative", 
            F.log(F.col("sale_price_2")) - F.log(F.col("sale_price_1"))
        ).withColumn(
            "time_diff_years",
            F.datediff(F.col("sale_date_2"), F.col("sale_date_1")) / 365.25
        ).withColumn(
            "cagr",
            F.pow(F.col("sale_price_2") / F.col("sale_price_1"), 
                  1.0 / F.col("time_diff_years")) - 1
        )
        
        pairs_count = repeat_sales.count()
        self.logger.info(f"Created {pairs_count:,} repeat-sales pairs")
        
        return repeat_sales
    
    def apply_filters(self, repeat_sales: DataFrame) -> DataFrame:
        """
        Apply data quality filters as per PRD specifications
        
        Filters:
        1. Not in same 12-month period
        2. CAGR between -30% and +30%
        3. Cumulative appreciation between 0.25x and 10x
        
        Args:
            repeat_sales: DataFrame with repeat-sales pairs
            
        Returns:
            Filtered DataFrame
        """
        initial_count = repeat_sales.count()
        self.logger.info(f"Applying filters to {initial_count:,} repeat-sales pairs")
        
        filtered = repeat_sales.filter(
            # Not in same year (must be different years)
            (F.year("sale_date_2") != F.year("sale_date_1"))
        ).filter(
            # CAGR filter
            (F.abs(F.col("cagr")) <= 0.30)
        ).filter(
            # Cumulative appreciation filter
            (F.col("sale_price_2") / F.col("sale_price_1")).between(0.25, 10.0)
        )
        
        final_count = filtered.count()
        self.logger.info(
            f"Filtered pairs: {initial_count:,} → {final_count:,} "
            f"({(initial_count - final_count) / initial_count * 100:.1f}% removed)"
        )
        
        return filtered
    
    def calculate_half_pairs(self, repeat_sales: DataFrame) -> DataFrame:
        """
        Calculate half-pairs for each tract-period combination
        
        For a repeat-sales pair, we count:
        - 1 half-pair for the first sale period
        - 1 half-pair for the second sale period
        
        Args:
            repeat_sales: DataFrame with repeat-sales pairs
            
        Returns:
            DataFrame with half-pair counts by tract/year
        """
        self.logger.info("Calculating half-pairs by tract and year")
        
        # Create half-pairs for first sale
        half_pairs_1 = repeat_sales.select(
            F.col("census_tract"),
            F.col("cbsa_code"),
            F.year("sale_date_1").alias("year"),
            F.lit(1).alias("half_pairs")
        )
        
        # Create half-pairs for second sale
        half_pairs_2 = repeat_sales.select(
            F.col("census_tract"),
            F.col("cbsa_code"),
            F.year("sale_date_2").alias("year"),
            F.lit(1).alias("half_pairs")
        )
        
        # Union and aggregate
        half_pairs = half_pairs_1.union(half_pairs_2).groupBy(
            "census_tract", "cbsa_code", "year"
        ).agg(
            F.sum("half_pairs").alias("total_half_pairs")
        )
        
        self.logger.info(f"Calculated half-pairs for {half_pairs.count():,} tract-year combinations")
        
        return half_pairs
    
    def add_supertract_mapping(
        self, 
        repeat_sales: DataFrame, 
        supertract_mapping: DataFrame
    ) -> DataFrame:
        """
        Add supertract ID to repeat-sales pairs
        
        Args:
            repeat_sales: DataFrame with repeat-sales pairs
            supertract_mapping: DataFrame with tract to supertract mapping
            
        Returns:
            DataFrame with supertract_id added
        """
        return repeat_sales.join(
            supertract_mapping.select("census_tract", "supertract_id"),
            on="census_tract",
            how="left"
        )
    
    def create_regression_ready_data(
        self,
        repeat_sales: DataFrame,
        start_year: int,
        end_year: int
    ) -> DataFrame:
        """
        Prepare data for BMN regression by filtering to specific time period
        
        Args:
            repeat_sales: DataFrame with repeat-sales pairs
            start_year: Start year for analysis
            end_year: End year for analysis
            
        Returns:
            DataFrame ready for regression
        """
        return repeat_sales.filter(
            (F.year("sale_date_1") >= start_year) &
            (F.year("sale_date_2") <= end_year)
        )