"""Data validation utilities for HPI-FHFA pipeline"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from typing import Dict, List, Optional
import logging

from ..utils.logging_config import setup_logging


class DataValidator:
    """Validate data quality throughout the pipeline"""
    
    def __init__(self, spark):
        self.spark = spark
        self.logger = setup_logging(self.__class__.__name__)
    
    def validate_transactions(self, df: DataFrame) -> Dict[str, bool]:
        """
        Validate transaction data quality
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Dictionary of validation results
        """
        self.logger.info("Validating transaction data")
        
        results = {}
        
        # Check for required columns
        required_cols = ["property_id", "transaction_date", "transaction_price", 
                        "census_tract", "cbsa_code"]
        results["has_required_columns"] = all(col in df.columns for col in required_cols)
        
        # Check for nulls in critical columns
        null_counts = {}
        for col in required_cols:
            if col in df.columns:
                null_count = df.filter(F.col(col).isNull()).count()
                null_counts[col] = null_count
                results[f"no_nulls_{col}"] = null_count == 0
        
        # Check price ranges
        if "transaction_price" in df.columns:
            price_stats = df.select(
                F.min("transaction_price").alias("min_price"),
                F.max("transaction_price").alias("max_price"),
                F.avg("transaction_price").alias("avg_price")
            ).collect()[0]
            
            results["valid_price_range"] = (
                price_stats["min_price"] > 0 and 
                price_stats["max_price"] < 100_000_000  # $100M cap
            )
            
            self.logger.info(
                f"Price range: ${price_stats['min_price']:,.0f} - "
                f"${price_stats['max_price']:,.0f} "
                f"(avg: ${price_stats['avg_price']:,.0f})"
            )
        
        # Check date ranges
        if "transaction_date" in df.columns:
            date_stats = df.select(
                F.min("transaction_date").alias("min_date"),
                F.max("transaction_date").alias("max_date")
            ).collect()[0]
            
            self.logger.info(
                f"Date range: {date_stats['min_date']} - {date_stats['max_date']}"
            )
        
        # Log validation summary
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        self.logger.info(f"Validation summary: {passed}/{total} checks passed")
        
        if null_counts:
            self.logger.warning(f"Null counts: {null_counts}")
        
        return results
    
    def validate_repeat_sales(self, df: DataFrame) -> Dict[str, bool]:
        """
        Validate repeat-sales pairs data quality
        
        Args:
            df: Repeat-sales DataFrame
            
        Returns:
            Dictionary of validation results
        """
        self.logger.info("Validating repeat-sales data")
        
        results = {}
        
        # Check that second sale is after first sale
        invalid_dates = df.filter(F.col("sale_date_2") <= F.col("sale_date_1")).count()
        results["valid_date_order"] = invalid_dates == 0
        
        if invalid_dates > 0:
            self.logger.warning(f"Found {invalid_dates} pairs with invalid date order")
        
        # Check that prices are positive
        invalid_prices = df.filter(
            (F.col("sale_price_1") <= 0) | (F.col("sale_price_2") <= 0)
        ).count()
        results["positive_prices"] = invalid_prices == 0
        
        # Check CAGR distribution
        cagr_stats = df.select(
            F.min("cagr").alias("min_cagr"),
            F.max("cagr").alias("max_cagr"),
            F.avg("cagr").alias("avg_cagr"),
            F.stddev("cagr").alias("std_cagr")
        ).collect()[0]
        
        self.logger.info(
            f"CAGR stats: min={cagr_stats['min_cagr']:.3f}, "
            f"max={cagr_stats['max_cagr']:.3f}, "
            f"avg={cagr_stats['avg_cagr']:.3f}, "
            f"std={cagr_stats['std_cagr']:.3f}"
        )
        
        # Check time differences
        time_diff_stats = df.select(
            F.min("time_diff_years").alias("min_years"),
            F.max("time_diff_years").alias("max_years"),
            F.avg("time_diff_years").alias("avg_years")
        ).collect()[0]
        
        self.logger.info(
            f"Time between sales: {time_diff_stats['min_years']:.1f} - "
            f"{time_diff_stats['max_years']:.1f} years "
            f"(avg: {time_diff_stats['avg_years']:.1f})"
        )
        
        results["reasonable_time_gaps"] = (
            time_diff_stats["min_years"] > 0 and 
            time_diff_stats["max_years"] < 50
        )
        
        return results
    
    def validate_half_pairs(self, df: DataFrame, min_threshold: int = 40) -> Dict[str, any]:
        """
        Validate half-pairs data and check for sufficient observations
        
        Args:
            df: Half-pairs DataFrame
            min_threshold: Minimum half-pairs threshold
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating half-pairs data")
        
        results = {}
        
        # Count tracts below threshold
        below_threshold = df.filter(
            F.col("total_half_pairs") < min_threshold
        ).count()
        
        total_tracts = df.select("census_tract").distinct().count()
        
        results["tracts_below_threshold"] = below_threshold
        results["total_tracts"] = total_tracts
        results["percent_below_threshold"] = (below_threshold / total_tracts * 100) if total_tracts > 0 else 0
        
        self.logger.info(
            f"{below_threshold:,} of {total_tracts:,} tracts "
            f"({results['percent_below_threshold']:.1f}%) have fewer than "
            f"{min_threshold} half-pairs"
        )
        
        # Distribution of half-pairs
        distribution = df.select(
            F.min("total_half_pairs").alias("min"),
            F.percentile_approx("total_half_pairs", 0.25).alias("q1"),
            F.percentile_approx("total_half_pairs", 0.50).alias("median"),
            F.percentile_approx("total_half_pairs", 0.75).alias("q3"),
            F.max("total_half_pairs").alias("max")
        ).collect()[0]
        
        self.logger.info(
            f"Half-pairs distribution: min={distribution['min']}, "
            f"Q1={distribution['q1']}, median={distribution['median']}, "
            f"Q3={distribution['q3']}, max={distribution['max']}"
        )
        
        results["distribution"] = distribution.asDict()
        
        return results