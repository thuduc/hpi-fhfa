"""
Data validation module for RSAI using PySpark.

This module provides validation functionality for transactions,
properties, and repeat sales data.
"""

import logging
from typing import Dict, List, Optional
from datetime import date, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from rsai.src.data.models import (
    QualityMetrics,
    RSAIConfig,
    TransactionType
)

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality for RSAI model."""
    
    def __init__(self, spark: SparkSession, config: RSAIConfig):
        """
        Initialize data validator.
        
        Args:
            spark: SparkSession instance
            config: RSAI configuration
        """
        self.spark = spark
        self.config = config
        
    def validate_transactions(self, df: DataFrame) -> QualityMetrics:
        """
        Validate transaction data quality.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            QualityMetrics with validation results
        """
        total_records = df.count()
        
        # Check for missing required fields
        missing_fields = {}
        required_fields = ["transaction_id", "property_id", "sale_date", "sale_price"]
        
        for field in required_fields:
            if field in df.columns:
                missing_count = df.filter(F.col(field).isNull()).count()
                if missing_count > 0:
                    missing_fields[field] = missing_count
                    
        # Validation rules
        validation_errors = {}
        
        # Price range validation
        if "sale_price" in df.columns:
            invalid_price = df.filter(
                (F.col("sale_price") <= 0) |
                (F.col("sale_price") < self.config.min_price) |
                (F.col("sale_price") > self.config.max_price)
            ).count()
            if invalid_price > 0:
                validation_errors["price_range"] = invalid_price
                
        # Date validation
        if "sale_date" in df.columns:
            # Check for future dates
            future_dates = df.filter(
                F.col("sale_date") > F.current_date()
            ).count()
            if future_dates > 0:
                validation_errors["future_dates"] = future_dates
                
            # Check for very old dates
            min_date = date(1900, 1, 1)
            old_dates = df.filter(
                F.col("sale_date") < F.lit(min_date)
            ).count()
            if old_dates > 0:
                validation_errors["old_dates"] = old_dates
                
        # Transaction type validation
        if "transaction_type" in df.columns:
            valid_types = [t.value for t in TransactionType]
            invalid_types = df.filter(
                ~F.col("transaction_type").isin(valid_types)
            ).count()
            if invalid_types > 0:
                validation_errors["invalid_transaction_type"] = invalid_types
                
        # Calculate valid records
        valid_records = df
        for field in required_fields:
            if field in df.columns:
                valid_records = valid_records.filter(F.col(field).isNotNull())
                
        if "sale_price" in df.columns:
            valid_records = valid_records.filter(
                (F.col("sale_price") > 0) &
                (F.col("sale_price") >= self.config.min_price) &
                (F.col("sale_price") <= self.config.max_price)
            )
            
        valid_count = valid_records.count()
        invalid_count = total_records - valid_count
        
        # Calculate scores (with safe division)
        total_expected_fields = total_records * len(required_fields)
        if total_expected_fields > 0:
            completeness_score = 1.0 - (sum(missing_fields.values()) / total_expected_fields)
        else:
            completeness_score = 1.0 if total_records == 0 else 0.0
            
        validity_score = valid_count / total_records if total_records > 0 else 0.0
        overall_score = (completeness_score + validity_score) / 2
        
        return QualityMetrics(
            total_records=total_records,
            valid_records=valid_count,
            invalid_records=invalid_count,
            missing_fields=missing_fields,
            validation_errors=validation_errors,
            completeness_score=completeness_score,
            validity_score=validity_score,
            overall_score=overall_score
        )
        
    def validate_repeat_sales(self, df: DataFrame) -> QualityMetrics:
        """
        Validate repeat sales data quality.
        
        Args:
            df: Repeat sales DataFrame
            
        Returns:
            QualityMetrics with validation results
        """
        total_records = df.count()
        
        # Check for missing required fields
        missing_fields = {}
        required_fields = [
            "pair_id", "property_id",
            "sale1_date", "sale1_price",
            "sale2_date", "sale2_price",
            "holding_period_days", "log_price_ratio"
        ]
        
        for field in required_fields:
            if field in df.columns:
                missing_count = df.filter(F.col(field).isNull()).count()
                if missing_count > 0:
                    missing_fields[field] = missing_count
                    
        # Validation rules
        validation_errors = {}
        
        # Holding period validation
        if "holding_period_days" in df.columns:
            max_days = self.config.max_holding_period_years * 365
            invalid_holding = df.filter(
                (F.col("holding_period_days") <= 0) |
                (F.col("holding_period_days") > max_days)
            ).count()
            if invalid_holding > 0:
                validation_errors["holding_period"] = invalid_holding
                
        # Price ratio validation
        if "log_price_ratio" in df.columns:
            # Check for extreme price changes (e.g., > 500% or < -80%)
            extreme_changes = df.filter(
                (F.col("log_price_ratio") > F.log(F.lit(5.0))) |
                (F.col("log_price_ratio") < F.log(F.lit(0.2)))
            ).count()
            if extreme_changes > 0:
                validation_errors["extreme_price_changes"] = extreme_changes
                
        # Date consistency
        if all(field in df.columns for field in ["sale1_date", "sale2_date"]):
            date_errors = df.filter(
                F.col("sale2_date") <= F.col("sale1_date")
            ).count()
            if date_errors > 0:
                validation_errors["date_consistency"] = date_errors
                
        # Price consistency
        if all(field in df.columns for field in ["sale1_price", "sale2_price", "log_price_ratio"]):
            # Verify log price ratio calculation
            expected_ratio = F.log(F.col("sale2_price") / F.col("sale1_price"))
            ratio_errors = df.filter(
                F.abs(F.col("log_price_ratio") - expected_ratio) > 0.001
            ).count()
            if ratio_errors > 0:
                validation_errors["price_ratio_calculation"] = ratio_errors
                
        # Calculate valid records
        valid_records = df
        for field in required_fields:
            if field in df.columns:
                valid_records = valid_records.filter(F.col(field).isNotNull())
                
        if "holding_period_days" in df.columns:
            max_days = self.config.max_holding_period_years * 365
            valid_records = valid_records.filter(
                (F.col("holding_period_days") > 0) &
                (F.col("holding_period_days") <= max_days)
            )
            
        valid_count = valid_records.count()
        invalid_count = total_records - valid_count
        
        # Calculate scores (with safe division)
        total_expected_fields = total_records * len(required_fields)
        if total_expected_fields > 0:
            completeness_score = 1.0 - (sum(missing_fields.values()) / total_expected_fields)
        else:
            completeness_score = 1.0 if total_records == 0 else 0.0
            
        validity_score = valid_count / total_records if total_records > 0 else 0.0
        overall_score = (completeness_score + validity_score) / 2
        
        return QualityMetrics(
            total_records=total_records,
            valid_records=valid_count,
            invalid_records=invalid_count,
            missing_fields=missing_fields,
            validation_errors=validation_errors,
            completeness_score=completeness_score,
            validity_score=validity_score,
            overall_score=overall_score
        )
        
    def validate_properties(self, df: DataFrame) -> QualityMetrics:
        """
        Validate property data quality.
        
        Args:
            df: Property DataFrame
            
        Returns:
            QualityMetrics with validation results
        """
        total_records = df.count()
        
        # Check for missing required fields
        missing_fields = {}
        required_fields = ["property_id", "latitude", "longitude", "tract"]
        
        for field in required_fields:
            if field in df.columns:
                missing_count = df.filter(F.col(field).isNull()).count()
                if missing_count > 0:
                    missing_fields[field] = missing_count
                    
        # Validation rules
        validation_errors = {}
        
        # Coordinate validation
        if all(field in df.columns for field in ["latitude", "longitude"]):
            invalid_coords = df.filter(
                (F.col("latitude") < -90) | (F.col("latitude") > 90) |
                (F.col("longitude") < -180) | (F.col("longitude") > 180) |
                F.col("latitude").isNull() | F.col("longitude").isNull()
            ).count()
            if invalid_coords > 0:
                validation_errors["invalid_coordinates"] = invalid_coords
                
        # Year built validation
        if "year_built" in df.columns:
            current_year = date.today().year
            invalid_year = df.filter(
                (F.col("year_built") < 1800) |
                (F.col("year_built") > current_year)
            ).count()
            if invalid_year > 0:
                validation_errors["invalid_year_built"] = invalid_year
                
        # Square feet validation
        if "square_feet" in df.columns:
            invalid_sqft = df.filter(
                (F.col("square_feet") <= 0) |
                (F.col("square_feet") > 50000)  # Reasonable max
            ).count()
            if invalid_sqft > 0:
                validation_errors["invalid_square_feet"] = invalid_sqft
                
        # Calculate valid records
        valid_count = total_records - sum(validation_errors.values())
        invalid_count = sum(validation_errors.values())
        
        # Calculate scores (with safe division)
        total_expected_fields = total_records * len(required_fields)
        if total_expected_fields > 0:
            completeness_score = 1.0 - (sum(missing_fields.values()) / total_expected_fields)
        else:
            completeness_score = 1.0 if total_records == 0 else 0.0
            
        validity_score = valid_count / total_records if total_records > 0 else 0.0
        overall_score = (completeness_score + validity_score) / 2
        
        return QualityMetrics(
            total_records=total_records,
            valid_records=valid_count,
            invalid_records=invalid_count,
            missing_fields=missing_fields,
            validation_errors=validation_errors,
            completeness_score=completeness_score,
            validity_score=validity_score,
            overall_score=overall_score
        )