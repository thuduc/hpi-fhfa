"""Data validation rules for HPI-FHFA."""

import polars as pl
from typing import List, Dict, Any, Optional
import structlog

from .schemas import validate_schema, TRANSACTION_SCHEMA, GEOGRAPHIC_SCHEMA
from ..config.constants import (
    MIN_TRANSACTION_DATE,
    MAX_TRANSACTION_DATE,
    CENSUS_TRACT_YEAR
)
from ..utils.exceptions import DataValidationError

logger = structlog.get_logger()


class DataValidator:
    """Validate data integrity and quality."""
    
    def __init__(self, config: Any):
        """Initialize validator with configuration.
        
        Args:
            config: HPI configuration object
        """
        self.config = config
        self.strict = config.strict_validation
        
    def validate_transactions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate transaction data.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            Validated DataFrame
            
        Raises:
            DataValidationError: If validation fails in strict mode
        """
        logger.info("Validating transaction data", n_rows=len(df))
        
        # Validate schema
        try:
            validate_schema(df, TRANSACTION_SCHEMA)
        except ValueError as e:
            raise DataValidationError(f"Schema validation failed: {e}")
        
        # Run validation checks
        issues = []
        
        # Check for missing values
        null_counts = self._check_nulls(df)
        if null_counts:
            issues.append(f"Found null values: {null_counts}")
        
        # Validate price range
        invalid_prices = self._check_price_range(df)
        if invalid_prices > 0:
            issues.append(f"Found {invalid_prices} invalid prices")
        
        # Validate date range
        invalid_dates = self._check_date_range(df)
        if invalid_dates > 0:
            issues.append(f"Found {invalid_dates} invalid dates")
        
        # Check for duplicate transactions
        duplicates = self._check_duplicates(df)
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate transactions")
        
        # Handle validation results
        if issues:
            logger.warning("Data validation issues found", issues=issues)
            if self.strict:
                raise DataValidationError(
                    f"Validation failed with {len(issues)} issues: " + "; ".join(issues)
                )
        else:
            logger.info("Transaction data validation passed")
        
        # Remove invalid records if not in strict mode
        if not self.strict and issues:
            df = self._clean_transactions(df)
            logger.info("Cleaned transaction data", n_rows_after=len(df))
        
        return df
    
    def validate_geographic_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate geographic/census tract data.
        
        Args:
            df: Geographic DataFrame
            
        Returns:
            Validated DataFrame
            
        Raises:
            DataValidationError: If validation fails
        """
        logger.info("Validating geographic data", n_rows=len(df))
        
        # Validate schema
        try:
            validate_schema(df, GEOGRAPHIC_SCHEMA)
        except ValueError as e:
            raise DataValidationError(f"Schema validation failed: {e}")
        
        issues = []
        
        # Check for missing tract IDs
        if df["tract_id"].null_count() > 0:
            issues.append("Found null tract IDs")
        
        # Check for duplicate tracts
        duplicate_tracts = len(df) - df["tract_id"].n_unique()
        if duplicate_tracts > 0:
            issues.append(f"Found {duplicate_tracts} duplicate tract IDs")
        
        # Validate coordinate ranges
        invalid_coords = (
            (df["centroid_lat"].is_null() | 
             df["centroid_lon"].is_null() |
             (df["centroid_lat"].abs() > 90) |
             (df["centroid_lon"].abs() > 180))
            .sum()
        )
        if invalid_coords > 0:
            issues.append(f"Found {invalid_coords} invalid coordinates")
        
        # Validate shares are between 0 and 1
        invalid_shares = (
            ((df["college_share"] < 0) | (df["college_share"] > 1) |
             (df["nonwhite_share"] < 0) | (df["nonwhite_share"] > 1))
            .sum()
        )
        if invalid_shares > 0:
            issues.append(f"Found {invalid_shares} invalid share values")
        
        if issues:
            logger.error("Geographic data validation failed", issues=issues)
            raise DataValidationError(
                f"Geographic validation failed: " + "; ".join(issues)
            )
        
        logger.info("Geographic data validation passed")
        return df
    
    def _check_nulls(self, df: pl.DataFrame) -> Dict[str, int]:
        """Check for null values in required columns."""
        null_counts = {}
        for col in ["property_id", "transaction_date", "transaction_price", "census_tract"]:
            count = df[col].null_count()
            if count > 0:
                null_counts[col] = count
        return null_counts
    
    def _check_price_range(self, df: pl.DataFrame) -> int:
        """Check for invalid transaction prices."""
        return (
            (df["transaction_price"] <= 0) | 
            (df["transaction_price"] > 100_000_000)  # $100M upper bound
        ).sum()
    
    def _check_date_range(self, df: pl.DataFrame) -> int:
        """Check for dates outside valid range."""
        return (
            (df["transaction_date"] < MIN_TRANSACTION_DATE) |
            (df["transaction_date"] > MAX_TRANSACTION_DATE)
        ).sum()
    
    def _check_duplicates(self, df: pl.DataFrame) -> int:
        """Check for duplicate transactions (same property, same date)."""
        # Count total duplicates (not unique groups)
        dup_groups = df.group_by(["property_id", "transaction_date"]).len().filter(
            pl.col("len") > 1
        )
        
        if len(dup_groups) == 0:
            return 0
        
        # Sum up the duplicate counts minus 1 for each group (to get extra records)
        return (dup_groups["len"] - 1).sum()
    
    def _clean_transactions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove invalid transactions from DataFrame."""
        # Remove nulls
        df = df.drop_nulls(subset=["property_id", "transaction_date", "transaction_price", "census_tract"])
        
        # Filter valid prices
        df = df.filter(
            (pl.col("transaction_price") > 0) & 
            (pl.col("transaction_price") <= 100_000_000)
        )
        
        # Filter valid dates
        df = df.filter(
            (pl.col("transaction_date") >= MIN_TRANSACTION_DATE) &
            (pl.col("transaction_date") <= MAX_TRANSACTION_DATE)
        )
        
        # Remove duplicates (keep first)
        df = df.unique(subset=["property_id", "transaction_date"], keep="first")
        
        return df