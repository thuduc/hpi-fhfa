"""
Data validation module using Polars.

This module provides comprehensive data validation for transaction data,
repeat sales pairs, and geographic information using Polars DataFrames.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime
from collections import defaultdict

import polars as pl
import numpy as np

from rsai.src.data.models import (
    QualityMetrics,
    TransactionType,
    PropertyType,
    GeographyLevel,
    RSAIConfig
)

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality for RSAI model using Polars."""
    
    def __init__(self, config: RSAIConfig):
        """
        Initialize validator with configuration.
        
        Args:
            config: RSAI configuration object
        """
        self.config = config
        self.validation_results: Dict[str, QualityMetrics] = {}
        
    def validate_transactions(self, df: pl.DataFrame) -> QualityMetrics:
        """
        Validate transaction data quality.
        
        Args:
            df: Polars DataFrame with transaction data
            
        Returns:
            QualityMetrics object with validation results
        """
        logger.info("Validating transaction data")
        
        total_records = len(df)
        missing_counts = {}
        invalid_counts = {}
        issues = []
        
        # Check required columns
        required_cols = ["property_id", "sale_price", "sale_date", "transaction_id"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            
        # Count missing values
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                missing_counts[col] = null_count
                
        # Validate property_id
        if "property_id" in df.columns:
            invalid_ids = df.filter(
                pl.col("property_id").is_null() | 
                (pl.col("property_id").str.len_chars() == 0)
            ).height
            if invalid_ids > 0:
                invalid_counts["property_id"] = invalid_ids
                issues.append(f"{invalid_ids} invalid property IDs")
                
        # Validate sale_price
        if "sale_price" in df.columns:
            price_issues = df.filter(
                pl.col("sale_price").is_null() |
                (pl.col("sale_price") <= 0) |
                (pl.col("sale_price") < self.config.min_price) |
                (pl.col("sale_price") > self.config.max_price)
            ).height
            if price_issues > 0:
                invalid_counts["sale_price"] = price_issues
                issues.append(f"{price_issues} invalid sale prices")
                
        # Validate sale_date
        if "sale_date" in df.columns:
            # Check for future dates
            future_dates = df.filter(pl.col("sale_date") > datetime.now().date()).height
            if future_dates > 0:
                invalid_counts["sale_date_future"] = future_dates
                issues.append(f"{future_dates} future sale dates")
                
            # Check for very old dates
            old_date_threshold = date(1900, 1, 1)
            old_dates = df.filter(pl.col("sale_date") < old_date_threshold).height
            if old_dates > 0:
                invalid_counts["sale_date_old"] = old_dates
                issues.append(f"{old_dates} sale dates before 1900")
                
        # Validate transaction_type if present
        if "transaction_type" in df.columns:
            valid_types = [t.value for t in TransactionType]
            invalid_types = df.filter(
                ~pl.col("transaction_type").is_in(valid_types)
            ).height
            if invalid_types > 0:
                invalid_counts["transaction_type"] = invalid_types
                issues.append(f"{invalid_types} invalid transaction types")
                
        # Check for duplicates
        if all(col in df.columns for col in ["property_id", "sale_date"]):
            duplicate_count = df.group_by(["property_id", "sale_date"]).agg(
                pl.count().alias("count")
            ).filter(pl.col("count") > 1).select(
                pl.sum("count") - pl.count()
            ).item()
            
            if duplicate_count and duplicate_count > 0:
                issues.append(f"{duplicate_count} duplicate transactions")
                
        # Calculate valid records
        valid_records = total_records
        for col, count in invalid_counts.items():
            valid_records = max(0, valid_records - count)
            
        # Calculate quality scores
        completeness_score = 1 - (sum(missing_counts.values()) / (total_records * len(df.columns)))
        validity_score = valid_records / total_records if total_records > 0 else 0
        consistency_score = 1 - (len(issues) / 10)  # Arbitrary scale
        overall_score = (completeness_score + validity_score + consistency_score) / 3
        
        metrics = QualityMetrics(
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=total_records - valid_records,
            missing_counts=missing_counts,
            invalid_counts=invalid_counts,
            completeness_score=completeness_score,
            validity_score=validity_score,
            consistency_score=max(0, consistency_score),
            overall_score=overall_score,
            issues=issues
        )
        
        self.validation_results["transactions"] = metrics
        return metrics
        
    def validate_repeat_sales(self, df: pl.DataFrame) -> QualityMetrics:
        """
        Validate repeat sales pair data.
        
        Args:
            df: Polars DataFrame with repeat sales data
            
        Returns:
            QualityMetrics object with validation results
        """
        logger.info("Validating repeat sales data")
        
        total_records = len(df)
        missing_counts = {}
        invalid_counts = {}
        issues = []
        
        # Check required columns
        required_cols = [
            "pair_id", "property_id",
            "sale1_price", "sale1_date",
            "sale2_price", "sale2_date",
            "price_ratio", "holding_period_days"
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            
        # Count missing values
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                missing_counts[col] = null_count
                
        # Validate price ratio
        if "price_ratio" in df.columns:
            # Check for extreme ratios
            extreme_ratios = df.filter(
                (pl.col("price_ratio") < 0.1) |
                (pl.col("price_ratio") > 10)
            ).height
            if extreme_ratios > 0:
                invalid_counts["extreme_price_ratio"] = extreme_ratios
                issues.append(f"{extreme_ratios} extreme price ratios")
                
            # Verify calculation
            if all(col in df.columns for col in ["sale2_price", "sale1_price"]):
                calc_error = df.filter(
                    (pl.col("sale2_price") / pl.col("sale1_price") - pl.col("price_ratio")).abs() > 0.001
                ).height
                if calc_error > 0:
                    invalid_counts["price_ratio_calc"] = calc_error
                    issues.append(f"{calc_error} price ratio calculation errors")
                    
        # Validate holding period
        if "holding_period_days" in df.columns:
            invalid_periods = df.filter(
                (pl.col("holding_period_days") <= 0) |
                (pl.col("holding_period_days") > self.config.max_holding_period_years * 365)
            ).height
            if invalid_periods > 0:
                invalid_counts["holding_period"] = invalid_periods
                issues.append(f"{invalid_periods} invalid holding periods")
                
        # Validate date consistency
        if all(col in df.columns for col in ["sale1_date", "sale2_date"]):
            date_issues = df.filter(pl.col("sale2_date") <= pl.col("sale1_date")).height
            if date_issues > 0:
                invalid_counts["date_order"] = date_issues
                issues.append(f"{date_issues} pairs with sale2_date <= sale1_date")
                
        # Check for self-pairs (same transaction)
        if all(col in df.columns for col in ["sale1_transaction_id", "sale2_transaction_id"]):
            self_pairs = df.filter(
                pl.col("sale1_transaction_id") == pl.col("sale2_transaction_id")
            ).height
            if self_pairs > 0:
                invalid_counts["self_pairs"] = self_pairs
                issues.append(f"{self_pairs} self-pairs detected")
                
        # Calculate valid records
        valid_records = total_records
        for col, count in invalid_counts.items():
            valid_records = max(0, valid_records - count)
            
        # Calculate quality scores
        completeness_score = 1 - (sum(missing_counts.values()) / (total_records * len(required_cols)))
        validity_score = valid_records / total_records if total_records > 0 else 0
        consistency_score = 1 - (len(issues) / 10)
        overall_score = (completeness_score + validity_score + consistency_score) / 3
        
        metrics = QualityMetrics(
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=total_records - valid_records,
            missing_counts=missing_counts,
            invalid_counts=invalid_counts,
            completeness_score=completeness_score,
            validity_score=validity_score,
            consistency_score=max(0, consistency_score),
            overall_score=overall_score,
            issues=issues
        )
        
        self.validation_results["repeat_sales"] = metrics
        return metrics
        
    def validate_geographic_data(self, df: pl.DataFrame) -> QualityMetrics:
        """
        Validate geographic data quality.
        
        Args:
            df: Polars DataFrame with geographic data
            
        Returns:
            QualityMetrics object with validation results
        """
        logger.info("Validating geographic data")
        
        total_records = len(df)
        missing_counts = {}
        invalid_counts = {}
        issues = []
        
        # Count missing values
        geo_cols = ["latitude", "longitude", "zip_code", "tract", "county_fips", "msa_code"]
        for col in geo_cols:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    missing_counts[col] = null_count
                    
        # Validate coordinates
        if all(col in df.columns for col in ["latitude", "longitude"]):
            # Check coordinate bounds
            coord_issues = df.filter(
                pl.col("latitude").is_not_null() & 
                pl.col("longitude").is_not_null() &
                (
                    (pl.col("latitude") < -90) | (pl.col("latitude") > 90) |
                    (pl.col("longitude") < -180) | (pl.col("longitude") > 180)
                )
            ).height
            if coord_issues > 0:
                invalid_counts["coordinates"] = coord_issues
                issues.append(f"{coord_issues} invalid coordinates")
                
            # Check for (0,0) coordinates
            zero_coords = df.filter(
                (pl.col("latitude") == 0) & (pl.col("longitude") == 0)
            ).height
            if zero_coords > 0:
                invalid_counts["zero_coordinates"] = zero_coords
                issues.append(f"{zero_coords} properties at (0,0)")
                
        # Validate ZIP codes
        if "zip_code" in df.columns:
            # Check ZIP format (5 or 9 digits)
            invalid_zips = df.filter(
                pl.col("zip_code").is_not_null() &
                ~pl.col("zip_code").str.match(r"^\d{5}(-\d{4})?$")
            ).height
            if invalid_zips > 0:
                invalid_counts["zip_format"] = invalid_zips
                issues.append(f"{invalid_zips} invalid ZIP code formats")
                
        # Validate FIPS codes
        if "county_fips" in df.columns:
            # FIPS should be 5 digits (state + county)
            invalid_fips = df.filter(
                pl.col("county_fips").is_not_null() &
                ~pl.col("county_fips").str.match(r"^\d{5}$")
            ).height
            if invalid_fips > 0:
                invalid_counts["fips_format"] = invalid_fips
                issues.append(f"{invalid_fips} invalid FIPS codes")
                
        # Check geographic consistency
        if all(col in df.columns for col in ["zip_code", "county_fips"]):
            # Group by ZIP and check for multiple counties (potential issue)
            zip_county_groups = df.filter(
                pl.col("zip_code").is_not_null() & 
                pl.col("county_fips").is_not_null()
            ).group_by("zip_code").agg(
                pl.col("county_fips").n_unique().alias("num_counties")
            ).filter(pl.col("num_counties") > 1)
            
            if len(zip_county_groups) > 0:
                issues.append(f"{len(zip_county_groups)} ZIP codes span multiple counties")
                
        # Calculate valid records
        valid_records = total_records - sum(invalid_counts.values())
        
        # Calculate quality scores
        geo_coverage = sum(1 for col in geo_cols if col in df.columns and df[col].null_count() < total_records * 0.5) / len(geo_cols)
        completeness_score = 1 - (sum(missing_counts.values()) / (total_records * len(geo_cols)))
        validity_score = valid_records / total_records if total_records > 0 else 0
        consistency_score = 1 - (len(issues) / 10)
        overall_score = (completeness_score + validity_score + consistency_score + geo_coverage) / 4
        
        metrics = QualityMetrics(
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=total_records - valid_records,
            missing_counts=missing_counts,
            invalid_counts=invalid_counts,
            completeness_score=completeness_score,
            validity_score=validity_score,
            consistency_score=max(0, consistency_score),
            overall_score=overall_score,
            issues=issues
        )
        
        self.validation_results["geographic"] = metrics
        return metrics
        
    def validate_time_series_consistency(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Validate time series consistency for repeat sales.
        
        Args:
            df: Polars DataFrame with repeat sales data
            
        Returns:
            Dictionary with time series validation results
        """
        logger.info("Validating time series consistency")
        
        results = {
            "coverage": {},
            "gaps": [],
            "density": {},
            "issues": []
        }
        
        if "sale2_date" not in df.columns:
            results["issues"].append("Missing sale2_date column")
            return results
            
        # Check date range coverage
        min_date = df["sale2_date"].min()
        max_date = df["sale2_date"].max()
        results["coverage"] = {
            "start": min_date,
            "end": max_date,
            "total_days": (max_date - min_date).days if min_date and max_date else 0
        }
        
        # Analyze by month
        monthly_counts = df.with_columns([
            pl.col("sale2_date").dt.year().alias("year"),
            pl.col("sale2_date").dt.month().alias("month")
        ]).group_by(["year", "month"]).agg(
            pl.count().alias("count")
        ).sort(["year", "month"])
        
        # Check for gaps in monthly data
        if len(monthly_counts) > 0:
            # Create complete month series
            all_months = []
            current = min_date.replace(day=1)
            end = max_date.replace(day=1)
            
            while current <= end:
                all_months.append((current.year, current.month))
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
                    
            # Find missing months
            existing_months = set(zip(monthly_counts["year"], monthly_counts["month"]))
            missing_months = [m for m in all_months if m not in existing_months]
            
            if missing_months:
                results["gaps"] = missing_months
                results["issues"].append(f"{len(missing_months)} months with no data")
                
        # Calculate density statistics
        daily_counts = df.group_by("sale2_date").agg(
            pl.count().alias("count")
        )
        
        results["density"] = {
            "mean_daily": daily_counts["count"].mean(),
            "median_daily": daily_counts["count"].median(),
            "min_daily": daily_counts["count"].min(),
            "max_daily": daily_counts["count"].max(),
            "total_days_with_data": len(daily_counts)
        }
        
        # Check for sufficient data per period
        periods_below_threshold = monthly_counts.filter(
            pl.col("count") < self.config.min_pairs_threshold
        ).height
        
        if periods_below_threshold > 0:
            results["issues"].append(
                f"{periods_below_threshold} months below minimum threshold ({self.config.min_pairs_threshold} pairs)"
            )
            
        return results
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary with validation summary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "min_price": self.config.min_price,
                "max_price": self.config.max_price,
                "max_holding_period_years": self.config.max_holding_period_years,
                "min_pairs_threshold": self.config.min_pairs_threshold
            },
            "results": {},
            "summary": {
                "overall_score": 0,
                "total_issues": 0,
                "critical_issues": []
            }
        }
        
        # Add individual validation results
        for name, metrics in self.validation_results.items():
            report["results"][name] = {
                "total_records": metrics.total_records,
                "valid_records": metrics.valid_records,
                "invalid_records": metrics.invalid_records,
                "scores": {
                    "completeness": metrics.completeness_score,
                    "validity": metrics.validity_score,
                    "consistency": metrics.consistency_score,
                    "overall": metrics.overall_score
                },
                "issues": metrics.issues,
                "missing_counts": metrics.missing_counts,
                "invalid_counts": metrics.invalid_counts
            }
            
            report["summary"]["total_issues"] += len(metrics.issues)
            
            # Identify critical issues
            if metrics.overall_score < 0.7:
                report["summary"]["critical_issues"].append(
                    f"{name}: Low overall score ({metrics.overall_score:.2f})"
                )
                
        # Calculate overall score
        if self.validation_results:
            report["summary"]["overall_score"] = np.mean([
                m.overall_score for m in self.validation_results.values()
            ])
            
        return report