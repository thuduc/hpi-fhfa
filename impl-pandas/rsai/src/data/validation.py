"""Data validation utilities for RSAI"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data integrity and quality throughout the pipeline"""
    
    @staticmethod
    def validate_transactions(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate transaction data quality"""
        issues = {
            'missing_values': {},
            'data_quality': {},
            'statistics': {}
        }
        
        # Check for missing values
        for col in ['property_id', 'transaction_date', 'transaction_price', 
                    'census_tract_2010', 'cbsa_id']:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues['missing_values'][col] = missing_count
        
        # Check for negative or zero prices
        invalid_prices = (df['transaction_price'] <= 0).sum()
        if invalid_prices > 0:
            issues['data_quality']['invalid_prices'] = invalid_prices
        
        # Check for future dates
        if pd.Timestamp.now() < df['transaction_date'].max():
            issues['data_quality']['future_dates'] = True
        
        # Basic statistics
        issues['statistics'] = {
            'total_records': len(df),
            'unique_properties': df['property_id'].nunique(),
            'date_range': f"{df['transaction_date'].min()} to {df['transaction_date'].max()}",
            'price_range': f"${df['transaction_price'].min():,.0f} to ${df['transaction_price'].max():,.0f}",
            'unique_cbsas': df['cbsa_id'].nunique(),
            'unique_tracts': df['census_tract_2010'].nunique()
        }
        
        return issues
    
    @staticmethod
    def validate_repeat_sales(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate repeat sales pairs"""
        issues = {
            'data_quality': {},
            'statistics': {}
        }
        
        # Check for invalid time periods
        invalid_periods = (df['years_between_sales'] <= 0).sum()
        if invalid_periods > 0:
            issues['data_quality']['invalid_periods'] = invalid_periods
        
        # Check for extreme growth rates
        extreme_growth = (abs(df['annual_growth_rate']) > 1.0).sum()  # >100% annual
        if extreme_growth > 0:
            issues['data_quality']['extreme_growth_rates'] = extreme_growth
        
        # Statistics
        issues['statistics'] = {
            'total_pairs': len(df),
            'avg_years_between': df['years_between_sales'].mean(),
            'median_appreciation': df['cumulative_appreciation'].median(),
            'avg_annual_growth': df['annual_growth_rate'].mean()
        }
        
        return issues
    
    @staticmethod
    def validate_geographic_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate geographic data"""
        issues = {
            'data_quality': {},
            'statistics': {}
        }
        
        # Check coordinate bounds
        invalid_lat = ((df['centroid_lat'] < -90) | (df['centroid_lat'] > 90)).sum()
        invalid_lon = ((df['centroid_lon'] < -180) | (df['centroid_lon'] > 180)).sum()
        
        if invalid_lat > 0:
            issues['data_quality']['invalid_latitude'] = invalid_lat
        if invalid_lon > 0:
            issues['data_quality']['invalid_longitude'] = invalid_lon
        
        # Check for duplicates
        duplicate_tracts = df['census_tract_2010'].duplicated().sum()
        if duplicate_tracts > 0:
            issues['data_quality']['duplicate_tracts'] = duplicate_tracts
        
        issues['statistics'] = {
            'total_tracts': len(df),
            'unique_tracts': df['census_tract_2010'].nunique()
        }
        
        return issues


def log_validation_results(validation_results: Dict[str, Any], data_type: str):
    """Log validation results in a structured format"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Validation Results for {data_type}")
    logger.info(f"{'='*50}")
    
    if validation_results.get('missing_values'):
        logger.warning("Missing Values:")
        for col, count in validation_results['missing_values'].items():
            logger.warning(f"  - {col}: {count} missing")
    
    if validation_results.get('data_quality'):
        logger.warning("Data Quality Issues:")
        for issue, details in validation_results['data_quality'].items():
            logger.warning(f"  - {issue}: {details}")
    
    if validation_results.get('statistics'):
        logger.info("Data Statistics:")
        for stat, value in validation_results['statistics'].items():
            logger.info(f"  - {stat}: {value}")