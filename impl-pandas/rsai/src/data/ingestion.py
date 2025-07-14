"""Data ingestion and preparation module for RSAI"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data loading, validation, and initial processing"""
    
    def __init__(self):
        self.transactions_df = None
        self.geographic_df = None
        self.weighting_df = None
        
    def load_transaction_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate transaction data from CSV file"""
        logger.info(f"Loading transaction data from {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=['transaction_date'])
        
        # Validate required columns
        required_cols = ['property_id', 'transaction_date', 'transaction_price', 
                        'census_tract_2010', 'cbsa_id']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Basic data validation
        df = df[df['transaction_price'] > 0]
        df = df.dropna(subset=required_cols)
        
        # Sort by property_id and transaction_date for repeat sales identification
        df = df.sort_values(['property_id', 'transaction_date'])
        
        self.transactions_df = df
        logger.info(f"Loaded {len(df)} transactions")
        return df
    
    def load_geographic_data(self, filepath: str) -> pd.DataFrame:
        """Load census tract geographic data"""
        logger.info(f"Loading geographic data from {filepath}")
        
        df = pd.read_csv(filepath)
        required_cols = ['census_tract_2010', 'centroid_lat', 'centroid_lon']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate coordinates
        df = df[(df['centroid_lat'] >= -90) & (df['centroid_lat'] <= 90)]
        df = df[(df['centroid_lon'] >= -180) & (df['centroid_lon'] <= 180)]
        
        self.geographic_df = df
        logger.info(f"Loaded geographic data for {len(df)} census tracts")
        return df
    
    def load_weighting_data(self, filepath: str) -> pd.DataFrame:
        """Load tract-level weighting data"""
        logger.info(f"Loading weighting data from {filepath}")
        
        df = pd.read_csv(filepath)
        required_cols = ['census_tract_2010', 'year']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.weighting_df = df
        logger.info(f"Loaded weighting data with {len(df)} records")
        return df


class RepeatSalesProcessor:
    """Identifies and filters repeat sales pairs"""
    
    def __init__(self, min_period_months: int = 12, 
                 max_annual_growth: float = 0.30,
                 max_appreciation_factor: float = 10.0,
                 min_appreciation_factor: float = 0.25):
        self.min_period_months = min_period_months
        self.max_annual_growth = max_annual_growth
        self.max_appreciation_factor = max_appreciation_factor
        self.min_appreciation_factor = min_appreciation_factor
    
    def identify_repeat_sales(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Identify all repeat sales pairs from transaction data"""
        logger.info("Identifying repeat sales pairs")
        
        # Group by property_id and find properties with multiple sales
        property_groups = transactions_df.groupby('property_id')
        
        repeat_sales = []
        for property_id, group in property_groups:
            if len(group) < 2:
                continue
            
            # Create all consecutive pairs
            sorted_group = group.sort_values('transaction_date')
            for i in range(len(sorted_group) - 1):
                first_sale = sorted_group.iloc[i]
                second_sale = sorted_group.iloc[i + 1]
                
                repeat_sales.append({
                    'property_id': property_id,
                    'first_sale_date': first_sale['transaction_date'],
                    'first_sale_price': first_sale['transaction_price'],
                    'second_sale_date': second_sale['transaction_date'],
                    'second_sale_price': second_sale['transaction_price'],
                    'census_tract_2010': first_sale['census_tract_2010'],
                    'cbsa_id': first_sale['cbsa_id']
                })
        
        repeat_sales_df = pd.DataFrame(repeat_sales)
        logger.info(f"Identified {len(repeat_sales_df)} repeat sales pairs")
        return repeat_sales_df
    
    def calculate_price_relatives(self, repeat_sales_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate log price relatives and growth metrics"""
        df = repeat_sales_df.copy()
        
        # Calculate log price relative
        df['log_price_relative'] = np.log(df['second_sale_price'] / df['first_sale_price'])
        
        # Calculate time between sales in years
        df['years_between_sales'] = (
            (df['second_sale_date'] - df['first_sale_date']).dt.days / 365.25
        )
        
        # Calculate compound annual growth rate
        df['annual_growth_rate'] = (
            (df['second_sale_price'] / df['first_sale_price']) ** 
            (1 / df['years_between_sales']) - 1
        )
        
        # Calculate cumulative appreciation
        df['cumulative_appreciation'] = (
            df['second_sale_price'] / df['first_sale_price']
        )
        
        return df
    
    def apply_filters(self, repeat_sales_df: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering rules to remove outliers and quality changes"""
        logger.info("Applying filters to repeat sales pairs")
        initial_count = len(repeat_sales_df)
        
        df = repeat_sales_df.copy()
        
        # Filter 1: Remove same-year transactions
        months_between = (
            (df['second_sale_date'] - df['first_sale_date']).dt.days / 30.44
        )
        df = df[months_between >= self.min_period_months]
        logger.info(f"After same-period filter: {len(df)} pairs")
        
        # Filter 2: Remove excessive annual growth
        df = df[abs(df['annual_growth_rate']) <= self.max_annual_growth]
        logger.info(f"After growth rate filter: {len(df)} pairs")
        
        # Filter 3: Remove extreme cumulative appreciation
        df = df[
            (df['cumulative_appreciation'] <= self.max_appreciation_factor) &
            (df['cumulative_appreciation'] >= self.min_appreciation_factor)
        ]
        logger.info(f"After appreciation filter: {len(df)} pairs")
        
        final_count = len(df)
        logger.info(f"Filtered {initial_count - final_count} pairs ({(initial_count - final_count) / initial_count * 100:.1f}%)")
        
        return df
    
    def process_repeat_sales(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Complete repeat sales processing pipeline"""
        # Identify repeat sales
        repeat_sales_df = self.identify_repeat_sales(transactions_df)
        
        # Calculate price relatives and metrics
        repeat_sales_df = self.calculate_price_relatives(repeat_sales_df)
        
        # Apply filters
        filtered_df = self.apply_filters(repeat_sales_df)
        
        return filtered_df