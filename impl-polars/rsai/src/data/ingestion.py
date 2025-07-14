"""
Data ingestion and repeat sales processing using Polars.

This module handles loading property transaction data, identifying repeat sales,
and preparing data for index calculation using Polars DataFrames.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import datetime, date

import polars as pl
import numpy as np
from pydantic import ValidationError

from rsai.src.data.models import (
    Transaction,
    RepeatSalePair,
    GeographicLocation,
    PropertyCharacteristics,
    TransactionType,
    RSAIConfig
)

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data ingestion and repeat sales identification using Polars."""
    
    def __init__(self, config: RSAIConfig):
        """
        Initialize data ingestion with configuration.
        
        Args:
            config: RSAI configuration object
        """
        self.config = config
        self.transactions_df: Optional[pl.DataFrame] = None
        self.properties_df: Optional[pl.DataFrame] = None
        self.repeat_sales_df: Optional[pl.DataFrame] = None
        
    def load_transactions(self, file_path: Union[str, Path]) -> pl.DataFrame:
        """
        Load transaction data from file using Polars.
        
        Args:
            file_path: Path to transaction data file
            
        Returns:
            Polars DataFrame with transaction data
        """
        file_path = Path(file_path)
        logger.info(f"Loading transactions from {file_path}")
        
        # Determine file format and load
        if file_path.suffix == '.parquet':
            df = pl.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pl.read_csv(file_path, try_parse_dates=True)
        elif file_path.suffix == '.feather':
            df = pl.read_ipc(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Convert date columns
        date_cols = [pl.col("sale_date").cast(pl.Date)]
        if "recording_date" in df.columns:
            date_cols.append(pl.col("recording_date").cast(pl.Date).fill_null(pl.col("sale_date")))
        else:
            # Add recording_date as same as sale_date if missing
            date_cols.append(pl.col("sale_date").alias("recording_date"))
        
        df = df.with_columns(date_cols)
        
        # Filter by date range
        df = df.filter(
            (pl.col("sale_date") >= self.config.start_date) &
            (pl.col("sale_date") <= self.config.end_date)
        )
        
        # Filter by price range
        df = df.filter(
            (pl.col("sale_price") >= self.config.min_price) &
            (pl.col("sale_price") <= self.config.max_price)
        )
        
        logger.info(f"Loaded {len(df)} transactions")
        self.transactions_df = df
        return df
    
    def load_properties(self, file_path: Union[str, Path]) -> pl.DataFrame:
        """
        Load property characteristics and geographic data.
        
        Args:
            file_path: Path to property data file
            
        Returns:
            Polars DataFrame with property data
        """
        file_path = Path(file_path)
        logger.info(f"Loading properties from {file_path}")
        
        if file_path.suffix == '.parquet':
            df = pl.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pl.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Standardize column names
        df = self._standardize_property_columns(df)
        
        logger.info(f"Loaded {len(df)} properties")
        self.properties_df = df
        return df
    
    def identify_repeat_sales(self) -> pl.DataFrame:
        """
        Identify repeat sales pairs from transaction data.
        
        Returns:
            Polars DataFrame with repeat sale pairs
        """
        if self.transactions_df is None:
            raise ValueError("No transactions loaded")
        
        logger.info("Identifying repeat sales")
        
        # Sort transactions by property and date
        sorted_df = self.transactions_df.sort(["property_id", "sale_date"])
        
        # Create lagged columns for previous sale
        repeat_df = sorted_df.with_columns([
            pl.col("transaction_id").shift(1).over("property_id").alias("prev_transaction_id"),
            pl.col("sale_price").shift(1).over("property_id").alias("prev_sale_price"),
            pl.col("sale_date").shift(1).over("property_id").alias("prev_sale_date"),
            pl.col("transaction_type").shift(1).over("property_id").alias("prev_transaction_type")
        ])
        
        # Filter to only repeat sales (where previous sale exists)
        repeat_df = repeat_df.filter(pl.col("prev_transaction_id").is_not_null())
        
        # Filter to arms-length transactions only
        if self.config.weighting_scheme != "CUSTOM":
            repeat_df = repeat_df.filter(
                (pl.col("transaction_type") == TransactionType.ARMS_LENGTH.value) &
                (pl.col("prev_transaction_type") == TransactionType.ARMS_LENGTH.value)
            )
        
        # Calculate holding period
        repeat_df = repeat_df.with_columns([
            ((pl.col("sale_date") - pl.col("prev_sale_date")).dt.total_milliseconds() / (24 * 60 * 60 * 1000)).cast(pl.Int32).alias("holding_period_days")
        ])
        
        # Filter by maximum holding period
        max_days = self.config.max_holding_period_years * 365
        repeat_df = repeat_df.filter(
            (pl.col("holding_period_days") > 0) &
            (pl.col("holding_period_days") <= max_days)
        )
        
        # Calculate price ratios and returns
        repeat_df = repeat_df.with_columns([
            (pl.col("sale_price") / pl.col("prev_sale_price")).alias("price_ratio"),
            (pl.col("sale_price") / pl.col("prev_sale_price")).log().alias("log_price_ratio"),
            pl.concat_str([
                pl.col("property_id"),
                pl.lit("_"),
                pl.col("prev_transaction_id"),
                pl.lit("_"),
                pl.col("transaction_id")
            ]).alias("pair_id")
        ])
        
        # Calculate annualized return
        repeat_df = repeat_df.with_columns([
            ((pl.col("price_ratio") ** (365.0 / pl.col("holding_period_days"))) - 1).alias("annualized_return")
        ])
        
        # Rename columns to match RepeatSalePair model
        repeat_df = repeat_df.select([
            pl.col("pair_id"),
            pl.col("property_id"),
            pl.col("prev_transaction_id").alias("sale1_transaction_id"),
            pl.col("prev_sale_price").alias("sale1_price"),
            pl.col("prev_sale_date").alias("sale1_date"),
            pl.col("transaction_id").alias("sale2_transaction_id"),
            pl.col("sale_price").alias("sale2_price"),
            pl.col("sale_date").alias("sale2_date"),
            pl.col("price_ratio"),
            pl.col("log_price_ratio"),
            pl.col("holding_period_days"),
            pl.col("annualized_return"),
            pl.lit(True).alias("is_valid"),
            pl.lit([]).alias("validation_flags")
        ])
        
        logger.info(f"Identified {len(repeat_df)} repeat sale pairs")
        self.repeat_sales_df = repeat_df
        return repeat_df
    
    def merge_geographic_data(self) -> pl.DataFrame:
        """
        Merge geographic data with repeat sales.
        
        Returns:
            Polars DataFrame with geographic information added
        """
        if self.repeat_sales_df is None:
            raise ValueError("No repeat sales identified")
        if self.properties_df is None:
            logger.warning("No property data loaded, skipping geographic merge")
            return self.repeat_sales_df
        
        logger.info("Merging geographic data")
        
        # Select relevant geographic columns
        geo_cols = ["property_id", "latitude", "longitude", "address", 
                   "zip_code", "tract", "block", "county_fips", "msa_code", "state"]
        
        # Filter to existing columns
        available_cols = [col for col in geo_cols if col in self.properties_df.columns]
        geo_df = self.properties_df.select(available_cols)
        
        # Merge with repeat sales
        merged_df = self.repeat_sales_df.join(
            geo_df,
            on="property_id",
            how="left"
        )
        
        self.repeat_sales_df = merged_df
        return merged_df
    
    def filter_outliers(self, method: str = "iqr") -> pl.DataFrame:
        """
        Filter outlier repeat sales based on price ratios.
        
        Args:
            method: Outlier detection method ('iqr', 'zscore', or 'percentile')
            
        Returns:
            Filtered Polars DataFrame
        """
        if self.repeat_sales_df is None:
            raise ValueError("No repeat sales identified")
        
        logger.info(f"Filtering outliers using {method} method")
        df = self.repeat_sales_df
        
        if method == "iqr":
            # Calculate IQR bounds
            q1 = df["log_price_ratio"].quantile(0.25)
            q3 = df["log_price_ratio"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter outliers
            df = df.filter(
                (pl.col("log_price_ratio") >= lower_bound) &
                (pl.col("log_price_ratio") <= upper_bound)
            )
            
        elif method == "zscore":
            # Calculate z-scores
            mean = df["log_price_ratio"].mean()
            std = df["log_price_ratio"].std()
            
            df = df.with_columns([
                ((pl.col("log_price_ratio") - mean) / std).abs().alias("zscore")
            ])
            
            # Filter by z-score threshold
            df = df.filter(pl.col("zscore") <= self.config.outlier_std_threshold)
            df = df.drop("zscore")
            
        elif method == "percentile":
            # Filter by percentiles
            lower = df["log_price_ratio"].quantile(0.01)
            upper = df["log_price_ratio"].quantile(0.99)
            
            df = df.filter(
                (pl.col("log_price_ratio") >= lower) &
                (pl.col("log_price_ratio") <= upper)
            )
        
        logger.info(f"Retained {len(df)} pairs after outlier filtering")
        self.repeat_sales_df = df
        return df
    
    def _standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize transaction column names."""
        column_mapping = {
            # Common variations
            "SalePrice": "sale_price",
            "sale_amount": "sale_price",
            "price": "sale_price",
            "SaleDate": "sale_date",
            "sale_dt": "sale_date",
            "date": "sale_date",
            "PropertyID": "property_id",
            "property_id": "property_id",
            "parcel_id": "property_id",
            "TransactionID": "transaction_id",
            "trans_id": "transaction_id",
            "RecordingDate": "recording_date",
            "record_date": "recording_date",
            "TransactionType": "transaction_type",
            "trans_type": "transaction_type",
            "sale_type": "transaction_type"
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename({old_name: new_name})
        
        # Add transaction_id if missing
        if "transaction_id" not in df.columns:
            df = df.with_row_count("transaction_id")
            df = df.with_columns(pl.col("transaction_id").cast(pl.Utf8))
        
        # Add transaction_type if missing
        if "transaction_type" not in df.columns:
            df = df.with_columns(pl.lit(TransactionType.ARMS_LENGTH.value).alias("transaction_type"))
        
        return df
    
    def _standardize_property_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize property column names."""
        column_mapping = {
            # Property identifiers
            "PropertyID": "property_id",
            "parcel_id": "property_id",
            "ParcelID": "property_id",
            
            # Geographic columns
            "Latitude": "latitude",
            "lat": "latitude",
            "Longitude": "longitude",
            "lon": "longitude",
            "lng": "longitude",
            "Address": "address",
            "street_address": "address",
            "ZipCode": "zip_code",
            "zip": "zip_code",
            "postal_code": "zip_code",
            "Tract": "tract",
            "census_tract": "tract",
            "Block": "block",
            "census_block": "block",
            "County": "county_fips",
            "county_code": "county_fips",
            "MSA": "msa_code",
            "msa": "msa_code",
            "State": "state",
            "state_code": "state",
            
            # Property characteristics
            "LivingArea": "living_area",
            "sqft": "living_area",
            "square_feet": "living_area",
            "Bedrooms": "bedrooms",
            "beds": "bedrooms",
            "Bathrooms": "bathrooms",
            "baths": "bathrooms",
            "YearBuilt": "year_built",
            "year_built": "year_built",
            "LotSize": "lot_size",
            "lot_sqft": "lot_size",
            "PropertyType": "property_type",
            "prop_type": "property_type"
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename({old_name: new_name})
        
        return df
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for the loaded data.
        
        Returns:
            Dictionary with summary statistics
        """
        stats = {}
        
        if self.transactions_df is not None:
            stats["total_transactions"] = len(self.transactions_df)
            stats["unique_properties"] = self.transactions_df["property_id"].n_unique()
            stats["date_range"] = {
                "min": self.transactions_df["sale_date"].min(),
                "max": self.transactions_df["sale_date"].max()
            }
            stats["price_range"] = {
                "min": self.transactions_df["sale_price"].min(),
                "max": self.transactions_df["sale_price"].max(),
                "median": self.transactions_df["sale_price"].median()
            }
        
        if self.repeat_sales_df is not None:
            stats["total_repeat_pairs"] = len(self.repeat_sales_df)
            stats["repeat_properties"] = self.repeat_sales_df["property_id"].n_unique()
            stats["holding_period"] = {
                "min_days": self.repeat_sales_df["holding_period_days"].min(),
                "max_days": self.repeat_sales_df["holding_period_days"].max(),
                "median_days": self.repeat_sales_df["holding_period_days"].median()
            }
            stats["price_ratio"] = {
                "min": self.repeat_sales_df["price_ratio"].min(),
                "max": self.repeat_sales_df["price_ratio"].max(),
                "median": self.repeat_sales_df["price_ratio"].median()
            }
        
        return stats