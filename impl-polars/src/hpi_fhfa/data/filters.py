"""Transaction filtering logic for HPI-FHFA."""

import polars as pl
import numpy as np
from typing import Tuple
import structlog

from ..config.constants import (
    MAX_CAGR_THRESHOLD,
    MAX_CUMULATIVE_APPRECIATION,
    MIN_CUMULATIVE_APPRECIATION
)

logger = structlog.get_logger()


class TransactionFilter:
    """Apply filters to repeat-sales transactions."""
    
    def __init__(self):
        """Initialize transaction filter."""
        self.filters_applied = []
        
    def apply_filters(self, repeat_sales_df: pl.DataFrame) -> pl.DataFrame:
        """Apply all filters to repeat-sales data.
        
        According to PRD, remove pairs where:
        1. Same 12-month transaction period
        2. Compound annual growth rate > |30%|
        3. Cumulative appreciation > 10x or < 0.25x previous sale
        
        Args:
            repeat_sales_df: DataFrame with repeat sales
            
        Returns:
            Filtered DataFrame
        """
        initial_count = len(repeat_sales_df)
        logger.info("Applying transaction filters", initial_count=initial_count)
        
        # Apply each filter and track results
        df = self._filter_same_period(repeat_sales_df)
        df = self._filter_cagr(df)
        df = self._filter_cumulative_appreciation(df)
        
        final_count = len(df)
        removed_count = initial_count - final_count
        removal_pct = (removed_count / initial_count * 100) if initial_count > 0 else 0
        
        logger.info(
            "Filtering complete",
            initial_count=initial_count,
            final_count=final_count,
            removed_count=removed_count,
            removal_pct=f"{removal_pct:.2f}%",
            filters_applied=self.filters_applied
        )
        
        return df
    
    def _filter_same_period(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove transactions in same 12-month period.
        
        Args:
            df: DataFrame with repeat sales
            
        Returns:
            Filtered DataFrame
        """
        before_count = len(df)
        
        # Extract year from dates and filter
        df = df.with_columns([
            pl.col("transaction_date").dt.year().alias("sale_year"),
            pl.col("prev_transaction_date").dt.year().alias("prev_sale_year")
        ]).filter(
            pl.col("sale_year") != pl.col("prev_sale_year")
        ).drop(["sale_year", "prev_sale_year"])
        
        after_count = len(df)
        removed = before_count - after_count
        
        self.filters_applied.append({
            "filter": "same_period",
            "removed": removed,
            "pct": f"{removed / before_count * 100:.2f}%" if before_count > 0 else "0%"
        })
        
        logger.debug(f"Same period filter removed {removed} transactions")
        return df
    
    def _filter_cagr(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove transactions with CAGR > |30%|.
        
        CAGR formula: |(V1/V0)^(1/(t1-t0)) - 1| > 0.30
        
        Args:
            df: DataFrame with repeat sales
            
        Returns:
            Filtered DataFrame
        """
        before_count = len(df)
        
        # Calculate CAGR if not already present
        if "cagr" not in df.columns:
            df = df.with_columns([
                (
                    ((pl.col("transaction_price") / pl.col("prev_transaction_price"))
                     .pow(1.0 / pl.col("time_diff_years")) - 1)
                    .abs()
                ).alias("cagr")
            ])
        
        # Filter by CAGR threshold
        df = df.filter(pl.col("cagr") <= MAX_CAGR_THRESHOLD)
        
        after_count = len(df)
        removed = before_count - after_count
        
        self.filters_applied.append({
            "filter": "cagr",
            "removed": removed,
            "pct": f"{removed / before_count * 100:.2f}%" if before_count > 0 else "0%"
        })
        
        logger.debug(f"CAGR filter removed {removed} transactions")
        return df
    
    def _filter_cumulative_appreciation(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove transactions with extreme cumulative appreciation.
        
        Remove if cumulative appreciation > 10x or < 0.25x previous sale.
        
        Args:
            df: DataFrame with repeat sales
            
        Returns:
            Filtered DataFrame
        """
        before_count = len(df)
        
        # Calculate price ratio
        df = df.with_columns([
            (pl.col("transaction_price") / pl.col("prev_transaction_price")).alias("price_ratio")
        ])
        
        # Filter by cumulative appreciation bounds
        df = df.filter(
            (pl.col("price_ratio") >= MIN_CUMULATIVE_APPRECIATION) &
            (pl.col("price_ratio") <= MAX_CUMULATIVE_APPRECIATION)
        ).drop("price_ratio")
        
        after_count = len(df)
        removed = before_count - after_count
        
        self.filters_applied.append({
            "filter": "cumulative_appreciation", 
            "removed": removed,
            "pct": f"{removed / before_count * 100:.2f}%" if before_count > 0 else "0%"
        })
        
        logger.debug(f"Cumulative appreciation filter removed {removed} transactions")
        return df
    
    def get_filter_summary(self) -> pl.DataFrame:
        """Get summary of applied filters.
        
        Returns:
            DataFrame with filter statistics
        """
        if not self.filters_applied:
            return pl.DataFrame()
        
        return pl.DataFrame(self.filters_applied)