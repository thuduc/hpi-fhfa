"""Repeat-sales pair identification and processing."""

import polars as pl
from typing import Tuple, Optional
import structlog

from ..data.schemas import REPEAT_SALES_SCHEMA

logger = structlog.get_logger()


class RepeatSalesIdentifier:
    """Identify and process repeat sales from transaction data."""
    
    def __init__(self):
        """Initialize repeat sales identifier."""
        self.stats = {}
    
    def identify_repeat_sales(
        self, 
        transactions: pl.DataFrame,
        min_days_between_sales: int = 365
    ) -> pl.DataFrame:
        """Identify repeat sales from transaction data.
        
        Uses Polars window functions to efficiently identify properties
        with multiple transactions and create repeat-sales pairs.
        
        Args:
            transactions: DataFrame with transaction data
            min_days_between_sales: Minimum days between sales (default 365)
            
        Returns:
            DataFrame with repeat sales pairs
        """
        logger.info("Identifying repeat sales", n_transactions=len(transactions))
        
        # Sort by property and date
        sorted_df = transactions.sort(["property_id", "transaction_date"])
        
        # Add previous transaction information using window functions
        repeat_sales = sorted_df.with_columns([
            pl.col("transaction_date").shift(1).over("property_id").alias("prev_transaction_date"),
            pl.col("transaction_price").shift(1).over("property_id").alias("prev_transaction_price"),
        ])
        
        # Filter to only repeat sales (where previous transaction exists)
        repeat_sales = repeat_sales.filter(pl.col("prev_transaction_date").is_not_null())
        
        # Add time difference
        repeat_sales = repeat_sales.with_columns([
            (pl.col("transaction_date") - pl.col("prev_transaction_date")).dt.total_days().alias("days_between_sales")
        ])
        
        # Filter by minimum days between sales
        if min_days_between_sales > 0:
            initial_count = len(repeat_sales)
            repeat_sales = repeat_sales.filter(
                pl.col("days_between_sales") >= min_days_between_sales
            )
            filtered_count = initial_count - len(repeat_sales)
            logger.debug(f"Filtered {filtered_count} sales with < {min_days_between_sales} days between")
        
        # Calculate derived fields
        repeat_sales = self._add_derived_fields(repeat_sales)
        
        # Store statistics
        self._calculate_statistics(transactions, repeat_sales)
        
        logger.info(
            "Repeat sales identification complete",
            n_repeat_sales=len(repeat_sales),
            n_unique_properties=repeat_sales["property_id"].n_unique()
        )
        
        return repeat_sales
    
    def _add_derived_fields(self, repeat_sales: pl.DataFrame) -> pl.DataFrame:
        """Add derived fields for analysis.
        
        Adds:
        - log_price_diff: Log price difference (p_itτ)
        - time_diff_years: Time between sales in years
        - cagr: Compound annual growth rate
        """
        return repeat_sales.with_columns([
            # Log price difference
            (pl.col("transaction_price").log() - pl.col("prev_transaction_price").log())
            .alias("log_price_diff"),
            
            # Time difference in years
            (pl.col("days_between_sales") / 365.25).alias("time_diff_years"),
        ]).with_columns([
            # CAGR: |(V1/V0)^(1/(t1-t0)) - 1|
            (
                ((pl.col("transaction_price") / pl.col("prev_transaction_price"))
                 .pow(1.0 / pl.col("time_diff_years")) - 1)
                .abs()
            ).alias("cagr")
        ])
    
    def _calculate_statistics(
        self, 
        transactions: pl.DataFrame, 
        repeat_sales: pl.DataFrame
    ) -> None:
        """Calculate and store repeat sales statistics."""
        n_transactions = len(transactions)
        n_properties = transactions["property_id"].n_unique()
        n_repeat_sales = len(repeat_sales)
        n_repeat_properties = repeat_sales["property_id"].n_unique()
        
        self.stats = {
            "n_transactions": n_transactions,
            "n_properties": n_properties,
            "n_repeat_sales": n_repeat_sales,
            "n_repeat_properties": n_repeat_properties,
            "repeat_sales_pct": n_repeat_sales / n_transactions * 100 if n_transactions > 0 else 0,
            "properties_with_repeats_pct": n_repeat_properties / n_properties * 100 if n_properties > 0 else 0,
            "avg_time_between_sales_years": repeat_sales["time_diff_years"].mean() if n_repeat_sales > 0 else 0,
            "median_time_between_sales_years": repeat_sales["time_diff_years"].median() if n_repeat_sales > 0 else 0,
            "avg_price_appreciation": ((repeat_sales["transaction_price"] / repeat_sales["prev_transaction_price"]).mean() - 1) if n_repeat_sales > 0 else 0,
            "median_cagr": repeat_sales["cagr"].median() if n_repeat_sales > 0 else 0
        }
    
    def get_statistics(self) -> dict:
        """Get repeat sales statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.stats
    
    def create_balanced_panel(
        self,
        repeat_sales: pl.DataFrame,
        start_period: int,
        end_period: int
    ) -> pl.DataFrame:
        """Create a balanced panel of repeat sales.
        
        Ensures all period pairs are represented, even with zero observations.
        
        Args:
            repeat_sales: DataFrame with repeat sales
            start_period: First period (year)
            end_period: Last period (year)
            
        Returns:
            Balanced panel DataFrame
        """
        # Extract periods
        repeat_sales = repeat_sales.with_columns([
            pl.col("transaction_date").dt.year().alias("sale_period"),
            pl.col("prev_transaction_date").dt.year().alias("prev_period")
        ])
        
        # Count observations by period pair
        period_counts = (
            repeat_sales
            .group_by(["prev_period", "sale_period"])
            .agg(pl.len().alias("n_observations"))
        )
        
        # Create all possible period pairs
        all_periods = list(range(start_period, end_period + 1))
        all_pairs = []
        for prev in all_periods:
            for curr in all_periods:
                if curr > prev:  # Only forward-looking pairs
                    all_pairs.append({"prev_period": prev, "sale_period": curr})
        
        all_pairs_df = pl.DataFrame(all_pairs)
        
        # Create balanced panel
        balanced_panel = (
            all_pairs_df
            .join(period_counts, on=["prev_period", "sale_period"], how="left")
            .with_columns(pl.col("n_observations").fill_null(0))
        )
        
        return balanced_panel