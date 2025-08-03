"""Half-pairs calculation for supertract formation."""

import polars as pl
from typing import Dict, List, Tuple
import structlog

logger = structlog.get_logger()


class HalfPairsCalculator:
    """Calculate half-pairs for census tracts.
    
    Half-pairs represent the number of repeat-sales observations
    involving a tract in a given period. For a property with
    transactions at times [t1, t2, t3]:
    - Half-pairs at t1: 1 (pair with t2)
    - Half-pairs at t2: 2 (pairs with t1 and t3)
    - Half-pairs at t3: 1 (pair with t2)
    """
    
    def calculate_half_pairs(
        self,
        repeat_sales: pl.DataFrame,
        periods: List[int]
    ) -> pl.DataFrame:
        """Calculate half-pairs by tract and period.
        
        Args:
            repeat_sales: DataFrame with repeat sales
                Must contain: census_tract, transaction_date, prev_transaction_date
            periods: List of periods (years) to calculate for
            
        Returns:
            DataFrame with tract_id, period, half_pairs_count
        """
        logger.info("Calculating half-pairs", n_periods=len(periods))
        
        # Extract periods from dates
        repeat_sales = repeat_sales.with_columns([
            pl.col("transaction_date").dt.year().alias("sale_period"),
            pl.col("prev_transaction_date").dt.year().alias("prev_period")
        ])
        
        # Calculate half-pairs for each period
        half_pairs_list = []
        
        for period in periods:
            # Count sales in current period (as second transaction)
            current_sales = (
                repeat_sales
                .filter(pl.col("sale_period") == period)
                .group_by("census_tract")
                .agg(pl.len().alias("current_count"))
            )
            
            # Count sales in current period (as first transaction)
            previous_sales = (
                repeat_sales
                .filter(pl.col("prev_period") == period)
                .group_by("census_tract")
                .agg(pl.len().alias("previous_count"))
            )
            
            # Combine counts
            period_half_pairs = (
                current_sales
                .join(previous_sales, on="census_tract", how="full")
                .with_columns([
                    pl.col("current_count").fill_null(0),
                    pl.col("previous_count").fill_null(0),
                    pl.lit(period).alias("period")
                ])
                .with_columns(
                    (pl.col("current_count") + pl.col("previous_count")).alias("half_pairs_count")
                )
                .select([
                    pl.col("census_tract").alias("tract_id"),
                    "period",
                    "half_pairs_count"
                ])
            )
            
            half_pairs_list.append(period_half_pairs)
        
        # Combine all periods
        half_pairs_df = pl.concat(half_pairs_list)
        
        # Add zero counts for missing tract-period combinations
        half_pairs_df = self._fill_missing_periods(half_pairs_df, repeat_sales, periods)
        
        logger.info(
            "Half-pairs calculation complete",
            n_tract_periods=len(half_pairs_df),
            total_half_pairs=half_pairs_df["half_pairs_count"].sum()
        )
        
        return half_pairs_df
    
    def _fill_missing_periods(
        self,
        half_pairs_df: pl.DataFrame,
        repeat_sales: pl.DataFrame,
        periods: List[int]
    ) -> pl.DataFrame:
        """Fill in zero counts for tract-period combinations with no data."""
        # Get all unique tracts
        all_tracts = repeat_sales["census_tract"].unique().to_list()
        
        # Create all tract-period combinations
        all_combinations = []
        for tract in all_tracts:
            for period in periods:
                all_combinations.append({
                    "tract_id": tract,
                    "period": period
                })
        
        all_combinations_df = pl.DataFrame(all_combinations)
        
        # Join with actual data and fill nulls
        complete_df = (
            all_combinations_df
            .join(half_pairs_df, on=["tract_id", "period"], how="left")
            .with_columns(pl.col("half_pairs_count").fill_null(0))
        )
        
        return complete_df
    
    def get_tract_summary(self, half_pairs_df: pl.DataFrame) -> pl.DataFrame:
        """Get summary statistics by tract.
        
        Args:
            half_pairs_df: DataFrame with half-pairs counts
            
        Returns:
            Summary DataFrame by tract
        """
        return (
            half_pairs_df
            .group_by("tract_id")
            .agg([
                pl.col("half_pairs_count").sum().alias("total_half_pairs"),
                pl.col("half_pairs_count").mean().alias("avg_half_pairs_per_period"),
                pl.col("half_pairs_count").min().alias("min_half_pairs"),
                pl.col("half_pairs_count").max().alias("max_half_pairs"),
                (pl.col("half_pairs_count") > 0).sum().alias("n_periods_with_data")
            ])
            .sort("total_half_pairs", descending=True)
        )
    
    def identify_sparse_tracts(
        self,
        half_pairs_df: pl.DataFrame,
        min_threshold: int = 40
    ) -> pl.DataFrame:
        """Identify tracts with insufficient half-pairs.
        
        Args:
            half_pairs_df: DataFrame with half-pairs counts
            min_threshold: Minimum half-pairs threshold
            
        Returns:
            DataFrame of sparse tract-period combinations
        """
        return (
            half_pairs_df
            .filter(pl.col("half_pairs_count") < min_threshold)
            .sort(["period", "half_pairs_count"])
        )