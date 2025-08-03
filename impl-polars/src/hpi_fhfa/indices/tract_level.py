"""Tract-level index construction."""

import polars as pl
from typing import Dict, List
import structlog

from ..config.constants import BASE_YEAR, BASE_INDEX_VALUE

logger = structlog.get_logger()


class TractLevelIndex:
    """Build tract-level house price indices."""
    
    def calculate_indices(
        self,
        regression_results: Dict[str, Dict[int, float]],
        all_supertracts: Dict[int, pl.DataFrame],
        start_year: int,
        end_year: int,
        base_year: int = BASE_YEAR
    ) -> pl.DataFrame:
        """Calculate tract-level indices from regression results.
        
        Creates a balanced panel where each tract has an index value
        for each year, based on its supertract's regression results.
        
        Args:
            regression_results: BMN regression results by supertract ID
            all_supertracts: Supertract mappings by period
            start_year: First year of indices
            end_year: Last year of indices
            base_year: Base year for normalization (default 1989)
            
        Returns:
            DataFrame with tract_id, year, index_value, appreciation_rate
        """
        logger.info(
            "Calculating tract-level indices",
            n_supertracts=len(regression_results),
            start_year=start_year,
            end_year=end_year
        )
        
        # Collect all unique tracts across all periods
        all_tracts = set()
        for period_mapping in all_supertracts.values():
            all_tracts.update(period_mapping["tract_id"].unique().to_list())
        
        logger.info(f"Building indices for {len(all_tracts)} tracts")
        
        # Build indices for each tract
        index_records = []
        
        for tract_id in all_tracts:
            # For each year, find which supertract this tract belongs to
            for year in range(start_year, end_year + 1):
                if year in all_supertracts:
                    # Find supertract for this tract in this year
                    year_mapping = all_supertracts[year]
                    supertract_match = year_mapping.filter(
                        pl.col("tract_id") == tract_id
                    )
                    
                    if len(supertract_match) > 0:
                        supertract_id = supertract_match["supertract_id"][0]
                        
                        # Get index value from regression results
                        if supertract_id in regression_results:
                            if year in regression_results[supertract_id]:
                                index_value = regression_results[supertract_id][year]
                                
                                # Calculate appreciation rate (year-over-year)
                                appreciation_rate = None
                                if year > start_year:
                                    prev_year = year - 1
                                    if prev_year in regression_results[supertract_id]:
                                        prev_value = regression_results[supertract_id][prev_year]
                                        if prev_value > 0:
                                            appreciation_rate = (
                                                (index_value / prev_value) - 1
                                            ) * 100  # Percentage
                                
                                index_records.append({
                                    "tract_id": tract_id,
                                    "year": year,
                                    "index_value": index_value,
                                    "appreciation_rate": appreciation_rate,
                                    "supertract_id": supertract_id,
                                    "n_component_tracts": supertract_match["n_component_tracts"][0]
                                })
        
        # Create DataFrame
        if not index_records:
            logger.warning("No index records created")
            return pl.DataFrame()
        
        indices_df = pl.DataFrame(index_records)
        
        # Ensure all tracts have all years (balanced panel)
        indices_df = self._create_balanced_panel(
            indices_df, all_tracts, start_year, end_year
        )
        
        # Normalize to base year if specified
        if base_year and base_year >= start_year and base_year <= end_year:
            indices_df = self._normalize_to_base_year(indices_df, base_year)
        
        # Sort for consistent output
        indices_df = indices_df.sort(["tract_id", "year"])
        
        # Calculate summary statistics
        self._log_summary_statistics(indices_df)
        
        return indices_df
    
    def _create_balanced_panel(
        self,
        indices_df: pl.DataFrame,
        all_tracts: set,
        start_year: int,
        end_year: int
    ) -> pl.DataFrame:
        """Ensure every tract has an entry for every year.
        
        For missing tract-year combinations, forward-fill from the
        most recent available value.
        """
        # Create all tract-year combinations
        all_combinations = []
        for tract_id in all_tracts:
            for year in range(start_year, end_year + 1):
                all_combinations.append({
                    "tract_id": tract_id,
                    "year": year
                })
        
        all_combinations_df = pl.DataFrame(all_combinations)
        
        # Join with actual indices
        balanced_df = all_combinations_df.join(
            indices_df,
            on=["tract_id", "year"],
            how="left"
        )
        
        # Forward-fill missing values within each tract
        balanced_df = (
            balanced_df
            .sort(["tract_id", "year"])
            .with_columns([
                pl.col("index_value").forward_fill().over("tract_id"),
                pl.col("supertract_id").forward_fill().over("tract_id"),
                pl.col("n_component_tracts").forward_fill().over("tract_id")
            ])
        )
        
        # Handle tracts with no data at all (use overall average)
        if balanced_df["index_value"].null_count() > 0:
            # Calculate average index by year
            year_averages = (
                balanced_df
                .filter(pl.col("index_value").is_not_null())
                .group_by("year")
                .agg(pl.col("index_value").mean().alias("avg_index"))
            )
            
            # Fill remaining nulls with year average
            balanced_df = balanced_df.join(
                year_averages,
                on="year",
                how="left"
            ).with_columns(
                pl.when(pl.col("index_value").is_null())
                .then(pl.col("avg_index"))
                .otherwise(pl.col("index_value"))
                .alias("index_value")
            ).drop("avg_index")
        
        # Recalculate appreciation rates for filled values
        balanced_df = balanced_df.with_columns([
            pl.when(pl.col("year") > start_year)
            .then(
                ((pl.col("index_value") / pl.col("index_value").shift(1).over("tract_id")) - 1) * 100
            )
            .otherwise(None)
            .alias("appreciation_rate")
        ])
        
        return balanced_df
    
    def _normalize_to_base_year(
        self,
        indices_df: pl.DataFrame,
        base_year: int
    ) -> pl.DataFrame:
        """Normalize indices so base year = 100."""
        # Get base year values for each tract
        base_values = (
            indices_df
            .filter(pl.col("year") == base_year)
            .select(["tract_id", "index_value"])
            .rename({"index_value": "base_value"})
        )
        
        # Join and normalize
        normalized_df = (
            indices_df
            .join(base_values, on="tract_id", how="left")
            .with_columns(
                (pl.col("index_value") / pl.col("base_value") * BASE_INDEX_VALUE)
                .alias("index_value")
            )
            .drop("base_value")
        )
        
        return normalized_df
    
    def _log_summary_statistics(self, indices_df: pl.DataFrame) -> None:
        """Log summary statistics about the indices."""
        # Overall statistics
        n_tracts = indices_df["tract_id"].n_unique()
        n_years = indices_df["year"].n_unique()
        
        # Appreciation statistics
        appreciation_stats = indices_df.filter(
            pl.col("appreciation_rate").is_not_null()
        )["appreciation_rate"]
        
        if len(appreciation_stats) > 0:
            logger.info(
                "Tract index summary",
                n_tracts=n_tracts,
                n_years=n_years,
                n_observations=len(indices_df),
                avg_appreciation=f"{appreciation_stats.mean():.2f}%",
                median_appreciation=f"{appreciation_stats.median():.2f}%",
                min_appreciation=f"{appreciation_stats.min():.2f}%",
                max_appreciation=f"{appreciation_stats.max():.2f}%"
            )
        else:
            logger.info(
                "Tract index summary",
                n_tracts=n_tracts,
                n_years=n_years,
                n_observations=len(indices_df)
            )