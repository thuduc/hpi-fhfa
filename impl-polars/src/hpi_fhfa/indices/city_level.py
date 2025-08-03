"""City-level index construction with multiple weighting schemes."""

import polars as pl
import numpy as np
from typing import Dict, List, Optional
import structlog

from ..models.weighting import WeightingFactory
from ..config.constants import BASE_INDEX_VALUE

logger = structlog.get_logger()


class CityLevelIndex:
    """Build city-level (CBSA) house price indices."""
    
    def __init__(self, weight_schemes: List[str]):
        """Initialize with list of weighting schemes to calculate.
        
        Args:
            weight_schemes: List of weight scheme names
        """
        self.weight_schemes = weight_schemes
        
    def calculate_all_indices(
        self,
        regression_results: Dict[str, Dict[int, float]],
        all_supertracts: Dict[int, pl.DataFrame],
        geographic_df: pl.DataFrame,
        transaction_df: Optional[pl.DataFrame],
        start_year: int,
        end_year: int
    ) -> Dict[str, pl.DataFrame]:
        """Calculate city-level indices for all weight schemes.
        
        Args:
            regression_results: BMN regression results by supertract
            all_supertracts: Supertract mappings by period
            geographic_df: Geographic data with CBSA codes
            transaction_df: Transaction data (for dynamic weights)
            start_year: First year
            end_year: Last year
            
        Returns:
            Dictionary mapping weight scheme to index DataFrame
        """
        logger.info(
            "Calculating city-level indices",
            n_weight_schemes=len(self.weight_schemes),
            start_year=start_year,
            end_year=end_year
        )
        
        # Get all CBSAs
        cbsa_codes = geographic_df["cbsa_code"].unique().to_list()
        logger.info(f"Building indices for {len(cbsa_codes)} CBSAs")
        
        # Calculate indices for each weight scheme
        all_indices = {}
        
        for weight_scheme in self.weight_schemes:
            try:
                indices = self._calculate_index_for_scheme(
                    weight_scheme,
                    regression_results,
                    all_supertracts,
                    geographic_df,
                    transaction_df,
                    cbsa_codes,
                    start_year,
                    end_year
                )
                all_indices[weight_scheme] = indices
                
                logger.info(
                    f"Calculated {weight_scheme} indices",
                    n_observations=len(indices)
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to calculate {weight_scheme} indices",
                    error=str(e)
                )
                raise
        
        return all_indices
    
    def _calculate_index_for_scheme(
        self,
        weight_scheme: str,
        regression_results: Dict[str, Dict[int, float]],
        all_supertracts: Dict[int, pl.DataFrame],
        geographic_df: pl.DataFrame,
        transaction_df: Optional[pl.DataFrame],
        cbsa_codes: List[str],
        start_year: int,
        end_year: int
    ) -> pl.DataFrame:
        """Calculate city indices for a specific weighting scheme.
        
        Following Algorithm from PRD Section 3.6.1:
        1. Initialize P_a(t=0) = 1 (or 100)
        2. For each period t:
           - Construct supertracts
           - Calculate BMN indices for each supertract  
           - Calculate weights
           - Aggregate: p̂_a(t) = Σ w_n * (δ̂_n,t - δ̂_n,t-1)
           - Update index: P̂_a(t) = P̂_a(t-1) * exp(p̂_a(t))
        """
        # Create weighting scheme instance
        weighter = WeightingFactory.create(weight_scheme)
        
        index_records = []
        
        for cbsa_code in cbsa_codes:
            # Get tracts in this CBSA
            cbsa_tracts = geographic_df.filter(
                pl.col("cbsa_code") == cbsa_code
            )["tract_id"].to_list()
            
            # Initialize index
            cbsa_indices = {start_year - 1: BASE_INDEX_VALUE}
            
            for year in range(start_year, end_year + 1):
                if year not in all_supertracts:
                    continue
                
                # Get supertracts for this CBSA in this year
                year_supertracts = all_supertracts[year].filter(
                    pl.col("tract_id").is_in(cbsa_tracts)
                )
                
                if len(year_supertracts) == 0:
                    continue
                
                # Get unique supertracts
                unique_supertracts = year_supertracts["supertract_id"].unique().to_list()
                
                # Prepare data for weight calculation
                # Add tract-level data from geographic_df
                supertract_data = (
                    year_supertracts
                    .join(geographic_df, left_on="tract_id", right_on="tract_id")
                    .filter(pl.col("cbsa_code") == cbsa_code)
                )
                
                # Calculate weights
                weights_df = weighter.calculate_weights(
                    supertract_data,
                    year,
                    geographic_df,
                    transaction_df
                )
                
                # Calculate weighted appreciation
                weighted_appreciation = 0.0
                total_weight = 0.0
                
                for row in weights_df.iter_rows(named=True):
                    supertract_id = row["supertract_id"]
                    weight = row["weight"]
                    
                    if supertract_id in regression_results:
                        reg_results = regression_results[supertract_id]
                        if year in reg_results and (year - 1) in reg_results:
                            # Calculate log appreciation (δ_t - δ_{t-1})
                            # Note: regression results are already index values, 
                            # so we need to convert back to log space
                            delta_t = np.log(reg_results[year] / 100.0)
                            delta_t_1 = np.log(reg_results[year - 1] / 100.0)
                            appreciation = delta_t - delta_t_1
                            
                            weighted_appreciation += weight * appreciation
                            total_weight += weight
                
                # Calculate new index value
                if total_weight > 0:
                    # P̂_a(t) = P̂_a(t-1) * exp(p̂_a(t))
                    prev_index = cbsa_indices.get(year - 1, BASE_INDEX_VALUE)
                    new_index = prev_index * np.exp(weighted_appreciation)
                    cbsa_indices[year] = new_index
                    
                    # Calculate year-over-year appreciation rate
                    appreciation_rate = ((new_index / prev_index) - 1) * 100
                    
                    index_records.append({
                        "cbsa_code": cbsa_code,
                        "year": year,
                        "index_value": new_index,
                        "appreciation_rate": appreciation_rate,
                        "weight_scheme": weight_scheme,
                        "n_supertracts": len(unique_supertracts),
                        "total_weight": total_weight
                    })
        
        # Create DataFrame
        if not index_records:
            logger.warning(f"No index records created for {weight_scheme}")
            return pl.DataFrame()
        
        indices_df = pl.DataFrame(index_records)
        
        # Ensure balanced panel
        indices_df = self._create_balanced_panel(
            indices_df, cbsa_codes, start_year, end_year, weight_scheme
        )
        
        # Sort for consistent output
        indices_df = indices_df.sort(["cbsa_code", "year"])
        
        # Log summary statistics
        self._log_summary_statistics(indices_df, weight_scheme)
        
        return indices_df
    
    def _create_balanced_panel(
        self,
        indices_df: pl.DataFrame,
        cbsa_codes: List[str],
        start_year: int,
        end_year: int,
        weight_scheme: str
    ) -> pl.DataFrame:
        """Ensure every CBSA has an entry for every year."""
        # Create all CBSA-year combinations
        all_combinations = []
        for cbsa_code in cbsa_codes:
            for year in range(start_year, end_year + 1):
                all_combinations.append({
                    "cbsa_code": cbsa_code,
                    "year": year,
                    "weight_scheme": weight_scheme
                })
        
        all_combinations_df = pl.DataFrame(all_combinations)
        
        # Join with actual indices
        balanced_df = all_combinations_df.join(
            indices_df,
            on=["cbsa_code", "year", "weight_scheme"],
            how="left"
        )
        
        # Forward-fill missing values within each CBSA
        balanced_df = (
            balanced_df
            .sort(["cbsa_code", "year"])
            .with_columns([
                pl.col("index_value").forward_fill().over("cbsa_code"),
                pl.col("n_supertracts").forward_fill().over("cbsa_code")
            ])
        )
        
        # Fill any remaining nulls with base value
        balanced_df = balanced_df.with_columns(
            pl.col("index_value").fill_null(BASE_INDEX_VALUE)
        )
        
        # Recalculate appreciation rates
        balanced_df = balanced_df.with_columns([
            pl.when(pl.col("year") > start_year)
            .then(
                ((pl.col("index_value") / pl.col("index_value").shift(1).over("cbsa_code")) - 1) * 100
            )
            .otherwise(None)
            .alias("appreciation_rate")
        ])
        
        return balanced_df
    
    def _log_summary_statistics(self, indices_df: pl.DataFrame, weight_scheme: str) -> None:
        """Log summary statistics for the indices."""
        n_cbsas = indices_df["cbsa_code"].n_unique()
        n_years = indices_df["year"].n_unique()
        
        appreciation_stats = indices_df.filter(
            pl.col("appreciation_rate").is_not_null()
        )["appreciation_rate"]
        
        if len(appreciation_stats) > 0:
            logger.info(
                f"City index summary ({weight_scheme})",
                n_cbsas=n_cbsas,
                n_years=n_years,
                avg_appreciation=f"{appreciation_stats.mean():.2f}%",
                median_appreciation=f"{appreciation_stats.median():.2f}%",
                std_appreciation=f"{appreciation_stats.std():.2f}%"
            )