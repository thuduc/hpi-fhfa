"""
City-level and geographic aggregation using Polars.

This module implements methods for aggregating price indices from lower-level
geographies (tracts, counties) to higher levels (MSAs, states, national).
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import date
from collections import defaultdict

import polars as pl
import numpy as np
from scipy import stats

from rsai.src.data.models import (
    IndexValue,
    GeographyLevel,
    WeightingScheme
)

logger = logging.getLogger(__name__)


class IndexAggregator:
    """Aggregates price indices across geographic hierarchies."""
    
    def __init__(
        self,
        aggregation_method: str = "weighted_mean",
        weight_by: str = "transaction_count",
        min_components: int = 3
    ):
        """
        Initialize index aggregator.
        
        Args:
            aggregation_method: Method for aggregation ('weighted_mean', 'geometric_mean', 'median')
            weight_by: Weighting variable ('transaction_count', 'property_count', 'population', 'equal')
            min_components: Minimum number of components required for aggregation
        """
        self.aggregation_method = aggregation_method
        self.weight_by = weight_by
        self.min_components = min_components
        
    def aggregate_indices(
        self,
        index_df: pl.DataFrame,
        from_level: GeographyLevel,
        to_level: GeographyLevel,
        geography_mapping: pl.DataFrame,
        weights_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Aggregate indices from one geographic level to another.
        
        Args:
            index_df: DataFrame with index values at lower level
            from_level: Source geography level
            to_level: Target geography level
            geography_mapping: DataFrame mapping lower to higher geography
            weights_df: Optional DataFrame with aggregation weights
            
        Returns:
            DataFrame with aggregated indices
        """
        logger.info(f"Aggregating indices from {from_level.value} to {to_level.value}")
        
        # Validate aggregation direction
        if not self._is_valid_aggregation(from_level, to_level):
            raise ValueError(f"Invalid aggregation from {from_level.value} to {to_level.value}")
            
        # Join with mapping
        mapped_df = index_df.join(
            geography_mapping,
            left_on=f"{from_level.value}_id",
            right_on=f"{from_level.value}_id",
            how="inner"
        )
        
        # Add weights if provided
        if weights_df is not None:
            mapped_df = mapped_df.join(
                weights_df,
                on=[f"{from_level.value}_id", "period"],
                how="left"
            )
        else:
            # Create default weights based on weight_by parameter
            mapped_df = self._create_default_weights(mapped_df, from_level)
            
        # Perform aggregation
        if self.aggregation_method == "weighted_mean":
            aggregated_df = self._weighted_mean_aggregation(mapped_df, from_level, to_level)
        elif self.aggregation_method == "geometric_mean":
            aggregated_df = self._geometric_mean_aggregation(mapped_df, from_level, to_level)
        elif self.aggregation_method == "median":
            aggregated_df = self._median_aggregation(mapped_df, from_level, to_level)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
        return aggregated_df
        
    def create_geography_mapping(
        self,
        properties_df: pl.DataFrame,
        from_col: str,
        to_col: str
    ) -> pl.DataFrame:
        """
        Create mapping between geographic levels from property data.
        
        Args:
            properties_df: DataFrame with property locations
            from_col: Column name for lower-level geography
            to_col: Column name for higher-level geography
            
        Returns:
            DataFrame with unique geography mappings
        """
        # Get unique mappings
        mapping_df = properties_df.select([from_col, to_col]).unique().filter(
            pl.col(from_col).is_not_null() & 
            pl.col(to_col).is_not_null()
        )
        
        # Check for many-to-many mappings
        from_counts = mapping_df.group_by(from_col).agg(
            pl.count().alias("num_to")
        ).filter(pl.col("num_to") > 1)
        
        if len(from_counts) > 0:
            logger.warning(f"Found {len(from_counts)} {from_col} values mapping to multiple {to_col}")
            
        return mapping_df
        
    def aggregate_to_msa(
        self,
        county_indices: pl.DataFrame,
        county_to_msa: pl.DataFrame,
        population_weights: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Aggregate county-level indices to MSA level.
        
        Args:
            county_indices: DataFrame with county-level indices
            county_to_msa: Mapping from county to MSA
            population_weights: Optional population weights by county
            
        Returns:
            DataFrame with MSA-level indices
        """
        # Rename columns for consistency
        county_indices = county_indices.rename({
            "geography_id": "county_id",
            "index_value": "county_index"
        })
        
        # Join with mapping
        msa_data = county_indices.join(
            county_to_msa,
            on="county_id",
            how="inner"
        )
        
        # Add population weights if provided
        if population_weights is not None:
            msa_data = msa_data.join(
                population_weights.select(["county_id", "population"]),
                on="county_id",
                how="left"
            ).with_columns([
                pl.col("population").fill_null(1).alias("weight")
            ])
        else:
            # Use transaction counts as weights
            msa_data = msa_data.with_columns([
                pl.col("num_pairs").alias("weight")
            ])
            
        # Aggregate by MSA and period
        msa_indices = msa_data.group_by(["msa_id", "period"]).agg([
            # Weighted average of county indices
            (pl.col("county_index") * pl.col("weight")).sum() / pl.col("weight").sum(),
            
            # Sum statistics across counties
            pl.col("num_pairs").sum().alias("num_pairs"),
            pl.col("num_properties").sum().alias("num_properties"),
            
            # Weighted median price
            pl.col("median_price").median().alias("median_price"),
            
            # Number of counties
            pl.count().alias("num_counties"),
            
            # Total weight
            pl.col("weight").sum().alias("total_weight")
        ]).rename({
            "county_index": "index_value"
        })
        
        # Filter by minimum components
        msa_indices = msa_indices.filter(
            pl.col("num_counties") >= self.min_components
        )
        
        # Add geography level
        msa_indices = msa_indices.with_columns([
            pl.lit(GeographyLevel.MSA.value).alias("geography_level"),
            pl.col("msa_id").alias("geography_id")
        ])
        
        return msa_indices
        
    def aggregate_to_state(
        self,
        lower_indices: pl.DataFrame,
        geography_to_state: pl.DataFrame,
        state_weights: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Aggregate indices to state level.
        
        Args:
            lower_indices: DataFrame with lower-level indices
            geography_to_state: Mapping to state
            state_weights: Optional state-level weights
            
        Returns:
            DataFrame with state-level indices
        """
        # Join with state mapping
        state_data = lower_indices.join(
            geography_to_state,
            left_on="geography_id",
            right_on="geography_id",
            how="inner"
        )
        
        # Add weights
        if state_weights is not None:
            state_data = state_data.join(
                state_weights,
                on=["geography_id", "period"],
                how="left"
            )
        else:
            state_data = state_data.with_columns([
                pl.col("num_pairs").alias("weight")
            ])
            
        # Aggregate by state
        state_indices = state_data.group_by(["state", "period"]).agg([
            # Weighted index
            ((pl.col("index_value") * pl.col("weight")).sum() / pl.col("weight").sum()).alias("index_value"),
            
            # Sum statistics
            pl.col("num_pairs").sum().alias("num_pairs"),
            pl.col("num_properties").sum().alias("num_properties"),
            pl.col("median_price").median().alias("median_price"),
            pl.count().alias("num_components")
        ])
        
        # Add metadata
        state_indices = state_indices.with_columns([
            pl.lit(GeographyLevel.STATE.value).alias("geography_level"),
            pl.col("state").alias("geography_id")
        ])
        
        return state_indices
        
    def aggregate_to_national(
        self,
        state_indices: pl.DataFrame,
        state_weights: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Aggregate state indices to national level.
        
        Args:
            state_indices: DataFrame with state-level indices
            state_weights: Optional weights by state
            
        Returns:
            DataFrame with national-level indices
        """
        # Add weights
        if state_weights is not None:
            national_data = state_indices.join(
                state_weights,
                on=["geography_id", "period"],
                how="left"
            )
        else:
            # Weight by number of transactions
            national_data = state_indices.with_columns([
                pl.col("num_pairs").alias("weight")
            ])
            
        # Aggregate to national
        national_indices = national_data.group_by("period").agg([
            # Weighted index
            ((pl.col("index_value") * pl.col("weight")).sum() / pl.col("weight").sum()).alias("index_value"),
            
            # Sum statistics
            pl.col("num_pairs").sum().alias("num_pairs"),
            pl.col("num_properties").sum().alias("num_properties"),
            pl.col("median_price").median().alias("median_price"),
            pl.count().alias("num_states")
        ])
        
        # Add metadata
        national_indices = national_indices.with_columns([
            pl.lit(GeographyLevel.NATIONAL.value).alias("geography_level"),
            pl.lit("USA").alias("geography_id")
        ])
        
        return national_indices
        
    def calculate_diffusion_index(
        self,
        indices_df: pl.DataFrame,
        threshold: float = 0.0
    ) -> pl.DataFrame:
        """
        Calculate diffusion index showing percentage of areas with positive growth.
        
        Args:
            indices_df: DataFrame with indices by geography
            threshold: Growth threshold (0 = any positive growth)
            
        Returns:
            DataFrame with diffusion index by period
        """
        # Calculate period-over-period growth
        growth_df = indices_df.sort(["geography_id", "period"]).with_columns([
            (pl.col("index_value") / pl.col("index_value").shift(1).over("geography_id") - 1)
            .alias("growth_rate")
        ])
        
        # Calculate diffusion index
        diffusion_df = growth_df.filter(
            pl.col("growth_rate").is_not_null()
        ).group_by("period").agg([
            (pl.col("growth_rate") > threshold).mean().alias("diffusion_index"),
            pl.col("growth_rate").mean().alias("mean_growth"),
            pl.col("growth_rate").median().alias("median_growth"),
            pl.col("growth_rate").std().alias("growth_dispersion"),
            pl.count().alias("num_areas")
        ])
        
        return diffusion_df
        
    def calculate_contribution_weights(
        self,
        indices_df: pl.DataFrame,
        base_period: date
    ) -> pl.DataFrame:
        """
        Calculate contribution weights for index components.
        
        Args:
            indices_df: DataFrame with component indices
            base_period: Base period for weight calculation
            
        Returns:
            DataFrame with contribution weights
        """
        # Get base period values
        base_values = indices_df.filter(
            pl.col("period") == base_period
        ).select([
            "geography_id",
            pl.col("num_pairs").alias("base_pairs"),
            pl.col("median_price").alias("base_price")
        ])
        
        # Join with current data
        weighted_df = indices_df.join(
            base_values,
            on="geography_id",
            how="left"
        )
        
        # Calculate weights based on base period share
        weighted_df = weighted_df.with_columns([
            (pl.col("base_pairs") / pl.col("base_pairs").sum().over("period"))
            .alias("transaction_weight"),
            
            (pl.col("base_price") * pl.col("base_pairs") / 
             (pl.col("base_price") * pl.col("base_pairs")).sum().over("period"))
            .alias("value_weight")
        ])
        
        return weighted_df
        
    def decompose_aggregate_change(
        self,
        component_indices: pl.DataFrame,
        aggregate_index: pl.DataFrame,
        period1: date,
        period2: date
    ) -> Dict[str, float]:
        """
        Decompose aggregate index change into component contributions.
        
        Args:
            component_indices: DataFrame with component-level indices
            aggregate_index: DataFrame with aggregate index
            period1: Start period
            period2: End period
            
        Returns:
            Dictionary with decomposition results
        """
        # Get component changes
        components_p1 = component_indices.filter(pl.col("period") == period1)
        components_p2 = component_indices.filter(pl.col("period") == period2)
        
        # Join periods
        component_changes = components_p1.join(
            components_p2,
            on="geography_id",
            suffix="_p2"
        )
        
        # Calculate changes and weights
        component_changes = component_changes.with_columns([
            (pl.col("index_value_p2") / pl.col("index_value") - 1).alias("component_return"),
            (pl.col("num_pairs") / pl.col("num_pairs").sum()).alias("weight")
        ])
        
        # Get aggregate change
        agg_p1 = aggregate_index.filter(pl.col("period") == period1)["index_value"][0]
        agg_p2 = aggregate_index.filter(pl.col("period") == period2)["index_value"][0]
        aggregate_return = agg_p2 / agg_p1 - 1
        
        # Calculate contributions
        contributions = component_changes.with_columns([
            (pl.col("component_return") * pl.col("weight")).alias("contribution")
        ])
        
        # Decomposition results
        decomposition = {
            "aggregate_return": float(aggregate_return),
            "weighted_component_return": float(contributions["contribution"].sum()),
            "interaction_effect": float(aggregate_return - contributions["contribution"].sum()),
            "max_contribution": {
                "geography_id": contributions.sort("contribution", descending=True)["geography_id"][0],
                "contribution": float(contributions["contribution"].max())
            },
            "min_contribution": {
                "geography_id": contributions.sort("contribution")["geography_id"][0],
                "contribution": float(contributions["contribution"].min())
            }
        }
        
        return decomposition
        
    def _is_valid_aggregation(
        self,
        from_level: GeographyLevel,
        to_level: GeographyLevel
    ) -> bool:
        """Check if aggregation direction is valid."""
        hierarchy = [
            GeographyLevel.PROPERTY,
            GeographyLevel.BLOCK,
            GeographyLevel.TRACT,
            GeographyLevel.SUPERTRACT,
            GeographyLevel.ZIP,
            GeographyLevel.COUNTY,
            GeographyLevel.MSA,
            GeographyLevel.STATE,
            GeographyLevel.NATIONAL
        ]
        
        from_idx = hierarchy.index(from_level)
        to_idx = hierarchy.index(to_level)
        
        return to_idx > from_idx
        
    def _create_default_weights(
        self,
        df: pl.DataFrame,
        from_level: GeographyLevel
    ) -> pl.DataFrame:
        """Create default weights based on weight_by parameter."""
        if self.weight_by == "transaction_count":
            return df.with_columns([
                pl.col("num_pairs").alias("weight")
            ])
        elif self.weight_by == "property_count":
            return df.with_columns([
                pl.col("num_properties").alias("weight")
            ])
        elif self.weight_by == "equal":
            return df.with_columns([
                pl.lit(1.0).alias("weight")
            ])
        else:
            raise ValueError(f"Unknown weight_by parameter: {self.weight_by}")
            
    def _weighted_mean_aggregation(
        self,
        df: pl.DataFrame,
        from_level: GeographyLevel,
        to_level: GeographyLevel
    ) -> pl.DataFrame:
        """Perform weighted mean aggregation."""
        # Group by target geography and period
        aggregated = df.group_by([f"{to_level.value}_id", "period"]).agg([
            # Weighted mean index
            ((pl.col("index_value") * pl.col("weight")).sum() / pl.col("weight").sum())
            .alias("index_value"),
            
            # Sum statistics
            pl.col("num_pairs").sum().alias("num_pairs"),
            pl.col("num_properties").sum().alias("num_properties"),
            pl.col("median_price").median().alias("median_price"),
            
            # Count components
            pl.count().alias("num_components"),
            
            # Total weight
            pl.col("weight").sum().alias("total_weight"),
            
            # Calculate weighted standard error if available
            pl.when(pl.col("standard_error").is_not_null())
            .then(
                ((pl.col("standard_error")**2 * pl.col("weight")**2).sum() / 
                 pl.col("weight").sum()**2).sqrt()
            )
            .otherwise(None)
            .alias("standard_error")
        ])
        
        # Filter by minimum components
        aggregated = aggregated.filter(
            pl.col("num_components") >= self.min_components
        )
        
        # Add metadata
        aggregated = aggregated.with_columns([
            pl.lit(to_level.value).alias("geography_level"),
            pl.col(f"{to_level.value}_id").alias("geography_id")
        ])
        
        return aggregated
        
    def _geometric_mean_aggregation(
        self,
        df: pl.DataFrame,
        from_level: GeographyLevel,
        to_level: GeographyLevel
    ) -> pl.DataFrame:
        """Perform geometric mean aggregation."""
        # Convert to log space for geometric mean
        df = df.with_columns([
            pl.col("index_value").log().alias("log_index")
        ])
        
        # Weighted average in log space
        aggregated = df.group_by([f"{to_level.value}_id", "period"]).agg([
            # Geometric mean (exp of weighted average of logs)
            ((pl.col("log_index") * pl.col("weight")).sum() / pl.col("weight").sum()).exp()
            .alias("index_value"),
            
            # Other statistics
            pl.col("num_pairs").sum().alias("num_pairs"),
            pl.col("num_properties").sum().alias("num_properties"),
            pl.col("median_price").median().alias("median_price"),
            pl.count().alias("num_components")
        ])
        
        # Filter and add metadata
        aggregated = aggregated.filter(
            pl.col("num_components") >= self.min_components
        ).with_columns([
            pl.lit(to_level.value).alias("geography_level"),
            pl.col(f"{to_level.value}_id").alias("geography_id")
        ])
        
        return aggregated
        
    def _median_aggregation(
        self,
        df: pl.DataFrame,
        from_level: GeographyLevel,
        to_level: GeographyLevel
    ) -> pl.DataFrame:
        """Perform median aggregation."""
        # For median, we don't use weights directly
        aggregated = df.group_by([f"{to_level.value}_id", "period"]).agg([
            # Median index
            pl.col("index_value").median().alias("index_value"),
            
            # Other statistics
            pl.col("num_pairs").sum().alias("num_pairs"),
            pl.col("num_properties").sum().alias("num_properties"),
            pl.col("median_price").median().alias("median_price"),
            pl.count().alias("num_components"),
            
            # MAD for robust dispersion measure
            (pl.col("index_value") - pl.col("index_value").median()).abs().median()
            .alias("mad")
        ])
        
        # Filter and add metadata
        aggregated = aggregated.filter(
            pl.col("num_components") >= self.min_components
        ).with_columns([
            pl.lit(to_level.value).alias("geography_level"),
            pl.col(f"{to_level.value}_id").alias("geography_id")
        ])
        
        return aggregated